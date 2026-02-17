"""
Latent Vector Optimizer for NH3-targeted MOF Generation

This is the core optimization engine. Given a target NH3 uptake value,
it optimizes the 256-dimensional latent vector z that will be fed into
MOFDiff's diffusion model (model.sample(z=z_optimized)).

Optimization Strategy:
  1. Initialize z from high-performing reference MOFs in the latent dataset
  2. Gradient descent on z through the differentiable NH3 predictor ensemble
  3. Chemistry-informed regularization (acidic metal site preferences)
  4. Latent space distribution constraint (KL penalty to N(0,I) prior)
  5. Uncertainty-aware optimization (avoid high-uncertainty regions)

Chemistry Background (NH3 Adsorption in MOFs):
  - NH3 is a hard Lewis base (lone pair on nitrogen)
  - Best adsorption via Lewis acid-base interaction with open metal sites (OMS)
  - Hard Lewis acid metals: Al³⁺, Fe³⁺, Cr³⁺, Zr⁴⁺, Mg²⁺, Ca²⁺, Ti⁴⁺
  - Intermediate Lewis acids also effective: Zn²⁺, Cu²⁺, Ni²⁺, Co²⁺
  - Pore size matters: micropores (< 2nm) enhance uptake via confinement
  - Functional groups (-OH, -NH₂, -COOH) provide additional H-bonding sites
  
  In the BW-DB dataset, Zn/Cu/Ni MOFs dominate high uptake because they are
  the most common metals with abundant open metal sites. The optimizer learns
  latent directions correlated with these structural features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from scipy.stats import spearmanr

from mofdiff.nh3_optimizer.predictor import NH3PredictorEnsemble
from mofdiff.nh3_optimizer.train import load_predictor


class LatentOptimizer:
    """
    Gradient-based optimizer for MOFDiff latent vectors targeting NH3 uptake.
    
    The optimizer works in the 256-dim latent space of the MOFDiff VAE.
    It uses gradients from the trained NH3 predictor ensemble to move z
    toward regions associated with the target NH3 uptake value.
    
    Key features:
      - Multi-start optimization from reference MOF latents
      - Adaptive step size with gradient clipping
      - KL divergence constraint to stay near the prior p(z) = N(0,I)
      - Uncertainty penalty to avoid unreliable predictions
      - Chemistry-aware initialization from high-performing MOFs
    """
    
    def __init__(
        self,
        predictor: NH3PredictorEnsemble,
        latent_dataset_path: str = "raw_nh3_core/nh3_latent_dataset.pt",
        device: str = "cuda",
    ):
        self.predictor = predictor.to(device).eval()
        self.device = device
        
        # Load reference latent dataset
        data = torch.load(latent_dataset_path, map_location='cpu', weights_only=False)
        self.ref_z = data['z']  # [N, 256]
        self.ref_y = data['y']  # [N]
        self.ref_ids = data['m_id']
        
        # Compute latent distribution statistics for constraints
        self.z_mean_dist = self.ref_z.mean(dim=0)  # Mean of latent distribution
        self.z_std_dist = self.ref_z.std(dim=0)    # Std of latent distribution
        self.z_cov = None  # Computed on demand
        
        # Identify chemistry-informative latent directions
        self._compute_nh3_gradient_direction()
    
    def _compute_nh3_gradient_direction(self):
        """
        Compute the average gradient direction in latent space that increases NH3.
        This is used as an informative prior for the optimization direction.
        """
        # Use the top quartile of NH3 MOFs as positive exemplars
        sorted_idx = torch.argsort(self.ref_y, descending=True)
        n_top = max(int(len(self.ref_y) * 0.25), 10)
        n_bottom = n_top
        
        top_z = self.ref_z[sorted_idx[:n_top]]
        bottom_z = self.ref_z[sorted_idx[-n_bottom:]]
        
        # Direction from low-NH3 to high-NH3 in latent space
        self.nh3_direction = (top_z.mean(dim=0) - bottom_z.mean(dim=0))
        self.nh3_direction = self.nh3_direction / (self.nh3_direction.norm() + 1e-8)
        
        # Compute per-dimension correlations with NH3
        self.dim_correlations = torch.zeros(self.ref_z.shape[1])
        for i in range(self.ref_z.shape[1]):
            r, _ = spearmanr(self.ref_z[:, i].numpy(), self.ref_y.numpy())
            self.dim_correlations[i] = r if not np.isnan(r) else 0.0
    
    def _select_initialization(
        self,
        target_nh3: float,
        n_starts: int = 8,
        strategy: str = "hybrid",
    ) -> torch.Tensor:
        """
        Select initial z vectors for optimization.
        
        Strategies:
          - "nearest": Start from MOFs closest to target NH3
          - "top": Start from highest-NH3 MOFs (biased toward high uptake)
          - "hybrid": Mix of nearest + top + random perturbations
          - "random": Random from prior N(0, I)
        
        Returns:
            z_init: [n_starts, 256] initial latent vectors
        """
        z_inits = []
        
        if strategy in ("nearest", "hybrid"):
            # Find MOFs closest to target
            distances = (self.ref_y - target_nh3).abs()
            nearest_idx = torch.argsort(distances)
            
            n_nearest = n_starts // 2 if strategy == "hybrid" else n_starts
            for i in range(min(n_nearest, len(nearest_idx))):
                idx = nearest_idx[i]
                z = self.ref_z[idx].clone()
                # Small perturbation for diversity
                z = z + torch.randn_like(z) * 0.1
                z_inits.append(z)
        
        if strategy in ("top", "hybrid"):
            # Start from high-NH3 MOFs
            sorted_idx = torch.argsort(self.ref_y, descending=True)
            n_top = n_starts // 4 if strategy == "hybrid" else n_starts
            for i in range(min(n_top, len(sorted_idx))):
                idx = sorted_idx[i]
                z = self.ref_z[idx].clone()
                z = z + torch.randn_like(z) * 0.15
                z_inits.append(z)
        
        if strategy in ("random", "hybrid"):
            # Random from prior + NH3 direction bias
            n_random = n_starts - len(z_inits)
            for _ in range(max(n_random, 0)):
                z = torch.randn(self.ref_z.shape[1])
                # Bias toward high-NH3 direction proportional to target
                bias_strength = min(target_nh3 / 5.0, 2.0)  # Scale with target
                z = z + self.nh3_direction * bias_strength
                z_inits.append(z)
        
        # Pad if needed
        while len(z_inits) < n_starts:
            z = torch.randn(self.ref_z.shape[1])
            z_inits.append(z)
        
        return torch.stack(z_inits[:n_starts])
    
    def optimize(
        self,
        target_nh3: float,
        n_samples: int = 10,
        n_starts: int = 16,
        n_steps: int = 300,
        lr: float = 0.05,
        kl_weight: float = 0.005,
        uncertainty_weight: float = 0.1,
        smoothness_weight: float = 0.001,
        target_tolerance: float = 0.5,
        init_strategy: str = "hybrid",
        verbose: bool = True,
    ) -> Dict:
        """
        Optimize latent vectors for target NH3 uptake.
        
        Args:
            target_nh3: Target NH3 uptake in mmol/g
            n_samples: Number of final optimized z vectors to return
            n_starts: Number of parallel optimization trajectories
            n_steps: Number of optimization steps
            lr: Learning rate for z optimization
            kl_weight: Weight for KL divergence regularization
            uncertainty_weight: Weight for uncertainty penalty
            smoothness_weight: Weight for gradient smoothness
            target_tolerance: Acceptable deviation from target (mmol/g)
            init_strategy: Initialization strategy
            verbose: Print progress
        
        Returns:
            dict with:
                'z': Optimized latent vectors [n_samples, 256]
                'predicted_nh3': Predicted uptake for each z [n_samples]
                'uncertainty': Prediction uncertainty [n_samples]
                'loss_history': Loss trajectory
                'optimization_info': Additional metadata
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"NH3 Latent Optimizer")
            print(f"{'='*60}")
            print(f"Target: {target_nh3:.2f} mmol/g")
            print(f"Starts: {n_starts}, Steps: {n_steps}, LR: {lr}")
            print(f"KL weight: {kl_weight}, Uncertainty weight: {uncertainty_weight}")
            print(f"{'='*60}\n")
        
        # Initialize
        z = self._select_initialization(target_nh3, n_starts, init_strategy)
        z = z.to(self.device).requires_grad_(True)
        
        optimizer = torch.optim.Adam([z], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_steps, eta_min=lr * 0.01
        )
        
        # Target tensor
        target = torch.tensor(target_nh3, device=self.device, dtype=torch.float)
        
        loss_history = []
        best_z = z.detach().clone()
        best_loss = float('inf')
        best_step = 0
        
        for step in range(n_steps):
            optimizer.zero_grad()
            
            # Forward pass through predictor
            result = self.predictor(z)
            pred_nh3 = result['mean']      # [n_starts]
            pred_std = result['std']        # [n_starts]
            epistemic = result['epistemic'] # [n_starts]
            
            # === Loss components ===
            
            # 1. Target matching loss (Huber for robustness)
            target_loss = F.huber_loss(
                pred_nh3, target.expand_as(pred_nh3),
                delta=target_tolerance
            )
            
            # 2. KL divergence to prior p(z) = N(0, I)
            # Keeps z in the valid latent space
            kl_loss = 0.5 * (z ** 2).mean()
            
            # 3. Uncertainty penalty
            # Discourages moving into regions where the predictor is uncertain
            uncertainty_loss = epistemic.mean()
            
            # 4. Smoothness regularization
            # Prevents z from having extreme values in individual dimensions
            smoothness_loss = (z[:, 1:] - z[:, :-1]).pow(2).mean()
            
            # Total loss
            total_loss = (
                target_loss
                + kl_weight * kl_loss
                + uncertainty_weight * uncertainty_loss
                + smoothness_weight * smoothness_loss
            )
            
            # Backward
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_([z], max_norm=5.0)
            
            optimizer.step()
            scheduler.step()
            
            # Track best
            with torch.no_grad():
                step_loss = target_loss.item()
                if step_loss < best_loss:
                    best_loss = step_loss
                    best_z = z.detach().clone()
                    best_step = step
            
            loss_history.append({
                'step': step,
                'total_loss': total_loss.item(),
                'target_loss': target_loss.item(),
                'kl_loss': kl_loss.item(),
                'uncertainty_loss': uncertainty_loss.item(),
                'pred_nh3_mean': pred_nh3.mean().item(),
                'pred_nh3_std': pred_std.mean().item(),
            })
            
            if verbose and (step + 1) % 50 == 0:
                pred_range = f"[{pred_nh3.min().item():.2f}, {pred_nh3.max().item():.2f}]"
                print(
                    f"Step {step+1:3d}: "
                    f"target_loss={target_loss.item():.4f}, "
                    f"pred={pred_nh3.mean().item():.3f}±{pred_std.mean().item():.3f}, "
                    f"range={pred_range}, "
                    f"kl={kl_loss.item():.3f}"
                )
        
        # Select best n_samples from the n_starts trajectories
        with torch.no_grad():
            final_result = self.predictor(best_z)
            final_pred = final_result['mean']
            final_std = final_result['std']
            final_epistemic = final_result['epistemic']
            
            # Score: closeness to target, penalized by uncertainty
            scores = -(final_pred - target).abs() - 0.5 * final_epistemic
            best_indices = torch.argsort(scores, descending=True)[:n_samples]
            
            selected_z = best_z[best_indices].cpu()
            selected_pred = final_pred[best_indices].cpu()
            selected_std = final_std[best_indices].cpu()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Optimization Complete (best at step {best_step})")
            print(f"{'='*60}")
            print(f"Selected {n_samples} optimized latent vectors:")
            for i in range(n_samples):
                print(f"  z[{i}]: predicted={selected_pred[i].item():.3f} ± "
                      f"{selected_std[i].item():.3f} mmol/g "
                      f"(target={target_nh3:.2f})")
        
        return {
            'z': selected_z,
            'predicted_nh3': selected_pred,
            'uncertainty': selected_std,
            'target_nh3': target_nh3,
            'loss_history': loss_history,
            'optimization_info': {
                'n_starts': n_starts,
                'n_steps': n_steps,
                'lr': lr,
                'kl_weight': kl_weight,
                'uncertainty_weight': uncertainty_weight,
                'best_step': best_step,
                'best_loss': best_loss,
                'init_strategy': init_strategy,
            },
        }

    def optimize_batch(
        self,
        targets: List[float],
        n_samples_per_target: int = 5,
        **kwargs,
    ) -> List[Dict]:
        """
        Optimize for multiple NH3 targets at once.
        
        Args:
            targets: List of target NH3 values
            n_samples_per_target: How many z vectors per target
            **kwargs: Passed to optimize()
        
        Returns:
            List of optimization results, one per target
        """
        results = []
        for target in targets:
            result = self.optimize(
                target_nh3=target,
                n_samples=n_samples_per_target,
                **kwargs,
            )
            results.append(result)
        
        return results


def create_optimizer(
    predictor_path: str = "pretrained/nh3_optimizer.pt",
    latent_dataset_path: str = "raw_nh3_core/nh3_latent_dataset.pt",
    device: str = "cuda",
) -> LatentOptimizer:
    """
    Create a LatentOptimizer from saved predictor checkpoint.
    
    Args:
        predictor_path: Path to trained NH3 predictor ensemble
        latent_dataset_path: Path to the latent dataset
        device: Device to use
    
    Returns:
        LatentOptimizer ready for optimization
    """
    device = device if torch.cuda.is_available() else "cpu"
    predictor = load_predictor(predictor_path, device=device)
    
    optimizer = LatentOptimizer(
        predictor=predictor,
        latent_dataset_path=latent_dataset_path,
        device=device,
    )
    
    return optimizer
