"""
NH3-Guided MOF Generation Script

This script generates MOF structures guided by target NH3 uptake values.
It uses the trained NH3 head model to optimize latent vectors towards
desired NH3 uptake, then decodes them to MOF structures.

Usage:
    python scripts/sample_nh3_guided.py --target_nh3 5.0 --n_samples 3
    python scripts/sample_nh3_guided.py --target_nh3 10.0 --output_dir my_samples
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import seed_everything
from scipy.spatial import KDTree

from mofdiff.common.atomic_utils import arrange_decoded_mofs
from mofdiff.common.eval_utils import load_mofdiff_model


# ============================================================================
# Model Architectures (must match training)
# ============================================================================

class RobustPredictor(nn.Module):
    """Robust NH3 predictor."""
    def __init__(self, input_dim=256, hidden_dims=[128, 64], dropout=0.3):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)
    
    def forward(self, x):
        x = self.input_norm(x)
        x = self.hidden(x)
        return self.output(x).squeeze(-1)


class GradientGuidedPredictor(nn.Module):
    """Predictor optimized for gradient-based optimization."""
    def __init__(self, input_dim=256, hidden_dim=128, n_layers=3, dropout=0.2):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ))
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        x = self.input_norm(x)
        x = self.input_proj(x)
        for layer in self.layers:
            x = x + 0.1 * layer(x)
        return self.output(x).squeeze(-1)


class EnsemblePredictor(nn.Module):
    """Ensemble of models."""
    def __init__(self, input_dim=256, n_models=5):
        super().__init__()
        self.models = nn.ModuleList([
            RobustPredictor(input_dim, hidden_dims=[64 + i*16, 32], dropout=0.2 + i*0.05)
            for i in range(n_models)
        ])
    
    def forward(self, x):
        preds = torch.stack([m(x) for m in self.models], dim=0)
        return preds.mean(dim=0)


def load_nh3_predictor(model_path, device='cuda'):
    """Load the trained NH3 predictor."""
    print(f"Loading NH3 predictor from {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    
    model_config = ckpt.get('model_config', 'robust_small_combined')
    input_dim = ckpt.get('input_dim', 256)
    
    # Create model based on config
    if 'ensemble' in model_config:
        model = EnsemblePredictor(input_dim, 5)
    elif 'gradient_guided' in model_config:
        model = GradientGuidedPredictor(input_dim, 128, 3, 0.2)
    elif 'robust_medium' in model_config:
        model = RobustPredictor(input_dim, [128, 64], 0.3)
    else:
        model = RobustPredictor(input_dim, [64, 32], 0.3)
    
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, ckpt


def optimize_latent_for_target(
    model, ckpt, target_nh3,
    n_samples=5, n_init=30, steps=500, lr=0.1,
    device='cuda'
):
    """
    Optimize latent vectors to achieve target NH3 uptake.
    
    Args:
        model: NH3 predictor model
        ckpt: Model checkpoint with normalization params
        target_nh3: Target NH3 uptake (mmol/g)
        n_samples: Number of samples to return
        n_init: Number of random initializations
        steps: Optimization steps per init
        lr: Learning rate
    
    Returns:
        optimized_z: Optimized latent vectors
        predicted_nh3: Predicted NH3 values
    """
    y_mean = ckpt['y_mean']
    y_std = ckpt['y_std']
    X_mean = ckpt['X_mean'].to(device)
    X_std = ckpt['X_std'].to(device)
    use_log = ckpt.get('use_log_transform', False)
    latent_dim = ckpt.get('input_dim', 256)
    
    # Transform target if using log scale
    if use_log:
        target_transformed = np.log1p(target_nh3)
    else:
        target_transformed = target_nh3
    
    target_normalized = (target_transformed - y_mean) / y_std
    
    print(f"\n{'='*60}")
    print(f"GRADIENT-BASED LATENT OPTIMIZATION")
    print(f"{'='*60}")
    print(f"Target NH3: {target_nh3:.2f} mmol/g")
    print(f"Using log transform: {use_log}")
    print(f"Initializations: {n_init}, Steps: {steps}")
    
    all_z = []
    all_nh3 = []
    all_losses = []
    
    for init_idx in range(n_init):
        # Initialize with random latent
        z = torch.randn(1, latent_dim, device=device) * 0.5
        z.requires_grad = True
        
        optimizer = optim.Adam([z], lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        
        for step in range(steps):
            optimizer.zero_grad()
            
            # Normalize input
            z_normalized = (z - X_mean) / X_std
            
            # Predict
            pred = model(z_normalized)
            
            # Loss: MSE to target + regularization to keep z reasonable
            target_loss = (pred - target_normalized) ** 2
            reg_loss = 0.01 * (z ** 2).mean()
            loss = target_loss + reg_loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # Get final prediction
        with torch.no_grad():
            z_normalized = (z - X_mean) / X_std
            final_pred_norm = model(z_normalized)
            final_pred = final_pred_norm * y_std + y_mean
            
            if use_log:
                final_nh3 = np.expm1(final_pred.item())
            else:
                final_nh3 = final_pred.item()
            
            all_z.append(z.detach().clone())
            all_nh3.append(final_nh3)
            all_losses.append(abs(final_nh3 - target_nh3))
        
        if (init_idx + 1) % 10 == 0:
            print(f"  Init {init_idx+1}/{n_init}: NH3 = {final_nh3:.4f}, "
                  f"distance = {abs(final_nh3 - target_nh3):.4f}")
    
    # Select best results
    all_z = torch.cat(all_z, dim=0)
    all_nh3 = np.array(all_nh3)
    all_losses = np.array(all_losses)
    
    # Sort by distance to target
    best_idx = np.argsort(all_losses)[:n_samples]
    
    optimized_z = all_z[best_idx]
    predicted_nh3 = all_nh3[best_idx]
    
    print(f"\nBest {n_samples} results:")
    for i, (nh3, dist) in enumerate(zip(predicted_nh3, all_losses[best_idx])):
        print(f"  Sample {i}: NH3 = {nh3:.4f}, distance = {dist:.4f}")
    
    return optimized_z, predicted_nh3


def decode_latents_to_mofs(
    latent_vectors, mofdiff_model, bb_emb_space,
    output_dir, device='cuda'
):
    """
    Decode optimized latent vectors to MOF structures.
    
    Args:
        latent_vectors: Optimized latent vectors
        mofdiff_model: Loaded MOFDiff model
        bb_emb_space: Building block embedding space
        output_dir: Directory to save outputs
    """
    print(f"\n{'='*60}")
    print(f"DECODING LATENT VECTORS TO MOFs")
    print(f"{'='*60}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build KD-tree for nearest neighbor search
    bb_data = torch.load(bb_emb_space, weights_only=False)
    bb_embs = bb_data['bb_emb'].numpy()
    bb_names = bb_data['m_id']
    kdtree = KDTree(bb_embs)
    
    all_samples = defaultdict(list)
    
    for i, z in enumerate(latent_vectors):
        print(f"\nDecoding sample {i}...")
        
        z = z.unsqueeze(0).to(device) if z.dim() == 1 else z.to(device)
        
        # Sample MOF from latent
        with torch.no_grad():
            sample = mofdiff_model.sample_from_z(z, num_steps=1000)
        
        if sample is None:
            print(f"  Failed to decode sample {i}")
            continue
        
        # Store sample
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                all_samples[k].append(v.cpu())
            else:
                all_samples[k].append(v)
        
        print(f"  Successfully decoded sample {i}")
    
    if not all_samples:
        print("No samples were successfully decoded!")
        return None
    
    # Combine samples
    combined_samples = {}
    for k, v_list in all_samples.items():
        if isinstance(v_list[0], torch.Tensor):
            combined_samples[k] = torch.cat(v_list, dim=0)
        else:
            combined_samples[k] = v_list
    
    # Save samples
    samples_path = output_dir / 'samples.pt'
    torch.save(combined_samples, samples_path)
    print(f"\nSaved raw samples to {samples_path}")
    
    return combined_samples


def main():
    parser = argparse.ArgumentParser(description='NH3-Guided MOF Generation')
    parser.add_argument('--target_nh3', type=float, required=True,
                        help='Target NH3 uptake in mmol/g')
    parser.add_argument('--n_samples', type=int, default=3,
                        help='Number of MOF samples to generate')
    parser.add_argument('--n_init', type=int, default=30,
                        help='Number of optimization initializations')
    parser.add_argument('--steps', type=int, default=500,
                        help='Optimization steps per initialization')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Optimization learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: pretrained/mofdiff_ckpt/nh3_guided_<target>)')
    parser.add_argument('--nh3_model', type=str, default='raw_nh3_core/nh3_predictor_mlp.pt',
                        help='Path to NH3 predictor model')
    parser.add_argument('--mofdiff_ckpt', type=str, default='pretrained/mofdiff_ckpt',
                        help='Path to MOFDiff checkpoint')
    parser.add_argument('--bb_emb', type=str, default='pretrained/bb_emb_space.pt',
                        help='Path to building block embeddings')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--skip_decode', action='store_true',
                        help='Skip decoding (only optimize latents)')
    args = parser.parse_args()
    
    # Set random seed
    seed_everything(args.seed)
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = f"pretrained/mofdiff_ckpt/nh3_guided_{args.target_nh3:.1f}"
    
    print("=" * 70)
    print("NH3-GUIDED MOF GENERATION")
    print("=" * 70)
    print(f"Target NH3: {args.target_nh3} mmol/g")
    print(f"Samples: {args.n_samples}")
    print(f"Output: {args.output_dir}")
    print()
    
    # Load NH3 predictor
    model, ckpt = load_nh3_predictor(args.nh3_model, args.device)
    
    # Optimize latents
    optimized_z, predicted_nh3 = optimize_latent_for_target(
        model, ckpt, args.target_nh3,
        n_samples=args.n_samples,
        n_init=args.n_init,
        steps=args.steps,
        lr=args.lr,
        device=args.device
    )
    
    # Save optimized latents
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    latent_save = {
        'optimized_z': optimized_z.cpu(),
        'predicted_nh3': predicted_nh3,
        'target_nh3': args.target_nh3,
        'args': vars(args)
    }
    torch.save(latent_save, output_dir / 'optimized_latents.pt')
    print(f"\nSaved optimized latents to {output_dir / 'optimized_latents.pt'}")
    
    # Save summary
    summary = {
        'target_nh3': args.target_nh3,
        'n_samples': args.n_samples,
        'predicted_nh3': predicted_nh3.tolist(),
        'seed': args.seed
    }
    with open(output_dir / 'optimization_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    if args.skip_decode:
        print("\nSkipping MOF decoding (--skip_decode flag set)")
        return
    
    # Load MOFDiff model for decoding
    print(f"\nLoading MOFDiff model from {args.mofdiff_ckpt}")
    mofdiff_model = load_mofdiff_model(args.mofdiff_ckpt, args.device)
    
    # Decode latents to MOFs
    samples = decode_latents_to_mofs(
        optimized_z, mofdiff_model, args.bb_emb,
        args.output_dir, args.device
    )
    
    print("\n" + "=" * 70)
    print("NH3-GUIDED GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\nTarget NH3: {args.target_nh3} mmol/g")
    print(f"Generated {args.n_samples} samples")
    print(f"Predicted NH3 values: {predicted_nh3}")
    print(f"\nOutput saved to: {args.output_dir}")
    print("\nNext steps:")
    print(f"  1. Assemble: python mofdiff/scripts/assemble.py --sample_dir {args.output_dir}")
    print(f"  2. Relax: python scripts/simple_relax.py {args.output_dir}/cif")
    print(f"  3. GCMC: python mofdiff/scripts/gcmc_nh3_screen.py --input {args.output_dir}/relaxed")


if __name__ == '__main__':
    main()
