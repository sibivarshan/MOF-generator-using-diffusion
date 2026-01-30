"""
NH3-Guided MOF Generation with Latent Interpolation

This approach uses actual high-NH3 MOF latent vectors as anchors,
then interpolates and perturbs around them to generate new structures.

This is more reliable than pure gradient optimization when the
predictor has limited accuracy.
"""

import argparse
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import seed_everything


# Model definitions (same as training)
class RobustPredictor(nn.Module):
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


def load_nh3_predictor(model_path, device='cuda'):
    """Load NH3 predictor."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model_config = ckpt.get('model_config', 'robust_medium_mse')
    input_dim = ckpt.get('input_dim', 256)
    
    if 'robust_medium' in model_config:
        model = RobustPredictor(input_dim, [128, 64], 0.3)
    else:
        model = RobustPredictor(input_dim, [64, 32], 0.3)
    
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    return model, ckpt


def predict_nh3(model, ckpt, z, device='cuda'):
    """Predict NH3 for latent vector(s)."""
    X_mean = ckpt['X_mean'].to(device)
    X_std = ckpt['X_std'].to(device)
    y_mean = ckpt['y_mean']
    y_std = ckpt['y_std']
    use_log = ckpt.get('use_log_transform', False)
    
    with torch.no_grad():
        z_norm = (z - X_mean) / X_std
        pred_norm = model(z_norm)
        pred = pred_norm * y_std + y_mean
        
        if use_log:
            return np.expm1(pred.cpu().numpy())
        return pred.cpu().numpy()


def find_high_nh3_latents(latent_data, nh3_threshold=5.0, top_k=10):
    """Find latent vectors with high NH3 uptake."""
    z = latent_data['z']
    y = latent_data['y'].squeeze().numpy()
    m_ids = latent_data['m_id']
    
    # Sort by NH3 descending
    sorted_idx = np.argsort(y)[::-1]
    
    # Filter by threshold
    high_nh3_idx = [i for i in sorted_idx if y[i] >= nh3_threshold][:top_k]
    
    if len(high_nh3_idx) == 0:
        # Fall back to top_k highest
        high_nh3_idx = sorted_idx[:top_k].tolist()
    
    return {
        'z': z[high_nh3_idx],
        'y': y[high_nh3_idx],
        'm_id': [m_ids[i] for i in high_nh3_idx]
    }


def generate_latents_by_interpolation(
    anchor_latents, n_samples=5,
    perturbation_scale=0.1, device='cuda'
):
    """
    Generate new latents by interpolating between high-NH3 anchors
    and adding small perturbations.
    """
    z_anchors = anchor_latents['z'].to(device)
    n_anchors = len(z_anchors)
    
    generated_z = []
    
    for i in range(n_samples):
        # Randomly select 2-3 anchors for interpolation
        n_interp = min(np.random.randint(2, 4), n_anchors)
        idx = np.random.choice(n_anchors, n_interp, replace=False)
        
        # Random weights for interpolation
        weights = np.random.dirichlet(np.ones(n_interp))
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        
        # Interpolate
        z_interp = sum(w * z_anchors[j] for w, j in zip(weights, idx))
        
        # Add small perturbation
        noise = torch.randn_like(z_interp) * perturbation_scale
        z_new = z_interp + noise
        
        generated_z.append(z_new)
    
    return torch.stack(generated_z)


def select_best_latents(
    model, ckpt, candidates, target_nh3,
    n_select=3, device='cuda'
):
    """Select best latents based on predicted NH3."""
    predictions = predict_nh3(model, ckpt, candidates, device)
    
    # Sort by distance to target
    distances = np.abs(predictions - target_nh3)
    best_idx = np.argsort(distances)[:n_select]
    
    return candidates[best_idx], predictions[best_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_nh3', type=float, required=True)
    parser.add_argument('--n_samples', type=int, default=3)
    parser.add_argument('--n_candidates', type=int, default=100)
    parser.add_argument('--perturbation_scale', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--latent_data', type=str, default='raw_nh3_core/nh3_latent_dataset.pt')
    parser.add_argument('--nh3_model', type=str, default='raw_nh3_core/nh3_predictor_mlp.pt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_actual_nh3', action='store_true', default=True,
                        help='Use actual NH3 values instead of model predictions')
    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    if args.output_dir is None:
        args.output_dir = f'pretrained/mofdiff_ckpt/nh3_interp_{args.target_nh3:.1f}'
    
    print("=" * 70)
    print("NH3-GUIDED MOF GENERATION (INTERPOLATION)")
    print("=" * 70)
    print(f"Target NH3: {args.target_nh3} mmol/g")
    print(f"Output: {args.output_dir}")
    print()
    
    # Load model (optional, for predictions)
    model, ckpt = load_nh3_predictor(args.nh3_model, args.device)
    
    # Load latent data
    print("Loading latent dataset...")
    latent_data = torch.load(args.latent_data, weights_only=False)
    
    z_all = latent_data['z']
    y_all = latent_data['y'].squeeze().numpy()
    m_ids = latent_data['m_id']
    
    # Find MOFs closest to target NH3
    print(f"\nFinding MOFs closest to target NH3 = {args.target_nh3} mmol/g...")
    distances = np.abs(y_all - args.target_nh3)
    sorted_idx = np.argsort(distances)
    
    # Select top 20 anchors
    anchor_idx = sorted_idx[:20]
    anchors = {
        'z': z_all[anchor_idx],
        'y': y_all[anchor_idx],
        'm_id': [m_ids[i] for i in anchor_idx]
    }
    
    print(f"Selected {len(anchors['z'])} anchor MOFs")
    print(f"Anchor NH3 range: [{anchors['y'].min():.2f}, {anchors['y'].max():.2f}] mmol/g")
    print(f"Closest to target: {anchors['m_id'][0]} ({anchors['y'][0]:.2f} mmol/g)")
    print(f"Anchor MOFs: {anchors['m_id'][:5]}...")
    
    # Generate candidates by interpolation between close-to-target MOFs
    print(f"\nGenerating {args.n_candidates} candidate latents...")
    
    z_anchors = anchors['z'].to(args.device)
    y_anchors = anchors['y']
    n_anchors = len(z_anchors)
    
    candidates_z = []
    candidates_y_est = []
    
    for i in range(args.n_candidates):
        # Randomly select 2-3 anchors for interpolation
        n_interp = min(np.random.randint(2, 4), n_anchors)
        idx = np.random.choice(n_anchors, n_interp, replace=False)
        
        # Weights that favor anchors closer to target
        base_weights = 1.0 / (np.abs(y_anchors[idx] - args.target_nh3) + 0.1)
        weights = base_weights / base_weights.sum()
        weights = torch.tensor(weights, dtype=torch.float32, device=args.device)
        
        # Interpolate
        z_interp = sum(w * z_anchors[j] for w, j in zip(weights, idx))
        
        # Add small perturbation
        noise = torch.randn_like(z_interp) * args.perturbation_scale
        z_new = z_interp + noise
        
        # Estimate NH3 by weighted average of anchor NH3s (more reliable than model)
        y_est = sum(w.item() * y_anchors[j] for w, j in zip(weights, idx))
        
        candidates_z.append(z_new)
        candidates_y_est.append(y_est)
    
    candidates_z = torch.stack(candidates_z)
    candidates_y_est = np.array(candidates_y_est)
    
    print(f"\nCandidate estimated NH3:")
    print(f"  Range: [{candidates_y_est.min():.2f}, {candidates_y_est.max():.2f}] mmol/g")
    print(f"  Mean: {candidates_y_est.mean():.2f} mmol/g")
    
    # Select best by estimated NH3 (closest to target)
    distances = np.abs(candidates_y_est - args.target_nh3)
    best_idx = np.argsort(distances)[:args.n_samples]
    
    selected_z = candidates_z[best_idx]
    selected_y_est = candidates_y_est[best_idx]
    
    # Also get model predictions for comparison
    selected_y_pred = predict_nh3(model, ckpt, selected_z, args.device)
    
    print(f"\nSelected {args.n_samples} best candidates:")
    for i, (y_est, y_pred) in enumerate(zip(selected_y_est, selected_y_pred)):
        print(f"  {i}: estimated NH3 = {y_est:.2f} mmol/g, model pred = {y_pred:.2f} mmol/g")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'selected_z': selected_z.cpu(),
        'estimated_nh3': selected_y_est,
        'predicted_nh3': selected_y_pred,
        'target_nh3': args.target_nh3,
        'anchor_mofs': anchors['m_id'],
        'anchor_nh3': anchors['y'].tolist(),
        'args': vars(args)
    }, output_dir / 'selected_latents.pt')
    
    summary = {
        'target_nh3': args.target_nh3,
        'n_samples': args.n_samples,
        'estimated_nh3': selected_y_est.tolist(),
        'predicted_nh3': selected_y_pred.tolist(),
        'anchor_mofs': anchors['m_id'],
        'anchor_nh3': anchors['y'].tolist()[:10],
        'seed': args.seed
    }
    with open(output_dir / 'generation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved to: {output_dir}")
    print("\nNext steps:")
    print(f"  Note: These latent vectors are interpolations of high-NH3 MOF latents.")
    print(f"  Estimated NH3 values are based on the actual values of anchor MOFs.")
    print(f"  To decode to structures, the latents need to be fed to MOFDiff decoder.")
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
