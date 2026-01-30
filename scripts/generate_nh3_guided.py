#!/usr/bin/env python
"""
NH3-Guided MOF Generation using trained ensemble predictor.
Uses gradient-based optimization to find latent vectors that maximize predicted NH3 uptake.
"""

import torch
import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SimpleNet(nn.Module):
    """Simple MLP for NH3 prediction."""
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


class NH3Predictor:
    """Wrapper for NH3 ensemble predictor."""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        self.feature_indices = ckpt['feature_indices']
        self.in_dim = ckpt['in_dim']
        self.hidden_dim = ckpt['hidden_dim']
        
        # Reconstruct scaler
        self.scaler_mean = torch.FloatTensor(ckpt['scaler_mean']).to(device)
        self.scaler_scale = torch.FloatTensor(ckpt['scaler_scale']).to(device)
        
        # Load models
        self.models = []
        for model_state in ckpt['models']:
            model = SimpleNet(self.in_dim, self.hidden_dim).to(device)
            model.load_state_dict(model_state)
            model.eval()
            self.models.append(model)
        
        print(f"Loaded ensemble with {len(self.models)} models")
        print(f"Test Spearman: {ckpt.get('test_spearman', 'N/A'):.4f}")
        print(f"Overall Spearman: {ckpt.get('overall_spearman', 'N/A'):.4f}")
    
    def preprocess(self, z):
        """Preprocess latent vector for prediction."""
        # Select features
        z_selected = z[:, self.feature_indices]
        
        # Create polynomial features
        z_poly = torch.cat([
            z_selected,
            z_selected ** 2,
            torch.log(torch.abs(z_selected) + 1) * torch.sign(z_selected)
        ], dim=1)
        
        # Scale
        z_scaled = (z_poly - self.scaler_mean) / self.scaler_scale
        
        return z_scaled
    
    def predict(self, z, return_log=False):
        """Predict NH3 uptake from latent vector."""
        z_preprocessed = self.preprocess(z)
        
        # Ensemble prediction
        preds = []
        for model in self.models:
            with torch.no_grad():
                pred = model(z_preprocessed)
            preds.append(pred)
        
        # Average predictions (in log space)
        avg_pred = torch.stack(preds).mean(dim=0)
        
        if return_log:
            return avg_pred
        else:
            # Convert from log space
            return torch.exp(avg_pred) - 1
    
    def predict_with_grad(self, z):
        """Predict NH3 with gradients enabled."""
        z_preprocessed = self.preprocess(z)
        
        # Enable gradients for all models
        preds = []
        for model in self.models:
            model.eval()
            pred = model(z_preprocessed)
            preds.append(pred)
        
        # Average predictions (in log space)
        avg_pred = torch.stack(preds).mean(dim=0)
        
        # Return in original scale
        return torch.exp(avg_pred) - 1


def optimize_latent_for_nh3(predictor, start_z, target_nh3, 
                            n_steps=100, lr=0.1, regularization=0.01,
                            max_norm_change=2.0, verbose=True):
    """
    Optimize latent vector to maximize NH3 prediction.
    Uses strong regularization to stay within distribution.
    
    Args:
        predictor: NH3Predictor instance
        start_z: Starting latent vector (1, 256)
        target_nh3: Target NH3 value to optimize towards
        n_steps: Number of optimization steps
        lr: Learning rate
        regularization: L2 regularization to stay near original
        max_norm_change: Maximum allowed change in L2 norm from original
        verbose: Print progress
    
    Returns:
        Optimized latent vector
    """
    z = start_z.clone().requires_grad_(True)
    original_z = start_z.clone()
    original_norm = original_z.norm().item()
    
    optimizer = torch.optim.Adam([z], lr=lr)
    
    best_z = z.clone()
    best_pred = predictor.predict(z.detach()).item()
    
    for step in range(n_steps):
        optimizer.zero_grad()
        
        # Predict NH3
        pred = predictor.predict_with_grad(z)
        
        # Loss: maximize NH3 (minimize negative) + strong regularization
        dist_penalty = ((z - original_z) ** 2).sum()
        loss = -pred.mean() + regularization * dist_penalty
        
        loss.backward()
        optimizer.step()
        
        # Project back if too far from original
        with torch.no_grad():
            delta = z - original_z
            delta_norm = delta.norm().item()
            if delta_norm > max_norm_change:
                z.copy_(original_z + delta * (max_norm_change / delta_norm))
            
            current_pred = predictor.predict(z).item()
            
            # Only keep if prediction is realistic (< 20 mmol/g)
            if current_pred > best_pred and current_pred < 20:
                best_pred = current_pred
                best_z = z.clone()
        
        if verbose and (step + 1) % 20 == 0:
            print(f"  Step {step+1}/{n_steps}: pred={current_pred:.2f}, best={best_pred:.2f}")
    
    return best_z.detach(), best_pred


def find_high_nh3_samples(predictor, latent_data, device='cuda'):
    """Find samples with highest predicted NH3 in the dataset."""
    z_all = latent_data['z'].to(device)
    
    # Predict all
    with torch.no_grad():
        preds = predictor.predict(z_all)
    
    # Sort by prediction
    sorted_idx = torch.argsort(preds, descending=True)
    
    return sorted_idx, preds


def interpolate_to_high_nh3(predictor, start_z, high_z, n_steps=10):
    """Interpolate between start and high-NH3 latent, finding best point."""
    best_z = start_z.clone()
    best_pred = 0
    
    for alpha in np.linspace(0, 1, n_steps):
        z = (1 - alpha) * start_z + alpha * high_z
        with torch.no_grad():
            pred = predictor.predict(z).item()
        
        if pred > best_pred:
            best_pred = pred
            best_z = z.clone()
            best_alpha = alpha
    
    return best_z, best_pred


def main():
    parser = argparse.ArgumentParser(description='NH3-Guided MOF Generation')
    parser.add_argument('--checkpoint', type=str, 
                        default='nh3_head_training/nh3_head_ensemble.pt',
                        help='Path to NH3 predictor checkpoint')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_samples', type=int, default=5,
                        help='Number of samples to generate')
    parser.add_argument('--n_steps', type=int, default=100,
                        help='Optimization steps per sample')
    parser.add_argument('--output', type=str, default='nh3_optimized_latents.pt',
                        help='Output file for optimized latents')
    args = parser.parse_args()
    
    print('='*70)
    print('NH3-GUIDED MOF GENERATION')
    print('='*70)
    
    # Load predictor
    print('\n1. Loading NH3 Predictor...')
    predictor = NH3Predictor(args.checkpoint, args.device)
    
    # Load latent dataset
    print('\n2. Loading latent dataset...')
    latent_data = torch.load('raw_nh3_core/nh3_latent_dataset.pt')
    z_all = latent_data['z'].to(args.device)
    y_all = latent_data['y'].to(args.device)
    
    print(f'   Dataset size: {len(y_all)}')
    print(f'   NH3 range: [{y_all.min():.2f}, {y_all.max():.2f}]')
    
    # Find high-NH3 samples
    print('\n3. Finding high-NH3 samples...')
    sorted_idx, preds = find_high_nh3_samples(predictor, latent_data, args.device)
    
    print('\n   Top 10 predicted NH3:')
    for i in range(10):
        idx = sorted_idx[i].item()
        print(f'   {i+1}. Predicted={preds[idx]:.2f}, True={y_all[idx]:.2f}')
    
    # Generate optimized samples
    print(f'\n4. Generating {args.n_samples} optimized samples...')
    
    optimized_latents = []
    optimized_preds = []
    
    for i in range(args.n_samples):
        print(f'\n   Sample {i+1}/{args.n_samples}:')
        
        # Start from random sample in top 50%
        start_idx = np.random.choice(len(z_all) // 2)
        start_z = z_all[start_idx:start_idx+1]
        
        # Method 1: Gradient optimization (conservative)
        print('   Method 1: Gradient optimization')
        opt_z_grad, pred_grad = optimize_latent_for_nh3(
            predictor, start_z, target_nh3=10.0,
            n_steps=args.n_steps, lr=0.05, regularization=0.1,
            max_norm_change=2.0, verbose=True
        )
        print(f'   -> Gradient: pred={pred_grad:.2f}')
        
        # Method 2: Interpolation to best sample
        print('   Method 2: Interpolation to best sample')
        best_idx = sorted_idx[0].item()
        best_z = z_all[best_idx:best_idx+1]
        opt_z_interp, pred_interp = interpolate_to_high_nh3(
            predictor, start_z, best_z, n_steps=20
        )
        print(f'   -> Interpolation: pred={pred_interp:.2f}')
        
        # Use better result
        if pred_grad > pred_interp:
            optimized_latents.append(opt_z_grad)
            optimized_preds.append(pred_grad)
            print(f'   -> Using gradient result: {pred_grad:.2f}')
        else:
            optimized_latents.append(opt_z_interp)
            optimized_preds.append(pred_interp)
            print(f'   -> Using interpolation result: {pred_interp:.2f}')
    
    # Save results
    print(f'\n5. Saving results to {args.output}...')
    
    results = {
        'latents': torch.cat(optimized_latents, dim=0),
        'predicted_nh3': torch.tensor(optimized_preds),
        'n_samples': args.n_samples
    }
    torch.save(results, args.output)
    
    print('\n   Summary:')
    print(f'   Mean predicted NH3: {np.mean(optimized_preds):.2f} mmol/g')
    print(f'   Max predicted NH3: {np.max(optimized_preds):.2f} mmol/g')
    print(f'   Min predicted NH3: {np.min(optimized_preds):.2f} mmol/g')
    
    print('\n' + '='*70)
    print('DONE!')
    print('='*70)


if __name__ == '__main__':
    main()
