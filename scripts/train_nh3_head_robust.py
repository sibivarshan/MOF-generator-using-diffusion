"""
Robust NH3 Head Training Script

This script addresses the challenge of weak latent-NH3 correlation by:
1. Using log-transformed targets to handle skewed distribution
2. Implementing ranking-based losses for better guidance
3. Using ensemble methods for uncertainty estimation
4. Heavy regularization to prevent overfitting
5. Cross-validation for robust model selection

The trained model enables NH3-guided MOF generation even with weak correlations.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import KFold, train_test_split


# ============================================================================
# Model Architectures
# ============================================================================

class RobustPredictor(nn.Module):
    """
    Robust NH3 predictor with multiple regularization techniques.
    Designed to work with weak correlations.
    """
    def __init__(self, input_dim=256, hidden_dims=[128, 64], dropout=0.3):
        super().__init__()
        
        # Input projection with strong regularization
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
        
        # Initialize with small weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.input_norm(x)
        x = self.hidden(x)
        return self.output(x).squeeze(-1)


class GradientGuidedPredictor(nn.Module):
    """
    Predictor optimized for gradient-based latent optimization.
    Uses smooth activations and skip connections.
    """
    def __init__(self, input_dim=256, hidden_dim=128, n_layers=3, dropout=0.2):
        super().__init__()
        
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),  # Smooth activation for better gradients
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
            x = x + 0.1 * layer(x)  # Small residual connections
        
        return self.output(x).squeeze(-1)


class EnsemblePredictor(nn.Module):
    """Ensemble of multiple small models for uncertainty estimation."""
    def __init__(self, input_dim=256, n_models=5):
        super().__init__()
        
        self.models = nn.ModuleList([
            RobustPredictor(input_dim, hidden_dims=[64 + i*16, 32], dropout=0.2 + i*0.05)
            for i in range(n_models)
        ])
    
    def forward(self, x):
        preds = torch.stack([m(x) for m in self.models], dim=0)
        return preds.mean(dim=0)
    
    def forward_with_uncertainty(self, x):
        preds = torch.stack([m(x) for m in self.models], dim=0)
        return preds.mean(dim=0), preds.std(dim=0)


# ============================================================================
# Loss Functions
# ============================================================================

class RankingLoss(nn.Module):
    """
    Pairwise ranking loss - focuses on relative ordering rather than absolute values.
    Better for guidance when absolute prediction is difficult.
    """
    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin
    
    def forward(self, pred, target):
        n = len(pred)
        if n < 2:
            return torch.tensor(0.0, device=pred.device)
        
        # Create all pairs
        idx_i, idx_j = torch.triu_indices(n, n, offset=1)
        
        pred_i, pred_j = pred[idx_i], pred[idx_j]
        target_i, target_j = target[idx_i], target[idx_j]
        
        # Target direction: 1 if target_i > target_j, -1 otherwise
        target_diff = torch.sign(target_i - target_j)
        pred_diff = pred_i - pred_j
        
        # Hinge loss for ranking
        loss = F.relu(self.margin - target_diff * pred_diff)
        return loss.mean()


class CombinedLoss(nn.Module):
    """Combined MSE + Ranking loss."""
    def __init__(self, ranking_weight=0.3, margin=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ranking = RankingLoss(margin)
        self.ranking_weight = ranking_weight
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        rank_loss = self.ranking(pred, target)
        return mse_loss + self.ranking_weight * rank_loss


# ============================================================================
# Training Utilities
# ============================================================================

class EarlyStopping:
    def __init__(self, patience=30, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = float('inf')
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, score, model):
        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def load_best_model(self, model):
        if self.best_model_state:
            model.load_state_dict(self.best_model_state)


def augment_data(X, y, noise_std=0.02, n_augment=3):
    """Augment data with Gaussian noise."""
    X_aug = [X]
    y_aug = [y]
    
    for _ in range(n_augment):
        noise = torch.randn_like(X) * noise_std
        X_aug.append(X + noise)
        y_aug.append(y)
    
    return torch.cat(X_aug, dim=0), torch.cat(y_aug, dim=0)


def train_single_model(
    model, X_train, y_train, X_val, y_val,
    criterion, device, max_epochs=500, patience=30,
    lr=5e-4, weight_decay=1e-3, use_augmentation=True
):
    """Train a single model."""
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    early_stopping = EarlyStopping(patience=patience)
    
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)
    
    for epoch in range(max_epochs):
        model.train()
        
        # Augment training data each epoch
        if use_augmentation:
            X_aug, y_aug = augment_data(X_train, y_train, noise_std=0.02, n_augment=2)
        else:
            X_aug, y_aug = X_train, y_train
        
        # Shuffle
        perm = torch.randperm(len(X_aug))
        X_aug, y_aug = X_aug[perm], y_aug[perm]
        
        # Mini-batch training
        batch_size = min(32, len(X_aug))
        total_loss = 0
        n_batches = 0
        
        for i in range(0, len(X_aug), batch_size):
            X_batch = X_aug[i:i+batch_size]
            y_batch = y_aug[i:i+batch_size]
            
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = F.mse_loss(val_pred, y_val).item()
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            break
    
    early_stopping.load_best_model(model)
    return model


def evaluate_model(model, X, y, y_mean, y_std, device):
    """Evaluate model performance."""
    model.eval()
    X, y = X.to(device), y.to(device)
    
    with torch.no_grad():
        pred = model(X)
        
        # Denormalize
        pred_denorm = pred * y_std + y_mean
        y_denorm = y * y_std + y_mean
        
        mae = (pred_denorm - y_denorm).abs().mean().item()
        mse = ((pred_denorm - y_denorm) ** 2).mean().item()
        rmse = np.sqrt(mse)
        
        # R² score
        ss_res = ((y_denorm - pred_denorm) ** 2).sum().item()
        ss_tot = ((y_denorm - y_denorm.mean()) ** 2).sum().item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Ranking correlation (Spearman)
        pred_np = pred_denorm.cpu().numpy()
        y_np = y_denorm.cpu().numpy()
        from scipy.stats import spearmanr
        spearman, _ = spearmanr(pred_np, y_np)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'spearman': spearman,
        'preds': pred_denorm.cpu().numpy(),
        'targets': y_denorm.cpu().numpy()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='raw_nh3_core/nh3_latent_dataset.pt')
    parser.add_argument('--output_dir', default='nh3_head_training')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--n_folds', type=int, default=5)
    args = parser.parse_args()
    
    print("=" * 70)
    print("Robust NH3 Head Training")
    print("=" * 70)
    print(f"Device: {args.device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    data = torch.load(args.data_path, weights_only=False)
    X = data['z'].float()
    y = data['y'].float().squeeze()
    
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"NH3 range: [{y.min():.4f}, {y.max():.4f}] mmol/g")
    
    # Log-transform target for better distribution
    y_log = torch.log1p(y)  # log(1 + y) to handle zeros
    
    # Normalize features
    X_mean = X.mean(dim=0)
    X_std = X.std(dim=0).clamp(min=1e-6)
    X_norm = (X - X_mean) / X_std
    
    # Split data
    indices = np.arange(len(X))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_trainval = X_norm[train_val_idx]
    y_trainval = y_log[train_val_idx]
    X_test = X_norm[test_idx]
    y_test = y_log[test_idx]
    
    # Normalize targets based on train data
    y_mean = y_trainval.mean().item()
    y_std = y_trainval.std().item()
    y_trainval_norm = (y_trainval - y_mean) / y_std
    y_test_norm = (y_test - y_mean) / y_std
    
    print(f"\nData splits: {len(train_val_idx)} train+val, {len(test_idx)} test")
    
    # Model configurations to try
    model_configs = [
        ('robust_small', lambda: RobustPredictor(256, [64, 32], 0.3)),
        ('robust_medium', lambda: RobustPredictor(256, [128, 64], 0.3)),
        ('gradient_guided', lambda: GradientGuidedPredictor(256, 128, 3, 0.2)),
        ('ensemble', lambda: EnsemblePredictor(256, 5)),
    ]
    
    loss_configs = [
        ('mse', nn.MSELoss()),
        ('combined', CombinedLoss(ranking_weight=0.3)),
    ]
    
    best_result = {'spearman': -1, 'model': None, 'config': None}
    all_results = []
    
    for model_name, model_fn in model_configs:
        for loss_name, criterion in loss_configs:
            config_name = f"{model_name}_{loss_name}"
            print(f"\n{'='*50}")
            print(f"Training: {config_name}")
            print(f"{'='*50}")
            
            # Cross-validation
            kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
            fold_results = []
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X_trainval)):
                print(f"  Fold {fold+1}/{args.n_folds}", end=" ")
                
                X_train = X_trainval[train_idx]
                y_train = y_trainval_norm[train_idx]
                X_val = X_trainval[val_idx]
                y_val = y_trainval_norm[val_idx]
                
                model = model_fn().to(args.device)
                model = train_single_model(
                    model, X_train, y_train, X_val, y_val,
                    criterion, args.device, args.max_epochs, patience=30
                )
                
                # Evaluate on validation
                # For evaluation, we need to reverse log transform
                metrics = evaluate_model(model, X_val, y_val, y_mean, y_std, args.device)
                # Reverse log transform for MAE
                metrics['mae_original'] = np.mean(np.abs(
                    np.expm1(metrics['preds']) - np.expm1(metrics['targets'])
                ))
                
                fold_results.append(metrics)
                print(f"- MAE: {metrics['mae_original']:.4f}, Spearman: {metrics['spearman']:.4f}")
            
            # Aggregate
            avg_mae = np.mean([r['mae_original'] for r in fold_results])
            avg_spearman = np.mean([r['spearman'] for r in fold_results])
            std_mae = np.std([r['mae_original'] for r in fold_results])
            std_spearman = np.std([r['spearman'] for r in fold_results])
            
            result = {
                'config': config_name,
                'mae_mean': avg_mae,
                'mae_std': std_mae,
                'spearman_mean': avg_spearman,
                'spearman_std': std_spearman,
            }
            all_results.append(result)
            
            print(f"\n  CV Results: MAE={avg_mae:.4f}±{std_mae:.4f}, Spearman={avg_spearman:.4f}±{std_spearman:.4f}")
            
            if avg_spearman > best_result['spearman']:
                best_result = {
                    'spearman': avg_spearman,
                    'config': config_name,
                    'model_fn': model_fn,
                    'criterion': criterion,
                }
    
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL")
    print("=" * 70)
    print(f"Best config: {best_result['config']} (Spearman: {best_result['spearman']:.4f})")
    
    # Train final model on all train+val data
    final_model = best_result['model_fn']().to(args.device)
    final_model = train_single_model(
        final_model, X_trainval, y_trainval_norm, X_test, y_test_norm,
        best_result['criterion'], args.device, args.max_epochs, patience=50
    )
    
    # Final evaluation
    test_metrics = evaluate_model(final_model, X_test, y_test_norm, y_mean, y_std, args.device)
    test_metrics['mae_original'] = np.mean(np.abs(
        np.expm1(test_metrics['preds']) - np.expm1(test_metrics['targets'])
    ))
    
    print(f"\nTest Results:")
    print(f"  MAE: {test_metrics['mae_original']:.4f} mmol/g")
    print(f"  Spearman: {test_metrics['spearman']:.4f}")
    print(f"  R²: {test_metrics['r2']:.4f}")
    
    # Save model
    model_save_path = output_dir / 'nh3_head_best.pt'
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'model_config': best_result['config'],
        'y_mean': y_mean,
        'y_std': y_std,
        'X_mean': X_mean,
        'X_std': X_std,
        'input_dim': X.shape[1],
        'use_log_transform': True,
        'test_metrics': {
            'mae': float(test_metrics['mae_original']),
            'spearman': float(test_metrics['spearman']),
            'r2': float(test_metrics['r2'])
        }
    }, model_save_path)
    print(f"\nSaved model to: {model_save_path}")
    
    # Also save to standard location
    standard_path = Path('raw_nh3_core/nh3_predictor_mlp.pt')
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'model_config': best_result['config'],
        'y_mean': y_mean,
        'y_std': y_std,
        'X_mean': X_mean,
        'X_std': X_std,
        'input_dim': X.shape[1],
        'use_log_transform': True,
        'test_metrics': {
            'mae': float(test_metrics['mae_original']),
            'spearman': float(test_metrics['spearman']),
            'r2': float(test_metrics['r2'])
        }
    }, standard_path)
    print(f"Saved to standard location: {standard_path}")
    
    # Save summary
    summary = {
        'best_config': best_result['config'],
        'test_mae': float(test_metrics['mae_original']),
        'test_spearman': float(test_metrics['spearman']),
        'test_r2': float(test_metrics['r2']),
        'all_results': [{k: float(v) if isinstance(v, (np.floating, float)) else v 
                        for k, v in r.items()} for r in all_results],
        'timestamp': datetime.now().isoformat()
    }
    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nBest Model: {best_result['config']}")
    print(f"Test MAE: {test_metrics['mae_original']:.4f} mmol/g")
    print(f"Spearman Correlation: {test_metrics['spearman']:.4f}")


if __name__ == '__main__':
    main()
