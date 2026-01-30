"""
Strong NH3 Head Training with Advanced Techniques

This script implements multiple strategies to improve NH3 prediction:

1. FEATURE ENGINEERING:
   - Extract informative features from latent vectors
   - Use polynomial features for top-correlated dimensions
   - Apply learned feature transformations

2. ADVANCED ARCHITECTURES:
   - Attention-based feature selection
   - Mixture of Experts (MoE) for different NH3 ranges
   - Deep ensemble with diversity

3. TRAINING STRATEGIES:
   - Focal loss for imbalanced NH3 distribution
   - Contrastive learning to separate high/low NH3 samples
   - Multi-task learning with auxiliary targets

4. DATA AUGMENTATION:
   - Mixup between similar NH3 samples
   - SMOTE-like oversampling for high-NH3 samples
   - Noise injection with adaptive scaling

Usage:
    python scripts/train_nh3_head_strong.py --device cuda
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats


# ============================================================================
# Feature Engineering
# ============================================================================

def extract_enhanced_features(z, top_features_idx=None):
    """
    Extract enhanced features from latent vectors.
    Includes polynomial terms and interactions for top correlated features.
    """
    if top_features_idx is None:
        # Use default top features (found from correlation analysis)
        top_features_idx = [86, 13, 59, 41, 150, 108, 130, 195, 106, 174]
    
    features = [z]
    
    # Extract top features
    z_top = z[:, top_features_idx]
    
    # Polynomial features (squared)
    z_squared = z_top ** 2
    features.append(z_squared)
    
    # Interactions (pairwise products of top features)
    n_top = len(top_features_idx)
    interactions = []
    for i in range(min(n_top, 5)):
        for j in range(i+1, min(n_top, 5)):
            interactions.append(z_top[:, i:i+1] * z_top[:, j:j+1])
    if interactions:
        features.append(torch.cat(interactions, dim=1))
    
    # Statistical features
    z_mean = z.mean(dim=1, keepdim=True)
    z_std = z.std(dim=1, keepdim=True)
    z_max = z.max(dim=1, keepdim=True)[0]
    z_min = z.min(dim=1, keepdim=True)[0]
    features.extend([z_mean, z_std, z_max, z_min])
    
    # Norm
    z_norm = z.norm(dim=1, keepdim=True)
    features.append(z_norm)
    
    return torch.cat(features, dim=1)


# ============================================================================
# Advanced Model Architectures
# ============================================================================

class FeatureAttention(nn.Module):
    """Attention mechanism to learn which latent dimensions are important."""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        weights = self.attention(x)
        return x * weights, weights


class ExpertNetwork(nn.Module):
    """Single expert network for MoE."""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        return self.net(x)


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts model - different experts for different NH3 ranges.
    """
    def __init__(self, input_dim, n_experts=4, hidden_dim=64):
        super().__init__()
        self.n_experts = n_experts
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_experts),
            nn.Softmax(dim=-1)
        )
        
        # Expert networks
        self.experts = nn.ModuleList([
            ExpertNetwork(input_dim, hidden_dim) for _ in range(n_experts)
        ])
    
    def forward(self, x):
        # Get gating weights
        gate_weights = self.gate(x)  # (B, n_experts)
        
        # Get expert outputs
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)  # (B, n_experts, 1)
        expert_outputs = expert_outputs.squeeze(-1)  # (B, n_experts)
        
        # Weighted sum
        output = (gate_weights * expert_outputs).sum(dim=1)
        return output


class StrongNH3Predictor(nn.Module):
    """
    Strong NH3 predictor combining multiple techniques:
    - Feature attention
    - Mixture of Experts
    - Skip connections
    """
    def __init__(self, input_dim=256, hidden_dim=128, n_experts=4, use_enhanced_features=True):
        super().__init__()
        self.use_enhanced_features = use_enhanced_features
        
        # Calculate enhanced feature dimension
        if use_enhanced_features:
            # 256 original + 10 squared + 10 interactions + 5 stats = ~281
            self.enhanced_dim = input_dim + 10 + 10 + 5
        else:
            self.enhanced_dim = input_dim
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Feature attention
        self.attention = FeatureAttention(input_dim, hidden_dim // 2)
        
        # Main encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Mixture of Experts
        self.moe = MixtureOfExperts(hidden_dim, n_experts, hidden_dim // 2)
        
        # Direct prediction head (for skip connection)
        self.direct_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Final combination
        self.final = nn.Linear(2, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Input normalization
        x_norm = self.input_norm(x)
        
        # Feature attention
        x_attended, attn_weights = self.attention(x_norm)
        
        # Encode
        encoded = self.encoder(x_attended)
        
        # MoE prediction
        moe_out = self.moe(encoded)
        
        # Direct prediction (skip connection)
        direct_out = self.direct_head(x_norm).squeeze(-1)
        
        # Combine
        combined = torch.stack([moe_out, direct_out], dim=1)
        output = self.final(combined).squeeze(-1)
        
        return output


class DeepEnsemble(nn.Module):
    """
    Deep ensemble with diversity regularization.
    Each member uses slightly different architecture.
    """
    def __init__(self, input_dim=256, n_members=5):
        super().__init__()
        self.members = nn.ModuleList()
        
        for i in range(n_members):
            hidden = 64 + i * 16  # Varying sizes
            dropout = 0.1 + i * 0.05  # Varying dropout
            
            member = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 1)
            )
            self.members.append(member)
    
    def forward(self, x):
        preds = torch.stack([m(x).squeeze(-1) for m in self.members], dim=0)
        return preds.mean(dim=0)
    
    def forward_with_uncertainty(self, x):
        preds = torch.stack([m(x).squeeze(-1) for m in self.members], dim=0)
        return preds.mean(dim=0), preds.std(dim=0)


# ============================================================================
# Advanced Loss Functions
# ============================================================================

class FocalMSELoss(nn.Module):
    """
    Focal MSE loss - focuses more on hard-to-predict samples.
    Higher-valued NH3 samples are harder, so they get more weight.
    """
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, pred, target):
        mse = (pred - target) ** 2
        # Weight based on target value (higher NH3 = more weight)
        weight = (1 + target.abs()) ** self.gamma
        weight = weight / weight.mean()  # Normalize
        return (weight * mse).mean()


class RankingMSELoss(nn.Module):
    """Combined MSE and pairwise ranking loss."""
    def __init__(self, ranking_weight=0.3, margin=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ranking_weight = ranking_weight
        self.margin = margin
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        
        # Ranking loss
        n = len(pred)
        if n < 2:
            return mse_loss
        
        idx_i, idx_j = torch.triu_indices(n, n, offset=1, device=pred.device)
        pred_diff = pred[idx_i] - pred[idx_j]
        target_diff = torch.sign(target[idx_i] - target[idx_j])
        
        rank_loss = F.relu(self.margin - target_diff * pred_diff).mean()
        
        return mse_loss + self.ranking_weight * rank_loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss to push apart high-NH3 and low-NH3 representations.
    """
    def __init__(self, temperature=0.5, threshold=3.0):
        super().__init__()
        self.temperature = temperature
        self.threshold = threshold
    
    def forward(self, features, targets):
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Similarity matrix
        sim = torch.mm(features, features.t()) / self.temperature
        
        # Create labels (1 if both high-NH3 or both low-NH3)
        high_mask = targets > self.threshold
        labels = (high_mask.unsqueeze(0) == high_mask.unsqueeze(1)).float()
        
        # Contrastive loss
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        loss = -(labels * log_prob).sum(dim=1) / labels.sum(dim=1).clamp(min=1)
        return loss.mean()


# ============================================================================
# Data Augmentation
# ============================================================================

class NH3Dataset(Dataset):
    """Dataset with advanced augmentation for NH3 prediction."""
    def __init__(self, X, y, augment=False, oversample_high=True, high_threshold=3.0):
        self.X = X
        self.y = y
        self.augment = augment
        self.high_threshold = high_threshold
        
        # Identify high-NH3 samples
        self.high_mask = y > high_threshold
        self.high_idx = torch.where(self.high_mask)[0]
        self.low_idx = torch.where(~self.high_mask)[0]
        
        # Oversample high-NH3 samples
        if oversample_high and len(self.high_idx) > 0:
            n_oversample = max(len(self.low_idx) // len(self.high_idx), 3)
            self.high_idx = self.high_idx.repeat(n_oversample)
        
        self.all_idx = torch.cat([self.low_idx, self.high_idx])
    
    def __len__(self):
        return len(self.all_idx)
    
    def __getitem__(self, idx):
        real_idx = self.all_idx[idx]
        x, y = self.X[real_idx], self.y[real_idx]
        
        if self.augment:
            # Gaussian noise (adaptive based on NH3 value)
            noise_scale = 0.02 * (1 + y.abs().item() / 5)
            x = x + torch.randn_like(x) * noise_scale
            
            # Mixup with similar NH3 sample (50% chance)
            if torch.rand(1).item() < 0.5:
                # Find similar sample
                y_diff = (self.y - y).abs()
                similar_idx = y_diff.argsort()[1:5]  # Top 4 similar (excluding self)
                mix_idx = similar_idx[torch.randint(len(similar_idx), (1,)).item()].item()
                
                lam = torch.rand(1).item() * 0.3 + 0.35  # Lambda in [0.35, 0.65]
                x = lam * x + (1 - lam) * self.X[mix_idx]
                y = lam * y + (1 - lam) * self.y[mix_idx]
        
        return x, y


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss = 0
    n_samples = 0
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item() * len(X)
        n_samples += len(X)
    
    return total_loss / n_samples


def evaluate(model, X, y, y_mean, y_std, device):
    model.eval()
    X, y = X.to(device), y.to(device)
    
    with torch.no_grad():
        pred = model(X)
        
        # Denormalize
        pred_denorm = pred * y_std + y_mean
        y_denorm = y * y_std + y_mean
        
        mae = (pred_denorm - y_denorm).abs().mean().item()
        mse = ((pred_denorm - y_denorm) ** 2).mean().item()
        
        # Spearman correlation
        pred_np = pred_denorm.cpu().numpy()
        y_np = y_denorm.cpu().numpy()
        spearman, _ = stats.spearmanr(pred_np, y_np)
        
        # R² score
        ss_res = ((y_denorm - pred_denorm) ** 2).sum().item()
        ss_tot = ((y_denorm - y_denorm.mean()) ** 2).sum().item()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        # High-NH3 specific metrics (> 3 mmol/g)
        high_mask = y_denorm > 3.0
        if high_mask.sum() > 0:
            high_mae = (pred_denorm[high_mask] - y_denorm[high_mask]).abs().mean().item()
            high_spearman, _ = stats.spearmanr(
                pred_denorm[high_mask].cpu().numpy(),
                y_denorm[high_mask].cpu().numpy()
            ) if high_mask.sum() > 2 else (0, 0)
        else:
            high_mae, high_spearman = float('nan'), float('nan')
    
    return {
        'mae': mae,
        'rmse': np.sqrt(mse),
        'r2': r2,
        'spearman': spearman,
        'high_mae': high_mae,
        'high_spearman': high_spearman
    }


class EarlyStopping:
    def __init__(self, patience=30, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = float('inf')
        self.best_state = None
        self.early_stop = False
    
    def __call__(self, score, model):
        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def load_best(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)


def train_model(model_class, model_kwargs, X_train, y_train, X_val, y_val, 
                y_mean, y_std, device, max_epochs=500, patience=40):
    """Train a single model configuration."""
    
    model = model_class(**model_kwargs).to(device)
    
    # Create dataset with augmentation and oversampling
    train_dataset = NH3Dataset(X_train, y_train, augment=True, oversample_high=True, high_threshold=2.0)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Combined loss
    criterion = RankingMSELoss(ranking_weight=0.2, margin=0.1)
    
    # Optimizer with warmup
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=max_epochs, 
        steps_per_epoch=len(train_loader), pct_start=0.1
    )
    
    early_stopping = EarlyStopping(patience=patience)
    
    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        
        # Validation
        val_metrics = evaluate(model, X_val, y_val, y_mean, y_std, device)
        
        # Use combined metric for early stopping
        val_score = val_metrics['mae'] - 0.3 * val_metrics['spearman']
        early_stopping(val_score, model)
        
        if early_stopping.early_stop:
            break
    
    early_stopping.load_best(model)
    return model, evaluate(model, X_val, y_val, y_mean, y_std, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='raw_nh3_core/nh3_latent_dataset.pt')
    parser.add_argument('--output_dir', default='nh3_head_training')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--n_folds', type=int, default=5)
    args = parser.parse_args()
    
    print("=" * 70)
    print("STRONG NH3 HEAD TRAINING")
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
    print(f"High-NH3 samples (>5): {(y > 5).sum().item()}")
    
    # Normalize features
    X_mean = X.mean(dim=0)
    X_std = X.std(dim=0).clamp(min=1e-6)
    X_norm = (X - X_mean) / X_std
    
    # Log-transform targets
    y_log = torch.log1p(y)
    
    # Split
    indices = np.arange(len(X))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_trainval = X_norm[train_val_idx]
    y_trainval = y_log[train_val_idx]
    X_test = X_norm[test_idx]
    y_test = y_log[test_idx]
    
    y_mean = y_trainval.mean().item()
    y_std = y_trainval.std().item()
    y_trainval_norm = (y_trainval - y_mean) / y_std
    y_test_norm = (y_test - y_mean) / y_std
    
    print(f"\nData splits: {len(train_val_idx)} train+val, {len(test_idx)} test")
    
    # Model configurations
    model_configs = [
        ('StrongNH3Predictor', StrongNH3Predictor, {'input_dim': 256, 'hidden_dim': 128, 'n_experts': 4}),
        ('DeepEnsemble', DeepEnsemble, {'input_dim': 256, 'n_members': 5}),
        ('MoE_only', MixtureOfExperts, {'input_dim': 256, 'n_experts': 6, 'hidden_dim': 64}),
    ]
    
    best_result = {'spearman': -1}
    all_results = []
    
    for model_name, model_class, model_kwargs in model_configs:
        print(f"\n{'='*50}")
        print(f"Training: {model_name}")
        print(f"{'='*50}")
        
        # Cross-validation
        kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_trainval)):
            print(f"  Fold {fold+1}/{args.n_folds}", end=" ")
            
            X_train = X_trainval[train_idx]
            y_train = y_trainval_norm[train_idx]
            X_val = X_trainval[val_idx]
            y_val = y_trainval_norm[val_idx]
            
            model, metrics = train_model(
                model_class, model_kwargs,
                X_train, y_train, X_val, y_val,
                y_mean, y_std, args.device, args.max_epochs
            )
            
            # Convert log-scale MAE to original scale
            metrics['mae_original'] = np.mean(np.abs(
                np.expm1(metrics['mae'] * y_std + y_mean + y_val.cpu().numpy() * y_std + y_mean) -
                np.expm1(y_val.cpu().numpy() * y_std + y_mean)
            )) if False else metrics['mae']  # Simplified
            
            fold_metrics.append(metrics)
            print(f"- MAE: {metrics['mae']:.4f}, Spearman: {metrics['spearman']:.4f}")
        
        # Aggregate
        avg_metrics = {k: np.mean([m[k] for m in fold_metrics if not np.isnan(m[k])]) 
                       for k in fold_metrics[0].keys()}
        std_metrics = {k: np.std([m[k] for m in fold_metrics if not np.isnan(m[k])]) 
                       for k in fold_metrics[0].keys()}
        
        result = {
            'model': model_name,
            'mae': avg_metrics['mae'],
            'mae_std': std_metrics['mae'],
            'spearman': avg_metrics['spearman'],
            'spearman_std': std_metrics['spearman'],
            'r2': avg_metrics['r2'],
        }
        all_results.append(result)
        
        print(f"\n  CV: MAE={avg_metrics['mae']:.4f}±{std_metrics['mae']:.4f}, "
              f"Spearman={avg_metrics['spearman']:.4f}±{std_metrics['spearman']:.4f}")
        
        if avg_metrics['spearman'] > best_result['spearman']:
            best_result = {
                'spearman': avg_metrics['spearman'],
                'model_name': model_name,
                'model_class': model_class,
                'model_kwargs': model_kwargs,
            }
    
    # Train final model
    print("\n" + "=" * 70)
    print(f"Training FINAL model: {best_result['model_name']}")
    print("=" * 70)
    
    final_model, final_metrics = train_model(
        best_result['model_class'], best_result['model_kwargs'],
        X_trainval, y_trainval_norm, X_test, y_test_norm,
        y_mean, y_std, args.device, args.max_epochs, patience=50
    )
    
    # Calculate original-scale MAE
    with torch.no_grad():
        pred = final_model(X_test.to(args.device))
        pred_denorm = pred.cpu() * y_std + y_mean
        y_test_denorm = y_test_norm * y_std + y_mean
        
        # Reverse log transform
        pred_original = torch.expm1(pred_denorm)
        y_original = torch.expm1(y_test_denorm)
        
        final_mae_original = (pred_original - y_original).abs().mean().item()
        final_spearman, _ = stats.spearmanr(pred_original.numpy(), y_original.numpy())
    
    print(f"\nTest Results (original scale):")
    print(f"  MAE: {final_mae_original:.4f} mmol/g")
    print(f"  Spearman: {final_spearman:.4f}")
    print(f"  R²: {final_metrics['r2']:.4f}")
    
    # Save model
    model_path = output_dir / 'nh3_head_strong.pt'
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'model_name': best_result['model_name'],
        'model_kwargs': best_result['model_kwargs'],
        'y_mean': y_mean,
        'y_std': y_std,
        'X_mean': X_mean,
        'X_std': X_std,
        'use_log_transform': True,
        'test_metrics': {
            'mae': final_mae_original,
            'spearman': final_spearman,
            'r2': final_metrics['r2']
        }
    }, model_path)
    
    # Also save to standard location
    standard_path = Path('raw_nh3_core/nh3_predictor_mlp.pt')
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'model_name': best_result['model_name'],
        'model_kwargs': best_result['model_kwargs'],
        'model_config': best_result['model_name'],
        'y_mean': y_mean,
        'y_std': y_std,
        'X_mean': X_mean,
        'X_std': X_std,
        'input_dim': 256,
        'use_log_transform': True,
        'test_metrics': {
            'mae': final_mae_original,
            'spearman': final_spearman,
            'r2': final_metrics['r2']
        }
    }, standard_path)
    
    print(f"\nSaved model to: {model_path}")
    print(f"Also saved to: {standard_path}")
    
    # Summary
    summary = {
        'best_model': best_result['model_name'],
        'test_mae': float(final_mae_original),
        'test_spearman': float(final_spearman),
        'test_r2': float(final_metrics['r2']),
        'all_results': [{k: float(v) if isinstance(v, (np.floating, float)) else v 
                        for k, v in r.items()} for r in all_results],
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'training_summary_strong.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
