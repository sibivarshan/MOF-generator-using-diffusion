"""
Comprehensive NH3 Head Training Script

This script trains multiple NH3 prediction models with different architectures
and hyperparameters to find the best model for guiding latent vector optimization.

Features:
- Multiple model architectures (MLP, ResNet, Attention-based)
- Hyperparameter grid search
- Data augmentation (noise injection, mixup)
- Proper train/val/test split with cross-validation
- Early stopping to prevent overfitting
- Learning rate scheduling
- Comprehensive evaluation metrics
- GPU support

Usage:
    python scripts/train_nh3_head_comprehensive.py --data_path raw_nh3_core/nh3_latent_dataset.pt
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
from sklearn.preprocessing import StandardScaler

# Try importing matplotlib, but make it optional
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plots will be skipped")


# ============================================================================
# Model Architectures
# ============================================================================

class SimpleMLP(nn.Module):
    """Simple MLP baseline."""
    def __init__(self, input_dim=256, hidden_dims=[128, 64], dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


class MLPWithBatchNorm(nn.Module):
    """MLP with BatchNorm for better generalization."""
    def __init__(self, input_dim=256, hidden_dims=[256, 128, 64], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout if i < len(hidden_dims) - 1 else dropout/2),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


class MLPWithLayerNorm(nn.Module):
    """MLP with LayerNorm - works better with small batches."""
    def __init__(self, input_dim=256, hidden_dims=[256, 128, 64], dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.activation = nn.GELU()
    
    def forward(self, x):
        return self.activation(x + self.block(x))


class ResNetMLP(nn.Module):
    """MLP with residual connections."""
    def __init__(self, input_dim=256, hidden_dim=256, n_blocks=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)
        ])
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x).squeeze(-1)


class AttentionMLP(nn.Module):
    """MLP with self-attention for capturing feature interactions."""
    def __init__(self, input_dim=256, hidden_dim=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # Project input
        x = self.input_proj(x)
        # Add batch dimension for attention
        x = x.unsqueeze(1)  # (B, 1, D)
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        # FFN
        x = self.norm2(x + self.ffn(x))
        # Output
        x = x.squeeze(1)  # (B, D)
        return self.output(x).squeeze(-1)


class EnsembleModel(nn.Module):
    """Ensemble of multiple models."""
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, x):
        preds = torch.stack([m(x) for m in self.models], dim=0)
        return preds.mean(dim=0)


# ============================================================================
# Data Augmentation
# ============================================================================

class LatentDataset(Dataset):
    """Dataset with optional augmentation."""
    def __init__(self, X, y, augment=False, noise_std=0.05, mixup_alpha=0.2):
        self.X = X
        self.y = y
        self.augment = augment
        self.noise_std = noise_std
        self.mixup_alpha = mixup_alpha
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        
        if self.augment and self.training:
            # Gaussian noise injection
            if np.random.random() < 0.5:
                noise = torch.randn_like(x) * self.noise_std
                x = x + noise
            
            # Mixup with random sample
            if np.random.random() < 0.3:
                mix_idx = np.random.randint(len(self.X))
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                x = lam * x + (1 - lam) * self.X[mix_idx]
                y = lam * y + (1 - lam) * self.y[mix_idx]
        
        return x, y
    
    @property
    def training(self):
        return self.augment


# ============================================================================
# Training Utilities
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=20, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif self._is_improvement(score):
            self.best_score = score
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta
    
    def load_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


def train_epoch(model, train_loader, optimizer, criterion, device, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_samples = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item() * len(X_batch)
        n_samples += len(X_batch)
    
    return total_loss / n_samples


def evaluate(model, data_loader, criterion, device, y_mean=0, y_std=1):
    """Evaluate model on data."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            total_loss += loss.item() * len(X_batch)
            
            # Denormalize for metrics
            pred_denorm = pred * y_std + y_mean
            target_denorm = y_batch * y_std + y_mean
            all_preds.extend(pred_denorm.cpu().numpy())
            all_targets.extend(target_denorm.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    mse = np.mean((all_preds - all_targets) ** 2)
    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(mse)
    
    # R² score
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'loss': total_loss / len(all_preds),
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'preds': all_preds,
        'targets': all_targets
    }


# ============================================================================
# Model Configurations
# ============================================================================

MODEL_CONFIGS = {
    'simple_small': {
        'class': SimpleMLP,
        'params': {'hidden_dims': [64, 32], 'dropout': 0.1},
        'description': 'Simple MLP (small)'
    },
    'simple_medium': {
        'class': SimpleMLP,
        'params': {'hidden_dims': [128, 64], 'dropout': 0.2},
        'description': 'Simple MLP (medium)'
    },
    'batchnorm_small': {
        'class': MLPWithBatchNorm,
        'params': {'hidden_dims': [128, 64], 'dropout': 0.2},
        'description': 'MLP with BatchNorm (small)'
    },
    'batchnorm_medium': {
        'class': MLPWithBatchNorm,
        'params': {'hidden_dims': [256, 128, 64], 'dropout': 0.2},
        'description': 'MLP with BatchNorm (medium)'
    },
    'batchnorm_large': {
        'class': MLPWithBatchNorm,
        'params': {'hidden_dims': [512, 256, 128], 'dropout': 0.3},
        'description': 'MLP with BatchNorm (large)'
    },
    'layernorm_small': {
        'class': MLPWithLayerNorm,
        'params': {'hidden_dims': [128, 64], 'dropout': 0.1},
        'description': 'MLP with LayerNorm (small)'
    },
    'layernorm_medium': {
        'class': MLPWithLayerNorm,
        'params': {'hidden_dims': [256, 128, 64], 'dropout': 0.1},
        'description': 'MLP with LayerNorm (medium)'
    },
    'resnet_small': {
        'class': ResNetMLP,
        'params': {'hidden_dim': 128, 'n_blocks': 2, 'dropout': 0.1},
        'description': 'ResNet MLP (small)'
    },
    'resnet_medium': {
        'class': ResNetMLP,
        'params': {'hidden_dim': 256, 'n_blocks': 3, 'dropout': 0.1},
        'description': 'ResNet MLP (medium)'
    },
    'resnet_large': {
        'class': ResNetMLP,
        'params': {'hidden_dim': 256, 'n_blocks': 5, 'dropout': 0.2},
        'description': 'ResNet MLP (large)'
    },
    'attention_small': {
        'class': AttentionMLP,
        'params': {'hidden_dim': 64, 'n_heads': 2, 'dropout': 0.1},
        'description': 'Attention MLP (small)'
    },
    'attention_medium': {
        'class': AttentionMLP,
        'params': {'hidden_dim': 128, 'n_heads': 4, 'dropout': 0.1},
        'description': 'Attention MLP (medium)'
    },
}

HYPERPARAMETER_CONFIGS = [
    {'lr': 1e-3, 'weight_decay': 1e-4, 'batch_size': 32},
    {'lr': 5e-4, 'weight_decay': 1e-4, 'batch_size': 32},
    {'lr': 1e-3, 'weight_decay': 1e-3, 'batch_size': 16},
    {'lr': 5e-4, 'weight_decay': 1e-5, 'batch_size': 64},
]


# ============================================================================
# Main Training Function
# ============================================================================

def train_model(
    model_config: Dict,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    y_mean: float,
    y_std: float,
    hyperparams: Dict,
    device: str = 'cuda',
    max_epochs: int = 500,
    patience: int = 30,
    use_augmentation: bool = True,
    verbose: bool = True
) -> Tuple[nn.Module, Dict]:
    """Train a single model configuration."""
    
    # Create model
    model = model_config['class'](input_dim=X_train.shape[1], **model_config['params']).to(device)
    
    # Create data loaders
    if use_augmentation:
        train_dataset = LatentDataset(X_train, y_train, augment=True, noise_std=0.02)
    else:
        train_dataset = TensorDataset(X_train, y_train)
    
    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=hyperparams['batch_size'])
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=hyperparams['lr'], weight_decay=hyperparams['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    criterion = nn.MSELoss()
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, mode='min')
    
    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_r2': []}
    
    for epoch in range(max_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, y_mean, y_std)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_r2'].append(val_metrics['r2'])
        
        # Early stopping
        early_stopping(val_metrics['loss'], model)
        
        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, val_mae={val_metrics['mae']:.4f}, val_r2={val_metrics['r2']:.4f}")
        
        if early_stopping.early_stop:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    early_stopping.load_best_model(model)
    
    # Final evaluation
    final_metrics = evaluate(model, val_loader, criterion, device, y_mean, y_std)
    final_metrics['epochs_trained'] = epoch + 1
    final_metrics['history'] = history
    
    return model, final_metrics


def cross_validate_model(
    model_config: Dict,
    X: torch.Tensor,
    y: torch.Tensor,
    hyperparams: Dict,
    device: str = 'cuda',
    n_folds: int = 5,
    max_epochs: int = 500,
    verbose: bool = True
) -> Tuple[List[nn.Module], Dict]:
    """Perform k-fold cross-validation."""
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    fold_models = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        if verbose:
            print(f"\n  Fold {fold + 1}/{n_folds}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Normalize using training data
        y_mean = y_train.mean().item()
        y_std = y_train.std().item()
        y_train_norm = (y_train - y_mean) / y_std
        y_val_norm = (y_val - y_mean) / y_std
        
        model, metrics = train_model(
            model_config, X_train, y_train_norm, X_val, y_val_norm,
            y_mean, y_std, hyperparams, device, max_epochs, 
            patience=30, verbose=False
        )
        
        fold_results.append(metrics)
        fold_models.append(model)
        
        if verbose:
            print(f"    MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
    
    # Aggregate results
    cv_results = {
        'mae_mean': np.mean([r['mae'] for r in fold_results]),
        'mae_std': np.std([r['mae'] for r in fold_results]),
        'r2_mean': np.mean([r['r2'] for r in fold_results]),
        'r2_std': np.std([r['r2'] for r in fold_results]),
        'rmse_mean': np.mean([r['rmse'] for r in fold_results]),
        'fold_results': fold_results
    }
    
    return fold_models, cv_results


def main():
    parser = argparse.ArgumentParser(description='Train NH3 prediction head')
    parser.add_argument('--data_path', type=str, default='raw_nh3_core/nh3_latent_dataset.pt',
                        help='Path to latent dataset')
    parser.add_argument('--output_dir', type=str, default='nh3_head_training',
                        help='Output directory for models and results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--max_epochs', type=int, default=500, help='Maximum epochs')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--quick', action='store_true', help='Quick mode - fewer models and epochs')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Specific models to train (default: all)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("NH3 Head Comprehensive Training")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Data path: {args.data_path}")
    print(f"Output dir: {args.output_dir}")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading data...")
    data = torch.load(args.data_path, weights_only=False)
    
    X = data['z'].float()
    y = data['y'].float().squeeze()
    
    print(f"Dataset size: {len(X)} samples")
    print(f"Latent dimension: {X.shape[1]}")
    print(f"NH3 uptake range: [{y.min():.2f}, {y.max():.2f}] mmol/g")
    print(f"NH3 uptake mean: {y.mean():.2f} ± {y.std():.2f} mmol/g")
    
    # Normalize input features
    X_mean = X.mean(dim=0)
    X_std = X.std(dim=0).clamp(min=1e-6)
    X_normalized = (X - X_mean) / X_std
    
    # Move to device
    X_normalized = X_normalized.to(args.device)
    y = y.to(args.device)
    X_mean = X_mean.to(args.device)
    X_std = X_std.to(args.device)
    
    # Split data: 80% train+val, 20% test
    indices = np.arange(len(X))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_trainval = X_normalized[train_val_idx]
    y_trainval = y[train_val_idx]
    X_test = X_normalized[test_idx]
    y_test = y[test_idx]
    
    print(f"\nData splits: {len(train_val_idx)} train+val, {len(test_idx)} test")
    
    # Determine which models to train
    if args.quick:
        model_names = ['simple_medium', 'batchnorm_medium', 'layernorm_medium', 'resnet_small']
        hyperparams_list = [HYPERPARAMETER_CONFIGS[0]]  # Just one config
        max_epochs = 200
    else:
        model_names = args.models if args.models else list(MODEL_CONFIGS.keys())
        hyperparams_list = HYPERPARAMETER_CONFIGS
        max_epochs = args.max_epochs
    
    print(f"\nModels to train: {model_names}")
    print(f"Hyperparameter configs: {len(hyperparams_list)}")
    print()
    
    # Store all results
    all_results = {}
    best_overall = {'mae': float('inf'), 'model_name': None, 'config': None}
    
    # Train all model configurations
    for model_name in model_names:
        model_config = MODEL_CONFIGS[model_name]
        print(f"\n{'='*60}")
        print(f"Training: {model_config['description']}")
        print(f"{'='*60}")
        
        for hp_idx, hyperparams in enumerate(hyperparams_list):
            config_name = f"{model_name}_hp{hp_idx}"
            print(f"\n--- Hyperparams {hp_idx + 1}: lr={hyperparams['lr']}, wd={hyperparams['weight_decay']}, bs={hyperparams['batch_size']} ---")
            
            try:
                # Cross-validation
                fold_models, cv_results = cross_validate_model(
                    model_config, X_trainval, y_trainval, hyperparams,
                    device=args.device, n_folds=args.n_folds, max_epochs=max_epochs, verbose=True
                )
                
                print(f"\n  CV Results: MAE={cv_results['mae_mean']:.4f}±{cv_results['mae_std']:.4f}, R²={cv_results['r2_mean']:.4f}±{cv_results['r2_std']:.4f}")
                
                all_results[config_name] = {
                    'model_config': model_name,
                    'hyperparams': hyperparams,
                    'cv_results': cv_results,
                    'description': model_config['description']
                }
                
                # Track best model
                if cv_results['mae_mean'] < best_overall['mae']:
                    best_overall['mae'] = cv_results['mae_mean']
                    best_overall['model_name'] = config_name
                    best_overall['config'] = model_config
                    best_overall['hyperparams'] = hyperparams
                    best_overall['cv_results'] = cv_results
                    best_overall['fold_models'] = fold_models
                
            except Exception as e:
                print(f"  Error training {config_name}: {e}")
                continue
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    # Print leaderboard
    print("\n--- Model Leaderboard (by CV MAE) ---")
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['cv_results']['mae_mean'])
    
    for i, (name, result) in enumerate(sorted_results[:10], 1):
        cv = result['cv_results']
        print(f"{i}. {name}: MAE={cv['mae_mean']:.4f}±{cv['mae_std']:.4f}, R²={cv['r2_mean']:.4f}")
    
    # Train final model on all train+val data
    print(f"\n{'='*60}")
    print(f"Training FINAL model: {best_overall['model_name']}")
    print(f"{'='*60}")
    
    # Normalize targets
    y_mean = y_trainval.mean().item()
    y_std = y_trainval.std().item()
    y_trainval_norm = (y_trainval - y_mean) / y_std
    y_test_norm = (y_test - y_mean) / y_std
    
    # Train final model
    final_model, final_metrics = train_model(
        best_overall['config'], 
        X_trainval, y_trainval_norm,
        X_test, y_test_norm,
        y_mean, y_std,
        best_overall['hyperparams'],
        device=args.device,
        max_epochs=max_epochs,
        patience=50,
        use_augmentation=True,
        verbose=True
    )
    
    print(f"\nFinal Test Results:")
    print(f"  MAE: {final_metrics['mae']:.4f} mmol/g")
    print(f"  RMSE: {final_metrics['rmse']:.4f} mmol/g")
    print(f"  R²: {final_metrics['r2']:.4f}")
    
    # Save the best model
    model_save_path = output_dir / 'nh3_head_best.pt'
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'model_config': best_overall['model_name'],
        'model_class': best_overall['config']['class'].__name__,
        'model_params': best_overall['config']['params'],
        'hyperparams': best_overall['hyperparams'],
        'y_mean': y_mean,
        'y_std': y_std,
        'X_mean': X_mean.cpu(),
        'X_std': X_std.cpu(),
        'input_dim': X.shape[1],
        'test_metrics': {
            'mae': final_metrics['mae'],
            'rmse': final_metrics['rmse'],
            'r2': final_metrics['r2']
        },
        'cv_results': best_overall['cv_results']
    }, model_save_path)
    print(f"\nSaved best model to: {model_save_path}")
    
    # Also save to the standard location for sampling scripts
    standard_path = Path('raw_nh3_core/nh3_predictor_mlp.pt')
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'model_config': best_overall['model_name'],
        'model_class': best_overall['config']['class'].__name__,
        'model_params': best_overall['config']['params'],
        'hyperparams': best_overall['hyperparams'],
        'y_mean': y_mean,
        'y_std': y_std,
        'X_mean': X_mean.cpu(),
        'X_std': X_std.cpu(),
        'input_dim': X.shape[1],
        'test_metrics': {
            'mae': final_metrics['mae'],
            'rmse': final_metrics['rmse'],
            'r2': final_metrics['r2']
        }
    }, standard_path)
    print(f"Saved model to standard location: {standard_path}")
    
    # Save training summary
    summary = {
        'best_model': best_overall['model_name'],
        'best_mae': best_overall['mae'],
        'test_mae': final_metrics['mae'],
        'test_rmse': final_metrics['rmse'],
        'test_r2': final_metrics['r2'],
        'dataset_size': len(X),
        'train_val_size': len(train_val_idx),
        'test_size': len(test_idx),
        'all_results': {k: {
            'model_config': v['model_config'],
            'description': v['description'],
            'mae_mean': v['cv_results']['mae_mean'],
            'mae_std': v['cv_results']['mae_std'],
            'r2_mean': v['cv_results']['r2_mean'],
            'r2_std': v['cv_results']['r2_std']
        } for k, v in all_results.items()},
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Plot training results (if matplotlib available)
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Model comparison
        ax = axes[0]
        names = [r['model_config'] for _, r in sorted_results[:10]]
        maes = [r['cv_results']['mae_mean'] for _, r in sorted_results[:10]]
        mae_stds = [r['cv_results']['mae_std'] for _, r in sorted_results[:10]]
        ax.barh(range(len(names)), maes, xerr=mae_stds, color='steelblue')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('MAE (mmol/g)')
        ax.set_title('Model Comparison (CV MAE)')
        ax.invert_yaxis()
        
        # Plot 2: Predictions vs actual (test set)
        ax = axes[1]
        ax.scatter(final_metrics['targets'], final_metrics['preds'], alpha=0.6, s=20)
        min_val = min(final_metrics['targets'].min(), final_metrics['preds'].min())
        max_val = max(final_metrics['targets'].max(), final_metrics['preds'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax.set_xlabel('Actual NH3 Uptake (mmol/g)')
        ax.set_ylabel('Predicted NH3 Uptake (mmol/g)')
        ax.set_title(f'Test Set Predictions (R²={final_metrics["r2"]:.3f})')
        
        # Plot 3: Residual distribution
        ax = axes[2]
        residuals = final_metrics['preds'] - final_metrics['targets']
        ax.hist(residuals, bins=20, color='steelblue', edgecolor='white')
        ax.axvline(x=0, color='r', linestyle='--')
        ax.set_xlabel('Residual (mmol/g)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Residual Distribution (MAE={final_metrics["mae"]:.3f})')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'training_results.png', dpi=150)
        print(f"\nSaved training plots to: {output_dir / 'training_results.png'}")
    else:
        print("\nPlotting skipped (matplotlib not available)")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nBest Model: {best_overall['model_name']}")
    print(f"Test MAE: {final_metrics['mae']:.4f} mmol/g")
    print(f"Test R²: {final_metrics['r2']:.4f}")
    print(f"\nModel saved to: {model_save_path}")
    print(f"Also saved to: {standard_path}")


if __name__ == '__main__':
    main()
