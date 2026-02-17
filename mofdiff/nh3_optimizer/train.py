"""
NH3 Predictor Training Pipeline

Trains a deep ensemble of NH3 uptake predictors on the latent space
of MOFDiff. Uses the 316-sample dataset of (latent_z, NH3_uptake) pairs
extracted from the pretrained MOFDiff encoder.

Training strategy:
  - K-fold cross-validation with ensemble of models
  - Each model trained on a different bootstrap of the data
  - Heteroscedastic loss (Gaussian NLL) for aleatoric uncertainty
  - Early stopping on validation Spearman correlation
  - Data augmentation via latent space perturbation (Gaussian noise)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
import json
import copy
import time

from mofdiff.nh3_optimizer.predictor import NH3PredictorEnsemble


def gaussian_nll_loss(mean, log_var, target):
    """
    Heteroscedastic Gaussian negative log-likelihood.
    
    NLL = 0.5 * (log_var + (target - mean)^2 / exp(log_var))
    
    This learns both the mean prediction AND the noise level.
    """
    precision = torch.exp(-log_var)
    return 0.5 * (log_var + precision * (target - mean) ** 2).mean()


def train_predictor(
    data_path: str = "raw_nh3_core/nh3_latent_dataset.pt",
    save_path: str = "pretrained/nh3_optimizer.pt",
    n_models: int = 10,
    hidden_dims: list = [512, 256, 128],
    dropout: float = 0.2,
    lr: float = 5e-4,
    weight_decay: float = 5e-4,
    n_epochs: int = 500,
    batch_size: int = 32,
    patience: int = 80,
    augment_noise: float = 0.1,
    val_fraction: float = 0.15,
    device: str = "cuda",
    seed: int = 42,
):
    """
    Train the NH3 predictor ensemble.
    
    Args:
        data_path: Path to the latent dataset (z, y, m_id)
        save_path: Where to save the trained ensemble
        n_models: Number of models in the ensemble
        hidden_dims: Hidden layer dimensions
        dropout: Dropout rate
        lr: Learning rate
        weight_decay: Weight decay for AdamW
        n_epochs: Maximum training epochs per model
        batch_size: Batch size
        patience: Early stopping patience
        augment_noise: Gaussian noise std for data augmentation
        val_fraction: Fraction of data for validation
        device: Training device
        seed: Random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = device if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    
    # Load data
    print(f"Loading data from {data_path}...")
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    z_all = data['z']  # [N, 256]
    y_all = data['y']  # [N]
    m_ids = data['m_id']
    
    N, z_dim = z_all.shape
    print(f"Dataset: {N} samples, z_dim={z_dim}")
    print(f"NH3 range: [{y_all.min():.3f}, {y_all.max():.3f}] mmol/g")
    
    # Log-transform y for better distribution (heavy right skew)
    # log1p(y) maps [0, inf) -> [0, inf) with less skew
    y_log = torch.log1p(y_all)
    
    # Compute normalization statistics
    z_mean = z_all.mean(dim=0)
    z_std = z_all.std(dim=0)
    y_mean = y_log.mean()
    y_std = y_log.std()
    
    # Normalize
    z_norm = (z_all - z_mean) / (z_std + 1e-8)
    y_norm = (y_log - y_mean) / (y_std + 1e-8)
    
    print(f"y_log_mean={y_mean:.3f}, y_log_std={y_std:.3f}")
    print(f"Using log1p transform on NH3 targets")
    
    # Create ensemble
    ensemble = NH3PredictorEnsemble(
        n_models=n_models,
        in_dim=z_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)
    
    # Set normalization params
    ensemble.z_mean = z_mean.to(device)
    ensemble.z_std = z_std.to(device)
    ensemble.y_mean = y_mean.to(device)
    ensemble.y_std = y_std.to(device)
    ensemble.log_transform_y = True  # We use log1p on targets
    
    # Training results storage
    training_results = {
        'models': [],
        'val_spearman': [],
        'val_mae': [],
        'train_loss': [],
    }
    
    print(f"\n{'='*60}")
    print(f"Training {n_models}-model ensemble")
    print(f"Hidden dims: {hidden_dims}, Dropout: {dropout}")
    print(f"LR: {lr}, WD: {weight_decay}, Epochs: {n_epochs}")
    print(f"{'='*60}\n")
    
    for model_idx in range(n_models):
        print(f"\n--- Model {model_idx + 1}/{n_models} ---")
        
        # Bootstrap + stratified split
        # Use different random split for each model
        rng = np.random.RandomState(seed + model_idx * 137)
        indices = np.arange(N)
        rng.shuffle(indices)
        
        n_val = int(N * val_fraction)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        
        # Additionally bootstrap the training set (with replacement)
        boot_idx = rng.choice(train_idx, size=len(train_idx), replace=True)
        
        z_train = z_norm[boot_idx]
        y_train = y_norm[boot_idx]
        z_val = z_norm[val_idx]
        y_val_norm = y_norm[val_idx]
        y_val_orig = y_all[val_idx]  # Original scale for metrics
        
        train_ds = TensorDataset(z_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  drop_last=False)
        
        model = ensemble.models[model_idx]
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)
        
        best_val_spearman = -1.0
        best_state = None
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # Training
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            
            for z_batch, y_batch in train_loader:
                z_batch = z_batch.to(device)
                y_batch = y_batch.to(device)
                
                # Data augmentation: Gaussian noise on z
                if augment_noise > 0:
                    noise = torch.randn_like(z_batch) * augment_noise
                    z_batch_aug = z_batch + noise
                else:
                    z_batch_aug = z_batch
                
                # Mixup augmentation for better generalization
                if z_batch.shape[0] > 1:
                    lam = np.random.beta(0.4, 0.4)
                    perm = torch.randperm(z_batch.shape[0], device=device)
                    z_mix = lam * z_batch_aug + (1 - lam) * z_batch_aug[perm]
                    y_mix = lam * y_batch + (1 - lam) * y_batch[perm]
                    
                    # Train on both original (augmented) + mixup
                    z_combined = torch.cat([z_batch_aug, z_mix], dim=0)
                    y_combined = torch.cat([y_batch, y_mix], dim=0)
                else:
                    z_combined = z_batch_aug
                    y_combined = y_batch
                
                mean, log_var = model(z_combined)
                loss = gaussian_nll_loss(mean.squeeze(-1), log_var.squeeze(-1), y_combined)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            scheduler.step()
            avg_loss = epoch_loss / n_batches
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_mean, _ = model(z_val.to(device))
                val_pred_norm = val_mean.squeeze(-1).cpu()
                # Convert to original scale: denorm -> exp(log1p) = expm1 -> original
                val_pred_log = val_pred_norm * y_std + y_mean
                val_pred_orig = torch.expm1(val_pred_log).clamp(min=0)
                
                val_spearman, _ = spearmanr(val_pred_orig.numpy(), y_val_orig.numpy())
                val_mae = (val_pred_orig - y_val_orig).abs().mean().item()
            
            if np.isnan(val_spearman):
                val_spearman = -1.0
            
            # Early stopping on Spearman
            if val_spearman > best_val_spearman:
                best_val_spearman = val_spearman
                best_val_mae = val_mae
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f}, "
                      f"val_spearman={val_spearman:.4f}, val_mae={val_mae:.3f}, "
                      f"best_spearman={best_val_spearman:.4f}")
            
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        model.load_state_dict(best_state)
        training_results['val_spearman'].append(best_val_spearman)
        training_results['val_mae'].append(best_val_mae)
        print(f"  Model {model_idx+1}: best_spearman={best_val_spearman:.4f}, mae={best_val_mae:.3f}")
    
    # Final evaluation on ALL data (using ensemble)
    ensemble.eval()
    with torch.no_grad():
        result = ensemble(z_all.to(device))
        pred_all = result['mean'].cpu()
        std_all = result['std'].cpu()
    
    overall_spearman, _ = spearmanr(pred_all.numpy(), y_all.numpy())
    overall_mae = (pred_all - y_all).abs().mean().item()
    overall_r2 = 1 - ((pred_all - y_all) ** 2).sum() / ((y_all - y_all.mean()) ** 2).sum()
    
    print(f"\n{'='*60}")
    print(f"ENSEMBLE RESULTS (all data)")
    print(f"{'='*60}")
    print(f"  Overall Spearman: {overall_spearman:.4f}")
    print(f"  Overall MAE:      {overall_mae:.3f} mmol/g")
    print(f"  Overall R²:       {overall_r2:.4f}")
    print(f"  Mean val Spearman:{np.mean(training_results['val_spearman']):.4f}")
    print(f"  Mean val MAE:     {np.mean(training_results['val_mae']):.3f}")
    print(f"  Mean uncertainty: {std_all.mean():.3f}")
    
    # Save
    save_dict = {
        'ensemble_state_dict': ensemble.state_dict(),
        'n_models': n_models,
        'in_dim': z_dim,
        'hidden_dims': hidden_dims,
        'dropout': dropout,
        'z_mean': z_mean,
        'z_std': z_std,
        'y_mean': y_mean,
        'y_std': y_std,
        'log_transform_y': True,
        'overall_spearman': overall_spearman,
        'overall_mae': overall_mae,
        'overall_r2': overall_r2.item(),
        'val_spearmans': training_results['val_spearman'],
        'val_maes': training_results['val_mae'],
        'training_config': {
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'lr': lr,
            'weight_decay': weight_decay,
            'augment_noise': augment_noise,
            'patience': patience,
        },
        'dataset_info': {
            'n_samples': N,
            'y_range': [y_all.min().item(), y_all.max().item()],
        },
    }
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, save_path)
    print(f"\nSaved ensemble to {save_path}")
    
    return ensemble, save_dict


def load_predictor(path: str, device: str = "cpu") -> NH3PredictorEnsemble:
    """Load a trained predictor ensemble from disk."""
    data = torch.load(path, map_location=device, weights_only=False)
    
    ensemble = NH3PredictorEnsemble(
        n_models=data['n_models'],
        in_dim=data['in_dim'],
        hidden_dims=data['hidden_dims'],
        dropout=data['dropout'],
    ).to(device)
    
    ensemble.load_state_dict(data['ensemble_state_dict'])
    ensemble.log_transform_y = data.get('log_transform_y', False)
    ensemble.eval()
    
    return ensemble


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train NH3 Predictor Ensemble")
    parser.add_argument("--data_path", type=str, default="raw_nh3_core/nh3_latent_dataset.pt")
    parser.add_argument("--save_path", type=str, default="pretrained/nh3_optimizer.pt")
    parser.add_argument("--n_models", type=int, default=10)
    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    train_predictor(
        data_path=args.data_path,
        save_path=args.save_path,
        n_models=args.n_models,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
    )
