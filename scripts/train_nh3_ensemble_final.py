#!/usr/bin/env python
"""
Final NH3 Head Training with Ensemble of Feature-Selected Models
This approach achieves Spearman ~0.72 on full dataset and ~0.38 on held-out test set
"""

import torch
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

def main():
    print('='*70)
    print('FINAL NH3 HEAD ENSEMBLE TRAINING')
    print('='*70)
    
    # Load data
    data = torch.load('raw_nh3_core/nh3_latent_dataset.pt')
    X = data['z'].numpy()
    y = data['y'].numpy()
    
    print(f'\nDataset: {len(y)} samples')
    print(f'NH3 range: [{y.min():.4f}, {y.max():.4f}] mmol/g')
    
    # 1. Feature selection based on Spearman correlation
    print('\n1. Feature Selection:')
    correlations = []
    for i in range(X.shape[1]):
        r, _ = spearmanr(X[:, i], y)
        correlations.append((i, abs(r), r))
    correlations.sort(key=lambda x: -x[1])
    
    n_features = 30
    top_features = [c[0] for c in correlations[:n_features]]
    print(f'   Selected top {n_features} features')
    print(f'   Correlation range: [{correlations[n_features-1][1]:.4f}, {correlations[0][1]:.4f}]')
    
    X_selected = X[:, top_features]
    
    # 2. Create polynomial and log features
    print('\n2. Feature Engineering:')
    X_poly = np.concatenate([
        X_selected,
        X_selected ** 2,
        np.log(np.abs(X_selected) + 1) * np.sign(X_selected)
    ], axis=1)
    print(f'   Original: {X_selected.shape[1]} features')
    print(f'   Engineered: {X_poly.shape[1]} features')
    
    # Log transform target (helps with skewed distribution)
    y_log = np.log(y + 1)
    
    # 3. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)
    
    # Train/test split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_log, test_size=0.2, random_state=42,
        stratify=(y > np.median(y)).astype(int)
    )
    y_test_orig = np.exp(y_test) - 1
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n   Device: {device}')
    
    # Model definition
    class SimpleNet(nn.Module):
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
    
    # Ranking loss function
    def ranking_loss(pred, target, margin=0.1):
        mse_loss = nn.functional.mse_loss(pred, target)
        
        n = len(pred)
        if n < 2:
            return mse_loss
        
        # Sample pairs for ranking
        n_pairs = min(100, n * (n - 1) // 2)
        idx1 = torch.randint(0, n, (n_pairs,), device=pred.device)
        idx2 = torch.randint(0, n, (n_pairs,), device=pred.device)
        
        pred_diff = pred[idx1] - pred[idx2]
        target_diff = target[idx1] - target[idx2]
        
        # Ranking loss: predictions should have same ordering as targets
        rank_loss = torch.relu(margin - pred_diff * target_diff.sign()).mean()
        
        return mse_loss + 0.5 * rank_loss
    
    # 4. Train ensemble
    print('\n3. Training Ensemble:')
    models = []
    test_preds = []
    
    n_models = 10
    for trial in range(n_models):
        torch.manual_seed(trial * 123)
        np.random.seed(trial * 123)
        
        model = SimpleNet(X_poly.shape[1], hidden=64).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
        
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).to(device)
        X_test_t = torch.FloatTensor(X_test).to(device)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Training loop
        for epoch in range(300):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = ranking_loss(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            pred_test = model(X_test_t).cpu().numpy()
        
        pred_orig = np.exp(pred_test) - 1
        spearman = spearmanr(pred_orig, y_test_orig)[0]
        
        print(f'   Model {trial+1}/{n_models}: Spearman = {spearman:.4f}')
        
        models.append(model.state_dict())
        test_preds.append(pred_orig)
    
    # Ensemble predictions on test set
    ensemble_pred = np.mean(test_preds, axis=0)
    ensemble_spearman = spearmanr(ensemble_pred, y_test_orig)[0]
    print(f'\n   Ensemble Test Spearman: {ensemble_spearman:.4f}')
    
    # 5. Full dataset evaluation
    print('\n4. Full Dataset Evaluation:')
    X_all_t = torch.FloatTensor(X_scaled).to(device)
    all_preds = []
    
    for model_state in models:
        model = SimpleNet(X_poly.shape[1], hidden=64).to(device)
        model.load_state_dict(model_state)
        model.eval()
        with torch.no_grad():
            pred = model(X_all_t).cpu().numpy()
        all_preds.append(np.exp(pred) - 1)
    
    final_pred = np.mean(all_preds, axis=0)
    final_spearman = spearmanr(final_pred, y)[0]
    final_mae = np.mean(np.abs(final_pred - y))
    
    print(f'   Overall Spearman: {final_spearman:.4f}')
    print(f'   Overall MAE: {final_mae:.4f} mmol/g')
    
    # Top-K identification
    print('\n5. Top-K Identification:')
    for k in [5, 10, 20]:
        true_top_k = set(int(i) for i in np.argsort(y)[-k:])
        pred_top_k = set(int(i) for i in np.argsort(final_pred)[-k:])
        overlap = len(true_top_k & pred_top_k)
        print(f'   Top-{k}: {overlap}/{k} correct ({100*overlap/k:.1f}%)')
    
    # 6. Save
    print('\n6. Saving models...')
    os.makedirs('nh3_head_training', exist_ok=True)
    
    save_dict = {
        'models': models,
        'feature_indices': top_features,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'in_dim': X_poly.shape[1],
        'hidden_dim': 64,
        'test_spearman': float(ensemble_spearman),
        'overall_spearman': float(final_spearman),
        'overall_mae': float(final_mae)
    }
    
    torch.save(save_dict, 'nh3_head_training/nh3_head_ensemble.pt')
    torch.save(save_dict, 'raw_nh3_core/nh3_predictor_mlp.pt')
    
    print('   Saved to nh3_head_training/nh3_head_ensemble.pt')
    print('   Saved to raw_nh3_core/nh3_predictor_mlp.pt')
    
    print('\n' + '='*70)
    print('TRAINING COMPLETE!')
    print('='*70)

if __name__ == '__main__':
    main()
