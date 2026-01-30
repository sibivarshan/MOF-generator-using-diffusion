"""
Sample MOFs with TARGET NH3 uptake using GRADIENT-BASED OPTIMIZATION.

Instead of screening random candidates, this script:
1. Starts with random latent vectors
2. Uses gradient descent to optimize them towards the target NH3 value
3. Decodes the optimized latent vectors to MOF structures

This requires a differentiable NH3 predictor (neural network).
"""
from pathlib import Path
from collections import defaultdict
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gc
from pytorch_lightning import seed_everything
from scipy.spatial import KDTree
from tqdm import tqdm

from mofdiff.common.atomic_utils import arrange_decoded_mofs
from mofdiff.common.eval_utils import load_mofdiff_model


class NH3PredictorMLP(nn.Module):
    """
    Differentiable NH3 predictor using a neural network.
    This replaces the SVR model to enable gradient-based optimization.
    """
    def __init__(self, input_dim=256, hidden_dims=[512, 256, 128, 64], dropout=0.1, use_batchnorm=False):
        super().__init__()
        
        if use_batchnorm:
            # Architecture with BatchNorm (matches retrained model)
            self.network = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        else:
            # Original architecture with LayerNorm
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
    
    def forward(self, z):
        return self.network(z).squeeze(-1)


def train_differentiable_predictor(latent_data_path, device="cuda", epochs=500):
    """
    Train a differentiable NH3 predictor from the latent dataset.
    """
    print("Training differentiable NH3 predictor...")
    
    # Load the latent dataset
    data = torch.load(latent_data_path, weights_only=False)
    
    # Get latents and targets
    if 'latents' in data:
        X = data['latents']
    elif 'z' in data:
        X = data['z']
    else:
        raise KeyError(f"Could not find latents in {latent_data_path}. Keys: {data.keys()}")
    
    if 'nh3_uptake' in data:
        y = data['nh3_uptake']
    elif 'NH3_uptake_298K_1bar [mmol/g]' in data:
        y = data['NH3_uptake_298K_1bar [mmol/g]']
    elif 'y' in data:
        y = data['y']
    else:
        raise KeyError(f"Could not find NH3 uptake in {latent_data_path}. Keys: {data.keys()}")
    
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()
    
    X = X.to(device)
    y = y.to(device)
    
    print(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"NH3 range: [{y.min():.4f}, {y.max():.4f}] mmol/g")
    
    # Normalize targets for better training
    y_mean = y.mean()
    y_std = y.std()
    y_normalized = (y - y_mean) / y_std
    
    # Create model
    model = NH3PredictorMLP(input_dim=X.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    model.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X)
        loss = nn.MSELoss()(pred, y_normalized)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        if (epoch + 1) % 100 == 0:
            rmse = np.sqrt(loss.item()) * y_std.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}, RMSE: {rmse:.4f} mmol/g")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        pred = model(X) * y_std + y_mean
        mae = (pred - y).abs().mean().item()
        print(f"\nFinal MAE: {mae:.4f} mmol/g")
    
    return model, y_mean, y_std


def load_or_train_predictor(latent_data_path, model_save_path=None, device="cuda", force_retrain=False):
    """Load existing predictor or train a new one."""
    if model_save_path is None:
        model_save_path = Path(latent_data_path).parent / "nh3_predictor_mlp.pt"
    
    if Path(model_save_path).exists() and not force_retrain:
        print(f"Loading existing predictor from {model_save_path}")
        ckpt = torch.load(model_save_path, weights_only=False)
        
        # Check if using BatchNorm architecture (new model)
        use_batchnorm = any('net.' in k for k in ckpt['model_state_dict'].keys())
        model = NH3PredictorMLP(input_dim=ckpt['input_dim'], use_batchnorm=use_batchnorm).to(device)
        
        # Handle different key names
        if use_batchnorm:
            # Rename keys from 'net.' to 'network.'
            new_state_dict = {}
            for k, v in ckpt['model_state_dict'].items():
                new_key = k.replace('net.', 'network.')
                new_state_dict[new_key] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(ckpt['model_state_dict'])
        
        y_mean = ckpt['y_mean']
        y_std = ckpt['y_std']
        # Load input normalization if available
        X_mean = ckpt.get('X_mean', None)
        X_std = ckpt.get('X_std', None)
        if X_mean is not None:
            return model, y_mean, y_std, X_mean.to(device), X_std.to(device)
    else:
        model, y_mean, y_std = train_differentiable_predictor(latent_data_path, device)
        
        # Save the model
        torch.save({
            'model_state_dict': model.state_dict(),
            'y_mean': y_mean,
            'y_std': y_std,
            'input_dim': 256,
        }, model_save_path)
        print(f"Saved predictor to {model_save_path}")
    
    return model, y_mean, y_std, None, None


def optimize_latent_for_target(
    model,
    y_mean,
    y_std,
    target_nh3,
    latent_dim=256,
    n_samples=5,
    n_init=20,
    lr=0.1,
    steps=500,
    regularization=0.01,
    device="cuda",
    X_mean=None,
    X_std=None,
):
    """
    Optimize latent vectors to achieve target NH3 uptake using gradient descent.
    
    Args:
        model: Differentiable NH3 predictor
        y_mean, y_std: Normalization parameters
        target_nh3: Target NH3 uptake value (mmol/g)
        latent_dim: Dimension of latent space
        n_samples: Number of final samples to return
        n_init: Number of random initializations to try
        lr: Learning rate
        steps: Number of optimization steps
        regularization: L2 regularization on latent vectors (keeps them near origin)
        device: cuda or cpu
    
    Returns:
        optimized_z: Optimized latent vectors
        final_nh3: Predicted NH3 values after optimization
    """
    model.eval()
    
    # Normalize target
    target_normalized = (target_nh3 - y_mean) / y_std
    
    print(f"\n{'='*60}")
    print(f"GRADIENT-BASED OPTIMIZATION")
    print(f"{'='*60}")
    print(f"Target NH3: {target_nh3:.4f} mmol/g")
    print(f"Initializations: {n_init}, Steps: {steps}, LR: {lr}")
    
    all_z = []
    all_nh3 = []
    all_losses = []
    
    for init_idx in range(n_init):
        # Initialize random latent vector
        z = torch.randn(1, latent_dim, device=device, requires_grad=True)
        
        optimizer = optim.Adam([z], lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        
        for step in range(steps):
            optimizer.zero_grad()
            
            # Normalize input if normalization params available
            if X_mean is not None and X_std is not None:
                z_input = (z - X_mean) / X_std
            else:
                z_input = z
            
            # Predict NH3 (normalized)
            pred_normalized = model(z_input)
            
            # Loss: MSE to target + regularization
            target_loss = (pred_normalized - target_normalized) ** 2
            reg_loss = regularization * (z ** 2).mean()
            loss = target_loss + reg_loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # Store result
        with torch.no_grad():
            if X_mean is not None and X_std is not None:
                z_input = (z - X_mean) / X_std
            else:
                z_input = z
            final_pred = model(z_input) * y_std + y_mean
            all_z.append(z.detach().clone())
            all_nh3.append(final_pred.item())
            all_losses.append(abs(final_pred.item() - target_nh3))
        
        if (init_idx + 1) % 5 == 0:
            print(f"  Init {init_idx+1}/{n_init}: NH3 = {final_pred.item():.4f} mmol/g, "
                  f"distance = {abs(final_pred.item() - target_nh3):.4f}")
    
    # Select best results
    all_z = torch.cat(all_z, dim=0)
    all_nh3 = np.array(all_nh3)
    all_losses = np.array(all_losses)
    
    # Sort by distance to target
    best_idx = np.argsort(all_losses)[:n_samples]
    
    optimized_z = all_z[best_idx]
    final_nh3 = all_nh3[best_idx]
    
    print(f"\nBest {n_samples} results:")
    for i, (nh3, dist) in enumerate(zip(final_nh3, all_losses[best_idx])):
        print(f"  Sample {i}: NH3 = {nh3:.4f} mmol/g, distance = {dist:.4f}")
    
    return optimized_z, final_nh3


def optimize_with_constraint(
    model,
    y_mean,
    y_std,
    target_nh3,
    mofdiff_model,
    latent_dim=256,
    n_samples=5,
    n_init=20,
    lr=0.1,
    steps=500,
    regularization=0.01,
    validity_weight=0.0,
    device="cuda",
):
    """
    Advanced optimization with additional constraints.
    
    This version can optionally include:
    - Regularization to keep latents near the data manifold
    - (Future) Validity constraints from MOFDiff
    """
    return optimize_latent_for_target(
        model, y_mean, y_std, target_nh3,
        latent_dim=latent_dim,
        n_samples=n_samples,
        n_init=n_init,
        lr=lr,
        steps=steps,
        regularization=regularization,
        device=device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample MOFs with target NH3 using gradient optimization")
    parser.add_argument("--model_path", type=str, default="pretrained/mofdiff_ckpt")
    parser.add_argument("--bb_cache_path", type=str, default="pretrained/bb_emb_space.pt")
    parser.add_argument("--latent_data_path", type=str, default="raw_nh3_core/nh3_latent_dataset.pt",
                        help="Path to latent dataset for training the differentiable predictor")
    parser.add_argument("--predictor_path", type=str, default=None,
                        help="Path to save/load the differentiable predictor")
    
    parser.add_argument("--target_nh3", type=float, required=True,
                        help="Target NH3 uptake value (mmol/g)")
    parser.add_argument("--n_samples", default=5, type=int, 
                        help="Number of final MOF samples")
    parser.add_argument("--n_init", default=20, type=int,
                        help="Number of random initializations for optimization")
    parser.add_argument("--lr", default=0.1, type=float,
                        help="Learning rate for optimization")
    parser.add_argument("--steps", default=500, type=int,
                        help="Number of optimization steps")
    parser.add_argument("--regularization", default=0.01, type=float,
                        help="L2 regularization strength")
    
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--relax", action="store_true")
    parser.add_argument("--force_retrain", action="store_true",
                        help="Force retraining of the differentiable predictor")

    args = parser.parse_args()
    seed_everything(args.seed)
    
    model_path = Path(args.model_path).resolve()
    
    # Set output directory
    if args.output_dir is None:
        output_dir = Path(f"nh3_gradient_{args.target_nh3:.2f}")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load or train differentiable NH3 predictor
    result = load_or_train_predictor(
        args.latent_data_path,
        args.predictor_path,
        device=device,
        force_retrain=args.force_retrain,
    )
    if len(result) == 5:
        predictor, y_mean, y_std, X_mean, X_std = result
    else:
        predictor, y_mean, y_std = result
        X_mean, X_std = None, None
    
    # Optimize latent vectors towards target
    optimized_z, predicted_nh3 = optimize_latent_for_target(
        predictor, y_mean, y_std,
        target_nh3=args.target_nh3,
        n_samples=args.n_samples,
        n_init=args.n_init,
        lr=args.lr,
        steps=args.steps,
        regularization=args.regularization,
        device=device,
        X_mean=X_mean,
        X_std=X_std,
    )
    
    # Load MOFDiff model for decoding
    print(f"\nLoading MOFDiff model from {model_path}...")
    mofdiff_model, cfg, bb_encoder = load_mofdiff_model(model_path)
    mofdiff_model = mofdiff_model.to(device)
    
    # Load building block cache
    print(f"Loading building block cache from {args.bb_cache_path}...")
    all_data, all_z = torch.load(args.bb_cache_path)
    kdtree = KDTree(all_z)
    
    # Decode to MOF structures
    print(f"\nDecoding {len(optimized_z)} optimized MOFs...")
    output = defaultdict(list)
    
    n_batch = int(np.ceil(len(optimized_z) / args.batch_size))
    for idx in range(n_batch):
        print(f"Processing batch {idx + 1}/{n_batch}...")
        start = idx * args.batch_size
        end = min(start + args.batch_size, len(optimized_z))
        z_batch = optimized_z[start:end].to(device)
        
        samples = mofdiff_model.sample(z_batch.shape[0], z_batch, save_freq=False)
        mofs = arrange_decoded_mofs(samples, kdtree, all_data, False)
        output["mofs"].extend(mofs)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    output["z"] = optimized_z.cpu()
    output["predicted_nh3"] = predicted_nh3
    output["target_nh3"] = args.target_nh3
    output["optimization_method"] = "gradient"
    
    # Save samples
    save_path = output_dir / "samples.pt"
    print(f"Saving samples to {save_path}...")
    torch.save(output, save_path)
    
    # Print summary
    print("\n" + "="*60)
    print("GRADIENT OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Target NH3 uptake: {args.target_nh3:.4f} mmol/g")
    print(f"Generated {len(output['mofs'])} MOF structures")
    print(f"Predicted NH3 uptake:")
    print(f"  Mean: {predicted_nh3.mean():.4f} mmol/g")
    print(f"  Min:  {predicted_nh3.min():.4f} mmol/g")
    print(f"  Max:  {predicted_nh3.max():.4f} mmol/g")
    print(f"  Avg distance from target: {np.abs(predicted_nh3 - args.target_nh3).mean():.4f}")
    print(f"\nSaved to: {output_dir}")
    
    # Run assembly if requested
    if args.assemble:
        print(f"\n{'='*60}")
        print("ASSEMBLING MOF STRUCTURES")
        print("="*60)
        from mofdiff.scripts.assemble import main as assemble_main
        assemble_main(str(save_path), verbose=False, rounds=3)
        
        if args.relax:
            print(f"\n{'='*60}")
            print("RELAXING MOF STRUCTURES")
            print("="*60)
            import subprocess
            subprocess.run([
                "python", "scripts/simple_relax.py",
                "--input_dir", str(output_dir)
            ])
    
    print(f"\nNext steps:")
    if not args.assemble:
        print(f"  1. Assemble: python mofdiff/scripts/assemble.py --input {save_path}")
    if not args.relax:
        print(f"  2. Relax: python scripts/simple_relax.py --input_dir {output_dir}")
