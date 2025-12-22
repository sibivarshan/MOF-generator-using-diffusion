"""
Sample MOFs with NH3-guided optimization using the trained NH3 head.
This script samples from the latent space, uses the NH3 head to predict uptake,
and generates structures predicted to have high NH3 uptake.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import torch
import numpy as np
import gc
from pytorch_lightning import seed_everything
from scipy.spatial import KDTree
from tqdm import tqdm

from mofdiff.common.atomic_utils import arrange_decoded_mofs
from mofdiff.common.eval_utils import load_mofdiff_model


def load_nh3_head(path="raw_nh3_core/nh3_head_best_final.pt"):
    """Load the trained NH3 prediction head"""
    ckpt = torch.load(path, weights_only=False)
    return ckpt


def predict_nh3(latent_z, nh3_ckpt):
    """Predict NH3 uptake from latent vectors"""
    if isinstance(latent_z, torch.Tensor):
        z = latent_z.cpu().numpy()
    else:
        z = latent_z
    
    if z.ndim == 1:
        z = z.reshape(1, -1)
    
    # Standardize
    scaler = nh3_ckpt['scaler_X']
    z_scaled = scaler.transform(z)
    
    # Feature selection
    selector = nh3_ckpt.get('selector', None)
    if selector is not None:
        z_scaled = selector.transform(z_scaled)
    
    # Predict
    model = nh3_ckpt['model']
    predictions = model.predict(z_scaled)
    
    return predictions


def sample_with_nh3_optimization(
    model, 
    nh3_ckpt,
    n_samples,
    n_candidates=1000,
    device="cuda",
    min_nh3=None,
    top_k=None,
):
    """
    Sample MOFs optimized for high NH3 uptake.
    
    Strategy: Generate many candidate latent vectors, predict NH3 uptake,
    and keep only those with highest predicted uptake.
    
    Args:
        model: MOFDiff model
        nh3_ckpt: NH3 head checkpoint
        n_samples: Number of final samples to return
        n_candidates: Number of candidates to generate (should be >> n_samples)
        device: cuda or cpu
        min_nh3: Minimum NH3 uptake threshold (optional)
        top_k: Keep top k samples by predicted NH3 (default: n_samples)
    """
    if top_k is None:
        top_k = n_samples
    
    latent_dim = model.hparams.latent_dim
    
    print(f"Generating {n_candidates} candidate latent vectors...")
    all_z = torch.randn(n_candidates, latent_dim)
    
    print("Predicting NH3 uptake for candidates...")
    nh3_pred = predict_nh3(all_z, nh3_ckpt)
    
    print(f"NH3 predictions: min={nh3_pred.min():.4f}, max={nh3_pred.max():.4f}, mean={nh3_pred.mean():.4f}")
    
    # Select top candidates
    if min_nh3 is not None:
        mask = nh3_pred >= min_nh3
        selected_idx = np.where(mask)[0]
        print(f"Found {len(selected_idx)} candidates with NH3 >= {min_nh3}")
        if len(selected_idx) < n_samples:
            print(f"Warning: Only {len(selected_idx)} candidates meet threshold, using top {n_samples} instead")
            selected_idx = np.argsort(nh3_pred)[::-1][:n_samples].copy()
    else:
        selected_idx = np.argsort(nh3_pred)[::-1][:top_k].copy()
    
    selected_z = all_z[selected_idx].clone()
    selected_nh3 = nh3_pred[selected_idx].copy()
    
    print(f"Selected {len(selected_idx)} samples with predicted NH3: {selected_nh3.min():.4f} - {selected_nh3.max():.4f}")
    
    return selected_z, selected_nh3


def sample_random(model, n_samples, device="cuda"):
    """Generate random samples from latent space"""
    latent_dim = model.hparams.latent_dim
    z = torch.randn(n_samples, latent_dim)
    return z


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="pretrained/mofdiff_ckpt",
    )
    parser.add_argument(
        "--bb_cache_path",
        type=str,
        default="pretrained/bb_emb_space.pt",
    )
    parser.add_argument(
        "--nh3_head_path",
        type=str,
        default="raw_nh3_core/nh3_head_best_final.pt",
    )
    parser.add_argument("--n_samples", default=10, type=int, help="Number of final MOF samples")
    parser.add_argument("--n_candidates", default=1000, type=int, help="Number of candidate latent vectors to generate")
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--output_dir", default="nh3_samples", type=str)
    parser.add_argument("--random", action="store_true", help="Use random sampling instead of NH3-guided")

    args = parser.parse_args()
    seed_everything(args.seed)
    
    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load MOFDiff model
    print(f"Loading MOFDiff model from {model_path}...")
    model, cfg, bb_encoder = load_mofdiff_model(model_path)
    model = model.to(device)
    
    # Load NH3 head
    print(f"Loading NH3 head from {args.nh3_head_path}...")
    nh3_ckpt = load_nh3_head(args.nh3_head_path)
    print(f"NH3 head CV MAE: {nh3_ckpt.get('cv_mae', 'N/A')}")
    
    # Load building block cache
    print(f"Loading building block cache from {args.bb_cache_path}...")
    all_data, all_z = torch.load(args.bb_cache_path)
    kdtree = KDTree(all_z)
    
    # Generate latent vectors
    if args.random:
        print(f"\nGenerating {args.n_samples} random samples...")
        selected_z = sample_random(model, args.n_samples, device)
        selected_nh3 = predict_nh3(selected_z, nh3_ckpt)
    else:
        print(f"\nGenerating NH3-optimized samples...")
        selected_z, selected_nh3 = sample_with_nh3_optimization(
            model, nh3_ckpt,
            n_samples=args.n_samples,
            n_candidates=args.n_candidates,
            device=device,
        )
    
    print(f"\nSelected latent vectors: {selected_z.shape}")
    print(f"Predicted NH3 uptake range: [{selected_nh3.min():.4f}, {selected_nh3.max():.4f}] mmol/g")
    
    # Decode to MOF structures
    print(f"\nDecoding {len(selected_z)} MOFs...")
    output = defaultdict(list)
    
    n_batch = int(np.ceil(len(selected_z) / args.batch_size))
    for idx in range(n_batch):
        print(f"Processing batch {idx + 1}/{n_batch}...")
        start = idx * args.batch_size
        end = min(start + args.batch_size, len(selected_z))
        z_batch = selected_z[start:end].to(device)
        
        samples = model.sample(z_batch.shape[0], z_batch, save_freq=False)
        mofs = arrange_decoded_mofs(samples, kdtree, all_data, False)
        output["mofs"].extend(mofs)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    output["z"] = selected_z
    output["predicted_nh3"] = selected_nh3
    
    # Save samples
    save_path = output_dir / "samples.pt"
    print(f"Saving samples to {save_path}...")
    torch.save(output, save_path)
    
    # Print summary
    print("\n" + "="*60)
    print("SAMPLING COMPLETE")
    print("="*60)
    print(f"Generated {len(output['mofs'])} MOF structures")
    print(f"Predicted NH3 uptake:")
    print(f"  Mean: {selected_nh3.mean():.4f} mmol/g")
    print(f"  Min:  {selected_nh3.min():.4f} mmol/g")
    print(f"  Max:  {selected_nh3.max():.4f} mmol/g")
    print(f"\nSaved to: {output_dir}")
    print(f"Next step: python mofdiff/scripts/assemble.py --input {save_path}")
