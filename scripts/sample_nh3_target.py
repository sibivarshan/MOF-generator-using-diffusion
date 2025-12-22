"""
Sample MOFs with a TARGET NH3 uptake value.
This script samples from the latent space and finds structures with predicted 
NH3 uptake close to the user-specified target value.
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


def predict_nh3(latent_z, nh3_ckpt, return_original_scale=True):
    """Predict NH3 uptake from latent vectors
    
    Args:
        latent_z: Latent vectors
        nh3_ckpt: NH3 head checkpoint
        return_original_scale: If True, return predictions in original mmol/g scale
                               If False, return standardized predictions
    
    Returns:
        predictions: NH3 uptake predictions (mmol/g if return_original_scale=True)
    """
    if isinstance(latent_z, torch.Tensor):
        z = latent_z.cpu().numpy()
    else:
        z = latent_z
    
    if z.ndim == 1:
        z = z.reshape(1, -1)
    
    # Standardize input
    scaler = nh3_ckpt['scaler_X']
    z_scaled = scaler.transform(z)
    
    # Feature selection
    selector = nh3_ckpt.get('selector', None)
    if selector is not None:
        z_scaled = selector.transform(z_scaled)
    
    # Predict (standardized)
    model = nh3_ckpt['model']
    predictions = model.predict(z_scaled)
    
    # Convert back to original scale if requested
    if return_original_scale:
        y_mean = nh3_ckpt.get('y_mean', 0.0)
        y_std = nh3_ckpt.get('y_std', 1.0)
        predictions = predictions * y_std + y_mean
    
    return predictions


def sample_with_target_nh3(
    model, 
    nh3_ckpt,
    target_nh3,
    n_samples,
    n_candidates=5000,
    tolerance=0.1,
    device="cuda",
):
    """
    Sample MOFs with NH3 uptake close to a target value.
    
    Strategy: Generate many candidate latent vectors, predict NH3 uptake,
    and keep those closest to the target value.
    
    Args:
        model: MOFDiff model
        nh3_ckpt: NH3 head checkpoint
        target_nh3: Target NH3 uptake value (mmol/g)
        n_samples: Number of final samples to return
        n_candidates: Number of candidates to generate (should be >> n_samples)
        tolerance: Acceptable deviation from target (optional, for filtering)
        device: cuda or cpu
    
    Returns:
        selected_z: Latent vectors closest to target
        selected_nh3: Predicted NH3 values
        distances: Absolute distance from target
    """
    latent_dim = model.hparams.latent_dim
    
    print(f"Target NH3 uptake: {target_nh3:.4f} mmol/g")
    print(f"Generating {n_candidates} candidate latent vectors...")
    all_z = torch.randn(n_candidates, latent_dim)
    
    print("Predicting NH3 uptake for candidates...")
    nh3_pred = predict_nh3(all_z, nh3_ckpt)
    
    print(f"NH3 predictions range: [{nh3_pred.min():.4f}, {nh3_pred.max():.4f}], mean={nh3_pred.mean():.4f}")
    
    # Calculate distance from target
    distances = np.abs(nh3_pred - target_nh3)
    
    # Sort by distance to target (closest first)
    sorted_idx = np.argsort(distances)
    
    # Select n_samples closest to target
    selected_idx = sorted_idx[:n_samples].copy()
    
    selected_z = all_z[selected_idx].clone()
    selected_nh3 = nh3_pred[selected_idx].copy()
    selected_distances = distances[selected_idx].copy()
    
    print(f"\nSelected {len(selected_idx)} samples closest to target {target_nh3:.4f}:")
    print(f"  Predicted NH3 range: [{selected_nh3.min():.4f}, {selected_nh3.max():.4f}]")
    print(f"  Distance from target: [{selected_distances.min():.4f}, {selected_distances.max():.4f}]")
    
    # Check if any are within tolerance
    within_tolerance = np.sum(selected_distances <= tolerance)
    print(f"  Samples within tolerance (±{tolerance}): {within_tolerance}/{n_samples}")
    
    return selected_z, selected_nh3, selected_distances


def sample_with_target_range(
    model, 
    nh3_ckpt,
    min_nh3,
    max_nh3,
    n_samples,
    n_candidates=5000,
    device="cuda",
):
    """
    Sample MOFs with NH3 uptake within a specified range.
    
    Args:
        model: MOFDiff model
        nh3_ckpt: NH3 head checkpoint
        min_nh3: Minimum NH3 uptake value (mmol/g)
        max_nh3: Maximum NH3 uptake value (mmol/g)
        n_samples: Number of final samples to return
        n_candidates: Number of candidates to generate
        device: cuda or cpu
    """
    latent_dim = model.hparams.latent_dim
    
    print(f"Target NH3 uptake range: [{min_nh3:.4f}, {max_nh3:.4f}] mmol/g")
    print(f"Generating {n_candidates} candidate latent vectors...")
    all_z = torch.randn(n_candidates, latent_dim)
    
    print("Predicting NH3 uptake for candidates...")
    nh3_pred = predict_nh3(all_z, nh3_ckpt)
    
    print(f"NH3 predictions range: [{nh3_pred.min():.4f}, {nh3_pred.max():.4f}], mean={nh3_pred.mean():.4f}")
    
    # Find candidates within range
    mask = (nh3_pred >= min_nh3) & (nh3_pred <= max_nh3)
    in_range_idx = np.where(mask)[0]
    
    print(f"Found {len(in_range_idx)} candidates within range")
    
    if len(in_range_idx) >= n_samples:
        # Randomly select from those in range
        selected_idx = np.random.choice(in_range_idx, size=n_samples, replace=False)
    else:
        # Not enough in range - take all in range + closest ones
        print(f"Warning: Only {len(in_range_idx)} candidates in range, finding closest alternatives...")
        
        # Calculate distance to range center
        target_center = (min_nh3 + max_nh3) / 2
        distances = np.abs(nh3_pred - target_center)
        sorted_idx = np.argsort(distances)
        selected_idx = sorted_idx[:n_samples].copy()
    
    selected_z = all_z[selected_idx].clone()
    selected_nh3 = nh3_pred[selected_idx].copy()
    
    print(f"\nSelected {len(selected_idx)} samples:")
    print(f"  Predicted NH3 range: [{selected_nh3.min():.4f}, {selected_nh3.max():.4f}]")
    
    return selected_z, selected_nh3


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample MOFs with target NH3 uptake value")
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
    parser.add_argument("--target_nh3", type=float, required=True,
                        help="Target NH3 uptake value (mmol/g)")
    parser.add_argument("--tolerance", type=float, default=0.1,
                        help="Acceptable tolerance from target (default: 0.1)")
    parser.add_argument("--min_nh3", type=float, default=None,
                        help="Minimum NH3 uptake (for range mode)")
    parser.add_argument("--max_nh3", type=float, default=None,
                        help="Maximum NH3 uptake (for range mode)")
    parser.add_argument("--n_samples", default=5, type=int, 
                        help="Number of final MOF samples")
    parser.add_argument("--n_candidates", default=5000, type=int, 
                        help="Number of candidate latent vectors to generate")
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--output_dir", default=None, type=str,
                        help="Output directory (default: nh3_target_<value>)")
    parser.add_argument("--assemble", action="store_true",
                        help="Also run assembly after sampling")
    parser.add_argument("--relax", action="store_true",
                        help="Also run relaxation after assembly")

    args = parser.parse_args()
    seed_everything(args.seed)
    
    model_path = Path(args.model_path).resolve()
    
    # Set output directory
    if args.output_dir is None:
        output_dir = Path(f"nh3_target_{args.target_nh3:.2f}")
    else:
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
    
    # Generate latent vectors with target NH3
    print(f"\n{'='*60}")
    print(f"SAMPLING MOFs WITH TARGET NH3 = {args.target_nh3:.4f} mmol/g")
    print(f"{'='*60}")
    
    if args.min_nh3 is not None and args.max_nh3 is not None:
        # Range mode
        selected_z, selected_nh3 = sample_with_target_range(
            model, nh3_ckpt,
            min_nh3=args.min_nh3,
            max_nh3=args.max_nh3,
            n_samples=args.n_samples,
            n_candidates=args.n_candidates,
            device=device,
        )
        distances = np.abs(selected_nh3 - args.target_nh3)
    else:
        # Target mode
        selected_z, selected_nh3, distances = sample_with_target_nh3(
            model, nh3_ckpt,
            target_nh3=args.target_nh3,
            n_samples=args.n_samples,
            n_candidates=args.n_candidates,
            tolerance=args.tolerance,
            device=device,
        )
    
    print(f"\nSelected latent vectors: {selected_z.shape}")
    print(f"Predicted NH3 uptake values:")
    for i, (nh3, dist) in enumerate(zip(selected_nh3, distances)):
        print(f"  Sample {i}: {nh3:.4f} mmol/g (distance from target: {dist:.4f})")
    
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
    output["target_nh3"] = args.target_nh3
    output["distances_from_target"] = distances
    
    # Save samples
    save_path = output_dir / "samples.pt"
    print(f"Saving samples to {save_path}...")
    torch.save(output, save_path)
    
    # Print summary
    print("\n" + "="*60)
    print("SAMPLING COMPLETE")
    print("="*60)
    print(f"Target NH3 uptake: {args.target_nh3:.4f} mmol/g")
    print(f"Generated {len(output['mofs'])} MOF structures")
    print(f"Predicted NH3 uptake:")
    print(f"  Mean: {selected_nh3.mean():.4f} mmol/g")
    print(f"  Min:  {selected_nh3.min():.4f} mmol/g")
    print(f"  Max:  {selected_nh3.max():.4f} mmol/g")
    print(f"  Avg distance from target: {distances.mean():.4f}")
    print(f"\nSaved to: {output_dir}")
    
    # Run assembly if requested
    if args.assemble:
        print(f"\n{'='*60}")
        print("ASSEMBLING MOF STRUCTURES")
        print("="*60)
        from mofdiff.scripts.assemble import main as assemble_main
        assemble_main(str(save_path), verbose=False, rounds=3)
        
        # Run relaxation if requested
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
