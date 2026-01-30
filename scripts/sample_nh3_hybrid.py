"""
HYBRID approach: Screening + Gradient Optimization

1. Screen many random candidates to find good starting points
2. Use gradient optimization to refine them towards the target
3. This avoids local optima while achieving precise targeting
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

# Import from our other scripts
from sample_nh3_gradient import NH3PredictorMLP, load_or_train_predictor
from sample_nh3_target import load_nh3_head, predict_nh3


def hybrid_optimization(
    nn_model,
    y_mean,
    y_std,
    svr_ckpt,
    target_nh3,
    latent_dim=256,
    n_samples=5,
    n_screening=5000,
    top_k_refine=50,
    lr=0.05,
    steps=200,
    regularization=0.01,
    device="cuda",
):
    """
    Hybrid optimization: Screening + Gradient refinement.
    
    1. Generate n_screening random candidates
    2. Use SVR to predict NH3 and select top_k closest to target
    3. Use gradient optimization to refine each towards target
    4. Return best n_samples
    """
    print(f"\n{'='*60}")
    print(f"HYBRID OPTIMIZATION (Screening + Gradient)")
    print(f"{'='*60}")
    print(f"Target NH3: {target_nh3:.4f} mmol/g")
    print(f"Step 1: Screening {n_screening} candidates")
    print(f"Step 2: Refining top {top_k_refine} with {steps} gradient steps")
    
    # Step 1: Screening with SVR
    print(f"\n[Step 1] Generating and screening candidates...")
    all_z = torch.randn(n_screening, latent_dim)
    nh3_pred = predict_nh3(all_z, svr_ckpt)  # Uses SVR, returns original scale
    
    print(f"SVR predictions: range=[{nh3_pred.min():.4f}, {nh3_pred.max():.4f}], mean={nh3_pred.mean():.4f}")
    
    # Select top_k closest to target
    distances = np.abs(nh3_pred - target_nh3)
    top_idx = np.argsort(distances)[:top_k_refine]
    
    candidates_z = all_z[top_idx].clone()
    candidates_nh3 = nh3_pred[top_idx]
    
    print(f"Selected {top_k_refine} candidates with SVR-predicted NH3: "
          f"[{candidates_nh3.min():.4f}, {candidates_nh3.max():.4f}]")
    
    # Step 2: Gradient refinement with NN
    print(f"\n[Step 2] Gradient refinement...")
    nn_model.eval()
    
    # Normalize target for NN
    target_normalized = (target_nh3 - y_mean) / y_std
    
    refined_z = []
    refined_nh3 = []
    
    for i, z_init in enumerate(tqdm(candidates_z, desc="Refining")):
        z = z_init.unsqueeze(0).to(device).clone().requires_grad_(True)
        optimizer = optim.Adam([z], lr=lr)
        
        for step in range(steps):
            optimizer.zero_grad()
            pred_normalized = nn_model(z)
            loss = (pred_normalized - target_normalized) ** 2 + regularization * (z ** 2).mean()
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            final_pred = nn_model(z) * y_std + y_mean
            refined_z.append(z.detach().cpu())
            refined_nh3.append(final_pred.item())
    
    refined_z = torch.cat(refined_z, dim=0)
    refined_nh3 = np.array(refined_nh3)
    
    print(f"After refinement: NH3 range=[{refined_nh3.min():.4f}, {refined_nh3.max():.4f}]")
    
    # Select best n_samples
    final_distances = np.abs(refined_nh3 - target_nh3)
    best_idx = np.argsort(final_distances)[:n_samples]
    
    final_z = refined_z[best_idx]
    final_nh3 = refined_nh3[best_idx]
    
    print(f"\nFinal {n_samples} samples:")
    for i, (nh3, dist) in enumerate(zip(final_nh3, final_distances[best_idx])):
        print(f"  Sample {i}: NH3 = {nh3:.4f} mmol/g, distance = {dist:.4f}")
    
    return final_z, final_nh3


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid: Screening + Gradient optimization")
    parser.add_argument("--model_path", type=str, default="pretrained/mofdiff_ckpt")
    parser.add_argument("--bb_cache_path", type=str, default="pretrained/bb_emb_space.pt")
    parser.add_argument("--latent_data_path", type=str, default="raw_nh3_core/nh3_latent_dataset.pt")
    parser.add_argument("--svr_ckpt_path", type=str, default="raw_nh3_core/nh3_head_best_final.pt")
    
    parser.add_argument("--target_nh3", type=float, required=True)
    parser.add_argument("--n_samples", default=5, type=int)
    parser.add_argument("--n_screening", default=5000, type=int,
                        help="Number of random candidates to screen")
    parser.add_argument("--top_k_refine", default=50, type=int,
                        help="Number of top candidates to refine with gradient")
    parser.add_argument("--lr", default=0.05, type=float)
    parser.add_argument("--steps", default=200, type=int)
    parser.add_argument("--regularization", default=0.01, type=float)
    
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--relax", action="store_true")

    args = parser.parse_args()
    seed_everything(args.seed)
    
    model_path = Path(args.model_path).resolve()
    
    if args.output_dir is None:
        output_dir = Path(f"nh3_hybrid_{args.target_nh3:.2f}")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load SVR predictor (for screening)
    print(f"Loading SVR predictor from {args.svr_ckpt_path}...")
    svr_ckpt = load_nh3_head(args.svr_ckpt_path)
    
    # Load NN predictor (for gradient refinement)
    nn_model, y_mean, y_std = load_or_train_predictor(
        args.latent_data_path, device=device
    )
    
    # Hybrid optimization
    optimized_z, predicted_nh3 = hybrid_optimization(
        nn_model, y_mean, y_std, svr_ckpt,
        target_nh3=args.target_nh3,
        n_samples=args.n_samples,
        n_screening=args.n_screening,
        top_k_refine=args.top_k_refine,
        lr=args.lr,
        steps=args.steps,
        regularization=args.regularization,
        device=device,
    )
    
    # Load MOFDiff for decoding
    print(f"\nLoading MOFDiff model from {model_path}...")
    mofdiff_model, cfg, bb_encoder = load_mofdiff_model(model_path)
    mofdiff_model = mofdiff_model.to(device)
    
    print(f"Loading building block cache from {args.bb_cache_path}...")
    all_data, all_z = torch.load(args.bb_cache_path)
    kdtree = KDTree(all_z)
    
    # Decode
    print(f"\nDecoding {len(optimized_z)} MOFs...")
    output = defaultdict(list)
    
    n_batch = int(np.ceil(len(optimized_z) / args.batch_size))
    for idx in range(n_batch):
        start = idx * args.batch_size
        end = min(start + args.batch_size, len(optimized_z))
        z_batch = optimized_z[start:end].to(device)
        
        samples = mofdiff_model.sample(z_batch.shape[0], z_batch, save_freq=False)
        mofs = arrange_decoded_mofs(samples, kdtree, all_data, False)
        output["mofs"].extend(mofs)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    output["z"] = optimized_z
    output["predicted_nh3"] = predicted_nh3
    output["target_nh3"] = args.target_nh3
    output["method"] = "hybrid"
    
    save_path = output_dir / "samples.pt"
    torch.save(output, save_path)
    
    print("\n" + "="*60)
    print("HYBRID OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Target: {args.target_nh3:.4f} mmol/g")
    print(f"Generated {len(output['mofs'])} MOFs")
    print(f"NH3: mean={predicted_nh3.mean():.4f}, range=[{predicted_nh3.min():.4f}, {predicted_nh3.max():.4f}]")
    print(f"Avg distance from target: {np.abs(predicted_nh3 - args.target_nh3).mean():.4f}")
    print(f"Saved to: {output_dir}")
    
    if args.assemble:
        from mofdiff.scripts.assemble import main as assemble_main
        assemble_main(str(save_path), verbose=False, rounds=3)
        
        if args.relax:
            import subprocess
            subprocess.run(["python", "scripts/simple_relax.py", "--input_dir", str(output_dir)])
