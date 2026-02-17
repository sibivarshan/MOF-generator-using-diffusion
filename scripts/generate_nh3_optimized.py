#!/usr/bin/env python
"""
NH3-Optimized MOF Generation Pipeline

This script provides the complete end-to-end pipeline for generating MOFs
with optimized NH3 adsorption properties. It:

  1. Trains (or loads) a deep ensemble NH3 predictor on the latent space
  2. Optimizes latent vectors z for a user-specified NH3 target
  3. Feeds optimized z into the ORIGINAL MOFDiff sampling pipeline
  4. Assembles, relaxes (LAMMPS UFF), and validates via GCMC

CRITICAL: This script does NOT modify any of the original MOFDiff methods.
  - Sampling: mofdiff/scripts/sample.py (model.sample(z=z_optimized))
  - Assembly: mofdiff/scripts/assemble.py  
  - Relaxation: mofdiff/scripts/uff_relax.py (LAMMPS UFF)
  - GCMC: mofdiff/scripts/gcmc_nh3_screen.py (RASPA2)

Usage:
    # First time (trains predictor + optimizes + generates):
    python scripts/generate_nh3_optimized.py --target 5.0 --train_predictor

    # Subsequent runs (reuses trained predictor):
    python scripts/generate_nh3_optimized.py --target 10.0
    python scripts/generate_nh3_optimized.py --target 2.0 --n_samples 20

    # Just train the predictor:
    python scripts/generate_nh3_optimized.py --train_only

    # Just optimize (no downstream pipeline):
    python scripts/generate_nh3_optimized.py --target 8.0 --optimize_only
"""

import os
import sys
import json
import argparse
import shutil
import subprocess
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def train_predictor_step(args):
    """Train the NH3 predictor ensemble."""
    from mofdiff.nh3_optimizer.train import train_predictor
    
    print("\n" + "=" * 70)
    print("STEP 0: Training NH3 Predictor Ensemble")
    print("=" * 70)
    
    ensemble, results = train_predictor(
        data_path=str(PROJECT_ROOT / args.latent_dataset),
        save_path=str(PROJECT_ROOT / args.predictor_path),
        n_models=args.n_ensemble,
        hidden_dims=[512, 256, 128],
        dropout=0.15,
        lr=1e-3,
        weight_decay=1e-4,
        n_epochs=args.train_epochs,
        batch_size=32,
        patience=60,
        augment_noise=0.05,
        device=args.device,
        seed=args.seed,
    )
    
    return ensemble


def optimize_latents_step(args, predictor=None):
    """Optimize latent vectors for target NH3."""
    from mofdiff.nh3_optimizer.optimizer import LatentOptimizer, create_optimizer
    
    print("\n" + "=" * 70)
    print(f"STEP 1: Optimizing Latent Vectors for {args.target} mmol/g NH3")
    print("=" * 70)
    
    if predictor is not None:
        optimizer = LatentOptimizer(
            predictor=predictor,
            latent_dataset_path=str(PROJECT_ROOT / args.latent_dataset),
            device=args.device,
        )
    else:
        optimizer = create_optimizer(
            predictor_path=str(PROJECT_ROOT / args.predictor_path),
            latent_dataset_path=str(PROJECT_ROOT / args.latent_dataset),
            device=args.device,
        )
    
    result = optimizer.optimize(
        target_nh3=args.target,
        n_samples=args.n_samples,
        n_starts=max(args.n_samples * 2, 16),
        n_steps=args.opt_steps,
        lr=args.opt_lr,
        kl_weight=args.kl_weight,
        uncertainty_weight=args.uncertainty_weight,
        init_strategy="hybrid",
        verbose=True,
    )
    
    return result


def sample_with_optimized_z(args, opt_result):
    """
    Feed optimized z into the ORIGINAL MOFDiff sample method.
    
    This uses the exact same model.sample(z=z_optimized) as sample.py,
    preserving all diffusion model behavior.
    """
    from mofdiff.common.eval_utils import load_mofdiff_model
    from mofdiff.common.atomic_utils import arrange_decoded_mofs
    from scipy.spatial import KDTree
    import gc
    
    print("\n" + "=" * 70)
    print("STEP 2: Generating MOFs with Optimized Latent Vectors")
    print("=" * 70)
    
    model_path = Path(args.model_path).resolve()
    
    # Load model (same as OG sample.py)
    print(f"Loading MOFDiff model from {model_path}...")
    model, cfg, bb_encoder = load_mofdiff_model(model_path)
    device = args.device if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Load BB cache (same as OG sample.py)
    print(f"Loading building block cache from {args.bb_cache_path}...")
    all_data, all_z_bb = torch.load(args.bb_cache_path, map_location='cpu', weights_only=False)
    
    print("Building KDTree...")
    kdtree = KDTree(all_z_bb)
    
    # Use optimized z vectors
    z_optimized = opt_result['z'].to(device)
    n_samples = z_optimized.shape[0]
    
    print(f"Sampling {n_samples} MOFs with optimized latent vectors...")
    
    output = defaultdict(list)
    
    # Process in batches if needed (same pattern as OG sample.py)
    batch_size = min(n_samples, args.batch_size)
    n_batch = int(np.ceil(n_samples / batch_size))
    
    all_latent_z = []
    
    for idx in range(n_batch):
        start = idx * batch_size
        end = min(start + batch_size, n_samples)
        z_batch = z_optimized[start:end]
        
        print(f"  Batch {idx + 1}/{n_batch} ({end - start} samples)...")
        
        # ORIGINAL model.sample() - no modifications
        samples = model.sample(z_batch.shape[0], z_batch, save_freq=False)
        mofs = arrange_decoded_mofs(samples, kdtree, all_data, False)
        output["mofs"].extend(mofs)
        all_latent_z.append(z_batch.cpu())
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    output["z"] = torch.cat(all_latent_z, dim=0)
    
    # Add optimization metadata
    output["optimization"] = {
        'target_nh3': opt_result['target_nh3'],
        'predicted_nh3': opt_result['predicted_nh3'].tolist(),
        'uncertainty': opt_result['uncertainty'].tolist(),
    }
    
    # Save to output directory
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = output_dir / "samples.pt"
    print(f"Saving samples to {save_path}...")
    torch.save(output, save_path)
    
    return output_dir


def run_assembly(output_dir: Path):
    """Run OG assemble.py"""
    print("\n" + "=" * 70)
    print("STEP 3: Assembling MOF Structures")
    print("=" * 70)
    
    samples_path = output_dir / "samples.pt"
    cmd = f"conda run -n mofdiff-gpu python mofdiff/scripts/assemble.py --input {samples_path}"
    
    result = subprocess.run(
        cmd, shell=True, cwd=str(PROJECT_ROOT),
        capture_output=True, text=True, timeout=1800,
    )
    
    if result.returncode != 0:
        print(f"Assembly error: {result.stderr[-500:]}")
        return False
    
    print(result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
    
    cif_dir = output_dir / "cif"
    if cif_dir.exists():
        n_cifs = len(list(cif_dir.glob("*.cif")))
        print(f"✓ Assembled {n_cifs} structures")
    
    return True


def run_relaxation(output_dir: Path):
    """Run OG uff_relax.py (LAMMPS UFF)"""
    print("\n" + "=" * 70)
    print("STEP 4: LAMMPS UFF Relaxation")
    print("=" * 70)
    
    cmd = f"conda run -n mofdiff-gpu python mofdiff/scripts/uff_relax.py --input {output_dir}"
    
    result = subprocess.run(
        cmd, shell=True, cwd=str(PROJECT_ROOT),
        capture_output=True, text=True, timeout=3600,
    )
    
    if result.returncode != 0:
        print(f"Relaxation error: {result.stderr[-500:]}")
        # Don't fail - relaxation errors are common
    
    print(result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
    
    relaxed_dir = output_dir / "relaxed"
    if relaxed_dir.exists():
        n_relaxed = len(list(relaxed_dir.glob("*.cif")))
        print(f"✓ Relaxed {n_relaxed} structures")
    
    return True


def run_gcmc(output_dir: Path):
    """Run OG gcmc_nh3_screen.py"""
    print("\n" + "=" * 70)
    print("STEP 5: GCMC NH3 Screening")
    print("=" * 70)
    
    relaxed_dir = output_dir / "relaxed"
    if not relaxed_dir.exists() or len(list(relaxed_dir.glob("*.cif"))) == 0:
        # Fall back to CIF directory
        relaxed_dir = output_dir / "cif"
    
    # Set RASPA environment
    env = os.environ.copy()
    raspa_dir = env.get('RASPA_DIR', '/home/sibivarshan_m7/gcmc_tools/raspa_install')
    env['RASPA_DIR'] = raspa_dir
    env['RASPA_PATH'] = f"{raspa_dir}/share/raspa"
    env['RASPA_SIM_PATH'] = f"{raspa_dir}/bin/simulate"
    
    cmd = (
        f"conda run -n mofdiff-gpu python mofdiff/scripts/gcmc_nh3_screen.py "
        f"--input {relaxed_dir} "
        f"--ncpu 4 "
        f"--simulation_type uptake "
        f"--temperature 298 "
        f"--pressure 101325"
    )
    
    result = subprocess.run(
        cmd, shell=True, cwd=str(PROJECT_ROOT),
        capture_output=True, text=True, env=env,
    )
    
    if result.returncode != 0:
        print(f"GCMC error: {result.stderr[-500:]}")
        return False
    
    print(result.stdout[-1500:] if len(result.stdout) > 1500 else result.stdout)
    return True


def save_final_summary(output_dir: Path, opt_result: dict, args):
    """Save a comprehensive summary of the run."""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'target_nh3_mmol_g': args.target,
        'n_samples': args.n_samples,
        'optimization': {
            'predicted_nh3': opt_result['predicted_nh3'].tolist(),
            'uncertainty': opt_result['uncertainty'].tolist(),
            'n_steps': opt_result['optimization_info']['n_steps'],
            'best_step': opt_result['optimization_info']['best_step'],
            'best_loss': opt_result['optimization_info']['best_loss'],
        },
        'pipeline': [
            'mofdiff/scripts/sample.py (with optimized z)',
            'mofdiff/scripts/assemble.py',
            'mofdiff/scripts/uff_relax.py',
            'mofdiff/scripts/gcmc_nh3_screen.py',
        ],
    }
    
    # Check for GCMC results
    gcmc_dir = output_dir / "relaxed" / "gcmc_nh3"
    results_file = gcmc_dir / "nh3_uptake_results.json"
    if results_file.exists():
        with open(results_file) as f:
            gcmc_results = json.load(f)
        
        successful = [r for r in gcmc_results if r.get('adsorption_info')]
        if successful:
            uptakes = [r['adsorption_info']['NH3_uptake_mmol_g'] for r in successful]
            best_idx = np.argmax(uptakes)
            
            summary['gcmc_results'] = {
                'n_total': len(gcmc_results),
                'n_successful': len(successful),
                'best_structure': successful[best_idx]['uid'],
                'best_nh3_mmol_g': uptakes[best_idx],
                'mean_nh3_mmol_g': float(np.mean(uptakes)),
                'all_uptakes': {r['uid']: r['adsorption_info']['NH3_uptake_mmol_g']
                               for r in successful},
            }
            
            print(f"\n{'='*70}")
            print(f"FINAL RESULTS")
            print(f"{'='*70}")
            print(f"Target NH3:      {args.target:.2f} mmol/g")
            print(f"Best achieved:   {uptakes[best_idx]:.3f} mmol/g ({successful[best_idx]['uid']})")
            print(f"Mean achieved:   {np.mean(uptakes):.3f} mmol/g")
            print(f"Successful GCMC: {len(successful)}/{len(gcmc_results)}")
    
    with open(output_dir / "run_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to {output_dir / 'run_summary.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="NH3-Optimized MOF Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train predictor and generate for 5 mmol/g target:
  python scripts/generate_nh3_optimized.py --target 5.0 --train_predictor

  # Generate for 10 mmol/g (predictor already trained):
  python scripts/generate_nh3_optimized.py --target 10.0

  # Generate more samples with custom output:
  python scripts/generate_nh3_optimized.py --target 8.0 --n_samples 20

  # Just train the predictor:
  python scripts/generate_nh3_optimized.py --train_only

  # Just optimize (no sampling/assembly/GCMC):
  python scripts/generate_nh3_optimized.py --target 5.0 --optimize_only
        """,
    )
    
    # Target
    parser.add_argument("--target", type=float, default=5.0,
                        help="Target NH3 uptake in mmol/g")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Number of MOF samples to generate")
    
    # Paths
    parser.add_argument("--model_path", type=str,
                        default="pretrained/mofdiff_ckpt",
                        help="Path to MOFDiff model checkpoint")
    parser.add_argument("--bb_cache_path", type=str,
                        default="pretrained/bb_emb_space.pt",
                        help="Path to building block cache")
    parser.add_argument("--predictor_path", type=str,
                        default="pretrained/nh3_optimizer.pt",
                        help="Path to NH3 predictor ensemble")
    parser.add_argument("--latent_dataset", type=str,
                        default="raw_nh3_core/nh3_latent_dataset.pt",
                        help="Path to latent dataset")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: results/nh3_optimized_<target>)")
    
    # Training
    parser.add_argument("--train_predictor", action="store_true",
                        help="Train the predictor (even if checkpoint exists)")
    parser.add_argument("--train_only", action="store_true",
                        help="Only train the predictor, don't generate")
    parser.add_argument("--train_epochs", type=int, default=500,
                        help="Training epochs for predictor")
    parser.add_argument("--n_ensemble", type=int, default=10,
                        help="Number of models in ensemble")
    
    # Optimization
    parser.add_argument("--opt_steps", type=int, default=300,
                        help="Number of optimization steps")
    parser.add_argument("--opt_lr", type=float, default=0.05,
                        help="Optimization learning rate")
    parser.add_argument("--kl_weight", type=float, default=0.005,
                        help="KL divergence weight")
    parser.add_argument("--uncertainty_weight", type=float, default=0.1,
                        help="Uncertainty penalty weight")
    parser.add_argument("--optimize_only", action="store_true",
                        help="Only optimize latents, skip downstream pipeline")
    
    # Pipeline control
    parser.add_argument("--skip_relax", action="store_true",
                        help="Skip LAMMPS relaxation step")
    parser.add_argument("--skip_gcmc", action="store_true",
                        help="Skip GCMC validation step")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for MOFDiff sampling")
    
    # General
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = f"results/nh3_optimized_{args.target:.1f}"
    
    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    predictor_exists = Path(PROJECT_ROOT / args.predictor_path).exists()
    predictor = None
    
    # Step 0: Train predictor if needed
    if args.train_predictor or args.train_only or not predictor_exists:
        if not predictor_exists:
            print("No trained predictor found. Training from scratch...")
        predictor = train_predictor_step(args)
        
        if args.train_only:
            print("\nPredictor training complete. Exiting (--train_only).")
            return
    
    # Step 1: Optimize latent vectors
    opt_result = optimize_latents_step(args, predictor)
    
    if args.optimize_only:
        # Save optimized z
        output_dir = PROJECT_ROOT / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(opt_result, output_dir / "optimized_latents.pt")
        print(f"\nOptimized latents saved to {output_dir / 'optimized_latents.pt'}")
        print("Exiting (--optimize_only).")
        return
    
    # Step 2: Sample with optimized z (uses OG model.sample)
    output_dir = sample_with_optimized_z(args, opt_result)
    
    # Step 3: Assemble (uses OG assemble.py)
    if not run_assembly(output_dir):
        print("Assembly failed. Exiting.")
        return
    
    # Step 4: Relax (uses OG uff_relax.py with LAMMPS)
    if not args.skip_relax:
        run_relaxation(output_dir)
    
    # Step 5: GCMC (uses OG gcmc_nh3_screen.py)
    if not args.skip_gcmc:
        run_gcmc(output_dir)
    
    # Save summary
    save_final_summary(output_dir, opt_result, args)
    
    print(f"\n{'='*70}")
    print("Pipeline complete!")
    print(f"Results: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
