#!/usr/bin/env python3
"""
Generate high-quality MOF samples with proper structure validation.
Uses more optimization rounds and better assembly parameters.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import gc
from pytorch_lightning import seed_everything
from scipy.spatial import KDTree
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mofdiff.common.atomic_utils import arrange_decoded_mofs, mof2cif_with_bonds
from mofdiff.common.eval_utils import load_mofdiff_model
from mofdiff.common.optimization import (
    annealed_optimization,
    assemble_mof,
    feasibility_check,
)


def validate_structure(cif_path):
    """Validate that a CIF file contains a reasonable MOF structure."""
    try:
        from pymatgen.core import Structure
        struct = Structure.from_file(str(cif_path))
        
        # Check for reasonable size
        n_atoms = len(struct)
        volume = struct.volume
        
        # MOFs typically have >50 atoms and reasonable volume
        if n_atoms < 30:
            return False, f"Too few atoms: {n_atoms}"
        if volume < 100:
            return False, f"Volume too small: {volume:.1f} Å³"
        if volume / n_atoms < 5:
            return False, f"Density too high: {volume/n_atoms:.1f} Å³/atom"
        if volume / n_atoms > 200:
            return False, f"Density too low: {volume/n_atoms:.1f} Å³/atom"
            
        # Check lattice parameters
        a, b, c = struct.lattice.abc
        if min(a, b, c) < 5:
            return False, f"Lattice too small: {min(a,b,c):.1f} Å"
        if max(a, b, c) > 100:
            return False, f"Lattice too large: {max(a,b,c):.1f} Å"
            
        return True, f"Valid: {n_atoms} atoms, {volume:.1f} Å³"
    except Exception as e:
        return False, f"Error reading structure: {e}"


def assemble_one_quality(
    mof,
    verbose=True,
    rounds=5,  # More rounds for better optimization
    sigma_start=5.0,  # Higher starting sigma
    sigma_end=0.2,   # Lower ending sigma for tighter fit
    max_neighbors_start=40,  # More neighbors to consider
    max_neighbors_end=1,
    maxiter=300,  # More iterations
):
    """Assemble MOF with higher quality parameters."""
    if not feasibility_check(mof):
        return None

    sigma_schedule = np.linspace(sigma_start, sigma_end, rounds)
    max_neighbors_schedule = (
        np.linspace(max_neighbors_start, max_neighbors_end, rounds).round().astype(int)
    )

    now = time.time()
    results, v = annealed_optimization(
        mof,
        0,
        sigma_schedule=sigma_schedule,
        max_neighbors_schedule=max_neighbors_schedule,
        maxiter=maxiter,
        verbose=verbose,
    )
    elapsed = time.time() - now

    vecs = torch.from_numpy(results["x"]).view(mof.num_atoms, 3).float()
    bb_local_vectors = [bb.local_vectors for bb in mof.bbs]
    assembled_rec = assemble_mof(mof, vecs, bb_local_vectors=bb_local_vectors)

    assembled_rec.opt_v = v
    assembled_rec.assemble_time = elapsed
    return assembled_rec


def generate_samples(
    model, 
    kdtree, 
    all_data, 
    n_samples, 
    device,
    batch_size=32,
    max_attempts=3,
):
    """Generate samples with retry logic for quality."""
    output = defaultdict(list)
    all_latent_z = []
    
    generated = 0
    attempt = 0
    
    while generated < n_samples and attempt < max_attempts * n_samples:
        attempt += 1
        current_batch_size = min(batch_size, n_samples - generated)
        
        # Use slightly different random seeds for diversity
        z = torch.randn(current_batch_size, model.hparams.latent_dim).to(device)
        
        try:
            samples = model.sample(z.shape[0], z, save_freq=False)
            mofs = arrange_decoded_mofs(samples, kdtree, all_data, False)
            
            for mof in mofs:
                if generated >= n_samples:
                    break
                # Basic check - ensure MOF has reasonable number of building blocks
                if hasattr(mof, 'num_atoms') and mof.num_atoms >= 3:
                    output["mofs"].append(mof)
                    generated += 1
                    
            all_latent_z.append(z[:len(mofs)].cpu())
        except Exception as e:
            print(f"Batch failed: {e}")
            continue
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    if all_latent_z:
        output["z"] = torch.cat(all_latent_z, dim=0)[:n_samples]
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Generate high-quality MOF samples")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="nh3_all_samples", help="Output directory")
    parser.add_argument("--model_path", type=str, default="pretrained/mofdiff_ckpt", help="Path to MOFDiff model")
    parser.add_argument("--bb_cache_path", type=str, default="pretrained/bb_emb_space.pt", help="Building block cache")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--rounds", type=int, default=5, help="Assembly optimization rounds")
    parser.add_argument("--maxiter", type=int, default=300, help="Max iterations per round")
    parser.add_argument("--relax", action="store_true", help="Run UFF relaxation")
    parser.add_argument("--target_nh3", type=float, default=None, help="Target NH3 uptake for gradient optimization")
    args = parser.parse_args()
    
    seed_everything(args.seed)
    base_dir = Path(__file__).parent.parent
    
    # Setup paths
    model_path = (base_dir / args.model_path).resolve()
    bb_cache_path = (base_dir / args.bb_cache_path).resolve()
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cif_dir = output_dir / "cif"
    cif_dir.mkdir(exist_ok=True)
    
    # Load model
    print(f"Loading model from {model_path}...")
    model, cfg, bb_encoder = load_mofdiff_model(model_path)
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Load building block cache
    print(f"Loading building block cache from {bb_cache_path}...")
    all_data, all_z = torch.load(bb_cache_path)
    kdtree = KDTree(all_z)
    
    # Generate samples
    print(f"\nGenerating {args.n_samples} samples...")
    output = generate_samples(model, kdtree, all_data, args.n_samples, device)
    
    n_generated = len(output["mofs"])
    print(f"Generated {n_generated} candidate MOFs")
    
    # Assembly with quality parameters
    print(f"\nAssembling MOFs with {args.rounds} optimization rounds...")
    output["assembled"] = []
    output["assemble_info"] = []
    
    valid_samples = []
    assembly_stats = {"success": 0, "failed": 0, "invalid": 0}
    
    for i in tqdm(range(n_generated), desc="Assembling"):
        try:
            mof = output["mofs"][i].detach().cpu()
            assembled = assemble_one_quality(
                mof, 
                verbose=False, 
                rounds=args.rounds,
                maxiter=args.maxiter
            )
            
            if assembled is None:
                output["assembled"].append(None)
                output["assemble_info"].append("infeasible")
                assembly_stats["failed"] += 1
                continue
            
            # Save CIF
            cif_path = cif_dir / f"sample_{i}.cif"
            mof2cif_with_bonds(assembled, cif_path)
            
            # Validate structure
            is_valid, msg = validate_structure(cif_path)
            if is_valid:
                valid_samples.append(i)
                output["assembled"].append(assembled)
                output["assemble_info"].append(msg)
                assembly_stats["success"] += 1
                print(f"  Sample {i}: {msg}")
            else:
                # Remove invalid CIF
                cif_path.unlink()
                output["assembled"].append(None)
                output["assemble_info"].append(msg)
                assembly_stats["invalid"] += 1
                print(f"  Sample {i}: INVALID - {msg}")
                
        except Exception as e:
            print(f"  Sample {i}: ERROR - {e}")
            output["assembled"].append(None)
            output["assemble_info"].append(str(e))
            assembly_stats["failed"] += 1
    
    # Save assembled data
    torch.save(output, output_dir / "assembled.pt")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Assembly Summary:")
    print(f"  Successful: {assembly_stats['success']}")
    print(f"  Invalid: {assembly_stats['invalid']}")
    print(f"  Failed: {assembly_stats['failed']}")
    print(f"  Valid samples: {valid_samples}")
    
    # Relaxation
    if args.relax and assembly_stats["success"] > 0:
        print(f"\n{'='*60}")
        print("Running UFF relaxation...")
        
        relaxed_dir = output_dir / "relaxed"
        relaxed_dir.mkdir(exist_ok=True)
        
        from mofdiff.common.relaxation import relax_cif
        
        relax_info = {}
        for i in valid_samples:
            cif_path = cif_dir / f"sample_{i}.cif"
            if cif_path.exists():
                print(f"Relaxing sample {i}...")
                try:
                    relaxed_cif, info = relax_cif(
                        str(cif_path), 
                        str(relaxed_dir / f"sample_{i}_relaxed.cif"),
                        fmax=0.5,
                        steps=500
                    )
                    relax_info[f"sample_{i}"] = info
                    
                    # Validate relaxed structure
                    is_valid, msg = validate_structure(relaxed_cif)
                    if is_valid:
                        print(f"  Sample {i} relaxed: {msg}")
                    else:
                        print(f"  Sample {i} relaxed but invalid: {msg}")
                except Exception as e:
                    print(f"  Sample {i} relaxation failed: {e}")
                    relax_info[f"sample_{i}"] = {"error": str(e)}
        
        with open(relaxed_dir / "relax_info.json", "w") as f:
            json.dump(relax_info, f, indent=2, default=str)
    
    # Final summary
    final_cifs = list(cif_dir.glob("*.cif"))
    print(f"\n{'='*60}")
    print(f"Final Output:")
    print(f"  Output directory: {output_dir}")
    print(f"  CIF files: {len(final_cifs)}")
    if args.relax:
        relaxed_cifs = list((output_dir / "relaxed").glob("*_relaxed.cif"))
        print(f"  Relaxed CIFs: {len(relaxed_cifs)}")
    
    # Save summary
    summary = {
        "n_requested": args.n_samples,
        "n_generated": n_generated,
        "n_valid": assembly_stats["success"],
        "valid_indices": valid_samples,
        "assembly_stats": assembly_stats,
        "parameters": {
            "rounds": args.rounds,
            "maxiter": args.maxiter,
            "seed": args.seed
        }
    }
    with open(output_dir / "samples_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return output_dir


if __name__ == "__main__":
    main()
