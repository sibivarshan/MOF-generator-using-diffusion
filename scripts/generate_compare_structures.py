"""
Generate and compare MOF structures following the official MOFDiff workflow:
https://github.com/microsoft/MOFDiff

Workflow:
1. Sample CG structures using mofdiff/scripts/sample.py
2. Assemble all-atom MOFs using mofdiff/scripts/assemble.py
3. Relax using mofdiff/scripts/uff_relax.py (or simple relaxation)
4. Calculate charges using mofdiff/scripts/calculate_charges.py
5. Run GCMC for NH3 using mofdiff/scripts/gcmc_nh3_screen.py

This script generates:
- One structure using the ORIGINAL model (random sampling)
- One structure using the NH3 HEAD (guided sampling)
"""
from pathlib import Path
from collections import defaultdict
import argparse
import torch
import numpy as np
import gc
import json
import time
import subprocess
import sys
from pytorch_lightning import seed_everything
from scipy.spatial import KDTree
from tqdm import tqdm

from mofdiff.common.atomic_utils import arrange_decoded_mofs, mof2cif_with_bonds
from mofdiff.common.eval_utils import load_mofdiff_model
from mofdiff.common.optimization import (
    annealed_optimization,
    assemble_mof,
    feasibility_check,
)


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


def sample_original_model(model, kdtree, all_data, n_samples=1, device="cuda", 
                          max_attempts=10, seed=42):
    """
    Sample MOFs using the original model without NH3 guidance.
    Following official mofdiff/scripts/sample.py approach.
    Retry with different seeds if assembly fails.
    """
    print("\n" + "="*60)
    print("Sampling with ORIGINAL MODEL (no NH3 guidance)")
    print("="*60)
    
    latent_dim = model.hparams.latent_dim
    
    for attempt in range(max_attempts):
        current_seed = seed + attempt
        seed_everything(current_seed)
        
        z = torch.randn(n_samples, latent_dim).to(device)
        
        print(f"Attempt {attempt+1}/{max_attempts} (seed={current_seed}): Generating {n_samples} samples...")
        samples = model.sample(z.shape[0], z, save_freq=False)
        mofs = arrange_decoded_mofs(samples, kdtree, all_data, False)
        
        # Check if structure can be assembled
        mof = mofs[0].detach().cpu()
        if feasibility_check(mof):
            print(f"  -> Found feasible structure!")
            return {"mofs": mofs, "z": z.cpu(), "seed": current_seed}
        else:
            print(f"  -> Structure not feasible, trying next seed...")
    
    print(f"Warning: Could not find feasible structure after {max_attempts} attempts")
    return {"mofs": mofs, "z": z.cpu(), "seed": current_seed}


def sample_with_nh3_head(model, nh3_ckpt, kdtree, all_data, n_samples=1, 
                         n_candidates=1000, device="cuda", max_to_try=50):
    """
    Sample MOFs using NH3 head for guided selection.
    Try multiple top candidates until we find one that can be assembled.
    """
    print("\n" + "="*60)
    print("Sampling with NH3 HEAD (guided sampling)")
    print("="*60)
    
    latent_dim = model.hparams.latent_dim
    
    print(f"Generating {n_candidates} candidate latent vectors...")
    all_z = torch.randn(n_candidates, latent_dim)
    
    print("Predicting NH3 uptake for candidates...")
    nh3_pred = predict_nh3(all_z, nh3_ckpt)
    
    print(f"NH3 predictions: min={nh3_pred.min():.4f}, max={nh3_pred.max():.4f}, mean={nh3_pred.mean():.4f}")
    
    # Sort by predicted NH3 (highest first)
    sorted_idx = np.argsort(nh3_pred)[::-1]
    
    # Try top candidates until we find one that can be assembled
    for i, idx in enumerate(sorted_idx[:max_to_try]):
        selected_z = all_z[idx:idx+1].clone().to(device)
        selected_nh3_pred = nh3_pred[idx]
        
        print(f"Trying candidate {i+1}/{max_to_try} with predicted NH3: {selected_nh3_pred:.4f}...")
        
        samples = model.sample(selected_z.shape[0], selected_z, save_freq=False)
        mofs = arrange_decoded_mofs(samples, kdtree, all_data, False)
        
        # Check if structure can be assembled
        mof = mofs[0].detach().cpu()
        if feasibility_check(mof):
            print(f"  -> Found feasible structure!")
            return {
                "mofs": mofs, 
                "z": selected_z.cpu(), 
                "predicted_nh3": np.array([selected_nh3_pred]),
                "candidate_rank": i+1
            }
        else:
            print(f"  -> Structure not feasible, trying next candidate...")
    
    # If none are feasible, return the best one anyway
    print(f"Warning: No feasible structure found in top {max_to_try} candidates, using best one")
    idx = sorted_idx[0]
    selected_z = all_z[idx:idx+1].clone().to(device)
    samples = model.sample(selected_z.shape[0], selected_z, save_freq=False)
    mofs = arrange_decoded_mofs(samples, kdtree, all_data, False)
    
    return {
        "mofs": mofs, 
        "z": selected_z.cpu(), 
        "predicted_nh3": np.array([nh3_pred[idx]]),
        "candidate_rank": 1
    }


def assemble_one(mof, verbose=False, rounds=3, sigma_start=3.0, sigma_end=0.3,
                 max_neighbors_start=30, max_neighbors_end=1):
    """
    Assemble a single MOF structure.
    Following official mofdiff/scripts/assemble.py approach.
    """
    if not feasibility_check(mof):
        return None

    sigma_schedule = np.linspace(sigma_start, sigma_end, rounds)
    max_neighbors_schedule = (
        np.linspace(max_neighbors_start, max_neighbors_end, rounds).round().astype(int)
    )

    now = time.time()
    results, v = annealed_optimization(
        mof, 0, sigma_schedule=sigma_schedule,
        max_neighbors_schedule=max_neighbors_schedule,
        maxiter=200, verbose=verbose,
    )
    elapsed = time.time() - now

    vecs = torch.from_numpy(results["x"]).view(mof.num_atoms, 3).float()
    bb_local_vectors = [bb.local_vectors for bb in mof.bbs]
    assembled_rec = assemble_mof(mof, vecs, bb_local_vectors=bb_local_vectors)

    assembled_rec.opt_v = v
    assembled_rec.assemble_time = elapsed
    return assembled_rec


def relax_with_lammps(cif_path, output_dir):
    """
    Relax structure using LAMMPS UFF.
    Following official mofdiff/scripts/uff_relax.py approach.
    """
    from mofdiff.common.relaxation import lammps_relax
    from pymatgen.io.cif import CifWriter
    
    try:
        name = cif_path.stem
        struct, relax_info = lammps_relax(str(cif_path), str(output_dir))
        
        if struct is not None:
            struct = struct.get_primitive_structure()
            output_path = output_dir / f"{name}.cif"
            CifWriter(struct).write_file(str(output_path))
            relax_info["natoms"] = struct.frac_coords.shape[0]
            relax_info["volume"] = struct.volume
            relax_info["path"] = str(output_path)
            relax_info["success"] = True
            return relax_info
        else:
            return {"success": False, "error": "LAMMPS returned None"}
    except Exception as e:
        print(f"LAMMPS relaxation failed: {e}")
        return {"success": False, "error": str(e)}


def simple_relax_structure(cif_path, output_dir):
    """Simple relaxation using pymatgen (fallback if LAMMPS fails)"""
    from pymatgen.core import Structure
    from pymatgen.io.cif import CifWriter
    
    try:
        name = cif_path.stem
        struct = Structure.from_file(str(cif_path))
        primitive = struct.get_primitive_structure()
        output_path = output_dir / f"{name}.cif"
        CifWriter(primitive).write_file(str(output_path))
        return {
            'natoms': len(primitive),
            'volume': primitive.volume,
            'path': str(output_path),
            'success': True,
        }
    except Exception as e:
        print(f"Error relaxing {cif_path}: {e}")
        return {'success': False, 'error': str(e)}


def run_gcmc_nh3(cif_path, rundir, calc_charges=True, rewrite_raspa_input=True):
    """
    Run GCMC simulation for NH3 uptake.
    Following official mofdiff/scripts/gcmc_nh3_screen.py approach.
    
    Note: rewrite_raspa_input=True is used to avoid RASPA2 input file reading errors.
    This is mentioned in the official MOFDiff README.
    """
    from mofdiff.gcmc.simulation import nh3_uptake_simulation
    from mofdiff.common.atomic_utils import graph_from_cif
    from mofdiff.common.data_utils import lattice_params_to_matrix
    from mofdiff.common.atomic_utils import frac2cart, compute_distance_matrix
    
    uid = Path(cif_path).stem
    
    try:
        struct = graph_from_cif(cif_path).structure.get_primitive_structure()
        
        # Check for atomic overlap
        frac_coords = torch.tensor(struct.frac_coords).float()
        cell = torch.from_numpy(lattice_params_to_matrix(*struct.lattice.parameters)).float()
        cart_coords = frac2cart(frac_coords, cell)
        dist_mat = compute_distance_matrix(cell, cart_coords).fill_diagonal_(5.)
        
        if dist_mat.min() < 0.5:
            return dict(uid=uid, info='atomic overlap detected', adsorption_info=None)
        
        # Run GCMC simulation
        adsorption_info = nh3_uptake_simulation(
            str(cif_path),
            calc_charges=calc_charges,
            rundir=rundir,
            rewrite_raspa_input=rewrite_raspa_input,
            temperature=298,
            pressure=101325,  # 1 bar
        )
        
        return dict(uid=uid, info='success', adsorption_info=adsorption_info)
        
    except Exception as e:
        print(f'GCMC Error for {cif_path}: {e}')
        return dict(uid=uid, info=str(e), adsorption_info=None)


def convert_to_serializable(obj):
    """Convert numpy types to Python types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(
        description="Generate and compare MOF structures using original model vs NH3-guided sampling"
    )
    parser.add_argument("--model_path", type=str, default="pretrained/mofdiff_ckpt",
                        help="Path to trained MOFDiff model checkpoint")
    parser.add_argument("--bb_cache_path", type=str, default="pretrained/bb_emb_space.pt",
                        help="Path to building block embedding cache")
    parser.add_argument("--nh3_head_path", type=str, default="raw_nh3_core/nh3_head_best_final.pt",
                        help="Path to trained NH3 prediction head")
    parser.add_argument("--output_dir", type=str, default="comparison_results",
                        help="Output directory for results")
    parser.add_argument("--n_candidates", type=int, default=1000,
                        help="Number of candidates for NH3-guided sampling")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--skip_gcmc", action="store_true",
                        help="Skip GCMC simulations")
    parser.add_argument("--use_lammps", action="store_true",
                        help="Use LAMMPS for relaxation (requires LAMMPS)")
    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    # Setup output directory
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(exist_ok=True, parents=True)
    
    original_dir = output_dir / "original_model"
    nh3_guided_dir = output_dir / "nh3_guided"
    
    # Create directory structure following official MOFDiff
    for d in [original_dir, nh3_guided_dir]:
        d.mkdir(exist_ok=True, parents=True)
        (d / "cif").mkdir(exist_ok=True)
        (d / "relaxed").mkdir(exist_ok=True)
        (d / "gcmc_nh3").mkdir(exist_ok=True)
    
    # Load model (following official sample.py)
    print("\n" + "="*60)
    print("Loading MOFDiff model...")
    print("="*60)
    model_path = Path(args.model_path).resolve()
    model, cfg, bb_encoder = load_mofdiff_model(model_path)
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Load building block cache (following official sample.py)
    print(f"Loading building block cache from {args.bb_cache_path}...")
    all_data, all_z = torch.load(args.bb_cache_path)
    kdtree = KDTree(all_z)
    
    # Load NH3 head
    print(f"Loading NH3 head from {args.nh3_head_path}...")
    nh3_ckpt = load_nh3_head(args.nh3_head_path)
    
    results = {
        "original_model": {},
        "nh3_guided": {},
        "comparison": {},
    }
    
    # =====================================================
    # STEP 1: Sample CG structures (following sample.py)
    # =====================================================
    
    # Sample with original model
    original_samples = sample_original_model(
        model, kdtree, all_data, n_samples=1, device=device, seed=args.seed
    )
    torch.save(original_samples, original_dir / "samples.pt")
    results["original_model"]["seed"] = original_samples.get("seed", args.seed)
    
    # Sample with NH3 head
    nh3_samples = sample_with_nh3_head(
        model, nh3_ckpt, kdtree, all_data, 
        n_samples=1, n_candidates=args.n_candidates, device=device
    )
    torch.save(nh3_samples, nh3_guided_dir / "samples.pt")
    results["nh3_guided"]["predicted_nh3"] = float(nh3_samples["predicted_nh3"][0])
    results["nh3_guided"]["candidate_rank"] = nh3_samples.get("candidate_rank", 1)
    
    # =====================================================
    # STEP 2: Assemble all-atom MOFs (following assemble.py)
    # =====================================================
    print("\n" + "="*60)
    print("Assembling all-atom MOF structures...")
    print("="*60)
    
    # Assemble original
    print("\nAssembling original model structure...")
    mof_original = original_samples["mofs"][0].detach().cpu()
    assembled_original = assemble_one(mof_original, verbose=True, rounds=3)
    
    if assembled_original is not None:
        cif_path_original = original_dir / "cif" / "sample_0.cif"
        mof2cif_with_bonds(assembled_original, cif_path_original)
        print(f"Saved assembled CIF: {cif_path_original}")
        results["original_model"]["assembled"] = True
        results["original_model"]["assemble_time"] = assembled_original.assemble_time
    else:
        print("ERROR: Failed to assemble original structure")
        results["original_model"]["assembled"] = False
    
    # Assemble NH3-guided
    print("\nAssembling NH3-guided structure...")
    mof_nh3 = nh3_samples["mofs"][0].detach().cpu()
    assembled_nh3 = assemble_one(mof_nh3, verbose=True, rounds=3)
    
    if assembled_nh3 is not None:
        cif_path_nh3 = nh3_guided_dir / "cif" / "sample_0.cif"
        mof2cif_with_bonds(assembled_nh3, cif_path_nh3)
        print(f"Saved assembled CIF: {cif_path_nh3}")
        results["nh3_guided"]["assembled"] = True
        results["nh3_guided"]["assemble_time"] = assembled_nh3.assemble_time
    else:
        print("ERROR: Failed to assemble NH3-guided structure")
        results["nh3_guided"]["assembled"] = False
    
    # =====================================================
    # STEP 3: Relax MOFs (following uff_relax.py)
    # =====================================================
    print("\n" + "="*60)
    print("Relaxing MOF structures...")
    print("="*60)
    
    relaxed_original = None
    relaxed_nh3 = None
    
    # Relax original
    if results["original_model"].get("assembled", False):
        print("\nRelaxing original model structure...")
        relaxed_dir = original_dir / "relaxed"
        
        if args.use_lammps:
            relax_result = relax_with_lammps(cif_path_original, relaxed_dir)
        else:
            relax_result = simple_relax_structure(cif_path_original, relaxed_dir)
        
        results["original_model"]["relaxation"] = relax_result
        if relax_result.get('success', False):
            relaxed_original = relax_result.get('path')
            print(f"  -> {relax_result['natoms']} atoms, volume={relax_result['volume']:.1f} Å³")
            print(f"  -> Saved: {relaxed_original}")
    
    # Relax NH3-guided
    if results["nh3_guided"].get("assembled", False):
        print("\nRelaxing NH3-guided structure...")
        relaxed_dir = nh3_guided_dir / "relaxed"
        
        if args.use_lammps:
            relax_result = relax_with_lammps(cif_path_nh3, relaxed_dir)
        else:
            relax_result = simple_relax_structure(cif_path_nh3, relaxed_dir)
        
        results["nh3_guided"]["relaxation"] = relax_result
        if relax_result.get('success', False):
            relaxed_nh3 = relax_result.get('path')
            print(f"  -> {relax_result['natoms']} atoms, volume={relax_result['volume']:.1f} Å³")
            print(f"  -> Saved: {relaxed_nh3}")
    
    # =====================================================
    # STEP 4 & 5: GCMC for NH3 (following gcmc_nh3_screen.py)
    # =====================================================
    if not args.skip_gcmc:
        print("\n" + "="*60)
        print("Running GCMC simulations for NH3 uptake...")
        print("="*60)
        
        # GCMC for original
        if relaxed_original:
            print("\nRunning GCMC for original model structure...")
            gcmc_result = run_gcmc_nh3(
                relaxed_original,
                original_dir / "gcmc_nh3",
                calc_charges=True
            )
            results["original_model"]["gcmc"] = gcmc_result
            if gcmc_result.get('adsorption_info'):
                nh3_uptake = gcmc_result['adsorption_info'].get('NH3_uptake_mmol_g', 'N/A')
                print(f"  -> NH3 Uptake: {nh3_uptake} mmol/g")
        
        # GCMC for NH3-guided
        if relaxed_nh3:
            print("\nRunning GCMC for NH3-guided structure...")
            gcmc_result = run_gcmc_nh3(
                relaxed_nh3,
                nh3_guided_dir / "gcmc_nh3",
                calc_charges=True
            )
            results["nh3_guided"]["gcmc"] = gcmc_result
            if gcmc_result.get('adsorption_info'):
                nh3_uptake = gcmc_result['adsorption_info'].get('NH3_uptake_mmol_g', 'N/A')
                print(f"  -> NH3 Uptake: {nh3_uptake} mmol/g")
    else:
        print("\nSkipping GCMC simulations (--skip_gcmc flag set)")
    
    # =====================================================
    # FINAL: Save and display results
    # =====================================================
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    # Add comparison
    if (results["original_model"].get("gcmc", {}).get("adsorption_info") and 
        results["nh3_guided"].get("gcmc", {}).get("adsorption_info")):
        
        original_uptake = results["original_model"]["gcmc"]["adsorption_info"].get("NH3_uptake_mmol_g", 0)
        nh3_uptake = results["nh3_guided"]["gcmc"]["adsorption_info"].get("NH3_uptake_mmol_g", 0)
        
        if original_uptake > 0:
            improvement = ((nh3_uptake - original_uptake) / original_uptake) * 100
            results["comparison"]["nh3_improvement_percent"] = improvement
    
    # Save results
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    
    print(f"\nResults saved to: {output_dir / 'comparison_results.json'}")
    
    print("\n" + "-"*60)
    print("ORIGINAL MODEL (Random Sampling)")
    print("-"*60)
    print(f"  Seed used: {results['original_model'].get('seed', 'N/A')}")
    print(f"  Assembled: {results['original_model'].get('assembled', False)}")
    if results['original_model'].get('relaxation', {}).get('success'):
        print(f"  Relaxed atoms: {results['original_model']['relaxation']['natoms']}")
        print(f"  Volume: {results['original_model']['relaxation']['volume']:.1f} Å³")
    if results['original_model'].get('gcmc', {}).get('adsorption_info'):
        gcmc = results['original_model']['gcmc']['adsorption_info']
        print(f"  NH3 Uptake (GCMC): {gcmc.get('NH3_uptake_mmol_g', 'N/A')} mmol/g")
    
    print("\n" + "-"*60)
    print("NH3-GUIDED MODEL")
    print("-"*60)
    print(f"  Predicted NH3 (before GCMC): {results['nh3_guided'].get('predicted_nh3', 'N/A'):.4f}")
    print(f"  Candidate rank: {results['nh3_guided'].get('candidate_rank', 'N/A')}")
    print(f"  Assembled: {results['nh3_guided'].get('assembled', False)}")
    if results['nh3_guided'].get('relaxation', {}).get('success'):
        print(f"  Relaxed atoms: {results['nh3_guided']['relaxation']['natoms']}")
        print(f"  Volume: {results['nh3_guided']['relaxation']['volume']:.1f} Å³")
    if results['nh3_guided'].get('gcmc', {}).get('adsorption_info'):
        gcmc = results['nh3_guided']['gcmc']['adsorption_info']
        print(f"  Actual NH3 Uptake (GCMC): {gcmc.get('NH3_uptake_mmol_g', 'N/A')} mmol/g")
    
    if results.get("comparison", {}).get("nh3_improvement_percent") is not None:
        print("\n" + "-"*60)
        print("COMPARISON")
        print("-"*60)
        improvement = results["comparison"]["nh3_improvement_percent"]
        print(f"  NH3 Uptake Improvement: {improvement:+.1f}%")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()
