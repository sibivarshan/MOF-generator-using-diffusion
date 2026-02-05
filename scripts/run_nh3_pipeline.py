#!/usr/bin/env python
"""
Modular MOFDiff Pipeline for NH3-Targeted MOF Generation

This script provides a complete end-to-end pipeline for generating MOFs
with specific NH3 uptake targets. Simply provide the target value and
the script handles generation, assembly, relaxation, and GCMC validation.

Usage:
    python scripts/run_nh3_pipeline.py --target 10.0
    python scripts/run_nh3_pipeline.py --target 2.0 --n_samples 20 --seed 42
    python scripts/run_nh3_pipeline.py --target 5.0 --skip_gcmc  # Skip GCMC step
"""

import os
import sys
import json
import argparse
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_command(cmd, description, cwd=None):
    """Run a shell command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd or PROJECT_ROOT,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return False, result.stderr
    
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    return True, result.stdout


def generate_samples(output_dir: Path, n_samples: int, seed: int):
    """Generate MOF samples using MOFDiff"""
    cmd = f"""python mofdiff/scripts/sample.py \
        --model_path pretrained/mofdiff_ckpt \
        --bb_cache_path pretrained/bb_emb_space.pt \
        --n_samples {n_samples} \
        --seed {seed}"""
    
    success, output = run_command(cmd, f"Generating {n_samples} MOF samples (seed={seed})")
    
    if success:
        # Copy samples to output directory
        samples_path = PROJECT_ROOT / f"pretrained/mofdiff_ckpt/samples_{n_samples}_seed_{seed}/samples.pt"
        if samples_path.exists():
            shutil.copy(samples_path, output_dir / "samples.pt")
            print(f"✓ Samples copied to {output_dir / 'samples.pt'}")
    
    return success


def assemble_structures(output_dir: Path):
    """Assemble MOF structures from coarse-grained samples"""
    samples_path = output_dir / "samples.pt"
    cmd = f"python mofdiff/scripts/assemble.py --input {samples_path}"
    
    success, output = run_command(cmd, "Assembling MOF structures")
    
    if success:
        # Check how many were assembled
        cif_dir = output_dir / "cif"
        if cif_dir.exists():
            n_cifs = len(list(cif_dir.glob("*.cif")))
            print(f"✓ Assembled {n_cifs} structures")
    
    return success


def relax_structures(output_dir: Path):
    """Relax MOF structures using Pymatgen"""
    cmd = f"python scripts/simple_relax.py --input_dir {output_dir}"
    
    success, output = run_command(cmd, "Relaxing MOF structures")
    
    if success:
        relaxed_dir = output_dir / "relaxed"
        if relaxed_dir.exists():
            n_relaxed = len(list(relaxed_dir.glob("*.cif")))
            print(f"✓ Relaxed {n_relaxed} structures")
    
    return success


def run_gcmc(output_dir: Path, target_nh3: float):
    """Run GCMC simulations to validate NH3 uptake"""
    relaxed_dir = output_dir / "relaxed"
    
    cmd = f"""python scripts/run_gcmc_parallel.py \
        --input_dir {relaxed_dir} \
        --target_nh3 {target_nh3}"""
    
    success, output = run_command(cmd, f"Running GCMC (target: {target_nh3} mmol/g)")
    
    return success


def save_final_results(output_dir: Path, target_nh3: float, config: dict):
    """Save final results and identify best structure"""
    gcmc_results_file = output_dir / "gcmc_parallel" / "all_results.json"
    
    if not gcmc_results_file.exists():
        print("Warning: GCMC results not found")
        return None
    
    with open(gcmc_results_file) as f:
        gcmc_results = json.load(f)
    
    # Find successful results
    successful = {
        k: v for k, v in gcmc_results.items() 
        if 'NH3_uptake_mmol_g' in v
    }
    
    if not successful:
        print("No successful GCMC results")
        return None
    
    # Find best match
    best_name = min(successful.keys(), 
                    key=lambda k: abs(successful[k]['NH3_uptake_mmol_g'] - target_nh3))
    best_uptake = successful[best_name]['NH3_uptake_mmol_g']
    
    # Copy best structure
    best_cif_src = output_dir / "relaxed" / f"{best_name}.cif"
    best_cif_dst = output_dir / f"best_structure_{target_nh3}mmol.cif"
    if best_cif_src.exists():
        shutil.copy(best_cif_src, best_cif_dst)
    
    # Create final results
    final_results = {
        "target_nh3_mmol_g": target_nh3,
        "config": config,
        "timestamp": datetime.now().isoformat(),
        "best_result": {
            "structure": best_name,
            "nh3_uptake_mmol_g": best_uptake,
            "difference_from_target": abs(best_uptake - target_nh3)
        },
        "all_results": {
            k: v.get('NH3_uptake_mmol_g', 'ERROR') 
            for k, v in gcmc_results.items()
        },
        "n_successful": len(successful),
        "n_total": len(gcmc_results)
    }
    
    results_file = output_dir / "final_results.json"
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    return final_results


def print_summary(results: dict, target_nh3: float):
    """Print a summary of results"""
    print("\n" + "="*60)
    print("PIPELINE COMPLETE - SUMMARY")
    print("="*60)
    
    if results:
        best = results['best_result']
        print(f"\nTarget NH3 Uptake: {target_nh3} mmol/g")
        print(f"Best Structure:    {best['structure']}")
        print(f"Achieved Uptake:   {best['nh3_uptake_mmol_g']:.4f} mmol/g")
        print(f"Difference:        {best['difference_from_target']:.4f} mmol/g")
        print(f"\nSuccessful: {results['n_successful']}/{results['n_total']} structures")
        
        print("\nAll Results:")
        for name, uptake in sorted(results['all_results'].items(), 
                                   key=lambda x: x[1] if isinstance(x[1], float) else 999):
            if isinstance(uptake, float):
                diff = uptake - target_nh3
                print(f"  {name}: {uptake:.4f} mmol/g (diff: {diff:+.2f})")
            else:
                print(f"  {name}: {uptake}")
    else:
        print("No results available")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Modular MOFDiff Pipeline for NH3-Targeted MOF Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_nh3_pipeline.py --target 10.0
  python scripts/run_nh3_pipeline.py --target 2.0 --n_samples 20
  python scripts/run_nh3_pipeline.py --target 5.0 --seed 42
  python scripts/run_nh3_pipeline.py --target 8.0 --skip_gcmc
        """
    )
    
    parser.add_argument('--target', type=float, required=True,
                        help='Target NH3 uptake in mmol/g (e.g., 2.0, 5.0, 10.0)')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of MOF samples to generate (default: 10)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: based on target)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: results/nh3_target_Xmmol)')
    parser.add_argument('--skip_generation', action='store_true',
                        help='Skip sample generation (use existing samples)')
    parser.add_argument('--skip_assembly', action='store_true',
                        help='Skip assembly (use existing CIFs)')
    parser.add_argument('--skip_relaxation', action='store_true',
                        help='Skip relaxation (use existing relaxed CIFs)')
    parser.add_argument('--skip_gcmc', action='store_true',
                        help='Skip GCMC simulation')
    
    args = parser.parse_args()
    
    # Setup
    target_nh3 = args.target
    n_samples = args.n_samples
    seed = args.seed if args.seed else int(target_nh3 * 1000 + 1234)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / f"results/nh3_target_{target_nh3}mmol"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        "target_nh3": target_nh3,
        "n_samples": n_samples,
        "seed": seed,
        "output_dir": str(output_dir)
    }
    
    print("\n" + "="*60)
    print("MOFDiff NH3-Targeted Generation Pipeline")
    print("="*60)
    print(f"Target NH3 Uptake: {target_nh3} mmol/g")
    print(f"Number of Samples: {n_samples}")
    print(f"Random Seed:       {seed}")
    print(f"Output Directory:  {output_dir}")
    print("="*60)
    
    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Step 1: Generate samples
    if not args.skip_generation:
        if not generate_samples(output_dir, n_samples, seed):
            print("ERROR: Sample generation failed")
            return 1
    else:
        print("\n[Skipping generation - using existing samples]")
    
    # Step 2: Assemble structures
    if not args.skip_assembly:
        if not assemble_structures(output_dir):
            print("ERROR: Assembly failed")
            return 1
    else:
        print("\n[Skipping assembly - using existing CIFs]")
    
    # Step 3: Relax structures
    if not args.skip_relaxation:
        if not relax_structures(output_dir):
            print("ERROR: Relaxation failed")
            return 1
    else:
        print("\n[Skipping relaxation - using existing relaxed CIFs]")
    
    # Step 4: Run GCMC
    if not args.skip_gcmc:
        if not run_gcmc(output_dir, target_nh3):
            print("ERROR: GCMC failed")
            return 1
        
        # Save and display results
        results = save_final_results(output_dir, target_nh3, config)
        print_summary(results, target_nh3)
    else:
        print("\n[Skipping GCMC]")
    
    print(f"\n✓ Pipeline complete! Results saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
