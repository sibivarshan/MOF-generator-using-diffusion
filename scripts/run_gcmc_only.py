"""
Run GCMC simulations for NH3 uptake on already relaxed structures.
"""
import os
import sys
import json
from pathlib import Path

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mofdiff.gcmc.simulation import nh3_uptake_simulation


def run_gcmc_for_structure(cif_path, rundir):
    """Run GCMC simulation for NH3 uptake"""
    print(f"\nRunning GCMC for: {cif_path}")
    print(f"Output directory: {rundir}")
    
    try:
        result = nh3_uptake_simulation(
            str(cif_path),
            calc_charges=True,
            rundir=str(rundir),
            temperature=298,
            pressure=101325,
            rewrite_raspa_input=True,  # Enable this to fix input file issues
            equilibration_cycles=5000,  # Reduced for faster testing
            production_cycles=5000,
        )
        return result
    except Exception as e:
        print(f"GCMC error: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def main():
    base_dir = Path("/home/sibivarshan_m7/MOFDiff/MOFDiff/comparison_results")
    
    results = {}
    
    # Run GCMC for original model structure
    original_cif = base_dir / "original_model" / "relaxed" / "sample_0.cif"
    if original_cif.exists():
        print("\n" + "="*60)
        print("Running GCMC for ORIGINAL MODEL structure")
        print("="*60)
        
        # Clean up old GCMC files
        gcmc_dir = base_dir / "original_model" / "gcmc_nh3"
        if gcmc_dir.exists():
            import shutil
            shutil.rmtree(gcmc_dir)
        gcmc_dir.mkdir(parents=True, exist_ok=True)
        
        result = run_gcmc_for_structure(original_cif, gcmc_dir)
        results["original_model"] = result
        
        if 'NH3_uptake_mmol_g' in result:
            print(f"\n✓ Original Model NH3 Uptake: {result['NH3_uptake_mmol_g']:.4f} mmol/g")
        else:
            print(f"\n✗ GCMC failed: {result.get('error', 'Unknown error')}")
    else:
        print(f"Original CIF not found: {original_cif}")
    
    # Run GCMC for NH3-guided structure
    nh3_cif = base_dir / "nh3_guided" / "relaxed" / "sample_0.cif"
    if nh3_cif.exists():
        print("\n" + "="*60)
        print("Running GCMC for NH3-GUIDED structure")
        print("="*60)
        
        # Clean up old GCMC files
        gcmc_dir = base_dir / "nh3_guided" / "gcmc_nh3"
        if gcmc_dir.exists():
            import shutil
            shutil.rmtree(gcmc_dir)
        gcmc_dir.mkdir(parents=True, exist_ok=True)
        
        result = run_gcmc_for_structure(nh3_cif, gcmc_dir)
        results["nh3_guided"] = result
        
        if 'NH3_uptake_mmol_g' in result:
            print(f"\n✓ NH3-Guided Model NH3 Uptake: {result['NH3_uptake_mmol_g']:.4f} mmol/g")
        else:
            print(f"\n✗ GCMC failed: {result.get('error', 'Unknown error')}")
    else:
        print(f"NH3-guided CIF not found: {nh3_cif}")
    
    # Save results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    with open(base_dir / "gcmc_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {base_dir / 'gcmc_results.json'}")
    
    # Print comparison
    if "original_model" in results and "nh3_guided" in results:
        orig = results["original_model"]
        guided = results["nh3_guided"]
        
        if 'NH3_uptake_mmol_g' in orig and 'NH3_uptake_mmol_g' in guided:
            print("\n--- Comparison ---")
            print(f"Original Model NH3 Uptake: {orig['NH3_uptake_mmol_g']:.4f} mmol/g")
            print(f"NH3-Guided Model NH3 Uptake: {guided['NH3_uptake_mmol_g']:.4f} mmol/g")
            
            diff = guided['NH3_uptake_mmol_g'] - orig['NH3_uptake_mmol_g']
            if orig['NH3_uptake_mmol_g'] > 0:
                pct = (diff / orig['NH3_uptake_mmol_g']) * 100
                print(f"Difference: {diff:+.4f} mmol/g ({pct:+.1f}%)")


if __name__ == "__main__":
    main()
