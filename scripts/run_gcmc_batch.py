"""
Run GCMC simulation for NH3 uptake on all relaxed structures in a directory.
"""
import os
import sys
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from mofdiff.gcmc.simulation import nh3_uptake_simulation


def run_gcmc_for_structure(cif_path, rundir):
    """Run GCMC simulation for NH3 uptake"""
    try:
        result = nh3_uptake_simulation(
            str(cif_path),
            calc_charges=True,
            rundir=str(rundir),
            temperature=298,
            pressure=101325,
            rewrite_raspa_input=True,
            equilibration_cycles=5000,
            production_cycles=5000,
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def main():
    input_dir = Path("/home/sibivarshan_m7/MOFDiff/MOFDiff/results/nh3_target_10mmol/relaxed")
    output_dir = Path("/home/sibivarshan_m7/MOFDiff/MOFDiff/results/nh3_target_10mmol/gcmc")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all CIF files
    cif_files = sorted(input_dir.glob("*.cif"))
    print(f"Found {len(cif_files)} CIF files to process")
    
    results = {}
    
    for i, cif_path in enumerate(cif_files):
        name = cif_path.stem
        print(f"\n[{i+1}/{len(cif_files)}] Processing {name}...")
        
        gcmc_dir = output_dir / name
        gcmc_dir.mkdir(parents=True, exist_ok=True)
        
        result = run_gcmc_for_structure(cif_path, gcmc_dir)
        results[name] = result
        
        if 'NH3_uptake_mmol_g' in result:
            print(f"  ✓ NH3 Uptake: {result['NH3_uptake_mmol_g']:.4f} mmol/g")
        else:
            print(f"  ✗ Error: {result.get('error', 'Unknown')}")
    
    # Summary
    print("\n" + "="*60)
    print("GCMC Results Summary")
    print("="*60)
    
    successful = [(k, v) for k, v in results.items() if 'NH3_uptake_mmol_g' in v]
    successful.sort(key=lambda x: x[1]['NH3_uptake_mmol_g'], reverse=True)
    
    print(f"\nSuccessful: {len(successful)}/{len(results)}")
    print("\nRanking by NH3 uptake:")
    for i, (name, r) in enumerate(successful):
        uptake = r['NH3_uptake_mmol_g']
        print(f"  {i+1}. {name}: {uptake:.4f} mmol/g")
    
    # Find best match to 10 mmol/g target
    target = 10.0
    if successful:
        best_match = min(successful, key=lambda x: abs(x[1]['NH3_uptake_mmol_g'] - target))
        print(f"\nBest match to {target} mmol/g target:")
        print(f"  {best_match[0]}: {best_match[1]['NH3_uptake_mmol_g']:.4f} mmol/g")
        print(f"  Difference: {abs(best_match[1]['NH3_uptake_mmol_g'] - target):.4f} mmol/g")
    
    # Save all results
    results_file = output_dir / "all_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    main()
