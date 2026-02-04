"""
Run GCMC simulation for NH3 uptake on a single structure.
"""
import os
import sys
import json
import argparse
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
            rewrite_raspa_input=True,
            equilibration_cycles=5000,
            production_cycles=5000,
        )
        return result
    except Exception as e:
        print(f"GCMC error: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Run GCMC for single MOF structure')
    parser.add_argument('--input_cif', type=str, required=True, help='Path to input CIF file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for GCMC results')
    args = parser.parse_args()
    
    input_cif = Path(args.input_cif)
    output_dir = Path(args.output_dir)
    
    if not input_cif.exists():
        print(f"Error: CIF file not found: {input_cif}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("GCMC NH3 Uptake Simulation")
    print("="*60)
    print(f"Input CIF: {input_cif}")
    print(f"Output dir: {output_dir}")
    
    # Run GCMC
    result = run_gcmc_for_structure(input_cif, output_dir)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    if 'NH3_uptake_mmol_g' in result:
        print(f"\n✓ NH3 Uptake: {result['NH3_uptake_mmol_g']:.4f} mmol/g")
        print(f"  (Uncertainty: {result.get('NH3_uptake_mmol_g_error', 'N/A')})")
    else:
        print(f"\n✗ GCMC failed: {result.get('error', 'Unknown error')}")
    
    # Save result
    result_file = output_dir / "gcmc_result.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResult saved to: {result_file}")
    
    return result


if __name__ == "__main__":
    main()
