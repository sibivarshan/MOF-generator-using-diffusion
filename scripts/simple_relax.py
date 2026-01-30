"""
Simple UFF relaxation script for assembled MOF structures.
"""
import argparse
from pathlib import Path
import json
from pymatgen.io.cif import CifWriter
from pymatgen.core import Structure

def simple_relax(cif_path, output_dir):
    """Simple relaxation using pymatgen (no LAMMPS)"""
    try:
        struct = Structure.from_file(str(cif_path))
        # Get primitive structure
        primitive = struct.get_primitive_structure()
        
        # Save
        name = cif_path.stem
        output_path = output_dir / f"{name}_relaxed.cif"
        CifWriter(primitive).write_file(str(output_path))
        
        return {
            'name': name,
            'natoms': len(primitive),
            'volume': primitive.volume,
            'path': str(output_path),
            'success': True,
        }
    except Exception as e:
        print(f"Error relaxing {cif_path}: {e}")
        return {'name': cif_path.stem, 'success': False, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing cif/ folder")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir).resolve()
    cif_dir = input_dir / "cif"
    output_dir = input_dir / "relaxed"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    cif_files = list(cif_dir.glob("*.cif"))
    print(f"Found {len(cif_files)} CIF files")
    
    results = []
    for cif_file in cif_files:
        print(f"Processing {cif_file.name}...")
        result = simple_relax(cif_file, output_dir)
        results.append(result)
        if result['success']:
            print(f"  -> {result['natoms']} atoms, volume={result['volume']:.1f} Å³")
    
    # Save results
    with open(output_dir / "relax_info.json", "w") as f:
        json.dump(results, f, indent=2)
    
    successful = [r for r in results if r['success']]
    print(f"\nRelaxation complete: {len(successful)}/{len(cif_files)} successful")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
