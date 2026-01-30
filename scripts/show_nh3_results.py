"""
Summary of the NH3-guided MOF sampling pipeline.
"""
import torch
import json
from pathlib import Path
from pymatgen.core import Structure

def main():
    output_dir = Path("nh3_samples")
    
    print("="*60)
    print("NH3-GUIDED MOF SAMPLING RESULTS")
    print("="*60)
    
    # Load samples
    samples = torch.load(output_dir / "samples.pt")
    print(f"\n1. SAMPLING PHASE")
    print(f"   - Total samples generated: {len(samples['mofs'])}")
    print(f"   - Predicted NH3 uptake range: [{samples['predicted_nh3'].min():.4f}, {samples['predicted_nh3'].max():.4f}] mmol/g")
    print(f"   - Mean predicted NH3 uptake: {samples['predicted_nh3'].mean():.4f} mmol/g")
    
    # Load assembled
    assembled = torch.load(output_dir / "assembled.pt")
    successful_assembly = sum(1 for a in assembled['assembled'] if a[0] is not None)
    print(f"\n2. ASSEMBLY PHASE")
    print(f"   - Successfully assembled: {successful_assembly}/{len(assembled['mofs'])}")
    
    # Load relaxed
    with open(output_dir / "relaxed" / "relax_info.json") as f:
        relax_info = json.load(f)
    
    successful_relax = [r for r in relax_info if r['success']]
    print(f"\n3. RELAXATION PHASE")
    print(f"   - Successfully relaxed: {len(successful_relax)}/{successful_assembly}")
    
    print(f"\n4. GENERATED MOF STRUCTURES")
    print("-"*60)
    
    for i, info in enumerate(successful_relax):
        # Get predicted NH3 for this sample
        sample_idx = int(info['name'].split('_')[1])
        pred_nh3 = samples['predicted_nh3'][sample_idx]
        
        # Load structure
        struct = Structure.from_file(info['path'])
        formula = struct.composition.reduced_formula
        
        print(f"\n   Sample {sample_idx}:")
        print(f"     Formula: {formula}")
        print(f"     Atoms: {info['natoms']}")
        print(f"     Volume: {info['volume']:.1f} Å³")
        print(f"     Predicted NH3 uptake: {pred_nh3:.4f} mmol/g")
        print(f"     CIF path: {info['path']}")
    
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print(f"""
    Total samples requested:     5
    Successfully decoded:        {len(samples['mofs'])}
    Successfully assembled:      {successful_assembly}
    Successfully relaxed:        {len(successful_relax)}
    
    Output directory: {output_dir.resolve()}
    
    Files:
      - samples.pt        : Raw sampled MOF data
      - assembled.pt      : Assembled atomic structures  
      - cif/              : CIF files for assembled MOFs
      - relaxed/          : Relaxed primitive cell CIFs
    """)

if __name__ == "__main__":
    main()
