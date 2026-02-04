"""
Run GCMC simulation batch using shell command for RASPA.
"""
import os
import sys
import json
import subprocess
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mofdiff.gcmc.gcmc_wrapper import gcmc_simulation


def parse_raspa_output(output_file):
    """Parse RASPA output to extract NH3 uptake"""
    with open(output_file, 'r') as f:
        content = f.read()
    
    result = {}
    
    # Extract absolute loading in mol/kg
    mol_kg_match = re.search(
        r'Average loading absolute \[mol/kg framework\]\s+([\d.]+)\s+\+/-\s+([\d.]+)',
        content
    )
    if mol_kg_match:
        result['NH3_uptake_mol_kg'] = float(mol_kg_match.group(1))
        result['NH3_uptake_mol_kg_error'] = float(mol_kg_match.group(2))
        result['NH3_uptake_mmol_g'] = result['NH3_uptake_mol_kg']
    
    # Extract mg/g
    mg_g_match = re.search(
        r'Average loading absolute \[milligram/gram framework\]\s+([\d.]+)\s+\+/-\s+([\d.]+)',
        content
    )
    if mg_g_match:
        result['NH3_uptake_mg_g'] = float(mg_g_match.group(1))
    
    return result


def run_gcmc_shell(cif_path, output_dir, forcefield="NH3_GCMC"):
    """Run GCMC using shell command"""
    import shutil
    from openbabel import pybel
    
    cif_path = Path(cif_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    identifier = cif_path.stem + "_" + str(hash(str(cif_path)))[-6:]
    workdir = output_dir / identifier
    workdir.mkdir(parents=True, exist_ok=True)
    
    # Copy CIF to RASPA structures
    raspa_path = os.getenv("RASPA_PATH")
    raspa_sim = os.getenv("RASPA_SIM_PATH")
    
    # Calculate charges with OpenBabel EQeq
    charges_dir = output_dir / "charges"
    charges_dir.mkdir(parents=True, exist_ok=True)
    charged_cif = charges_dir / f"{cif_path.stem}_charged.cif"
    
    # Read CIF and calculate EQeq charges
    mol = next(pybel.readfile("cif", str(cif_path)))
    charges = mol.calccharges("eqeq")
    
    # Read original CIF and add charges
    with open(cif_path, 'r') as f:
        lines = f.readlines()
    
    # Simple approach: write a new CIF with charges
    new_lines = []
    in_atom_block = False
    atom_idx = 0
    added_charge_header = False
    
    for line in lines:
        stripped = line.strip()
        if '_atom_site_' in stripped:
            new_lines.append(line)
            in_atom_block = True
        elif in_atom_block and stripped and not stripped.startswith('_') and not stripped.startswith('loop_'):
            if not added_charge_header:
                new_lines.append('  _atom_site_charge\n')
                added_charge_header = True
            # Add charge to atom line
            charge = charges[atom_idx] if atom_idx < len(charges) else 0.0
            new_lines.append(line.rstrip() + f'  {charge:.6f}\n')
            atom_idx += 1
        else:
            new_lines.append(line)
            if stripped == '' or stripped.startswith('loop_'):
                in_atom_block = False
    
    with open(charged_cif, 'w') as f:
        f.writelines(new_lines)
    
    # Copy charged CIF to RASPA
    shutil.copy(str(charged_cif), f"{raspa_path}/share/raspa/structures/cif/")
    
    # Create simulation input
    simulation_input = f"""SimulationType                MonteCarlo
NumberOfCycles                5000
NumberOfInitializationCycles  0
NumberOfEquilibrationCycles   5000
PrintEvery                    1000

Forcefield                    {forcefield}
UseChargesFromCIFFile         yes
CutOffChargeCharge            12
ChargeMethod                  Ewald
EwaldPrecision                1e-6
CutOffVDW                     12

Framework                     0
FrameworkName                 {charged_cif.stem}
InputFileType                 cif
UnitCells                     3 2 2
HeliumVoidFraction            1.0
ExternalTemperature           298
ExternalPressure              101325

Movies                        no

Component 0 MoleculeName            NH3
         MoleculeDefinition            {forcefield}
         MolFraction                   1.0
         BlockPockets                  no
         IdealGasRosenbluthWeight      1.0
         IdentityChangeProbability     0.0
           NumberOfIdentityChanges       1
           IdentityChangeList            0
         TranslationProbability        0.5
         RotationProbability           0.5
         ReinsertionProbability        0.5
         SwapProbability               1.0
         CreateNumberOfMolecules       0
"""
    
    with open(workdir / "simulation.input", "w") as f:
        f.write(simulation_input)
    
    # Run RASPA
    print(f"Running RASPA for {cif_path.name}...")
    env = os.environ.copy()
    env['RASPA_DIR'] = raspa_path
    
    result = subprocess.run(
        [raspa_sim, "simulation.input"],
        cwd=str(workdir),
        env=env,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"RASPA error: {result.stderr}")
        return {'error': result.stderr}
    
    # Parse output
    output_dir_path = workdir / "Output" / "System_0"
    if not output_dir_path.exists():
        return {'error': 'RASPA output not found'}
    
    data_files = list(output_dir_path.glob("*.data"))
    if not data_files:
        return {'error': 'No RASPA data file found'}
    
    result = parse_raspa_output(data_files[0])
    result['file'] = str(cif_path)
    result['temperature_K'] = 298
    result['pressure_Pa'] = 101325
    
    return result


def main():
    input_dir = Path("/home/sibivarshan_m7/MOFDiff/MOFDiff/results/nh3_target_10mmol/relaxed")
    output_dir = Path("/home/sibivarshan_m7/MOFDiff/MOFDiff/results/nh3_target_10mmol/gcmc_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cif_files = sorted(input_dir.glob("*.cif"))
    print(f"Found {len(cif_files)} CIF files to process")
    
    results = {}
    target_nh3 = 10.0
    
    for i, cif_path in enumerate(cif_files):
        name = cif_path.stem
        print(f"\n[{i+1}/{len(cif_files)}] Processing {name}...")
        
        gcmc_dir = output_dir / name
        result = run_gcmc_shell(cif_path, gcmc_dir)
        results[name] = result
        
        if 'NH3_uptake_mmol_g' in result:
            uptake = result['NH3_uptake_mmol_g']
            diff = abs(uptake - target_nh3)
            print(f"  ✓ NH3 Uptake: {uptake:.4f} mmol/g (diff from {target_nh3}: {diff:.2f})")
        else:
            print(f"  ✗ Error: {result.get('error', 'Unknown')}")
    
    # Summary
    print("\n" + "="*60)
    print("GCMC Results Summary (Target: 10 mmol/g)")
    print("="*60)
    
    successful = [(k, v) for k, v in results.items() if 'NH3_uptake_mmol_g' in v]
    successful.sort(key=lambda x: x[1]['NH3_uptake_mmol_g'], reverse=True)
    
    print(f"\nSuccessful: {len(successful)}/{len(results)}")
    print("\nRanking by NH3 uptake:")
    for i, (name, r) in enumerate(successful):
        uptake = r['NH3_uptake_mmol_g']
        diff = uptake - target_nh3
        print(f"  {i+1}. {name}: {uptake:.4f} mmol/g (diff: {diff:+.2f})")
    
    if successful:
        best_match = min(successful, key=lambda x: abs(x[1]['NH3_uptake_mmol_g'] - target_nh3))
        print(f"\nBest match to {target_nh3} mmol/g target:")
        print(f"  {best_match[0]}: {best_match[1]['NH3_uptake_mmol_g']:.4f} mmol/g")
        print(f"  Difference: {abs(best_match[1]['NH3_uptake_mmol_g'] - target_nh3):.4f} mmol/g")
    
    # Save results
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_dir / 'all_results.json'}")
    
    return results


if __name__ == "__main__":
    main()
