"""
Run GCMC simulation for 2 mmol/g target.
"""
import os
import sys
import json
import subprocess
import re
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_raspa_output(output_file):
    """Parse RASPA output to extract NH3 uptake"""
    with open(output_file, 'r') as f:
        content = f.read()
    
    result = {}
    
    mol_kg_match = re.search(
        r'Average loading absolute \[mol/kg framework\]\s+([\d.]+)\s+\+/-\s+([\d.]+)',
        content
    )
    if mol_kg_match:
        result['NH3_uptake_mol_kg'] = float(mol_kg_match.group(1))
        result['NH3_uptake_mol_kg_error'] = float(mol_kg_match.group(2))
        result['NH3_uptake_mmol_g'] = result['NH3_uptake_mol_kg']
    
    mg_g_match = re.search(
        r'Average loading absolute \[milligram/gram framework\]\s+([\d.]+)\s+\+/-\s+([\d.]+)',
        content
    )
    if mg_g_match:
        result['NH3_uptake_mg_g'] = float(mg_g_match.group(1))
    
    return result


def run_gcmc_single(args):
    """Run GCMC for a single structure"""
    cif_path, output_dir, forcefield = args
    
    from openbabel import pybel
    from pymatgen.core import Structure
    
    name = Path(cif_path).stem
    workdir = Path(output_dir)
    workdir.mkdir(parents=True, exist_ok=True)
    
    raspa_path = "/home/sibivarshan_m7/gcmc_tools/raspa_install/share/raspa"
    
    try:
        # Calculate EQeq charges
        mol = next(pybel.readfile('cif', cif_path))
        mol.calccharges('eqeq')
        charges = [a.partialcharge for a in mol.atoms]
        
        # Create charged CIF for RASPA
        charged_cif = workdir / f"{name}_charged.cif"
        with open(cif_path, 'r') as f:
            lines = f.readlines()
        
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
        shutil.copy(str(charged_cif), f"{raspa_path}/structures/cif/")
        
        # Create simulation input
        simulation_input = f"""SimulationType                MonteCarlo
NumberOfCycles                2000
NumberOfInitializationCycles  0
NumberOfEquilibrationCycles   2000
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
UnitCells                     2 2 2
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
        
        # Setup RASPA environment
        env = os.environ.copy()
        env['RASPA_DIR'] = raspa_path
        env['DYLD_LIBRARY_PATH'] = "/home/sibivarshan_m7/gcmc_tools/raspa_install/lib"
        env['LD_LIBRARY_PATH'] = "/home/sibivarshan_m7/gcmc_tools/raspa_install/lib"
        
        # Run RASPA
        proc = subprocess.run(
            ["/home/sibivarshan_m7/gcmc_tools/raspa_install/bin/simulate", "simulation.input"],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            env=env,
            timeout=1200
        )
        
        if proc.returncode != 0:
            return name, {'error': f'RASPA failed: {proc.stderr[:200]}'}
        
        # Parse output
        output_path = workdir / "Output" / "System_0"
        if not output_path.exists():
            return name, {'error': 'RASPA output not found'}
        
        data_files = list(output_path.glob("*.data"))
        if not data_files:
            return name, {'error': 'No RASPA data file found'}
        
        result = parse_raspa_output(data_files[0])
        result['file'] = str(cif_path)
        result['temperature_K'] = 298
        result['pressure_Pa'] = 101325
        
        return name, result
        
    except Exception as e:
        return name, {'error': str(e)}


def main():
    input_dir = Path("/home/sibivarshan_m7/MOFDiff/MOFDiff/results/nh3_target_2mmol/relaxed")
    output_dir = Path("/home/sibivarshan_m7/MOFDiff/MOFDiff/results/nh3_target_2mmol/gcmc_parallel")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cif_files = sorted(input_dir.glob("*.cif"))
    n_files = len(cif_files)
    print(f"Found {n_files} CIF files to process")
    
    n_workers = min(6, n_files, cpu_count() - 2)
    print(f"Using {n_workers} parallel workers (available CPUs: {cpu_count()})")
    
    target_nh3 = 2.0  # Target: 2 mmol/g
    forcefield = "NH3_GCMC"
    
    args_list = [
        (str(cif), str(output_dir / cif.stem), forcefield)
        for cif in cif_files
    ]
    
    results = {}
    
    print(f"\nStarting parallel GCMC simulations...")
    print("="*60)
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_gcmc_single, args): args[0] for args in args_list}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            name, result = future.result()
            results[name] = result
            
            if 'NH3_uptake_mmol_g' in result:
                uptake = result['NH3_uptake_mmol_g']
                diff = uptake - target_nh3
                print(f"[{completed}/{n_files}] {name}: {uptake:.2f} mmol/g (diff: {diff:+.2f})")
            else:
                print(f"[{completed}/{n_files}] {name}: ERROR - {result.get('error', 'Unknown')[:50]}")
    
    # Summary
    print("\n" + "="*60)
    print(f"GCMC Results Summary (Target: {target_nh3} mmol/g)")
    print("="*60)
    
    successful = [(k, v) for k, v in results.items() if 'NH3_uptake_mmol_g' in v]
    successful.sort(key=lambda x: x[1]['NH3_uptake_mmol_g'], reverse=True)
    
    print(f"\nSuccessful: {len(successful)}/{len(results)}")
    
    if successful:
        print("\nRanking by NH3 uptake:")
        for i, (name, r) in enumerate(successful):
            uptake = r['NH3_uptake_mmol_g']
            diff = uptake - target_nh3
            print(f"  {i+1}. {name}: {uptake:.4f} mmol/g (diff: {diff:+.2f})")
        
        best_match = min(successful, key=lambda x: abs(x[1]['NH3_uptake_mmol_g'] - target_nh3))
        print(f"\n*** Best match to {target_nh3} mmol/g target ***")
        print(f"    {best_match[0]}: {best_match[1]['NH3_uptake_mmol_g']:.4f} mmol/g")
        print(f"    Difference: {abs(best_match[1]['NH3_uptake_mmol_g'] - target_nh3):.4f} mmol/g")
    
    # Save results
    results_file = output_dir / "all_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    main()
