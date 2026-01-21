"""
GCMC Screening Script for NH3 Uptake Simulations

This script runs GCMC (Grand Canonical Monte Carlo) simulations using RASPA2
to calculate NH3 adsorption properties of MOF structures.

Requirements:
- RASPA2 installed and configured (https://github.com/iRASPA/RASPA2)
- eGULP for charge calculations (https://github.com/danieleongari/egulp)
- Environment variables set:
    - RASPA_PATH: Path to RASPA installation
    - RASPA_SIM_PATH: Path to RASPA simulate executable
    - ZEO_PATH: Path to Zeo++ (optional)
    - EGULP_PATH: Path to eGULP executable
    - EGULP_PARAMETER_PATH: Path to eGULP parameter files

Usage:
    python mofdiff/scripts/gcmc_nh3_screen.py --input <cif_directory> [options]
"""

from pathlib import Path
import json
import argparse
import torch

from mofdiff.common.atomic_utils import graph_from_cif
from mofdiff.common.data_utils import lattice_params_to_matrix
from mofdiff.common.atomic_utils import frac2cart, compute_distance_matrix
from mofdiff.gcmc.simulation import (
    nh3_uptake_simulation,
    nh3_working_capacity,
    nh3_isotherm_simulation,
)
from p_tqdm import p_umap


def compute_nh3_gcmc(ciffile, rundir, calc_charges=True, max_natom=10000,
                      simulation_type='uptake', rewrite_raspa_input=False,
                      temperature=298, pressure=101325,
                      desorption_temperature=373, desorption_pressure=10000,
                      pressures=None):
    """
    Compute NH3 GCMC simulation for a single CIF file.
    
    Args:
        ciffile: Path to CIF file
        rundir: Directory for intermediate files
        calc_charges: Whether to calculate charges
        max_natom: Maximum number of atoms (skip larger structures)
        simulation_type: 'uptake', 'working_capacity', or 'isotherm'
        rewrite_raspa_input: Whether to rewrite RASPA input files
        temperature: Simulation temperature in K
        pressure: Simulation pressure in Pa
        desorption_temperature: Desorption temperature for working capacity
        desorption_pressure: Desorption pressure for working capacity
        pressures: List of pressures for isotherm
    
    Returns:
        Dictionary with simulation results
    """
    uid = ciffile.parts[-1].split('.')[0]
    
    try:
        struct = graph_from_cif(ciffile).structure.get_primitive_structure()
        
        if struct.frac_coords.shape[0] > max_natom:
            return dict(
                uid=uid,
                info=f'too large: {struct.frac_coords.shape[0]} atoms',
                adsorption_info=None
            )
        
        # Check for atomic overlap
        frac_coords = torch.tensor(struct.frac_coords).float()
        cell = torch.from_numpy(lattice_params_to_matrix(*struct.lattice.parameters)).float()
        cart_coords = frac2cart(frac_coords, cell)
        dist_mat = compute_distance_matrix(cell, cart_coords).fill_diagonal_(5.)
        
        if dist_mat.min() < 0.5:
            return dict(
                uid=uid,
                info='atomic overlap detected',
                adsorption_info=None
            )
        
        # Run appropriate simulation type
        if simulation_type == 'uptake':
            adsorption_info = nh3_uptake_simulation(
                str(ciffile),
                calc_charges=calc_charges,
                rundir=rundir,
                rewrite_raspa_input=rewrite_raspa_input,
                temperature=temperature,
                pressure=pressure,
            )
        elif simulation_type == 'working_capacity':
            adsorption_info = nh3_working_capacity(
                str(ciffile),
                calc_charges=calc_charges,
                rundir=rundir,
                rewrite_raspa_input=rewrite_raspa_input,
                adsorption_temperature=temperature,
                adsorption_pressure=pressure,
                desorption_temperature=desorption_temperature,
                desorption_pressure=desorption_pressure,
            )
        elif simulation_type == 'isotherm':
            adsorption_info = nh3_isotherm_simulation(
                str(ciffile),
                calc_charges=calc_charges,
                rundir=rundir,
                rewrite_raspa_input=rewrite_raspa_input,
                temperature=temperature,
                pressures=pressures,
            )
        else:
            raise ValueError(f"Unknown simulation type: {simulation_type}")
        
        return dict(uid=uid, info='success', adsorption_info=adsorption_info)
        
    except Exception as e:
        print(f'Error in {ciffile}: {e}')
        return dict(uid=uid, info=str(e), adsorption_info=None)


def main(input_dir, ncpu=24, rewrite_raspa_input=False, simulation_type='uptake',
         temperature=298, pressure=101325, desorption_temperature=373,
         desorption_pressure=10000, pressures=None, output_name=None):
    """
    Main function to run NH3 GCMC screening.
    
    Args:
        input_dir: Directory containing CIF files or path to valid_mof_paths.json
        ncpu: Number of CPUs for parallel processing
        rewrite_raspa_input: Whether to rewrite RASPA input files
        simulation_type: 'uptake', 'working_capacity', or 'isotherm'
        temperature: Simulation temperature in K
        pressure: Simulation pressure in Pa
        desorption_temperature: Desorption temperature for working capacity
        desorption_pressure: Desorption pressure for working capacity
        pressures: List of pressures for isotherm (comma-separated string)
        output_name: Custom output filename
    """
    input_path = Path(input_dir)
    rundir = input_path / 'gcmc_nh3'
    rundir.mkdir(exist_ok=True)
    
    # Parse pressures if provided as string
    if pressures is not None and isinstance(pressures, str):
        pressures = [float(p.strip()) for p in pressures.split(',')]
    
    # Determine if charges need to be calculated
    if input_path.parts[-1].startswith('mepo'):
        all_files = list(input_path.glob('*.cif'))
        calc_charges = False
    elif (input_path / 'valid_mof_paths.json').exists():
        with open(input_path / 'valid_mof_paths.json', "r") as f:
            all_files = [Path(x) for x in json.load(f)]
        calc_charges = True
    else:
        # Just use all CIF files in the directory
        all_files = list(input_path.glob('*.cif'))
        # Also check subdirectories
        all_files.extend(list(input_path.glob('**/*.cif')))
        calc_charges = True
    
    # Skip data entries (from dataset)
    all_files = [x for x in all_files if 'data' not in x.parts[-1]]
    
    if len(all_files) == 0:
        print(f"No CIF files found in {input_dir}")
        return
    
    print(f"Found {len(all_files)} CIF files to process")
    print(f"Simulation type: {simulation_type}")
    print(f"Temperature: {temperature} K")
    print(f"Pressure: {pressure} Pa")
    if simulation_type == 'working_capacity':
        print(f"Desorption temperature: {desorption_temperature} K")
        print(f"Desorption pressure: {desorption_pressure} Pa")
    if simulation_type == 'isotherm' and pressures:
        print(f"Isotherm pressures: {pressures}")
    
    # Create wrapper function with fixed parameters
    def compute_wrapper(ciffile):
        return compute_nh3_gcmc(
            ciffile,
            rundir=rundir,
            calc_charges=calc_charges,
            simulation_type=simulation_type,
            rewrite_raspa_input=rewrite_raspa_input,
            temperature=temperature,
            pressure=pressure,
            desorption_temperature=desorption_temperature,
            desorption_pressure=desorption_pressure,
            pressures=pressures,
        )
    
    # Run simulations in parallel
    results = p_umap(compute_wrapper, all_files, num_cpus=ncpu)
    
    # Save results
    if output_name is None:
        output_name = f'nh3_{simulation_type}_results.json'
    
    with open(rundir / output_name, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    successful = [r for r in results if r['adsorption_info'] is not None]
    failed = [r for r in results if r['adsorption_info'] is None]
    
    print(f"\n{'='*60}")
    print(f"GCMC NH3 Screening Complete")
    print(f"{'='*60}")
    print(f"Total structures: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Results saved to: {rundir / output_name}")
    
    if successful and simulation_type == 'uptake':
        uptakes = [r['adsorption_info']['NH3_uptake_mmol_g'] for r in successful]
        print(f"\nNH3 Uptake Statistics (mmol/g):")
        print(f"  Min: {min(uptakes):.3f}")
        print(f"  Max: {max(uptakes):.3f}")
        print(f"  Mean: {sum(uptakes)/len(uptakes):.3f}")
        
        # Top 5 performers
        sorted_results = sorted(successful, 
                                key=lambda x: x['adsorption_info']['NH3_uptake_mmol_g'],
                                reverse=True)
        print(f"\nTop 5 NH3 Uptake MOFs:")
        for i, r in enumerate(sorted_results[:5]):
            print(f"  {i+1}. {r['uid']}: {r['adsorption_info']['NH3_uptake_mmol_g']:.3f} mmol/g")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run GCMC simulations for NH3 uptake in MOF structures'
    )
    parser.add_argument(
        '--input', type=str, required=True,
        help='Directory containing CIF files to screen'
    )
    parser.add_argument(
        '--ncpu', type=int, default=24,
        help='Number of CPUs for parallel processing (default: 24)'
    )
    parser.add_argument(
        '--simulation_type', type=str, default='uptake',
        choices=['uptake', 'working_capacity', 'isotherm'],
        help='Type of GCMC simulation to run (default: uptake)'
    )
    parser.add_argument(
        '--temperature', type=float, default=298,
        help='Simulation temperature in K (default: 298)'
    )
    parser.add_argument(
        '--pressure', type=float, default=101325,
        help='Simulation pressure in Pa (default: 101325 = 1 bar)'
    )
    parser.add_argument(
        '--desorption_temperature', type=float, default=373,
        help='Desorption temperature in K for working capacity (default: 373)'
    )
    parser.add_argument(
        '--desorption_pressure', type=float, default=10000,
        help='Desorption pressure in Pa for working capacity (default: 10000 = 0.1 bar)'
    )
    parser.add_argument(
        '--pressures', type=str, default=None,
        help='Comma-separated list of pressures for isotherm (default: auto)'
    )
    parser.add_argument(
        '--output_name', type=str, default=None,
        help='Custom output filename (default: nh3_<simulation_type>_results.json)'
    )
    parser.add_argument(
        '--rewrite_raspa_input', action='store_true',
        help='Rewrite RASPA input files to avoid reading errors'
    )
    parser.set_defaults(rewrite_raspa_input=False)
    
    args = parser.parse_args()
    
    main(
        args.input,
        ncpu=args.ncpu,
        rewrite_raspa_input=args.rewrite_raspa_input,
        simulation_type=args.simulation_type,
        temperature=args.temperature,
        pressure=args.pressure,
        desorption_temperature=args.desorption_temperature,
        desorption_pressure=args.desorption_pressure,
        pressures=args.pressures,
        output_name=args.output_name,
    )
