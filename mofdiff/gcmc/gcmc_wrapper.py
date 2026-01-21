"""
# functions for running gcmc simulations
# requires RASPA2 for all simulations (https://github.com/iRASPA/RASPA2)
# Charge equilibration options:
#   - EQeq via OpenBabel (recommended, no external dependencies)
#   - eGULP if atomic charges are to be assigned (https://github.com/danieleongari/egulp)
"""

# import libraries
import os
import subprocess
import re
import shutil
from pathlib import Path
from time import time
from math import cos, radians
from textwrap import dedent
from openbabel import pybel
from openbabel.pybel import readfile

raspa_path = os.getenv("RASPA_PATH")
raspa_sim_path = os.getenv("RASPA_SIM_PATH")
raspa_dir = os.getenv("RASPA_DIR")
zeo_path = os.getenv("ZEO_PATH")
egulp_path = os.getenv("EGULP_PATH")
egulp_parameter_path = os.getenv("EGULP_PARAMETER_PATH")


def fix_cif_for_raspa(cif_file, output_file=None):
    """
    Fix CIF file format for RASPA compatibility.
    Adds required space group information if missing.
    
    Args:
        cif_file: Path to input CIF file
        output_file: Path to output CIF file (default: overwrites input)
    
    Returns:
        Path to the fixed CIF file
    """
    if output_file is None:
        output_file = cif_file
    
    with open(cif_file, 'r') as f:
        content = f.read()
    
    # Check and add space group name if missing
    if '_symmetry_space_group_name_H-M' not in content:
        # Add after _symmetry_cell_setting or at the beginning of symmetry section
        if '_symmetry_cell_setting' in content:
            content = content.replace(
                '_symmetry_cell_setting',
                "_symmetry_space_group_name_H-M   'P 1'\n_symmetry_cell_setting"
            )
        elif '_cell_length_a' in content:
            content = content.replace(
                '_cell_length_a',
                "_symmetry_space_group_name_H-M   'P 1'\n_cell_length_a"
            )
    
    # Check and add space group number if missing
    if '_symmetry_Int_Tables_number' not in content:
        if '_symmetry_space_group_name_H-M' in content:
            content = content.replace(
                '_symmetry_space_group_name_H-M',
                "_symmetry_Int_Tables_number      1\n_symmetry_space_group_name_H-M"
            )
    
    with open(output_file, 'w') as f:
        f.write(content)
    
    return output_file


# Parent class for aggregating results.
class gcmc_simulation:
    def __init__(
        self,
        cif_file,
        sorbates=["CO2"],
        sorbates_mol_fraction=[1],
        temperature=298,
        pressure=101325,
        rundir="./temp",
    ):
        # pull unit cell dimensions from cif_file
        self.sorbent = next(readfile("cif", cif_file))
        self.dim = [0, 0, 0]
        self.angle = [0, 0, 0]
        with open(cif_file, "r") as file:
            dim_match = re.findall("_cell_length_.\s+\d+.\d+", file.read())
        with open(cif_file, "r") as file:
            angle_match = re.findall("_cell_angle_\S+\s+\d+", file.read())
        for i in range(3):
            self.dim[i] = float(re.findall("\d+.\d+", dim_match[i])[0])
            self.angle[i] = float(re.findall("\d+", angle_match[i])[0])

        # a unique string for this simulation's intermediates to use
        self.identifier = (
            ".".join(cif_file.split("/")[-1].split(".")[:-1])
            + "_"
            + str(time()).split(".")[1]
        )

        # rewrite out the adsorbent cif file to standardize
        # for later functions, use simulation.rundir instead of cwd
        self.rundir = Path(rundir)
        Path(rundir).mkdir(parents=True, exist_ok=True)
        assert self.rundir.exists(), "must provide an existing rundir."
        self.sorbent_file = str(cif_file)

        # user-defined simulation conditions
        self.sorbates = sorbates  # from user definition
        self.sorbates_mol_fraction = [
            i / sum(sorbates_mol_fraction) for i in sorbates_mol_fraction
        ]  # from user definition, normalized
        self.temperature = temperature  # from user definition, K
        self.pressure = pressure  # from user definition, Pa

        # initialize variables to be filled in by software-specific scripts
        # Zeo++ related parameters
        self.block_files = None  # a list to hold blocking sphere file names

        # RASPA related parameters
        self.helium_void_fraction = 1.0  # assumed if not calculated
        self.rosenbluth_weights = [
            1.0 for i in range(len(sorbates))
        ]  # assumed if not calculated
        self.raspa_config = None  # to hold generated input file for RASPA
        self.raspa_output = None  # to hold RASPA output

    # a convenience function that returns the kinetic diameter for common gasses
    def get_sorbate_radius(self, sorbate):
        # sorbate kinetic diameters in Angstrom
        # https://doi.org/10.1039/B802426J
        kinetic_diameter = {
            # noble gases
            "He": 2.551,
            "Ne": 2.82,
            "Ar": 3.542,
            "Kr": 3.655,
            "Xe": 4.047,
            # diatomic gases
            "H2": 2.8585,
            "D2": 2.8585,
            "N2": 3.72,
            "O2": 3.467,
            "Cl2": 4.217,
            "Br2": 4.296,
            # oxides
            "CO": 3.69,
            "CO2": 3.3,
            "NO": 3.492,
            "N2O": 3.838,
            "SO2": 4.112,
            "COS": 4.130,
            # others
            "H2O": 2.641,
            "CH4": 3.758,
            "NH3": 3.62,
            "H2S": 3.623,
        }

        # check sorbate is present and return radius
        try:
            return kinetic_diameter[sorbate] * 0.5
        except Exception as e:
            print("Unknown sorbate " + sorbate + ".")
            print(e)
            exit()

    # Determines the number of unit cells required for reasonable gcmc calculations at a given forcefield cutoff.
    def calculate_unit_cells(self, forcefield_cutoff):
        # calculate length of unit cell along each dimension
        perpendicular_length = [0, 0, 0]

        for i in range(3):
            perpendicular_length[i] = self.dim[i] * abs(
                cos(radians(self.angle[i] - 90))
            )

        # stack unit cells in each dimension until larger than twice forcefield cutoff
        unit_cells = [1, 1, 1]

        for i in range(3):
            while unit_cells[i] < 2 * forcefield_cutoff / perpendicular_length[i]:
                unit_cells[i] += 1

        return unit_cells

    def write_out(self, output_path):
        with open(output_path, "w") as log_file:
            log_file.write(self.raspa_output)

# assigns charges to the atoms in the simulation file using the MEPO Qeq charge equilibration method
def calculate_mepo_qeq_charges(simulation, egulp_parameter_set="MEPO"):
    simulation.sorbent_file = str(simulation.rundir / f"{simulation.identifier}.cif")
    simulation.sorbent.write("cif", simulation.sorbent_file)
    rundir = simulation.rundir / "charges" / simulation.identifier
    rundir.mkdir(exist_ok=True, parents=True)

    # write out egulp config file
    config = dedent(
        """
            build_grid 0
            build_grid_from_scratch 1 none 0.25 0.25 0.25 1.0 2.0 0 0.3
            save_grid 0 grid.cube
            calculate_pot_diff 0
            calculate_pot 0 repeat.cube
            skip_everything 0
            point_charges_present 0
            include_pceq 0
            imethod 0
            """.format(
            **locals()
        )
    ).strip()

    with open(rundir / "temp_config.input", "w") as file:
        file.write(config)

    # run egulp
    subprocess.run(
        [
            egulp_path,
            simulation.sorbent_file,
            os.path.join(egulp_parameter_path, egulp_parameter_set + ".param"),
            "temp_config.input",
        ],
        cwd=str(rundir),
    )

    # update sorbent file path
    simulation.sorbent_file = str(rundir / "charges.cif")


def calculate_eqeq_charges(simulation, charge_method="eqeq"):
    """
    Assigns charges to atoms using the EQeq charge equilibration method via OpenBabel.
    This is the recommended method as it doesn't require external dependencies like eGULP.
    
    Args:
        simulation: gcmc_simulation object
        charge_method: OpenBabel charge method ("eqeq", "gasteiger", "mmff94", etc.)
    
    Returns:
        None (modifies simulation object in place)
    """
    # Create output directory
    rundir = simulation.rundir / "charges" / simulation.identifier
    rundir.mkdir(exist_ok=True, parents=True)
    
    # Read the CIF file with OpenBabel
    mol = next(pybel.readfile("cif", simulation.sorbent_file))
    
    # Calculate charges using specified method
    charges = mol.calccharges(charge_method)
    
    # Read the original CIF file content
    with open(simulation.sorbent_file, 'r') as f:
        cif_content = f.read()
    
    # Check if the CIF already has charges column
    if '_atom_site_charge' not in cif_content and '_atom_type_partial_charge' not in cif_content:
        # We need to add charges to the CIF file
        # Parse the CIF and reconstruct with charges
        lines = cif_content.split('\n')
        new_lines = []
        in_atom_site = False
        atom_site_headers = []
        atom_idx = 0
        
        for line in lines:
            if line.strip().startswith('_atom_site_'):
                in_atom_site = True
                atom_site_headers.append(line.strip())
                new_lines.append(line)
            elif in_atom_site and line.strip().startswith('_'):
                atom_site_headers.append(line.strip())
                new_lines.append(line)
            elif in_atom_site and not line.strip().startswith('_') and line.strip() and not line.strip().startswith('loop_'):
                # This is an atom line - add charge
                if atom_idx == 0:
                    # Add charge column header first
                    new_lines.append('  _atom_site_charge')
                if atom_idx < len(charges):
                    new_lines.append(f"{line}  {charges[atom_idx]:.6f}")
                else:
                    new_lines.append(f"{line}  0.0")
                atom_idx += 1
            else:
                if in_atom_site and (line.strip().startswith('loop_') or line.strip() == ''):
                    in_atom_site = False
                new_lines.append(line)
        
        cif_content = '\n'.join(new_lines)
    
    # Write charged CIF file
    charged_cif_path = rundir / f"{simulation.identifier}_charged.cif"
    with open(charged_cif_path, 'w') as f:
        f.write(cif_content)
    
    # Update sorbent file path
    simulation.sorbent_file = str(charged_cif_path)
    
    return charges


def calculate_charges(simulation, method="eqeq"):
    """
    Universal charge calculation function that selects the appropriate method.
    
    Args:
        simulation: gcmc_simulation object
        method: "eqeq" (recommended), "mepo" (requires eGULP), or "gasteiger"
    
    Returns:
        None (modifies simulation object in place)
    """
    if method.lower() == "mepo":
        if egulp_path is None or not os.path.exists(egulp_path):
            print("Warning: eGULP not found, falling back to EQeq method")
            calculate_eqeq_charges(simulation, charge_method="eqeq")
        else:
            calculate_mepo_qeq_charges(simulation)
    elif method.lower() in ["eqeq", "gasteiger", "mmff94"]:
        calculate_eqeq_charges(simulation, charge_method=method.lower())
    else:
        print(f"Warning: Unknown charge method '{method}', using EQeq")
        calculate_eqeq_charges(simulation, charge_method="eqeq")


# Runs a gcmc simulation to estimate the quantity of each sorbate adsorbed to the sorbent framework under the simulation conditions.
def run_gcmc_simulation(
    simulation,
    initialization_cycles=0,
    equilibration_cycles=2000,
    production_cycles=2000,
    forcefield=None,  # Auto-detect based on sorbates
    forcefield_cutoff=12,
    molecule_definitions=None,  # Auto-detect based on sorbates
    unit_cells=[0, 0, 0],
    cleanup=False,
    rewrite_raspa_input=False,
    use_charges=False,  # Disable charges by default to avoid memory issues
):
    # Auto-detect force field and molecule definitions based on sorbates
    if "NH3" in simulation.sorbates:
        if forcefield is None:
            forcefield = "NH3_GCMC"
        if molecule_definitions is None:
            molecule_definitions = "NH3_GCMC"
    else:
        if forcefield is None:
            forcefield = "ExampleMoleculeForceField"
        if molecule_definitions is None:
            molecule_definitions = "ExampleDefinitions"
    
    # copy cif file into parent RASPA folder
    shutil.copy(simulation.sorbent_file, raspa_path + "/share/raspa/structures/cif/")
    workdir = simulation.rundir / "raspa_output" / simulation.identifier
    workdir.mkdir(exist_ok=True, parents=True)

    sorbent_file = ".".join(simulation.sorbent_file.split("/")[-1].split(".")[:-1])

    # calculate number of unit cells needed if not user defined
    if sum(unit_cells) == 0:
        unit_cells = simulation.calculate_unit_cells(forcefield_cutoff)

    # build a RASPA config file, starting with high level parameters
    # Determine charge settings based on use_charges flag
    if use_charges:
        # Use Coulomb truncated instead of Ewald to avoid memory issues
        charge_config = f"""UseChargesFromCIFFile         yes
             CutOffChargeCharge            {forcefield_cutoff}
             ChargeMethod                  Coulomb"""
    else:
        charge_config = """UseChargesFromCIFFile         no
             ChargeMethod                  None"""
    
    simulation.raspa_config = dedent(
        f"""
             SimulationType                MonteCarlo
             NumberOfCycles                {production_cycles}
             NumberOfInitializationCycles  {initialization_cycles}
             NumberOfEquilibrationCycles   {equilibration_cycles}
             PrintEvery                    1000
             
             Forcefield                    {forcefield}
             {charge_config}
             CutOffVDW                     {forcefield_cutoff}
             
             Framework                     0
             FrameworkName                 {sorbent_file}
             InputFileType                 cif
             UnitCells                     {unit_cells[0]} {unit_cells[1]} {unit_cells[2]}
             HeliumVoidFraction            {simulation.helium_void_fraction}
             ExternalTemperature           {simulation.temperature}
             ExternalPressure              {simulation.pressure}
             
             Movies                        no
             """
    ).strip()

    # for each sorbate in the simulation object, add appropriate parameters to RASPA config file
    total_sorbates = len(simulation.sorbates)

    # identity change MC moves are only defined with more than one sorbate
    if total_sorbates > 1:
        identity_change_prob = 1.0
    else:
        identity_change_prob = 0.0

    sorbate_list = " ".join(str(n) for n in range(total_sorbates))

    for i in range(total_sorbates):
        # unpack vector variables
        sorbate = simulation.sorbates[i]
        sorbate_mol_fraction = simulation.sorbates_mol_fraction[i]
        rosenbluth_weight = simulation.rosenbluth_weights[i]

        # set blocking flag, dependent on existance of Zeo++ calculated blocking spheres for each sorbate
        if simulation.block_files is None:
            block_file_line = ""
            block_flag = "no"
        else:
            block_file_line = (
                "\nBlockPocketsFileName" + " " * 10 + simulation.block_files[i] + "\n"
            )
            block_flag = "yes"

        # add linebreaks
        simulation.raspa_config += "\n\n"

        # append each sorbate
        simulation.raspa_config += dedent(
            """
            Component {i} MoleculeName            {sorbate}
                     MoleculeDefinition            {molecule_definitions}
                     MolFraction                   {sorbate_mol_fraction}
                     BlockPockets                  {block_flag}{block_file_line}
                     IdealGasRosenbluthWeight      {rosenbluth_weight}
                     IdentityChangeProbability     {identity_change_prob}
                       NumberOfIdentityChanges       {total_sorbates}
                       IdentityChangeList            {sorbate_list}
                     TranslationProbability        0.5
                     RotationProbability           0.5
                     ReinsertionProbability        0.5
                     SwapProbability               1.0
                     CreateNumberOfMolecules       0
         """.format(
                **locals()
            )
        ).strip()

    # write out raspa input file
    with open(workdir / "simulation.input", "w") as raspa_input:
        raspa_input.write(simulation.raspa_config)
    
    # optionally rewrite input file to avoid errors with RASPA reading it
    if rewrite_raspa_input:
        raspa_input_path = workdir / "simulation.input"
        subprocess.run(["mv", raspa_input_path, f"{raspa_input_path}.orig"])
        command = f"printf '%s\\n' \"$(cat {raspa_input_path}.orig)\" > {raspa_input_path}"
        subprocess.Popen(command, shell = True)

    # Set environment for RASPA (ensure RASPA_DIR is set correctly)
    env = os.environ.copy()
    if raspa_dir:
        env['RASPA_DIR'] = raspa_dir
    elif raspa_path:
        env['RASPA_DIR'] = raspa_path
    
    # run raspa simulation
    subprocess.run([raspa_sim_path, "simulation.input"], cwd=str(workdir), env=env)

    # collect raspa output file
    file_list = os.listdir(str(workdir / "Output" / "System_0"))
    raspa_log = [item for item in file_list if re.match(r".*\.data", item)][0]

    with open(str(workdir / "Output" / "System_0" / raspa_log), "r") as log:
        simulation.raspa_output = log.read()

    # clear temp directory
    if cleanup:
        shutil.rmtree(str(workdir))
