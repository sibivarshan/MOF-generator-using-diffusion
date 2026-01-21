# GCMC (Grand Canonical Monte Carlo) Simulation Setup for MOFDiff

This document describes how to set up and run GCMC simulations for gas adsorption calculations in MOF structures.

## Overview

GCMC simulations allow prediction of gas adsorption properties (uptake, working capacity, isotherms) for MOF structures. This is implemented using RASPA2, a molecular simulation software for adsorption.

## Prerequisites

### Software Requirements
- **RASPA2**: Molecular simulation software for adsorption calculations
- **OpenBabel** (pybel): For charge equilibration calculations
- **Python 3.8+** with the mofdiff environment

### Hardware Requirements
- Recommended: At least 8GB RAM for small systems
- Large systems with charges enabled may require 16GB+ RAM

## Installation

### 1. RASPA2 Installation

RASPA2 must be compiled from source:

```bash
# Create installation directory
mkdir -p ~/gcmc_tools
cd ~/gcmc_tools

# Clone RASPA2 repository
git clone https://github.com/iRASPA/RASPA2.git
cd RASPA2

# Configure and build
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/gcmc_tools/raspa_install
make -j4
make install
```

### 2. Environment Configuration

Add the following to your `.env` file or shell profile:

```bash
# RASPA2 Environment Variables
export RASPA_PATH=$HOME/gcmc_tools/raspa_install
export RASPA_SIM_PATH=$HOME/gcmc_tools/raspa_install/bin/simulate
export RASPA_DIR=$HOME/gcmc_tools/raspa_install

# Optional: Zeo++ for pore analysis
export ZEO_PATH=/path/to/zeo++/bin

# Optional: eGULP for MEPO charge equilibration
export EGULP_PATH=/path/to/egulp
export EGULP_PARAMETER_PATH=/path/to/egulp/parameters
```

### 3. Force Field Setup for NH3

For NH3 simulations, a custom force field (`NH3_GCMC`) has been created. The force field files are located at:
```
$RASPA_PATH/share/raspa/forcefield/NH3_GCMC/
$RASPA_PATH/share/raspa/molecules/NH3_GCMC/
```

#### Force Field Files Structure

**force_field_mixing_rules.def** - Lennard-Jones parameters:
```
# NH3 parameters (TraPPE model)
N_nh3   lennard-jones  185.0  3.42   # epsilon=185K, sigma=3.42Å
H_nh3   none
# Framework parameters use UFF defaults
```

**pseudo_atoms.def** - Atom definitions:
```
N_nh3   N    14.0067   -1.02  0  0  absolute  0
H_nh3   H    1.00794   0.34   0  0  absolute  0
```

**NH3.def** (molecule definition):
```
# Number of atoms
4
# Bonds, angles, etc.
# Atom definitions with charges
```

## Usage

### Basic NH3 Uptake Simulation

```python
from mofdiff.gcmc.simulation import nh3_uptake_simulation

# Run simulation at 298K, 1 bar
result = nh3_uptake_simulation(
    cif_file="path/to/mof.cif",
    calc_charges=True,      # Calculate EQeq charges
    temperature=298,        # K
    pressure=101325,        # Pa (1 bar)
    rundir="./gcmc_output"
)

print(f"NH3 uptake: {result['NH3_uptake_mmol_g']:.3f} mmol/g")
```

### NH3 Working Capacity

```python
from mofdiff.gcmc.simulation import nh3_working_capacity

# Calculate working capacity between adsorption and desorption conditions
result = nh3_working_capacity(
    cif_file="path/to/mof.cif",
    adsorption_temperature=298,    # K
    adsorption_pressure=101325,    # Pa (1 bar)
    desorption_temperature=373,    # K (100°C)
    desorption_pressure=10000,     # Pa (0.1 bar)
)

print(f"Working capacity: {result['NH3_working_capacity_mmol_g']:.3f} mmol/g")
```

### NH3 Adsorption Isotherm

```python
from mofdiff.gcmc.simulation import nh3_isotherm_simulation

# Generate isotherm at multiple pressures
result = nh3_isotherm_simulation(
    cif_file="path/to/mof.cif",
    temperature=298,
    pressures=[100, 1000, 10000, 50000, 101325],  # Pa
)

for point in result['isotherm']:
    print(f"P={point['pressure_Pa']} Pa: {point['NH3_uptake_mmol_g']:.3f} mmol/g")
```

### CO2/N2 Vacuum Swing Adsorption (Original Function)

```python
from mofdiff.gcmc.simulation import working_capacity_vacuum_swing

result = working_capacity_vacuum_swing(
    cif_file="path/to/mof.cif",
    calc_charges=True,
)

print(f"CO2 working capacity: {result['working_capacity_vacuum_swing']:.3f} mol/kg")
```

## Charge Equilibration Methods

The wrapper supports multiple charge calculation methods:

1. **EQeq** (Recommended): OpenBabel's charge equilibration
   - No external dependencies
   - Fast and reliable
   - Default method

2. **Gasteiger**: Classic Gasteiger-Marsili charges
   - Available via OpenBabel
   - Good for organic molecules

3. **MEPO**: MEPO-QEq via eGULP
   - Requires eGULP installation
   - Better for periodic systems

```python
from mofdiff.gcmc.gcmc_wrapper import calculate_charges, gcmc_simulation

sim = gcmc_simulation(cif_file, sorbates=["NH3"], ...)
calculate_charges(sim, method="eqeq")  # or "gasteiger", "mepo"
```

## CIF File Requirements

RASPA requires specific CIF file formatting. The wrapper includes `fix_cif_for_raspa()` to handle common issues:

```python
from mofdiff.gcmc.gcmc_wrapper import fix_cif_for_raspa

# Fix CIF file formatting
fix_cif_for_raspa("input.cif", "output.cif")
```

The function adds:
- `_symmetry_space_group_name_H-M` if missing
- `_symmetry_Int_Tables_number` if missing

## Batch Screening

For screening multiple MOF structures:

```python
import os
from mofdiff.gcmc.simulation import nh3_uptake_simulation

cif_dir = "path/to/cifs/"
results = []

for cif_file in os.listdir(cif_dir):
    if cif_file.endswith(".cif"):
        try:
            result = nh3_uptake_simulation(
                os.path.join(cif_dir, cif_file),
                calc_charges=False,  # Faster without charges
                rundir="./gcmc_screening"
            )
            results.append(result)
        except Exception as e:
            print(f"Error with {cif_file}: {e}")
```

## Troubleshooting

### Memory Issues
If RASPA uses excessive memory (>16GB):
1. Ensure `RASPA_DIR` environment variable points to the correct installation
2. Disable charges: `use_charges=False` in `run_gcmc_simulation()`
3. Reduce system size (smaller MOF supercell)

### CIF File Errors
If RASPA can't read CIF files:
1. Use `fix_cif_for_raspa()` to add required fields
2. Ensure atom labels are standard (e.g., `Co1`, `C1`, `H1`)
3. Check for space group information

### Zero or Very Low Adsorption
Possible causes:
1. Force field parameters not matching atom types in CIF
2. Pore blocking (pores too small for adsorbate)
3. Very weak interactions (check force field parameters)

### Force Field Atom Type Matching
RASPA matches atoms by label. If your CIF uses labels like `Co1`, `H1`, the force field must include:
```
Co1   lennard-jones  ...
H1    lennard-jones  ...
```
Or use wildcard patterns (RASPA2 feature):
```
Co_   lennard-jones  ...  # Matches Co1, Co2, etc.
```

## Known Limitations

1. **Atom Type Matching**: RASPA requires exact match between CIF atom labels and force field definitions. CIF files with numbered atoms (e.g., `Co1`, `H1`) need corresponding force field entries.

2. **Force Field Calibration**: The NH3_GCMC force field uses generic UFF parameters for framework atoms. For accurate predictions matching experimental or literature values, force field parameters need calibration for specific MOF chemistry.

3. **Charge Calculation**: EQeq charges are approximate. For best accuracy, use charges from DFT calculations (e.g., DDEC6 charges).

## References

- RASPA2: https://github.com/iRASPA/RASPA2
- Dubbeldam et al., "RASPA: molecular simulation software for adsorption and diffusion in flexible nanoporous materials"
- Boyd et al., "Data-driven design of metal-organic frameworks for wet flue gas CO2 capture"
- TraPPE-UA Force Field: http://trappe.oit.umn.edu/
