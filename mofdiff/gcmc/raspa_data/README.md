# RASPA Force Field and Molecule Definitions

This directory contains the force field parameters and molecule definitions required for GCMC simulations using RASPA2.

## Directory Structure

```
raspa_data/
├── forcefield/
│   └── NH3_GCMC/
│       ├── force_field.def          # Main force field definition
│       ├── force_field_mixing_rules.def  # LJ parameters and mixing rules
│       └── pseudo_atoms.def         # Atom type definitions with charges
└── molecules/
    └── NH3_GCMC/
        └── NH3.def                  # NH3 molecule definition
```

## Installation

Copy these files to your RASPA installation:

```bash
# Assuming RASPA is installed at $RASPA_PATH
cp -r forcefield/NH3_GCMC $RASPA_PATH/share/raspa/forcefield/
cp -r molecules/NH3_GCMC $RASPA_PATH/share/raspa/molecules/
```

## Force Field Parameters

### NH3 (Ammonia)
- Model: TraPPE-based with Rizzo/Jorgensen geometry
- N_nh3: ε = 185 K, σ = 3.42 Å, charge = -1.02 e
- H_nh3: ε = 0 K, σ = 0 Å, charge = +0.34 e

### Framework Atoms
- Uses UFF (Universal Force Field) parameters with wildcard matching
- Wildcards (e.g., `Co_`, `Zn_`) match numbered atom labels (Co1, Zn2, etc.)

## Notes

1. The force field uses Lorentz-Berthelot mixing rules
2. Tail corrections are disabled (`no`)
3. Potential is shifted (not truncated)
4. Atom matching uses underscore wildcards for numbered labels

## References

- TraPPE Force Field: http://trappe.oit.umn.edu/
- UFF: Rappé et al., JACS 1992, 114, 10024-10035
- Rizzo & Jorgensen, JACS 1999, 121, 4827-4836
