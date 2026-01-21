# RASPA Force Field and Molecule Definitions

This directory contains the force field parameters and molecule definitions required for GCMC simulations using RASPA2.

## Directory Structure

```
raspa_data/
├── forcefield/
│   └── NH3_GCMC/
│       ├── force_field.def          # Main force field definition
│       ├── force_field_mixing_rules.def  # LJ parameters and mixing rules
│       └── pseudo_atoms.def         # Atom type definitions with masses
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
- Uses UFF (Universal Force Field) parameters
- **Plain element names** (Ce, Zn, Cu, etc.) match standard CIF labels (Ce1, Zn2, etc.)
- Underscore wildcards (Ce_, Zn_) provided as fallback for alternative formats
- 30 framework atom types defined including lanthanides (Ce, La)

## Notes

1. The force field uses Lorentz-Berthelot mixing rules
2. **Tail corrections are ENABLED** (`yes`) for accurate uptake values
3. Potential is shifted (not truncated)
4. CIF charges are read from files (set `UseChargesFromCIFFile yes` in simulation)

## References

- TraPPE Force Field: http://trappe.oit.umn.edu/
- UFF: Rappé et al., JACS 1992, 114, 10024-10035
- Rizzo & Jorgensen, JACS 1999, 121, 4827-4836
