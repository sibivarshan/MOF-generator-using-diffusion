#!/bin/bash
# ==============================================================================
# GCMC Environment Setup Script
# ==============================================================================
# Source this file to set up GCMC environment variables for RASPA2
# Usage: source setup_gcmc_env.sh
# ==============================================================================

# RASPA2 paths
export RASPA_PATH="/home/sibivarshan_m7/gcmc_tools/raspa_install"
export RASPA_SIM_PATH="/home/sibivarshan_m7/gcmc_tools/raspa_install/bin/simulate"

# Optional: Zeo++ path (uncomment if installed)
# export ZEO_PATH="/path/to/zeo++"

# Optional: eGULP paths (not required - using OpenBabel EQeq instead)
# export EGULP_PATH="/path/to/egulp"
# export EGULP_PARAMETER_PATH="/path/to/egulp/parameters"

# Add RASPA to PATH
export PATH="${RASPA_PATH}/bin:${PATH}"

# Set RASPA_DIR for some configurations
export RASPA_DIR="${RASPA_PATH}/share/raspa"

# Verify setup
echo "GCMC Environment Variables Set:"
echo "  RASPA_PATH: ${RASPA_PATH}"
echo "  RASPA_SIM_PATH: ${RASPA_SIM_PATH}"
echo "  RASPA_DIR: ${RASPA_DIR}"

# Test RASPA installation
if [ -x "${RASPA_SIM_PATH}" ]; then
    echo "  RASPA simulate: OK"
else
    echo "  WARNING: RASPA simulate not found at ${RASPA_SIM_PATH}"
fi

# Check if RASPA share directory exists
if [ -d "${RASPA_DIR}" ]; then
    echo "  RASPA data files: OK"
else
    echo "  WARNING: RASPA data directory not found at ${RASPA_DIR}"
fi

echo ""
echo "To use GCMC simulations, make sure to activate the mofdiff environment:"
echo "  conda activate mofdiff-gpu"
