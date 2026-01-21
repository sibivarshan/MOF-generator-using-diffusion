#!/bin/bash
# ==============================================================================
# MOFDiff Command Reference
# ==============================================================================
# This file contains all commands for running different processes in MOFDiff
# Run commands from the MOFDiff directory: /home/sibivarshan_m7/MOFDiff/MOFDiff
# Activate conda environment: conda activate mofdiff-gpu
# Or use: conda run -n mofdiff-gpu <command>
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. NORMAL MOF SAMPLING (Random generation using pretrained model)
# ------------------------------------------------------------------------------

# Generate random MOF samples (outputs samples.pt file)
# --n_samples: number of MOFs to generate
# --model_path: path to pretrained MOFDiff checkpoint
# --bb_cache_path: path to building block embeddings
conda run -n mofdiff-gpu python mofdiff/scripts/sample.py \
    --model_path pretrained/mofdiff_ckpt \
    --bb_cache_path pretrained/bb_emb_space.pt \
    --n_samples 10

# Assemble the generated samples into MOF structures (creates CIF files)
# --sample_path: path to the samples.pt file from sampling
# --output_dir: directory to save assembled CIF files
conda run -n mofdiff-gpu python mofdiff/scripts/assemble.py \
    --model_path pretrained/mofdiff_ckpt \
    --sample_path samples.pt \
    --output_dir samples_normal

# Relax the assembled structures using UFF force field
# --cif_dir: directory containing CIF files to relax
# --output_dir: directory to save relaxed CIF files
conda run -n mofdiff-gpu python mofdiff/scripts/uff_relax.py \
    --cif_dir samples_normal/cif \
    --output_dir samples_normal/relaxed

# ------------------------------------------------------------------------------
# 2. NH3-TARGETED SAMPLING (Gradient-guided generation for high NH3 uptake)
# ------------------------------------------------------------------------------

# Generate MOFs optimized for target NH3 uptake using gradient optimization
# --target_nh3: target NH3 uptake in mmol/g (typical range: 5-12)
# --latent_data_path: training data for the differentiable predictor
# --predictor_path: path to save/load the NH3 predictor model
# --assemble: automatically assemble generated samples
# --relax: automatically relax assembled structures
# --n_samples: number of MOFs to generate
# --n_init: number of random initializations (more = better coverage)
# --lr: learning rate for optimization
# --steps: number of optimization steps
conda run -n mofdiff-gpu python scripts/sample_nh3_gradient.py \
    --model_path pretrained/mofdiff_ckpt \
    --bb_cache_path pretrained/bb_emb_space.pt \
    --latent_data_path raw_nh3_core/nh3_latent_dataset.pt \
    --predictor_path raw_nh3_core/nh3_predictor_mlp.pt \
    --target_nh3 8.0 \
    --n_samples 10 \
    --n_init 100 \
    --lr 0.1 \
    --steps 200 \
    --output_dir samples_nh3_targeted \
    --assemble \
    --relax

# ------------------------------------------------------------------------------
# 3. FIX CIF FILES (Fix common CIF format issues)
# ------------------------------------------------------------------------------

# Fix CIF file issues: chemical names, bond types, symmetry operators, metadata
# Processes all CIF files in nh3_all_samples/cif and nh3_all_samples/relaxed
conda run -n mofdiff-gpu python scripts/fix_cif_files.py

# ------------------------------------------------------------------------------
# 4. TRAINING NH3 PREDICTION HEAD
# ------------------------------------------------------------------------------

# Train the NH3 prediction head on latent representations
# This creates a model that predicts NH3 uptake from MOF latent vectors
conda run -n mofdiff-gpu python scripts/train_nh3_head_final.py

# Train with feature selection (experimental)
conda run -n mofdiff-gpu python scripts/train_nh3_feature_selection.py

# ------------------------------------------------------------------------------
# 5. EXPORT NH3 LATENTS (Prepare training data for NH3 head)
# ------------------------------------------------------------------------------

# Export latent representations for MOFs with known NH3 uptake
# This creates the training dataset for the NH3 prediction head
conda run -n mofdiff-gpu python scripts/export_nh3_latents_v2.py

# ------------------------------------------------------------------------------
# 6. PREDICT NH3 UPTAKE (Predict NH3 for existing MOFs)
# ------------------------------------------------------------------------------

# Predict NH3 uptake for MOFs using the trained head
conda run -n mofdiff-gpu python scripts/predict_nh3.py

# Show results summary
conda run -n mofdiff-gpu python scripts/show_nh3_results.py

# ------------------------------------------------------------------------------
# 7. GCMC SIMULATIONS FOR NH3 UPTAKE
# ------------------------------------------------------------------------------

# IMPORTANT: GCMC simulations require RASPA2 and eGULP to be installed.
# Set the following environment variables before running:
#   export RASPA_PATH=/path/to/raspa
#   export RASPA_SIM_PATH=/path/to/raspa/bin/simulate
#   export ZEO_PATH=/path/to/zeo++
#   export EGULP_PATH=/path/to/egulp
#   export EGULP_PARAMETER_PATH=/path/to/egulp/parameters

# Basic NH3 uptake simulation (single pressure point at 1 bar, 298 K)
# Calculates: NH3 uptake in mol/kg, mmol/g, mg/g and heat of adsorption
conda run -n mofdiff-gpu python mofdiff/scripts/gcmc_nh3_screen.py \
    --input nh3_all_samples/cif \
    --simulation_type uptake \
    --temperature 298 \
    --pressure 101325 \
    --ncpu 24

# NH3 working capacity (adsorption vs desorption)
# Default: Adsorption at 298K/1bar, Desorption at 373K/0.1bar
conda run -n mofdiff-gpu python mofdiff/scripts/gcmc_nh3_screen.py \
    --input nh3_all_samples/cif \
    --simulation_type working_capacity \
    --temperature 298 \
    --pressure 101325 \
    --desorption_temperature 373 \
    --desorption_pressure 10000 \
    --ncpu 24

# NH3 adsorption isotherm (multiple pressure points)
# Generates uptake data at different pressures for isotherm plotting
conda run -n mofdiff-gpu python mofdiff/scripts/gcmc_nh3_screen.py \
    --input nh3_all_samples/cif \
    --simulation_type isotherm \
    --temperature 298 \
    --pressures "100,500,1000,5000,10000,50000,101325" \
    --ncpu 24

# Run on relaxed structures (often more accurate)
conda run -n mofdiff-gpu python mofdiff/scripts/gcmc_nh3_screen.py \
    --input nh3_all_samples/relaxed \
    --simulation_type uptake \
    --temperature 298 \
    --pressure 101325 \
    --ncpu 24

# Low pressure NH3 uptake (for low concentration applications)
conda run -n mofdiff-gpu python mofdiff/scripts/gcmc_nh3_screen.py \
    --input nh3_all_samples/cif \
    --simulation_type uptake \
    --temperature 298 \
    --pressure 1000 \
    --output_name nh3_uptake_lowP_results.json \
    --ncpu 24

# Original CO2/N2 vacuum swing adsorption screening (for reference)
conda run -n mofdiff-gpu python mofdiff/scripts/gcmc_screen.py \
    --input nh3_all_samples/cif

# ------------------------------------------------------------------------------
# 8. SIMPLE RELAXATION (Standalone UFF relaxation)
# ------------------------------------------------------------------------------

# Relax a single CIF file or directory of CIF files
conda run -n mofdiff-gpu python scripts/simple_relax.py \
    --input_path path/to/cif_file_or_directory \
    --output_dir path/to/output

# ------------------------------------------------------------------------------
# 9. BATCH GENERATION EXAMPLE
# ------------------------------------------------------------------------------

# Generate samples to a specific folder with both normal and NH3-targeted

# Create output directory
mkdir -p samples001/normal samples001/nh3_targeted

# Normal samples
conda run -n mofdiff-gpu python mofdiff/scripts/sample.py \
    --model_path pretrained/mofdiff_ckpt \
    --bb_cache_path pretrained/bb_emb_space.pt \
    --n_samples 10

conda run -n mofdiff-gpu python mofdiff/scripts/assemble.py \
    --model_path pretrained/mofdiff_ckpt \
    --sample_path samples.pt \
    --output_dir samples001/normal

conda run -n mofdiff-gpu python mofdiff/scripts/uff_relax.py \
    --cif_dir samples001/normal/cif \
    --output_dir samples001/normal/relaxed

# NH3-targeted samples
conda run -n mofdiff-gpu python scripts/sample_nh3_gradient.py \
    --model_path pretrained/mofdiff_ckpt \
    --bb_cache_path pretrained/bb_emb_space.pt \
    --latent_data_path raw_nh3_core/nh3_latent_dataset.pt \
    --predictor_path raw_nh3_core/nh3_predictor_mlp.pt \
    --target_nh3 8.0 \
    --n_samples 10 \
    --output_dir samples001/nh3_targeted \
    --assemble \
    --relax

# ------------------------------------------------------------------------------
# 10. USEFUL OPTIONS
# ------------------------------------------------------------------------------

# Low memory mode (for GPUs with limited VRAM)
# Add --low_memory flag to sample.py

# Change device
# Add --device cuda or --device cpu

# Set random seed for reproducibility
# Add --seed 42

# Batch size (affects memory usage)
# Add --batch_size 4

# ------------------------------------------------------------------------------
# 11. ENVIRONMENT SETUP
# ------------------------------------------------------------------------------

# Create conda environment
# conda env create -f env.yml

# Activate environment
# conda activate mofdiff-gpu

# Install package in development mode
# pip install -e .

# ------------------------------------------------------------------------------
# 12. GCMC ENVIRONMENT SETUP (RASPA2 + Charge Equilibration)
# ------------------------------------------------------------------------------

# ============================================================================
# GCMC IS NOW SET UP AND READY TO USE!
# ============================================================================
# RASPA2 is installed at: /home/sibivarshan_m7/gcmc_tools/raspa_install
# Charge equilibration uses OpenBabel's EQeq method (no external deps needed)
#
# Environment variables are already added to ~/.bashrc:
#   RASPA_PATH=/home/sibivarshan_m7/gcmc_tools/raspa_install
#   RASPA_SIM_PATH=/home/sibivarshan_m7/gcmc_tools/raspa_install/bin/simulate
#   RASPA_DIR=/home/sibivarshan_m7/gcmc_tools/raspa_install
#
# Force field: NH3_GCMC (custom force field with UFF parameters + NH3 TraPPE)
# Molecule definitions: NH3_GCMC (NH3 with Rizzo/Jorgensen parameters)

# Verify GCMC setup:
source ~/.bashrc && conda run -n mofdiff-gpu python -c "
from mofdiff.gcmc import gcmc_wrapper
print('RASPA_PATH:', gcmc_wrapper.raspa_path)
print('RASPA_SIM_PATH:', gcmc_wrapper.raspa_sim_path)
print('RASPA_DIR:', gcmc_wrapper.raspa_dir)
print('GCMC setup OK!')
"

# ------------------------------------------------------------------------------
# 13. NH3 GCMC SIMULATIONS (Using the wrapper functions)
# ------------------------------------------------------------------------------

# Run single NH3 uptake simulation
source ~/.bashrc && conda run -n mofdiff-gpu python -c "
import os
os.environ['RASPA_PATH'] = '/home/sibivarshan_m7/gcmc_tools/raspa_install'
os.environ['RASPA_SIM_PATH'] = '/home/sibivarshan_m7/gcmc_tools/raspa_install/bin/simulate'
os.environ['RASPA_DIR'] = '/home/sibivarshan_m7/gcmc_tools/raspa_install'

from mofdiff.gcmc.gcmc_wrapper import fix_cif_for_raspa
from mofdiff.gcmc.simulation import nh3_uptake_simulation

# Fix CIF for RASPA compatibility
cif_path = 'nh3_all_samples/cif/sample_1.cif'
fixed_cif = 'gcmc_results/sample_1_fixed.cif'
import os; os.makedirs('gcmc_results', exist_ok=True)
fix_cif_for_raspa(cif_path, fixed_cif)

# Run NH3 GCMC simulation
result = nh3_uptake_simulation(
    fixed_cif,
    calc_charges=False,  # Set True for more accurate results
    rundir='./gcmc_results',
    temperature=298,  # K
    pressure=101325   # Pa (1 bar)
)
print('NH3 uptake:', result['NH3_uptake_mol_kg'], 'mol/kg')
print('NH3 uptake:', result['NH3_uptake_mg_g'], 'mg/g')
"

# Run batch NH3 GCMC screening on all samples
source ~/.bashrc && conda run -n mofdiff-gpu python mofdiff/scripts/gcmc_nh3_screen.py \
    --cif_dir nh3_all_samples/cif \
    --output_dir gcmc_results \
    --mode uptake \
    --temperature 298 \
    --pressure 101325 \
    --n_workers 4

# Run NH3 working capacity calculation (adsorption/desorption cycle)
source ~/.bashrc && conda run -n mofdiff-gpu python mofdiff/scripts/gcmc_nh3_screen.py \
    --cif_dir nh3_all_samples/cif \
    --output_dir gcmc_results \
    --mode working_capacity \
    --ads_temp 298 \
    --ads_pressure 101325 \
    --des_temp 373 \
    --des_pressure 10000 \
    --n_workers 4

# Run NH3 adsorption isotherm (multiple pressure points)
source ~/.bashrc && conda run -n mofdiff-gpu python mofdiff/scripts/gcmc_nh3_screen.py \
    --cif_dir nh3_all_samples/cif \
    --output_dir gcmc_results \
    --mode isotherm \
    --temperature 298 \
    --n_workers 4

# ============================================================================
# Manual installation instructions (already done on this system):
# ============================================================================
# RASPA2 Installation:
# cd ~/gcmc_tools
# git clone https://github.com/iRASPA/RASPA2.git
# cd RASPA2
# conda install -c conda-forge autoconf automake libtool
# aclocal && autoreconf -i
# ./configure --prefix=$HOME/gcmc_tools/raspa_install
# make -j4 && make install

# Charge equilibration options:
# 1. EQeq via OpenBabel (RECOMMENDED - already available in mofdiff-gpu env)
#    - No additional installation needed
#    - Uses: gcmc_wrapper.calculate_eqeq_charges() or charge_method="eqeq"
#
# 2. MEPO-Qeq via eGULP (optional, for higher accuracy)
#    - Not currently available, falls back to EQeq
#    - Uses: charge_method="mepo"

# ------------------------------------------------------------------------------
# END OF COMMANDS
# ------------------------------------------------------------------------------
