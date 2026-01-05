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
# 7. SIMPLE RELAXATION (Standalone UFF relaxation)
# ------------------------------------------------------------------------------

# Relax a single CIF file or directory of CIF files
conda run -n mofdiff-gpu python scripts/simple_relax.py \
    --input_path path/to/cif_file_or_directory \
    --output_dir path/to/output

# ------------------------------------------------------------------------------
# 8. BATCH GENERATION EXAMPLE
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
# 9. USEFUL OPTIONS
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
# 10. ENVIRONMENT SETUP
# ------------------------------------------------------------------------------

# Create conda environment
# conda env create -f env.yml

# Activate environment
# conda activate mofdiff-gpu

# Install package in development mode
# pip install -e .

# ------------------------------------------------------------------------------
# END OF COMMANDS
# ------------------------------------------------------------------------------
