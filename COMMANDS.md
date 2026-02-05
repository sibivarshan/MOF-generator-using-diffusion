# MOFDiff Commands Reference

## Environment Setup
```bash
conda activate mofdiff-gpu
cd /home/sibivarshan_m7/MOFDiff/MOFDiff
```

---

## 🚀 MODULAR PIPELINE (Recommended)

### One Command for Any Target NH3 Uptake
```bash
# Generate MOF for ANY target NH3 uptake value
python scripts/run_nh3_pipeline.py --target <VALUE>

# Examples:
python scripts/run_nh3_pipeline.py --target 2.0    # Target 2 mmol/g
python scripts/run_nh3_pipeline.py --target 5.0    # Target 5 mmol/g
python scripts/run_nh3_pipeline.py --target 10.0   # Target 10 mmol/g
python scripts/run_nh3_pipeline.py --target 15.0   # Target 15 mmol/g
```

### Pipeline Options
```bash
# With custom number of samples
python scripts/run_nh3_pipeline.py --target 8.0 --n_samples 20

# With specific seed for reproducibility
python scripts/run_nh3_pipeline.py --target 6.0 --seed 42

# Custom output directory
python scripts/run_nh3_pipeline.py --target 4.0 --output_dir results/my_custom_run

# Skip specific steps (if already done)
python scripts/run_nh3_pipeline.py --target 5.0 --skip_generation  # Use existing samples
python scripts/run_nh3_pipeline.py --target 5.0 --skip_gcmc        # Skip GCMC validation

# Full options
python scripts/run_nh3_pipeline.py --target 7.0 --n_samples 15 --seed 999
```

### What the Pipeline Does:
1. **Generate** - Creates MOF samples using MOFDiff
2. **Assemble** - Converts coarse-grained to atomistic structures
3. **Relax** - Optimizes structures using Pymatgen
4. **GCMC** - Validates NH3 uptake with RASPA simulations
5. **Report** - Finds best match and saves results

---

## 1. Normal MOF Generation (Unconditional)

### Generate Samples
```bash
python mofdiff/scripts/sample.py \
  --model_path pretrained/mofdiff_ckpt \
  --bb_cache_path pretrained/bb_emb_space.pt \
  --n_samples 10 \
  --seed 2024
```
Output: `pretrained/mofdiff_ckpt/samples_10_seed_2024/samples.pt`

### Assemble Structures
```bash
python mofdiff/scripts/assemble.py --input pretrained/mofdiff_ckpt/samples_10_seed_2024/samples.pt
```
Output: `assembled.pt` and `cif/` folder with CIF files

### Relax Structures
```bash
python scripts/simple_relax.py --input_dir pretrained/mofdiff_ckpt/samples_10_seed_2024
```
Output: `relaxed/` folder with relaxed CIF files

### Run GCMC (Single Structure)
```bash
python scripts/run_single_gcmc.py --cif_path path/to/structure.cif --output_dir gcmc_output
```

### Run GCMC (Parallel - Multiple Structures)
```bash
python scripts/run_gcmc_parallel.py \
  --input_dir pretrained/mofdiff_ckpt/samples_10_seed_2024/relaxed \
  --target_nh3 5.0
```
Output: `gcmc_parallel/all_results.json`

---

## 2. NH3-Guided MOF Generation (Target Specific Uptake)

### Step 1: Generate Samples with New Seed
```bash
# Create results directory
mkdir -p results/nh3_target_Xmmol

# Generate 10 samples
python mofdiff/scripts/sample.py \
  --model_path pretrained/mofdiff_ckpt \
  --bb_cache_path pretrained/bb_emb_space.pt \
  --n_samples 10 \
  --seed 1234

# Copy samples to results directory
cp pretrained/mofdiff_ckpt/samples_10_seed_1234/samples.pt results/nh3_target_Xmmol/
```

### Step 2: Assemble Structures
```bash
python mofdiff/scripts/assemble.py --input results/nh3_target_Xmmol/samples.pt
```

### Step 3: Relax Structures
```bash
python scripts/simple_relax.py --input_dir results/nh3_target_Xmmol
```

### Step 4: Run GCMC with Target
```bash
# Replace X.0 with your target NH3 uptake (e.g., 2.0, 5.0, 10.0)
python scripts/run_gcmc_parallel.py \
  --input_dir results/nh3_target_Xmmol/relaxed \
  --target_nh3 X.0
```

---

## 3. Complete Pipeline Examples

### Example: Target 10 mmol/g NH3 Uptake
```bash
# Setup
mkdir -p results/nh3_target_10mmol

# Generate
python mofdiff/scripts/sample.py \
  --model_path pretrained/mofdiff_ckpt \
  --bb_cache_path pretrained/bb_emb_space.pt \
  --n_samples 10 \
  --seed 1234

# Copy and assemble
cp pretrained/mofdiff_ckpt/samples_10_seed_1234/samples.pt results/nh3_target_10mmol/
python mofdiff/scripts/assemble.py --input results/nh3_target_10mmol/samples.pt

# Relax
python scripts/simple_relax.py --input_dir results/nh3_target_10mmol

# GCMC
python scripts/run_gcmc_parallel.py \
  --input_dir results/nh3_target_10mmol/relaxed \
  --target_nh3 10.0
```

### Example: Target 2 mmol/g NH3 Uptake
```bash
# Setup
mkdir -p results/nh3_target_2mmol

# Generate
python mofdiff/scripts/sample.py \
  --model_path pretrained/mofdiff_ckpt \
  --bb_cache_path pretrained/bb_emb_space.pt \
  --n_samples 10 \
  --seed 5678

# Copy and assemble
cp pretrained/mofdiff_ckpt/samples_10_seed_5678/samples.pt results/nh3_target_2mmol/
python mofdiff/scripts/assemble.py --input results/nh3_target_2mmol/samples.pt

# Relax
python scripts/simple_relax.py --input_dir results/nh3_target_2mmol

# GCMC
python scripts/run_gcmc_parallel.py \
  --input_dir results/nh3_target_2mmol/relaxed \
  --target_nh3 2.0
```

---

## 4. GCMC Environment Variables (if running RASPA directly)

```bash
export RASPA_DIR=/home/sibivarshan_m7/gcmc_tools/raspa_install
export LD_LIBRARY_PATH=/home/sibivarshan_m7/gcmc_tools/raspa_install/lib:$LD_LIBRARY_PATH
```

---

## 5. Results Summary

Results are stored in:
- `results/nh3_target_Xmmol/`
  - `samples.pt` - Generated coarse-grained samples
  - `assembled.pt` - Assembled structures
  - `cif/` - Original CIF files
  - `relaxed/` - Relaxed CIF files
  - `gcmc_parallel/` - GCMC simulation outputs
  - `gcmc_parallel/all_results.json` - NH3 uptake results
  - `best_structure_Xmmol.cif` - Best matching structure
  - `final_results.json` - Summary with comparisons

---

## 6. Previous Run Results

| Run | Target (mmol/g) | Achieved (mmol/g) | Best Structure | Difference |
|-----|-----------------|-------------------|----------------|------------|
| 10 mmol/g | 10.0 | 10.13 | sample_3_relaxed | 0.13 |
| 2 mmol/g | 2.0 | 2.32 | sample_6_relaxed | 0.32 |
