#!/usr/bin/env python3
"""Merge all NH3 sample folders into a single organized folder."""

import os
import shutil
import json

# Define source folders and their prefixes for naming
source_folders = {
    'nh3_samples': 'random',           # Random NH3 sampling
    'nh3_guided_samples': 'guided',    # Guided sampling
    'nh3_target_1.5': 'target_1.5',    # Target 1.5 mmol/g
    'nh3_target_5.00': 'target_5.0',   # Target 5.0 mmol/g
    'nh3_gradient_2.0': 'gradient_2.0_v1',   # Gradient optimization 2.0 (v1)
    'nh3_gradient_2.0_v2': 'gradient_2.0',   # Gradient optimization 2.0 (v2 - best)
    'nh3_gradient_5.00': 'gradient_5.0',     # Gradient optimization 5.0
    'nh3_hybrid_1.00': 'hybrid_1.0',   # Hybrid approach 1.0
}

base_dir = '/home/sibivarshan_m7/MOFDiff/MOFDiff'
output_dir = os.path.join(base_dir, 'nh3_all_samples')

# Create output structure
os.makedirs(os.path.join(output_dir, 'cif'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'relaxed'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'pt_files'), exist_ok=True)

# Track all samples info
all_samples_info = {}
file_count = {'cif': 0, 'relaxed': 0, 'pt': 0}

for folder, prefix in source_folders.items():
    src_path = os.path.join(base_dir, folder)
    if not os.path.exists(src_path):
        print(f"Skipping {folder} - not found")
        continue
    
    print(f"\nProcessing {folder} -> prefix: {prefix}")
    all_samples_info[prefix] = {'cif_files': [], 'relaxed_files': []}
    
    # Copy CIF files
    cif_src = os.path.join(src_path, 'cif')
    if os.path.exists(cif_src):
        for f in sorted(os.listdir(cif_src)):
            if f.endswith('.cif'):
                # Extract sample number
                sample_num = f.replace('sample_', '').replace('.cif', '')
                new_name = f"{prefix}_sample_{sample_num}.cif"
                shutil.copy2(os.path.join(cif_src, f), os.path.join(output_dir, 'cif', new_name))
                all_samples_info[prefix]['cif_files'].append(new_name)
                file_count['cif'] += 1
                print(f"  CIF: {f} -> {new_name}")
    
    # Copy relaxed CIF files
    relaxed_src = os.path.join(src_path, 'relaxed')
    if os.path.exists(relaxed_src):
        for f in sorted(os.listdir(relaxed_src)):
            if f.endswith('.cif'):
                # Extract sample number
                sample_num = f.replace('sample_', '').replace('_relaxed.cif', '')
                new_name = f"{prefix}_sample_{sample_num}_relaxed.cif"
                shutil.copy2(os.path.join(relaxed_src, f), os.path.join(output_dir, 'relaxed', new_name))
                all_samples_info[prefix]['relaxed_files'].append(new_name)
                file_count['relaxed'] += 1
                print(f"  Relaxed: {f} -> {new_name}")
        
        # Copy relax_info.json if exists
        relax_info_path = os.path.join(relaxed_src, 'relax_info.json')
        if os.path.exists(relax_info_path):
            with open(relax_info_path, 'r') as fin:
                relax_info = json.load(fin)
            # Save with prefix
            with open(os.path.join(output_dir, 'relaxed', f'{prefix}_relax_info.json'), 'w') as fout:
                json.dump(relax_info, fout, indent=2)
    
    # Copy PT files
    for pt_file in ['samples.pt', 'assembled.pt']:
        pt_src = os.path.join(src_path, pt_file)
        if os.path.exists(pt_src):
            new_name = f"{prefix}_{pt_file}"
            shutil.copy2(pt_src, os.path.join(output_dir, 'pt_files', new_name))
            file_count['pt'] += 1
            print(f"  PT: {pt_file} -> {new_name}")

# Save summary info
with open(os.path.join(output_dir, 'samples_summary.json'), 'w') as f:
    json.dump(all_samples_info, f, indent=2)

print(f"\n{'='*50}")
print(f"Merge complete!")
print(f"  CIF files: {file_count['cif']}")
print(f"  Relaxed files: {file_count['relaxed']}")
print(f"  PT files: {file_count['pt']}")
print(f"\nOutput directory: {output_dir}")
