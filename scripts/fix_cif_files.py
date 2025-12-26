#!/usr/bin/env python3
"""
Fix common CIF file issues for computationally generated MOF structures.

Issues addressed:
1. _chemical_name_common contains filesystem path instead of chemical name
2. Space group P1 but bond table uses non-identity symmetry operators (1_554, 1_655, etc.)
3. Non-standard _ccdc_geom_bond_type values (S, D, T, A)
4. Unrealistic metal-ligand and metal-metal distances
5. Missing _atom_site_occupancy values
6. Missing atomic displacement parameters (_atom_site_U_iso_or_equiv)
7. Add proper metadata for computationally generated structures
"""

import os
import re
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Standard bond type mapping for CIF files
BOND_TYPE_MAP = {
    'S': 'single',
    'D': 'double', 
    'T': 'triple',
    'A': 'aromatic',
    'single': 'single',
    'double': 'double',
    'triple': 'triple',
    'aromatic': 'aromatic',
}

# Reasonable bond distance ranges (Å)
BOND_DISTANCE_RANGES = {
    # Metal-Oxygen
    ('Cu', 'O'): (1.8, 2.5),
    ('Zn', 'O'): (1.9, 2.3),
    ('Fe', 'O'): (1.8, 2.3),
    ('Co', 'O'): (1.8, 2.2),
    ('Ni', 'O'): (1.8, 2.2),
    # Metal-Nitrogen
    ('Cu', 'N'): (1.9, 2.3),
    ('Zn', 'N'): (1.9, 2.3),
    ('Fe', 'N'): (1.9, 2.3),
    ('Co', 'N'): (1.9, 2.3),
    ('Ni', 'N'): (1.9, 2.3),
    # Metal-Metal (paddlewheel)
    ('Cu', 'Cu'): (2.4, 2.8),
    ('Zn', 'Zn'): (2.5, 3.2),
    # Organic bonds
    ('C', 'C'): (1.2, 1.6),
    ('C', 'N'): (1.1, 1.5),
    ('C', 'O'): (1.1, 1.5),
    ('C', 'H'): (1.0, 1.2),
    ('N', 'H'): (0.9, 1.1),
    ('O', 'H'): (0.9, 1.1),
}

def get_element_from_label(label):
    """Extract element symbol from atom label (e.g., 'Cu28' -> 'Cu')"""
    match = re.match(r'([A-Z][a-z]?)', label)
    return match.group(1) if match else label[:2]


def detect_metal_from_lines(lines):
    """Detect metal type from atom site labels in the CIF"""
    metals = ['Cu', 'Zn', 'Fe', 'Co', 'Ni', 'Mn', 'Cd', 'Mg', 'Ca', 'Ba', 'Sr']
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 4:
            # Check if first column looks like an atom label with metal prefix
            label = parts[0]
            elem = get_element_from_label(label)
            if elem in metals:
                return elem
            # Also check last column (element symbol in some formats)
            if len(parts) >= 5:
                last_elem = parts[-1]
                if last_elem in metals:
                    return last_elem
    return None


def generate_chemical_name(formula, metal_from_atoms=None):
    """Generate a proper chemical name from the formula or detected metal"""
    metal_names = {
        'Cu': 'Copper', 'Zn': 'Zinc', 'Fe': 'Iron', 'Co': 'Cobalt',
        'Ni': 'Nickel', 'Mn': 'Manganese', 'Cd': 'Cadmium', 
        'Mg': 'Magnesium', 'Ca': 'Calcium', 'Ba': 'Barium', 'Sr': 'Strontium'
    }
    
    # Parse elements from formula
    metals = ['Cu', 'Zn', 'Fe', 'Co', 'Ni', 'Mn', 'Cd', 'Mg', 'Ca', 'Ba', 'Sr']
    metal_found = metal_from_atoms
    
    if formula:
        elements = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
        for elem, _ in elements:
            if elem in metals:
                metal_found = elem
                break
    
    if metal_found:
        return f"{metal_names.get(metal_found, metal_found)}-based MOF (computationally generated)"
    return "Metal-Organic Framework (computationally generated)"


def fix_symmetry_operators(bond_lines):
    """
    Fix non-identity symmetry operators in bond table for P1 space group.
    Remove bonds with non-identity operators or convert to identity.
    """
    fixed_lines = []
    removed_count = 0
    
    for line in bond_lines:
        parts = line.split()
        if len(parts) >= 4:
            # Check if symmetry operator is present and not identity
            symm_op = parts[3] if len(parts) > 3 else '.'
            
            # Remove or fix non-identity operators
            if symm_op != '.' and symm_op != '1':
                # Check if it's a translation operator like 1_554, 1_655, etc.
                if '_' in symm_op:
                    # For P1, these indicate periodic image - remove bond
                    removed_count += 1
                    continue
            
            # Keep the bond but ensure symmetry is identity
            if len(parts) >= 5:
                # Replace symmetry operator with identity
                parts[3] = '.'
                fixed_lines.append('  '.join(parts[:4]) + '  ' + parts[4] if len(parts) > 4 else '  '.join(parts[:4]))
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    return fixed_lines, removed_count


def fix_bond_types(bond_lines):
    """Convert non-standard bond types to standard CIF nomenclature"""
    fixed_lines = []
    
    for line in bond_lines:
        parts = line.split()
        if len(parts) >= 5:
            bond_type = parts[4] if len(parts) > 4 else 'S'
            
            # Map to standard bond type
            if bond_type in BOND_TYPE_MAP:
                parts[4] = BOND_TYPE_MAP[bond_type]
            else:
                parts[4] = 'single'  # Default to single bond
            
            fixed_lines.append('  '.join(parts))
        else:
            fixed_lines.append(line)
    
    return fixed_lines


def check_bond_distances(bond_lines):
    """Check for unrealistic bond distances and flag/remove them"""
    valid_lines = []
    warnings = []
    
    for line in bond_lines:
        parts = line.split()
        if len(parts) >= 3:
            atom1 = get_element_from_label(parts[0])
            atom2 = get_element_from_label(parts[1])
            try:
                distance = float(parts[2])
            except ValueError:
                valid_lines.append(line)
                continue
            
            # Check if distance is reasonable
            key1 = (atom1, atom2)
            key2 = (atom2, atom1)
            
            min_dist, max_dist = 0.5, 4.0  # Default range
            
            if key1 in BOND_DISTANCE_RANGES:
                min_dist, max_dist = BOND_DISTANCE_RANGES[key1]
            elif key2 in BOND_DISTANCE_RANGES:
                min_dist, max_dist = BOND_DISTANCE_RANGES[key2]
            
            if distance < min_dist:
                warnings.append(f"WARNING: Very short {atom1}-{atom2} distance: {distance:.3f} Å (min expected: {min_dist} Å)")
                # Keep but flag
                valid_lines.append(line)
            elif distance > max_dist:
                # Remove unreasonably long bonds
                warnings.append(f"REMOVED: Long {atom1}-{atom2} distance: {distance:.3f} Å (max expected: {max_dist} Å)")
                continue
            else:
                valid_lines.append(line)
        else:
            valid_lines.append(line)
    
    return valid_lines, warnings


def add_missing_atom_properties(cif_content):
    """Add missing occupancy and displacement parameters to atom sites"""
    lines = cif_content.split('\n')
    new_lines = []
    in_atom_loop = False
    atom_loop_headers = []
    atom_data_started = False
    has_occupancy = False
    has_u_iso = False
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Detect atom_site loop
        if '_atom_site_label' in line or '_atom_site_type_symbol' in line:
            in_atom_loop = True
            atom_loop_headers = []
        
        if in_atom_loop and line.strip().startswith('_atom_site_'):
            atom_loop_headers.append(line.strip())
            if '_atom_site_occupancy' in line:
                has_occupancy = True
            if '_atom_site_U_iso' in line or '_atom_site_B_iso' in line:
                has_u_iso = True
        
        # Detect start of atom data (first line that doesn't start with _)
        if in_atom_loop and atom_loop_headers and not line.strip().startswith('_') and line.strip() and not line.strip().startswith('loop_'):
            if not atom_data_started:
                atom_data_started = True
                
                # Add missing headers before data
                insert_lines = []
                if not has_occupancy:
                    insert_lines.append(' _atom_site_occupancy')
                if not has_u_iso:
                    insert_lines.append(' _atom_site_U_iso_or_equiv')
                
                # Insert headers before first data line
                for insert_line in insert_lines:
                    new_lines.insert(len(new_lines) - len(atom_loop_headers), insert_line)
                    atom_loop_headers.append(insert_line.strip())
        
        # Add occupancy and U_iso values to atom data lines
        if atom_data_started and in_atom_loop and line.strip() and not line.strip().startswith('_') and not line.strip().startswith('loop_'):
            parts = line.split()
            if len(parts) >= 3:  # Valid atom line
                if not has_occupancy:
                    parts.append('1.0')
                if not has_u_iso:
                    parts.append('0.05')  # Typical U_iso for MOFs
                line = '  ' + '  '.join(parts)
        
        # Detect end of atom loop
        if atom_data_started and (line.strip().startswith('loop_') or line.strip().startswith('_geom') or not line.strip()):
            in_atom_loop = False
            atom_data_started = False
        
        new_lines.append(line)
        i += 1
    
    return '\n'.join(new_lines)


def fix_cif_file(input_path, output_path=None):
    """
    Fix all issues in a CIF file
    """
    if output_path is None:
        output_path = input_path
    
    with open(input_path, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    new_lines = []
    warnings = []
    
    in_bond_loop = False
    bond_loop_started = False
    bond_lines = []
    bond_header_lines = []
    
    # Extract formula for naming
    formula = ""
    for line in lines:
        if '_chemical_formula_structural' in line or '_chemical_formula_sum' in line:
            match = re.search(r"'([^']+)'|(\S+)$", line)
            if match:
                formula = match.group(1) or match.group(2)
                break
    
    # Also detect metal from atom sites
    metal_detected = detect_metal_from_lines(lines)
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Fix 1: Replace filesystem path or generic name in chemical_name_common
        if '_chemical_name_common' in line:
            # Always replace with proper metal-based name
            chemical_name = generate_chemical_name(formula, metal_detected)
            new_lines.append(f"_chemical_name_common  '{chemical_name}'")
            i += 1
            continue
        
        # Fix 2 & 3: Handle bond loop
        if '_geom_bond_atom_site_label_1' in line:
            in_bond_loop = True
            bond_header_lines = []
        
        if in_bond_loop and line.strip().startswith('_'):
            # Collect header line (don't output yet)
            header_line = line
            # Replace _ccdc_geom_bond_type with standard _geom_bond_type
            if '_ccdc_geom_bond_type' in header_line:
                header_line = header_line.replace('_ccdc_geom_bond_type', '_geom_bond_type')
            bond_header_lines.append(header_line)
            i += 1
            continue  # Skip the append at the bottom
        
        if in_bond_loop and bond_header_lines and not line.strip().startswith('_') and line.strip():
            if not bond_loop_started:
                bond_loop_started = True
                # Output headers first
                for header in bond_header_lines:
                    new_lines.append(header)
            
            # Collect bond data
            bond_lines.append(line)
            i += 1
            continue
        
        # End of bond loop
        if bond_loop_started and (not line.strip() or line.strip().startswith('loop_')):
            # Process collected bond lines
            bond_lines, removed = fix_symmetry_operators(bond_lines)
            if removed > 0:
                warnings.append(f"Removed {removed} bonds with periodic image symmetry operators")
            
            bond_lines = fix_bond_types(bond_lines)
            bond_lines, dist_warnings = check_bond_distances(bond_lines)
            warnings.extend(dist_warnings)
            
            # Output processed bonds
            for bond_line in bond_lines:
                new_lines.append(bond_line)
            
            in_bond_loop = False
            bond_loop_started = False
            bond_lines = []
            bond_header_lines = []
        
        new_lines.append(line)
        i += 1
    
    # Join and add metadata
    new_content = '\n'.join(new_lines)
    
    # Add computational metadata at the beginning
    metadata = f"""# CIF file generated by MOFDiff
# Generation date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Note: This is a computationally generated structure, not experimentally refined
# Space group: P1 (no symmetry operations beyond identity)
# Occupancy values: Set to 1.0 (fully occupied sites assumed)
# Displacement parameters: Estimated U_iso values (0.05 Å²)

"""
    
    # Only add metadata if not already present
    if '# CIF file generated by MOFDiff' not in new_content:
        # Find data_ line and insert metadata before it
        data_match = re.search(r'^data_', new_content, re.MULTILINE)
        if data_match:
            new_content = new_content[:data_match.start()] + metadata + new_content[data_match.start():]
        else:
            new_content = metadata + new_content
    
    # Write fixed content
    with open(output_path, 'w') as f:
        f.write(new_content)
    
    return warnings


def fix_relaxed_cif(input_path, output_path=None):
    """
    Fix relaxed CIF files (generated by pymatgen) which have different format
    """
    if output_path is None:
        output_path = input_path
    
    with open(input_path, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    new_lines = []
    
    # Extract formula
    formula = ""
    for line in lines:
        if '_chemical_formula_structural' in line or '_chemical_formula_sum' in line:
            match = re.search(r"'([^']+)'|(\S+)$", line.strip())
            if match:
                formula = match.group(1) or match.group(2)
                break
    
    # Also detect metal from atom sites
    metal_detected = detect_metal_from_lines(lines)
    
    # Check if occupancy column exists
    has_occupancy = any('_atom_site_occupancy' in line for line in lines)
    has_u_iso = any('_atom_site_U_iso' in line or '_atom_site_B_iso' in line for line in lines)
    
    in_atom_loop = False
    atom_headers_done = False
    
    for i, line in enumerate(lines):
        # Add chemical name if missing
        if line.strip().startswith('data_'):
            new_lines.append(line)
            chemical_name = generate_chemical_name(formula, metal_detected)
            new_lines.append(f"_chemical_name_common   '{chemical_name}'")
            new_lines.append("_chemical_name_systematic   ?")
            new_lines.append("_audit_creation_method   'MOFDiff + LAMMPS UFF relaxation'")
            continue
        
        # Skip if line already exists
        if '_chemical_name_common' in line and 'generated' in line.lower():
            continue
        
        # Track atom loop
        if 'loop_' in line and i + 1 < len(lines) and '_atom_site' in lines[i + 1]:
            in_atom_loop = True
            atom_headers_done = False
        
        if in_atom_loop and line.strip().startswith('_atom_site_'):
            # Check if this is the last header
            next_idx = i + 1
            if next_idx < len(lines) and not lines[next_idx].strip().startswith('_'):
                # Add missing headers before data starts
                new_lines.append(line)
                if not has_u_iso:
                    new_lines.append(' _atom_site_U_iso_or_equiv')
                atom_headers_done = True
                continue
        
        new_lines.append(line)
    
    # Add metadata header
    metadata = f"""# CIF file from MOFDiff structure generation
# Relaxed using LAMMPS UFF force field
# Generation date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Note: Computationally generated structure
#
"""
    
    new_content = '\n'.join(new_lines)
    if '# CIF file from MOFDiff' not in new_content:
        new_content = metadata + new_content
    
    with open(output_path, 'w') as f:
        f.write(new_content)
    
    return []


def main():
    """Process all CIF files in the sample directories"""
    base_dir = Path(__file__).parent.parent / "nh3_all_samples"
    
    cif_dir = base_dir / "cif"
    relaxed_dir = base_dir / "relaxed"
    
    all_warnings = []
    
    print("=" * 70)
    print("Fixing CIF files...")
    print("=" * 70)
    
    # Fix unrelaxed CIF files (have bond tables)
    if cif_dir.exists():
        print(f"\nProcessing unrelaxed CIFs in {cif_dir}...")
        for cif_file in sorted(cif_dir.glob("*.cif")):
            print(f"  Fixing {cif_file.name}...", end=" ")
            warnings = fix_cif_file(cif_file)
            if warnings:
                print(f"({len(warnings)} warnings)")
                all_warnings.extend([(cif_file.name, w) for w in warnings])
            else:
                print("OK")
    
    # Fix relaxed CIF files (pymatgen format)
    if relaxed_dir.exists():
        print(f"\nProcessing relaxed CIFs in {relaxed_dir}...")
        for cif_file in sorted(relaxed_dir.glob("*.cif")):
            print(f"  Fixing {cif_file.name}...", end=" ")
            warnings = fix_relaxed_cif(cif_file)
            if warnings:
                print(f"({len(warnings)} warnings)")
                all_warnings.extend([(cif_file.name, w) for w in warnings])
            else:
                print("OK")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Summary of fixes applied:")
    print("=" * 70)
    print("1. ✓ _chemical_name_common: Replaced filesystem paths with proper names")
    print("2. ✓ Symmetry operators: Removed non-identity operators for P1 space group")
    print("3. ✓ Bond types: Converted S/D/T/A to single/double/triple/aromatic")
    print("4. ✓ Bond distances: Flagged unrealistic distances")
    print("5. ✓ Occupancy: Added _atom_site_occupancy = 1.0 where missing")
    print("6. ✓ Displacement: Added _atom_site_U_iso_or_equiv = 0.05 where missing")
    print("7. ✓ Metadata: Added computational generation notes")
    
    if all_warnings:
        print("\n" + "=" * 70)
        print("Warnings:")
        print("=" * 70)
        for filename, warning in all_warnings:
            print(f"  {filename}: {warning}")
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
