#!/usr/bin/env python3
"""
Script to update INCAR files with System and MAGMOM based on POSCAR data.
"""

import os
import re
import sys
from pathlib import Path

def parse_poscar(poscar_path):
    """Parse POSCAR to extract element names and counts."""
    with open(poscar_path, 'r') as f:
        lines = [line.rstrip() for line in f.readlines()]
    
    elements = []
    counts = []
    
    # POSCAR format:
    # Line 0: comment or element names (K  Mn O)
    # Line 1: scaling factor (1.0)
    # Lines 2-4: lattice vectors (3 numbers each)
    # Line 5: element names (K   Mn  O)  
    # Line 6: element counts (2  24  41)
    
    # Find non-empty lines with their original indices
    non_empty = [(i, line.strip()) for i, line in enumerate(lines) if line.strip()]
    
    if len(non_empty) >= 7:
        # Standard POSCAR format: element names at line 5, counts at line 6 (0-indexed)
        # But we need to find them by content, not just position
        
        for i in range(len(non_empty) - 1):
            idx1, line1 = non_empty[i]
            idx2, line2 = non_empty[i+1]
            
            parts1 = line1.split()
            parts2 = line2.split()
            
            # Check if line2 contains all integers (counts)
            if parts2 and all(part.isdigit() for part in parts2):
                # Check if line1 contains element symbols (letters, not all numbers/decimals)
                # Element symbols contain letters
                if parts1 and any(any(c.isalpha() for c in part) for part in parts1):
                    # Verify: line2 should be right after line1 in original file (or close)
                    if idx2 == idx1 + 1:
                        elements = parts1
                        counts = [int(c) for c in parts2]
                        break
    
    return elements, counts

def update_incar(incar_path, elements, counts):
    """Update INCAR with System and MAGMOM based on element data."""
    with open(incar_path, 'r') as f:
        lines = f.readlines()
    
    # MAGMOM values: K=0, Mn=3, O=0 (based on example)
    magmom_map = {'K': 0, 'Mn': 3, 'O': 0}
    
    # Build System string: K2Mn24O41 format
    system_parts = []
    for elem, count in zip(elements, counts):
        system_parts.append(f"{elem}{count}")
    system_str = ''.join(system_parts)
    
    # Build MAGMOM string: 2*0 24*3 41*0 format
    magmom_parts = []
    for elem, count in zip(elements, counts):
        magmom_val = magmom_map.get(elem, 0)
        magmom_parts.append(f"{count}*{magmom_val}")
    magmom_str = ' '.join(magmom_parts)
    
    # Update lines
    updated_lines = []
    for line in lines:
        # Update System line
        if re.match(r'^\s*System\s*=', line, re.IGNORECASE):
            updated_lines.append(f"System = {system_str}\n")
        # Update MAGMOM line
        elif re.match(r'^\s*MAGMOM\s*=', line, re.IGNORECASE):
            updated_lines.append(f"MAGMOM = {magmom_str}\n")
        else:
            updated_lines.append(line)
    
    # Write back
    with open(incar_path, 'w') as f:
        f.writelines(updated_lines)
    
    return system_str, magmom_str

def main():
    # Accept base directory from command line or use default
    if len(sys.argv) >= 2:
        base_dir = Path(sys.argv[1])
    else:
        base_dir = Path("/Users/924322630/amorphous_structures/more_amorphous/selected_structures_stratified/dft")
    
    if not base_dir.exists():
        print(f"Error: Base directory '{base_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    # Find all directories containing both POSCAR and INCAR recursively
    processed = 0
    errors = []
    
    for poscar_path in base_dir.rglob("POSCAR"):
        # Check if there's an INCAR in the same directory
        incar_path = poscar_path.parent / "INCAR"
        
        if incar_path.exists():
            try:
                elements, counts = parse_poscar(poscar_path)
                if elements and counts and len(elements) == len(counts):
                    system_str, magmom_str = update_incar(incar_path, elements, counts)
                    print(f"Updated {incar_path}: System={system_str}, MAGMOM={magmom_str}")
                    processed += 1
                else:
                    errors.append(f"{poscar_path}: Could not parse elements/counts correctly")
            except Exception as e:
                errors.append(f"{poscar_path}: Error - {str(e)}")
    
    print(f"\nProcessed {processed} INCAR files successfully.")
    if errors:
        print(f"\n{len(errors)} errors encountered:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

if __name__ == "__main__":
    main()
