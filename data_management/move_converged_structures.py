#!/usr/bin/env python3
"""
Move VASP runs (by OSZICAR) into ./converged/ or ./unconverged/, preserving subpaths.
Converged = OSZICAR does NOT contain a line with "RMM: <NELM>" where NELM is read from INCAR.
Non-converged = OSZICAR contains a line with "RMM: <NELM>".

Usage:
    ./move_converged_structures.py [ROOT_DIR]

Notes:
- We PRUNE the ./converged and ./unconverged directories from the search so we don't re-scan moved runs.
- We mirror the original directory layout under ./converged and ./unconverged to avoid name collisions.
- NELM is read from INCAR files in each calculation directory (not hardcoded).
"""

import os
import re
import shutil
import sys
from pathlib import Path


def read_nelm_from_incar(incar_path):
    """
    Read NELM value from INCAR file.
    Returns the NELM value as an integer, or None if not found.
    """
    try:
        with open(incar_path, 'r') as f:
            for line in f:
                # Match patterns like "NELM = 200", "NELM=200", "NELM 200", etc.
                # Case-insensitive, ignore whitespace variations
                match = re.search(r'NELM\s*=\s*(\d+)', line, re.IGNORECASE)
                if match:
                    return int(match.group(1))
    except (IOError, ValueError) as e:
        print(f"WARN: Could not read NELM from {incar_path}: {e}", file=sys.stderr)
    return None


def is_converged(oszicar_path, nelm):
    """
    Check if OSZICAR indicates convergence.
    Returns True if converged (no "RMM: <NELM>" line found), False otherwise.
    """
    if nelm is None:
        print(f"WARN: NELM not found for {oszicar_path}, skipping convergence check", file=sys.stderr)
        return False
    
    try:
        with open(oszicar_path, 'r') as f:
            for line in f:
                # Match pattern like "RMM: 200" where 200 is the NELM value
                # The pattern should match exactly: RMM: <NELM> followed by whitespace or end of line
                pattern = rf'^\s*RMM:\s*{nelm}(\b|[^0-9])'
                if re.search(pattern, line):
                    return False  # Found RMM: NELM, so NOT converged
        return True  # No RMM: NELM found, so converged
    except IOError as e:
        print(f"WARN: Could not read {oszicar_path}: {e}", file=sys.stderr)
        return False


def find_unique_destination(dest_path):
    """
    If destination exists, find a unique name by appending _1, _2, etc.
    """
    if not dest_path.exists():
        return dest_path
    
    base = dest_path
    i = 1
    while True:
        candidate = Path(f"{base}_{i}")
        if not candidate.exists():
            return candidate
        i += 1


def main():
    # Parse command line argument
    root_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    root_dir = root_dir.resolve()
    
    conv_dir = root_dir / 'converged'
    unconv_dir = root_dir / 'unconverged'
    conv_dir.mkdir(exist_ok=True)
    unconv_dir.mkdir(exist_ok=True)
    
    # Statistics
    total_osz = 0
    moved_conv = 0
    moved_unconv = 0
    skipped_no_incar = 0
    skipped_no_nelm = 0
    errors = 0
    
    # Find all OSZICAR files, excluding those in converged/ and unconverged/ directories
    for oszicar_path in root_dir.rglob('OSZICAR'):
        # Skip if in converged or unconverged directory
        try:
            oszicar_path.relative_to(conv_dir)
            continue  # Skip files in converged/
        except ValueError:
            pass
        try:
            oszicar_path.relative_to(unconv_dir)
            continue  # Skip files in unconverged/
        except ValueError:
            pass  # Not in either directory, continue processing
        
        total_osz += 1
        calc_dir = oszicar_path.parent
        
        # Find INCAR in the same directory
        incar_path = calc_dir / 'INCAR'
        if not incar_path.exists():
            print(f"WARN: No INCAR found in {calc_dir}, skipping", file=sys.stderr)
            skipped_no_incar += 1
            continue
        
        # Read NELM from INCAR
        nelm = read_nelm_from_incar(incar_path)
        if nelm is None:
            print(f"WARN: Could not find NELM in {incar_path}, skipping", file=sys.stderr)
            skipped_no_nelm += 1
            continue
        
        # Check convergence and determine target directory
        if is_converged(oszicar_path, nelm):
            target_dir = conv_dir
            status = "converged"
        else:
            target_dir = unconv_dir
            status = "unconverged"
        
        # Build destination path preserving the directory tree relative to ROOT
        try:
            rel_dir = calc_dir.relative_to(root_dir)
        except ValueError:
            # If calc_dir is not under root_dir, use just the directory name
            rel_dir = Path(calc_dir.name)
        
        dest_parent = target_dir / rel_dir.parent
        dest = target_dir / rel_dir
        
        # Create parent directories
        dest_parent.mkdir(parents=True, exist_ok=True)
        
        # Find unique destination if needed
        dest = find_unique_destination(dest)
        
        print(f"Moving {status}: {calc_dir} -> {dest}")
        try:
            shutil.move(str(calc_dir), str(dest))
            # Update statistics only after successful move
            if status == "converged":
                moved_conv += 1
            else:
                moved_unconv += 1
        except Exception as e:
            print(f"WARN: failed to move {calc_dir}: {e}", file=sys.stderr)
            errors += 1
    
    # Print summary
    print()
    print("===== Summary =====")
    print(f"OSZICAR files scanned: {total_osz}")
    print(f"Converged directories moved: {moved_conv}")
    print(f"Non-converged directories moved: {moved_unconv}")
    print(f"Skipped (no INCAR): {skipped_no_incar}")
    print(f"Skipped (no NELM found): {skipped_no_nelm}")
    print(f"Move errors: {errors}")


if __name__ == '__main__':
    main()

