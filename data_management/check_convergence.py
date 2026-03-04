#!/usr/bin/env python3
"""
Check electronic convergence of VASP interface calculations.

For each VASP run found (identified by an OSZICAR file), reads NELM from
the corresponding INCAR and checks whether the electronic self-consistency
loop converged within fewer than NELM steps.  A calculation is considered
NOT converged if any ionic step used exactly NELM electronic steps.

Usage:
    ./check_convergence.py <directory>

The directory is searched recursively for OSZICAR files.
"""

import argparse
import os
import re
import sys
from pathlib import Path


def read_nelm_from_incar(incar_path):
    """
    Read NELM value from an INCAR file.
    Returns the NELM value as an integer, or None if not found.
    """
    try:
        with open(incar_path, 'r') as f:
            for line in f:
                match = re.search(r'NELM\s*=\s*(\d+)', line, re.IGNORECASE)
                if match:
                    return int(match.group(1))
    except (IOError, ValueError) as e:
        print(f"WARN: Could not read NELM from {incar_path}: {e}",
              file=sys.stderr)
    return None


def check_convergence(oszicar_path, nelm):
    """
    Parse OSZICAR and determine convergence status.

    Returns a dict with:
        converged    : bool or None (None if file is empty / unreadable)
        ionic_steps  : total number of ionic steps completed
        max_escf     : maximum electronic steps taken in any ionic step
        hit_nelm_at  : list of ionic step indices that hit NELM
    """
    result = {
        'converged': None,
        'ionic_steps': 0,
        'max_escf': 0,
        'hit_nelm_at': [],
    }

    try:
        with open(oszicar_path, 'r') as f:
            content = f.read().strip()
    except IOError as e:
        print(f"WARN: Could not read {oszicar_path}: {e}", file=sys.stderr)
        return result

    if not content:
        return result

    current_max_step = 0
    ionic_index = 0

    for line in content.split('\n'):
        # Electronic step line: DAV/RMM/CG:  NNN  ...
        em = re.match(r'\s*(DAV|RMM|CG):\s*(\d+)', line)
        if em:
            step = int(em.group(2))
            current_max_step = max(current_max_step, step)
            continue

        # Ionic step summary line: starts with ionic step number and F=
        if re.match(r'\s*\d+\s+F=', line):
            ionic_index += 1
            if current_max_step > 0:
                result['max_escf'] = max(result['max_escf'], current_max_step)
                if current_max_step >= nelm:
                    result['hit_nelm_at'].append(ionic_index)
            current_max_step = 0

    result['ionic_steps'] = ionic_index
    if ionic_index == 0:
        return result

    result['converged'] = len(result['hit_nelm_at']) == 0
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Check electronic convergence of VASP calculations.')
    parser.add_argument(
        'directory',
        help='Root directory to search recursively for VASP calculations.')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Print details for every calculation, not just non-converged.')
    args = parser.parse_args()

    root = Path(args.directory).resolve()
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory.", file=sys.stderr)
        sys.exit(1)

    converged_list = []
    not_converged_list = []
    no_data_list = []
    no_incar_list = []
    no_nelm_list = []

    for oszicar_path in sorted(root.rglob('OSZICAR')):
        calc_dir = oszicar_path.parent
        try:
            label = str(calc_dir.relative_to(root))
        except ValueError:
            label = str(calc_dir)

        incar_path = calc_dir / 'INCAR'
        if not incar_path.exists():
            no_incar_list.append(label)
            continue

        nelm = read_nelm_from_incar(incar_path)
        if nelm is None:
            no_nelm_list.append(label)
            continue

        info = check_convergence(oszicar_path, nelm)

        if info['converged'] is None:
            no_data_list.append(label)
            continue

        entry = {
            'label': label,
            'nelm': nelm,
            'ionic_steps': info['ionic_steps'],
            'max_escf': info['max_escf'],
            'hit_nelm_at': info['hit_nelm_at'],
        }

        if info['converged']:
            converged_list.append(entry)
        else:
            not_converged_list.append(entry)

    # --- Report ---
    total = len(converged_list) + len(not_converged_list)

    if args.verbose and converged_list:
        print("=== Converged ===")
        for e in converged_list:
            print(f"  {e['label']}  "
                  f"(NELM={e['nelm']}, ionic_steps={e['ionic_steps']}, "
                  f"max_escf={e['max_escf']})")
        print()

    if not_converged_list:
        print("=== NOT Converged (hit NELM) ===")
        for e in not_converged_list:
            bad_steps = ', '.join(str(s) for s in e['hit_nelm_at'])
            print(f"  {e['label']}  "
                  f"(NELM={e['nelm']}, ionic_steps={e['ionic_steps']}, "
                  f"max_escf={e['max_escf']}, "
                  f"hit NELM at ionic step(s): {bad_steps})")
        print()

    if no_data_list:
        print("=== No OSZICAR data (empty / running) ===")
        for label in no_data_list:
            print(f"  {label}")
        print()

    if no_incar_list:
        print("=== Missing INCAR ===")
        for label in no_incar_list:
            print(f"  {label}")
        print()

    if no_nelm_list:
        print("=== NELM not found in INCAR ===")
        for label in no_nelm_list:
            print(f"  {label}")
        print()

    # Summary
    print("===== Summary =====")
    print(f"Total calculations found:     {total + len(no_data_list)}")
    print(f"  Converged:                   {len(converged_list)}")
    print(f"  NOT converged (hit NELM):    {len(not_converged_list)}")
    if no_data_list:
        print(f"  No OSZICAR data:             {len(no_data_list)}")
    if no_incar_list:
        print(f"  Skipped (no INCAR):          {len(no_incar_list)}")
    if no_nelm_list:
        print(f"  Skipped (NELM not in INCAR): {len(no_nelm_list)}")
    if total > 0:
        pct = 100.0 * len(converged_list) / total
        print(f"  Convergence rate:            {pct:.1f}%")


if __name__ == '__main__':
    main()
