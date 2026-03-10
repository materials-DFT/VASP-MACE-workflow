#!/usr/bin/env python3
"""
Extract structures from extended XYZ files into individual POSCAR directories.

Each structure gets its own directory named:
    {phase_tag}_{chemical_formula}_{index:04d}

and contains POSCAR, INCAR, and KPOINTS files ready for DFT calculations.
"""

import argparse
import os
from pathlib import Path

from ase.io import read, write


MAGMOM_MAP = {
    'Mn': 3.0,
}
DEFAULT_MAGMOM = 0.0


def generate_magmom(symbols, counts):
    parts = []
    for sym, n in zip(symbols, counts):
        mag = MAGMOM_MAP.get(sym, DEFAULT_MAGMOM)
        parts.append(f"{n}*{mag:g}")
    return ' '.join(parts)


def write_incar(path, system_name, magmom):
    content = f"""\
System = {system_name}

Starting parameters for this run:
ISTART = 0
ICHARG = 2

Electronic Relaxtion:
PREC = Accurate
ENCUT = 520
NELMIN = 6
NELM = 500
EDIFF = 1E-6
LREAL = .FALSE.
ISPIN = 2
MAGMOM = {magmom}
#LNONCOLLINEAR = .TRUE.
ALGO = All
#LDAU = .TRUE.
#LDAUTYPE = 4
#LDAUL = -1 -1 2 -1
#LDAUU = 0 0 5.5 0 
#LDAUJ = 0 0 0 0 
#LMAXMIX = 4
#AMIX = 0.2
#BMIX = 0.001
#LHFCALC = .TRUE.
#GGA = PE

METAGGA = SCAN
LMIXTAU = .TRUE.
LASPH = .TRUE.
LDIAG = .TRUE.

Ionic Molecular Dynamics:
NSW = 1
IBRION = 2
EDIFFG = -5E-02
ISIF = 3
POTIM = 0.5
ISYM = 0

DOS related values:
LORBIT =     10        
##  NEDOS  =     1501
##  EMIN   =    -20.0
##  EMAX   =     10.0
ISMEAR = 0
SIGMA = 0.05

Parallelization flags:
NCORE = 8
NSIM = 4
LPLANE = .TRUE.
LSCALU = .FALSE.
KPAR = 8
NPAR = 2
# LSCALAPACK = .FALSE.
LWAVE = .FALSE.
LCHARG = .FALSE.
"""
    with open(path, 'w') as f:
        f.write(content)


def write_kpoints(path):
    content = """\
Gamma KPOINTS
0
Monkhorst-Pack
1 1 1
0 0 0
"""
    with open(path, 'w') as f:
        f.write(content)


def get_element_counts(atoms):
    """Return (symbols, counts) sorted alphabetically to match ASE's sort=True."""
    from collections import Counter
    c = Counter(atoms.get_chemical_symbols())
    symbols = sorted(c.keys())
    return symbols, [c[s] for s in symbols]


def extract_poscars(xyz_file, output_dir=None, dry_run=False):
    xyz_path = Path(xyz_file).resolve()
    if not xyz_path.is_file():
        print(f"Error: '{xyz_file}' does not exist")
        return 1

    stem = xyz_path.stem
    if output_dir is None:
        output_base = xyz_path.parent / stem
    else:
        output_base = Path(output_dir).resolve()

    print(f"Reading {xyz_path} ...")
    atoms_list = read(str(xyz_path), index=':')
    print(f"Found {len(atoms_list)} structures")

    if dry_run:
        print("\n--- Dry run (no files written) ---")

    created = 0
    for i, atoms in enumerate(atoms_list):
        defects = atoms.info.get('defects_generic', {})
        phase = defects.get('phase_tag', 'unknown')
        formula = atoms.get_chemical_formula()
        dir_name = f"{phase}_{formula}_{i:04d}"

        if dry_run:
            print(f"  Would create: {dir_name}/")
            continue

        dest = output_base / dir_name
        dest.mkdir(parents=True, exist_ok=True)

        poscar_path = dest / 'POSCAR'
        write(str(poscar_path), atoms, format='vasp', vasp5=True, sort=True)

        symbols, counts = get_element_counts(atoms)
        system_name = ''.join(f"{s}{n}" for s, n in zip(symbols, counts))
        magmom = generate_magmom(symbols, counts)

        write_incar(dest / 'INCAR', system_name, magmom)
        write_kpoints(dest / 'KPOINTS')

        created += 1
        if created % 100 == 0:
            print(f"  Processed {created}/{len(atoms_list)} ...")

    if not dry_run:
        print(f"\nDone! Created {created} directories under {output_base}")
    else:
        print(f"\nWould create {len(atoms_list)} directories under {output_base}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Extract structures from XYZ into POSCAR directories for DFT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s filtered_min_dist_1.2A.xyz
  %(prog)s filtered_min_dist_1.5A.xyz -o ./my_output_dir
  %(prog)s filtered_min_dist_1.2A.xyz --dry-run
"""
    )
    parser.add_argument('xyz_file', help='Extended XYZ file containing structures')
    parser.add_argument('-o', '--output-dir',
                        help='Output directory (default: same location as XYZ file, '
                             'named after the file stem)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be created without writing any files')

    args = parser.parse_args()
    exit(extract_poscars(args.xyz_file, args.output_dir, args.dry_run))


if __name__ == '__main__':
    main()
