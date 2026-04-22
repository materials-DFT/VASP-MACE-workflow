#!/usr/bin/env python3
"""
Convert a POSCAR file to a LAMMPS data file using ASE.

By default we use ASE's triclinic LAMMPS box (force_skew=True). Orthogonal-only
exports omit xy/xz/yz even when the POSCAR cell is slightly non-orthogonal; that
representation has triggered segfaults in LAMMPS pair allegro/kk + LibTorch on
some GPU stacks. force_skew writes explicit tilt factors and avoids that path.

Velocities: by default ASE writes a ``Velocities`` block when ``velocities=True``.
Cartesian speeds from VASP (Å/fs in the POSCAR/CONTCAR convention ASE uses) are
converted to LAMMPS ``metal`` units (Å/ps). Use ``--no-velocities`` to omit the
block (then LAMMPS initializes v=0 unless you ``velocity create``).

Usage:
    python poscar_to_lammps.py POSCAR [output_filename]
    python poscar_to_lammps.py POSCAR data.lammps --no-force-skew   # legacy orthogonal
    python poscar_to_lammps.py CONTCAR data.lammps --no-velocities

Example:
    python poscar_to_lammps.py POSCAR data.lammps
"""

import argparse
import sys

import numpy as np
from ase.io import read, write


def main() -> None:
    p = argparse.ArgumentParser(description="Convert POSCAR to LAMMPS data (ASE).")
    p.add_argument("poscar", help="Input POSCAR path")
    p.add_argument(
        "output",
        nargs="?",
        default="data.lammps",
        help="Output LAMMPS data file (default: data.lammps)",
    )
    p.add_argument(
        "--no-force-skew",
        action="store_true",
        help="Let ASE omit triclinic tilt if nearly orthogonal (not recommended for Allegro/Kokkos GPU)",
    )
    p.add_argument(
        "--no-velocities",
        action="store_true",
        help="Do not write a Velocities section (default: write if ASE has velocities; VASP files without a block appear as zeros)",
    )
    args = p.parse_args()

    try:
        atoms = read(args.poscar, format="vasp")
        write_vel = not args.no_velocities
        write(
            args.output,
            atoms,
            format="lammps-data",
            atom_style="atomic",
            masses=True,
            force_skew=not args.no_force_skew,
            velocities=write_vel,
            units="metal",
        )
        skew = "orthogonal (no force_skew)" if args.no_force_skew else "triclinic (force_skew=True)"
        vmax = None
        if atoms.get_velocities() is not None:
            vmax = float(np.max(np.abs(atoms.get_velocities())))
        vel_note = (
            "no Velocities section"
            if args.no_velocities
            else (f"velocities (|v|_max ≈ {vmax:.4g} Å/fs in ASE)" if vmax is not None else "velocities")
        )
        print(
            f"Wrote {args.output} from {args.poscar} ({len(atoms)} atoms, {skew}, {vel_note})."
        )
    except OSError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
