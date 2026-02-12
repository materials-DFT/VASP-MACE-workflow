#!/usr/bin/env python3
"""
Extract the lowest-energy frame from each VASP OUTCAR into a single extended XYZ file.

For each OUTCAR found under the given directory, reads the full ionic trajectory,
picks the frame with the minimum total energy, and writes one combined XYZ with
one frame per structure (energy, forces, lattice, and stress when available).

Usage:
  python extract_frames.py <directory> [-o output.xyz]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from ase.io import read, write
    from ase.calculators.singlepoint import SinglePointCalculator
except ImportError:
    print("Error: ASE is required. Install with: pip install ase", file=sys.stderr)
    sys.exit(1)

# VASP stress in OUTCAR is in kB. 1 kB = 0.1 GPa; 1 eV/Å³ = 160.2176634 GPa.
KB_TO_EV_ANG3 = 0.1 / 160.2176634


def get_stress_at_step_from_outcar(outcar_path: Path, step_index: int) -> list[float] | None:
    """
    Parse OUTCAR and return the stress at the given ionic step as Voigt 6 (xx, yy, zz, yz, xz, xy)
    in eV/Å³. Returns None if not found.
    """
    try:
        with open(outcar_path, "r") as f:
            lines = f.readlines()
    except OSError:
        return None
    stress_list = []
    i = 0
    ionic_step = -1
    while i < len(lines):
        if "FREE ENERGIE OF THE ION-ELECTRON SYSTEM" in lines[i]:
            ionic_step += 1
        if " in kB " in lines[i].strip():
            stress_3x3 = []
            for j in range(i + 1, min(i + 4, len(lines))):
                parts = lines[j].split()
                if len(parts) >= 3:
                    try:
                        row = [float(parts[0]), float(parts[1]), float(parts[2])]
                        stress_3x3.append(row)
                    except ValueError:
                        break
            if len(stress_3x3) == 3:
                s = stress_3x3
                voigt_kb = [
                    s[0][0], s[1][1], s[2][2],
                    s[1][2], s[0][2], s[0][1]
                ]
                stress_list.append([v * KB_TO_EV_ANG3 for v in voigt_kb])
            i += 4
            continue
        i += 1
    if step_index < len(stress_list):
        return stress_list[step_index]
    return None


def ensure_stress_on_atoms(atoms, outcar_path: Path, step_index: int) -> None:
    """If atoms do not have stress, try to read it from OUTCAR for the given step and set it."""
    try:
        atoms.get_stress()
        return
    except (AttributeError, KeyError, RuntimeError):
        pass
    stress = get_stress_at_step_from_outcar(outcar_path, step_index)
    if stress is not None:
        # Attach a SinglePointCalculator with stress so that write(extxyz) includes it
        calc = SinglePointCalculator(atoms, stress=stress)
        atoms.calc = calc


def find_outcars(root: Path) -> list[Path]:
    """Find all OUTCAR files under root (recursive)."""
    return sorted(root.rglob("OUTCAR"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract lowest-energy frame from each VASP OUTCAR into one extended XYZ file."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Root directory to search for OUTCAR files",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("optimized_frames.xyz"),
        help="Output extended XYZ file (default: optimized_frames.xyz)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )
    args = parser.parse_args()

    root = args.directory.resolve()
    if not root.is_dir():
        print(f"Error: Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    outcars = find_outcars(root)
    if not outcars:
        print("Error: No OUTCAR files found.", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"Found {len(outcars)} OUTCAR(s). Extracting lowest-energy frame per structure...")

    best_frames = []
    for p in outcars:
        try:
            images = read(str(p), index=":")
            min_idx = min(range(len(images)), key=lambda i: images[i].get_potential_energy())
            best = images[min_idx]
            ensure_stress_on_atoms(best, p, min_idx)
            best_frames.append(best)
            if not args.quiet:
                e = best.get_potential_energy()
                print(f"  {p.relative_to(root)}: {len(images)} frame(s) -> E = {e:.4f} eV")
        except Exception as e:
            print(f"Warning: Could not read {p}: {e}", file=sys.stderr)

    if not best_frames:
        print("Error: No frames could be read from any OUTCAR.", file=sys.stderr)
        sys.exit(1)

    write(str(args.output), best_frames, format="extxyz")
    if not args.quiet:
        print(f"Wrote {len(best_frames)} frame(s) to {args.output}")


if __name__ == "__main__":
    main()
