#!/usr/bin/env python3
"""
Extract the lowest-energy frame from each VASP OUTCAR into a single extended XYZ file.

For each OUTCAR found under the given directory, reads the full ionic trajectory,
picks the frame with the minimum total energy, and writes one combined XYZ with
one frame per structure (energy, forces, lattice included).

Usage:
  python extract_frames.py <directory> [-o output.xyz]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from ase.io import read, write
except ImportError:
    print("Error: ASE is required. Install with: pip install ase", file=sys.stderr)
    sys.exit(1)


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
            best = min(images, key=lambda a: a.get_potential_energy())
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
