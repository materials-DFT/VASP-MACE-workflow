#!/usr/bin/env python3
"""
Extract ALL frames from VASP OUTCAR(s) in a directory into a single extended XYZ file.

Uses the same extraction method as extract_frames_for_mlff.py and extract_optimized_frames.py:
ASE reads OUTCAR trajectory (energy, forces, stress, lattice) and writes extended XYZ.

Unlike extract_frames_for_mlff.py (stride/skip/cap) or extract_optimized_frames.py (lowest-energy
only), this script extracts every frame from every OUTCAR found under the given directory.

Usage:
  python extract_all_frames.py <directory> [-o output.xyz]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from ase.io import read, write
except ImportError:
    print("Error: ASE is required. Install with: pip install ase", file=sys.stderr)
    sys.exit(1)


# ============================================================================
# Logging (tee to stdout + log file) — same style as extract_frames_for_mlff.py
# ============================================================================

class Tee:
    """Write to both stdout and a log file."""
    def __init__(self, log_path, stream=None):
        self._stream = stream if stream is not None else sys.stdout
        self._log_path = log_path
        self._file = open(log_path, 'w', encoding='utf-8')

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        self._file.close()


def find_outcars(root: Path) -> list[Path]:
    """Find all OUTCAR files under root (recursive)."""
    return sorted(root.rglob("OUTCAR"))


def make_run_id(root: Path, outcar_path: Path) -> str:
    """Full directory path from ~ as run_id."""
    abs_dir = str(outcar_path.resolve().parent)
    home = str(Path.home())
    if abs_dir.startswith(home):
        return "~" + abs_dir[len(home):]
    return abs_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract ALL frames from VASP OUTCAR(s) into one extended XYZ file."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory to search for OUTCAR files (searches recursively)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output extended XYZ file (default: all_frames_<count>.xyz)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Log file path (default: <output_stem>.log)",
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

    # Default output: all_frames_<count>.xyz (we'll set count after extraction)
    output_path = args.output
    if output_path is None:
        output_path = Path("all_frames.xyz")  # placeholder, renamed after

    # Log file: default to <output_stem>.log
    if args.log is None:
        args.log = os.path.splitext(str(output_path))[0] + ".log"

    tee = None
    try:
        tee = Tee(args.log)
        sys.stdout = tee

        if not args.quiet:
            print(f"Found {len(outcars)} OUTCAR(s). Extracting all frames...")

        all_frames = []
        for p in outcars:
            try:
                images = read(str(p), index=":")
                run_id = make_run_id(root, p)
                for atoms in images:
                    atoms.info["run_id"] = run_id
                all_frames.extend(images)
                if not args.quiet:
                    print(f"  {p.relative_to(root)}: {len(images)} frame(s)")
            except Exception as e:
                print(f"Warning: Could not read {p}: {e}", file=sys.stderr)

        if not all_frames:
            print("Error: No frames could be read from any OUTCAR.", file=sys.stderr)
            sys.exit(1)

        if args.output is None:
            output_path = Path(f"all_frames_{len(all_frames)}_frames.xyz")

        write(str(output_path), all_frames, format="extxyz")

        if not args.quiet:
            print(f"Wrote {len(all_frames)} frame(s) to {output_path}")
    finally:
        if tee is not None:
            sys.stdout = tee._stream
            tee.close()
            print(f"Log written to {args.log}")


if __name__ == "__main__":
    main()
