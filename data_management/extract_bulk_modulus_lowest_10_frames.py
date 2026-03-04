#!/usr/bin/env python3
"""
Scan specified directories for VASP OUTCARs, find the lowest-energy
frame per directory (phase), then extract the N frames closest in energy
to that minimum. Uses ASE to read OUTCAR and write extended XYZ (lattice,
energy, forces, stress). OUTCAR only.

Difference from extract_optimized_frames.py:
  - Optimized frames: ONE frame per OUTCAR (every volume point), extended XYZ.
  - This script: N frames per PHASE (only the N volumes closest to equilibrium),
    same extended XYZ format. Use for bulk-modulus / E(V) fitting.
"""

import re
import sys
import argparse
import os
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


def get_final_energy_from_outcar(outcar_path):
    """Extract final TOTEN (eV) from OUTCAR. Returns None if not found. Lightweight (no ASE)."""
    pattern = re.compile(r"free\s+energy\s+TOTEN\s*=\s*([-\d.Ee+]+)\s*eV")
    last_energy = None
    try:
        with open(outcar_path, "r") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    last_energy = float(m.group(1))
        return last_energy
    except (OSError, ValueError):
        return None


def _collect_candidates(top_path):
    """Direct children that have OUTCAR (OUTCAR only)."""
    candidates = []
    for sub in sorted(top_path.iterdir()):
        if not sub.is_dir():
            continue
        outcar = sub / "OUTCAR"
        if not outcar.is_file():
            continue
        energy = get_final_energy_from_outcar(outcar)
        if energy is None:
            continue
        candidates.append((sub, energy))
    return candidates


def make_run_id(root: Path, outcar_path: Path) -> str:
    """Full directory path from ~ as run_id."""
    abs_dir = str(outcar_path.resolve().parent)
    home = str(Path.home())
    if abs_dir.startswith(home):
        return "~" + abs_dir[len(home):]
    return abs_dir


def process_directory(top_dir, num_frames=10):
    """
    For one directory (e.g. alpha/), find subdirs with OUTCAR, get final energy
    from each, find minimum, then the num_frames subdirs closest in energy.
    Return list of ASE Atoms (final frame from each selected OUTCAR).
    """
    top_path = Path(top_dir).resolve()
    if not top_path.is_dir():
        print(f"Not a directory: {top_path}", file=sys.stderr)
        return []

    candidates = _collect_candidates(top_path)
    if not candidates:
        all_frames = []
        for child in sorted([p for p in top_path.iterdir() if p.is_dir()]):
            all_frames.extend(process_directory(child, num_frames))
        return all_frames

    min_energy = min(e for _, e in candidates)
    by_dist = sorted(candidates, key=lambda x: abs(x[1] - min_energy))
    selected = by_dist[:num_frames]

    frames = []
    for sub, en in selected:
        outcar = sub / "OUTCAR"
        try:
            images = read(str(outcar), index=":")
            atoms = images[-1]
            atoms.info["comment"] = f"{sub.name}  E = {en:.6f} eV"
            atoms.info["run_id"] = make_run_id(top_path, outcar)
            frames.append(atoms)
        except Exception as e:
            print(f"Failed to read OUTCAR with ASE: {outcar} ({e})", file=sys.stderr)
            continue

    if not frames:
        print(f"No valid structures under {top_path}", file=sys.stderr)
        return []
    print(f"Collected {len(frames)} frames from {top_path} (min E = {min_energy:.6f} eV)")
    return frames


def main():
    parser = argparse.ArgumentParser(
        description="Extract N frames closest to minimum energy per phase (OUTCAR only, ASE, extended XYZ)."
    )
    parser.add_argument(
        "directories",
        nargs="+",
        help="Phase directories to scan (e.g. alpha beta gamma); each should contain subdirs with OUTCAR",
    )
    parser.add_argument(
        "-o", "--output",
        default="combined_lowest_frames.xyz",
        help="Output extended XYZ path (default: combined_lowest_frames.xyz)",
    )
    parser.add_argument(
        "-n", "--num-frames",
        type=int,
        default=10,
        help="Frames per phase closest to min energy (default: 10)",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Log file path (default: <output_stem>.log, e.g. combined_lowest_frames.log)",
    )
    args = parser.parse_args()

    # Log file: default to <output_stem>.log
    if args.log is None:
        args.log = os.path.splitext(args.output)[0] + ".log"

    tee = None
    try:
        tee = Tee(args.log)
        sys.stdout = tee

        all_frames = []
        for d in args.directories:
            all_frames.extend(process_directory(d, args.num_frames))

        if not all_frames:
            print("No frames collected.", file=sys.stderr)
            sys.exit(1)

        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write(str(out_path), all_frames, format="extxyz")
        print(f"Wrote {len(all_frames)} frames to {out_path}")
    finally:
        if tee is not None:
            sys.stdout = tee._stream
            tee.close()
            print(f"Log written to {args.log}")


if __name__ == "__main__":
    main()
