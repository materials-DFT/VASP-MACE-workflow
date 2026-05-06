#!/usr/bin/env python3
"""
Scan specified directories for VASP OUTCARs, find the lowest-energy
frame per directory (phase), then extract the N frames closest in energy
to that minimum. Uses ASE to read OUTCAR and write extended XYZ (lattice,
energy, forces, stress). OUTCAR only. By default, only electronically
converged frames are used: if the last ionic step is unconverged, the script
falls back to the nearest earlier converged frame. Use --no-convergence-check
for the previous behavior.

Difference from extract_optimized_frames.py:
  - Optimized frames: ONE frame per OUTCAR (every volume point), extended XYZ.
  - This script: N frames per PHASE (only the N volumes closest to equilibrium),
    same extended XYZ format. Use for bulk-modulus / E(V) fitting.
"""

import sys
import argparse
import os
from pathlib import Path

try:
    from ase.io import read, write
except ImportError:
    print("Error: ASE is required. Install with: pip install ase", file=sys.stderr)
    sys.exit(1)

from vasp_step_convergence import per_step_electronically_converged


def _last_converged_index(ok: list[bool]) -> int | None:
    """Return the last converged frame index, or None if no converged frame exists."""
    for idx in range(len(ok) - 1, -1, -1):
        if ok[idx]:
            return idx
    return None


# ============================================================================
# Logging (tee to stdout + log file) — same style as other extract_* scripts
# ============================================================================

class Tee:
    """Write to both stdout and a log file."""

    def __init__(self, log_path, stream=None):
        self._stream = stream if stream is not None else sys.stdout
        self._log_path = log_path
        self._file = open(log_path, "w", encoding="utf-8")

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        self._file.close()


def _collect_candidates(top_path, check_convergence: bool = True):
    """
    Direct children that have OUTCAR (OUTCAR only).

    For consistency with other extraction scripts, use ASE to read the
    OUTCAR and take the potential energy of the selected ionic step.
    When check_convergence is True, selection starts at the final step and
    falls back to the nearest earlier converged step if needed.
    """
    candidates = []
    for sub in sorted(top_path.iterdir()):
        if not sub.is_dir():
            continue
        outcar = sub / "OUTCAR"
        if not outcar.is_file():
            continue
        try:
            images = read(str(outcar), index=":")
        except Exception:
            continue
        if not images:
            continue
        selected_idx = len(images) - 1
        if check_convergence:
            ok = per_step_electronically_converged(
                outcar, sub / "INCAR", sub / "OSZICAR"
            )
            if (
                ok is None
                or len(ok) != len(images)
            ):
                continue
            selected = _last_converged_index(ok)
            if selected is None:
                continue
            selected_idx = selected
        last = images[selected_idx]
        try:
            energy = float(last.get_potential_energy())
        except Exception:
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


def process_directory(
    top_dir, num_frames: int = 10, check_convergence: bool = True
):
    """
    For one directory (e.g. alpha/), find subdirs with OUTCAR, get selected-frame
    energy from each, find minimum, then the num_frames subdirs closest in energy.
    Return list of ASE Atoms (selected frame from each chosen OUTCAR).
    """
    top_path = Path(top_dir).resolve()
    if not top_path.is_dir():
        print(f"Not a directory: {top_path}", file=sys.stderr)
        return []

    candidates = _collect_candidates(top_path, check_convergence=check_convergence)
    if not candidates:
        all_frames = []
        for child in sorted([p for p in top_path.iterdir() if p.is_dir()]):
            all_frames.extend(
                process_directory(
                    child, num_frames, check_convergence=check_convergence
                )
            )
        return all_frames

    min_energy = min(e for _, e in candidates)
    by_dist = sorted(candidates, key=lambda x: abs(x[1] - min_energy))
    selected = by_dist[:num_frames]

    frames = []
    for sub, en in selected:
        outcar = sub / "OUTCAR"
        try:
            images = read(str(outcar), index=":")
        except Exception as e:
            print(f"Failed to read OUTCAR with ASE: {outcar} ({e})", file=sys.stderr)
            continue
        if not images:
            continue
        selected_idx = len(images) - 1
        if check_convergence:
            ok = per_step_electronically_converged(
                outcar, sub / "INCAR", sub / "OSZICAR"
            )
            if (
                ok is None
                or len(ok) != len(images)
            ):
                print(
                    f"Warning: convergence list invalid on re-read, skipping: {outcar}",
                    file=sys.stderr,
                )
                continue
            selected = _last_converged_index(ok)
            if selected is None:
                print(
                    f"Warning: no converged frame in trajectory, skipping: {outcar}",
                    file=sys.stderr,
                )
                continue
            selected_idx = selected
        atoms = images[selected_idx]
        atoms.info["run_id"] = make_run_id(top_path, outcar)
        frames.append(atoms)

    if not frames:
        print(f"No valid structures under {top_path}", file=sys.stderr)
        return []
    print(
        f"Collected {len(frames)} frames from {top_path} (min E = {min_energy:.6f} eV)"
    )
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
    parser.add_argument(
        "--no-convergence-check",
        action="store_true",
        help="Use final ionic step even if OSZICAR/OUTCAR suggest non-converged SCF",
    )
    args = parser.parse_args()

    # Log file: default to <output_stem>.log
    if args.log is None:
        args.log = os.path.splitext(args.output)[0] + ".log"

    tee = None
    try:
        tee = Tee(args.log)
        sys.stdout = tee

        check = not args.no_convergence_check
        all_frames = []
        for d in args.directories:
            all_frames.extend(
                process_directory(d, args.num_frames, check_convergence=check)
            )

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
