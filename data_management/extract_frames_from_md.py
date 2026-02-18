#!/usr/bin/env python3
# The following SBATCH directives are optional. They set job parameters when
# submitting this script directly with sbatch. The --output and --error
# directives are commented out so that only the internal log (e.g. md_two_frames.log)
# and xyz file are written.
#
#SBATCH --job-name=extract_md
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
##SBATCH --output=extract_md_%j.out
##SBATCH --error=extract_md_%j.err
"""
Extract 2 frames from each VASP MD OUTCAR and write to a single extended XYZ file.

Frames are taken at 1/3 and 2/3 of each trajectory (mid-two rule),
to sample from the production region and avoid equilibration/end effects.

Output is extended XYZ (extxyz) with energy, forces, stress, and lattice,
like extract_frames_for_mlff.py. The log file lists each OUTCAR, frame indices
extracted, and energies.

Requires: ase
Run interactively:  python extract_frames_from_md.py .
Run on cluster:     sbatch extract_frames_from_md.py .
(For sbatch, make script executable: chmod +x extract_frames_from_md.py)

Usage:
  python extract_frames_from_md.py [directory] [--out frames.xyz] [--md-dir .] [--log ...]
  (directory defaults to . if omitted, like extract_frames_for_mlff.py)
"""

import argparse
import sys
from pathlib import Path

from ase.io import read, write


# =============================================================================
# Logging (tee to stdout + log file)
# =============================================================================

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


def find_outcars(md_dir: Path):
    """Yield (outcar_path, run_id) for each OUTCAR under md_dir."""
    for outcar in sorted(md_dir.rglob("OUTCAR")):
        try:
            rel = outcar.relative_to(md_dir)
        except ValueError:
            rel = outcar
        run_id = str(rel.parent).replace("/", "_")
        yield outcar, run_id


def select_mid_two(atoms_list):
    """Return (indices, list of 2 ASE Atoms) at 1/3 and 2/3 of the trajectory."""
    n = len(atoms_list)
    if n == 0:
        return [], []
    if n == 1:
        return [0], [atoms_list[0], atoms_list[0]]
    i, j = n // 3, 2 * n // 3
    if i == j:
        j = min(j + 1, n - 1)
    return [i, j], [atoms_list[i], atoms_list[j]]


def run_extraction(md_dir: Path, out_path: Path):
    """Extract 2 mid-trajectory frames per OUTCAR; write extended XYZ (energy, forces, stress) and log details."""
    md_dir = md_dir.resolve()
    if not md_dir.is_dir():
        raise SystemExit(f"Not a directory: {md_dir}")

    print("=" * 80)
    print("MD frame extraction (2 frames per OUTCAR: 1/3 and 2/3 of trajectory)")
    print("=" * 80)
    print(f"Search directory: {md_dir}")
    print(f"Output: {out_path} (extended XYZ: energy, forces, stress)")
    print()

    outcars_list = list(find_outcars(md_dir))
    if not outcars_list:
        raise SystemExit("No OUTCAR files found under --md-dir.")

    print(f"Found {len(outcars_list)} OUTCAR(s)")
    print()

    all_atoms = []
    processed = 0
    skipped = 0

    for idx, (outcar, run_id) in enumerate(outcars_list):
        try:
            traj = read(outcar, index=":")
        except Exception as e:
            print(f"[{idx + 1}/{len(outcars_list)}] Skip {outcar}: {e}")
            skipped += 1
            continue
        if not isinstance(traj, list):
            traj = [traj]
        n_frames = len(traj)
        indices, two = select_mid_two(traj)
        if not two:
            print(f"[{idx + 1}/{len(outcars_list)}] Skip {outcar}: no frames")
            skipped += 1
            continue

        # Log: path, frame count, exact indices, energies
        try:
            rel_path = outcar.relative_to(md_dir)
        except ValueError:
            rel_path = outcar
        try:
            e0 = two[0].get_potential_energy()
            e1 = two[1].get_potential_energy()
            e_str = f" E = {e0:.4f}, {e1:.4f} eV"
        except (AttributeError, RuntimeError):
            e_str = ""
        print(f"[{idx + 1}/{len(outcars_list)}] {rel_path}")
        print(f"  Frames: {n_frames}  ->  extracting indices {indices[0]}, {indices[1]} (1/3, 2/3){e_str}")
        print(f"  run_id: {run_id}")
        print()

        for a in two:
            a.info["run_id"] = run_id
            all_atoms.append(a)
        processed += 1

    if not all_atoms:
        raise SystemExit("No frames collected. Check that OUTCARs exist under --md-dir.")

    print("=" * 80)
    print("Writing output...")
    write(out_path, all_atoms, format="extxyz")
    print(f"✓ Written {len(all_atoms)} frames from {processed} calculations to {out_path}")
    print()
    print("Extraction statistics:")
    print(f"  OUTCARs found:    {len(outcars_list)}")
    print(f"  Processed:        {processed}")
    print(f"  Skipped:          {skipped}")
    print(f"  Total frames:     {len(all_atoms)}")
    print()


def main():
    p = argparse.ArgumentParser(description="Extract 2 mid-trajectory frames per OUTCAR to XYZ")
    p.add_argument(
        "base_path",
        nargs="?",
        default=".",
        type=Path,
        help="Base directory to work in (default: .)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("md_two_frames.xyz"),
        help="Output XYZ file (default: md_two_frames.xyz)",
    )
    p.add_argument(
        "--md-dir",
        type=Path,
        default=Path("."),
        help="Subdir under base_path to search for OUTCARs; . means base_path itself (default: .)",
    )
    p.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Log file path (default: <output_stem>.log)",
    )
    args = p.parse_args()

    work_dir = Path(args.base_path).resolve()
    if not work_dir.is_dir():
        raise SystemExit(f"Not a directory: {work_dir}")
    md_dir = work_dir if args.md_dir == Path(".") else work_dir / args.md_dir

    log_path = args.log
    if log_path is None:
        log_path = args.out.with_suffix(".log")

    tee = None
    try:
        tee = Tee(log_path)
        sys.stdout = tee
        run_extraction(md_dir, args.out)
    finally:
        if tee is not None:
            sys.stdout = tee._stream
            tee.close()
            print(f"Log written to {log_path}")


if __name__ == "__main__":
    main()
