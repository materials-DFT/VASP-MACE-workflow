#!/usr/bin/env python3
# The following SBATCH directives are optional. They set job parameters when
# submitting this script directly with sbatch. Route Slurm stdout/stderr to
# /dev/null so only the internal log (e.g. md_two_frames.log) and xyz file are
# written in the working directory.
#
#SBATCH --job-name=extract_md
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
"""
Extract 2 frames from each VASP MD OUTCAR and write to a single extended XYZ file.

Frames are taken at 1/3 and 2/3 of each trajectory (mid-two rule),
to sample from the production region and avoid equilibration/end effects.

Output is extended XYZ (extxyz) with energy, forces, stress, and lattice,
like extract_frames_for_mlff.py. The log file lists each OUTCAR, frame indices
extracted, and energies.

By default, only electronically converged MD steps are used: per-step NELM
(oszicar/INCAR) and OUTCAR chunk failure messages (see --no-convergence-check
to disable). If the exact 1/3 or 2/3 target step is not converged, the script
searches forward to the next converged step for that target.

Requires: ase
Run interactively:  python extract_frames_from_md.py .
Run on cluster:     sbatch extract_frames_from_md.py .
(For sbatch, make script executable: chmod +x extract_frames_from_md.py)

Usage:
  python extract_frames_from_md.py [directory] [--out frames.xyz] [--md-dir .] [--log ...] [--no-convergence-check]
  (directory defaults to . if omitted, like extract_frames_for_mlff.py)
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

from ase.io import iread, write

def _load_per_step_convergence():
    """
    Import helper from local module, including sbatch spool fallback paths.

    Under `sbatch script.py`, Slurm executes a copied script from /var/spool,
    so sibling imports are not discoverable unless we probe explicit locations.
    """
    try:
        from vasp_step_convergence import per_step_electronically_converged as func
        return func
    except ModuleNotFoundError:
        candidates = [
            Path.cwd() / "vasp_step_convergence.py",
            Path(__file__).resolve().parent / "vasp_step_convergence.py",
            Path(os.environ.get("SLURM_SUBMIT_DIR", "")) / "vasp_step_convergence.py",
            Path.home() / "VASP-MACE-workflow" / "data_management" / "vasp_step_convergence.py",
        ]
        for module_path in candidates:
            if not module_path.is_file():
                continue
            spec = importlib.util.spec_from_file_location(
                "vasp_step_convergence", module_path
            )
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "per_step_electronically_converged"):
                return module.per_step_electronically_converged
        raise


per_step_electronically_converged = _load_per_step_convergence()

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
    home = str(Path.home())
    for outcar in sorted(md_dir.rglob("OUTCAR")):
        abs_dir = str(outcar.resolve().parent)
        if abs_dir.startswith(home):
            run_id = "~" + abs_dir[len(home):]
        else:
            run_id = abs_dir
        yield outcar, run_id


def select_mid_two(atoms_list):
    """Return (indices, list of 2 ASE Atoms) at 1/3 and 2/3 of the trajectory."""
    n = len(atoms_list)
    if n == 0:
        return [], []
    if n == 1:
        return [0, 0], [atoms_list[0], atoms_list[0]]
    i, j = n // 3, 2 * n // 3
    if i == j:
        j = min(j + 1, n - 1)
    return [i, j], [atoms_list[i], atoms_list[j]]


def count_frames(outcar: Path) -> int:
    """Count frames in an OUTCAR trajectory without storing them all."""
    n = 0
    for _ in iread(outcar, index=":"):
        n += 1
    return n


def read_two_frames(outcar: Path, i: int, j: int):
    """Read frames i and j from OUTCAR using a streaming iterator."""
    want = {i, j}
    got = {}
    for k, a in enumerate(iread(outcar, index=":")):
        if k in want:
            got[k] = a
            if len(got) == len(want):
                break
    return [got[i], got[j]] if i in got and j in got else []


def find_next_converged(ok_list, start: int, min_index: int = 0):
    """Return first converged frame index >= max(start, min_index), else None."""
    begin = max(start, min_index)
    for k in range(begin, len(ok_list)):
        if ok_list[k]:
            return k
    return None


def run_extraction(md_dir: Path, out_path: Path, check_convergence: bool = True):
    """Extract 2 mid-trajectory frames per OUTCAR; write extended XYZ (energy, forces, stress) and log details."""
    md_dir = md_dir.resolve()
    if not md_dir.is_dir():
        raise SystemExit(f"Not a directory: {md_dir}")

    print("=" * 80)
    print("MD frame extraction (2 frames per OUTCAR: 1/3 and 2/3 of trajectory)")
    print("=" * 80)
    print(f"Search directory: {md_dir}")
    print(f"Output: {out_path} (extended XYZ: energy, forces, stress)")
    if check_convergence:
        print(
            "Convergence filter: ON (OSZICAR/NELM + OUTCAR per-step; "
            "for 1/3 and 2/3 targets, scan forward to next converged frame)"
        )
    else:
        print("Convergence filter: OFF")
    print()

    outcars_list = list(find_outcars(md_dir))
    if not outcars_list:
        raise SystemExit("No OUTCAR files found under --md-dir.")

    print(f"Found {len(outcars_list)} OUTCAR(s)")
    print()

    processed = 0
    skipped = 0
    skipped_noconv = 0
    frames_written = 0
    first_write = True

    # Avoid accumulating all frames in memory (and avoid mixing with stale outputs).
    try:
        out_path.unlink()
    except FileNotFoundError:
        pass

    for idx, (outcar, run_id) in enumerate(outcars_list):
        if check_convergence:
            sim = outcar.parent
            ok_list = per_step_electronically_converged(
                outcar, sim / "INCAR", sim / "OSZICAR"
            )
            if ok_list is None:
                print(
                    f"[{idx + 1}/{len(outcars_list)}] Skip {outcar}: could not parse "
                    "per-step convergence (OUTCAR chunk read failed)"
                )
                skipped += 1
                continue
            n_frames = len(ok_list)
        else:
            try:
                n_frames = count_frames(outcar)
            except Exception as e:
                print(f"[{idx + 1}/{len(outcars_list)}] Skip {outcar}: {e}")
                skipped += 1
                continue

        if n_frames <= 0:
            print(f"[{idx + 1}/{len(outcars_list)}] Skip {outcar}: no frames")
            skipped += 1
            continue

        i, j = n_frames // 3, 2 * n_frames // 3
        if i == j:
            j = min(j + 1, n_frames - 1)

        selected_i, selected_j = i, j
        if check_convergence:
            # Preserve the mid-trajectory intent while avoiding unconverged steps:
            # move each target forward to the next converged frame.
            selected_i = find_next_converged(ok_list, i)
            if selected_i is None:
                print(
                    f"[{idx + 1}/{len(outcars_list)}] Skip {outcar}: no converged frame "
                    f"at or after target index {i} (1/3)"
                )
                skipped_noconv += 1
                skipped += 1
                continue

            selected_j = find_next_converged(ok_list, j, min_index=selected_i + 1)
            if selected_j is None:
                print(
                    f"[{idx + 1}/{len(outcars_list)}] Skip {outcar}: no second converged "
                    f"frame at or after target index {j} (2/3) beyond first index {selected_i}"
                )
                skipped_noconv += 1
                skipped += 1
                continue

        try:
            two = read_two_frames(outcar, selected_i, selected_j)
        except Exception as e:
            print(f"[{idx + 1}/{len(outcars_list)}] Skip {outcar}: {e}")
            skipped += 1
            continue
        if not two:
            print(f"[{idx + 1}/{len(outcars_list)}] Skip {outcar}: could not read frames")
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
        if check_convergence:
            print(
                f"  Frames: {n_frames}  ->  targets {i}, {j}; extracting converged "
                f"indices {selected_i}, {selected_j}{e_str}"
            )
        else:
            print(f"  Frames: {n_frames}  ->  extracting indices {i}, {j} (1/3, 2/3){e_str}")
        print(f"  run_id: {run_id}")
        print()

        for a in two:
            a.info["run_id"] = run_id
            write(out_path, a, format="extxyz", append=(not first_write))
            first_write = False
            frames_written += 1
        processed += 1

    if frames_written == 0:
        raise SystemExit("No frames collected. Check that OUTCARs exist under --md-dir.")

    print("=" * 80)
    print("Writing output...")
    print(f"✓ Written {frames_written} frames from {processed} calculations to {out_path}")
    print()
    print("Extraction statistics:")
    print(f"  OUTCARs found:    {len(outcars_list)}")
    print(f"  Processed:        {processed}")
    print(f"  Skipped:          {skipped}")
    if check_convergence and skipped_noconv:
        print(f"  Skipped (not converged at 1/3 or 2/3): {skipped_noconv}")
    print(f"  Total frames:     {frames_written}")
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
    p.add_argument(
        "--no-convergence-check",
        action="store_true",
        help="Extract 1/3 and 2/3 frames even when OSZICAR/OUTCAR suggest non-converged SCF",
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
        run_extraction(md_dir, args.out, check_convergence=not args.no_convergence_check)
    finally:
        if tee is not None:
            sys.stdout = tee._stream
            tee.close()
            print(f"Log written to {log_path}")


if __name__ == "__main__":
    main()
