#!/usr/bin/env python3
"""
Identify structures where ALLEGRO (or MACE) performs poorly, using the same
logic as parity_plot_per_atom.py: relative per-atom energy, force parity, stress parity.
Reads an XYZ with REF_* and ALLEGRO_* (or MACE_*) in frame.info/arrays.
Prints analysis results to stdout; does not write any files.
"""
import numpy as np
from ase.io import read
import argparse
import json
import sys


def parse_stress(val):
    """Parse stress from frame.info; may be array or '_JSON [...]' string."""
    if isinstance(val, (list, np.ndarray)):
        return np.asarray(val).flatten()
    if isinstance(val, str) and val.strip().startswith("_JSON"):
        return np.array(json.loads(val.replace("_JSON ", ""))).flatten()
    return np.asarray(val).flatten()


def main():
    parser = argparse.ArgumentParser(
        description="Identify structures with largest ALLEGRO vs REF errors (relative E, forces, stress)."
    )
    parser.add_argument("xyz_file", type=str, help="Path to .xyz with REF_* and ALLEGRO_* (or MACE_*)")
    parser.add_argument("--ref-index", type=int, default=None, help="Reference frame index for relative energy (default: lowest REF E/atom)")
    parser.add_argument("--top-n", type=int, default=15, help="Number of worst structures to report per metric (default: 15)")
    args = parser.parse_args()

    frames = read(args.xyz_file, index=":")
    if not frames:
        print(f"No frames in {args.xyz_file}", file=sys.stderr)
        sys.exit(1)

    first_info = frames[0].info
    if "ALLEGRO_energy" in first_info:
        ml_energy_key = "ALLEGRO_energy"
        ml_forces_key = "ALLEGRO_forces"
        ml_stress_key = "ALLEGRO_stress"
        ml_label = "ALLEGRO"
    elif "MACE_energy" in first_info:
        ml_energy_key = "MACE_energy"
        ml_forces_key = "MACE_forces"
        ml_stress_key = "MACE_stress"
        ml_label = "MACE"
    else:
        print("No ALLEGRO_energy or MACE_energy in frame.info.", file=sys.stderr)
        sys.exit(1)
    if "REF_energy" not in first_info:
        print("No REF_energy in frame.info.", file=sys.stderr)
        sys.exit(1)

    n_frames = len(frames)
    ml_energies = np.array([f.info[ml_energy_key] for f in frames])
    ref_energies = np.array([f.info["REF_energy"] for f in frames])
    n_atoms_list = np.array([len(f) for f in frames])

    ref_per_atom = ref_energies / n_atoms_list
    ml_per_atom = ml_energies / n_atoms_list

    ref_idx = args.ref_index if args.ref_index is not None else int(np.argmin(ref_per_atom))
    delta_ref_per_atom = ref_per_atom - ref_per_atom[ref_idx]
    delta_ml_per_atom = ml_per_atom - ml_per_atom[ref_idx]
    error_rel_per_atom = np.abs(delta_ml_per_atom - delta_ref_per_atom)

    # Per-structure force RMSE
    force_rmse_per_frame = np.zeros(n_frames)
    for i, frame in enumerate(frames):
        ref_f = frame.arrays["REF_forces"].flatten()
        ml_f = frame.arrays[ml_forces_key].flatten()
        force_rmse_per_frame[i] = np.sqrt(np.mean((ml_f - ref_f) ** 2))

    has_stress = ml_stress_key in first_info and "REF_stress" in first_info
    stress_rmse_per_frame = None
    if has_stress:
        stress_rmse_per_frame = np.zeros(n_frames)
        for i, frame in enumerate(frames):
            try:
                ref_s = parse_stress(frame.info["REF_stress"])
                ml_s = parse_stress(frame.info[ml_stress_key])
                stress_rmse_per_frame[i] = np.sqrt(np.mean((np.asarray(ml_s) - np.asarray(ref_s)) ** 2))
            except Exception as e:
                stress_rmse_per_frame[i] = np.nan

    # run_id or frame index as identifier
    def get_run_id(frame, idx):
        return frame.info.get("run_id", f"frame_{idx}")

    run_ids = [get_run_id(frames[i], i) for i in range(n_frames)]

    # Sort by each error metric (descending = worst first)
    order_rel_e = np.argsort(error_rel_per_atom)[::-1]
    order_force = np.argsort(force_rmse_per_frame)[::-1]
    order_stress = np.argsort(stress_rmse_per_frame)[::-1] if has_stress and stress_rmse_per_frame is not None else None

    n = args.top_n
    print(f"MLIP: {ml_label}  |  Reference for ΔE: frame index {ref_idx} ({run_ids[ref_idx]})")
    print(f"Total structures: {n_frames}")
    print()

    print("=" * 80)
    print("WORST STRUCTURES BY RELATIVE PER-ATOM ENERGY ERROR (eV/atom)")
    print("=" * 80)
    for rank, idx in enumerate(order_rel_e[:n], 1):
        rid = run_ids[idx]
        err = error_rel_per_atom[idx]
        delta_ref = delta_ref_per_atom[idx]
        delta_ml = delta_ml_per_atom[idx]
        print(f"  {rank:2d}. {rid}")
        print(f"      error = {err:.6f} eV/atom   (REF ΔE/atom = {delta_ref:.6f}, ML ΔE/atom = {delta_ml:.6f})")
    print()

    print("=" * 80)
    print("WORST STRUCTURES BY FORCE RMSE (eV/Å)")
    print("=" * 80)
    for rank, idx in enumerate(order_force[:n], 1):
        rid = run_ids[idx]
        rmse = force_rmse_per_frame[idx]
        print(f"  {rank:2d}. {rid}")
        print(f"      force RMSE = {rmse:.6f} eV/Å")
    print()

    if has_stress and order_stress is not None:
        print("=" * 80)
        print("WORST STRUCTURES BY STRESS RMSE (eV/Å³)")
        print("=" * 80)
        valid = ~np.isnan(stress_rmse_per_frame)
        if np.any(valid):
            for rank, idx in enumerate(order_stress[:n], 1):
                if np.isnan(stress_rmse_per_frame[idx]):
                    continue
                rid = run_ids[idx]
                rmse = stress_rmse_per_frame[idx]
                print(f"  {rank:2d}. {rid}")
                print(f"      stress RMSE = {rmse:.6f} eV/Å³")
        else:
            print("  (No valid stress data)")
        print()


if __name__ == "__main__":
    main()
