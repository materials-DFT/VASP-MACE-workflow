import argparse
import sys
from pathlib import Path

import numpy as np
from ase.io import iread, read

ML_TYPES = (
    ("MACE", "MACE_energy", "MACE_forces", "MACE_stress"),
    ("ALLEGRO", "ALLEGRO_energy", "ALLEGRO_forces", "ALLEGRO_stress"),
    ("UMA", "UMA_energy", "UMA_forces", "UMA_stress"),
)


def detect_mlip(info):
    for label, e_key, f_key, s_key in ML_TYPES:
        if e_key in info:
            return label, e_key, f_key, s_key
    return None


def is_eval_xyz(path):
    try:
        first = read(path, index=0)
    except Exception:
        return False
    if first is None:
        return False
    detected = detect_mlip(first.info)
    if detected is None:
        return False
    _, _, ml_forces_key, _ = detected
    return "REF_energy" in first.info and "REF_forces" in first.arrays and ml_forces_key in first.arrays


def discover_eval_xyzs(paths):
    xyzs = []
    for raw in paths:
        candidate = Path(raw).expanduser().resolve()
        if not candidate.exists():
            print(f"Warning: path does not exist, skipping: {candidate}")
            continue
        if candidate.is_file():
            if is_eval_xyz(candidate):
                xyzs.append(candidate)
            else:
                print(f"Warning: not an evaluation-format XYZ, skipping: {candidate}")
            continue
        for xyz_file in sorted(candidate.rglob("*.xyz")):
            if is_eval_xyz(xyz_file):
                xyzs.append(xyz_file)
    return sorted(set(xyzs))


def label_from_path(xyz_path):
    parts = xyz_path.parts
    if "npt_dataset" in parts:
        idx = parts.index("npt_dataset")
        tail = list(parts[idx + 1 : -1])
        dataset_id = tail[0] if tail else "unknown"
        eval_idx = tail.index("evaluation") if "evaluation" in tail else -1
        rel = tail[eval_idx + 1 :] if eval_idx >= 0 else tail[1:]
        rel_label = "/".join(rel) if rel else "evaluation"
        return f"npt_dataset/{dataset_id}: {rel_label}"
    return str(xyz_path.parent)


def compute_rmse(xyz_path, forced_ref_index=None):
    iterator = iread(str(xyz_path), index=":")
    first_frame = None
    try:
        first_frame = next(iterator)
    except StopIteration:
        raise RuntimeError(f"No frames in {xyz_path}")

    detected = detect_mlip(first_frame.info)
    if detected is None:
        raise RuntimeError(f"Could not detect MLIP type in {xyz_path}")
    ml_label, ml_energy_key, ml_forces_key, ml_stress_key = detected

    ml_energies, ref_energies = [], []
    n_atoms = []
    force_sqerr_sum = 0.0
    force_count = 0
    stress_sqerr_sum = 0.0
    stress_count = 0

    def process_frame(frame, i):
        if ml_energy_key not in frame.info or "REF_energy" not in frame.info:
            raise RuntimeError(
                f"Frame {i} in {xyz_path} missing '{ml_energy_key}' or 'REF_energy'."
            )
        if ml_forces_key not in frame.arrays or "REF_forces" not in frame.arrays:
            raise RuntimeError(
                f"Frame {i} in {xyz_path} missing '{ml_forces_key}' or 'REF_forces'."
            )
        ml_energies.append(frame.info[ml_energy_key])
        ref_energies.append(frame.info["REF_energy"])
        n_atoms.append(len(frame))
        ml_f = frame.arrays[ml_forces_key].ravel()
        ref_f = frame.arrays["REF_forces"].ravel()
        return ml_f, ref_f

    ml_f0, ref_f0 = process_frame(first_frame, 0)
    force_sqerr_sum += float(np.sum((ml_f0 - ref_f0) ** 2))
    force_count += int(ml_f0.size)
    if ml_stress_key in first_frame.info and "REF_stress" in first_frame.info:
        ml_s0 = np.asarray(first_frame.info[ml_stress_key]).ravel()
        ref_s0 = np.asarray(first_frame.info["REF_stress"]).ravel()
        stress_sqerr_sum += float(np.sum((ml_s0 - ref_s0) ** 2))
        stress_count += int(ml_s0.size)

    for i, frame in enumerate(iterator, start=1):
        ml_f, ref_f = process_frame(frame, i)
        force_sqerr_sum += float(np.sum((ml_f - ref_f) ** 2))
        force_count += int(ml_f.size)
        if ml_stress_key in frame.info and "REF_stress" in frame.info:
            ml_s = np.asarray(frame.info[ml_stress_key]).ravel()
            ref_s = np.asarray(frame.info["REF_stress"]).ravel()
            stress_sqerr_sum += float(np.sum((ml_s - ref_s) ** 2))
            stress_count += int(ml_s.size)

    ml_energies = np.asarray(ml_energies)
    ref_energies = np.asarray(ref_energies)
    n_atoms = np.asarray(n_atoms)
    has_stress = stress_count > 0

    ref_per_atom = ref_energies / n_atoms
    ml_per_atom = ml_energies / n_atoms

    if forced_ref_index is not None:
        if forced_ref_index < 0 or forced_ref_index >= len(ref_per_atom):
            raise RuntimeError(
                f"Invalid --ref-index {forced_ref_index} for {xyz_path}, "
                f"must be in [0, {len(ref_per_atom)-1}]"
            )
        ref_idx = forced_ref_index
    else:
        ref_idx = int(np.argmin(ref_per_atom))

    delta_ref_total = ref_energies - ref_energies[ref_idx]
    delta_ml_total = ml_energies - ml_energies[ref_idx]
    delta_ref_per_atom = ref_per_atom - ref_per_atom[ref_idx]
    delta_ml_per_atom = ml_per_atom - ml_per_atom[ref_idx]

    result = {
        "label": label_from_path(xyz_path),
        "path": xyz_path,
        "ml_label": ml_label,
        "ref_idx": ref_idx,
        "rmse_rel_total": float(np.sqrt(np.mean((delta_ml_total - delta_ref_total) ** 2))),
        "rmse_rel_per_atom": float(np.sqrt(np.mean((delta_ml_per_atom - delta_ref_per_atom) ** 2))),
        "rmse_forces": float(np.sqrt(force_sqerr_sum / force_count)),
        "has_stress": has_stress,
    }
    if has_stress:
        result["rmse_stress"] = float(np.sqrt(stress_sqerr_sum / stress_count))
    return result


def print_results_table(results):
    headers = [
        "Dataset",
        "MLIP",
        "RefIdx",
        "RMSE_dE_total(eV)",
        "RMSE_dE_atom(eV/atom)",
        "RMSE_forces(eV/A)",
        "RMSE_stress(eV/A^3)",
        "File",
    ]
    rows = []
    for r in results:
        rows.append(
            [
                r["label"],
                r["ml_label"],
                str(r["ref_idx"]),
                f"{r['rmse_rel_total']:.6f}",
                f"{r['rmse_rel_per_atom']:.6f}",
                f"{r['rmse_forces']:.6f}",
                f"{r['rmse_stress']:.6f}" if r["has_stress"] else "N/A",
                str(r["path"]),
            ]
        )

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(row):
        return " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt_row(row))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute RMSE metrics from one or more eval-format XYZs. Inputs can be "
            "files or directories (searched recursively)."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more XYZ files or directories containing eval-format XYZ files.",
    )
    parser.add_argument(
        "--ref-index",
        type=int,
        default=None,
        help=(
            "Index (0-based) used as relative-energy reference for each dataset. "
            "Default: per-dataset minimum REF energy/atom."
        ),
    )
    args = parser.parse_args()

    xyz_files = discover_eval_xyzs(args.paths)
    if not xyz_files:
        print("No evaluation-format XYZ files found in the provided inputs.")
        sys.exit(1)

    results = []
    for xyz in xyz_files:
        try:
            results.append(compute_rmse(xyz, forced_ref_index=args.ref_index))
        except Exception as exc:
            print(f"Warning: skipping {xyz}: {exc}")

    if not results:
        print("No valid datasets after loading.")
        sys.exit(1)

    print_results_table(results)


if __name__ == "__main__":
    main()
