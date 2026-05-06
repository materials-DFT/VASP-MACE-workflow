import argparse
import concurrent.futures
import os
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
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
    """Short label: up to three parent directory names above the XYZ's folder (outer → inner)."""
    resolved = xyz_path.resolve()
    names = []
    cur = resolved.parent
    for _ in range(3):
        parent = cur.parent
        if parent == cur:
            break
        if parent.name:
            names.append(parent.name)
        cur = parent
    if not names:
        return str(resolved.parent)
    return "/".join(reversed(names))


def load_dataset(xyz_path, forced_ref_index=None):
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
    ref_forces_plot_chunks = []
    ml_forces_plot_chunks = []
    ref_stresses_plot_chunks = []
    ml_stresses_plot_chunks = []
    force_plot_points = 0
    stress_plot_points = 0

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

    def update_plot_buffers(
        ref_vals, ml_vals, ref_chunks, ml_chunks, points_so_far, max_points, rng
    ):
        if max_points is None:
            ref_chunks.append(ref_vals.copy())
            ml_chunks.append(ml_vals.copy())
            return points_so_far + ref_vals.size
        remaining = max_points - points_so_far
        if remaining <= 0:
            return points_so_far
        if ref_vals.size <= remaining:
            ref_chunks.append(ref_vals.copy())
            ml_chunks.append(ml_vals.copy())
            return points_so_far + ref_vals.size
        idx = rng.choice(ref_vals.size, size=remaining, replace=False)
        ref_chunks.append(ref_vals[idx].copy())
        ml_chunks.append(ml_vals[idx].copy())
        return points_so_far + remaining

    rng = np.random.default_rng(12345)
    max_force_points = 200000
    max_stress_points = 200000

    # Process first frame then remainder.
    ml_f0, ref_f0 = process_frame(first_frame, 0)
    force_sqerr_sum += float(np.sum((ml_f0 - ref_f0) ** 2))
    force_count += int(ml_f0.size)
    force_plot_points = update_plot_buffers(
        ref_f0,
        ml_f0,
        ref_forces_plot_chunks,
        ml_forces_plot_chunks,
        force_plot_points,
        max_force_points,
        rng,
    )
    if ml_stress_key in first_frame.info and "REF_stress" in first_frame.info:
        ml_s0 = np.asarray(first_frame.info[ml_stress_key]).ravel()
        ref_s0 = np.asarray(first_frame.info["REF_stress"]).ravel()
        stress_sqerr_sum += float(np.sum((ml_s0 - ref_s0) ** 2))
        stress_count += int(ml_s0.size)
        stress_plot_points = update_plot_buffers(
            ref_s0,
            ml_s0,
            ref_stresses_plot_chunks,
            ml_stresses_plot_chunks,
            stress_plot_points,
            max_stress_points,
            rng,
        )

    for i, frame in enumerate(iterator, start=1):
        ml_f, ref_f = process_frame(frame, i)
        force_sqerr_sum += float(np.sum((ml_f - ref_f) ** 2))
        force_count += int(ml_f.size)
        force_plot_points = update_plot_buffers(
            ref_f,
            ml_f,
            ref_forces_plot_chunks,
            ml_forces_plot_chunks,
            force_plot_points,
            max_force_points,
            rng,
        )
        if ml_stress_key in frame.info and "REF_stress" in frame.info:
            ml_s = np.asarray(frame.info[ml_stress_key]).ravel()
            ref_s = np.asarray(frame.info["REF_stress"]).ravel()
            stress_sqerr_sum += float(np.sum((ml_s - ref_s) ** 2))
            stress_count += int(ml_s.size)
            stress_plot_points = update_plot_buffers(
                ref_s,
                ml_s,
                ref_stresses_plot_chunks,
                ml_stresses_plot_chunks,
                stress_plot_points,
                max_stress_points,
                rng,
            )

    ml_energies = np.asarray(ml_energies)
    ref_energies = np.asarray(ref_energies)
    n_atoms = np.asarray(n_atoms)
    if ref_forces_plot_chunks:
        ref_forces_plot = np.concatenate(ref_forces_plot_chunks)
        ml_forces_plot = np.concatenate(ml_forces_plot_chunks)
    else:
        ref_forces_plot = np.empty(0, dtype=float)
        ml_forces_plot = np.empty(0, dtype=float)
    has_stress = stress_count > 0
    if has_stress:
        if ref_stresses_plot_chunks:
            ref_stresses_plot = np.concatenate(ref_stresses_plot_chunks)
            ml_stresses_plot = np.concatenate(ml_stresses_plot_chunks)
        else:
            ref_stresses_plot = np.empty(0, dtype=float)
            ml_stresses_plot = np.empty(0, dtype=float)

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

    data = {
        "xyz_path": xyz_path,
        "label": label_from_path(xyz_path),
        "ml_label": ml_label,
        "ref_idx": ref_idx,
        "delta_ref_total": delta_ref_total,
        "delta_ml_total": delta_ml_total,
        "delta_ref_per_atom": delta_ref_per_atom,
        "delta_ml_per_atom": delta_ml_per_atom,
        "ref_forces_plot": ref_forces_plot,
        "ml_forces_plot": ml_forces_plot,
        "has_stress": has_stress,
    }
    if has_stress:
        data["ref_stresses_plot"] = ref_stresses_plot
        data["ml_stresses_plot"] = ml_stresses_plot

    data["rmse_rel_total"] = float(np.sqrt(np.mean((delta_ml_total - delta_ref_total) ** 2)))
    data["rmse_rel_per_atom"] = float(
        np.sqrt(np.mean((delta_ml_per_atom - delta_ref_per_atom) ** 2))
    )
    data["rmse_forces"] = float(np.sqrt(force_sqerr_sum / force_count))
    if has_stress:
        data["rmse_stress"] = float(np.sqrt(stress_sqerr_sum / stress_count))
    return data


def global_lims(*arrays):
    vals = np.concatenate([np.asarray(a).ravel() for a in arrays if len(a) > 0])
    return [float(np.min(vals)), float(np.max(vals))]


def add_top_left_metrics(ax, lines):
    if not lines:
        return
    wrapped_lines = [textwrap.shorten(line, width=110, placeholder="...") for line in lines]
    ax.text(
        0.02,
        0.98,
        "\n".join(wrapped_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.5", boxstyle="round,pad=0.3"),
    )


def _load_dataset_wrapper(payload):
    xyz, ref_index = payload
    try:
        return load_dataset(xyz, forced_ref_index=ref_index), None
    except Exception as exc:
        return None, f"Warning: skipping {xyz}: {exc}"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot parity across one or more eval datasets. Inputs can be files or "
            "directories. Directories are searched recursively for eval-format XYZs."
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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of processes used to load datasets in parallel. "
            "Use >1 to speed up loading when multiple files are provided."
        ),
    )
    args = parser.parse_args()

    xyz_files = discover_eval_xyzs(args.paths)
    if not xyz_files:
        print("No evaluation-format XYZ files found in the provided inputs.")
        sys.exit(1)

    datasets = []
    workers = max(1, int(args.workers))
    if workers == 1 or len(xyz_files) == 1:
        for xyz in xyz_files:
            ds, warn = _load_dataset_wrapper((xyz, args.ref_index))
            if warn:
                print(warn)
                continue
            datasets.append(ds)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            payloads = [(xyz, args.ref_index) for xyz in xyz_files]
            for ds, warn in executor.map(_load_dataset_wrapper, payloads):
                if warn:
                    print(warn)
                    continue
                datasets.append(ds)

    if not datasets:
        print("No valid datasets to plot after loading.")
        sys.exit(1)

    ml_types = sorted({d["ml_label"] for d in datasets})
    has_stress_any = any(d["has_stress"] for d in datasets)
    has_stress_all = all(d["has_stress"] for d in datasets)
    include_stress = has_stress_any
    if has_stress_any and not has_stress_all:
        print("Note: some datasets have stress, some do not. Plotting stress where available.")

    ncols = 2 if include_stress else 3
    nrows = 2 if include_stress else 1
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.4 * ncols, 6.2 * nrows), constrained_layout=True
    )
    axes = np.atleast_2d(axes)
    colors = plt.cm.tab20(np.linspace(0, 1, max(2, len(datasets))))

    # Relative Total Energy
    ax = axes[0, 0]
    energy_lines = []
    for i, ds in enumerate(datasets):
        ax.scatter(
            ds["delta_ref_total"],
            ds["delta_ml_total"],
            alpha=0.45,
            s=14,
            color=colors[i],
        )
        energy_lines.append(f'{ds["label"]} | RMSE={ds["rmse_rel_total"]:.4f} eV')
    lims = global_lims(
        *(ds["delta_ref_total"] for ds in datasets), *(ds["delta_ml_total"] for ds in datasets)
    )
    ax.plot(lims, lims, "k--")
    ax.set_xlabel("ΔE (DFT-calculated) (eV)")
    ax.set_ylabel("ΔE (ML-predicted) (eV)")
    ax.set_title("Relative Total Energy Parity")
    ax.axis("equal")
    ax.grid(True)
    add_top_left_metrics(ax, energy_lines)

    # Force
    ax = axes[0, 1]
    force_lines = []
    for i, ds in enumerate(datasets):
        ax.scatter(
            ds["ref_forces_plot"],
            ds["ml_forces_plot"],
            alpha=0.40,
            s=10,
            color=colors[i],
        )
        force_lines.append(f'{ds["label"]} | RMSE={ds["rmse_forces"]:.4f} eV/A')
    lims = global_lims(
        *(ds["ref_forces_plot"] for ds in datasets), *(ds["ml_forces_plot"] for ds in datasets)
    )
    ax.plot(lims, lims, "k--")
    ax.set_xlabel("DFT-calculated Force (eV/A)")
    ax.set_ylabel("ML-predicted Force (eV/A)")
    ax.set_title("Force Parity")
    ax.axis("equal")
    ax.grid(True)
    add_top_left_metrics(ax, force_lines)

    # Relative Per-Atom Energy
    ax = axes[1, 0] if include_stress else axes[0, 2]
    per_atom_lines = []
    for i, ds in enumerate(datasets):
        ax.scatter(
            ds["delta_ref_per_atom"],
            ds["delta_ml_per_atom"],
            alpha=0.45,
            s=14,
            color=colors[i],
        )
        per_atom_lines.append(f'{ds["label"]} | RMSE={ds["rmse_rel_per_atom"]:.5f} eV/atom')
    lims = global_lims(
        *(ds["delta_ref_per_atom"] for ds in datasets),
        *(ds["delta_ml_per_atom"] for ds in datasets),
    )
    ax.plot(lims, lims, "k--")
    ax.set_xlabel("ΔE per atom (DFT-calculated) (eV)")
    ax.set_ylabel("ΔE per atom (ML-predicted) (eV)")
    ax.set_title("Relative Per-Atom Energy Parity")
    ax.axis("equal")
    ax.grid(True)
    add_top_left_metrics(ax, per_atom_lines)

    # Stress
    if include_stress:
        ax = axes[1, 1]
        stress_lines = []
        for i, ds in enumerate(datasets):
            if not ds["has_stress"]:
                continue
            ax.scatter(
                ds["ref_stresses_plot"],
                ds["ml_stresses_plot"],
                alpha=0.40,
                s=10,
                color=colors[i],
            )
            stress_lines.append(f'{ds["label"]} | RMSE={ds["rmse_stress"]:.5f} eV/A^3')
        stressable = [ds for ds in datasets if ds["has_stress"]]
        lims = global_lims(
            *(ds["ref_stresses_plot"] for ds in stressable),
            *(ds["ml_stresses_plot"] for ds in stressable),
        )
        ax.plot(lims, lims, "k--")
        ax.set_xlabel("DFT-calculated Stress (eV/A^3)")
        ax.set_ylabel("ML-predicted Stress (eV/A^3)")
        ax.set_title("Stress Parity (all components)")
        ax.axis("equal")
        ax.grid(True)
        add_top_left_metrics(ax, stress_lines)

    context_text = ", ".join(ds["label"] for ds in datasets)
    ml_text = "/".join(ml_types)
    wrapped_context = textwrap.fill(context_text, width=120)
    fig.suptitle(
        f"Parity plots for {len(datasets)} dataset(s) | MLIP: {ml_text}\n{wrapped_context}",
        fontsize=10,
    )

    for ds in datasets:
        print(f"\nDataset: {ds['label']}")
        print(f"  File: {ds['xyz_path']}")
        print(f"  MLIP: {ds['ml_label']}")
        print(f"  Reference frame index: {ds['ref_idx']}")
        print(f"  RMSE of ΔE (total):  {ds['rmse_rel_total']:.6f} eV")
        print(f"  RMSE of ΔE/atom:     {ds['rmse_rel_per_atom']:.6f} eV/atom")
        print(f"  RMSE of forces:      {ds['rmse_forces']:.6f} eV/A")
        if ds["has_stress"]:
            print(f"  RMSE of stress:      {ds['rmse_stress']:.6f} eV/A^3")
        else:
            print("  RMSE of stress:      N/A")

    plt.show()


if __name__ == "__main__":
    main()
