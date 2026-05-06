#!/usr/bin/env python3
"""Plot stress distributions from XYZ files grouped by input directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import iread


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively find .xyz files under provided directories, parse per-frame "
            "stress values, and plot stress distributions by directory."
        )
    )
    parser.add_argument(
        "directories",
        nargs="+",
        help="One or more directories to scan for .xyz files",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output image path. If omitted, no file is written.",
    )
    parser.add_argument(
        "--component",
        choices=["hydrostatic", "xx", "yy", "zz", "xy", "xz", "yz"],
        default="hydrostatic",
        help=(
            "Stress component to plot. hydrostatic is the mean of sigma_xx, sigma_yy, "
            "sigma_zz."
        ),
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=60,
        help="Number of histogram bins (default: 60)",
    )
    parser.add_argument(
        "--format",
        choices=["xyz", "vasp"],
        default="xyz",
        help="Input file format. xyz is recommended (default: xyz).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display an interactive X11 plot window.",
    )
    return parser.parse_args()


def collect_input_files(directories: list[Path], file_format: str) -> list[Path]:
    pattern = "*.xyz" if file_format == "xyz" else "OUTCAR"
    files: list[Path] = []
    for directory in directories:
        files.extend(p for p in directory.rglob(pattern) if p.is_file())
    return sorted(files, key=lambda p: str(p.resolve()))


def _as_stress_vector(stress_raw: object) -> np.ndarray | None:
    """Convert stress-like payload to Voigt vector [xx, yy, zz, yz, xz, xy]."""
    arr = np.asarray(stress_raw, dtype=float)
    if arr.shape == (6,):
        return arr
    if arr.shape == (3, 3):
        return np.array(
            [
                arr[0, 0],
                arr[1, 1],
                arr[2, 2],
                arr[1, 2],
                arr[0, 2],
                arr[0, 1],
            ],
            dtype=float,
        )
    if arr.shape == (9,):
        mat = arr.reshape((3, 3))
        return np.array(
            [
                mat[0, 0],
                mat[1, 1],
                mat[2, 2],
                mat[1, 2],
                mat[0, 2],
                mat[0, 1],
            ],
            dtype=float,
        )
    return None


def extract_stress_component(stress_voigt: np.ndarray, component: str) -> float:
    if component == "hydrostatic":
        return float(np.mean(stress_voigt[:3]))

    index_map = {
        "xx": 0,
        "yy": 1,
        "zz": 2,
        "yz": 3,
        "xz": 4,
        "xy": 5,
    }
    return float(stress_voigt[index_map[component]])


def parse_frame_stress(frame, component: str) -> float | None:
    # extxyz commonly stores stress in frame.info["stress"].
    if "stress" in frame.info:
        stress_voigt = _as_stress_vector(frame.info["stress"])
        if stress_voigt is not None:
            return extract_stress_component(stress_voigt, component)

    # Fallback: if virial exists, convert to stress = -virial / volume.
    if "virial" in frame.info:
        virial = np.asarray(frame.info["virial"], dtype=float)
        vol = float(frame.get_volume())
        if vol > 0:
            if virial.shape == (9,):
                virial = virial.reshape((3, 3))
            if virial.shape == (3, 3):
                stress_tensor = -virial / vol
                stress_voigt = _as_stress_vector(stress_tensor)
                if stress_voigt is not None:
                    return extract_stress_component(stress_voigt, component)

    # ASE extxyz often stores stress in calculator results, not frame.info.
    if getattr(frame, "calc", None) is not None:
        calc_results = getattr(frame.calc, "results", {})
        if "stress" in calc_results:
            stress_voigt = _as_stress_vector(calc_results["stress"])
            if stress_voigt is not None:
                return extract_stress_component(stress_voigt, component)
        try:
            stress_voigt = _as_stress_vector(frame.get_stress(voigt=True))
            if stress_voigt is not None:
                return extract_stress_component(stress_voigt, component)
        except Exception:
            pass

    return None


def read_stress_values(path: Path, component: str, file_format: str) -> list[float]:
    values: list[float] = []
    formats_to_try = [file_format]
    if file_format == "xyz":
        # Most ML/ASE trajectories with metadata are extxyz, even with .xyz suffix.
        formats_to_try = ["extxyz", "xyz"]

    for fmt in formats_to_try:
        try:
            for frame in iread(str(path), index=":", format=fmt):
                val = parse_frame_stress(frame, component)
                if val is not None:
                    values.append(val)
            if values:
                break
        except Exception:
            continue
    return values


def main() -> int:
    args = parse_args()
    directories = [Path(d).expanduser().resolve() for d in args.directories]
    output_path = (
        Path(args.output).expanduser().resolve() if args.output is not None else None
    )

    bad_dirs = [d for d in directories if not d.is_dir()]
    if bad_dirs:
        for bad in bad_dirs:
            print(f"Error: not a directory: {bad}", file=sys.stderr)
        return 1

    files = collect_input_files(directories, args.format)
    if not files:
        print("No matching files found in provided directories.", file=sys.stderr)
        return 1

    values_by_directory: dict[Path, list[float]] = {d: [] for d in directories}
    file_counts: dict[Path, int] = {d: 0 for d in directories}
    skipped_files = 0

    for file_path in files:
        owner = next((d for d in directories if file_path.is_relative_to(d)), None)
        if owner is None:
            continue
        file_values = read_stress_values(file_path, args.component, args.format)
        if file_values:
            values_by_directory[owner].extend(file_values)
            file_counts[owner] += 1
        else:
            skipped_files += 1

    non_empty = {d: v for d, v in values_by_directory.items() if v}
    if not non_empty:
        print(
            "No stress values found. Ensure files contain per-frame stress metadata.",
            file=sys.stderr,
        )
        return 1

    all_values = np.concatenate([np.asarray(v, dtype=float) for v in non_empty.values()])
    global_min = float(np.min(all_values))
    global_max = float(np.max(all_values))
    if np.isclose(global_min, global_max):
        # Avoid zero-width histogram bins for near-constant datasets.
        pad = 1e-12 if global_min == 0.0 else abs(global_min) * 1e-6
        global_min -= pad
        global_max += pad
    bin_edges = np.linspace(global_min, global_max, args.bins + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for directory, values in non_empty.items():
        label = f"{directory.name} (n={len(values)})"
        ax.hist(
            values,
            bins=bin_edges,
            alpha=0.35,
            density=True,
            label=label,
            histtype="stepfilled",
        )

    ax.set_title(f"Stress distribution by directory ({args.component})")
    ax.set_xlabel("Stress")
    ax.set_ylabel("Probability density")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=200)
        print(f"Wrote plot to {output_path}")
    for d, values in non_empty.items():
        print(
            f"- {d}: {len(values)} stresses from {file_counts[d]} file(s)"
        )
    if skipped_files:
        print(f"Skipped {skipped_files} file(s) without readable stress metadata")

    if args.no_show:
        if output_path is None:
            print(
                "Error: nothing to do. Provide --output and/or omit --no-show.",
                file=sys.stderr,
            )
            return 1
        plt.close(fig)
        return 0

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
