#!/usr/bin/env python3
"""
Plot energies per atom and forces from an extended XYZ dataset.

Usage:
    python plot_xyz_energies_forces.py <path_to.xyz>

Reads the XYZ (extxyz) with ASE, detects energy and force keys from info/arrays,
then displays histograms and frame-wise plots via X11.
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")  # X11 display
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read


# Common keys used in extxyz for total energy (per structure) and per-atom forces
ENERGY_KEYS = (
    "energy",
    "free_energy",
    "REF_energy",
    "total_energy",
    "Energy",
    "potential_energy",
)
FORCE_KEYS = ("forces", "REF_forces", "force", "forces_dft", "ALLEGRO_forces")


def _energy_from_comment(comment):
    """Try to parse energy from comment string, e.g. '... E = -1335.637875 eV' or 'energy=-1335.63'."""
    if not comment:
        return None
    # "E = -1335.637875 eV" or "E=-1335.63"
    m = re.search(r"E\s*=\s*([-\d.]+)\s*(?:eV)?", comment, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"energy\s*=\s*([-\d.]+)", comment, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def find_energy_per_atom(atoms_list):
    """Extract energy per atom for each structure. Tries common info keys and comment fallback."""
    e_per_atom = []
    for atoms in atoms_list:
        n = len(atoms)
        if n == 0:
            continue
        total_energy = None
        for key in ENERGY_KEYS:
            if key in atoms.info:
                total_energy = float(atoms.info[key])
                break
        if total_energy is None:
            for k, v in atoms.info.items():
                if "energy" in k.lower() and not ("stress" in k.lower() or "per_atom" in k.lower()):
                    try:
                        total_energy = float(v)
                        break
                    except (TypeError, ValueError):
                        pass
        if total_energy is None and "comment" in atoms.info:
            total_energy = _energy_from_comment(str(atoms.info["comment"]))
        if total_energy is not None:
            e_per_atom.append(total_energy / n)
    return np.array(e_per_atom) if e_per_atom else np.array([])


def _parse_extxyz_forces(path):
    """
    Fallback: parse extxyz file manually when ASE doesn't put forces in arrays.
    Returns list of (n_atoms, 3) force arrays per frame, or [] if no forces in file.
    """
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        lines = [ln for ln in f]
    i = 0
    result = []
    while i < len(lines):
        try:
            n_atoms = int(lines[i].strip())
        except ValueError:
            break
        i += 1
        if i >= len(lines):
            break
        header = lines[i]
        i += 1
        # Parse Properties=... to find force column indices
        props_match = re.search(r"Properties\s*=\s*([^\s]+)", header)
        if not props_match:
            # skip this frame's atom lines and continue
            i += n_atoms
            continue
        props_str = props_match.group(1)
        # Split into name:type:n chunks (type is S or R, n is int)
        parts = props_str.split(":")
        col = 0
        force_start = None
        force_len = 3
        j = 0
        while j + 2 < len(parts):
            name, typ, n_str = parts[j], parts[j + 1], parts[j + 2]
            try:
                n_cols = int(n_str)
            except ValueError:
                j += 1
                continue
            if name in FORCE_KEYS and n_cols >= 3:
                force_start = col
                force_len = min(3, n_cols)
                break
            col += n_cols
            j += 3
        if force_start is None:
            i += n_atoms
            continue
        # Read atom lines: each has columns; forces are at force_start : force_start+3
        frame_forces = []
        for _ in range(n_atoms):
            if i >= len(lines):
                break
            toks = lines[i].split()
            i += 1
            if len(toks) < force_start + force_len:
                continue
            fx = float(toks[force_start])
            fy = float(toks[force_start + 1])
            fz = float(toks[force_start + 2])
            frame_forces.append([fx, fy, fz])
        if len(frame_forces) == n_atoms:
            result.append(np.array(frame_forces))
    return result


def find_force_magnitudes(atoms_list, xyz_path=None):
    """Extract force magnitudes (per atom) from all structures. Tries arrays first, then parses file."""
    all_mags = []
    for atoms in atoms_list:
        forces = None
        for key in FORCE_KEYS:
            if key in atoms.arrays:
                forces = np.asarray(atoms.arrays[key], dtype=float)
                break
        if forces is not None and forces.ndim >= 2:
            mags = np.linalg.norm(forces, axis=-1)
            all_mags.extend(mags.ravel().tolist())
    if all_mags:
        return np.array(all_mags)
    # Fallback: ASE didn't put forces in arrays (e.g. npt_dataset.xyz with magmoms)
    if xyz_path is not None:
        frames_forces = _parse_extxyz_forces(xyz_path)
        for forces in frames_forces:
            mags = np.linalg.norm(forces, axis=-1)
            all_mags.extend(mags.ravel().tolist())
    return np.array(all_mags) if all_mags else np.array([])


def main():
    parser = argparse.ArgumentParser(
        description="Plot energies per atom and forces from an extended XYZ file."
    )
    parser.add_argument(
        "xyz_path",
        type=Path,
        help="Path to the XYZ (or extxyz) dataset file",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default=None,
        help="ASE format (default: auto-detect from extension/content)",
    )
    args = parser.parse_args()

    xyz_path = args.xyz_path.resolve()
    if not xyz_path.exists():
        print(f"Error: file not found: {xyz_path}", file=sys.stderr)
        sys.exit(1)

    fmt = args.format or "extxyz"
    try:
        atoms_list = read(str(xyz_path), index=":", format=fmt)
    except Exception as e:
        try:
            atoms_list = read(str(xyz_path), index=":")
        except Exception as e2:
            print(f"Error reading XYZ: {e}\nFallback: {e2}", file=sys.stderr)
            sys.exit(1)

    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    n_frames = len(atoms_list)
    print(f"Read {n_frames} structure(s) from {xyz_path}")

    # Energy per atom
    e_per_atom = find_energy_per_atom(atoms_list)
    force_mags = find_force_magnitudes(atoms_list, xyz_path=xyz_path)

    if e_per_atom.size == 0:
        print("Warning: no energy data found in info.")
    if force_mags.size == 0:
        print("Warning: no force data found in arrays.")

    if e_per_atom.size == 0 and force_mags.size == 0:
        print("No energy or force data found. Check that your XYZ has info/arrays with standard keys.")
        sys.exit(1)

    # One figure, 2x2 subplots, single X11 window
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # (0,0): energy per atom histogram
    if e_per_atom.size > 0:
        ax = axes[0, 0]
        ax.hist(e_per_atom, bins=min(80, max(20, len(e_per_atom) // 5)), color="steelblue", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Energy per atom (eV)")
        ax.set_ylabel("Count")
        ax.set_title(f"Energy per atom (n={len(e_per_atom)})")
        ax.grid(True, alpha=0.3)
    else:
        axes[0, 0].axis("off")

    # (0,1): energy per atom vs frame
    if e_per_atom.size > 1:
        ax = axes[0, 1]
        ax.plot(np.arange(len(e_per_atom)), e_per_atom, "o-", markersize=3, color="steelblue", alpha=0.8)
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Energy per atom (eV)")
        ax.set_title("Energy per atom vs frame")
        ax.grid(True, alpha=0.3)
    else:
        axes[0, 1].axis("off")

    # (1,0): force magnitude histogram
    if force_mags.size > 0:
        ax = axes[1, 0]
        ax.hist(force_mags, bins=min(100, max(30, force_mags.size // 50)), color="coral", edgecolor="white", alpha=0.85)
        ax.set_xlabel("|Force| (eV/Å)")
        ax.set_ylabel("Count")
        ax.set_title(f"Force magnitude (n={force_mags.size} atoms)")
        ax.grid(True, alpha=0.3)
    else:
        axes[1, 0].axis("off")

    # (1,1): max force per frame
    max_f_per_frame = []
    if force_mags.size > 0 and n_frames > 1:
        for atoms in atoms_list:
            forces = None
            for key in FORCE_KEYS:
                if key in atoms.arrays:
                    forces = np.asarray(atoms.arrays[key], dtype=float)
                    break
            if forces is not None and forces.ndim >= 2:
                mags = np.linalg.norm(forces, axis=-1)
                max_f_per_frame.append(float(np.max(mags)))
        if not max_f_per_frame:
            # Forces came from file fallback; parse again for per-frame max
            frames_forces = _parse_extxyz_forces(xyz_path)
            for forces in frames_forces:
                mags = np.linalg.norm(forces, axis=-1)
                max_f_per_frame.append(float(np.max(mags)))
    if max_f_per_frame:
        ax = axes[1, 1]
        ax.plot(np.arange(len(max_f_per_frame)), max_f_per_frame, "o-", markersize=3, color="coral", alpha=0.8)
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Max |Force| (eV/Å)")
        ax.set_title("Max force per frame")
        ax.grid(True, alpha=0.3)
    else:
        axes[1, 1].axis("off")

    fig.suptitle(xyz_path.name, fontsize=11)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
