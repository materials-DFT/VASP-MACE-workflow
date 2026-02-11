#!/usr/bin/env python3
"""
Scan specified directories for VASP OUTCAR/CONTCAR, find the lowest-energy
frame per directory, then extract the 10 frames closest in energy to that
minimum. All frames from all directories are written into a single multi-frame
XYZ file.
"""

import re
import sys
import argparse
from pathlib import Path


def get_final_energy_from_outcar(outcar_path):
    """Extract final TOTEN (eV) from OUTCAR. Returns None if not found."""
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


def parse_contcar(contcar_path):
    """
    Parse VASP CONTCAR. Returns list of (species, x, y, z) in Cartesian Angstrom,
    or None on error. Handles Direct/Cartesian and optional Selective dynamics.
    """
    try:
        with open(contcar_path, "r") as f:
            lines = [l.rstrip() for l in f.readlines()]
    except OSError:
        return None

    if len(lines) < 9:
        return None

    try:
        scale = float(lines[1].split()[0])
        lattice = [
            [float(x) for x in lines[2].split()],
            [float(x) for x in lines[3].split()],
            [float(x) for x in lines[4].split()],
        ]
        symbols = lines[5].split()
        counts = [int(x) for x in lines[6].split()]
        if len(symbols) != len(counts):
            return None
        n_atoms = sum(counts)
        coord_start = 8
        if lines[7].strip().lower().startswith("selective"):
            coord_start = 9
        mode = lines[coord_start - 1].strip().lower()
        if "direct" in mode:
            fractional = True
        else:
            fractional = True if "cartesian" not in mode else False

        species_list = []
        for sym, cnt in zip(symbols, counts):
            species_list.extend([sym] * cnt)

        coords = []
        for i in range(n_atoms):
            parts = lines[coord_start + i].split()
            if len(parts) < 3:
                return None
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            if fractional:
                # Convert fractional to Cartesian: r_cart = scale * (a*x + b*y + c*z)
                cx = scale * (lattice[0][0] * x + lattice[1][0] * y + lattice[2][0] * z)
                cy = scale * (lattice[0][1] * x + lattice[1][1] * y + lattice[2][1] * z)
                cz = scale * (lattice[0][2] * x + lattice[1][2] * y + lattice[2][2] * z)
                coords.append((species_list[i], cx, cy, cz))
            else:
                coords.append((species_list[i], scale * x, scale * y, scale * z))

        return coords
    except (IndexError, ValueError):
        return None


def write_xyz_frames(filepath, frames):
    """Write multiple structures to a single multi-frame XYZ file."""
    with open(filepath, "w") as f:
        for comment, atoms in frames:
            n = len(atoms)
            f.write(f"{n}\n")
            f.write(f"{comment}\n")
            for spec, x, y, z in atoms:
                f.write(f"{spec} {x:.10f} {y:.10f} {z:.10f}\n")


def _collect_candidates(top_path):
    """Return list of (subdir_path, energy) for direct children with OUTCAR/CONTCAR."""
    candidates = []
    for sub in sorted(top_path.iterdir()):
        if not sub.is_dir():
            continue
        outcar = sub / "OUTCAR"
        contcar = sub / "CONTCAR"
        if not outcar.is_file() or not contcar.is_file():
            continue
        energy = get_final_energy_from_outcar(outcar)
        if energy is None:
            continue
        candidates.append((sub, energy))
    return candidates


def process_directory(top_dir, num_frames=10):
    """
    For one directory (e.g. gamma/), find subdirs with OUTCAR/CONTCAR,
    get final energy from each OUTCAR, find minimum, then the num_frames
    subdirs closest in energy. Return their CONTCARs as a list of (comment, atoms) frames.
    If direct children have no OUTCAR/CONTCAR but their children do (e.g.
    isif3/gamma/V_+00.5%), process each direct child and aggregate returned frames.
    """
    top_path = Path(top_dir).resolve()
    if not top_path.is_dir():
        print(f"Not a directory: {top_path}", file=sys.stderr)
        return []

    candidates = _collect_candidates(top_path)

    # No OUTCAR/CONTCAR in direct children: try one level deeper (e.g. isif3 -> gamma -> V_+00.5%)
    if not candidates:
        all_frames = []
        subdirs = sorted([p for p in top_path.iterdir() if p.is_dir()])
        for child in subdirs:
            all_frames.extend(process_directory(child, num_frames))
        return all_frames

    # Minimum energy
    min_energy = min(e for _, e in candidates)
    # Sort by distance from minimum energy, then take num_frames
    by_dist = sorted(candidates, key=lambda x: abs(x[1] - min_energy))
    selected = by_dist[:num_frames]

    # Build frames from CONTCARs
    frames = []
    for sub, en in selected:
        atoms = parse_contcar(sub / "CONTCAR")
        if atoms is None:
            print(f"Failed to parse CONTCAR: {sub / 'CONTCAR'}", file=sys.stderr)
            continue
        comment = f"{sub.name}  E = {en:.6f} eV"
        frames.append((comment, atoms))

    if not frames:
        print(f"No valid CONTCARs among selected frames under {top_path}", file=sys.stderr)
        return []

    print(f"Collected {len(frames)} frames from {top_path} (min E = {min_energy:.6f} eV)")
    return frames


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames closest to minimum energy per directory into a single multi-frame XYZ file"
    )
    parser.add_argument(
        "directories",
        nargs="+",
        help="Directories to scan (each should contain subdirs with OUTCAR and CONTCAR)",
    )
    parser.add_argument(
        "-o", "--output",
        default="combined_lowest_frames.xyz",
        help="Output XYZ file path for all frames (default: combined_lowest_frames.xyz)",
    )
    parser.add_argument(
        "-n", "--num-frames",
        type=int,
        default=10,
        help="Number of frames closest to minimum to extract per directory (default: 10)",
    )
    args = parser.parse_args()

    all_frames = []
    for d in args.directories:
        all_frames.extend(process_directory(d, args.num_frames))

    if not all_frames:
        print("No frames collected.", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_xyz_frames(out_path, all_frames)
    print(f"Wrote {len(all_frames)} frames to {out_path}")


if __name__ == "__main__":
    main()
