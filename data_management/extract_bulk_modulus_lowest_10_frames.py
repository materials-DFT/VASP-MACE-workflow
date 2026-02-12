#!/usr/bin/env python3
"""
Scan specified directories for VASP OUTCARs, find the lowest-energy
frame per directory, then extract the 10 frames closest in energy to that
minimum. All frames from all directories are written into a single multi-frame
extended XYZ file (with lattice and stress). Uses only OUTCAR (no CONTCAR).
"""

import re
import sys
import argparse
from pathlib import Path

# VASP stress in OUTCAR is in kB. 1 kB = 0.1 GPa; 1 eV/Å³ = 160.2176634 GPa.
KB_TO_EV_ANG3 = 0.1 / 160.2176634


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


def get_stress_from_outcar(outcar_path):
    """
    Extract the last 3x3 stress tensor from OUTCAR (in kB), convert to Voigt
    (xx, yy, zz, yz, xz, xy) in eV/Å³. Returns None if not found.
    """
    last_stress_3x3 = None
    try:
        with open(outcar_path, "r") as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            if " in kB " in lines[i].strip():
                stress_3x3 = []
                for j in range(i + 1, min(i + 4, len(lines))):
                    parts = lines[j].split()
                    if len(parts) >= 3:
                        try:
                            row = [float(parts[0]), float(parts[1]), float(parts[2])]
                            stress_3x3.append(row)
                        except ValueError:
                            break
                if len(stress_3x3) == 3:
                    last_stress_3x3 = stress_3x3
                i += 4
                continue
            i += 1
    except OSError:
        return None
    if last_stress_3x3 is None:
        return None
    # Voigt order: xx, yy, zz, yz, xz, xy
    s = last_stress_3x3
    voigt_kb = [
        s[0][0], s[1][1], s[2][2],
        s[1][2], s[0][2], s[0][1]
    ]
    return [v * KB_TO_EV_ANG3 for v in voigt_kb]


def _element_from_titel_line(line):
    """Extract element symbol from OUTCAR/POTCAR TITEL line (e.g. 'TITEL  = PAW P 15Jan2000')."""
    tokens = line.strip().split()
    for tok in tokens:
        base = tok.split("_", 1)[0]
        if re.fullmatch(r"[A-Z][a-z]?", base):
            return base
    return None


def parse_structure_from_outcar(outcar_path):
    """
    Parse the final structure (lattice + species + positions) from a VASP OUTCAR.
    Returns (lattice_cart, coords) where lattice_cart is 3x3 Cartesian Angstrom,
    coords is list of (species, x, y, z) in Cartesian Angstrom; or None on error.
    """
    try:
        with open(outcar_path, "r") as f:
            lines = [l.rstrip() for l in f.readlines()]
    except OSError:
        return None

    try:
        # Species: TITEL lines in order (from POTCAR echo in OUTCAR)
        species_from_titel = []
        for line in lines:
            if line.strip().startswith("TITEL"):
                el = _element_from_titel_line(line)
                if el:
                    species_from_titel.append(el)

        # Counts: "ions per type = n1 n2 ..."
        counts = None
        for line in lines:
            if "ions per type" in line.lower():
                parts = line.split("=")
                if len(parts) >= 2:
                    counts = [int(x) for x in parts[1].split()]
                    break
        if not counts:
            return None

        # If we have TITEL-derived species, use them; else fallback to generic (e.g. X1, X2)
        if len(species_from_titel) >= len(counts):
            symbols = species_from_titel[: len(counts)]
        else:
            symbols = [f"X{i+1}" for i in range(len(counts))]
        species_list = []
        for sym, cnt in zip(symbols, counts):
            species_list.extend([sym] * cnt)
        n_atoms = sum(counts)

        # Last lattice: "LATTICE" and "ANGSTROM" (or "direct lattice vectors")
        lattice_cart = None
        for i in range(len(lines) - 3):
            line_lower = lines[i].lower()
            if ("lattice" in line_lower and "angstrom" in line_lower) or (
                "direct" in line_lower and "lattice" in line_lower
            ):
                try:
                    lattice_cart = [
                        [float(x) for x in lines[i + 1].split()[:3]],
                        [float(x) for x in lines[i + 2].split()[:3]],
                        [float(x) for x in lines[i + 3].split()[:3]],
                    ]
                except (ValueError, IndexError):
                    pass

        if lattice_cart is None:
            return None

        # Last POSITION / TOTAL-FORCE block: positions are first 3 columns (Angstrom)
        coords = None
        for i in range(len(lines)):
            if "POSITION" in lines[i] and "TOTAL-FORCE" in lines[i]:
                pos_lines = []
                j = i + 2  # skip header and "---"
                while j < len(lines) and len(pos_lines) < n_atoms + 2:
                    stripped = lines[j].strip()
                    if not stripped or "---" in stripped:
                        if len(pos_lines) >= n_atoms:
                            break
                        j += 1
                        continue
                    parts = lines[j].split()
                    if len(parts) >= 3:
                        try:
                            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                            pos_lines.append((x, y, z))
                        except ValueError:
                            pass
                    j += 1
                if len(pos_lines) == n_atoms:
                    coords = [
                        (species_list[k], pos_lines[k][0], pos_lines[k][1], pos_lines[k][2])
                        for k in range(n_atoms)
                    ]

        if coords is None:
            return None
        return (lattice_cart, coords)
    except (IndexError, ValueError):
        return None


def write_xyz_frames(filepath, frames):
    """Write multiple structures to a single multi-frame extended XYZ file (lattice + optional stress)."""
    with open(filepath, "w") as f:
        for item in frames:
            comment = item[0]
            lattice_cart = item[1]
            atoms = item[2]
            stress_voigt = item[3] if len(item) > 3 else None
            n = len(atoms)
            f.write(f"{n}\n")
            lat_flat = [lattice_cart[i][j] for i in range(3) for j in range(3)]
            lat_str = " ".join(f"{x:.10f}" for x in lat_flat)
            # Extended XYZ: Lattice, Properties, optional stress, comment
            line_parts = [f'Lattice="{lat_str}"', 'Properties=species:S:1:pos:R:3']
            if stress_voigt is not None:
                s = stress_voigt
                line_parts.append(f"stress={s[0]:.10f} {s[1]:.10f} {s[2]:.10f} {s[3]:.10f} {s[4]:.10f} {s[5]:.10f}")
            line_parts.append(f'comment="{comment}"')
            f.write(" ".join(line_parts) + "\n")
            for spec, x, y, z in atoms:
                f.write(f"{spec} {x:.10f} {y:.10f} {z:.10f}\n")


def _collect_candidates(top_path):
    """Return list of (subdir_path, energy) for direct children with OUTCAR."""
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


def process_directory(top_dir, num_frames=10):
    """
    For one directory (e.g. gamma/), find subdirs with OUTCAR,
    get final energy from each OUTCAR, find minimum, then the num_frames
    subdirs closest in energy. Return their structures (from OUTCAR) as
    (comment, lattice, atoms, stress) frames.
    If direct children have no OUTCAR but their children do (e.g.
    isif3/gamma/V_+00.5%), process each direct child and aggregate returned frames.
    """
    top_path = Path(top_dir).resolve()
    if not top_path.is_dir():
        print(f"Not a directory: {top_path}", file=sys.stderr)
        return []

    candidates = _collect_candidates(top_path)

    # No OUTCAR in direct children: try one level deeper (e.g. isif3 -> gamma -> V_+00.5%)
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

    # Build frames from OUTCARs (structure + stress)
    frames = []
    for sub, en in selected:
        outcar = sub / "OUTCAR"
        parsed = parse_structure_from_outcar(outcar)
        if parsed is None:
            print(f"Failed to parse structure from OUTCAR: {outcar}", file=sys.stderr)
            continue
        lattice_cart, atoms = parsed
        stress_voigt = get_stress_from_outcar(outcar)
        comment = f"{sub.name}  E = {en:.6f} eV"
        frames.append((comment, lattice_cart, atoms, stress_voigt))

    if not frames:
        print(f"No valid structures among selected frames under {top_path}", file=sys.stderr)
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
        help="Directories to scan (each should contain subdirs with OUTCAR)",
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
