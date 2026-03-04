import glob
import os
import re
import sys
import numpy as np
from ase.neighborlist import neighbor_list
from ase import Atoms

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <directory_with_xyz_files>")
    sys.exit(1)

xyz_dir = sys.argv[1]
if not os.path.isdir(xyz_dir):
    print(f"Error: '{xyz_dir}' is not a valid directory")
    sys.exit(1)

xyz_files = sorted(glob.glob(os.path.join(xyz_dir, "*.xyz")))
if not xyz_files:
    print(f"No .xyz files found in '{xyz_dir}'")
    sys.exit(1)


def parse_lattice(comment_line):
    match = re.search(r'Lattice="([^"]+)"', comment_line)
    if not match:
        return None
    vals = list(map(float, match.group(1).split()))
    return np.array(vals).reshape(3, 3)


def parse_pbc(comment_line):
    match = re.search(r'pbc="([^"]+)"', comment_line)
    if not match:
        return [True, True, True]
    tokens = match.group(1).split()
    return [t.upper() in ("T", "TRUE", "1") for t in tokens]


def parse_xyz_frames(filepath):
    """Parse a multi-frame XYZ file, returning raw text blocks and parsed atom data."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    frames = []
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        try:
            natoms = int(line)
        except ValueError:
            idx += 1
            continue

        count_line = lines[idx]
        comment_line = lines[idx + 1]
        atom_lines = lines[idx + 2 : idx + 2 + natoms]

        raw_block = [count_line, comment_line] + atom_lines

        species = []
        positions = []
        for aline in atom_lines:
            parts = aline.split()
            species.append(parts[0])
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])

        lattice = parse_lattice(comment_line)
        pbc = parse_pbc(comment_line)

        frames.append({
            "raw": raw_block,
            "species": species,
            "positions": np.array(positions),
            "lattice": lattice,
            "pbc": pbc,
        })
        idx += 2 + natoms

    return frames


def has_close_contacts(species, positions, lattice, pbc, cutoff):
    atoms = Atoms(symbols=species, positions=positions, cell=lattice, pbc=pbc)
    dists = neighbor_list("d", atoms, cutoff=cutoff)
    return len(dists) > 0


print("Counting structures...", flush=True)
all_frames = []
for fpath in xyz_files:
    try:
        frames = parse_xyz_frames(fpath)
        all_frames.extend(frames)
    except Exception as e:
        print(f"ERROR reading {os.path.basename(fpath)}: {e}")

total_structure_count = len(all_frames)
print(f"Found {total_structure_count} structures in {len(xyz_files)} files.\n", flush=True)

raw_blocks_15 = []
raw_blocks_12 = []

for i, frame in enumerate(all_frames):
    if i % 100 == 0:
        print(f"  Working on {i}/{total_structure_count}...", flush=True)

    sp = frame["species"]
    pos = frame["positions"]
    lat = frame["lattice"]
    pbc = frame["pbc"]
    raw = frame["raw"]

    close_15 = has_close_contacts(sp, pos, lat, pbc, 1.5)
    if not close_15:
        raw_blocks_15.append(raw)
        raw_blocks_12.append(raw)
    else:
        close_12 = has_close_contacts(sp, pos, lat, pbc, 1.2)
        if not close_12:
            raw_blocks_12.append(raw)

print(f"\nTotal files: {len(xyz_files)}")
print(f"Total structures: {total_structure_count}")
print(f"Structures with min dist > 1.5 A: {len(raw_blocks_15)}")
print(f"Structures with min dist > 1.2 A: {len(raw_blocks_12)}")

if raw_blocks_15:
    with open("filtered_min_dist_1.5A.xyz", "w") as f:
        for block in raw_blocks_15:
            f.writelines(block)
    print(f"Wrote filtered_min_dist_1.5A.xyz ({len(raw_blocks_15)} structures)")
else:
    print("No structures passed the 1.5 A filter.")

if raw_blocks_12:
    with open("filtered_min_dist_1.2A.xyz", "w") as f:
        for block in raw_blocks_12:
            f.writelines(block)
    print(f"Wrote filtered_min_dist_1.2A.xyz ({len(raw_blocks_12)} structures)")
else:
    print("No structures passed the 1.2 A filter.")
