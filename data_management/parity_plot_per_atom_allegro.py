import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
import argparse
import sys

# --- Argument parser ---
parser = argparse.ArgumentParser(description="Plot ALLEGRO vs REF parity plots from an XYZ file.")
parser.add_argument("xyz_file", type=str, help="Path to the input .xyz file (e.g. output.xyz from ALLEGRO evaluation)")
args = parser.parse_args()

# --- Read file ---
try:
    frames = read(args.xyz_file, index=":")
except Exception as e:
    print(f"Error reading file {args.xyz_file}: {e}")
    sys.exit(1)

# --- Data extraction ---
allegro_energies = []
ref_energies = []
allegro_forces = []
ref_forces = []
n_atoms_list = []

for frame in frames:
    allegro_energies.append(frame.info["ALLEGRO_energy"])
    ref_energies.append(frame.info["REF_energy"])
    n_atoms_list.append(len(frame))

    allegro_forces.extend(frame.arrays["ALLEGRO_forces"].flatten())
    ref_forces.extend(frame.arrays["REF_forces"].flatten())

allegro_energies = np.array(allegro_energies)
ref_energies = np.array(ref_energies)
allegro_forces = np.array(allegro_forces)
ref_forces = np.array(ref_forces)
n_atoms_list = np.array(n_atoms_list)

# --- Per-atom energies ---
ref_per_atom = ref_energies / n_atoms_list
allegro_per_atom = allegro_energies / n_atoms_list

# --- Plotting ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Total Energy Plot
axes[0].scatter(ref_energies, allegro_energies, alpha=0.5)
lims = [min(ref_energies.min(), allegro_energies.min()), max(ref_energies.max(), allegro_energies.max())]
axes[0].plot(lims, lims, 'k--')
axes[0].set_xlabel("REF Energy (eV)")
axes[0].set_ylabel("ALLEGRO Energy (eV)")
axes[0].set_title("Total Energy Parity Plot")
axes[0].axis("equal")
axes[0].grid(True)

# Force Parity Plot
axes[1].scatter(ref_forces, allegro_forces, alpha=0.5)
lims = [min(ref_forces.min(), allegro_forces.min()), max(ref_forces.max(), allegro_forces.max())]
axes[1].plot(lims, lims, 'k--')
axes[1].set_xlabel("REF Force (eV/Å)")
axes[1].set_ylabel("ALLEGRO Force (eV/Å)")
axes[1].set_title("Force Parity Plot")
axes[1].axis("equal")
axes[1].grid(True)

# Per-Atom Energy Plot
axes[2].scatter(ref_per_atom, allegro_per_atom, alpha=0.5)
lims = [min(ref_per_atom.min(), allegro_per_atom.min()), max(ref_per_atom.max(), allegro_per_atom.max())]
axes[2].plot(lims, lims, 'k--')
axes[2].set_xlabel("REF Energy per Atom (eV)")
axes[2].set_ylabel("ALLEGRO Energy per Atom (eV)")
axes[2].set_title("Per-Atom Energy Parity Plot")
axes[2].axis("equal")
axes[2].grid(True)

plt.tight_layout()
plt.show()
