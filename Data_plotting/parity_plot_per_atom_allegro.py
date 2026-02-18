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
allegro_stresses = []
ref_stresses = []
n_atoms_list = []

for frame in frames:
    allegro_energies.append(frame.info["ALLEGRO_energy"])
    ref_energies.append(frame.info["REF_energy"])
    n_atoms_list.append(len(frame))

    allegro_forces.extend(frame.arrays["ALLEGRO_forces"].flatten())
    ref_forces.extend(frame.arrays["REF_forces"].flatten())

    # Stress (optional): 3x3 tensor per frame
    if "ALLEGRO_stress" in frame.info and "REF_stress" in frame.info:
        allegro_s = np.asarray(frame.info["ALLEGRO_stress"]).flatten()
        ref_s = np.asarray(frame.info["REF_stress"]).flatten()
        allegro_stresses.extend(allegro_s)
        ref_stresses.extend(ref_s)

allegro_energies = np.array(allegro_energies)
ref_energies = np.array(ref_energies)
allegro_forces = np.array(allegro_forces)
ref_forces = np.array(ref_forces)
n_atoms_list = np.array(n_atoms_list)
has_stress = len(allegro_stresses) > 0 and len(ref_stresses) > 0
if has_stress:
    allegro_stresses = np.array(allegro_stresses)
    ref_stresses = np.array(ref_stresses)

# --- Per-atom energies ---
ref_per_atom = ref_energies / n_atoms_list
allegro_per_atom = allegro_energies / n_atoms_list

# --- RMSE calculations ---
rmse_total_energy = np.sqrt(np.mean((allegro_energies - ref_energies) ** 2))
rmse_forces = np.sqrt(np.mean((allegro_forces - ref_forces) ** 2))
rmse_per_atom = np.sqrt(np.mean((allegro_per_atom - ref_per_atom) ** 2))
if has_stress:
    rmse_stress = np.sqrt(np.mean((allegro_stresses - ref_stresses) ** 2))

# --- Plotting ---
if has_stress:
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = np.atleast_2d(axes)
else:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Total Energy Plot
ax = axes[0, 0] if has_stress else axes[0]
ax.scatter(ref_energies, allegro_energies, alpha=0.5)
lims = [min(ref_energies.min(), allegro_energies.min()), max(ref_energies.max(), allegro_energies.max())]
ax.plot(lims, lims, 'k--')
ax.set_xlabel("REF Energy (eV)")
ax.set_ylabel("ALLEGRO Energy (eV)")
ax.set_title(f"Total Energy Parity Plot\nRMSE = {rmse_total_energy:.4f} eV")
ax.axis("equal")
ax.grid(True)

# Force Parity Plot
ax = axes[0, 1] if has_stress else axes[1]
ax.scatter(ref_forces, allegro_forces, alpha=0.5)
lims = [min(ref_forces.min(), allegro_forces.min()), max(ref_forces.max(), allegro_forces.max())]
ax.plot(lims, lims, 'k--')
ax.set_xlabel("REF Force (eV/Å)")
ax.set_ylabel("ALLEGRO Force (eV/Å)")
ax.set_title(f"Force Parity Plot\nRMSE = {rmse_forces:.4f} eV/Å")
ax.axis("equal")
ax.grid(True)

# Per-Atom Energy Plot
ax = axes[1, 0] if has_stress else axes[2]
ax.scatter(ref_per_atom, allegro_per_atom, alpha=0.5)
lims = [min(ref_per_atom.min(), allegro_per_atom.min()), max(ref_per_atom.max(), allegro_per_atom.max())]
ax.plot(lims, lims, 'k--')
ax.set_xlabel("REF Energy per Atom (eV)")
ax.set_ylabel("ALLEGRO Energy per Atom (eV)")
ax.set_title(f"Per-Atom Energy Parity Plot\nRMSE = {rmse_per_atom:.4f} eV/atom")
ax.axis("equal")
ax.grid(True)

# Stress Parity Plot (when REF_stress and ALLEGRO_stress present)
if has_stress:
    ax = axes[1, 1]
    ax.scatter(ref_stresses, allegro_stresses, alpha=0.5)
    lims = [min(ref_stresses.min(), allegro_stresses.min()), max(ref_stresses.max(), allegro_stresses.max())]
    ax.plot(lims, lims, 'k--')
    ax.set_xlabel("REF Stress (eV/Å³)")
    ax.set_ylabel("ALLEGRO Stress (eV/Å³)")
    ax.set_title(f"Stress Parity Plot (all tensor components)\nRMSE = {rmse_stress:.4f} eV/Å³")
    ax.axis("equal")
    ax.grid(True)

plt.tight_layout()
plt.show()
