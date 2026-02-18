import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
import argparse
import sys

# --- Argument parser ---
parser = argparse.ArgumentParser(description="Plot MACE vs REF parity plots from an XYZ file.")
parser.add_argument("xyz_file", type=str, help="Path to the input .xyz file")
args = parser.parse_args()

# --- Read file ---
try:
    frames = read(args.xyz_file, index=":")
except Exception as e:
    print(f"Error reading file {args.xyz_file}: {e}")
    sys.exit(1)

# --- Data extraction ---
mace_energies = []
ref_energies = []
mace_forces = []
ref_forces = []
mace_stresses = []
ref_stresses = []
n_atoms_list = []

for frame in frames:
    mace_energies.append(frame.info["MACE_energy"])
    ref_energies.append(frame.info["REF_energy"])
    n_atoms_list.append(len(frame))
    
    mace_forces.extend(frame.arrays["MACE_forces"].flatten())
    ref_forces.extend(frame.arrays["REF_forces"].flatten())

    # Stress (optional): 3x3 tensor per frame
    if "MACE_stress" in frame.info and "REF_stress" in frame.info:
        mace_s = np.asarray(frame.info["MACE_stress"]).flatten()
        ref_s = np.asarray(frame.info["REF_stress"]).flatten()
        mace_stresses.extend(mace_s)
        ref_stresses.extend(ref_s)

mace_energies = np.array(mace_energies)
ref_energies = np.array(ref_energies)
mace_forces = np.array(mace_forces)
ref_forces = np.array(ref_forces)
n_atoms_list = np.array(n_atoms_list)
has_stress = len(mace_stresses) > 0 and len(ref_stresses) > 0
if has_stress:
    mace_stresses = np.array(mace_stresses)
    ref_stresses = np.array(ref_stresses)

# --- Per-atom energies ---
ref_per_atom = ref_energies / n_atoms_list
mace_per_atom = mace_energies / n_atoms_list

# --- Plotting ---
n_plots = 4 if has_stress else 3
ncols = 2 if has_stress else 3
nrows = 2 if has_stress else 1
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
axes = np.atleast_2d(axes)

# Total Energy Plot
ax = axes[0, 0]
ax.scatter(ref_energies, mace_energies, alpha=0.5)
lims = [min(ref_energies.min(), mace_energies.min()), max(ref_energies.max(), mace_energies.max())]
ax.plot(lims, lims, 'k--')
ax.set_xlabel("REF Energy (eV)")
ax.set_ylabel("MACE Energy (eV)")
ax.set_title("Total Energy Parity Plot")
ax.axis("equal")
ax.grid(True)

# Force Parity Plot
ax = axes[0, 1]
ax.scatter(ref_forces, mace_forces, alpha=0.5)
lims = [min(ref_forces.min(), mace_forces.min()), max(ref_forces.max(), mace_forces.max())]
ax.plot(lims, lims, 'k--')
ax.set_xlabel("REF Force (eV/Å)")
ax.set_ylabel("MACE Force (eV/Å)")
ax.set_title("Force Parity Plot")
ax.axis("equal")
ax.grid(True)

# Per-Atom Energy Plot
ax = axes[1, 0] if has_stress else axes[0, 2]
ax.scatter(ref_per_atom, mace_per_atom, alpha=0.5)
lims = [min(ref_per_atom.min(), mace_per_atom.min()), max(ref_per_atom.max(), mace_per_atom.max())]
ax.plot(lims, lims, 'k--')
ax.set_xlabel("REF Energy per Atom (eV)")
ax.set_ylabel("MACE Energy per Atom (eV)")
ax.set_title("Per-Atom Energy Parity Plot")
ax.axis("equal")
ax.grid(True)

# Stress Parity Plot (when REF_stress and MACE_stress present)
if has_stress:
    ax = axes[1, 1]
    ax.scatter(ref_stresses, mace_stresses, alpha=0.5)
    lims = [min(ref_stresses.min(), mace_stresses.min()), max(ref_stresses.max(), mace_stresses.max())]
    ax.plot(lims, lims, 'k--')
    ax.set_xlabel("REF Stress (eV/Å³)")
    ax.set_ylabel("MACE Stress (eV/Å³)")
    ax.set_title("Stress Parity Plot (all tensor components)")
    ax.axis("equal")
    ax.grid(True)

plt.tight_layout()
plt.show()
