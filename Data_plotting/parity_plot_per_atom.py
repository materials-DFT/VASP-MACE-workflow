import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
import argparse
import sys

# --- Argument parser ---
parser = argparse.ArgumentParser(
    description=(
        "Plot MLIP (MACE, ALLEGRO, or UMA) vs REF parity plots from an XYZ file. "
        "Auto-detects MACE, ALLEGRO, or UMA from frame.info. "
        "Energy plots use relative energies (reference structure = 0)."
    )
)
parser.add_argument("xyz_file", type=str, help="Path to the input .xyz file")
parser.add_argument(
    "--ref-index",
    type=int,
    default=None,
    help=(
        "Index (0-based) of the reference structure for relative energy. "
        "If not provided, the structure with the lowest REF energy per atom is used."
    ),
)
args = parser.parse_args()

# --- Read file ---
try:
    frames = read(args.xyz_file, index=":")
except Exception as e:
    print(f"Error reading file {args.xyz_file}: {e}")
    sys.exit(1)

if len(frames) == 0:
    print(f"No frames found in file {args.xyz_file}")
    sys.exit(1)

# --- Detect MLIP type (MACE, ALLEGRO, or UMA) ---
first_info = frames[0].info
if "MACE_energy" in first_info:
    ml_energy_key = "MACE_energy"
    ml_forces_key = "MACE_forces"
    ml_stress_key = "MACE_stress"
    ml_label = "MACE"
elif "ALLEGRO_energy" in first_info:
    ml_energy_key = "ALLEGRO_energy"
    ml_forces_key = "ALLEGRO_forces"
    ml_stress_key = "ALLEGRO_stress"
    ml_label = "ALLEGRO"
elif "UMA_energy" in first_info:
    ml_energy_key = "UMA_energy"
    ml_forces_key = "UMA_forces"
    ml_stress_key = "UMA_stress"
    ml_label = "UMA"
else:
    print(
        "No 'MACE_energy', 'ALLEGRO_energy', or 'UMA_energy' in frame.info. "
        "Make sure the XYZ file was written with one of these fields."
    )
    sys.exit(1)
if "REF_energy" not in first_info:
    print("No 'REF_energy' in frame.info.")
    sys.exit(1)

# --- Data extraction ---
ml_energies = []
ref_energies = []
ml_forces = []
ref_forces = []
ml_stresses = []
ref_stresses = []
n_atoms_list = []

for i, frame in enumerate(frames):
    if ml_energy_key not in frame.info or "REF_energy" not in frame.info:
        print(
            f"Frame {i} is missing '{ml_energy_key}' or 'REF_energy' in frame.info."
        )
        sys.exit(1)
    ml_energies.append(frame.info[ml_energy_key])
    ref_energies.append(frame.info["REF_energy"])
    n_atoms_list.append(len(frame))

    ml_forces.extend(frame.arrays[ml_forces_key].flatten())
    ref_forces.extend(frame.arrays["REF_forces"].flatten())

    if ml_stress_key in frame.info and "REF_stress" in frame.info:
        ml_s = np.asarray(frame.info[ml_stress_key]).flatten()
        ref_s = np.asarray(frame.info["REF_stress"]).flatten()
        ml_stresses.extend(ml_s)
        ref_stresses.extend(ref_s)

ml_energies = np.array(ml_energies)
ref_energies = np.array(ref_energies)
ml_forces = np.array(ml_forces)
ref_forces = np.array(ref_forces)
n_atoms_list = np.array(n_atoms_list)
has_stress = len(ml_stresses) > 0 and len(ref_stresses) > 0
if has_stress:
    ml_stresses = np.array(ml_stresses)
    ref_stresses = np.array(ref_stresses)

# --- Per-atom energies ---
ref_per_atom = ref_energies / n_atoms_list
ml_per_atom = ml_energies / n_atoms_list

# --- Reference structure for relative energy ---
if args.ref_index is not None:
    if args.ref_index < 0 or args.ref_index >= len(ref_per_atom):
        print(
            f"Invalid ref-index {args.ref_index}. "
            f"Must be between 0 and {len(ref_per_atom) - 1}."
        )
        sys.exit(1)
    ref_idx = args.ref_index
else:
    ref_idx = int(np.argmin(ref_per_atom))

# --- Relative energies (total and per atom) ---
delta_ref_total = ref_energies - ref_energies[ref_idx]
delta_ml_total = ml_energies - ml_energies[ref_idx]
delta_ref_per_atom = ref_per_atom - ref_per_atom[ref_idx]
delta_ml_per_atom = ml_per_atom - ml_per_atom[ref_idx]

# --- RMSE (relative energy + absolute force/stress) ---
rmse_rel_total = float(np.sqrt(np.mean((delta_ml_total - delta_ref_total) ** 2)))
rmse_rel_per_atom = float(np.sqrt(np.mean((delta_ml_per_atom - delta_ref_per_atom) ** 2)))
rmse_forces = float(np.sqrt(np.mean((ml_forces - ref_forces) ** 2)))
if has_stress:
    rmse_stress = float(np.sqrt(np.mean((ml_stresses - ref_stresses) ** 2)))

print(f"Detected MLIP: {ml_label}")
print(f"Using frame {ref_idx} as reference for relative energy.")
print(f"  RMSE of ΔE (total):     {rmse_rel_total:.6f} eV")
print(f"  RMSE of ΔE per atom:    {rmse_rel_per_atom:.6f} eV/atom")
print(f"  RMSE of forces:         {rmse_forces:.6f} eV/Å")
if has_stress:
    print(f"  RMSE of stress:         {rmse_stress:.6f} eV/Å³")

# --- Plotting ---
ncols = 2 if has_stress else 3
nrows = 2 if has_stress else 1
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
axes = np.atleast_2d(axes)

# Relative Total Energy Plot
ax = axes[0, 0]
ax.scatter(delta_ref_total, delta_ml_total, alpha=0.5)
all_vals = np.concatenate([delta_ref_total, delta_ml_total])
lims = [float(all_vals.min()), float(all_vals.max())]
ax.plot(lims, lims, "k--", label="y = x")
ax.set_xlabel(r"$\Delta E_{\mathrm{REF}}$ (eV)")
ax.set_ylabel(r"$\Delta E_{\mathrm{" + ml_label + r"}}$ (eV)")
ax.set_title(f"Relative Total Energy Parity Plot\nRMSE = {rmse_rel_total:.6f} eV")
ax.axis("equal")
ax.grid(True)

# Force Parity Plot
ax = axes[0, 1]
ax.scatter(ref_forces, ml_forces, alpha=0.5)
lims = [min(ref_forces.min(), ml_forces.min()), max(ref_forces.max(), ml_forces.max())]
ax.plot(lims, lims, "k--")
ax.set_xlabel("REF Force (eV/Å)")
ax.set_ylabel(f"{ml_label} Force (eV/Å)")
ax.set_title(f"Force Parity Plot\nRMSE = {rmse_forces:.6f} eV/Å")
ax.axis("equal")
ax.grid(True)

# Relative Per-Atom Energy Plot
ax = axes[1, 0] if has_stress else axes[0, 2]
ax.scatter(delta_ref_per_atom, delta_ml_per_atom, alpha=0.5)
all_vals = np.concatenate([delta_ref_per_atom, delta_ml_per_atom])
lims = [float(all_vals.min()), float(all_vals.max())]
ax.plot(lims, lims, "k--", label="y = x")
ax.set_xlabel(r"$\Delta E_{\mathrm{REF}}$ per atom (eV)")
ax.set_ylabel(r"$\Delta E_{\mathrm{" + ml_label + r"}}$ per atom (eV)")
ax.set_title(f"Relative Per-Atom Energy Parity Plot\nRMSE = {rmse_rel_per_atom:.6f} eV/atom")
ax.axis("equal")
ax.grid(True)

# Stress Parity Plot (when REF_stress and ML stress present)
if has_stress:
    ax = axes[1, 1]
    ax.scatter(ref_stresses, ml_stresses, alpha=0.5)
    lims = [min(ref_stresses.min(), ml_stresses.min()), max(ref_stresses.max(), ml_stresses.max())]
    ax.plot(lims, lims, "k--")
    ax.set_xlabel("REF Stress (eV/Å³)")
    ax.set_ylabel(f"{ml_label} Stress (eV/Å³)")
    ax.set_title(f"Stress Parity Plot (all tensor components)\nRMSE = {rmse_stress:.6f} eV/Å³")
    ax.axis("equal")
    ax.grid(True)

plt.tight_layout()
plt.show()
