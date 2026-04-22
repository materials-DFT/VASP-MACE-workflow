import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ase.io import read
from collections import defaultdict
import argparse
import sys

# --- Argument parser ---
parser = argparse.ArgumentParser(
    description=(
        "Plot per-species force parity (MLIP vs REF) from an XYZ file. "
        "Auto-detects MACE, ALLEGRO, or UMA from frame.info. "
        "Shows combined force parity per species and directional (Fx, Fy, Fz) breakdown."
    )
)
parser.add_argument("xyz_file", type=str, help="Path to the input .xyz file")
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
    ml_forces_key = "MACE_forces"
    ml_label = "MACE"
elif "ALLEGRO_energy" in first_info:
    ml_forces_key = "ALLEGRO_forces"
    ml_label = "ALLEGRO"
elif "UMA_energy" in first_info:
    ml_forces_key = "UMA_forces"
    ml_label = "UMA"
else:
    print(
        "No 'MACE_energy', 'ALLEGRO_energy', or 'UMA_energy' in frame.info. "
        "Make sure the XYZ file was written with one of these fields."
    )
    sys.exit(1)

# --- Data extraction per species ---
species_forces = defaultdict(
    lambda: {k: [] for k in ("ref_fx", "ref_fy", "ref_fz", "ml_fx", "ml_fy", "ml_fz")}
)

for i, frame in enumerate(frames):
    ref_f = frame.arrays["REF_forces"]
    ml_f = frame.arrays[ml_forces_key]
    symbols = frame.get_chemical_symbols()
    for j, sym in enumerate(symbols):
        species_forces[sym]["ref_fx"].append(ref_f[j, 0])
        species_forces[sym]["ref_fy"].append(ref_f[j, 1])
        species_forces[sym]["ref_fz"].append(ref_f[j, 2])
        species_forces[sym]["ml_fx"].append(ml_f[j, 0])
        species_forces[sym]["ml_fy"].append(ml_f[j, 1])
        species_forces[sym]["ml_fz"].append(ml_f[j, 2])

for sym in species_forces:
    for k in species_forces[sym]:
        species_forces[sym][k] = np.array(species_forces[sym][k])
species_list = sorted(species_forces.keys())
n_species = len(species_list)

# --- Layout: row 0 = combined per species, rows 1-3 = Fx/Fy/Fz per species ---
nrows = 4
ncols = n_species
fig = plt.figure(figsize=(5.5 * ncols, 5 * nrows), constrained_layout=True)
gs = GridSpec(nrows, ncols, figure=fig)

dir_labels = ["x", "y", "z"]
dir_keys = [("ref_fx", "ml_fx"), ("ref_fy", "ml_fy"), ("ref_fz", "ml_fz")]

# Row 0: combined force parity per species
for col, sym in enumerate(species_list):
    sf = species_forces[sym]
    ref_all = np.concatenate([sf["ref_fx"], sf["ref_fy"], sf["ref_fz"]])
    ml_all = np.concatenate([sf["ml_fx"], sf["ml_fy"], sf["ml_fz"]])
    rmse = float(np.sqrt(np.mean((ml_all - ref_all) ** 2)))
    mae = float(np.mean(np.abs(ml_all - ref_all)))
    n_atoms_species = len(sf["ref_fx"])

    ax = fig.add_subplot(gs[0, col])
    ax.scatter(ref_all, ml_all, alpha=0.3, s=4)
    lims = [min(ref_all.min(), ml_all.min()), max(ref_all.max(), ml_all.max())]
    ax.plot(lims, lims, "k--")
    ax.set_xlabel("REF Force (eV/Å)", fontsize=9)
    ax.set_ylabel(f"{ml_label} Force (eV/Å)", fontsize=9)
    ax.set_title(
        f"{sym} Forces ({n_atoms_species:,} atoms)\n"
        f"RMSE = {rmse:.4f}  MAE = {mae:.4f} eV/Å",
        fontsize=10,
    )
    ax.tick_params(labelsize=8)
    ax.axis("equal")
    ax.grid(True)

# Rows 1-3: directional breakdown per species
for col, sym in enumerate(species_list):
    sf = species_forces[sym]
    for row_off, (d_label, (ref_key, ml_key)) in enumerate(zip(dir_labels, dir_keys)):
        ref_d = sf[ref_key]
        ml_d = sf[ml_key]
        rmse = float(np.sqrt(np.mean((ml_d - ref_d) ** 2)))
        mae = float(np.mean(np.abs(ml_d - ref_d)))

        ax = fig.add_subplot(gs[1 + row_off, col])
        ax.scatter(ref_d, ml_d, alpha=0.3, s=4)
        lims = [min(ref_d.min(), ml_d.min()), max(ref_d.max(), ml_d.max())]
        ax.plot(lims, lims, "k--")
        ax.set_xlabel(f"REF $F_{{{d_label}}}$ (eV/Å)", fontsize=9)
        ax.set_ylabel(f"{ml_label} $F_{{{d_label}}}$ (eV/Å)", fontsize=9)
        ax.set_title(
            f"{sym} — $F_{{{d_label}}}$\n"
            f"RMSE = {rmse:.4f}  MAE = {mae:.4f} eV/Å",
            fontsize=10,
        )
        ax.tick_params(labelsize=8)
        ax.axis("equal")
        ax.grid(True)

fig.suptitle(
    f"Per-Species Force Parity ({ml_label} vs REF)", fontsize=16, y=0.995
)

# --- Print per-species force summary ---
print(f"Detected MLIP: {ml_label}")
print(f"Read {len(frames)} frames, species: {', '.join(species_list)}\n")
print("Force RMSE/MAE by species:")
for sym in species_list:
    sf = species_forces[sym]
    ref_all = np.concatenate([sf["ref_fx"], sf["ref_fy"], sf["ref_fz"]])
    ml_all = np.concatenate([sf["ml_fx"], sf["ml_fy"], sf["ml_fz"]])
    rmse = float(np.sqrt(np.mean((ml_all - ref_all) ** 2)))
    mae = float(np.mean(np.abs(ml_all - ref_all)))
    print(f"  {sym:>4s}:  RMSE = {rmse:.6f}  MAE = {mae:.6f} eV/Å")
    for d_label, (ref_key, ml_key) in zip(dir_labels, dir_keys):
        ref_d = sf[ref_key]
        ml_d = sf[ml_key]
        r = float(np.sqrt(np.mean((ml_d - ref_d) ** 2)))
        m = float(np.mean(np.abs(ml_d - ref_d)))
        print(f"         F_{d_label}:  RMSE = {r:.6f}  MAE = {m:.6f} eV/Å")

plt.show()
