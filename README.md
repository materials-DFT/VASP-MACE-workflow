# 🔬 VASP-MACE-workflow

A collection of Python scripts and shell utilities for automating VASP (Vienna Ab initio Simulation Package) calculations, preparing molecular dynamics simulations, managing MACE (Machine-learning Approach to Chemistry Emulation) workflows, and analyzing computational materials science data.

## 📋 Overview

This repository provides tools to streamline common tasks in computational materials science workflows, particularly those involving:
- **⚛️ VASP calculations**: Automated setup, parameter optimization, and job management
- **🤖 MACE machine learning**: Model evaluation and validation
- **🌡️ Molecular dynamics**: NPT simulation preparation with optimized parameters
- **📊 Data analysis**: Extraction and visualization of simulation results

## Repository Structure

### 📁 VASP_scripts/
Python utilities for VASP input file generation and manipulation:

- **`kpoints_calculator.py`** 🔢 - Automatically calculates and recommends KPOINTS mesh parameters based on POSCAR lattice dimensions. Supports both Gamma and Monkhorst-Pack meshes with configurable density constants.
- **`set_magmom_nelect.py`** 🧲 - Recursively updates INCAR files with magnetic moments (MAGMOM) and electron counts (NELECT) based on POSCAR atom counts and POTCAR ZVAL values.
- **`smass_set_recursive.py`** 🔄 - Sets SMASS parameter recursively across directories.
- **`standartize_poscar.py`** ✨ - Standardizes POSCAR file formatting.
- **`bulkmodulus_setup.py`** 📐 - Sets up bulk modulus calculations by generating multiple strained structures. Creates directories with volume-strained POSCAR files (default: -10% to +10% strain, 13 points) for equation of state (EOS) fitting. Automatically modifies INCAR files for single-point energy calculations and handles cleanup of template files.

### 📁 MACE_scripts/
Scripts for Machine Learning Atomic Cluster Expansion workflows:

- **`eval_configs.sh`** 🚀 - SLURM submission script for evaluating MACE models on configuration files. Automatically detects model files and handles GPU allocation.
- **`mace_1gpu.sh`** 🎯 - SLURM submission script for training MACE models using a single GPU. Automatically finds `.xyz` training files in the submission directory and runs training with standard MACE parameters (64x0e + 64x1o + 64x2e hidden irreps, r_max=5.0, float64 precision).
- **`mace_2gpus.sh`** 🔀 - SLURM submission script for distributed MACE training across 2 GPUs. Uses the same training parameters as `mace_1gpu.sh` but enables distributed training for faster model convergence on larger datasets.
- **`mace_4gpus.sh`** ⚡ - SLURM submission script for distributed MACE training across 4 GPUs. Provides maximum training speed for large-scale MACE model training using the same configuration as the single-GPU script.

### 📁 INCAR_NPT_preparation/
Tools for preparing NPT (isothermal-isobaric) molecular dynamics simulations:

- **`npt_incar_optimizer.py`** ⚙️ - Advanced parameter optimizer for VASP NPT MD simulations. Generates physically motivated Langevin thermostat parameters that:
  - ⚖️ Scale with atomic masses (lighter atoms get stronger damping)
  - 🌡️ Adjust with temperature
  - ⏱️ Compute safe POTIM values based on system composition
  - 📏 Optimize PMASS and LANGEVIN_GAMMA_L based on system size
- **`prepare_directories_for_md.py`** 📂 - Sets up directory structures for MD runs at multiple temperatures, handling file organization and cleanup.

### 📁 data_management/
Data extraction, analysis, and visualization tools:

- **`parity_plot_per_atom.py`** 📈 - Generates parity plots comparing MACE predictions vs reference (DFT) energies and forces. Creates visualizations for total energy, per-atom energy, and force components.
- **`converged_global_extract_frames_500+.py`** 💾 - Extracts converged frames from VASP OUTCAR files and writes them to XYZ format. Memory-efficient processing for large trajectory files.
- **`plot_md_temperature.sh`** 📉 - Shell script for plotting temperature evolution from MD simulations.

### 📁 Data_plotting/
Advanced plotting and analysis tools for simulation results:

- **`plot_bulk_modulus.py`** 📊 - Analyzes bulk modulus calculations from VASP OUTCAR files. Extracts volume-energy data, fits to Birch-Murnaghan equation of state, and generates comprehensive plots with normalized energy vs volume curves. Automatically detects compounds from directory structure and generates summary tables with fitted parameters (B₀, V₀, B₀', E₀).
- **`plot_md_pvt.py`** 🌡️ - Interactive plotter for VASP MD simulations showing temperature, volume, and pressure vs step. Supports both GUI (X11) and ASCII output modes. Automatically handles remote X11 connections with robust backend selection (TkAgg/Qt5Agg) and includes fallback ASCII plotting for environments without display.

### 📁 SLURM_management/
Utilities for managing computational jobs on SLURM clusters:

- **`submit_all_jobs.sh`** 📤 - Recursively finds and submits all `submit.vasp6.sh` scripts in a directory tree.
- **`check_jobs.sh`** 👀 - Monitors SLURM job status, showing job state, progress (steps completed vs total), and output paths.

### 📁 software/
Compilation guides and installation scripts for high-performance computing clusters:

- **`Polaris_software/`** 🏔️ - Guides and scripts for Polaris cluster:
  - 📘 VASP GPU compilation guide
  - 📗 LAMMPS with Kokkos and MACE installation guide
  - 🔧 Compilation scripts
- **`Palmetto_software/`** 🌴 - Scripts for Palmetto cluster:
  - 🔧 LAMMPS MACE build scripts

## ✨ Key Features

### 🤖 Automated Parameter Optimization
- **🔢 KPOINTS**: Automatic mesh generation based on lattice parameters
- **🌡️ NPT MD**: Mass- and temperature-aware Langevin thermostat parameters
- **🧲 Magnetic properties**: Automatic MAGMOM and NELECT calculation

### ⚡ Workflow Automation
- 🔄 Batch processing across directory structures
- 📁 Recursive operations for large datasets
- 🖥️ Integration with SLURM job schedulers

### 📊 Data Analysis
- 📈 Parity plots for ML model validation
- 💾 Converged frame extraction from trajectories
- 🌡️ Temperature, volume, and pressure analysis from MD simulations
- 📐 Bulk modulus analysis with Birch-Murnaghan EOS fitting

## 📦 Dependencies

### 🐍 Python Packages
- `numpy` - Numerical computations
- `matplotlib` - Plotting and visualization
- `scipy` - Scientific computing (for curve fitting in bulk modulus analysis)
- `ase` (Atomic Simulation Environment) - Structure manipulation and I/O
- `pymatgen` - Materials analysis (optional, for some scripts)

### 💻 External Software
- ⚛️ VASP 6.x
- 🤖 MACE (Machine Learning Atomic Cluster Expansion)
- 🖥️ SLURM (for job management scripts)
- 🔬 LAMMPS (for some installation scripts)

## 🚀 Usage Examples

### 🔢 Calculate KPOINTS for all structures
```bash
python VASP_scripts/kpoints_calculator.py /path/to/structures --k 25
```

### ⚙️ Optimize NPT MD parameters
```bash
python INCAR_NPT_preparation/npt_incar_optimizer.py /path/to/md/runs --backup
```

### 📂 Prepare MD directories for multiple temperatures
```bash
python INCAR_NPT_preparation/prepare_directories_for_md.py /path/to/structures --temps 300,500,700,900
```

### 🤖 Train MACE model (single GPU)
```bash
cd /path/to/training/data  # Directory containing .xyz training file
sbatch MACE_scripts/mace_1gpu.sh
```

### 🔀 Train MACE model (2 GPUs, distributed)
```bash
cd /path/to/training/data  # Directory containing .xyz training file
sbatch MACE_scripts/mace_2gpus.sh
```

### ⚡ Train MACE model (4 GPUs, distributed)
```bash
cd /path/to/training/data  # Directory containing .xyz training file
sbatch MACE_scripts/mace_4gpus.sh
```

### 🤖 Evaluate MACE model
```bash
sbatch MACE_scripts/eval_configs.sh --model model.model --configs structures.xyz
```

### 📤 Submit all VASP jobs
```bash
bash SLURM_management/submit_all_jobs.sh /path/to/calculations
```

### 👀 Check job status
```bash
bash SLURM_management/check_jobs.sh
```

### 📈 Generate parity plots
```bash
python data_management/parity_plot_per_atom.py mace_vs_ref.xyz
```

### 📐 Set up bulk modulus calculations
```bash
python VASP_scripts/bulkmodulus_setup.py /path/to/structures --min -10.0 --max 10.0 --npts 13
```

### 📊 Analyze bulk modulus results
```bash
python Data_plotting/plot_bulk_modulus.py /path/to/bulk_modulus/calculations
```

### 🌡️ Plot MD temperature, volume, and pressure
```bash
python Data_plotting/plot_md_pvt.py OUTCAR
# Or force ASCII output:
python Data_plotting/plot_md_pvt.py OUTCAR --ascii
```

## 💡 Notes

- ✅ Most scripts include `--dry-run` options for testing before making changes
- 💾 Many scripts preserve existing files by creating backups
- 🔄 Scripts are designed to work recursively across directory structures
- 🎮 GPU-related scripts are configured for NVIDIA A100 GPUs and CUDA-aware MPI

## 🤝 Contributing

This repository contains scripts developed for specific computational workflows. When adapting for your own use:
- 🔍 Review script parameters and adjust defaults as needed
- 🔧 Update cluster-specific paths in SLURM scripts
- ⚛️ Modify atomic data and parameters in optimization scripts for your systems

## 📄 License

This repository contains utility scripts for computational materials science workflows. Use and modify as needed for your research
hello
