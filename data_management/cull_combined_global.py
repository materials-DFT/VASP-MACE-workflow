#!/usr/bin/env python3
"""
Combine all structures from multiple files, then:
1. Filter by size (≤200 atoms)
2. Maximize diversity and minimize redundancy globally

This approach ensures maximum diversity across the entire dataset.
"""

import argparse
import json
import os
import sys
import glob as glob_module
import time
from pathlib import Path

import numpy as np
from ase.io import read, write

from quests.compression.compress import DatasetCompressor
from quests.descriptor import DEFAULT_CUTOFF, DEFAULT_K, get_descriptors
from quests.entropy import DEFAULT_BANDWIDTH, DEFAULT_BATCH


def combine_and_filter(directory, max_atoms=200, exclude_patterns=None, max_files=None):
    """
    Combine all structures from XYZ files in directory and filter by size.
    
    Returns:
        filtered_structures: List of structures with ≤max_atoms
        file_mapping: List of (file_index, structure_index_in_file, filename) tuples
    """
    xyz_files = sorted(glob_module.glob(os.path.join(directory, "*.xyz")))
    
    # Exclude already processed files
    if exclude_patterns:
        for pattern in exclude_patterns:
            xyz_files = [f for f in xyz_files if pattern not in f]
    
    # Limit files for test mode
    if max_files:
        xyz_files = xyz_files[:max_files]
    
    print(f"Found {len(xyz_files)} files to process")
    
    all_structures = []
    file_mapping = []
    
    print("Combining structures...")
    for file_idx, xyz_file in enumerate(xyz_files):
        try:
            structures = read(xyz_file, index=":")
            for struct_idx, struct in enumerate(structures):
                if len(struct) <= max_atoms:
                    all_structures.append(struct)
                    file_mapping.append((file_idx, struct_idx, os.path.basename(xyz_file)))
            
            if (file_idx + 1) % 50 == 0:
                print(f"  Processed {file_idx + 1}/{len(xyz_files)} files, {len(all_structures):,} structures so far...", end='\r')
        except Exception as e:
            print(f"\n  Warning: Error reading {os.path.basename(xyz_file)}: {e}")
    
    print(f"\n  Processed {len(xyz_files)} files")
    print(f"  Total structures with ≤{max_atoms} atoms: {len(all_structures):,}")
    
    return all_structures, file_mapping


def _compute_target_size(n_items: int, size: float) -> int:
    """
    Convert a user-provided size (fraction in (0,1) or absolute >=1) into an integer target,
    clamped to [1, n_items] (unless n_items == 0).
    """
    if n_items <= 0:
        return 0
    if 0 < size < 1:
        target = int(np.round(n_items * size))
    else:
        target = int(size)
    target = max(1, min(target, n_items))
    return target


def hierarchical_pre_cull(
    directory: str,
    *,
    max_atoms: int,
    pre_size: float,
    pre_method: str,
    cutoff: float,
    nbrs: int,
    bandwidth: float,
    batch_size: int,
    exclude_patterns=None,
    max_files=None,
    chunk_files: int = 25,
    progress_interval: int = 300,
):
    """
    Hierarchical pre-cull to reduce the dataset before a global cull.

    Strategy:
    - Read and size-filter structures in chunks of XYZ files.
    - For each chunk, run a fast compressor selection (default FPS) to keep only a fraction
      (or absolute count) from that chunk.
    - Return the union of all chunk selections + a file_mapping compatible with the final output.
    """
    xyz_files = sorted(glob_module.glob(os.path.join(directory, "*.xyz")))

    if exclude_patterns:
        for pattern in exclude_patterns:
            xyz_files = [f for f in xyz_files if pattern not in f]

    if max_files:
        xyz_files = xyz_files[:max_files]

    if len(xyz_files) == 0:
        return [], []

    def descriptor_fn(ds):
        return get_descriptors([ds], k=nbrs, cutoff=cutoff)

    selected_structures = []
    selected_mapping = []

    total_files = len(xyz_files)
    for chunk_start in range(0, total_files, max(1, chunk_files)):
        chunk = xyz_files[chunk_start:chunk_start + max(1, chunk_files)]

        chunk_structures = []
        chunk_mapping = []

        for file_idx, xyz_file in enumerate(chunk, start=chunk_start):
            try:
                structures = read(xyz_file, index=":")
                for struct_idx, struct in enumerate(structures):
                    if len(struct) <= max_atoms:
                        chunk_structures.append(struct)
                        chunk_mapping.append((file_idx, struct_idx, os.path.basename(xyz_file)))
            except Exception as e:
                print(f"  Warning: Error reading {os.path.basename(xyz_file)}: {e}")

        if len(chunk_structures) == 0:
            print(f"  Chunk {chunk_start}-{chunk_start+len(chunk)-1}: 0 structures after size filter; skipping.")
            continue

        chunk_target = _compute_target_size(len(chunk_structures), pre_size)
        print(
            f"  Chunk {chunk_start}-{chunk_start+len(chunk)-1}: "
            f"{len(chunk_structures):,} -> {chunk_target:,} using {pre_method.upper()}"
        )
        sys.stdout.flush()

        compressor = DatasetCompressor(
            chunk_structures,
            descriptor_fn,
            bandwidth=bandwidth,
            batch_size=batch_size,
        )

        start_time = time.time()
        import threading
        stop_progress = threading.Event()

        def progress_loop():
            while not stop_progress.is_set():
                if stop_progress.wait(progress_interval):
                    break
                elapsed = time.time() - start_time
                print(
                    f"    [Pre-cull progress] Chunk {chunk_start}-{chunk_start+len(chunk)-1} "
                    f"still computing... ({elapsed/60:.1f} minutes elapsed)",
                    flush=True,
                )

        progress_thread = threading.Thread(target=progress_loop, daemon=True)
        progress_thread.start()
        try:
            idx = compressor.get_indices(pre_method, chunk_target)
        finally:
            stop_progress.set()
            progress_thread.join(timeout=1)

        selected_structures.extend([chunk_structures[i] for i in idx])
        selected_mapping.extend([chunk_mapping[i] for i in idx])

    return selected_structures, selected_mapping


def print_summary(compressor, selected, total_original, filtered_size, output_file, max_atoms):
    """Print summary statistics."""
    comp_H = compressor.entropy(selected)
    comp_D = compressor.diversity(selected)
    overlap = compressor.overlap(selected)
    comp_structs = len(selected)
    orig_envs = compressor.num_envs()
    comp_envs = compressor.num_envs(selected)
    
    reduction_from_filtered = (1 - comp_structs/filtered_size) * 100 if filtered_size > 0 else 0
    reduction_from_original = (1 - comp_structs/total_original) * 100
    
    print(f"\n{'='*70}")
    print(f"  GLOBAL CULLING SUMMARY")
    print(f"{'='*70}")
    print(f"  Filter:               ≤{max_atoms} atoms")
    print(f"  Method:               MSC (Maximum Structural Coverage)")
    print(f"  Original structures:  {total_original:,}")
    print(f"  After size filter:    {filtered_size:,} ({filtered_size/total_original*100:.1f}%)")
    print(f"  Final (culled):      {comp_structs:,}")
    print(f"  Removed (from filtered): {filtered_size - comp_structs:,} ({reduction_from_filtered:.1f}%)")
    print(f"  Total reduction:      {total_original - comp_structs:,} ({reduction_from_original:.1f}%)")
    print(f"\n  Diversity metrics:")
    print(f"    Entropy:            {comp_H:.4f} nats")
    print(f"    Diversity:          {comp_D:.4f}")
    print(f"    Overlap:             {overlap*100:.1f}%")
    print(f"    Environments:       {comp_envs:,} / {orig_envs:,}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Combine all structures, filter by size (≤200 atoms), then maximize diversity globally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # Combine all files, filter ≤200 atoms, keep 25% (maximizes global diversity)
  python cull_combined_global.py outputs_kmno2_generic/ --max-atoms 200 --size 0.25 --output-dir selected_structures
        """
    )
    
    parser.add_argument(
        "directory",
        help="Directory containing XYZ files to combine"
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=200,
        help="Maximum number of atoms per structure (default: 200)"
    )
    parser.add_argument(
        "-s", "--size",
        type=float,
        default=0.25,
        help="Size: fraction (0-1) or absolute number after filtering (default: 0.25)"
    )
    parser.add_argument(
        "-m", "--method",
        type=str,
        default="msc",
        choices=["msc", "fps"],
        help="Culling method: 'msc' (recommended) or 'fps' (default: msc)"
    )
    parser.add_argument(
        "-c", "--cutoff",
        type=float,
        default=5.5,
        help=f"Cutoff radius in Å (default: 5.5)"
    )
    parser.add_argument(
        "-k", "--nbrs",
        type=int,
        default=24,
        help=f"Number of nearest neighbors (default: 24)"
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=DEFAULT_BANDWIDTH,
        help=f"Bandwidth for kernel (default: {DEFAULT_BANDWIDTH})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH,
        help=f"Batch size for computations (default: {DEFAULT_BATCH})"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="selected_structures",
        help="Output directory (default: selected_structures)"
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        default=True,
        help="Save selection indices and metrics to JSON (default: True)"
    )
    parser.add_argument(
        "--test-mode",
        type=int,
        default=None,
        help="Test mode: process only first N files (for debugging)"
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=300,
        help="Progress update interval in seconds (default: 300)"
    )
    parser.add_argument(
        "--hierarchical",
        action="store_true",
        default=False,
        help="Use a hierarchical strategy: pre-cull in file-chunks (fast) then do final global cull "
             "(recommended if global MSC exceeds walltime)."
    )
    parser.add_argument(
        "--pre-size",
        type=float,
        default=0.5,
        help="Hierarchical pre-cull size per chunk: fraction (0-1) or absolute number (default: 0.5)."
    )
    parser.add_argument(
        "--pre-method",
        type=str,
        default="fps",
        choices=["fps", "msc"],
        help="Hierarchical pre-cull method (default: fps)."
    )
    parser.add_argument(
        "--chunk-files",
        type=int,
        default=25,
        help="Number of XYZ files per hierarchical pre-cull chunk (default: 25)."
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"ERROR: {args.directory} is not a directory")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    
    # Step 1: Combine and filter
    print(f"\n{'='*70}")
    print(f"COMBINING AND FILTERING STRUCTURES")
    print(f"{'='*70}")
    if args.test_mode:
        print(f"⚠️  TEST MODE: Processing only first {args.test_mode} files")
    
    if args.hierarchical:
        print(f"\n{'='*70}")
        print(f"HIERARCHICAL PRE-CULL (to reduce global runtime)")
        print(f"{'='*70}")
        print(
            f"Pre-cull: method={args.pre_method.upper()}, pre_size={args.pre_size}, "
            f"chunk_files={args.chunk_files}"
        )
        filtered_structures, file_mapping = hierarchical_pre_cull(
            args.directory,
            max_atoms=args.max_atoms,
            pre_size=args.pre_size,
            pre_method=args.pre_method,
            cutoff=args.cutoff,
            nbrs=args.nbrs,
            bandwidth=args.bandwidth,
            batch_size=args.batch_size,
            exclude_patterns=["_culled_", "_filtered_"],
            max_files=args.test_mode,
            chunk_files=args.chunk_files,
            progress_interval=args.progress_interval,
        )

        print(f"\nHierarchical pre-cull result: {len(filtered_structures):,} structures kept")
        pre_file = output_dir / "preculled_pool.xyz"
        pre_map = output_dir / "preculled_pool_mapping.json"
        print(f"Writing pre-culled pool: {pre_file}")
        write(pre_file, filtered_structures, format="extxyz")
        with open(pre_map, "w") as f:
            json.dump(
                {
                    "directory": args.directory,
                    "max_atoms": args.max_atoms,
                    "pre_method": args.pre_method,
                    "pre_size": args.pre_size,
                    "chunk_files": args.chunk_files,
                    "n_preculled": len(filtered_structures),
                    "file_mapping": [
                        {"file_index": int(fi), "structure_index": int(si), "filename": fn}
                        for (fi, si, fn) in file_mapping
                    ],
                },
                f,
                indent=2,
            )
        print(f"Wrote pre-culled mapping: {pre_map}")
        sys.stdout.flush()
    else:
        filtered_structures, file_mapping = combine_and_filter(
            args.directory,
            max_atoms=args.max_atoms,
            exclude_patterns=["_culled_", "_filtered_"],
            max_files=args.test_mode
        )
    
    if len(filtered_structures) == 0:
        print("ERROR: No structures found with ≤{} atoms".format(args.max_atoms))
        sys.exit(1)
    
    # Count original structures
    xyz_files = glob_module.glob(os.path.join(args.directory, "*.xyz"))
    xyz_files = [f for f in xyz_files if "_culled_" not in f and "_filtered_" not in f]
    total_original = sum(len(read(f, index=":")) for f in xyz_files)
    
    # Determine target size
    target_size = _compute_target_size(len(filtered_structures), args.size)
    
    # Step 2: Maximize diversity globally
    print(f"\n{'='*70}")
    print(f"MAXIMIZING DIVERSITY AND MINIMIZING REDUNDANCY")
    print(f"{'='*70}")
    print(f"Computing descriptors (cutoff={args.cutoff}Å, neighbors={args.nbrs})...")
    print(f"This may take a while for {len(filtered_structures):,} structures...")
    print(f"Progress updates every {args.progress_interval} seconds...")
    sys.stdout.flush()
    
    descriptor_fn = lambda ds: get_descriptors([ds], k=args.nbrs, cutoff=args.cutoff)
    
    try:
        start_time = time.time()
        print(f"  Step 1/2: Computing descriptors for all structures...")
        print(f"  Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        sys.stdout.flush()
        
        compressor = DatasetCompressor(
            filtered_structures,
            descriptor_fn,
            bandwidth=args.bandwidth,
            batch_size=args.batch_size
        )
        
        elapsed = time.time() - start_time
        print(f"  ✓ Descriptors computed successfully")
        print(f"  Time elapsed: {elapsed/60:.1f} minutes ({elapsed:.1f} seconds)")
        sys.stdout.flush()
    except Exception as e:
        print(f"ERROR: Failed to create compressor: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"\n  Step 2/2: Culling to {target_size} structures using {args.method.upper()}...")
    print(f"  This is the most time-consuming step (may take 30-60+ minutes)...")
    print(f"  Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Progress updates every {args.progress_interval} seconds...")
    sys.stdout.flush()
    
    try:
        start_time = time.time()
        
        # Note: DatasetCompressor doesn't have a progress callback, so we'll use a thread
        # to print periodic updates
        import threading
        
        stop_progress = threading.Event()
        
        def progress_loop():
            while not stop_progress.is_set():
                if stop_progress.wait(args.progress_interval):
                    break
                elapsed = time.time() - start_time
                print(f"  [Progress] Still computing... ({elapsed/60:.1f} minutes elapsed)", flush=True)
        
        progress_thread = threading.Thread(target=progress_loop, daemon=True)
        progress_thread.start()
        
        try:
            selected_indices = compressor.get_indices(args.method, target_size)
        finally:
            stop_progress.set()
            progress_thread.join(timeout=1)
        
        elapsed = time.time() - start_time
        print(f"  ✓ Culling completed successfully")
        print(f"  Time elapsed: {elapsed/60:.1f} minutes ({elapsed:.1f} seconds)")
        sys.stdout.flush()
    except Exception as e:
        print(f"ERROR: Culling failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 3: Write output
    output_file = output_dir / "all_kmno2_culled.xyz"
    print(f"\nWriting {len(selected_indices):,} structures to {output_file}...")
    
    culled_structures = [filtered_structures[i] for i in selected_indices]
    write(output_file, culled_structures, format="extxyz")
    
    # Print summary
    print_summary(compressor, selected_indices, total_original, len(filtered_structures), output_file, args.max_atoms)
    
    # Save JSON if requested
    if args.save_json:
        json_file = output_dir / "culling_metadata.json"
        selected_mappings = [file_mapping[i] for i in selected_indices]
        
        results = {
            "input_directory": args.directory,
            "output_directory": str(output_dir.absolute()),
            "output_file": str(output_file),
            "max_atoms": args.max_atoms,
            "method": args.method,
            "k": args.nbrs,
            "cutoff": args.cutoff,
            "bandwidth": args.bandwidth,
            "total_original_structures": total_original,
            "filtered_size": len(filtered_structures),
            "final_size": len(selected_indices),
            "size_fraction": len(selected_indices) / len(filtered_structures),
            "selected_indices": [int(i) for i in selected_indices],
            "file_mapping": [
                {
                    "file_index": int(fi),
                    "structure_index": int(si),
                    "filename": fn
                }
                for fi, si, fn in selected_mappings
            ],
            "entropy": float(compressor.entropy(selected_indices)),
            "diversity": float(compressor.diversity(selected_indices)),
            "overlap": float(compressor.overlap(selected_indices)),
        }
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved metadata to: {json_file}")
        print(f"\n{'='*70}")
        print(f"SUCCESS! Culled structures saved to: {output_dir.absolute()}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

