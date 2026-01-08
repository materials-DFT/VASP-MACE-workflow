#!/usr/bin/env python3
"""
Stratified culling script to maximize amorphous structure inclusion.

This script extends the original cull_combined_global.py with stratified culling,
which ensures proportional representation from each structure type (defects, 
slabgen, bulk_stream, unknown). This prevents diversity-based methods from 
excluding similar structures like amorphous/defective ones.

Key feature: --stratified flag enables type-aware culling.
"""

import argparse
import json
import os
import sys
import glob as glob_module
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from ase.io import read, write

from quests.compression.compress import DatasetCompressor
from quests.descriptor import DEFAULT_CUTOFF, DEFAULT_K, get_descriptors
from quests.entropy import DEFAULT_BANDWIDTH, DEFAULT_BATCH

# Import functions from original script
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
try:
    from cull_combined_global import (
        combine_and_filter,
        _compute_target_size,
        hierarchical_pre_cull,
    )
except ImportError:
    print("ERROR: Could not import from cull_combined_global.py")
    print(f"  Make sure {parent_dir}/cull_combined_global.py exists")
    sys.exit(1)


def _classify_structure_type(struct, filename):
    """
    Classify structure type based on metadata and filename.
    Returns: 'defects', 'slabgen', 'bulk_stream', or 'unknown'
    """
    # Check metadata first
    if hasattr(struct, 'info') and struct.info:
        if 'defects_generic' in struct.info:
            return 'defects'
        elif 'slabgen_generic' in struct.info:
            return 'slabgen'
    
    # Fall back to filename
    fn_lower = filename.lower()
    if 'defective' in fn_lower or 'defects' in fn_lower:
        return 'defects'
    elif 'slabgen' in fn_lower:
        return 'slabgen'
    elif 'bulk_stream' in fn_lower:
        return 'bulk_stream'
    
    return 'unknown'


def stratified_cull(
    structures,
    file_mapping,
    target_size,
    method,
    descriptor_fn,
    bandwidth,
    batch_size,
    progress_interval=300,
):
    """
    Stratified culling: ensures proportional representation from each structure type.
    
    Strategy:
    1. Group structures by type (defects, slabgen, bulk_stream, unknown)
    2. Allocate target size proportionally to each type
    3. Run MSC/FPS within each type separately
    4. Combine results
    
    This prevents diversity-based methods from excluding similar structures
    (like amorphous/defective ones) that cluster together in descriptor space.
    """
    # Group structures by type
    type_groups = {}
    type_indices = {}
    
    for idx, (struct, (file_idx, struct_idx, filename)) in enumerate(zip(structures, file_mapping)):
        struct_type = _classify_structure_type(struct, filename)
        if struct_type not in type_groups:
            type_groups[struct_type] = []
            type_indices[struct_type] = []
        type_groups[struct_type].append(struct)
        type_indices[struct_type].append(idx)
    
    print(f"\nStratified culling: {len(type_groups)} structure types identified")
    for typ, group in sorted(type_groups.items()):
        print(f"  {typ}: {len(group):,} structures ({len(group)/len(structures)*100:.1f}%)")
    
    # Allocate target size proportionally
    type_targets = {}
    total_allocated = 0
    for typ, group in type_groups.items():
        if len(group) > 0:
            # Proportional allocation, but ensure at least 1 if group is large enough
            prop = len(group) / len(structures)
            target = max(1, int(np.round(target_size * prop))) if len(group) >= target_size * prop else 0
            target = min(target, len(group))  # Can't select more than available
            type_targets[typ] = target
            total_allocated += target
        else:
            type_targets[typ] = 0
    
    # Adjust if we're under/over target due to rounding
    if total_allocated < target_size:
        # Add to largest groups
        diff = target_size - total_allocated
        for typ in sorted(type_targets.keys(), key=lambda t: len(type_groups[t]), reverse=True):
            if type_targets[typ] < len(type_groups[typ]) and diff > 0:
                type_targets[typ] += 1
                diff -= 1
                total_allocated += 1
    elif total_allocated > target_size:
        # Remove from smallest groups
        diff = total_allocated - target_size
        for typ in sorted(type_targets.keys(), key=lambda t: type_targets[t]):
            if type_targets[typ] > 0 and diff > 0:
                type_targets[typ] -= 1
                diff -= 1
                total_allocated -= 1
    
    print(f"\nStratified allocation:")
    for typ in sorted(type_targets.keys()):
        if type_targets[typ] > 0:
            print(f"  {typ}: selecting {type_targets[typ]:,} from {len(type_groups[typ]):,}")
    
    # Cull within each type
    selected_indices = []
    import threading
    
    for typ, group in sorted(type_groups.items()):
        if type_targets[typ] == 0 or len(group) == 0:
            continue
        
        print(f"\n  Culling {typ} structures ({type_targets[typ]:,} target)...")
        sys.stdout.flush()
        
        # Get original indices for this type
        orig_indices = type_indices[typ]
        
        # Create compressor for this type
        compressor = DatasetCompressor(
            group,
            descriptor_fn,
            bandwidth=bandwidth,
            batch_size=batch_size
        )
        
        start_time = time.time()
        stop_progress = threading.Event()
        
        def progress_loop():
            while not stop_progress.is_set():
                if stop_progress.wait(progress_interval):
                    break
                elapsed = time.time() - start_time
                print(f"    [{typ}] Still computing... ({elapsed/60:.1f} minutes elapsed)", flush=True)
        
        progress_thread = threading.Thread(target=progress_loop, daemon=True)
        progress_thread.start()
        
        try:
            type_selected = compressor.get_indices(method, type_targets[typ])
        finally:
            stop_progress.set()
            progress_thread.join(timeout=1)
        
        # Map back to original indices
        selected_indices.extend([orig_indices[i] for i in type_selected])
        
        elapsed = time.time() - start_time
        print(f"    ✓ {typ}: selected {len(type_selected):,} structures ({elapsed:.1f}s)")
        sys.stdout.flush()
    
    return np.array(selected_indices)


def print_summary_with_types(compressor, selected, total_original, filtered_size, output_file, max_atoms, structures, file_mapping):
    """Print summary statistics including type distribution."""
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
    print(f"  Method:               MSC (Maximum Structural Coverage) + STRATIFIED")
    print(f"  Original structures:  {total_original:,}")
    print(f"  After size filter:    {filtered_size:,} ({filtered_size/total_original*100:.1f}%)")
    print(f"  Final (culled):      {comp_structs:,}")
    print(f"  Removed (from filtered): {filtered_size - comp_structs:,} ({reduction_from_filtered:.1f}%)")
    print(f"  Total reduction:      {total_original - comp_structs:,} ({reduction_from_original:.1f}%)")
    
    # Show type distribution
    type_counts = defaultdict(int)
    for idx in selected:
        struct = structures[idx]
        _, _, filename = file_mapping[idx]
        struct_type = _classify_structure_type(struct, filename)
        type_counts[struct_type] += 1
    
    print(f"\n  Selection by type:")
    for typ in sorted(type_counts.keys()):
        count = type_counts[typ]
        pct = count / comp_structs * 100
        print(f"    {typ:15s}: {count:3d} ({pct:5.1f}%)")
    
    print(f"\n  Diversity metrics:")
    print(f"    Entropy:            {comp_H:.4f} nats")
    print(f"    Diversity:          {comp_D:.4f}")
    print(f"    Overlap:             {overlap*100:.1f}%")
    print(f"    Environments:       {comp_envs:,} / {orig_envs:,}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Stratified culling: maximize amorphous structure inclusion by ensuring proportional representation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # Use stratified culling to ensure amorphous structures are included
  python cull_stratified.py outputs_kmno2_generic/ --max-atoms 100 --size 100 --output-dir selected_structures --hierarchical --pre-size 0.7
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
    
    # Step 2: Stratified culling (ALWAYS ON in this script)
    print(f"\n{'='*70}")
    print(f"STRATIFIED CULLING (ensuring proportional representation)")
    print(f"{'='*70}")
    print(f"Computing descriptors (cutoff={args.cutoff}Å, neighbors={args.nbrs})...")
    print(f"This may take a while for {len(filtered_structures):,} structures...")
    print(f"Progress updates every {args.progress_interval} seconds...")
    sys.stdout.flush()
    
    descriptor_fn = lambda ds: get_descriptors([ds], k=args.nbrs, cutoff=args.cutoff)
    
    # Use stratified culling
    print(f"\n  Step 2/2: Stratified culling to {target_size} structures using {args.method.upper()}...")
    print(f"  This ensures proportional representation of all structure types")
    print(f"  Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Progress updates every {args.progress_interval} seconds...")
    sys.stdout.flush()
    
    try:
        start_time = time.time()
        selected_indices = stratified_cull(
            filtered_structures,
            file_mapping,
            target_size,
            args.method,
            descriptor_fn,
            args.bandwidth,
            args.batch_size,
            args.progress_interval
        )
        
        elapsed = time.time() - start_time
        print(f"  ✓ Stratified culling completed successfully")
        print(f"  Time elapsed: {elapsed/60:.1f} minutes ({elapsed:.1f} seconds)")
        sys.stdout.flush()
    except Exception as e:
        print(f"ERROR: Stratified culling failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create compressor for summary metrics (on full filtered set)
    compressor = DatasetCompressor(
        filtered_structures,
        descriptor_fn,
        bandwidth=args.bandwidth,
        batch_size=args.batch_size
    )
    
    # Step 3: Write output
    output_file = output_dir / "all_kmno2_culled.xyz"
    print(f"\nWriting {len(selected_indices):,} structures to {output_file}...")
    
    culled_structures = [filtered_structures[i] for i in selected_indices]
    write(output_file, culled_structures, format="extxyz")
    
    # Print summary with type distribution
    print_summary_with_types(compressor, selected_indices, total_original, len(filtered_structures), output_file, args.max_atoms, filtered_structures, file_mapping)
    
    # Save JSON if requested
    if args.save_json:
        json_file = output_dir / "culling_metadata.json"
        selected_mappings = [file_mapping[i] for i in selected_indices]
        
        # Count types for JSON
        type_counts = defaultdict(int)
        for idx in selected_indices:
            struct = filtered_structures[idx]
            _, _, filename = file_mapping[idx]
            struct_type = _classify_structure_type(struct, filename)
            type_counts[struct_type] += 1
        
        results = {
            "input_directory": args.directory,
            "output_directory": str(output_dir.absolute()),
            "output_file": str(output_file),
            "max_atoms": args.max_atoms,
            "method": args.method,
            "stratified": True,
            "k": args.nbrs,
            "cutoff": args.cutoff,
            "bandwidth": args.bandwidth,
            "total_original_structures": total_original,
            "filtered_size": len(filtered_structures),
            "final_size": len(selected_indices),
            "size_fraction": len(selected_indices) / len(filtered_structures),
            "type_distribution": dict(type_counts),
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

