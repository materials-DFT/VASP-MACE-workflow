#!/usr/bin/env python3
"""
Plot frame distribution or temperature distribution from a dataset directory or CSV file(s).

Usage:
    # Scan current directory (default)
    python plot_frame_distribution.py

    # Scan a specific dataset directory
    python plot_frame_distribution.py /path/to/npt_dataset
    python plot_frame_distribution.py /path/to/npt_dataset --save dist.png

    # Frame distribution from CSV file(s)
    python plot_frame_distribution.py --csv 54/classic/extraction_log.csv
    python plot_frame_distribution.py --csv 54/extraction_log.csv 52/extraction_log.csv

    # Temperature distribution (requires CSV with temperature data)
    python plot_frame_distribution.py --csv 52/extraction_log.csv --temperature
"""

import argparse
from collections import defaultdict
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import re
import sys
import os


# ---------------------------------------------------------------------------
# Directory-scan helpers
# ---------------------------------------------------------------------------

def _categorize_run_id(run_id):
    """
    Map a run_id string (from extended-XYZ comment line) to
    (category, subcategory) for plotting.

    Returns (None, None) if the run_id cannot be categorised.
    """
    rid = run_id.strip()

    # --- MD frames (underscore-joined run_ids from mlff extraction) ----------
    if rid.startswith('kmno2_md_'):
        mp_match = re.search(r'(mp\d+)_(\d+K)$', rid)
        if mp_match:
            return ('MD: kmno2', mp_match.group(1))
        return ('MD: kmno2', rid)

    if rid.startswith('kmno_md_'):
        comp_match = re.search(r'_([A-Z][A-Za-z0-9]+)_(\d+K)$', rid)
        if comp_match:
            return ('MD: kmno', comp_match.group(1))
        return ('MD: kmno', rid)

    if rid.startswith('mno2_phases+k_md_'):
        if 'unitcells' in rid:
            phase_match = re.search(r'isif3_(\w+?)_\d+K$', rid)
            if phase_match:
                return ('MD: mno2 unitcells', phase_match.group(1))
            return ('MD: mno2 unitcells', rid)
        if 'supercells' in rid:
            phase_match = re.search(r'monkhorst-pack_calculated_(\w+?)_\d+K$', rid)
            if phase_match:
                return ('MD: mno2 supercells', phase_match.group(1))
            return ('MD: mno2 supercells', rid)

    # --- Path-style run_ids (tilde-prefixed absolute paths) ------------------
    if '/' in rid:
        path = rid.replace('~/', '')

        if 'amorphous_kmno2' in path:
            if '/amorphous/' in path:
                return ('Amorphous KMnO2', 'amorphous')
            if '/defective/' in path:
                return ('Amorphous KMnO2', 'defective')
            return ('Amorphous KMnO2', os.path.basename(path))

        if 'bulk_modulus' in path:
            parts = path.rstrip('/').split('/')
            for i, p in enumerate(parts):
                if p == 'bulk_modulus_calculations' and i + 1 < len(parts):
                    return ('Bulk modulus', parts[i + 1])
            return ('Bulk modulus', os.path.basename(path))

        if 'interfaces' in path:
            base = os.path.basename(path)
            pair_match = re.match(r'^([a-z]+_[a-z]+)_', base)
            if pair_match:
                return ('Interfaces', pair_match.group(1))
            return ('Interfaces', base)

        if 'optimized_structures' in path:
            if 'kmno2/' in path:
                name = path.rstrip('/').split('/')[-1]
                return ('Optimized: kmno2', name)
            if 'kmno/' in path:
                name = path.rstrip('/').split('/')[-1]
                return ('Optimized: kmno', name)
            if 'mno2_phases+k' in path:
                name = path.rstrip('/').split('/')[-1]
                kind = 'supercells' if 'supercells' in path else 'unitcells'
                return (f'Optimized: mno2 {kind}', name)
            return ('Optimized', os.path.basename(path))

    return (None, None)


def _find_xyz_file(directory):
    """Find the combined dataset .xyz in *directory* (top-level only)."""
    candidates = []
    for f in os.listdir(directory):
        if f.endswith('.xyz') and os.path.isfile(os.path.join(directory, f)):
            candidates.append(f)
    if not candidates:
        return None
    for c in candidates:
        if 'dataset' in c.lower():
            return os.path.join(directory, c)
    candidates.sort(key=lambda c: os.path.getsize(os.path.join(directory, c)), reverse=True)
    return os.path.join(directory, candidates[0])


def _count_frames_in_xyz(xyz_path):
    """
    Parse an extended-XYZ file and return a list of run_id strings
    (one per frame).  Only reads comment lines, so it is fast even
    for large files.
    """
    run_ids = []
    with open(xyz_path) as fh:
        line_no = 0
        skip_to = 0
        for line in fh:
            line_no += 1
            if line_no < skip_to:
                continue
            stripped = line.strip()
            if not stripped:
                continue
            try:
                n_atoms = int(stripped)
            except ValueError:
                continue
            comment = next(fh, '')
            line_no += 1
            m = re.search(r'run_id=(\S+)', comment)
            run_ids.append(m.group(1) if m else '__unknown__')
            skip_to = line_no + n_atoms + 1
    return run_ids


def process_directory(directory):
    """
    Scan a dataset directory.
    Returns dict of {category: frame_count} from the combined .xyz.
    """
    xyz_path = _find_xyz_file(directory)
    if xyz_path is None:
        raise FileNotFoundError(
            f"No .xyz file found in {directory}. "
            "Expected a combined dataset file (e.g. npt_dataset.xyz)."
        )

    print(f"Reading {os.path.basename(xyz_path)} ...")
    run_ids = _count_frames_in_xyz(xyz_path)
    print(f"  Found {len(run_ids)} frames")

    cat_counts = defaultdict(int)
    uncategorised = 0
    for rid in run_ids:
        cat, _sub = _categorize_run_id(rid)
        if cat is None:
            uncategorised += 1
        else:
            cat_counts[cat] += 1
    if uncategorised:
        cat_counts['Uncategorised'] = uncategorised

    return dict(cat_counts)


# ---------------------------------------------------------------------------
# CSV helpers (original functionality)
# ---------------------------------------------------------------------------

def extract_structure_name_54(dir_path):
    """Extract structure name from directory path (for 54/classic format)."""
    clean_path = dir_path.lstrip('./')
    dir_name = clean_path.split('/')[-1] if '/' in clean_path else clean_path

    if 'vasp_neutralized_structures' in dir_name or dir_name.isdigit():
        return None

    parts = dir_name.rsplit('.', 1)
    if len(parts) == 2 and len(parts[1]) == 36:
        structure = parts[0]
        phase_match = re.match(r'^([a-z]+(?:_[a-z]+)+)_\d+_\d+_\d+', structure)
        if phase_match:
            return phase_match.group(1)
        return structure
    return dir_name


def extract_structure_name_52(path):
    """Extract structure name from source_xyz_path (for 52 format)."""
    match = re.search(r'/([^/]+)/(\d+K)/', path)
    if match:
        return match.group(1)
    return None


def process_csv_54(csv_path):
    """Process CSV file from 54/classic format."""
    df = pd.read_csv(csv_path)
    df = df[df['source_file'] != 'source_file']

    def get_dir(path):
        if '/' in path:
            return '/'.join(path.split('/')[:-1])
        return path

    df['directory'] = df['source_file'].apply(get_dir)
    df['structure'] = df['directory'].apply(extract_structure_name_54)
    df = df[df['structure'].notna()]

    summary = df.groupby('structure')['directory'].nunique().reset_index(name='total_frames')
    summary = summary.sort_values('total_frames', ascending=False)
    summary['source'] = 'Interface Structures'

    return summary


def process_csv_52(csv_path):
    """Process CSV file from 52 format."""
    df = pd.read_csv(csv_path)
    df['structure'] = df['source_xyz_path'].apply(extract_structure_name_52)
    df = df[df['structure'].notna()]

    summary = df.groupby('structure').size().reset_index(name='total_frames')
    summary = summary.sort_values('total_frames', ascending=False)
    summary['source'] = 'Neutralized Structures'

    return summary


def process_csv_52_for_temperature(csv_path):
    """Process CSV file from 52 format for temperature distribution."""
    df = pd.read_csv(csv_path)
    df['structure'] = df['source_xyz_path'].apply(extract_structure_name_52)
    df = df[df['structure'].notna()]

    if 'temperature' not in df.columns:
        raise ValueError("CSV file does not contain 'temperature' column. Cannot plot temperature distribution.")

    df = df[df['temperature'].notna()]

    return df


def detect_csv_format(csv_path):
    """Detect CSV format by checking columns."""
    df = pd.read_csv(csv_path, nrows=1)
    if 'source_file' in df.columns:
        return '54'
    elif 'source_xyz_path' in df.columns:
        return '52'
    else:
        raise ValueError(f"Unknown CSV format. Expected 'source_file' or 'source_xyz_path' column.")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def create_dir_plot(included, output_file=None):
    """Create a bar chart from directory-scan results with count and percentage labels."""
    sorted_items = sorted(included.items(), key=lambda x: -x[1])
    categories = [c for c, _ in sorted_items]
    counts = [n for _, n in sorted_items]
    total = sum(counts)

    fig, ax = plt.subplots(figsize=(max(10, len(categories) * 0.95), 7))

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B9A6D',
              '#7B68AE', '#E8793A', '#4BA3C3', '#D65F8A', '#8CB43F',
              '#C07A3E']
    bar_colors = [colors[i % len(colors)] for i in range(len(categories))]

    x = np.arange(len(categories))
    bars = ax.bar(x, counts, color=bar_colors, alpha=0.85, edgecolor='white',
                  linewidth=0.5)

    for bar, count in zip(bars, counts):
        pct = 100.0 * count / total if total else 0
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                f'{count}  ({pct:.1f}%)',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=40, ha='right', fontsize=9)
    ax.set_ylabel('Number of Frames', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Dataset Frame Distribution  ({total} frames)',
        fontsize=14, fontweight='bold', pad=15,
    )
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved as '{output_file}'")
        plt.close()
    else:
        plt.show()


def create_frame_plot(summaries, output_file=None):
    """Create and display/save the frame distribution plot."""
    combined = pd.concat([s[['structure', 'total_frames', 'source']] for s in summaries],
                        ignore_index=True)

    fig, ax = plt.subplots(figsize=(16, 10))

    all_structures = combined.groupby('structure')['total_frames'].sum().sort_values(ascending=False).index

    sources = [s['source'].iloc[0] for s in summaries]
    source_counts = {source: [] for source in sources}

    for struct in all_structures:
        for source in sources:
            val = combined[(combined['structure'] == struct) & (combined['source'] == source)]['total_frames'].values
            source_counts[source].append(val[0] if len(val) > 0 else 0)

    x = np.arange(len(all_structures))
    width = 0.35 if len(sources) == 2 else 0.8 / len(sources)

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    bars_list = []
    for i, source in enumerate(sources):
        offset = (i - len(sources)/2 + 0.5) * width if len(sources) > 1 else 0
        bars = ax.bar(x + offset, source_counts[source], width,
                     label=source, color=colors[i % len(colors)], alpha=0.8)
        bars_list.append(bars)

    ax.set_xlabel('Structure Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Frames', fontsize=12, fontweight='bold')

    totals = {source: sum(source_counts[source]) for source in sources}
    grand_total = sum(totals.values())

    title = 'Frame Distribution: ' + ' vs '.join(sources)
    if len(totals) == 2:
        title += f'\n({totals[sources[0]]} {sources[0].split()[0]} + {totals[sources[1]]} {sources[1].split()[0]} = {grand_total} total)'
    else:
        title += f'\n(Total: {grand_total} frames)'

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(all_structures, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for bars in bars_list:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                pct = 100.0 * height / grand_total if grand_total else 0
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}  ({pct:.1f}%)',
                       ha='center', va='bottom', fontsize=7)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved as '{output_file}'")
        plt.close()
    else:
        plt.show()


def create_temperature_plot(df, output_file=None):
    """Create and display/save the temperature distribution plot."""
    fig, ax = plt.subplots(figsize=(16, 10))

    struct_temp_counts = df.groupby(['structure', 'temperature']).size().reset_index(name='count')

    structures = sorted(df['structure'].unique())
    temps = sorted(df['temperature'].unique())

    x = np.arange(len(structures))
    width = 0.12

    colors_map = plt.cm.viridis(np.linspace(0.2, 0.8, len(temps)))

    for i, temp in enumerate(temps):
        counts = []
        for struct in structures:
            count = struct_temp_counts[(struct_temp_counts['structure'] == struct) &
                                     (struct_temp_counts['temperature'] == temp)]['count'].values
            counts.append(count[0] if len(count) > 0 else 0)

        offset = (i - len(temps)/2 + 0.5) * width
        ax.bar(x + offset, counts, width, label=f'{temp:.0f}K',
               color=colors_map[i], alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Structure Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Frames', fontsize=12, fontweight='bold')
    ax.set_title('MD Temperature Distribution by Structure Type', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(structures, rotation=45, ha='right', fontsize=8)
    ax.legend(title='Temperature', fontsize=9, title_fontsize=10, ncol=len(temps), loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved as '{output_file}'")
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Plot frame distribution from a dataset directory or CSV file(s).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('path', nargs='?', default='.',
                       help='Dataset directory to scan (default: current directory)')
    parser.add_argument('--csv', nargs='+', metavar='CSV',
                       help='Use CSV mode: path(s) to extraction log CSV file(s)')
    parser.add_argument('--save', '-s', metavar='OUTPUT',
                       help='Save plot to PNG file instead of displaying')
    parser.add_argument('--temperature', '-t', action='store_true',
                       help='Plot temperature distribution (CSV mode only)')

    args = parser.parse_args()

    # ---- CSV mode -----------------------------------------------------------
    if args.csv:
        for csv_file in args.csv:
            if not os.path.exists(csv_file):
                print(f"Error: File not found: {csv_file}", file=sys.stderr)
                sys.exit(1)

        if args.temperature:
            if len(args.csv) > 1:
                print("Warning: Multiple CSV files provided. Using only first for temperature distribution.",
                      file=sys.stderr)
            try:
                df = process_csv_52_for_temperature(args.csv[0])
                print(f"Processed {args.csv[0]}: {len(df)} frames with temperature data")
                print(f"Temperature range: {df['temperature'].min():.0f}K - {df['temperature'].max():.0f}K")
                create_temperature_plot(df, output_file=args.save)
            except Exception as e:
                print(f"Error processing {args.csv[0]}: {e}", file=sys.stderr)
                sys.exit(1)
            return

        summaries = []
        for csv_file in args.csv:
            try:
                csv_format = detect_csv_format(csv_file)
                if csv_format == '54':
                    summary = process_csv_54(csv_file)
                else:
                    summary = process_csv_52(csv_file)
                summaries.append(summary)
                print(f"Processed {csv_file}: {len(summary)} structure types, "
                      f"{summary['total_frames'].sum()} frames")
            except Exception as e:
                print(f"Error processing {csv_file}: {e}", file=sys.stderr)
                sys.exit(1)

        create_frame_plot(summaries, output_file=args.save)
        return

    # ---- Directory mode (default) -------------------------------------------
    target = args.path
    if not os.path.isdir(target):
        print(f"Error: Not a directory: {target}", file=sys.stderr)
        sys.exit(1)

    try:
        included = process_directory(target)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    total = sum(included.values())
    print(f"\nDataset breakdown ({total} frames):")
    for cat in sorted(included, key=included.get, reverse=True):
        pct = 100.0 * included[cat] / total if total else 0
        print(f"  {cat:30s}  {included[cat]:>6d}  ({pct:5.1f}%)")

    create_dir_plot(included, output_file=args.save)


if __name__ == '__main__':
    main()
