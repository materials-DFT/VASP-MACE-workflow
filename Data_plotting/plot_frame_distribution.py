#!/usr/bin/env python3
"""
Plot frame distribution or temperature distribution from extraction log CSV file(s).

Usage:
    # Frame distribution - Display plot using X11 (default) - single CSV
    python plot_frame_distribution.py 54/classic/extraction_log.csv
    
    # Frame distribution - Display plot using X11 (default) - multiple CSVs
    python plot_frame_distribution.py 54/classic/extraction_log.csv 52/extraction_log.csv
    
    # Frame distribution - Save as PNG file
    python plot_frame_distribution.py 54/classic/extraction_log.csv --save output.png
    
    # Temperature distribution - Display plot (requires CSV with temperature data)
    python plot_frame_distribution.py 52/extraction_log.csv --temperature
    
    # Temperature distribution - Save as PNG
    python plot_frame_distribution.py 52/extraction_log.csv --temperature --save temp_dist.png
"""

import argparse
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import re
import sys
import os


def extract_structure_name_54(dir_path):
    """Extract structure name from directory path (for 54/classic format)."""
    clean_path = dir_path.lstrip('./')
    dir_name = clean_path.split('/')[-1] if '/' in clean_path else clean_path
    
    # Skip vasp_neutralized_structures - we only want interface structures
    if 'vasp_neutralized_structures' in dir_name or dir_name.isdigit():
        return None  # Skip these
    
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
    df = df[df['source_file'] != 'source_file']  # Filter header row if present
    
    def get_dir(path):
        if '/' in path:
            return '/'.join(path.split('/')[:-1])
        return path
    
    df['directory'] = df['source_file'].apply(get_dir)
    df['structure'] = df['directory'].apply(extract_structure_name_54)
    df = df[df['structure'].notna()]  # Filter out vasp_neutralized_structures
    
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
    
    # Check if temperature column exists
    if 'temperature' not in df.columns:
        raise ValueError("CSV file does not contain 'temperature' column. Cannot plot temperature distribution.")
    
    # Filter out NaN temperatures
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


def create_frame_plot(summaries, output_file=None):
    """Create and display/save the frame distribution plot."""
    # Combine all summaries
    combined = pd.concat([s[['structure', 'total_frames', 'source']] for s in summaries], 
                        ignore_index=True)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Get unique structures and sort by total frames
    all_structures = combined.groupby('structure')['total_frames'].sum().sort_values(ascending=False).index
    
    # Prepare data for grouped bar chart
    sources = [s['source'].iloc[0] for s in summaries]
    source_counts = {source: [] for source in sources}
    
    for struct in all_structures:
        for source in sources:
            val = combined[(combined['structure'] == struct) & (combined['source'] == source)]['total_frames'].values
            source_counts[source].append(val[0] if len(val) > 0 else 0)
    
    x = np.arange(len(all_structures))
    width = 0.35 if len(sources) == 2 else 0.8 / len(sources)
    
    # Color scheme
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # Create bars
    bars_list = []
    for i, source in enumerate(sources):
        offset = (i - len(sources)/2 + 0.5) * width if len(sources) > 1 else 0
        bars = ax.bar(x + offset, source_counts[source], width, 
                     label=source, color=colors[i % len(colors)], alpha=0.8)
        bars_list.append(bars)
    
    # Customize the plot
    ax.set_xlabel('Structure Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Frames', fontsize=12, fontweight='bold')
    
    # Calculate totals
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
    
    # Add value labels on bars
    for bars in bars_list:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved as '{output_file}'")
        plt.close()
    else:
        # Use default backend (X11) for display
        plt.show()


def create_temperature_plot(df, output_file=None):
    """Create and display/save the temperature distribution plot."""
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Group by structure and temperature
    struct_temp_counts = df.groupby(['structure', 'temperature']).size().reset_index(name='count')
    
    # Get unique structures and temperatures
    structures = sorted(df['structure'].unique())
    temps = sorted(df['temperature'].unique())
    
    # Get temperature data for statistics
    temperatures = df['temperature'].values
    
    # Create grouped bar chart
    x = np.arange(len(structures))
    width = 0.12  # Width for each temperature bar
    
    # Color map for temperatures
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
        # Use default backend (X11) for display
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Plot frame distribution or temperature distribution from extraction log CSV file(s).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('csv_files', nargs='+',
                       help='Path(s) to CSV file(s)')
    parser.add_argument('--save', '-s', metavar='OUTPUT',
                       help='Save plot to PNG file instead of displaying')
    parser.add_argument('--temperature', '-t', action='store_true',
                       help='Plot temperature distribution instead of frame distribution')
    
    args = parser.parse_args()
    
    # Check if files exist
    for csv_file in args.csv_files:
        if not os.path.exists(csv_file):
            print(f"Error: File not found: {csv_file}", file=sys.stderr)
            sys.exit(1)
    
    # Temperature distribution mode
    if args.temperature:
        if len(args.csv_files) > 1:
            print("Warning: Multiple CSV files provided. Using only first CSV file for temperature distribution.", file=sys.stderr)
        
        try:
            df = process_csv_52_for_temperature(args.csv_files[0])
            print(f"Processed {args.csv_files[0]}: {len(df)} frames with temperature data")
            print(f"Temperature range: {df['temperature'].min():.0f}K - {df['temperature'].max():.0f}K")
            create_temperature_plot(df, output_file=args.save)
        except Exception as e:
            print(f"Error processing {args.csv_files[0]}: {e}", file=sys.stderr)
            sys.exit(1)
        return
    
    # Frame distribution mode
    summaries = []
    
    # Process all CSV files
    for csv_file in args.csv_files:
        try:
            csv_format = detect_csv_format(csv_file)
            if csv_format == '54':
                summary = process_csv_54(csv_file)
            else:
                summary = process_csv_52(csv_file)
            summaries.append(summary)
            print(f"Processed {csv_file}: {len(summary)} structure types, {summary['total_frames'].sum()} frames")
        except Exception as e:
            print(f"Error processing {csv_file}: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Create and show/save plot
    create_frame_plot(summaries, output_file=args.save)


if __name__ == '__main__':
    main()
