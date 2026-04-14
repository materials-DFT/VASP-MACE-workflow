#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from ase.build.tools import sort
from ase.io import read, write

def main():
    """
    Main function to process POSCAR files from command-line input.
    """
    if len(sys.argv) < 2:
        print("Usage: python reorder_poscar.py <file_or_directory_path>")
        sys.exit(1)

    path = sys.argv[1]
    # Note: new_order is not used anymore since ASE's sort automatically sorts by element
    # But keeping it for potential future use or compatibility
    new_order = ['K', 'Mn', 'O']

    if os.path.isfile(path):
        process_file(path, new_order)
    elif os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            if 'POSCAR' in filenames:
                file_path = os.path.join(dirpath, 'POSCAR')
                process_file(file_path, new_order)
    else:
        print(f"Error: The path '{path}' does not exist or is not a valid file/directory.", file=sys.stderr)
        sys.exit(1)

def process_file(file_path, new_order):
    """
    Reads, sorts atoms by element type using ASE, and overwrites the POSCAR file.
    """
    print(f"Processing {file_path}...")
    try:
        # Read the POSCAR file using ASE
        atoms = read(file_path)
        
        # Sort atoms by element type (ASE's sort function automatically sorts by element)
        atoms_sorted = sort(atoms)
        
        # Write the sorted structure back to the same file
        write(file_path, atoms_sorted, format='vasp')
        
        print(f"Successfully reordered and overwrote {file_path}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

