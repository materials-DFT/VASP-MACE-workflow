#!/bin/bash

# Check if a directory is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <target_directory>"
    exit 1
fi

# Navigate to the target directory
TARGET_DIR="$1"
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory $TARGET_DIR does not exist."
    exit 1
fi

cd "$TARGET_DIR" || exit

# Recursively find and submit all submit.vasp6.sh scripts
find . -type f -name submit.vasp6.sh | while read -r script; do
    echo "Submitting $script"
    script_dir=$(dirname "$script")
    (cd "$script_dir" && sbatch submit.vasp6.sh)
done
