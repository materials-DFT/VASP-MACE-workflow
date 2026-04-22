#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 file_to_copy target_parent_directory"
  exit 1
fi

file_to_copy="$1"
parent_dir="$2"

if [ ! -f "$file_to_copy" ]; then
  echo "Error: File '$file_to_copy' does not exist."
  exit 2
fi

if [ ! -d "$parent_dir" ]; then
  echo "Error: Directory '$parent_dir' does not exist."
  exit 3
fi

# Find only the deepest-level subdirectories (leaf dirs)
leaf_dirs=$(find "$parent_dir" -type d -not -path "$parent_dir" | while read -r dir; do
  if [ -z "$(find "$dir" -mindepth 1 -type d 2>/dev/null)" ]; then
    echo "$dir"
  fi
done)

if [ -z "$leaf_dirs" ]; then
  echo "No leaf subdirectories found in '$parent_dir'."
  exit 0
fi

for dir in $leaf_dirs; do
  echo "Copying '$file_to_copy' to '$dir/'"
  cp "$file_to_copy" "$dir/"
done

echo "Done."
