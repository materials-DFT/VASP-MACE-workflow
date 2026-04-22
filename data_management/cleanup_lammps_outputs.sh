#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

target_dir="${1%/}"

if [[ ! -d "$target_dir" ]]; then
  echo "Error: '$target_dir' is not a directory or does not exist."
  exit 1
fi

find_patterns=( -type f '(' \
  -name 'restart*' -o \
  -name 'job*' -o \
  -name 'log.lammps' -o \
  -name 'trajectory*' \
')' )

# Phase 1: only the specified directory (not its subdirectories)
mapfile -d '' -t files_to_delete < <(
  find "$target_dir" -maxdepth 1 "${find_patterns[@]}" -print0
)

if [[ ${#files_to_delete[@]} -gt 0 ]]; then
  rm -f -- "${files_to_delete[@]}"
  echo "Deleted ${#files_to_delete[@]} file(s) from '$target_dir' (top level only)."
  exit 0
fi

# Phase 2: nothing at top level — search subdirectories recursively
mapfile -d '' -t files_to_delete < <(
  find "$target_dir" -mindepth 2 "${find_patterns[@]}" -print0
)

if [[ ${#files_to_delete[@]} -eq 0 ]]; then
  echo "No matching files in '$target_dir' or its subdirectories."
  exit 0
fi

rm -f -- "${files_to_delete[@]}"
echo "Deleted ${#files_to_delete[@]} file(s) under subdirectories of '$target_dir' (recursive; top level had no matches)."
