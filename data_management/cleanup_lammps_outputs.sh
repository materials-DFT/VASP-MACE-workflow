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

# Files to retain when cleaning a VASP directory (matched by basename anywhere under target_dir).
VASP_KEEP_NAMES=(INCAR POSCAR POTCAR KPOINTS submit.vasp6.sh)

is_vasp_directory() {
  [[ -f "$target_dir/INCAR" ]] || [[ -f "$target_dir/POSCAR" ]]
}

collect_vasp_files_to_delete() {
  local scan_root="$1"
  local f base k keep
  while IFS= read -r -d '' f; do
    base=$(basename "$f")
    keep=false
    for k in "${VASP_KEEP_NAMES[@]}"; do
      if [[ "$base" == "$k" ]]; then
        keep=true
        break
      fi
    done
    if [[ "$keep" == false ]]; then
      files_to_delete+=("$f")
    fi
  done < <(find "$scan_root" -type f -print0)
}

collect_lammps_files_to_delete() {
  local find_patterns=(
    -type f '('
    -name 'restart*' -o
    -name 'job*' -o
    -name 'log.lammps' -o
    -name 'trajectory*'
    ')'
  )

  mapfile -d '' -t files_to_delete < <(
    find "$target_dir" -maxdepth 1 "${find_patterns[@]}" -print0
  )

  if [[ ${#files_to_delete[@]} -gt 0 ]]; then
    lammps_scope="top level of '$target_dir' only"
    return 0
  fi

  mapfile -d '' -t files_to_delete < <(
    find "$target_dir" -mindepth 2 "${find_patterns[@]}" -print0
  )
  lammps_scope="subdirectories of '$target_dir' (recursive; top level had no matches)"
}

files_to_delete=()
lammps_scope=""
vasp_multi_scope=""

if is_vasp_directory; then
  mode="VASP"
  collect_vasp_files_to_delete "$target_dir"
else
  vasp_roots=()
  while IFS= read -r -d '' d; do
    [[ -f "$d/INCAR" ]] || [[ -f "$d/POSCAR" ]] || continue
    vasp_roots+=("$d")
  done < <(find "$target_dir" -mindepth 1 -maxdepth 1 -type d -print0)

  if [[ ${#vasp_roots[@]} -gt 0 ]]; then
    mode="VASP"
    labels=()
    for d in "${vasp_roots[@]}"; do
      labels+=("$(basename "$d")")
      collect_vasp_files_to_delete "$d"
    done
    saved_IFS="$IFS"
    IFS=','
    vasp_multi_scope="VASP run folder(s): ${labels[*]}"
    IFS="$saved_IFS"
  else
    mode="LAMMPS"
    collect_lammps_files_to_delete
  fi
fi

if [[ ${#files_to_delete[@]} -eq 0 ]]; then
  if [[ "$mode" == "VASP" ]]; then
    echo "No files to remove under '$target_dir' (only kept inputs may be present, or directory is empty)."
  else
    echo "No matching files in '$target_dir' or its subdirectories."
  fi
  exit 0
fi

echo "Mode: $mode"
echo "Directory: $target_dir"
if [[ "$mode" == "VASP" && -n "${vasp_multi_scope:-}" ]]; then
  echo "Scope: $vasp_multi_scope"
fi
if [[ "$mode" == "LAMMPS" ]]; then
  echo "Scope: $lammps_scope"
fi
echo "Files to delete (${#files_to_delete[@]}):"
printf '  %s\n' "${files_to_delete[@]}"
echo
read -r -p "Proceed with deletion? [y/N] " reply
if [[ ! "${reply:-}" =~ ^[Yy]([Ee][Ss])?$ ]]; then
  echo "Aborted."
  exit 1
fi

rm -f -- "${files_to_delete[@]}"
echo "Deleted ${#files_to_delete[@]} file(s)."
