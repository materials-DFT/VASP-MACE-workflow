#!/bin/bash
#
# Remove psm_shm.* and shm-col-space* from /dev/shm on themis compute nodes.
# Run this script ON THEMIS (head node).
#
# By default does NOT use sudo on compute nodes (you may have sudo on the head
# node but not on child nodes). Removes only files you own. For full cleanup on
# nodes, an admin must add your user to sudoers on the COMPUTE nodes, then use
# CLEAN_SHM_USE_SUDO=1.
#
# Usage:
#   ./clean_dev_shm_themis.sh                     # default nodes, no sudo (removes only your files)
#   ./clean_dev_shm_themis.sh n011 n012           # only listed nodes
#   ./clean_dev_shm_themis.sh --dry-run           # show what would be done
#   CLEAN_SHM_USE_SUDO=1 ./clean_dev_shm_themis.sh   # use sudo on nodes (requires sudo on compute nodes)
#

set -e

DRY_RUN=false
NODES=()

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    *)         NODES+=("$arg") ;;
  esac
done

# Default node list if none given (n001 through n012)
if [[ ${#NODES[@]} -eq 0 ]]; then
  for i in $(seq 1 12); do
    NODES+=("n$(printf '%03d' $i)")
  done
fi

# Default: no sudo on compute nodes (works when you have sudo on head but not on child nodes).
# Set CLEAN_SHM_USE_SUDO=1 only if your user is in sudoers on the COMPUTE nodes (not just the head node).
USE_SUDO=false
[[ -n "$CLEAN_SHM_USE_SUDO" ]] && USE_SUDO=true

if [[ "$USE_SUDO" == true && "$DRY_RUN" != true ]]; then
  REMOTE_CMD='sudo rm -f /dev/shm/psm_shm.* /dev/shm/shm-col-space*'
elif [[ "$USE_SUDO" == true && "$DRY_RUN" == true ]]; then
  REMOTE_CMD='echo "Would run: sudo rm -f /dev/shm/psm_shm.* /dev/shm/shm-col-space*"; echo -n "psm_shm: "; ls /dev/shm/psm_shm.* 2>/dev/null | wc -l; echo -n "shm-col-space: "; ls /dev/shm/shm-col-space* 2>/dev/null | wc -l'
else
  REMOTE_CMD='rm -f /dev/shm/psm_shm.* /dev/shm/shm-col-space* 2>/dev/null; echo "Removed (own files only, no sudo)"'
  [[ "$DRY_RUN" == true ]] && REMOTE_CMD='echo "Would run: rm -f /dev/shm/psm_shm.* /dev/shm/shm-col-space* (no sudo)"; ls /dev/shm/psm_shm.* /dev/shm/shm-col-space* 2>/dev/null | wc -l'
fi

# Use -t when sudo is used so sudo on the node gets a TTY (avoids "no tty present and no askpass program specified").
# Run from an interactive shell on themis, or set CLEAN_SHM_NO_SUDO=1 for non-interactive (removes only your files).
SSH_OPTS="-o ConnectTimeout=5 -o StrictHostKeyChecking=no"
[[ "$USE_SUDO" == true ]] && SSH_OPTS="-t $SSH_OPTS"

for node in "${NODES[@]}"; do
  echo "=== $node ==="
  if ssh $SSH_OPTS "$node" "$REMOTE_CMD" 2>&1; then
    if [[ "$DRY_RUN" != true ]]; then
      pcount=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$node" 'ls /dev/shm/psm_shm.* 2>/dev/null | wc -l')
      scount=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$node" 'ls /dev/shm/shm-col-space* 2>/dev/null | wc -l')
      echo "  Remaining psm_shm: $pcount, shm-col-space: $scount"
    fi
  else
    echo "  Failed or skipped."
  fi
  echo
done

echo "Done."
