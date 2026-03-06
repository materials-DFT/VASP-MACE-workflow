#!/usr/bin/env python3
import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------- shell helpers ----------
def run(cmd: List[str]) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()

def which(cmd: str) -> bool:
    return subprocess.run(["bash", "-lc", f"type -P {shlex.quote(cmd)} >/dev/null 2>&1"]).returncode == 0

# ---------- PBS parsing ----------
WORKDIR_RE = re.compile(r"\s*PBS_O_WORKDIR\s*=\s*(.+)", re.IGNORECASE)
JOBSTATE_RE = re.compile(r"\s*job_state\s*=\s*(\S+)", re.IGNORECASE)
JOBNAME_RE = re.compile(r"\s*Job_Name\s*=\s*(.+)", re.IGNORECASE)
OUTPUT_PATH_RE = re.compile(r"\s*Output_Path\s*=\s*(.+)", re.IGNORECASE)
ERROR_PATH_RE = re.compile(r"\s*Error_Path\s*=\s*(.+)", re.IGNORECASE)
RUNTIME_RE = re.compile(r"\s*resources_used\.walltime\s*=\s*(\S+)", re.IGNORECASE)

def get_user() -> str:
    import getpass
    return getpass.getuser()

def list_job_ids(user: str) -> List[str]:
    # qstat -u user returns jobs in format: jobid.hostname username jobname ...
    # We need to extract just the job IDs
    base = ["qstat", "-u", user]
    code, out, err = run(base)
    if code != 0:
        print(f"[ERROR] qstat failed: {err}", file=sys.stderr)
        return []
    
    jobids = []
    lines = out.splitlines()
    # Skip header lines (usually first 2 lines)
    for line in lines[2:]:
        if line.strip():
            # Job ID is typically the first field
            parts = line.split()
            if parts:
                # PBS job IDs are usually in format: number.hostname
                jobid = parts[0].split('.')[0]  # Take just the numeric part
                if jobid.isdigit():
                    jobids.append(jobid)
    
    return jobids

def qstat_show_job(jobid: str) -> Optional[str]:
    code, out, err = run(["qstat", "-f", jobid])
    if code != 0 or not out:
        return None
    return out

def parse_job_info(raw: str) -> Dict[str, Optional[str]]:
    info = {}
    for key, regex in [
        ("workdir", WORKDIR_RE),
        ("state", JOBSTATE_RE),
        ("name", JOBNAME_RE),
        ("output_path", OUTPUT_PATH_RE),
        ("error_path", ERROR_PATH_RE),
        ("runtime", RUNTIME_RE),
    ]:
        m = regex.search(raw)
        info[key] = m.group(1).strip() if m else None
    return info

def job_runtime(jobid: str) -> Optional[str]:
    raw = qstat_show_job(jobid)
    if not raw:
        return None
    info = parse_job_info(raw)
    return info.get("runtime") or "N/A"

# ---------- path logic ----------
def canonical(p: Optional[str]) -> Optional[Path]:
    if not p:
        return None
    try:
        return Path(p).expanduser().resolve()
    except Exception:
        return Path(p).expanduser()

def under_any(path: Path, roots: List[Path]) -> bool:
    try:
        for r in roots:
            common = os.path.commonpath([str(path), str(r)])
            if common == str(r):
                return True
    except Exception:
        pass
    return False

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Cancel PBS jobs whose job directories are under specified paths."
    )
    ap.add_argument("roots", nargs="+", help="Root directories to match (e.g., /path/to/project)")
    args = ap.parse_args()

    # required tools
    for cmd in ("qstat", "qdel"):
        if not which(cmd):
            print(f"[ERROR] `{cmd}` not found in PATH.", file=sys.stderr)
            sys.exit(2)

    roots = [canonical(r) for r in args.roots]
    bad = [r for r in roots if r is None or not r.exists()]
    if bad:
        for b in bad:
            print(f"[WARN] Root does not exist or cannot be resolved: {b}", file=sys.stderr)
    roots = [r for r in roots if r and r.exists()]
    if not roots:
        print("[ERROR] No valid roots to match against. Exiting.", file=sys.stderr)
        sys.exit(1)

    user = get_user()
    print("Roots:")
    for r in roots:
        print(f"  - {r}")

    jobids = list_job_ids(user)
    if not jobids:
        print("No jobs found for user.")
        return

    print(f"\nFound {len(jobids)} job(s) for user `{user}`.\n")

    # PBS states: Q (queued), R (running), E (exiting), H (held), C (completed)
    # We want to cancel Q, R, E, H (not C)
    target_states = {"Q", "R", "E", "H"}

    to_cancel: List[Tuple[str, str, str, str]] = []  # (jid, state, name, dir)

    for jid in jobids:
        raw = qstat_show_job(jid)
        if not raw:
            print(f"Job {jid}: unable to retrieve details via qstat -f (skipping).")
            continue

        meta = parse_job_info(raw)
        state = (meta.get("state") or "").upper()
        name = meta.get("name") or "(no-name)"
        workdir = canonical(meta.get("workdir"))
        output_path = canonical(meta.get("output_path"))
        runtime = job_runtime(jid) or "N/A"

        # Use workdir if available, otherwise try to infer from output_path
        job_dir = workdir
        if not job_dir and output_path:
            # Output path might be like /path/to/file.o12345, so get parent
            job_dir = output_path.parent if output_path else None
        
        job_dir_str = str(job_dir) if job_dir else "(unknown)"

        # Only look at active states (Q, R, E, H)
        if state not in target_states:
            continue

        # Require resolvable dir to match roots
        if not job_dir:
            print(f"Job {jid} | {name} | State={state} | Runtime={runtime} | Dir=(unknown) -> skipping (no directory)")
            continue

        if under_any(job_dir, roots):
            state_name = {"Q": "QUEUED", "R": "RUNNING", "E": "EXITING", "H": "HELD"}.get(state, state)
            print(f"Job {jid} | {name} | State={state_name} | Runtime={runtime} | Dir={job_dir_str} -> MATCH")
            to_cancel.append((jid, state, name, job_dir_str))

    if not to_cancel:
        print("\nNo matching jobs to cancel.")
        return

    print(f"\nMatched {len(to_cancel)} job(s) under the specified roots:")
    print("  " + " ".join(j for j, _, _, _ in to_cancel))

    # Single interactive confirmation (no flags)
    try:
        resp = input("\nCancel ALL matched jobs? [y/N]: ").strip().lower()
    except EOFError:
        resp = ""
    if resp not in ("y", "yes"):
        print("Aborted. No jobs were canceled.")
        return

    failed = []
    for jid, _, _, _ in to_cancel:
        code, out, err = run(["qdel", jid])
        if code == 0:
            print(f"[OK] qdel {jid}")
        else:
            print(f"[FAIL] qdel {jid}: {err or out}")
            failed.append(jid)

    if failed:
        print(f"\nSome cancellations failed ({len(failed)}): {' '.join(failed)}")
    else:
        print("\nAll matching jobs were canceled successfully.")

if __name__ == "__main__":
    main()

