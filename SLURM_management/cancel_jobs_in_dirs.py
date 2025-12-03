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

# ---------- slurm parsing ----------
STDOUT_RE = re.compile(r"\bStdOut=(\S+)")
WORKDIR_RE = re.compile(r"\bWorkDir=(\S+)")
JOBSTATE_RE = re.compile(r"\bJobState=(\S+)")
JOBNAME_RE = re.compile(r"\bJobName=(\S+)")
REASON_RE = re.compile(r"\bReason=(\S+)")

def get_user() -> str:
    import getpass
    return getpass.getuser()

def list_job_ids(user: str) -> List[str]:
    base = ["squeue", "-u", user, "-h", "-o", "%A"]
    code, out, err = run(base)
    if code != 0:
        print(f"[ERROR] squeue failed: {err}", file=sys.stderr)
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]

def scontrol_show_job(jobid: str) -> Optional[str]:
    code, out, err = run(["scontrol", "show", "job", jobid])
    if code != 0 or not out:
        return None
    return out

def parse_job_info(raw: str) -> Dict[str, Optional[str]]:
    info = {}
    for key, regex in [
        ("stdout", STDOUT_RE),
        ("workdir", WORKDIR_RE),
        ("state", JOBSTATE_RE),
        ("name", JOBNAME_RE),
        ("reason", REASON_RE),
    ]:
        m = regex.search(raw)
        info[key] = m.group(1) if m else None
    return info

def job_runtime(jobid: str) -> Optional[str]:
    code, out, _ = run(["squeue", "-j", jobid, "-h", "-o", "%M"])
    if code == 0 and out:
        return out.strip()
    raw = scontrol_show_job(jobid)
    if not raw:
        return None
    m = re.search(r"\bRunTime=(\S+)", raw) or re.search(r"\bElapsed=(\S+)", raw)
    return m.group(1) if m else None

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
        description="Cancel SLURM jobs whose job directories are under specified paths."
    )
    ap.add_argument("roots", nargs="+", help="Root directories to match (e.g., kmno2 /path/to/project)")
    args = ap.parse_args()

    # required tools
    for cmd in ("squeue", "scontrol", "scancel"):
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

    # Consider RUNNING, COMPLETING, and PENDING by default
    target_states = {"RUNNING", "COMPLETING", "PENDING"}

    to_cancel: List[Tuple[str, str, str, str]] = []  # (jid, state, name, dir)

    for jid in jobids:
        raw = scontrol_show_job(jid)
        if not raw:
            print(f"Job {jid}: unable to retrieve details via scontrol (skipping).")
            continue

        meta = parse_job_info(raw)
        state = (meta.get("state") or "").upper()
        name = meta.get("name") or "(no-name)"
        stdout = canonical(meta.get("stdout"))
        workdir = canonical(meta.get("workdir"))
        runtime = job_runtime(jid) or "N/A"
        reason = meta.get("reason")

        job_dir = stdout.parent if stdout else workdir
        job_dir_str = str(job_dir) if job_dir else "(unknown)"

        # Only look at RUNNING/COMPLETING/PENDING
        if state not in target_states:
            continue

        # Require resolvable dir to match roots
        if not job_dir:
            print(f"Job {jid} | {name} | State={state} | Runtime={runtime} | Dir=(unknown) -> skipping (no directory)")
            continue

        if under_any(job_dir, roots):
            suffix = ""
            if state == "PENDING" and reason:
                suffix = f" | Reason={reason}"
            print(f"Job {jid} | {name} | State={state} | Runtime={runtime} | Dir={job_dir_str}{suffix} -> MATCH")
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
        code, out, err = run(["scancel", jid])
        if code == 0:
            print(f"[OK] scancel {jid}")
        else:
            print(f"[FAIL] scancel {jid}: {err or out}")
            failed.append(jid)

    if failed:
        print(f"\nSome cancellations failed ({len(failed)}): {' '.join(failed)}")
    else:
        print("\nAll matching jobs were canceled successfully.")

if __name__ == "__main__":
    main()
