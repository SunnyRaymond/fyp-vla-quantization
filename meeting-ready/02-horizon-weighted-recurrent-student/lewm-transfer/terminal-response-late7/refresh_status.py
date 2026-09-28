#!/usr/bin/env python3
"""Read-only PBS status refresh and bounded fetch of small result files."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ACCESS = HERE.parents[3] / "nscc-access" / "aspire2a_shell.py"
REMOTE = "/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/terminal-response-late7"
JOB_ID_FILE = HERE / "results" / "submitted_job_id.txt"
SMALL_RESULT_LIMIT = 2_000_000
SMALL_LOG_LIMIT = 2_000_000
LOG_TAIL_BYTES = 4_000
RESULT_FILES = (
    "job_status",
    "final_exit_status.txt",
    "terminal_response_late7_summary.json",
    "terminal_response_late7_episodes.jsonl",
    "terminal_response_late7_solve_records.jsonl",
    "rng_pairing.json",
    "selected_tasks.json",
    "run_order.json",
    "gpu_info.csv",
)


def resolve_job_id() -> str:
    value = sys.argv[1].strip() if len(sys.argv) > 1 else JOB_ID_FILE.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"\d+(?:\.[A-Za-z0-9._-]+)?", value):
        raise ValueError("provide a valid PBS job id, or run submit.py first")
    return value


def load_connector():
    spec = importlib.util.spec_from_file_location("aspire2a_shell", ACCESS)
    if spec is None or spec.loader is None:
        raise RuntimeError("verified ASPIRE2A connector is unavailable")
    connector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(connector)
    return connector


def main() -> None:
    job_id = resolve_job_id()
    remote_out = f"{REMOTE}/artifacts/{job_id}"
    local_out = HERE / "results" / job_id
    connector = load_connector()
    jump, nscc = connector.connect()
    try:
        channel = nscc.open_session(timeout=20)
        try:
            channel.exec_command(f"qstat -fx {job_id}")
            raw = channel.makefile("rb").read().decode("utf-8", errors="replace")
            err = channel.makefile_stderr("rb").read().decode("utf-8", errors="replace")
            status = channel.recv_exit_status()
            if status != 0:
                print(f"qstat status={status}: {err.strip() or raw.strip()}")
            else:
                for line in raw.splitlines():
                    if any(key in line for key in (
                        "job_state =", "Exit_status =", "comment =", "queue =",
                        "exec_host =", "resources_used.walltime =",
                    )):
                        print(line.strip())
                if len(raw) <= 100_000:
                    local_out.mkdir(parents=True, exist_ok=True)
                    name = "qstat_final.txt" if "job_state = F" in raw else "qstat_latest.txt"
                    (local_out / name).write_text(raw, encoding="utf-8")
        finally:
            channel.close()

        with nscc.open_sftp_client() as sftp:
            for name in RESULT_FILES:
                path = f"{remote_out}/{name}"
                try:
                    size = sftp.stat(path).st_size
                except OSError:
                    print(f"{name}: not available yet")
                    continue
                if size > SMALL_RESULT_LIMIT:
                    if name == "terminal_response_late7_solve_records.jsonl":
                        print(f"{name}: {size} bytes; left remote because it exceeds the small-fetch limit")
                        continue
                    raise RuntimeError(f"refusing oversized result fetch: {name} ({size} bytes)")
                print(f"{name}: {size} bytes")
                local_out.mkdir(parents=True, exist_ok=True)
                sftp.get(path, str(local_out / name))

            log_path = f"{remote_out}/job.log"
            try:
                log_size = sftp.stat(log_path).st_size
            except OSError:
                print("job.log: not available yet")
            else:
                print(f"job.log: {log_size} bytes")
                with sftp.open(log_path, "rb") as handle:
                    handle.seek(max(0, log_size - LOG_TAIL_BYTES))
                    tail = handle.read(LOG_TAIL_BYTES).decode("utf-8", errors="replace")
                print(tail)
                if log_size <= SMALL_LOG_LIMIT:
                    local_out.mkdir(parents=True, exist_ok=True)
                    sftp.get(log_path, str(local_out / "job.log"))
    finally:
        nscc.close()
        jump.close()


if __name__ == "__main__":
    main()
