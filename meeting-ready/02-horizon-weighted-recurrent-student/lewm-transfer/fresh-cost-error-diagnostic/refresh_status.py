#!/usr/bin/env python3
"""Read-only PBS status and small result fetch for the diagnostic job."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ACCESS = HERE.parents[3] / "nscc-access" / "aspire2a_shell.py"
REMOTE = "/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/fresh-cost-error-diagnostic"
JOB_ID_FILE = HERE / "results" / "submitted_job_id.txt"
SMALL_RESULT_LIMIT = 2_000_000
STATUS_LIMIT = 64_000
LOG_TAIL_BYTES = 3_000


def resolve_job_id() -> str:
    value = sys.argv[1].strip() if len(sys.argv) > 1 else JOB_ID_FILE.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"\d+(?:\.[A-Za-z0-9._-]+)?", value):
        raise ValueError("provide a valid PBS job id, or run submit.py first")
    return value


def main() -> None:
    job_id = resolve_job_id()
    remote_out = f"{REMOTE}/artifacts/{job_id}"
    local_out = HERE / "results" / job_id
    spec = importlib.util.spec_from_file_location("aspire2a_shell", ACCESS)
    if spec is None or spec.loader is None:
        raise RuntimeError("verified ASPIRE2A connector is unavailable")
    connector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(connector)
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
            elif "job_state = F" in raw and len(raw) < 100_000:
                local_out.mkdir(parents=True, exist_ok=True)
                (local_out / "qstat_final.txt").write_text(raw, encoding="utf-8")
            for line in raw.splitlines():
                if any(key in line for key in ("job_state =", "Exit_status =", "comment =", "queue =", "exec_host =", "resources_used.walltime =")):
                    print(line.strip())
        finally:
            channel.close()

        with nscc.open_sftp_client() as sftp:
            for name in ("job_status", "fresh_cost_error_diagnostic_summary.json", "job.log"):
                path = f"{remote_out}/{name}"
                try:
                    size = sftp.stat(path).st_size
                except OSError:
                    continue
                print(f"{name}: {size} bytes")
                if name == "job.log":
                    if size <= 20_000:
                        local_out.mkdir(parents=True, exist_ok=True)
                        sftp.get(path, str(local_out / name))
                    with sftp.open(path, "rb") as handle:
                        handle.seek(max(0, size - LOG_TAIL_BYTES))
                        tail = handle.read(LOG_TAIL_BYTES).decode("utf-8", errors="replace")
                    print(tail)
                    continue
                limit = STATUS_LIMIT if name == "job_status" else SMALL_RESULT_LIMIT
                if size > limit:
                    raise RuntimeError(f"refusing oversized status result: {name}")
                local_out.mkdir(parents=True, exist_ok=True)
                sftp.get(path, str(local_out / name))
    finally:
        nscc.close()
        jump.close()


if __name__ == "__main__":
    main()
