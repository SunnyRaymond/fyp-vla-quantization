#!/usr/bin/env python3
"""Read-only status and small result fetch for the frozen PBS experiment."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ACCESS = HERE.parents[3] / "nscc-access" / "aspire2a_shell.py"
REMOTE = "/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/terminal-response-loss"
JOB_ID = "25549480.pbs101"
REMOTE_OUT = f"{REMOTE}/artifacts/{JOB_ID}"
LOCAL_OUT = HERE / "results" / JOB_ID


def main() -> None:
    spec = importlib.util.spec_from_file_location("aspire2a_shell", ACCESS)
    if spec is None or spec.loader is None:
        raise RuntimeError("ASPIRE2A connector unavailable")
    connector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(connector)
    jump, nscc = connector.connect()
    try:
        channel = nscc.open_session(timeout=20)
        try:
            channel.exec_command(f"qstat -fx {JOB_ID}")
            raw = channel.makefile("rb").read().decode("utf-8", errors="replace")
            err = channel.makefile_stderr("rb").read().decode("utf-8", errors="replace")
            status = channel.recv_exit_status()
            if status != 0:
                print(f"qstat status={status}: {err.strip()}")
            elif "job_state = F" in raw and len(raw) < 100_000:
                LOCAL_OUT.mkdir(parents=True, exist_ok=True)
                (LOCAL_OUT / "qstat_final.txt").write_text(raw, encoding="utf-8")
            for line in raw.splitlines():
                if any(key in line for key in ("job_state =", "Exit_status =", "comment =", "queue =", "exec_host =", "resources_used.walltime =")):
                    print(line.strip())
        finally:
            channel.close()

        with nscc.open_sftp_client() as sftp:
            for name in ("job_status", "selection.json", "terminal_response_loss_summary.json", "job.log"):
                path = f"{REMOTE_OUT}/{name}"
                try:
                    size = sftp.stat(path).st_size
                except OSError:
                    continue
                print(f"{name}: {size} bytes")
                if name in ("job_status", "selection.json", "terminal_response_loss_summary.json"):
                    if size > 2_000_000:
                        raise RuntimeError(f"refusing oversized small result: {name}")
                    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
                    sftp.get(path, str(LOCAL_OUT / name))
                elif name == "job.log":
                    if size <= 20_000:
                        LOCAL_OUT.mkdir(parents=True, exist_ok=True)
                        sftp.get(path, str(LOCAL_OUT / name))
                    with sftp.open(path, "rb") as handle:
                        handle.seek(max(0, size - 3000))
                        print(handle.read().decode("utf-8", errors="replace"))
    finally:
        nscc.close()
        jump.close()


if __name__ == "__main__":
    main()
