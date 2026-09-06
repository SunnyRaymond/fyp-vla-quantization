from __future__ import annotations

import re
import sys
import time
from datetime import datetime
from pathlib import Path

from nscc_compute_probe import connect, run


ACCESS_DIR = Path(__file__).resolve().parent
PROJECT_ID = "personal-yguo017"


def main() -> int:
    jump, nscc = connect()
    try:
        status, home, error = run(nscc, 'printf %s "$HOME"')
        if status or not home.startswith("/"):
            raise RuntimeError(error or "Could not determine the remote home directory.")

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        remote_output = f"{home}/codex-4xa100-{stamp}.out"
        local_output = ACCESS_DIR / f"aspire2a-4xa100-{stamp}.txt"
        job_script = r"""#!/bin/bash
# This batch job deliberately exits immediately after collecting evidence.
printf 'NSCC ASPIRE2A - 4 x A100 allocation snapshot\n'
printf 'Job ID: %s\n' "$PBS_JOBID"
printf 'Compute node: %s\n' "$(hostname)"
printf 'Captured: '
date --iso-8601=seconds
printf 'CUDA_VISIBLE_DEVICES=%s\n\n' "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
printf '\n'
nvidia-smi
"""

        channel = nscc.open_session(timeout=20)
        channel.settimeout(60)
        channel.exec_command(
            f"qsub -o {remote_output} -N codex_4xa100_snapshot "
            f"-q normal -P {PROJECT_ID} -l select=1:ngpus=4 "
            "-l walltime=00:02:00 -j oe"
        )
        channel.sendall(job_script.encode("utf-8"))
        channel.shutdown_write()
        job_id = channel.makefile("rb").read().decode("utf-8", errors="replace").strip()
        submit_error = channel.makefile_stderr("rb").read().decode("utf-8", errors="replace").strip()
        submit_status = channel.recv_exit_status()
        if submit_status or not re.fullmatch(r"\d+(?:\.[A-Za-z0-9._-]+)?", job_id):
            raise RuntimeError(f"PBS submission failed: {submit_error or job_id}")

        print(f"Submitted minimal 4-GPU snapshot job: {job_id}", flush=True)
        print("The job will exit automatically immediately after nvidia-smi.", flush=True)

        last_state = ""
        while True:
            _status, detail, _error = run(
                nscc,
                f"qstat -f {job_id} 2>/dev/null | "
                "awk -F ' = ' '/job_state =|queue =|comment =|resources_used.walltime =|Exit_status =/ "
                "{print $1 \"=\" $2}'",
            )
            state_match = re.search(r"job_state\s*=\s*([A-Z])", detail)
            state = state_match.group(1) if state_match else ""
            if state and state != last_state:
                print(f"PBS state: {state}", flush=True)
                if "comment" in detail:
                    print(detail, flush=True)
                last_state = state
            if state == "F" or not detail:
                break
            time.sleep(5)

        for _ in range(12):
            output_status, output, output_error = run(
                nscc, f"test -s {remote_output} && cat {remote_output}", timeout=60
            )
            if output_status == 0 and output:
                local_output.write_text(output + "\n", encoding="utf-8")
                print(f"LOCAL_OUTPUT={local_output}", flush=True)
                print(f"REMOTE_OUTPUT={remote_output}", flush=True)
                print("--- SNAPSHOT ---", flush=True)
                print(output, flush=True)
                return 0
            time.sleep(2)
        raise RuntimeError(output_error or "The completed job output was not available.")
    finally:
        nscc.close()
        jump.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"4-GPU snapshot failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
