#!/usr/bin/env python3
"""Upload frozen linear-adapter controls and submit the guarded PBS job."""

from __future__ import annotations

import importlib.util
import json
import re
import stat
from pathlib import Path
from pathlib import PurePosixPath


HERE = Path(__file__).resolve().parent
ACCESS = HERE.parents[3] / "nscc-access" / "aspire2a_shell.py"
REMOTE = "/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/linear-response-adapter"
PREVIOUS_RUN = "/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/terminal-response-loss/artifacts/25549480.pbs101"
FILES = (
    "FREEZE.json",
    "PROTOCOL.zh.md",
    "runner.py",
    "self_check.py",
    "run_linear_response_adapter.pbs",
)
MAX_CONTROL_FILE_BYTES = 200_000
LOCAL_JOB_ID = HERE / "results" / "submitted_job_id.txt"


def load_connector():
    spec = importlib.util.spec_from_file_location("aspire2a_shell", ACCESS)
    if spec is None or spec.loader is None:
        raise RuntimeError("verified ASPIRE2A connector is unavailable")
    connector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(connector)
    return connector


def validate_local_controls() -> dict[str, int]:
    sizes = {}
    for name in FILES:
        source = HERE / name
        if not source.is_file():
            raise FileNotFoundError(f"required control file is missing: {name}")
        size = source.stat().st_size
        if size > MAX_CONTROL_FILE_BYTES:
            raise RuntimeError(f"refusing oversized control file: {name} ({size} bytes)")
        sizes[name] = size

    freeze = json.loads((HERE / "FREEZE.json").read_text(encoding="utf-8"))
    source = freeze["source"]
    if source.get("previous_job_id") != "25549480.pbs101":
        raise RuntimeError("frozen previous job identity changed")
    if source.get("previous_dir") != PREVIOUS_RUN:
        raise RuntimeError("frozen previous result directory changed")
    previous_freeze = f"{PREVIOUS_RUN.rsplit('/artifacts/', 1)[0]}/FREEZE.json"
    required = [source["prior_summary"], source["prior_selection"], source["previous_freeze"], source["student_checkpoint"]]
    expected = {
        f"{PREVIOUS_RUN}/terminal_response_loss_summary.json",
        f"{PREVIOUS_RUN}/selection.json",
        f"{PREVIOUS_RUN}/terminal_response_step3000.pt",
        previous_freeze,
    }
    if len(required) != 4 or set(required) != expected or any(str(PurePosixPath(path).parent) != PREVIOUS_RUN for path in required if path != previous_freeze):
        raise RuntimeError("frozen summary, selection, or checkpoint paths changed")
    return sizes


def check_previous_inputs(sftp) -> None:
    for path in (
        f"{PREVIOUS_RUN.rsplit('/artifacts/', 1)[0]}/FREEZE.json",
        f"{PREVIOUS_RUN}/terminal_response_loss_summary.json",
        f"{PREVIOUS_RUN}/selection.json",
        f"{PREVIOUS_RUN}/terminal_response_step3000.pt",
    ):
        attrs = sftp.stat(path)
        if not stat.S_ISREG(attrs.st_mode) or attrs.st_size <= 0:
            raise RuntimeError(f"required previous result is missing or empty: {PurePosixPath(path).name}")
        print(f"source present ({attrs.st_size} bytes): {PurePosixPath(path).name}")


def main() -> None:
    sizes = validate_local_controls()
    connector = load_connector()
    jump, nscc = connector.connect()
    try:
        with nscc.open_sftp_client() as sftp:
            parent = REMOTE.rsplit("/", 1)[0]
            sftp.stat(parent)
            check_previous_inputs(sftp)
            try:
                sftp.stat(REMOTE)
            except OSError as exc:
                if getattr(exc, "errno", None) != 2:
                    raise
                sftp.mkdir(REMOTE)

            for name in FILES:
                source = HERE / name
                size = sizes[name]
                remote_file = f"{REMOTE}/{name}"
                sftp.put(str(source), remote_file)
                if sftp.stat(remote_file).st_size != size:
                    raise RuntimeError(f"uploaded size mismatch: {name}")
                print(f"uploaded {name} ({size} bytes)")

        channel = nscc.open_session(timeout=20)
        try:
            channel.exec_command(f"qsub {REMOTE}/run_linear_response_adapter.pbs")
            stdout = channel.makefile("rb").read().decode("utf-8", errors="replace").strip()
            stderr = channel.makefile_stderr("rb").read().decode("utf-8", errors="replace").strip()
            status = channel.recv_exit_status()
            if status != 0:
                raise RuntimeError(f"qsub failed (exit {status}): {stderr or stdout}")
            match = re.search(r"(?m)^(\d+(?:\.[A-Za-z0-9._-]+)?)\s*$", stdout)
            if match is None:
                raise RuntimeError(f"qsub succeeded but returned an unrecognized job id: {stdout}")
            job_id = match.group(1)
            LOCAL_JOB_ID.parent.mkdir(parents=True, exist_ok=True)
            LOCAL_JOB_ID.write_text(job_id + "\n", encoding="utf-8")
            print(f"submitted {job_id}")
        finally:
            channel.close()
    finally:
        nscc.close()
        jump.close()


if __name__ == "__main__":
    main()
