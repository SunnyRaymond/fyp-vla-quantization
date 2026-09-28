#!/usr/bin/env python3
"""Upload the frozen terminal-response controls and submit their guarded PBS job."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
ACCESS = HERE.parents[3] / "nscc-access" / "aspire2a_shell.py"
REMOTE = "/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/terminal-response-loss"
FILES = ("FREEZE.json", "PROTOCOL.zh.md", "runner.py", "self_check.py", "run_terminal_response_loss.pbs")
MAX_CONTROL_FILE_BYTES = 200_000
TRANSFER = "/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer"
REMOTE_PREREQUISITES = (
    f"{TRANSFER}/state-coverage/artifacts/24926383.pbs101/context_manifest_512.json",
    f"{TRANSFER}/teacher-screening/artifacts/25213164.pbs101/prepared_balanced_rows_reconstructed.pt",
)


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
    return sizes


def main() -> None:
    sizes = validate_local_controls()
    connector = load_connector()
    jump, nscc = connector.connect()
    try:
        with nscc.open_sftp_client() as sftp:
            parent = REMOTE.rsplit("/", 1)[0]
            sftp.stat(parent)
            for path in REMOTE_PREREQUISITES:
                size = sftp.stat(path).st_size
                if size <= 0:
                    raise RuntimeError(f"required source file is empty: {path}")
                print(f"source present ({size} bytes): {Path(path).name}")
            try:
                sftp.stat(REMOTE)
            except OSError as exc:
                if exc.errno != 2:
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
            channel.exec_command(f"qsub {REMOTE}/run_terminal_response_loss.pbs")
            stdout = channel.makefile("rb").read().decode("utf-8", errors="replace").strip()
            stderr = channel.makefile_stderr("rb").read().decode("utf-8", errors="replace").strip()
            status = channel.recv_exit_status()
            if status != 0:
                raise RuntimeError(f"qsub failed (exit {status}): {stderr or stdout}")
            print(f"submitted {stdout}")
        finally:
            channel.close()
    finally:
        nscc.close()
        jump.close()


if __name__ == "__main__":
    main()
