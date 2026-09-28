#!/usr/bin/env python3
"""Upload three small control files and submit the guarded PBS diagnostic."""

from __future__ import annotations

import argparse
import importlib.util
import socket
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACCESS = HERE.parents[4] / "nscc-access" / "aspire2a_shell.py"
REMOTE = "/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/student-induced-shadow-pilot/action-response-decomposition-posthoc"
FILES = ("FREEZE.json", "runner.py", "run_action_response_decomposition.pbs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct", action="store_true", help="Use verified direct ASPIRE2A SSH")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("aspire2a_shell", ACCESS)
    if spec is None or spec.loader is None:
        raise RuntimeError("verified connector is unavailable")
    connector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(connector)
    if args.direct:
        credentials = connector.read_credentials()
        sock = socket.create_connection((connector.ASPIRE2A_HOST, 22), timeout=20)
        nscc = connector.paramiko.Transport(sock)
        try:
            nscc.start_client(timeout=20)
            connector.verify_host_key(nscc, connector.ASPIRE2A_HOST, connector.NSCC_KNOWN_HOSTS)
            nscc.auth_password(credentials["NSCC_USERNAME"], credentials["NSCC_PASSWORD"])
            if not nscc.is_authenticated():
                raise RuntimeError("verified direct ASPIRE2A authentication failed")
        except BaseException:
            nscc.close()
            raise
        jump = None
    else:
        jump, nscc = connector.connect()
    try:
        with nscc.open_sftp_client() as sftp:
            parent = REMOTE.rsplit("/", 1)[0]
            sftp.stat(parent)
            try:
                sftp.stat(REMOTE)
            except OSError:
                sftp.mkdir(REMOTE)
            for name in FILES:
                source = HERE / name
                if not source.is_file() or source.stat().st_size > 200_000:
                    raise RuntimeError(f"missing or oversized control file: {name}")
                sftp.put(str(source), f"{REMOTE}/{name}")
                if sftp.stat(f"{REMOTE}/{name}").st_size != source.stat().st_size:
                    raise RuntimeError(f"control-file size mismatch: {name}")
                print(f"uploaded {name}")
            for path in (
                f"{parent}/latent-score-sensitivity-posthoc/runner.py",
                f"{parent}/onpolicy-fullbank-ranker/partial-horizon-teacher-hybrid/runner.py",
                f"{parent}/onpolicy-fullbank-ranker/artifacts/25538135.pbs101/train_banks.pt",
            ):
                sftp.stat(path)
        channel = nscc.open_session(timeout=20)
        try:
            channel.exec_command(f"qsub {REMOTE}/run_action_response_decomposition.pbs")
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
        if jump is not None:
            jump.close()


if __name__ == "__main__":
    main()
