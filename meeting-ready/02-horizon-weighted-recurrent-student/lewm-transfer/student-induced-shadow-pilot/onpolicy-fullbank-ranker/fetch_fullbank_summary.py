"""Fetch only the small result records from the frozen full-bank PBS job."""

from __future__ import annotations

import argparse
import importlib.util
import socket
from pathlib import Path


HERE = Path(__file__).resolve().parent
ACCESS = HERE.parents[4] / "nscc-access" / "aspire2a_shell.py"
JOB_ID = "25538135.pbs101"
REMOTE = (
    "/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/"
    "02-horizon-weighted-recurrent-student/lewm-transfer/"
    f"student-induced-shadow-pilot/onpolicy-fullbank-ranker/artifacts/{JOB_ID}"
)
FILES = (
    "collection_gate.json",
    "collection_manifest.json",
    "training/gate.json",
    "training/ranker_summary.json",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--oracle-shortlist", action="store_true")
    mode.add_argument("--real-observation", action="store_true")
    mode.add_argument("--real-observation-14", action="store_true")
    args = parser.parse_args()
    if args.real_observation_14:
        job_id = "25542467.pbs101"
        remote_root = (
            "/scratch/users/ntu/yguo017/lewm-pusht-iteration/artifacts/"
            f"student-induced-shadow-pilot/real-observation-rollout-probe-14-task/{job_id}"
        )
        files = ("real_observation_latent_alignment_14_task.json",)
        output = HERE.parent / "real-observation-rollout-probe-14-task" / "results" / job_id
    elif args.real_observation:
        job_id = "25542215.pbs101"
        remote_root = (
            "/scratch/users/ntu/yguo017/lewm-pusht-iteration/artifacts/"
            f"student-induced-shadow-pilot/real-observation-rollout-probe/{job_id}"
        )
        files = ("real_observation_latent_alignment.json",)
        output = HERE.parent / "real-observation-rollout-probe" / "results" / job_id
    elif args.oracle_shortlist:
        job_id = "25538259.pbs101"
        remote_root = (
            "/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/"
            "02-horizon-weighted-recurrent-student/lewm-transfer/"
            "student-induced-shadow-pilot/onpolicy-fullbank-ranker/"
            f"artifacts/oracle-shortlist-posthoc/{job_id}"
        )
        files = ("oracle_shortlist_summary.json",)
        output = HERE / "results" / job_id
    else:
        job_id = JOB_ID
        remote_root = REMOTE
        files = FILES
        output = HERE / "results" / job_id
    spec = importlib.util.spec_from_file_location("aspire2a_shell", ACCESS)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load verified ASPIRE2A connector")
    connector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(connector)
    credentials = connector.read_credentials()
    sock = socket.create_connection((connector.ASPIRE2A_HOST, 22), timeout=20)
    nscc = connector.paramiko.Transport(sock)
    try:
        nscc.start_client(timeout=20)
        connector.verify_host_key(nscc, connector.ASPIRE2A_HOST, connector.NSCC_KNOWN_HOSTS)
        nscc.auth_password(credentials["NSCC_USERNAME"], credentials["NSCC_PASSWORD"])
        if not nscc.is_authenticated():
            raise RuntimeError("verified ASPIRE2A authentication failed")
        sftp = nscc.open_sftp_client()
        try:
            output.mkdir(parents=True, exist_ok=True)
            for relative in files:
                remote = f"{remote_root}/{relative}"
                size = sftp.stat(remote).st_size
                if size > 200_000:
                    raise RuntimeError(f"refusing unexpectedly large result: {relative}")
                with sftp.open(remote, "rb") as source:
                    data = source.read()
                if len(data) != size:
                    raise RuntimeError(f"short read: {relative}")
                destination = output / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                print(f"saved {relative} ({size} bytes)")
            log_path = f"{remote_root}/job.log"
            size = sftp.stat(log_path).st_size
            with sftp.open(log_path, "rb") as source:
                source.seek(max(0, size - 8192))
                tail = source.read()
            (output / "job_log_tail.txt").write_bytes(tail)
            print(f"saved job log tail ({len(tail)} bytes)")
        finally:
            sftp.close()
    finally:
        nscc.close()


if __name__ == "__main__":
    main()
