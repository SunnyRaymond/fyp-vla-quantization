"""Fail closed unless this process runs inside the current CCDS SLURM job."""
from __future__ import annotations

import getpass
import os
import re
import socket
import subprocess
from typing import Any, Dict


_JOB_ID = re.compile(r"^[0-9]+(?:_[0-9]+)?$")
_HOST = re.compile(r"^TC1N[0-9]{2}$")


def _run(*args: str) -> str:
    result = subprocess.run(
        list(args), check=True, text=True, capture_output=True, timeout=30
    )
    return result.stdout.strip()


def require_allocation() -> Dict[str, Any]:
    """Return scheduler evidence, or raise before any workload/model I/O."""
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    host = socket.gethostname().split(".", 1)[0]
    env_nodelist = os.environ.get("SLURM_JOB_NODELIST", "").strip()
    if not _JOB_ID.fullmatch(job_id):
        raise RuntimeError("a real SLURM_JOB_ID is required; do not fabricate it")
    if not _HOST.fullmatch(host):
        raise RuntimeError(f"actual host {host!r} is not a CCDS TC1 compute node")
    if not env_nodelist:
        raise RuntimeError("SLURM_JOB_NODELIST is missing")

    record = _run("scontrol", "show", "job", "-o", job_id)
    fields = dict(item.split("=", 1) for item in record.split() if "=" in item)
    if fields.get("JobId", "").split("_", 1)[0] != job_id.split("_", 1)[0]:
        raise RuntimeError("scontrol job record does not match SLURM_JOB_ID")
    if fields.get("JobState") != "RUNNING":
        raise RuntimeError(f"SLURM job is not RUNNING: {fields.get('JobState')!r}")

    user = getpass.getuser()
    user_id = fields.get("UserId", "")
    recorded_user = user_id.split("(", 1)[0]
    if not recorded_user or recorded_user != user:
        raise RuntimeError("scontrol UserId does not match the current user")

    recorded_nodes = fields.get("NodeList", "")
    if not recorded_nodes:
        raise RuntimeError("scontrol NodeList is missing")
    try:
        allocated_hosts = {
            line.strip().split(".", 1)[0]
            for line in _run("scontrol", "show", "hostnames", recorded_nodes).splitlines()
            if line.strip()
        }
        env_hosts = {
            line.strip().split(".", 1)[0]
            for line in _run("scontrol", "show", "hostnames", env_nodelist).splitlines()
            if line.strip()
        }
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("scontrol could not expand the allocated NodeList") from exc
    if host not in allocated_hosts or host not in env_hosts:
        raise RuntimeError("actual hostname is absent from the SLURM allocation")
    if allocated_hosts != env_hosts:
        raise RuntimeError("SLURM_JOB_NODELIST disagrees with scontrol NodeList")

    return {
        "scheduler": "slurm",
        "job_id": job_id,
        "hostname": host,
        "nodelist": recorded_nodes,
        "user": user,
        "verified": True,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(require_allocation(), separators=(",", ":")))
