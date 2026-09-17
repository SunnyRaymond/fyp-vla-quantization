"""Fail-closed PBS allocation checks for the Wall Exp1 workload."""
from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any


EXPECTED_PROJECT = "personal-yguo017"


class AllocationError(RuntimeError):
    """Raised before any model/data workload is allowed to start."""


def require_compute_allocation(expected_project: str = EXPECTED_PROJECT) -> dict[str, Any]:
    job_id = os.environ.get("PBS_JOBID", "").strip()
    if not job_id:
        raise AllocationError("PBS_JOBID is required; refusing to run outside a PBS allocation")
    hostname = socket.gethostname()
    short_host = hostname.split(".", 1)[0]
    lowered = f"{hostname} {short_host}".lower()
    if any(token in lowered for token in ("login", "head", "submit")):
        raise AllocationError(f"login/head node is not a compute allocation: {hostname}")
    nodefile = os.environ.get("PBS_NODEFILE", "").strip()
    if not nodefile or not Path(nodefile).is_file():
        raise AllocationError("PBS_NODEFILE is missing; refusing an unverified allocation")

    qstat = shutil.which("qstat")
    if qstat is None:
        raise AllocationError("qstat is required to verify the PBS project/account")
    try:
        result = subprocess.run(
            [qstat, "-f", job_id],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise AllocationError("cannot verify the current PBS job with qstat") from exc
    project_match = re.search(r"(?:^|\n)\s*project\s*=\s*([^\s\n]+)", result.stdout)
    if project_match is None:
        raise AllocationError("qstat did not expose a project field; refusing to guess the account")
    project = project_match.group(1).strip()
    if project != expected_project:
        raise AllocationError(f"PBS project mismatch: {project!r} != {expected_project!r}")
    select_match = re.search(r"(?:Resource_List\.select|select)\s*=\s*([^\n]+)", result.stdout)
    exec_vnode_match = re.search(r"exec_vnode\s*=\s*([^\n]+)", result.stdout)
    resource_text = " ".join(
        item for item in (
            select_match.group(1) if select_match else "",
            exec_vnode_match.group(1) if exec_vnode_match else "",
        ) if item
    )
    gpu_match = re.search(
        r"(?:ngpus|gpu(?:s)?|accelerator)\s*=?\s*[:=]?\s*(\d+)",
        resource_text,
        re.IGNORECASE,
    )
    if gpu_match is None or int(gpu_match.group(1)) < 1:
        raise AllocationError("qstat does not prove ngpus>=1 in Resource_List.select/exec_vnode")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cuda_visible or cuda_visible.lower() in {"nodevfiles", "none", "void"}:
        raise AllocationError("CUDA_VISIBLE_DEVICES must expose at least one GPU")
    return {
        "verified": True,
        "scheduler": "pbs",
        "job_id": job_id,
        "hostname": hostname,
        "nodefile": nodefile,
        "project": project,
        "project_check": "qstat -f",
        "gpu_resource_check": "qstat select/exec_vnode ngpus>=1",
        "cuda_visible_devices": cuda_visible,
        "qstat_resource_text": resource_text,
    }


def verify_visible_a100() -> dict[str, Any]:
    """Verify the single visible A100 only after PBS checks have passed."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cuda_visible or cuda_visible.lower() in {"nodevfiles", "none", "void"}:
        raise AllocationError("CUDA_VISIBLE_DEVICES is empty or disables all GPUs")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise AllocationError("nvidia-smi GPU verification failed") from exc
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            rows.append({"index": int(fields[0]), "name": fields[1], "memory_mib": int(fields[2])})
        except ValueError:
            continue
    if len(rows) != 1:
        raise AllocationError(f"Exp1 requires exactly one visible GPU, found {len(rows)}")
    if "a100" not in rows[0]["name"].lower() or rows[0]["memory_mib"] < 30000:
        raise AllocationError(f"Exp1 requires a full A100, found {rows[0]!r}")
    return {"verified": True, "visible_gpus": rows, "cuda_visible_devices": cuda_visible}


if __name__ == "__main__":
    import json

    try:
        allocation = require_compute_allocation()
        allocation["gpu"] = verify_visible_a100()
        print(json.dumps(allocation, sort_keys=True))
    except Exception as exc:
        print(f"allocation guard refused: {type(exc).__name__}: {exc}")
        raise SystemExit(64)
