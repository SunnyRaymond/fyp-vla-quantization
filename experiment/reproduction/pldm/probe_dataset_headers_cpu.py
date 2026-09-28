"""Inspect selected official PLDM archive members and NPZ array headers in PBS CPU."""
from __future__ import annotations

import json
import os
import shutil
import socket
import tarfile
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/scratch/users/ntu/yguo017/pldm-reproduction")
ARCHIVE = ROOT / "datasets" / "wall_data.tar.gz"
CANDIDATES = {
    "good_quality_data_no_images.npz",
    "len_65_no_images.npz",
    "len_33_no_images.npz",
    "len_17_no_images.npz",
    "ds_size_1500K_no_images.npz",
}
MAX_MEMBER_BYTES = 2_000_000_000
TIME_LIMIT_SECONDS = 1500


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def assert_allocation() -> tuple[str, str]:
    job_id = os.environ.get("PBS_JOBID", "")
    nodefile = os.environ.get("PBS_NODEFILE", "")
    if not job_id or not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("PBS_JOBID and valid PBS_NODEFILE are required")
    host = socket.gethostname().split(".")[0].lower()
    if any(word in host for word in ("login", "head", "submit")):
        raise RuntimeError(f"Refusing probable login node: {host}")
    nodes = {line.split()[0].split(".")[0].lower() for line in Path(nodefile).read_text().splitlines() if line.strip()}
    if host not in nodes:
        raise RuntimeError(f"Current host {host} is not in PBS_NODEFILE")
    return job_id, host


def inspect_npz(path: Path) -> dict:
    arrays = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if not info.filename.endswith(".npy"):
                continue
            with archive.open(info) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise RuntimeError(f"Unsupported NPZ array header version {version} in {info.filename}")
            arrays[info.filename[:-4]] = {
                "shape": list(shape),
                "dtype": str(dtype),
                "fortran_order": bool(fortran_order),
                "npy_member_uncompressed_bytes": info.file_size,
            }
    transition_shape = arrays.get("actions", {}).get("shape") or arrays.get("locations", {}).get("shape")
    return {
        "npz_bytes": path.stat().st_size,
        "array_headers": arrays,
        "transition_array_shape": transition_shape,
        "transition_count_from_first_axis": transition_shape[0] if transition_shape else None,
        "payload_arrays_loaded": False,
    }


def atomic_json(path: Path, payload: dict, job_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{job_id}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    job_id, host = assert_allocation()
    started = time.monotonic()
    deadline = started + TIME_LIMIT_SECONDS
    report = {
        "status": "RUNNING",
        "job_id": job_id,
        "host": host,
        "archive": str(ARCHIVE),
        "started_utc": utc_now(),
        "candidates": {},
        "scope": "Read selected tar member sizes and NPZ .npy headers only; no array payload loads or hashes.",
    }
    report_path = ROOT / "reports" / job_id / "dataset_header_probe.json"
    atomic_json(report_path, report, job_id)
    if not ARCHIVE.is_file():
        raise RuntimeError(f"Official dataset archive missing: {ARCHIVE}")
    with tempfile.TemporaryDirectory(prefix=f"pldm-npz-header-{job_id}-") as temp_name:
        temp_dir = Path(temp_name)
        with tarfile.open(ARCHIVE, mode="r|gz") as archive:
            for member in archive:
                if time.monotonic() > deadline:
                    raise TimeoutError("Dataset header probe reached its 25 minute self-time limit")
                basename = Path(member.name).name
                if basename not in CANDIDATES:
                    continue
                item = {"archive_member": member.name, "archive_member_bytes": member.size}
                if not member.isfile() or member.size > MAX_MEMBER_BYTES:
                    item.update({"status": "SKIPPED", "reason": "not a regular file or larger than the 2 GB probe cap"})
                else:
                    free_bytes = shutil.disk_usage(temp_dir).free
                    if free_bytes < member.size + 64 * 1024 * 1024:
                        item.update({"status": "SKIPPED", "reason": "insufficient temporary filesystem space", "temp_free_bytes": free_bytes})
                    else:
                        extracted = temp_dir / f"{basename}.partial-{job_id}"
                        stream = archive.extractfile(member)
                        if stream is None:
                            item.update({"status": "FAIL", "reason": "tar member could not be read"})
                        else:
                            with stream, extracted.open("wb") as output:
                                shutil.copyfileobj(stream, output, length=4 * 1024 * 1024)
                                output.flush()
                                os.fsync(output.fileno())
                            item.update({"status": "PASS", **inspect_npz(extracted)})
                            extracted.unlink()
                report["candidates"][basename] = item
                atomic_json(report_path, report, job_id)
                print(f"header_probe {basename}: {item.get('status')} outer_bytes={item.get('archive_member_bytes')}", flush=True)
    missing = CANDIDATES - set(report["candidates"])
    report["missing_candidates"] = sorted(missing)
    report["finished_utc"] = utc_now()
    report["elapsed_seconds"] = time.monotonic() - started
    report["status"] = "PASS" if not missing and all(x.get("status") == "PASS" for x in report["candidates"].values()) else "PARTIAL"
    atomic_json(report_path, report, job_id)
    print(f"DATASET_HEADER_PROBE_{report['status']}", flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
