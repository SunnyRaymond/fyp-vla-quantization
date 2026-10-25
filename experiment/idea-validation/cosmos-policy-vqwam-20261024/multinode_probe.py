#!/usr/bin/env python3
"""Small PBSPro pbsdsh environment/host-mapping probe; no model or tensor work."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


JOB_NAME = "cpvq24mprobe"
PBSDSH = "/opt/pbs/bin/pbsdsh"


def _short_host(value: str) -> str:
    return value.strip().split(".", 1)[0].lower()


def _hostname_short() -> str:
    try:
        result = subprocess.run(
            ["hostname", "-s"], capture_output=True, text=True, timeout=5, check=False
        )
        if result.returncode == 0 and result.stdout.strip():
            return _short_host(result.stdout)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return _short_host(socket.gethostname())


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value) or "unknown"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _node_hosts(path: Path) -> list[str]:
    values = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            host = _short_host(line)
            if host:
                values.append(host)
    return values


def _qstat(job_id: str) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["qstat", "-f", job_id], capture_output=True, text=True, timeout=15, check=False
        )
        output = result.stdout + "\n" + result.stderr
        state_match = re.search(r"(?m)^\s*job_state\s*=\s*([^\s]+)", output)
        name_match = re.search(r"(?m)^\s*Job_Name\s*=\s*(.*?)\s*$", output)
        state = state_match.group(1).strip() if state_match else None
        name = name_match.group(1).strip() if name_match else None
        return {
            "query_rc": result.returncode,
            "job_state": state,
            "Job_Name": name,
            "matches_running_probe": result.returncode == 0 and state == "R" and name == JOB_NAME,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "query_rc": None,
            "job_state": None,
            "Job_Name": None,
            "matches_running_probe": False,
            "error": type(exc).__name__,
        }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def run_worker(launch_path: Path) -> int:
    launch = _read_json(launch_path)
    job_id = str(launch.get("pbs_job_id", ""))
    expected_hosts = [str(value).lower() for value in launch.get("expected_host_order", [])]
    probe_dir = Path(str(launch["probe_dir"]))
    receipts_dir = probe_dir / "receipts"
    claims_dir = probe_dir / "claims"
    host = _hostname_short()
    rank = expected_hosts.index(host) if host in expected_hosts else None
    slug = _slug(host)
    receipt_path = receipts_dir / f"{slug}.json"

    receipts_dir.mkdir(parents=True, exist_ok=True)
    claims_dir.mkdir(parents=True, exist_ok=True)
    claim_path = claims_dir / f"{slug}.claim"
    try:
        claim_path.mkdir()
    except FileExistsError:
        print(f"duplicate pbsdsh invocation ignored for host={host}", flush=True)
        return 0

    # Read only the named PBS identity fields. Never serialize the process environment.
    native = {
        key: os.environ.get(key)
        for key in ("PBS_JOBID", "PBS_NODEFILE", "PBS_NODENUM", "PBS_TASKNUM")
    }
    checks: dict[str, bool] = {
        "host_is_not_login": "login" not in host.lower(),
        "host_is_in_launch_mapping": rank is not None,
        "launch_job_id_is_valid": bool(job_id),
        "native_job_id_present": bool(native["PBS_JOBID"]),
        "native_job_id_matches_launch": native["PBS_JOBID"] == job_id,
        "native_nodenum_present": bool(native["PBS_NODENUM"]),
        "native_tasknum_present": bool(native["PBS_TASKNUM"]),
    }

    parent_nodefile = launch.get("parent_native_nodefile")
    effective_nodefile = native["PBS_NODEFILE"] or parent_nodefile
    checks["nodefile_path_is_original_parent_value"] = (
        isinstance(parent_nodefile, str) and bool(parent_nodefile)
        and effective_nodefile == parent_nodefile
        and Path(parent_nodefile).is_absolute()
        and Path(parent_nodefile).name == native["PBS_JOBID"] == job_id
    )
    nodefile_info: dict[str, Any] = {
        "path": effective_nodefile,
        "native_environment_value": native["PBS_NODEFILE"],
        "path_source": "native" if native["PBS_NODEFILE"] else "verified_parent_native_path",
        "exists_and_readable_on_this_host": False,
        "contains_this_host": False,
        "hosts_match_launch_mapping": False,
        "unique_host_order": [],
    }
    nodefile_value = effective_nodefile if checks["nodefile_path_is_original_parent_value"] else None
    if nodefile_value:
        try:
            nodefile = Path(nodefile_value)
            entries = _node_hosts(nodefile)
            unique_hosts = list(dict.fromkeys(entries))
            nodefile_info.update(
                {
                    "exists_and_readable_on_this_host": True,
                    "contains_this_host": host in entries,
                    "hosts_match_launch_mapping": unique_hosts == expected_hosts,
                    "unique_host_order": unique_hosts,
                }
            )
        except (OSError, UnicodeError, ValueError) as exc:
            nodefile_info["read_error"] = type(exc).__name__
    checks["allocation_nodefile_readable_locally"] = nodefile_info["exists_and_readable_on_this_host"]
    checks["allocation_nodefile_contains_this_host"] = nodefile_info["contains_this_host"]
    checks["allocation_nodefile_matches_launch_hosts"] = nodefile_info["hosts_match_launch_mapping"]

    qstat_info = _qstat(native["PBS_JOBID"] or "") if native["PBS_JOBID"] else {
        "query_rc": None,
        "job_state": None,
        "Job_Name": None,
        "matches_running_probe": False,
        "error": "native PBS_JOBID absent",
    }
    checks["qstat_running_job_name_match"] = bool(qstat_info["matches_running_probe"])
    success = all(checks.values())
    receipt = {
        "schema": "cosmos-multinode-probe-worker-v1",
        "pbs_job_id": job_id,
        "rank": rank,
        "host_short": host,
        "host_name": socket.gethostname(),
        "expected_host_order": expected_hosts,
        "native_pbs_identity": native,
        "native_nodefile": nodefile_info,
        "qstat": qstat_info,
        "checks": checks,
        "status": "complete" if success else "failed",
        "worker_exit_code": 0 if success else 1,
    }
    _atomic_json(receipt_path, receipt)
    print(
        f"worker rank={rank} host={host} native_job_match={checks['native_job_id_matches_launch']} "
        f"nodefile_local={checks['allocation_nodefile_readable_locally']} "
        f"qstat_match={checks['qstat_running_job_name_match']} status={receipt['status']}",
        flush=True,
    )
    return 0 if success else 1


def run_parent(root: Path, artifacts: Path) -> int:
    job_id = os.environ.get("PBS_JOBID", "")
    nodefile_value = os.environ.get("PBS_NODEFILE", "")
    host = _hostname_short()
    summary_path = root / "results" / "multinode_probe" / "summary.json"
    probe_dir = artifacts / "multinode_probe"
    summary: dict[str, Any] = {
        "schema": "cosmos-multinode-probe-v1",
        "pbs_job_id": job_id or None,
        "Job_Name_expected": JOB_NAME,
        "status": "failed",
        "complete": False,
        "parent_host_short": host,
        "worker_host_order": [],
        "worker_exitcodes": {},
        "pbsdsh_exit_code": None,
        "workers": [],
    }
    try:
        if not job_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", job_id):
            raise RuntimeError("missing or unsafe native PBS_JOBID")
        if not nodefile_value:
            raise RuntimeError("native PBS_NODEFILE is missing in parent")
        if "login" in host.lower():
            raise RuntimeError("refusing to run parent probe on a login host")
        parent_qstat = _qstat(job_id)
        summary["parent_qstat"] = parent_qstat
        if not parent_qstat["matches_running_probe"]:
            raise RuntimeError("parent qstat did not confirm R/cpvq24mprobe")

        nodefile = Path(nodefile_value)
        node_entries = _node_hosts(nodefile)
        host_order = list(dict.fromkeys(node_entries))
        summary["parent_nodefile"] = {
            "path": nodefile_value,
            "readable": True,
            "entry_count": len(node_entries),
            "unique_host_order": host_order,
            "contains_parent_host": host in node_entries,
        }
        summary["worker_host_order"] = host_order
        if host not in node_entries:
            raise RuntimeError("parent host is absent from native PBS_NODEFILE")
        if len(host_order) != 2:
            raise RuntimeError(f"expected exactly two distinct PBS hosts, found {len(host_order)}")

        receipts_dir = probe_dir / "receipts"
        claims_dir = probe_dir / "claims"
        receipts_dir.mkdir(parents=True, exist_ok=True)
        claims_dir.mkdir(parents=True, exist_ok=True)
        launch_path = probe_dir / "launch.json"
        launch = {
            "schema": "cosmos-multinode-probe-launch-v1",
            "pbs_job_id": job_id,
            "parent_native_nodefile": nodefile_value,
            "expected_host_order": host_order,
            "probe_dir": str(probe_dir),
        }
        _atomic_json(launch_path, launch)
        snapshot_shell = artifacts / "control_sources" / "multinode_probe.sh"
        if not snapshot_shell.is_file():
            raise RuntimeError("snapshotted worker shell is missing")
        command = [PBSDSH, "--", "/bin/bash", "-l", str(snapshot_shell), "worker", str(launch_path)]
        print(f"launching pbsdsh across {len(host_order)} allocated hosts", flush=True)
        try:
            launched = subprocess.run(command, timeout=600, check=False)
            pbsdsh_rc = launched.returncode
        except subprocess.TimeoutExpired:
            pbsdsh_rc = None
            summary["pbsdsh_error"] = "timeout_after_600_seconds"
        except OSError as exc:
            pbsdsh_rc = None
            summary["pbsdsh_error"] = type(exc).__name__
        summary["pbsdsh_exit_code"] = pbsdsh_rc

        receipt_files = sorted(receipts_dir.glob("*.json"))
        receipts = []
        receipt_errors = []
        for path in receipt_files:
            try:
                receipts.append(_read_json(path))
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                receipt_errors.append({"file": path.name, "error": type(exc).__name__})
        summary["workers"] = receipts
        summary["receipt_errors"] = receipt_errors

        expected_slugs = [_slug(value) for value in host_order]
        actual_slugs = [path.stem for path in receipt_files]
        by_host = {item.get("host_short"): item for item in receipts}
        mapping_ok = (
            len(expected_slugs) == 2
            and sorted(actual_slugs) == sorted(expected_slugs)
            and len(by_host) == len(receipts) == 2
            and all(
                by_host.get(expected_host, {}).get("rank") == rank
                and by_host.get(expected_host, {}).get("pbs_job_id") == job_id
                for rank, expected_host in enumerate(host_order)
            )
        )
        workers_ok = mapping_ok and all(
            by_host[expected_host].get("status") == "complete"
            and by_host[expected_host].get("worker_exit_code") == 0
            and all(by_host[expected_host].get("checks", {}).values())
            for expected_host in host_order
        )
        summary["host_mapping_matches_launch_order"] = mapping_ok
        summary["worker_exitcodes"] = {
            str(item.get("host_short", "unknown")): item.get("worker_exit_code", 1)
            for item in receipts
        }
        complete = pbsdsh_rc == 0 and workers_ok and not receipt_errors
        summary["complete"] = bool(complete)
        summary["status"] = "complete" if complete else "failed"
        if not complete:
            summary["failure_reason"] = "PBS dispatch, exact two-host receipt, job binding, host mapping, or worker checks failed"

    except Exception as exc:  # summarize a bounded probe failure without exposing process environment
        summary["failure_reason"] = f"{type(exc).__name__}: {exc}"

    _atomic_json(summary_path, summary)
    marker = artifacts / "PIPELINE_COMPLETE"
    if summary["status"] == "complete":
        try:
            with marker.open("x", encoding="utf-8") as stream:
                stream.write(job_id + "\n")
        except OSError as exc:
            summary["status"] = "failed"
            summary["complete"] = False
            summary["failure_reason"] = f"PIPELINE_COMPLETE write failed: {type(exc).__name__}"
            _atomic_json(summary_path, summary)
            return 1
        return 0
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("launch_json", type=Path)
    parent_parser = subparsers.add_parser("parent")
    parent_parser.add_argument("--root", type=Path, required=True)
    parent_parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "worker":
        return run_worker(args.launch_json)
    return run_parent(args.root, args.artifacts)


if __name__ == "__main__":
    raise SystemExit(main())
