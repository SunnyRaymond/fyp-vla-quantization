"""PBS-native two-node coordinator for the frozen eight-worker evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import time

import aggregate
import evaluate


def _host(value):
    return str(value).strip().lower().split(".", 1)[0]


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _qstat_attributes(output):
    attributes = {}
    pattern = re.compile(r"^(?: {4}|(?=\S))([A-Za-z_][A-Za-z0-9_.-]*)\s*=\s*(.*)$")
    lines = output.splitlines()
    for index, line in enumerate(lines):
        match = pattern.match(line)
        if not match:
            continue
        key = match.group(1)
        parts = [match.group(2).strip()]
        for continuation in lines[index + 1:]:
            if not continuation.strip():
                break
            expanded = continuation.expandtabs(8)
            indent = len(expanded) - len(expanded.lstrip())
            if indent <= 4:
                break
            parts.append(continuation.strip())
        attributes[key] = "".join(part for part in parts if part)
    return attributes


def _qstat(job_id):
    result = subprocess.run(["qstat", "-xf", job_id], check=False, capture_output=True,
                            text=True, timeout=35)
    if result.returncode:
        raise RuntimeError("Cannot query current PBS allocation: " + result.stderr[-1000:])
    return _qstat_attributes(result.stdout)


def _exec_vnodes(value):
    chunks = re.findall(r"\(([^()]*)\)", value or "")
    if not chunks:
        chunks = [part for part in (value or "").split("+") if part]
    result = []
    for chunk in chunks:
        fields = chunk.strip().strip("() ").split(":")
        if not fields or not fields[0]:
            continue
        resources = {}
        for field in fields[1:]:
            key, separator, content = field.partition("=")
            if separator:
                resources[key] = content
        result.append({"hostname": _host(fields[0]), "resources": resources})
    return result


def _nodefile_hosts(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 1_000_000:
        raise RuntimeError("PBS_NODEFILE is missing or too large")
    hosts = [_host(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(hosts) != 2 or len(set(hosts)) != 2:
        raise RuntimeError("The 8-GPU evaluation requires two distinct PBS_NODEFILE hosts")
    return hosts


def _verify_g4_qstat(job_id):
    attrs = _qstat(job_id)
    if (attrs.get("job_state") != "R" or attrs.get("Job_Name") != "cpvq24eval"
            or attrs.get("queue") != "g4"):
        raise RuntimeError("PBS evaluation must be the running cpvq24eval job routed to g4")
    chunks = _exec_vnodes(attrs.get("exec_vnode", ""))
    if len(chunks) != 2:
        raise RuntimeError("qstat exec_vnode must contain exactly two g4 node allocations")
    for chunk in chunks:
        try:
            gpus = int(chunk["resources"].get("ngpus", "-1"))
            cpus = int(chunk["resources"].get("ncpus", "-1"))
        except ValueError as exc:
            raise RuntimeError("qstat exec_vnode has an unverifiable per-node resource count") from exc
        if gpus != 4 or cpus != 64:
            raise RuntimeError("Each allocated g4 host must have exactly 4 GPUs and 64 CPUs")
    return attrs, chunks


def _verify_nodefile_hosts(nodefile, chunks, expected_hosts=None):
    hosts = _nodefile_hosts(nodefile)
    if expected_hosts is not None and hosts != [_host(host) for host in expected_hosts]:
        raise RuntimeError("PBS_NODEFILE node order differs from the launch receipt")
    if {chunk["hostname"] for chunk in chunks} != set(hosts):
        raise RuntimeError("qstat exec_vnode does not match the two PBS_NODEFILE hosts")
    return hosts


def _verify_g4_allocation(protocol, job_id, nodefile, expected_hosts=None):
    if evaluate.expected_worker_count(protocol) != 8:
        raise RuntimeError("PBSdsh coordinator is only valid for the two-node 8-worker profile")
    attrs, chunks = _verify_g4_qstat(job_id)
    hosts = _verify_nodefile_hosts(nodefile, chunks, expected_hosts)
    return hosts, attrs, chunks


def _load_protocol(path):
    protocol = evaluate.validate_protocol(evaluate.read_json(path))
    if evaluate.expected_worker_count(protocol) != 8:
        raise RuntimeError("This coordinator only runs the two-node 8-worker profile")
    return protocol


def make_launch_receipt(protocol, root, out, artifacts, soft_deadline_seconds):
    job_id = os.environ.get("PBS_JOBID")
    nodefile = os.environ.get("PBS_NODEFILE")
    hostname = socket.gethostname().lower()
    if not job_id or not nodefile or "login" in hostname:
        raise RuntimeError("Coordinator requires a genuine non-login PBS compute allocation")
    nodefile_path = Path(nodefile)
    if not nodefile_path.is_absolute() or nodefile_path.name != job_id:
        raise RuntimeError("Native parent PBS_NODEFILE path must be absolute and named for PBS_JOBID")
    evaluate._guard_allocation()
    hosts, attrs, chunks = _verify_g4_allocation(protocol, job_id, nodefile)
    if _host(hostname) not in hosts:
        raise RuntimeError("Coordinator host is not one of the allocated g4 nodes")
    artifacts = Path(artifacts).resolve()
    control_sources = artifacts / "control_sources"
    node_script = control_sources / "evaluation_node.sh"
    if not node_script.is_file():
        raise RuntimeError("Snapshotted evaluation_node.sh is missing")
    now = time.time()
    soft_deadline_seconds = float(soft_deadline_seconds)
    if soft_deadline_seconds <= 0:
        raise ValueError("Global soft deadline must be positive")
    job_start_epoch = float(os.environ.get("COSMOS_JOB_START_EPOCH", now))
    receipt = {
        "format": "cosmos_evaluation_multinode_launch_v1",
        "protocol": protocol["protocol"], "pbs_job_id": job_id,
        "pbs_nodefile": nodefile,
        "nodes": [{"rank": rank, "hostname": host} for rank, host in enumerate(hosts)],
        "expected_workers": 8, "workers_per_node": 4,
        "deadline_epoch": job_start_epoch + soft_deadline_seconds,
        "soft_deadline_seconds": soft_deadline_seconds,
        "started_epoch": job_start_epoch,
        "coordinator_started_epoch": now,
        "root": str(Path(root).resolve()), "out": str(Path(out).resolve()),
        "artifacts_root": str(artifacts),
        "control_sources": str(control_sources),
        "engine_sources": str(artifacts / "engine_sources"),
        "legacy_sources": str(artifacts / "legacy_sources"),
        "source_helpers": str(artifacts / "source_helpers"),
        "node_script": str(node_script),
        "protocol_path": str(control_sources / "protocol.json"),
        "exec_vnode": attrs["exec_vnode"],
        "per_node_allocations": chunks,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    return receipt


def pbsdsh_command(launch):
    return ["/opt/pbs/bin/pbsdsh", "--", "/bin/bash", "-l", launch["node_script"],
            str(Path(launch["artifacts_root"]) / "evaluation_multinode_launch.json")]


def _valid_worker_map(value, rank):
    expected = {f"cosmos-gpu-worker-{worker}" for worker in range(rank * 4, rank * 4 + 4)}
    if not isinstance(value, dict) or set(value) != expected:
        return False
    return all(type(code) is int for code in value.values())


def collect_node_receipts(launch, pbsdsh_rc):
    worker_codes = {f"cosmos-gpu-worker-{worker}": None for worker in range(8)}
    valid_ranks = set()
    errors = []
    node_receipts = []
    artifacts = Path(launch["artifacts_root"])
    for node in launch["nodes"]:
        rank, expected_host = node["rank"], _host(node["hostname"])
        path = artifacts / "nodes" / f"node_{rank}" / "node_receipt.json"
        if not path.is_file():
            errors.append(f"missing node receipt rank={rank}")
            continue
        try:
            receipt = evaluate.read_json(path)
        except Exception as exc:
            errors.append(f"invalid node receipt rank={rank}: {type(exc).__name__}")
            continue
        mapped = receipt.get("worker_exitcodes")
        if (receipt.get("protocol") != launch["protocol"]
                or receipt.get("pbs_job_id") != launch["pbs_job_id"]
                or receipt.get("node_rank") != rank or _host(receipt.get("host", "")) != expected_host
                or not _valid_worker_map(mapped, rank)):
            errors.append(f"node receipt identity or worker IDs mismatch rank={rank}")
            continue
        valid_ranks.add(rank)
        node_receipts.append(receipt)
        for worker_id, code in mapped.items():
            worker_codes[worker_id] = code
        if receipt.get("exit_code") != 0:
            errors.append(f"node wrapper failed rank={rank} exit={receipt.get('exit_code')}")
    if pbsdsh_rc != 0:
        errors.append(f"pbsdsh exited {pbsdsh_rc}")
    complete = len(valid_ranks) == 2 and pbsdsh_rc == 0 and not errors
    return worker_codes, node_receipts, complete, errors


def _summarize(protocol, root, out, pbs_job_id, worker_exitcodes, node_receipts_complete,
               launcher_rc, node_receipts, errors):
    manifest_path = Path(protocol["cohort"]["manifest"])
    _, rows = evaluate.load_manifest(manifest_path, protocol)
    new_records = evaluate.load_completed(out, rows, protocol)
    reused_records = evaluate.load_reused_records(root, rows, protocol, manifest_path)
    summary = aggregate.summarize(protocol, rows, new_records, reused_records,
                                  worker_exitcodes, pbs_job_id,
                                  node_receipts_complete=node_receipts_complete)
    summary["node_receipts"] = node_receipts
    summary["node_launcher_exit_code"] = launcher_rc
    summary["node_collection_errors"] = errors
    evaluate.save_json(Path(out) / "summary.json", summary)
    evaluate.save_json(Path(out) / "progress.json", {
        "protocol": evaluate.PROTOCOL, "status": summary["status"],
        "complete": summary["complete"], "completed_slots": summary["new_episode_slots"],
        "expected_slots": 1120, "pbs_job_id": pbs_job_id,
    })
    return summary


def run_coordinator(args):
    protocol = _load_protocol(args.protocol)
    root, out, artifacts = Path(args.root), Path(args.out), Path(args.artifacts)
    out.mkdir(parents=True, exist_ok=True)
    launch = make_launch_receipt(protocol, root, out, artifacts, args.soft_deadline_seconds)
    launch_path = artifacts / "evaluation_multinode_launch.json"
    if launch_path.exists():
        raise RuntimeError("A launch receipt already exists for this PBS evaluation; refusing replay")
    _atomic_json(launch_path, launch)
    command = pbsdsh_command(launch)
    pbsdsh_path = Path(command[0])
    if not pbsdsh_path.is_file():
        raise RuntimeError("Native PBS pbsdsh is unavailable at /opt/pbs/bin/pbsdsh")
    print("PBSDSH_COMMAND=" + " ".join(command), flush=True)
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", flush=True)
    worker_codes, receipts, nodes_complete, errors = collect_node_receipts(launch, result.returncode)
    evaluate.save_json(out / "worker_exitcodes.json", {
        "pbs_job_id": launch["pbs_job_id"], "expected_worker_count": 8,
        "worker_exitcodes": worker_codes, "node_receipts_complete": nodes_complete,
        "node_receipt_paths": [str(artifacts / "nodes" / f"node_{r}" / "node_receipt.json")
                               for r in range(2)],
    })
    summary = _summarize(protocol, root, out, launch["pbs_job_id"], worker_codes,
                         nodes_complete, result.returncode, receipts, errors)
    print(f"COSMOS_EVALUATION status={summary['status']} new_slots={summary['new_episode_slots']}/1120 "
          f"all_methods={summary['all_methods']['episode_slots']}/1680 nodes={len(receipts)}/2",
          flush=True)
    if not nodes_complete or any(code != 0 for code in worker_codes.values()):
        return 1
    return 0


def claim_node(launch_path):
    launch_path = Path(launch_path).resolve()
    launch = evaluate.read_json(launch_path)
    job_id = os.environ.get("PBS_JOBID")
    nodefile = os.environ.get("PBS_NODEFILE")
    tasknum = os.environ.get("PBS_TASKNUM")
    host_full = socket.gethostname().lower()
    hostname = _host(host_full)
    if (not job_id or job_id != launch.get("pbs_job_id") or not tasknum
            or "login" in host_full):
        raise RuntimeError("pbsdsh node lacks the native current PBS_JOBID/PBS_TASKNUM identity")
    parent_nodefile = launch.get("pbs_nodefile")
    parent_path = Path(parent_nodefile) if isinstance(parent_nodefile, str) else Path()
    if not parent_path.is_absolute() or parent_path.name != job_id:
        raise RuntimeError("Launch receipt parent PBS_NODEFILE path does not match the native job ID")
    protocol = evaluate.validate_protocol(evaluate.read_json(launch["protocol_path"]))
    if evaluate.expected_worker_count(protocol) != 8:
        raise RuntimeError("PBSdsh node is not assigned the two-node 8-worker profile")
    attrs, chunks = _verify_g4_qstat(job_id)
    if hostname not in {chunk["hostname"] for chunk in chunks}:
        raise RuntimeError("Current pbsdsh host is not present in qstat exec_vnode allocations")
    if nodefile:
        if nodefile != parent_nodefile:
            raise RuntimeError("Native PBS_NODEFILE differs from the verified parent allocation path")
        effective_nodefile = nodefile
        nodefile_source = "native_environment"
    else:
        effective_nodefile = parent_nodefile
        nodefile_source = "verified_parent_native_path"
    hosts = _verify_nodefile_hosts(effective_nodefile, chunks,
                                   [node["hostname"] for node in launch["nodes"]])
    ranks = [node["rank"] for node in launch["nodes"] if _host(node["hostname"]) == hostname]
    if len(ranks) != 1 or hosts[ranks[0]] != hostname:
        raise RuntimeError("Current pbsdsh host is not mapped to exactly one launch node rank")
    rank = ranks[0]
    artifacts_root = Path(launch["artifacts_root"])
    node_dir = artifacts_root / "nodes" / f"node_{rank}"
    claim_dir = artifacts_root / "node_claims" / f"node_{rank}"
    claim_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        claim_dir.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"Node rank {rank} already has an allocation-time claim; refusing duplicate launch") from exc
    node_dir.mkdir(parents=True, exist_ok=True)
    node_info = {
        "node_rank": rank, "host": hostname, "node_dir": str(node_dir),
        "claim_dir": str(claim_dir), "launch_receipt": str(launch_path),
        "root": launch["root"], "out": launch["out"], "artifacts_root": str(artifacts_root),
        "control_sources": launch["control_sources"], "engine_sources": launch["engine_sources"],
        "legacy_sources": launch["legacy_sources"], "source_helpers": launch["source_helpers"],
        "protocol_path": launch["protocol_path"], "node_script": launch["node_script"],
        "deadline_epoch": launch["deadline_epoch"],
        "soft_deadline_seconds": launch["soft_deadline_seconds"],
        "pbs_job_id": job_id, "pbs_tasknum": tasknum,
        "effective_nodefile": effective_nodefile, "nodefile_source": nodefile_source,
        "native_nodefile": nodefile if nodefile else "(missing)",
    }
    return node_info


def write_node_receipt(launch_path, node_rank, exit_code):
    launch = evaluate.read_json(launch_path)
    job_id = os.environ.get("PBS_JOBID")
    if job_id != launch.get("pbs_job_id") or node_rank not in (0, 1):
        raise RuntimeError("Cannot write node receipt for a different PBS job or invalid rank")
    host = _host(socket.gethostname())
    expected_host = _host(launch["nodes"][node_rank]["hostname"])
    if host != expected_host:
        raise RuntimeError("Node receipt host does not match its assigned rank")
    node_dir = Path(launch["artifacts_root"]) / "nodes" / f"node_{node_rank}"
    worker_path = node_dir / "worker_exitcodes.json"
    worker_codes = {}
    if worker_path.is_file():
        worker_receipt = evaluate.read_json(worker_path)
        if (worker_receipt.get("pbs_job_id") != job_id or worker_receipt.get("node_rank") != node_rank
                or _host(worker_receipt.get("host", "")) != host):
            raise RuntimeError("Node worker receipt identity does not match the native PBS node")
        worker_codes = worker_receipt.get("worker_exitcodes", {})
    receipt = {
        "protocol": launch["protocol"], "pbs_job_id": job_id,
        "node_rank": node_rank, "host": host, "exit_code": int(exit_code),
        "worker_exitcodes": worker_codes,
        "worker_exitcodes_path": str(worker_path),
        "pbs_tasknum": os.environ.get("PBS_TASKNUM"),
        "effective_pbs_nodefile": os.environ.get("PBS_NODEFILE"),
        "pbs_nodefile_source": os.environ.get("COSMOS_NODEFILE_SOURCE"),
        "native_pbs_nodefile": os.environ.get("COSMOS_NATIVE_PBS_NODEFILE"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(node_dir / "node_receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--protocol", type=Path, required=True)
    run_parser.add_argument("--root", type=Path, required=True)
    run_parser.add_argument("--out", type=Path, required=True)
    run_parser.add_argument("--artifacts", type=Path, required=True)
    run_parser.add_argument("--soft-deadline-seconds", type=float, default=85_500)
    claim_parser = subparsers.add_parser("claim-node")
    claim_parser.add_argument("--launch", type=Path, required=True)
    write_parser = subparsers.add_parser("write-node-receipt")
    write_parser.add_argument("--launch", type=Path, required=True)
    write_parser.add_argument("--node-rank", type=int, required=True)
    write_parser.add_argument("--exit-code", type=int, required=True)
    args = parser.parse_args(argv)
    if args.action == "run":
        return run_coordinator(args)
    if args.action == "claim-node":
        print(json.dumps(claim_node(args.launch), ensure_ascii=False))
        return 0
    print(json.dumps(write_node_receipt(args.launch, args.node_rank, args.exit_code), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
