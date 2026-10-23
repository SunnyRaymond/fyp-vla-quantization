"""CPU-only accounting for persisted R8+VQ8 training and evaluation timings."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import socket


PROTOCOL = "fastwam-vq-learned-r128-c8-v1"
REMOTE_ROOT = "/scratch/users/ntu/yguo017/fastwam-vq-learned-r128-c8-20261021"
UPSTREAM_JOB = "25902163.pbs101"
CALIBRATION_CASES = [2000, 2019, 2021, 2025, 2051, 2057, 2066, 2070]
HELDOUT_CASES = list(range(3000, 3040))
SEEDS = [0, 1]
EXPECTED_LINEARS = 614
ROTATION_CHANGE_THRESHOLD = 1e-7


def require_allocation():
    job_id = os.environ.get("PBS_JOBID", "").strip()
    nodefile = os.environ.get("PBS_NODEFILE", "").strip()
    receipt = os.environ.get("PBS_ALLOCATION_RECEIPT", "").strip()
    if not job_id or not nodefile or not receipt:
        raise RuntimeError("PBS_JOBID, PBS_NODEFILE, and allocation receipt are required")
    host = socket.gethostname().split(".", 1)[0]
    if "login" in host.lower():
        raise RuntimeError("Refusing to run accounting on a login node")
    path = Path(nodefile)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("PBS_NODEFILE is unavailable or empty")
    allocated = {
        line.strip().split("/", 1)[0].split(".", 1)[0]
        for line in path.read_text().splitlines()
        if line.strip()
    }
    if host not in allocated:
        raise RuntimeError(f"Current host {host} is not in the PBS allocation")
    receipt_text = Path(receipt).read_text(encoding="utf-8")
    receipt_job = re.search(r"^\s*Job Id:\s*(\S+)\s*$", receipt_text, re.MULTILINE)
    if (not receipt_job or receipt_job.group(1) != job_id
            or not re.search(r"^\s*job_state\s*=\s*R(?:\s|$)", receipt_text, re.MULTILINE)):
        raise RuntimeError("Host-side PBS allocation receipt does not verify this running job")
    return job_id, host


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_pt(torch, path):
    return torch.load(path, map_location="cpu", weights_only=True)


def require_completed_upstream(root, upstream_job):
    results = root / "results"
    summary = read_json(results / "summary.json")
    artifact_dir = root / "artifacts" / upstream_job
    worker_summary_path = artifact_dir / "worker_summary.json"
    exit_code_path = artifact_dir / "exit_code.txt"
    if summary.get("protocol") != PROTOCOL or summary.get("complete") is not True:
        raise RuntimeError("Upstream results/summary.json is not complete for this protocol")
    if summary.get("pbs_jobid") != upstream_job:
        raise RuntimeError("Upstream summary job ID does not match the requested job")
    if (summary.get("heldout_queries") != 80 or summary.get("calibration_queries") != 8
            or summary.get("new_teacher_queries") != 0):
        raise RuntimeError("Upstream summary query counts/teacher-forward scope differs")
    if not (artifact_dir / "PHASE_COMPLETE").is_file():
        raise RuntimeError("Upstream PBS PHASE_COMPLETE marker is missing")
    if not exit_code_path.is_file() or exit_code_path.read_text().strip() != "0":
        raise RuntimeError("Upstream PBS exit_code.txt does not report success")
    worker_summary = read_json(worker_summary_path)
    if (worker_summary.get("complete") is not True
            or worker_summary.get("protocol") != PROTOCOL
            or worker_summary.get("pbs_jobid") != upstream_job):
        raise RuntimeError("Upstream worker summary is missing or mismatched")
    return summary


def valid_seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0 else None


def phase_report(items, expected_labels):
    """items maps a stable label to seconds; absent/invalid values are missing."""
    valid = {label: valid_seconds(value) for label, value in items.items()}
    observed = {label: value for label, value in valid.items() if value is not None}
    missing = [label for label in expected_labels if valid.get(label) is None]
    values = list(observed.values())
    return {
        "expected_count": len(expected_labels),
        "observed_count": len(observed),
        "missing": missing,
        "sum_seconds": sum(values) if values else None,
        "min_seconds": min(values) if values else None,
        "max_seconds": max(values) if values else None,
        "measurements_seconds": observed,
    }


def history_seconds(history, expected_updates):
    by_update = {}
    duplicate_updates = []
    for row in history if isinstance(history, list) else []:
        if not isinstance(row, dict):
            continue
        update = row.get("update")
        if update in by_update:
            duplicate_updates.append(update)
        by_update[update] = row.get("seconds")
    expected = list(range(1, expected_updates + 1))
    report = phase_report(
        {str(update): by_update.get(update) for update in expected},
        [str(update) for update in expected],
    )
    report["duplicate_updates"] = sorted(set(duplicate_updates), key=str)
    return report


def qstat_walltime(upstream_job, receipt_path):
    """Read host-collected qstat walltime; a missing optional receipt is non-fatal."""
    if not receipt_path or not Path(receipt_path).is_file():
        return {"seconds": None, "status": "unavailable", "source": f"host qstat -xf {upstream_job}"}
    receipt = Path(receipt_path).read_text(encoding="utf-8").strip()
    if receipt == "unavailable":
        return {"seconds": None, "status": "unavailable", "source": f"host qstat -xf {upstream_job}"}
    match = re.fullmatch(r"available\s+(\d+):(\d{2}):(\d{2})", receipt)
    if not match:
        return {"seconds": None, "status": "invalid_receipt", "source": f"host qstat -xf {upstream_job}"}
    hours, minutes, seconds = (int(part) for part in match.groups())
    return {
        "seconds": hours * 3600 + minutes * 60 + seconds,
        "status": "available",
        "source": f"host qstat -xf {upstream_job}: resources_used.walltime",
    }


def rotation_report(torch, path, protocol):
    saved = read_pt(torch, path)
    if saved.get("protocol") != protocol or saved.get("updates") != 8:
        raise RuntimeError("rotation.pt protocol/update binding differs")
    raw = saved.get("raw")
    matrices = saved.get("residual_matrices")
    orthogonality = saved.get("orthogonality")
    if not isinstance(raw, dict) or not isinstance(matrices, dict) or not isinstance(orthogonality, dict):
        raise RuntimeError("rotation.pt is missing raw/matrix/orthogonality records")
    if set(raw) != set(matrices) or set(raw) != set(orthogonality):
        raise RuntimeError("rotation.pt stream sets do not match")

    streams = {}
    for stream in sorted(raw):
        a = raw[stream]
        u = matrices[stream]
        if tuple(a.shape) != (128, 128) or tuple(u.shape) != (128, 128):
            raise RuntimeError(f"Unexpected 128x128 rotation shape for {stream}")
        a64 = a.to(dtype=torch.float64, device="cpu")
        u64 = u.to(dtype=torch.float64, device="cpu")
        if not bool(torch.isfinite(a64).all()) or not bool(torch.isfinite(u64).all()):
            raise RuntimeError(f"Non-finite saved rotation for {stream}")
        delta = u64 - torch.eye(128, dtype=torch.float64, device="cpu")
        max_abs = float(delta.abs().max().item())
        frobenius = float(torch.linalg.vector_norm(delta).item())
        raw_max_abs = float(a64.abs().max().item())
        raw_frobenius = float(torch.linalg.vector_norm(a64).item())
        orth = valid_seconds(orthogonality[stream])
        streams[stream] = {
            "U_minus_I_max_abs": max_abs,
            "U_minus_I_frobenius": frobenius,
            "raw_A_max_abs": raw_max_abs,
            "raw_A_frobenius": raw_frobenius,
            "saved_orthogonality_max_abs": orth,
            "nontrivial_change": max_abs > ROTATION_CHANGE_THRESHOLD,
        }
    changed = [name for name, item in streams.items() if item["nontrivial_change"]]
    del saved
    return {
        "source": "results/rotation.pt",
        "updates": 8,
        "change_threshold_max_abs": ROTATION_CHANGE_THRESHOLD,
        "stream_count": len(streams),
        "changed_streams": changed,
        "any_nontrivial_change": bool(changed),
        "all_streams_nontrivial": bool(streams) and len(changed) == len(streams),
        "streams": streams,
    }


def build_costs(root, upstream_job, summary, walltime_receipt):
    import torch

    protocol = read_json(root / "protocol.json")
    if protocol.get("protocol") != PROTOCOL:
        raise RuntimeError("protocol.json identity differs")
    if (protocol.get("calibration", {}).get("case_ids") != CALIBRATION_CASES
            or protocol.get("evaluation", {}).get("case_ids") != HELDOUT_CASES
            or protocol.get("evaluation", {}).get("seed_indices") != SEEDS
            or protocol.get("target_linears") != EXPECTED_LINEARS):
        raise RuntimeError("Frozen case/seed/Linear counts differ from accounting scope")

    results = root / "results"
    state_payload = read_pt(torch, results / "state.pt")
    if state_payload.get("protocol") != protocol:
        raise RuntimeError("state.pt protocol binding differs")
    state = state_payload.get("state")
    if not isinstance(state, dict) or state.get("stage") != "complete":
        raise RuntimeError("state.pt is not at the complete stage")
    state_timing = state.get("timing", {})
    r_history = history_seconds(state.get("rotation_history"), 8)
    vq_history = history_seconds(state.get("vq_history"), 8)
    for phase, field, report in (
        ("R training", "rotation_updates_seconds", r_history),
        ("VQ training", "vq_updates_seconds", vq_history),
    ):
        recorded = valid_seconds(state_timing.get(field)) if isinstance(state_timing, dict) else None
        report["state_timing_field"] = field
        report["state_timing_seconds"] = recorded
        if recorded is not None and report["sum_seconds"] is not None:
            report["history_minus_state_seconds"] = report["sum_seconds"] - recorded
    lineage = list(state.get("lineage", []))
    del state_payload

    moment_items = {}
    for case in CALIBRATION_CASES:
        label = str(case)
        path = results / "moments" / f"{case}.pt"
        if path.is_file():
            record = read_pt(torch, path)
            if (record.get("protocol") != protocol or record.get("case_id") != case
                    or len(record.get("moments", {})) != EXPECTED_LINEARS):
                raise RuntimeError(f"Moment record binding/coverage differs for {case}")
            moment_items[label] = record.get("seconds")
            del record
    moments = phase_report(moment_items, [str(case) for case in CALIBRATION_CASES])

    fit_items = {}
    encoded_values = {}
    for index in range(EXPECTED_LINEARS):
        label = f"{index:04d}"
        path = results / "banks" / f"{label}.pt"
        if path.is_file():
            record = read_pt(torch, path)
            if (record.get("protocol") != PROTOCOL or record.get("module_index") != index
                    or record.get("rotation_updates") != 8):
                raise RuntimeError(f"Fitted bank binding differs for index {index}")
            fit_items[label] = record.get("seconds")
            encoded_values[label] = record.get("encoded_bytes")
            del record
    vq_fit = phase_report(fit_items, [f"{i:04d}" for i in range(EXPECTED_LINEARS)])
    encoded_bytes = {
        key: value for key, value in encoded_values.items()
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
    }
    encoded_missing = [
        f"{i:04d}" for i in range(EXPECTED_LINEARS)
        if f"{i:04d}" not in encoded_bytes
    ]
    encoded_storage = {
        "expected_bank_count": EXPECTED_LINEARS,
        "observed_count": len(encoded_bytes),
        "missing": encoded_missing,
        "total_encoded_bytes": sum(encoded_bytes.values()) if encoded_bytes else None,
        "total_encoded_mib": sum(encoded_bytes.values()) / (1024 ** 2) if encoded_bytes else None,
        "per_bank_bytes_min": min(encoded_bytes.values()) if encoded_bytes else None,
        "per_bank_bytes_max": max(encoded_bytes.values()) if encoded_bytes else None,
    }

    def endpoint_phase(split, case_ids, seeds):
        items = {}
        expected = []
        for case in case_ids:
            for seed in seeds:
                label = f"{case}_{seed}"
                expected.append(label)
                path = results / split / "queries" / f"{label}.json"
                if not path.is_file():
                    continue
                record = read_json(path)
                if (record.get("protocol") != PROTOCOL or record.get("split") != split
                        or record.get("case_id") != case or record.get("seed_index") != seed):
                    raise RuntimeError(f"Endpoint query binding differs for {split}/{label}")
                items[label] = record.get("seconds")
        return phase_report(items, expected)

    calibration = endpoint_phase("calibration", CALIBRATION_CASES, [0])
    heldout = endpoint_phase("heldout", HELDOUT_CASES, SEEDS)
    rotation = rotation_report(torch, results / "rotation.pt", protocol)
    del torch

    phases = {
        "R_training": r_history,
        "moments_8": moments,
        "VQ_fit_614": vq_fit,
        "VQ_training": vq_history,
        "calibration_8": calibration,
        "heldout_80": heldout,
    }
    complete = all(not item["missing"] for item in phases.values()) and not encoded_missing
    totals = [item["sum_seconds"] for item in phases.values() if item["sum_seconds"] is not None]
    return {
        "protocol": PROTOCOL,
        "upstream_job_id": upstream_job,
        "upstream_status": summary.get("status"),
        "accounting_pbs_job_id": os.environ.get("PBS_JOBID"),
        "accounting_host": socket.gethostname(),
        "complete_coverage": complete,
        "rotation_learning_confirmed": rotation["any_nontrivial_change"],
        "new_model_forwards": 0,
        "new_teacher_queries": summary.get("new_teacher_queries", 0),
        "state_lineage": lineage,
        "phases": phases,
        "total_measured_operation_seconds": sum(totals) if totals else None,
        "encoded_storage": encoded_storage,
        "rotation_learning_evidence": rotation,
        "pbs_walltime": qstat_walltime(upstream_job, walltime_receipt),
        "interpretation": {
            "operation_seconds": "Persisted monotonic stage/query timings; not GPU kernel latency.",
            "pbs_walltime": "Whole upstream PBS attempt walltime, reported separately; state timings may span resumed attempts.",
            "startup_checkpoint_overhead": "Not estimated from phase timers; inspect PBS walltime and per-job logs separately.",
            "equal_compute": False,
        },
        "sources": {
            "state": str(results / "state.pt"),
            "rotation": str(results / "rotation.pt"),
            "moments_pattern": str(results / "moments" / "{case_id}.pt"),
            "fitted_banks_pattern": str(results / "banks" / "{index:04d}.pt"),
            "calibration_queries_pattern": str(results / "calibration" / "queries" / "{case_id}_0.json"),
            "heldout_queries_pattern": str(results / "heldout" / "queries" / "{case_id}_{seed}.json"),
        },
    }


def render_markdown(costs):
    labels = {
        "R_training": "R8 rotation training",
        "moments_8": "8 个 C8 moments 收集查询",
        "VQ_fit_614": "614 个 Linear 的 VQ fitting/refitting",
        "VQ_training": "VQ8 task training",
        "calibration_8": "Calibration 8 queries",
        "heldout_80": "Heldout 80 queries",
    }
    lines = [
        "# H+VQ8 与 R8+VQ8：额外成本记录",
        "",
        f"- Protocol：`{costs['protocol']}`；上游 PBS job：`{costs['upstream_job_id']}`。",
        f"- 覆盖完整：`{costs['complete_coverage']}`；新增 model forwards：`{costs['new_model_forwards']}`；新增 teacher queries：`{costs['new_teacher_queries']}`。",
        "- 下表记录持久化的阶段/query 操作秒数。它们是阶段计时，不是 GPU kernel latency，也不代表等算力。",
        "",
        "| 阶段 | 观测/预期 | 合计秒数 | 单项范围（秒） | 缺失项 |",
        "|---|---:|---:|---:|---|",
    ]
    for key, label in labels.items():
        item = costs["phases"][key]
        total = "—" if item["sum_seconds"] is None else f"{item['sum_seconds']:.3f}"
        if item["min_seconds"] is None:
            span = "—"
        else:
            span = f"{item['min_seconds']:.3f}–{item['max_seconds']:.3f}"
        missing = "无" if not item["missing"] else ", ".join(item["missing"])
        lines.append(f"| {label} | {item['observed_count']}/{item['expected_count']} | {total} | {span} | {missing} |")
    total = costs["total_measured_operation_seconds"]
    lines.extend([
        "",
        "已测阶段合计：**{} 秒**。该合计只加总表中各阶段的已测操作时间。".format(
            "—" if total is None else f"{total:.3f}"
        ),
        "",
        f"Encoded storage：{costs['encoded_storage']['observed_count']}/{costs['encoded_storage']['expected_bank_count']} banks，",
        "总计 {} bytes ({} MiB)；".format(
            "—" if costs["encoded_storage"]["total_encoded_bytes"] is None
            else costs["encoded_storage"]["total_encoded_bytes"],
            "—" if costs["encoded_storage"]["total_encoded_mib"] is None
            else f"{costs['encoded_storage']['total_encoded_mib']:.3f}",
        ),
        f"缺失项：{', '.join(costs['encoded_storage']['missing']) if costs['encoded_storage']['missing'] else '无'}。",
        "",
        f"PBS walltime（单次上游作业）：{costs['pbs_walltime']['seconds']} 秒；读取状态：`{costs['pbs_walltime']['status']}`。",
        "Walltime 与阶段计时分开报告；其中包含启动、checkpoint、排队外的作业执行开销等，且 state 可能跨 resume 作业累积，因此不从两者相减推断启动/checkpoint 成本。失败尝试的 walltime 由外部 qstat 记录。",
        "",
        "## R8 实际变化证据",
        "",
        "R8 实际变化核验：**{}**。rotation.pt 声明 8 次更新；按阈值 `max_abs(U-I) > {:.0e}`，发生非平凡变化的 streams：".format(
            "已确认" if costs["rotation_learning_evidence"]["any_nontrivial_change"] else "未确认",
            costs["rotation_learning_evidence"]["change_threshold_max_abs"],
        ),
        (", ".join(costs['rotation_learning_evidence']['changed_streams']) or "无").join(["**", "**"]),
        "",
        "| Stream | max_abs(U-I) | Frobenius(U-I) | max_abs(raw A) | saved orthogonality max_abs |",
        "|---|---:|---:|---:|---:|",
    ])
    for stream, item in costs["rotation_learning_evidence"]["streams"].items():
        lines.append(
            f"| {stream} | {item['U_minus_I_max_abs']:.6g} | {item['U_minus_I_frobenius']:.6g} | "
            f"{item['raw_A_max_abs']:.6g} | {item['saved_orthogonality_max_abs']:.6g} |"
        )
    lines.extend([
        "",
        "R8 确认以保存的残差变换矩阵相对 identity 的数值为准；更新计数本身不作为变化证据。上述配对结果为描述性统计，不声称统计显著、闭环性能或 native-kernel 加速。",
        "",
        "## 来源",
        "",
        f"- `state.pt`：`{costs['sources']['state']}`（update history）。",
        f"- `rotation.pt`：`{costs['sources']['rotation']}`（raw A、residual U、保存的正交误差）。",
        f"- Moments：`{costs['sources']['moments_pattern']}`；VQ fits：`{costs['sources']['fitted_banks_pattern']}`。",
        f"- Calibration queries：`{costs['sources']['calibration_queries_pattern']}`；heldout queries：`{costs['sources']['heldout_queries_pattern']}`。",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(REMOTE_ROOT))
    parser.add_argument("--upstream-job", default=UPSTREAM_JOB)
    parser.add_argument(
        "--walltime-receipt",
        type=Path,
        default=os.environ.get("PBS_UPSTREAM_WALLTIME_RECEIPT"),
    )
    args = parser.parse_args()
    job_id, host = require_allocation()
    root = args.root.resolve()
    summary = require_completed_upstream(root, args.upstream_job)
    costs = build_costs(root, args.upstream_job, summary, args.walltime_receipt)
    costs["accounting_host"] = host
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    temporary = out / "costs.json.tmp"
    temporary.write_text(json.dumps(costs, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(out / "costs.json")
    (root / "COSTS.zh.md").write_text(render_markdown(costs), encoding="utf-8")
    print(f"COST_ACCOUNTING_COMPLETE upstream={args.upstream_job} accounting={job_id} coverage={costs['complete_coverage']}", flush=True)


if __name__ == "__main__":
    main()
