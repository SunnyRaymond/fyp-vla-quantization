"""CPU-only source receipt checks and import of the frozen BF16/SpinQuant episodes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile

HERE = Path(__file__).resolve().parent
REQUIRED_EPISODE_FIELDS = {
    "protocol", "arm", "index", "variant_id", "suite", "dimension", "task_id",
    "task_name", "language", "state_id", "env_seed", "sampler_seed", "complete",
    "closed_loop_complete", "success", "termination", "steps", "replans", "modelqueries",
    "next_observation_count", "episode_seconds", "first_action_chunk",
    "first_next_observation_proprio", "action_min", "action_max", "gripper_action_min",
    "gripper_action_max", "quantization_evidence", "query_metrics", "worker_id",
    "pbs_job_id", "execution_scope",
}
PAIRED_FIELDS = ("variant_id", "suite", "dimension", "task_id", "task_name", "state_id",
                 "env_seed", "sampler_seed", "language")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def require_compute_allocation() -> None:
    job_id, nodefile = os.environ.get("PBS_JOBID", ""), os.environ.get("PBS_NODEFILE", "")
    host = socket.gethostname().split(".", 1)[0]
    if not job_id or not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("CPU preparation requires a live PBS allocation and PBS_NODEFILE")
    if "login" in host.lower():
        raise RuntimeError("Refusing to prepare data on a login node")
    allocated = {line.strip().split(".", 1)[0] for line in Path(nodefile).read_text().splitlines()}
    if host not in allocated:
        raise RuntimeError(f"Current host {host} is absent from PBS_NODEFILE")
    import subprocess
    result = subprocess.run(["qstat", "-f", job_id], check=True, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not any(line.strip() == "job_state = R" for line in result.stdout.splitlines()):
        raise RuntimeError("PBS allocation is not in running state")


def validate_protocols(protocol: dict, old_protocol: dict, summary: dict, reuse: dict) -> None:
    if summary.get("protocol") != "cosmos-policy-c8-w4a4-v1" or summary.get("status") != "complete" \
            or summary.get("complete") is not True:
        raise RuntimeError("Frozen source evaluation is not a complete terminal result")
    if (summary.get("model") != protocol.get("model") or summary.get("mode") != protocol.get("mode")
            or summary.get("cohort_protocol") != protocol["cohort"]["manifest_protocol"]):
        raise RuntimeError("Frozen source summary model, mode, or cohort differs")
    expected = int(protocol["cohort"]["episodes_per_arm"])
    source_arms = {"bf16": expected, "spinquant_w4a4": expected}
    if not source_arms.items() <= summary.get("arm_episodes", {}).items():
        raise RuntimeError("Frozen source summary does not contain 280 episodes for both reused arms")
    if summary.get("episode_slots") != 840 or summary.get("expected_episode_slots") != 840:
        raise RuntimeError("Frozen source summary episode denominator differs from the completed 3 x 280 run")
    if any(summary.get("missing_slots_by_arm", {}).get(arm) != [] for arm in source_arms):
        raise RuntimeError("Frozen source summary reports missing reused episodes")
    if old_protocol.get("model") != protocol.get("model") or old_protocol.get("mode") != protocol.get("mode"):
        raise RuntimeError("Frozen source protocol model or mode differs")
    if old_protocol.get("inference") != protocol.get("inference"):
        raise RuntimeError("Frozen source inference configuration differs")
    old_cohort, new_cohort = old_protocol.get("cohort", {}), protocol["cohort"]
    for key in ("manifest_protocol", "suites", "perturbation_dimensions", "variants_per_cell",
                "state_id", "environment_seed_base", "sampler_seed_base", "seed_stride", "paired_across_arms"):
        if old_cohort.get(key) != new_cohort.get(key):
            raise RuntimeError(f"Frozen source cohort field differs: {key}")
    if reuse.get("whole_cohort_protocol") != new_cohort.get("manifest_protocol"):
        raise RuntimeError("Reuse manifest protocol differs from the frozen cohort")


def copy_episode(source: Path, destination: Path, source_record: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if read_json(destination) != source_record:
            raise RuntimeError(f"Existing imported episode differs from frozen source: {destination}")
        return
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    try:
        os.link(temporary, destination)
    except FileExistsError:
        if read_json(destination) != source_record:
            raise RuntimeError(f"Concurrent imported episode differs from frozen source: {destination}")
    finally:
        temporary.unlink(missing_ok=True)


def import_reused(protocol: dict, root: Path) -> dict:
    reuse = protocol["reuse"]
    legacy_root = Path(protocol.get("legacy_root", protocol["asset_root"]))
    summary_path, episode_root = Path(reuse["summary"]), Path(reuse["episodes"])
    old_protocol_path = legacy_root / "protocol.json"
    required_assets = {
        "legacy_python": legacy_root / "env" / "bin" / "python",
        "cosmos_source": legacy_root / "vendor" / "cosmos-policy",
        "asset_paths": legacy_root / "asset_paths.json",
        "language_requirements": legacy_root / "language_requirements.json",
        "huggingface_cache": legacy_root / "cache" / "huggingface",
        "evaluation_summary": summary_path,
        "evaluation_episodes": episode_root,
        "spinquant_summary": Path(reuse["spinquant_artifact"]) / "summary.json",
        "spinquant_deployment": Path(reuse["spinquant_artifact"]) / "deployment.pt",
        "source_protocol": old_protocol_path,
    }
    missing = [label for label, path in required_assets.items() if not path.exists()]
    if missing:
        raise RuntimeError("Required read-only Cosmos source paths are inaccessible: " + ", ".join(missing))
    summary, old_protocol = read_json(summary_path), read_json(old_protocol_path)
    asset_paths = read_json(required_assets["asset_paths"])
    language_requirements = read_json(required_assets["language_requirements"])
    if not isinstance(asset_paths, dict) or not asset_paths:
        raise RuntimeError("Frozen Cosmos asset_paths.json is empty or invalid")
    if not isinstance(language_requirements.get("source_cache"), str):
        raise RuntimeError("Frozen Cosmos language_requirements.json lacks its embedding cache metadata")
    validate_protocols(protocol, old_protocol, summary, reuse)
    spin = read_json(required_assets["spinquant_summary"])
    if (spin.get("protocol") != old_protocol.get("protocol") or spin.get("status") != "complete"
            or spin.get("arm") != "spinquant_w4a4" or spin.get("completed_updates") != 200
            or spin.get("expected_updates") != 200):
        raise RuntimeError("Reused SpinQuant C8 calibration receipt is incomplete or differently bound")
    for name, expected in (("evaluation_job", "PIPELINE_COMPLETE"), ("spinquant_job", "PIPELINE_COMPLETE")):
        job_id = reuse[name]
        receipt_root = legacy_root / "artifacts" / job_id
        if not (receipt_root / expected).is_file() or (receipt_root / "exit_code.txt").read_text().strip() != "0":
            raise RuntimeError(f"Archived terminal receipt missing for reused source job {job_id}")

    imported = {"bf16": [], "spinquant_w4a4": []}
    paired_records: dict[int, dict] = {}
    source_protocol = old_protocol["protocol"]
    for arm in imported:
        if summary["arm_episodes"].get(arm) != int(protocol["cohort"]["episodes_per_arm"]):
            raise RuntimeError(f"Frozen summary count differs for {arm}")
        for index in range(int(protocol["cohort"]["episodes_per_arm"])):
            source = episode_root / arm / f"{index:03d}.json"
            if not source.is_file():
                raise RuntimeError(f"Frozen source slot is absent: {source}")
            record = read_json(source)
            missing_fields = REQUIRED_EPISODE_FIELDS - record.keys()
            if missing_fields:
                raise RuntimeError(f"Frozen slot {arm}/{index} lacks fields: {sorted(missing_fields)}")
            if (record["protocol"] != source_protocol or record["arm"] != arm or record["index"] != index
                    or record["complete"] is not True or record["closed_loop_complete"] is not True
                    or not isinstance(record["success"], bool) or record["steps"] <= 0
                    or record["modelqueries"] != len(record["query_metrics"])):
                raise RuntimeError(f"Frozen slot binding or completeness is invalid: {arm}/{index}")
            if arm == "bf16":
                paired_records[index] = record
            elif any(record.get(field) != paired_records[index].get(field) for field in PAIRED_FIELDS):
                raise RuntimeError(f"Paired frozen cohort binding differs at slot {index}")
            destination = root / "results" / "reused" / "episodes" / arm / f"{index:03d}.json"
            copy_episode(source, destination, record)
            imported[arm].append(index)
    return {
        "source_protocol": source_protocol, "source_root": str(legacy_root),
        "source_summary": str(summary_path), "source_jobs": {
            "evaluation": reuse["evaluation_job"], "spinquant_calibration": reuse["spinquant_job"]},
        "source_calibration": reuse.get("spinquant_calibration"),
        "asset_paths_metadata": str(required_assets["asset_paths"]),
        "language_requirements_metadata": str(required_assets["language_requirements"]),
        "huggingface_cache": str(required_assets["huggingface_cache"]),
        "source_summary_episodes": summary["arm_episodes"],
        "imported_episodes": {arm: len(indices) for arm, indices in imported.items()},
        "imported_indices": {arm: [indices[0], indices[-1]] for arm, indices in imported.items()},
        "destination": str(root / "results" / "reused" / "episodes"),
        "cohort_protocol": protocol["cohort"]["manifest_protocol"],
        "inference_config": protocol["inference"],
        "training_data_mismatch_for_spinquant": reuse.get("allow_training_data_mismatch_for_reused_spinquant", False),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--soft-deadline-seconds", type=int, default=13_500)
    args = parser.parse_args()
    if args.soft_deadline_seconds <= 0:
        raise ValueError("A positive soft deadline is required")
    require_compute_allocation()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise RuntimeError("CPU preparation unexpectedly received a GPU allocation")
    protocol = read_json(args.protocol)
    root = args.root or Path(protocol["remote_root"])
    out = args.out or root / "results" / "preparation"
    out.mkdir(parents=True, exist_ok=True)
    receipt = {
        "protocol": protocol["protocol"], "status": "complete",
        "pbs_job_id": os.environ["PBS_JOBID"], "host": socket.gethostname(),
        "created_utc": datetime.now(timezone.utc).isoformat(), "model_loads": 0,
        "gpu_forwards": 0, "cpu_only": True,
        "soft_deadline_seconds": args.soft_deadline_seconds,
        "reused": import_reused(protocol, root),
    }
    atomic_json(out / "summary.json", receipt)
    atomic_json(root / "results" / "reused" / "summary.json", receipt["reused"])
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
