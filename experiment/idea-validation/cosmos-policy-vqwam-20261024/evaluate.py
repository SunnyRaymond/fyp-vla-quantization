"""Closed-loop evaluation for the four new Cosmos-adapted arms."""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import multiprocessing as mp
import os
import posixpath
from pathlib import Path
import socket
import tempfile
import time
import uuid

PROTOCOL = "cosmos-policy-vqwam-v1"
NEW_ARMS = ("quarot_w4a4", "svdquant_w4a4", "viditq_w4a4", "ours_vqwam")
REUSED_ARMS = ("bf16", "spinquant_w4a4")
ARMS = REUSED_ARMS + NEW_ARMS
EXPECTED_WEIGHT_NUMEL = 1_761_607_680
HERE = Path(__file__).resolve().parent
_LEGACY = None
_EVALUATION_PROFILES = {
    (4, "g3", 4, "evaluation.pbs"),
    (8, "g4", 8, "evaluation_g4.pbs"),
}


def expected_worker_count(protocol):
    resources = protocol.get("resources", {}).get("evaluation", {})
    balancing = protocol.get("load_balancing", {})
    stage = protocol.get("stages", {}).get("evaluation", {})
    profile = (resources.get("gpus"), resources.get("required_execution_queue"),
               balancing.get("persistent_workers"), stage.get("script", "evaluation.pbs"))
    common_bad = (resources.get("walltime") != "24:00:00" or resources.get("queue") != "normal"
                  or stage.get("job_name", "cpvq24eval") != "cpvq24eval")
    if profile == (4, "g3", 4, "evaluation.pbs"):
        node_bad = (resources.get("ncpus") != 64 or resources.get("memory_gb") != 440
                    or resources.get("nodes", 1) != 1 or resources.get("gpus_per_node", 4) != 4
                    or resources.get("ncpus_per_node", 64) != 64
                    or resources.get("memory_gb_per_node", 440) != 440)
    elif profile == (8, "g4", 8, "evaluation_g4.pbs"):
        node_bad = (resources.get("ncpus") != 128 or resources.get("memory_gb") != 880
                    or resources.get("nodes") != 2 or resources.get("gpus_per_node") != 4
                    or resources.get("ncpus_per_node") != 64
                    or resources.get("memory_gb_per_node") != 440)
    else:
        node_bad = True
    if common_bad or node_bad:
        raise ValueError("Evaluation resources must match the exact 4-GPU/g3 or 8-GPU/two-node g4 profile")
    return profile[2]


def visible_cuda_devices(value, expected):
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be provided by the PBS allocation")
    devices = [device.strip() for device in value.split(",")]
    if (len(devices) != expected or any(not device for device in devices)
            or len(set(devices)) != expected):
        raise RuntimeError(f"Expected {expected} unique allocated GPUs, found {len(devices)}")
    return devices


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
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


def _legacy_helpers(protocol):
    """Reuse the frozen C8 evaluator's manifest, queue, seed, and atomic-slot helpers."""
    global _LEGACY
    if _LEGACY is not None:
        return _LEGACY
    candidates = [Path(protocol["asset_root"]) / "evaluate.py",
                  HERE.parent / "cosmos-policy-c8-w4a4" / "evaluate.py"]
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise FileNotFoundError("Frozen legacy Cosmos evaluator is unavailable: "
                                + ", ".join(map(str, candidates)))
    spec = importlib.util.spec_from_file_location("cosmos_c8_evaluation_helpers", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load frozen evaluator helpers from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.LEGACY_PROTOCOL, module.LEGACY_ARMS = module.PROTOCOL, module.ARMS
    module.PROTOCOL, module.ARMS = PROTOCOL, NEW_ARMS
    _LEGACY = module
    return module


def validate_protocol(protocol):
    cohort = protocol.get("cohort", {})
    evaluation = protocol.get("resources", {}).get("evaluation", {})
    balancing = protocol.get("load_balancing", {})
    quantization = protocol.get("quantization", {})
    expected_worker_count(protocol)
    frozen_cohort = {
        "manifest_protocol": "fastwam-rotation-baselines-v1", "episodes_per_arm": 280,
        "total_episodes": 1120, "comparison_episodes": 1680, "new_arms": 4,
        "reused_episodes": 560, "suites": 4, "perturbation_dimensions": 7,
        "variants_per_cell": 10, "state_id": 0,
        "environment_seed_base": 800_000_000, "sampler_seed_base": 900_000_000,
        "seed_stride": 1000, "paired_across_arms": True,
    }
    frozen_inference = {
        "config": "cosmos_predict2_2b_480p_libero__inference_only", "chunk_size": 16,
        "execute_per_replan": 16, "denoising_steps": 5, "sigma_min": 4, "sigma_max": 80,
        "parallel_action_future_value": True, "planning": False, "image_size": 224,
        "use_wrist_image": True, "use_proprio": True, "normalize_proprio": True,
        "unnormalize_actions": True, "trained_with_image_aug": True,
        "use_jpeg_compression": True, "flip_images": True, "deterministic": True,
        "use_variance_scale": False, "settling_steps": 30,
        "max_steps_normal": 400, "max_steps_long": 700,
    }
    if (protocol.get("protocol") != PROTOCOL
            or protocol.get("arms") != list(ARMS)
            or protocol.get("new_arms") != list(NEW_ARMS)
            or protocol.get("reused_arms") != list(REUSED_ARMS)
            or any(cohort.get(key) != value for key, value in frozen_cohort.items())
            or any(protocol.get("inference", {}).get(key) != value
                   for key, value in frozen_inference.items())
            or protocol.get("model") != "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
            or protocol.get("mode") != "libero_direct_policy"
            or protocol.get("execution_boundary") != (
                "Numerical low-bit policy evaluation; report decoded-BF16/fake-A4 separately from any future native kernel evidence.")
            or quantization.get("target_effective_bpw") != 4.29
            or quantization.get("effective_bpw_max_relative_difference") != 0.05
            or balancing.get("index_bundle_size") != 10 or balancing.get("bundles") != 28):
        raise ValueError("Protocol differs from the frozen four-new-arm Cosmos evaluation contract")
    return protocol


def load_manifest(path, protocol):
    return _legacy_helpers(protocol).load_manifest(path, protocol)


def environment_seed(index, protocol):
    return _legacy_helpers(protocol).environment_seed(index, protocol)


def sampler_seed(index, protocol, replan=0):
    return _legacy_helpers(protocol).sampler_seed(index, protocol, replan)


def build_bundles(rows, protocol, manifest_seed=20261008):
    return _legacy_helpers(protocol).build_bundles(rows, protocol, manifest_seed)


def episode_path(out, arm, index):
    return Path(out) / "episodes" / arm / f"{index:03d}.json"


def reused_episode_path(root, arm, index):
    return Path(root) / "results" / "reused" / "episodes" / arm / f"{index:03d}.json"


def normalize_arm_evidence(metadata, arm, root, protocol):
    """Normalize evidence without imposing the legacy SpinQuant/VQ storage formats."""
    if not isinstance(metadata, dict) or metadata.get("arm") != arm:
        raise ValueError(f"Installer returned no matching evidence for {arm}")
    evidence = dict(metadata)
    if arm in NEW_ARMS[:3]:
        evidence["target_module_count"] = metadata.get("target_linears")
        evidence["effective_bpw"] = metadata.get("encoded_target_bpw", metadata.get("effective_bpw"))
        evidence["artifact_dir"] = posixpath.dirname(metadata["artifact_manifest"]) \
            if metadata.get("artifact_manifest") else posixpath.join(
                str(root).replace("\\", "/"), "results", arm.removesuffix("_w4a4"))
        evidence["execution_scope"] = "method_specific_eager_decoded_bf16_fake_a4"
        evidence["native_int4_kernel_used"] = metadata.get("native_int4_kernel_used", False)
    elif arm == "ours_vqwam":
        storage = metadata.get("export_storage")
        if not isinstance(storage, dict):
            raise ValueError("Ours installer did not expose the deployment storage receipt")
        evidence["target_module_count"] = metadata.get("target_module_count")
        evidence["effective_bpw"] = storage.get("target_effective_bpw")
        evidence["artifact_manifest"] = posixpath.join(metadata["artifact_dir"], "manifest.json") \
            if metadata.get("artifact_dir") else None
        evidence["execution"] = "packed_gpu_resident_decoded_bf16_gemm"
    else:
        raise ValueError(f"Unexpected evaluated arm {arm}")
    evidence["execution_scope"] = evidence.get("execution_scope") or evidence.get("execution")
    evidence["target_weight_numel"] = metadata.get("target_weight_numel")
    evidence["artifact_manifest"] = metadata.get("artifact_manifest", evidence.get("artifact_manifest"))
    if arm in NEW_ARMS[:3] and evidence.get("artifact_manifest") is None:
        evidence["artifact_manifest"] = str(Path(evidence["artifact_dir"]) / "manifest.json")
    quant = protocol.get("quantization", {})
    target_bpw = float(quant.get("target_effective_bpw", 4.29))
    tolerance = float(quant.get("effective_bpw_max_relative_difference", 0.05))
    bpw = evidence.get("effective_bpw")
    if (type(evidence.get("target_module_count")) is not int
            or evidence["target_module_count"] != 280
            or type(evidence.get("target_weight_numel")) is not int
            or evidence["target_weight_numel"] != EXPECTED_WEIGHT_NUMEL
            or isinstance(bpw, bool) or not isinstance(bpw, (int, float))
            or not math.isfinite(bpw) or bpw <= 0
            or abs(float(bpw) - target_bpw) / target_bpw > tolerance
            or evidence.get("native_int4_kernel_used") is not False
            or not isinstance(evidence.get("execution_scope"), str)
            or not evidence["execution_scope"]
            or not isinstance(evidence.get("artifact_dir"), str)
            or not evidence["artifact_dir"]
            or not isinstance(evidence.get("artifact_manifest"), str)
            or not evidence["artifact_manifest"]):
        raise ValueError(f"Invalid target scope, budget, artifact, or decoded execution evidence for {arm}")
    evidence["bpw_match_relative_delta"] = abs(float(bpw) - target_bpw) / target_bpw
    if arm == "ours_vqwam" and evidence.get("packed_gpu_resident") is not True:
        raise ValueError("Ours arm must report its packed GPU resident deployment boundary")
    if arm == "ours_vqwam" and not isinstance(evidence.get("cuda_graph"), dict):
        raise ValueError("Ours arm must retain its CUDA graph evidence fields")
    if arm == 'ours_vqwam' and evidence.get('parity_status') != 'validated_full80':
        raise ValueError('Ours arm requires full80 same-artifact engine parity before closed-loop evaluation')
    return evidence


def validate_quantization_evidence(evidence, arm, protocol):
    if not isinstance(evidence, dict) or evidence.get("arm") != arm:
        raise ValueError(f"Missing quantization evidence for {arm}")
    if arm not in NEW_ARMS:
        raise ValueError(f"New worker cannot evaluate reused arm {arm}")
    bpw = evidence.get("effective_bpw")
    target_bpw = float(protocol.get("quantization", {}).get("target_effective_bpw", 4.29))
    tolerance = float(protocol.get("quantization", {}).get("effective_bpw_max_relative_difference", 0.05))
    if (evidence.get("target_module_count") != 280
            or evidence.get("target_weight_numel") != EXPECTED_WEIGHT_NUMEL
            or evidence.get("native_int4_kernel_used") is not False
            or isinstance(bpw, bool) or not isinstance(bpw, (int, float))
            or not math.isfinite(bpw) or abs(float(bpw) - target_bpw) / target_bpw > tolerance):
        raise ValueError(f"Arm metadata violates the frozen module/BPW/execution contract: {arm}")
    if arm == 'ours_vqwam' and evidence.get('parity_status') != 'validated_full80':
        raise ValueError('Ours closed-loop evidence lacks full80 engine parity')
    return evidence


def validate_episode(record, row, protocol, arm=None, worker_id=None):
    if not isinstance(record, dict):
        raise ValueError("Invalid episode record")
    arm = arm or record.get("arm")
    index = record.get("index")
    if arm not in NEW_ARMS or type(index) is not int or index != row["index"]:
        raise ValueError("Episode arm/index differs from the frozen new-arm cohort")
    expected = {"protocol": PROTOCOL, "arm": arm, "complete": True,
                "variant_id": str(row["variant_id"]), "suite": row["suite"],
                "dimension": row["dimension"], "task_id": int(row["task_id"]),
                "state_id": 0, "env_seed": environment_seed(index, protocol),
                "sampler_seed": sampler_seed(index, protocol)}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Episode identity/seed mismatch for {arm}/{index}")
    if record.get("closed_loop_complete") is not True:
        raise ValueError("Episode is not a complete closed-loop rollout")
    if worker_id is not None and record.get("worker_id") != worker_id:
        raise ValueError("Episode worker_id mismatch")
    if type(record.get("success")) is not bool:
        raise ValueError("Episode success must be boolean")
    if not isinstance(record.get("pbs_job_id"), str) or not record["pbs_job_id"]:
        raise ValueError("Episode is missing its PBS job identity")
    seconds = record.get("episode_seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Invalid episode wall time")
    metrics = record.get("query_metrics")
    if not isinstance(metrics, list) or not metrics or type(record.get("replans")) is not int \
            or record["replans"] != len(metrics):
        raise ValueError("Missing/mismatched query_metrics")
    for replan, metric in enumerate(metrics):
        if not isinstance(metric, dict) or metric.get("seed") != sampler_seed(index, protocol, replan):
            raise ValueError(f"Invalid query seed at {arm}/{index}/{replan}")
        for field in ("query_seconds", "peak_allocated_gb", "peak_reserved_gb"):
            value = metric.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid query metric {field}")
        denoising = metric.get("denoising_seconds")
        if denoising is not None and (isinstance(denoising, bool) or not isinstance(denoising, (int, float))
                                      or not math.isfinite(denoising) or denoising < 0):
            raise ValueError("Invalid denoising_seconds")
        query_index = metric.get("query_index_in_worker_arm")
        if (type(metric.get("first_query")) is not bool or type(metric.get("first_query_in_worker_arm")) is not bool
                or type(metric.get("first_query_in_run")) is not bool
                or type(query_index) is not int or query_index < 0
                or metric["first_query_in_worker_arm"] != (query_index == 0)
                or metric.get("pbs_job_id") != record["pbs_job_id"]
                or not isinstance(metric.get("worker_process_id"), str) or not metric["worker_process_id"]):
            raise ValueError("Invalid query sequence/PBS identity fields")
    validate_quantization_evidence(record.get("quantization_evidence"), arm, protocol)
    return index


def _validate_query_sequences(records, expected_workers):
    query_indices, run_markers = collections.defaultdict(list), collections.defaultdict(list)
    for (arm, _), record in records.items():
        worker = record.get("worker_id")
        if type(worker) is not int or worker not in range(expected_workers):
            raise ValueError(f"Episode worker_id is outside the {expected_workers}-worker allocation")
        for metric in record["query_metrics"]:
            query_indices[worker, arm].append(metric["query_index_in_worker_arm"])
            run_markers[worker, arm, metric["worker_process_id"]].append(
                (metric["query_index_in_worker_arm"], metric["first_query_in_run"]))
    for key, indices in query_indices.items():
        if sorted(indices) != list(range(len(indices))):
            raise ValueError(f"Query indices are duplicated or discontinuous for worker/arm {key}")
    for key, markers in run_markers.items():
        first = min(index for index, _ in markers)
        if sum(marked for _, marked in markers) != 1 or not any(
                index == first and marked for index, marked in markers):
            raise ValueError(f"First formal query marker is invalid for worker/arm/run {key}")


def load_completed(out, rows, protocol):
    found = {}
    root = Path(out) / "episodes"
    if root.exists():
        for path in sorted(root.glob("*/*.json")):
            record = read_json(path)
            arm, index = record.get("arm"), record.get("index")
            if arm not in NEW_ARMS or type(index) is not int or not 0 <= index < len(rows):
                raise ValueError(f"Malformed new-arm episode slot in {path}")
            slot = arm, index
            if slot in found or path != episode_path(out, arm, index):
                raise ValueError(f"Duplicate or misnamed new-arm episode slot: {path}")
            validate_episode(record, rows[index], protocol)
            found[slot] = record
    _validate_query_sequences(found, expected_worker_count(protocol))
    return found


def completed_bundle_ids(bundles, completed):
    """Derive complete bundles only from records already validated by load_completed."""
    return {
        bundle["bundle_id"] for bundle in bundles
        if len(bundle["indices"]) == 10 and all(
            (arm, index) in completed for arm in NEW_ARMS for index in bundle["indices"])
    }


def load_reused_records(root, rows, protocol, manifest_path):
    legacy = _legacy_helpers(protocol)
    old_protocol_path = Path(protocol["reuse"]["root"]) / "protocol.json"
    if not old_protocol_path.is_file():
        old_protocol_path = HERE.parent / "cosmos-policy-c8-w4a4" / "protocol.json"
    old_protocol = legacy.read_json(old_protocol_path)
    _, old_rows = legacy.load_manifest(manifest_path, old_protocol)
    if old_rows != rows:
        raise ValueError("Reused and new arms do not share the identical frozen manifest rows")
    found = {}
    current_protocol, current_arms = legacy.PROTOCOL, legacy.ARMS
    legacy.PROTOCOL, legacy.ARMS = legacy.LEGACY_PROTOCOL, legacy.LEGACY_ARMS
    try:
        for arm in REUSED_ARMS:
            for index, row in enumerate(rows):
                path = reused_episode_path(root, arm, index)
                if not path.is_file():
                    continue
                record = read_json(path)
                slot = arm, index
                if slot in found:
                    raise ValueError(f"Duplicate reused episode slot: {slot}")
                legacy.validate_episode(record, row, old_protocol, arm=arm)
                if record.get("protocol") != legacy.LEGACY_PROTOCOL:
                    raise ValueError("Reused record protocol was rewritten")
                found[slot] = record
    finally:
        legacy.PROTOCOL, legacy.ARMS = current_protocol, current_arms
    old_workers = old_protocol.get("load_balancing", {}).get("persistent_workers", 4)
    if type(old_workers) is not int or old_workers < 1:
        raise ValueError("Legacy reused records have an invalid worker-count contract")
    _validate_query_sequences(found, old_workers)
    return found


def atomic_episode(path, record):
    return _legacy_helpers({"asset_root": str(HERE.parent / "cosmos-policy-c8-w4a4")}).atomic_episode(path, record)


_QUEUE_VERSION = 2


def initial_queue(bundles, run_id, completed=()):
    bundle_ids = [bundle["bundle_id"] for bundle in bundles]
    completed = sorted(set(completed))
    if len(bundle_ids) != len(set(bundle_ids)) or not set(completed) <= set(bundle_ids):
        raise ValueError("Completed bundle seed does not match the frozen queue")
    return {"protocol": PROTOCOL, "queue_version": _QUEUE_VERSION, "run_id": run_id,
            "bundle_ids": bundle_ids, "completed_bundles": completed, "claims": {}}


def claim_from_state(state, bundles, worker_id, run_id, host, pid, completed_seed=()):
    """Pure small-state queue transition; it never checks episode files."""
    bundle_ids = [bundle["bundle_id"] for bundle in bundles]
    if (state.get("protocol") != PROTOCOL
            or state.get("bundle_ids") != bundle_ids
            or len(bundle_ids) != len(set(bundle_ids))):
        raise ValueError("Shared claim queue does not match this frozen cohort")
    seed = set(completed_seed)
    if not seed <= set(bundle_ids):
        raise ValueError("Verified completed bundle seed is outside the frozen cohort")
    if state.get("run_id") != run_id or state.get("queue_version") != _QUEUE_VERSION:
        state.clear()
        state.update(initial_queue(bundles, run_id, seed))
    else:
        claims = state.get("claims")
        completed = state.get("completed_bundles")
        if not isinstance(claims, dict) or not isinstance(completed, list):
            raise ValueError("Shared claim queue has invalid claims or completion ledger")
        if not set(completed) <= set(bundle_ids):
            raise ValueError("Shared completion ledger is outside the frozen cohort")
        state["completed_bundles"] = sorted(set(completed) | seed)

    claims = state["claims"]
    completed = set(state["completed_bundles"])
    active = set(claims)
    available = [bundle for bundle in bundles
                 if bundle["bundle_id"] not in completed and bundle["bundle_id"] not in active]
    if not available:
        # Idle workers may exit; the coordinator will detect any unfinished bundle
        # left behind by a worker that later fails or checkpoints.
        return None, True
    bundle = available[0]
    claims[bundle["bundle_id"]] = {"worker_id": worker_id, "run_id": run_id,
                                   "pbs_job_id": run_id, "host": host, "pid": pid}
    return bundle, False


class _queue_lock:
    """Short cross-node O_EXCL lock; never reaps a remote owner by age."""

    def __init__(self, queue_path, timeout=120):
        self.path = Path(str(queue_path) + ".lock")
        self.timeout = timeout
        self.owner = None
        self.inode = None

    def _remove_if_owned(self):
        try:
            stat = self.path.stat()
            if self.inode == (stat.st_dev, stat.st_ino):
                try:
                    owner = read_json(self.path)
                except (OSError, ValueError, json.JSONDecodeError):
                    return
                if self.owner and owner.get("token") == self.owner["token"]:
                    self.path.unlink(missing_ok=True)
        except FileNotFoundError:
            pass

    def __enter__(self):
        pbs_job_id = os.environ.get("PBS_JOBID")
        if not pbs_job_id:
            raise RuntimeError("Queue lock requires the native PBS_JOBID")
        token = uuid.uuid4().hex
        self.owner = {"pbs_job_id": pbs_job_id, "host": socket.gethostname(),
                      "pid": os.getpid(), "token": token}
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                stat = os.fstat(fd)
                self.inode = (stat.st_dev, stat.st_ino)
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(self.owner, stream, separators=(",", ":"))
                    stream.flush()
                    os.fsync(stream.fileno())
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for queue lock {self.path}")
                time.sleep(0.05)
            except Exception:
                try:
                    stat = self.path.stat()
                    if self.inode == (stat.st_dev, stat.st_ino):
                        self.path.unlink(missing_ok=True)
                except FileNotFoundError:
                    pass
                raise

    def __exit__(self, *_):
        self._remove_if_owned()


def claim_next(out, bundles, worker_id, run_id, protocol, completed):
    queue_path = Path(out) / "claim_queue.json"
    seed = completed_bundle_ids(bundles, completed)
    with _queue_lock(queue_path):
        state = read_json(queue_path) if queue_path.exists() else {
            "protocol": PROTOCOL, "bundle_ids": [bundle["bundle_id"] for bundle in bundles]}
        bundle, drained = claim_from_state(state, bundles, worker_id, run_id,
                                           socket.gethostname(), os.getpid(), seed)
        save_json(queue_path, state)
    return bundle, drained


def release_claim(out, bundle, worker_id, run_id, protocol, completed):
    bundle_id = bundle["bundle_id"]
    is_complete = (len(bundle["indices"]) == 10 and all(
        (arm, index) in completed for arm in NEW_ARMS for index in bundle["indices"]))
    queue_path = Path(out) / "claim_queue.json"
    with _queue_lock(queue_path):
        state = read_json(queue_path)
        claim = state.get("claims", {}).get(bundle_id)
        if (claim and claim.get("worker_id") == worker_id and claim.get("run_id") == run_id
                and claim.get("pbs_job_id") == run_id and claim.get("host") == socket.gethostname()
                and claim.get("pid") == os.getpid()):
            if is_complete:
                done = set(state.get("completed_bundles", ()))
                done.add(bundle_id)
                state["completed_bundles"] = sorted(done)
            del state["claims"][bundle_id]
            save_json(queue_path, state)


def arm_order(worker_id, protocol):
    return _legacy_helpers(protocol).arm_order(worker_id % 4)


def worker_completed_slots(completed, worker_id):
    return sum(record.get("worker_id") == worker_id for record in completed.values())


def _worker(worker_id, gpu, protocol, manifest_path, root, out, bundles,
            start_monotonic, soft_deadline, deadline_epoch, run_id):
    _guard_allocation()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    from common import prepare_runtime
    from runtime import install_arm

    worker_out = Path(root) / "artifacts" / run_id / f"worker_{worker_id}"
    worker_out.mkdir(parents=True, exist_ok=True)
    out = Path(out)
    _, rows = load_manifest(manifest_path, protocol)
    completed = load_completed(out, rows, protocol)
    query_counts = collections.Counter()
    run_process_counts = collections.Counter()
    for (_, _), record in completed.items():
        if record.get("worker_id") == worker_id:
            query_counts[record["arm"]] += len(record["query_metrics"])
            if record.get("pbs_job_id") == run_id:
                for metric in record["query_metrics"]:
                    run_process_counts[(record["arm"], metric["worker_process_id"])] += 1
    process_id = f"{run_id}/worker_{worker_id}/pid_{os.getpid()}"
    runtime = prepare_runtime(protocol, Path(root), worker_out)
    runtime.soft_deadline_epoch = deadline_epoch
    stopped = False
    try:
        while True:
            if time.monotonic() - start_monotonic >= soft_deadline:
                stopped = True
                break
            bundle, done = claim_next(out, bundles, worker_id, run_id, protocol, completed)
            if done:
                break
            if bundle is None:
                time.sleep(1.0)
                continue
            try:
                for arm in arm_order(worker_id, protocol):
                    missing = [index for index in bundle["indices"] if (arm, index) not in completed]
                    if not missing:
                        continue
                    if time.monotonic() - start_monotonic >= soft_deadline:
                        stopped = True
                        break
                    restore = getattr(runtime, "restore_ours_vqwam", None)
                    if callable(restore):
                        restore()
                    runtime.controller._restore_current()
                    runtime.controller.arm = "bf16"
                    metadata = install_arm(runtime, protocol, root, arm, optimized=True)
                    evidence = normalize_arm_evidence(metadata, arm, root, protocol)
                    validate_quantization_evidence(evidence, arm, protocol)
                    for index in missing:
                        if time.monotonic() - start_monotonic >= soft_deadline:
                            stopped = True
                            break
                        row = dict(rows[index])
                        row.update({"env_seed": environment_seed(index, protocol),
                                    "sampler_seed": sampler_seed(index, protocol)})
                        save_json(worker_out / "progress.json", {
                            "protocol": PROTOCOL, "worker_id": worker_id, "run_id": run_id,
                            "status": "running", "bundle_id": bundle["bundle_id"],
                            "arm": arm, "index": index,
                            "completed_slots": worker_completed_slots(completed, worker_id),
                            "expected_slots": len(NEW_ARMS) * len(rows),
                        })
                        record = dict(runtime.episode(row, index))
                        if record.get("termination") == "soft_deadline":
                            stopped = True
                            break
                        if record.get("complete") is not True or record.get("closed_loop_complete") is not True:
                            raise RuntimeError(f"Incomplete closed-loop episode {arm}/{index}: "
                                               f"{record.get('termination')} {record.get('error')}")
                        metrics = record.get("queries")
                        if not isinstance(metrics, list):
                            raise RuntimeError("CosmosRuntime episode omitted its query metrics")
                        record["query_metrics"] = metrics
                        record.pop("queries", None)
                        record.update({"protocol": PROTOCOL, "arm": arm, "worker_id": worker_id,
                            "index": index, "variant_id": str(row["variant_id"]),
                            "suite": row["suite"], "dimension": row["dimension"],
                            "task_id": int(row["task_id"]), "state_id": 0,
                            "env_seed": row["env_seed"], "sampler_seed": row["sampler_seed"],
                            "complete": True, "pbs_job_id": run_id,
                            "execution_scope": "closed_loop_CosmosRuntime_episode; new arm installer evidence specifies decoded BF16/fake A4",
                            "quantization_evidence": evidence})
                        for replan, metric in enumerate(record["query_metrics"]):
                            query_id = query_counts[arm]
                            run_key = arm, process_id
                            metric.update({"replan": replan,
                                "query_index_in_worker_arm": query_id,
                                "first_query_in_worker_arm": query_id == 0,
                                "first_query_in_run": run_process_counts[run_key] == 0,
                                "pbs_job_id": run_id, "worker_process_id": process_id})
                            query_counts[arm] += 1
                            run_process_counts[run_key] += 1
                        validate_episode(record, rows[index], protocol, arm, worker_id)
                        atomic_episode(episode_path(out, arm, index), record)
                        completed[(arm, index)] = record
                        save_json(worker_out / "progress.json", {
                            "protocol": PROTOCOL, "worker_id": worker_id, "run_id": run_id,
                            "status": "running", "bundle_id": bundle["bundle_id"],
                            "arm": arm, "index": index,
                            "completed_slots": worker_completed_slots(completed, worker_id),
                            "expected_slots": len(NEW_ARMS) * len(rows),
                        })
                        print(f"SLOT_COMPLETE worker={worker_id} arm={arm} index={index} "
                              f"success={record['success']}", flush=True)
                    if stopped:
                        break
            finally:
                release_claim(out, bundle, worker_id, run_id, protocol, completed)
            if stopped:
                break
        save_json(worker_out / "progress.json", {
            "protocol": PROTOCOL, "worker_id": worker_id, "run_id": run_id,
            "status": "checkpointed" if stopped else "drained",
            "completed_slots": worker_completed_slots(completed, worker_id),
        })
    except Exception as exc:
        save_json(worker_out / "progress.json", {
            "protocol": PROTOCOL, "worker_id": worker_id, "run_id": run_id,
            "status": "failed", "completed_slots": worker_completed_slots(completed, worker_id),
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise
    finally:
        restore = getattr(runtime, "restore_ours_vqwam", None)
        try:
            if callable(restore):
                restore()
        finally:
            runtime.close()


def _guard_allocation():
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("Must run inside the requested PBS compute allocation")
    hostname = socket.gethostname().lower()
    nodefile_value = os.environ.get("PBS_NODEFILE")
    if "login" in hostname or not nodefile_value:
        raise RuntimeError("Evaluation requires a genuine compute-node PBS allocation")
    nodefile = Path(nodefile_value)
    if not nodefile.is_file() or nodefile.stat().st_size > 1_000_000:
        raise RuntimeError("PBS_NODEFILE is missing or unexpectedly large")
    hosts = {line.strip().lower() for line in nodefile.read_text(encoding="utf-8").splitlines()
             if line.strip()}
    if hostname not in hosts and hostname.split(".", 1)[0] not in {host.split(".", 1)[0] for host in hosts}:
        raise RuntimeError("This host is not listed in PBS_NODEFILE")


def run(args):
    _guard_allocation()
    protocol = validate_protocol(read_json(args.protocol))
    worker_count = expected_worker_count(protocol)
    if worker_count != 4:
        raise RuntimeError("8-worker g4 evaluation must use the PBSdsh node coordinator")
    if args.workers is not None and args.workers != worker_count:
        raise ValueError(f"--workers must match the protocol profile ({worker_count})")
    devices = visible_cuda_devices(os.environ.get("CUDA_VISIBLE_DEVICES"), worker_count)
    manifest_path = args.manifest or Path(protocol["cohort"]["manifest"])
    manifest, rows = load_manifest(manifest_path, protocol)
    root = Path(args.root or protocol["remote_root"])
    out = Path(args.out or root / "results" / "evaluation")
    out.mkdir(parents=True, exist_ok=True)
    bundles = build_bundles(rows, protocol, manifest.get("seed", 20261008))
    run_id = os.environ["PBS_JOBID"]
    soft_deadline = float(args.soft_deadline_seconds if args.soft_deadline_seconds is not None
                          else protocol.get("soft_deadline_seconds", 85_500))
    start_monotonic, deadline_epoch = time.monotonic(), time.time() + soft_deadline
    context = mp.get_context("spawn")
    processes = [context.Process(target=_worker, args=(worker, devices[worker], protocol,
                 str(manifest_path), str(root), str(out), bundles, start_monotonic,
                 soft_deadline, deadline_epoch, run_id), name=f"cosmos-gpu-worker-{worker}")
                 for worker in range(worker_count)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    worker_exitcodes = {process.name: process.exitcode for process in processes}
    save_json(out / "worker_exitcodes.json", {"pbs_job_id": run_id, "worker_exitcodes": worker_exitcodes})
    from aggregate import summarize
    new_records = load_completed(out, rows, protocol)
    reused_records = load_reused_records(root, rows, protocol, manifest_path)
    summary = summarize(protocol, rows, new_records, reused_records, worker_exitcodes, run_id)
    save_json(out / "summary.json", summary)
    save_json(out / "progress.json", {"protocol": PROTOCOL, "status": summary["status"],
        "complete": summary["complete"], "completed_slots": summary["new_episode_slots"],
        "expected_slots": len(NEW_ARMS) * len(rows), "pbs_job_id": run_id})
    if any(code != 0 for code in worker_exitcodes.values()):
        raise RuntimeError(f"One or more GPU workers failed: {worker_exitcodes}")
    print(f"COSMOS_EVALUATION status={summary['status']} "
          f"new_slots={summary['new_episode_slots']}/1120 all_methods={summary['all_methods']['episode_slots']}/1680",
          flush=True)
    return summary


def run_node(args):
    _guard_allocation()
    protocol = validate_protocol(read_json(args.protocol))
    if expected_worker_count(protocol) != 8:
        raise ValueError("Node-only evaluation requires the approved 8-worker g4 profile")
    if args.node_rank not in (0, 1) or args.launch_receipt is None:
        raise ValueError("Node-only evaluation requires node rank 0/1 and a launch receipt")
    if args.workers is not None and args.workers != 8:
        raise ValueError("--workers must match the 8-worker protocol profile")
    launch = read_json(args.launch_receipt)
    run_id = os.environ.get("PBS_JOBID")
    launch_nodes = launch.get("nodes")
    if (launch.get("protocol") != PROTOCOL or launch.get("pbs_job_id") != run_id
            or launch.get("expected_workers") != 8 or not isinstance(launch_nodes, list)
            or len(launch_nodes) != 2 or launch_nodes[args.node_rank].get("rank") != args.node_rank):
        raise RuntimeError("Node launch receipt does not match the current PBS evaluation")
    host = socket.gethostname().lower().split(".", 1)[0]
    expected_host = str(launch_nodes[args.node_rank]["hostname"]).lower().split(".", 1)[0]
    if host != expected_host:
        raise RuntimeError("Node rank does not match this PBS host")
    manifest_path = args.manifest or Path(protocol["cohort"]["manifest"])
    manifest, rows = load_manifest(manifest_path, protocol)
    root = Path(args.root or protocol["remote_root"])
    out = Path(args.out or root / "results" / "evaluation")
    out.mkdir(parents=True, exist_ok=True)
    devices = visible_cuda_devices(os.environ.get("CUDA_VISIBLE_DEVICES"), 4)
    bundles = build_bundles(rows, protocol, manifest.get("seed", 20261008))
    deadline_epoch = float(launch["deadline_epoch"])
    remaining = max(0.0, deadline_epoch - time.time())
    start_monotonic = time.monotonic()
    worker_ids = range(args.node_rank * 4, args.node_rank * 4 + 4)
    context = mp.get_context("spawn")
    processes = [context.Process(target=_worker, args=(worker_id, devices[worker_id % 4], protocol,
                 str(manifest_path), str(root), str(out), bundles, start_monotonic,
                 remaining, deadline_epoch, run_id), name=f"cosmos-gpu-worker-{worker_id}")
                 for worker_id in worker_ids]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    worker_exitcodes = {process.name: process.exitcode for process in processes}
    node_rank = args.node_rank
    worker_path = (Path(launch["artifacts_root"]) / "nodes" / f"node_{node_rank}" /
                   "worker_exitcodes.json")
    save_json(worker_path, {"protocol": PROTOCOL, "pbs_job_id": run_id, "node_rank": node_rank,
                            "host": host, "worker_exitcodes": worker_exitcodes})
    if set(worker_exitcodes) != {f"cosmos-gpu-worker-{i}" for i in range(node_rank * 4, node_rank * 4 + 4)} \
            or any(code != 0 for code in worker_exitcodes.values()):
        raise RuntimeError(f"Node {node_rank} workers did not all exit zero: {worker_exitcodes}")
    print(f"COSMOS_EVALUATION_NODE node_rank={node_rank} workers=4 status=complete", flush=True)
    return worker_exitcodes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--node-only", action="store_true")
    parser.add_argument("--node-rank", type=int)
    parser.add_argument("--launch-receipt", type=Path)
    parser.add_argument("--soft-deadline-seconds", type=float)
    args = parser.parse_args(argv)
    if args.node_only:
        if args.node_rank is None or args.launch_receipt is None:
            parser.error("--node-only requires --node-rank and --launch-receipt")
        return run_node(args)
    if args.node_rank is not None or args.launch_receipt is not None:
        parser.error("--node-rank/--launch-receipt require --node-only")
    return run(args)


if __name__ == "__main__":
    main()
