"""Four persistent GPU workers for the frozen, paired Cosmos Policy cohort."""
from __future__ import annotations

import argparse
import collections
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import socket
import tempfile
import time
import uuid

PROTOCOL = "cosmos-policy-c8-w4a4-v1"
COHORT_PROTOCOL = "fastwam-rotation-baselines-v1"
ARMS = ("bf16", "spinquant_w4a4", "action_latent_micro_vq_a4")
SUITES = {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
DIMENSIONS = {"camera_viewpoints", "objects_layout", "robot_initial_states",
              "language_instructions", "light_conditions", "background_textures", "sensor_noise"}
DEFAULT_SOFT_DEADLINE = 23 * 3600 + 45 * 60


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temp = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_episode(path, record):
    """Publish one complete slot without replacing an existing result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    with temp.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def validate_protocol(protocol):
    expected_manifest = "/scratch/users/ntu/yguo017/fastwam-rotation-baselines-20261008/results/manifest.json"
    expected_calibration = {
        "cohort": "C8", "source_root": "/scratch/users/ntu/yguo017/fastwam-vq-calibration-suites-20261013",
        "case_ids": [2000, 2019, 2021, 2025, 2051, 2057, 2066, 2070],
        "contexts_per_suite": 2, "seed_index": 0, "read_only_shared_inputs": True,
        "spinquant_and_vq_separate_jobs": True, "closed_loop_test_used_for_fitting": False,
    }
    expected_cohort = {
        "manifest_protocol": COHORT_PROTOCOL, "episodes_per_arm": 280, "total_episodes": 840,
        "suites": 4, "perturbation_dimensions": 7, "variants_per_cell": 10,
        "state_id": 0, "environment_seed_base": 800_000_000,
        "sampler_seed_base": 900_000_000, "seed_stride": 1000,
        "paired_across_arms": True,
    }
    expected_inference = {
        "config": "cosmos_predict2_2b_480p_libero__inference_only", "chunk_size": 16,
        "execute_per_replan": 16, "denoising_steps": 5, "sigma_min": 4, "sigma_max": 80,
        "parallel_action_future_value": True, "planning": False, "image_size": 224,
        "use_wrist_image": True, "use_proprio": True, "normalize_proprio": True,
        "unnormalize_actions": True, "trained_with_image_aug": True,
        "use_jpeg_compression": True, "flip_images": True, "deterministic": True,
        "use_variance_scale": False, "settling_steps": 30,
        "max_steps_normal": 400, "max_steps_long": 700,
    }
    expected_load_balancing = {
        "persistent_workers": 4, "index_bundle_size": 10, "bundles": 28,
    }
    if (protocol.get("protocol") != PROTOCOL or protocol.get("model") != "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
            or protocol.get("mode") != "libero_direct_policy"
            or protocol.get("remote_root") != "/scratch/users/ntu/yguo017/cosmos-policy-c8-w4a4-20261015"
            or protocol.get("arms") != list(ARMS)
            or protocol.get("cohort", {}).get("manifest") != expected_manifest
            or any(protocol.get("cohort", {}).get(k) != v for k, v in expected_cohort.items())
            or any(protocol.get("calibration", {}).get(k) != v for k, v in expected_calibration.items())
            or any(protocol.get("inference", {}).get(k) != v for k, v in expected_inference.items())
            or any(protocol.get("load_balancing", {}).get(k) != v for k, v in expected_load_balancing.items())):
        raise ValueError("Protocol differs from the frozen Cosmos Policy evaluation contract")
    if protocol.get("execution_boundary") != (
            "Numerical low-bit policy evaluation; report decoded-BF16/fake-A4 separately from any future native kernel evidence."):
        raise ValueError("Protocol execution boundary changed")
    return protocol


def load_manifest(path, protocol):
    doc = read_json(path)
    if doc.get("protocol") != protocol["cohort"]["manifest_protocol"]:
        raise ValueError("Unexpected frozen cohort manifest protocol")
    rows = doc.get("variants")
    if not isinstance(rows, list) or len(rows) != 280:
        raise ValueError("Expected exactly 280 frozen variants")
    ids, cells = set(), collections.Counter()
    for index, row in enumerate(rows):
        if (not isinstance(row, dict) or type(row.get("index")) is not int or row["index"] != index
                or not row.get("variant_id") or row["variant_id"] in ids
                or type(row.get("task_id")) is not int or row.get("state_id", 0) != 0):
            raise ValueError(f"Invalid frozen cohort row at index {index}")
        ids.add(row["variant_id"])
        cells[row.get("suite"), row.get("dimension")] += 1
    if (len(cells) != 28 or set(cells.values()) != {10}
            or {suite for suite, _ in cells} != SUITES
            or {dim for _, dim in cells} != DIMENSIONS):
        raise ValueError("Expected 4 suites x 7 dimensions x 10 variants")
    return doc, rows


def environment_seed(index, protocol):
    cohort = protocol["cohort"]
    return cohort["environment_seed_base"] + index * cohort["seed_stride"]


def sampler_seed(index, protocol, replan=0):
    cohort = protocol["cohort"]
    return cohort["sampler_seed_base"] + index * cohort["seed_stride"] + replan


def validate_quantization_evidence(evidence, arm):
    """Check the explicit numerical-execution boundary and frozen target scope."""
    if arm not in ARMS or not isinstance(evidence, dict) or evidence.get("arm") != arm:
        raise ValueError(f"Missing quantization evidence for {arm}")
    common = {
        "execution_scope": "decoded_bf16_numerical_comparison",
        "native_int4_kernel_used": False,
    }
    if any(evidence.get(key) != value for key, value in common.items()):
        raise ValueError(f"Invalid execution boundary in quantization evidence for {arm}")
    for key in ("target_module_count", "target_weight_numel"):
        if type(evidence.get(key)) is not int or evidence[key] <= 0:
            raise ValueError(f"Invalid {key} in quantization evidence for {arm}")
    if arm == "bf16":
        expected = {
            "execution": "bf16_reference", "weight_format": "bf16", "activation_format": "bf16",
            "same_module_effective_bpw": None, "whole_dit_effective_bpw": None,
            "artifact_dir": None, "artifact_manifest": None,
        }
        if any(evidence.get(key) != value for key, value in expected.items()):
            raise ValueError("BF16 evidence must describe the unmodified BF16 reference")
        return evidence

    expected = {
        "execution": "decoded_bf16_fake_a4",
        "activation_format": "dynamic_symmetric_int4_per_token_qmax7_ste",
    }
    formats = {
        "spinquant_w4a4": ("packed_int4_group128_bf16_scales", "learned_shared_cayley_r128"),
        "action_latent_micro_vq_a4": ("m2_k256_d4_bf16_books_b32_uint8_exponents",
                                       "smooth05_batched_h128_v2"),
    }
    weight_format, transform = formats[arm]
    expected.update({"weight_format": weight_format, "transform": transform})
    if any(evidence.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Unexpected quantized representation metadata for {arm}")
    for key in ("same_module_effective_bpw", "whole_dit_effective_bpw", "bpw_match_relative_delta"):
        value = evidence.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"Invalid {key} in quantization evidence for {arm}")
    if evidence["same_module_effective_bpw"] <= 0 or evidence["whole_dit_effective_bpw"] <= 0:
        raise ValueError(f"Effective BPW must be positive for {arm}")
    if evidence["bpw_match_relative_delta"] > 0.05:
        raise ValueError(f"Effective BPW accounting exceeds the frozen 5% tolerance for {arm}")
    if not all(isinstance(evidence.get(key), str) and evidence[key]
               for key in ("artifact_dir", "artifact_manifest")):
        raise ValueError(f"Missing calibrated deployment artifact references for {arm}")
    return evidence


def build_bundles(rows, protocol, manifest_seed=20261008):
    """Group paired tasks by frozen suite/dimension cell; long-suite cells go first."""
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row["suite"], row["dimension"])].append(row["index"])
    inference = protocol["inference"]
    rng = random.Random(manifest_seed)
    tie = {key: rng.random() for key in groups}
    bundles = []
    for (suite, dimension), indices in groups.items():
        indices = sorted(indices)
        if len(indices) != 10:
            raise ValueError(f"Bundle {suite}/{dimension} has {len(indices)} rather than 10 variants")
        step_cap = inference["max_steps_long"] if suite == "libero_10" else inference["max_steps_normal"]
        bundles.append({"bundle_id": f"{suite}::{dimension}", "suite": suite,
                        "dimension": dimension, "indices": indices,
                        "estimated_steps": step_cap * len(indices) * len(ARMS), "tie": tie[suite, dimension]})
    return sorted(bundles, key=lambda bundle: (-bundle["estimated_steps"], bundle["tie"], bundle["bundle_id"]))


def episode_path(out, arm, index):
    return Path(out) / "episodes" / arm / f"{index:03d}.json"


def validate_episode(record, row, protocol, arm=None, worker_id=None):
    if not isinstance(record, dict):
        raise ValueError("Invalid episode record")
    arm = arm or record.get("arm")
    if arm not in ARMS:
        raise ValueError("Invalid arm/episode record")
    index = record.get("index")
    if type(index) is not int or index != row["index"]:
        raise ValueError("Episode index differs from frozen cohort")
    expected = {
        "protocol": PROTOCOL, "arm": arm, "complete": True,
        "variant_id": str(row["variant_id"]), "suite": row["suite"],
        "dimension": row["dimension"], "task_id": int(row["task_id"]),
        "state_id": 0, "env_seed": environment_seed(index, protocol),
        "sampler_seed": sampler_seed(index, protocol),
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Episode identity/seed mismatch for {arm}/{index}")
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
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("Missing query_metrics")
    if type(record.get("replans")) is not int or record["replans"] != len(metrics):
        raise ValueError("Episode replans/query_metrics mismatch")
    for replan, metric in enumerate(metrics):
        if not isinstance(metric, dict) or metric.get("seed") != sampler_seed(index, protocol, replan):
            raise ValueError(f"Invalid query seed at {arm}/{index}/{replan}")
        for name in ("query_seconds", "peak_allocated_gb", "peak_reserved_gb"):
            value = metric.get(name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"Invalid query metric {name}")
        denoising = metric.get("denoising_seconds")
        if denoising is not None and (isinstance(denoising, bool) or not isinstance(denoising, (int, float))
                                      or not math.isfinite(denoising) or denoising < 0):
            raise ValueError("Invalid denoising_seconds")
        if type(metric.get("first_query")) is not bool:
            raise ValueError("Query metric must mark the first query in its episode")
        query_index = metric.get("query_index_in_worker_arm")
        if (type(query_index) is not int or query_index < 0
                or type(metric.get("first_query_in_worker_arm")) is not bool
                or metric["first_query_in_worker_arm"] != (query_index == 0)
                or type(metric.get("first_query_in_run")) is not bool
                or metric.get("pbs_job_id") != record["pbs_job_id"]
                or not isinstance(metric.get("worker_process_id"), str)
                or not metric["worker_process_id"]):
            raise ValueError("Invalid first/warm query sequence marker")
    validate_quantization_evidence(record.get("quantization_evidence"), arm)
    return index


def load_completed(out, rows, protocol):
    found = {}
    query_indices, run_markers = collections.defaultdict(list), collections.defaultdict(list)
    root = Path(out) / "episodes"
    if not root.exists():
        return found
    for path in sorted(root.glob("*/*.json")):
        record = read_json(path)
        if not isinstance(record, dict):
            raise ValueError(f"Malformed episode record in {path}")
        arm, index = record.get("arm"), record.get("index")
        if arm not in ARMS or type(index) is not int or not 0 <= index < len(rows):
            raise ValueError(f"Malformed episode slot in {path}")
        slot = arm, index
        if slot in found:
            raise ValueError(f"Duplicate episode slot: {slot}")
        if path != episode_path(out, arm, index):
            raise ValueError(f"Episode filename does not match its slot: {path}")
        validate_episode(record, rows[index], protocol)
        found[slot] = record
        worker = record.get("worker_id")
        if type(worker) is not int or worker not in range(4):
            raise ValueError("Episode worker_id is outside the four-worker allocation")
        for metric in record["query_metrics"]:
            key = worker, arm
            query_indices[key].append(metric["query_index_in_worker_arm"])
            run_markers[(worker, arm, metric["worker_process_id"])].append(
                (metric["query_index_in_worker_arm"], metric["first_query_in_run"]))
    for key, indices in query_indices.items():
        if sorted(indices) != list(range(len(indices))):
            raise ValueError(f"Query indices are duplicated or discontinuous for worker/arm {key}")
    for key, markers in run_markers.items():
        first = min(index for index, _ in markers)
        if sum(marked for _, marked in markers) != 1 or not any(index == first and marked for index, marked in markers):
            raise ValueError(f"First formal query marker is invalid for worker/arm/run {key}")
    return found


def pending_slots(bundle, completed):
    return [(arm, index) for arm in ARMS for index in bundle["indices"] if (arm, index) not in completed]


def initial_queue(bundles, run_id):
    return {"protocol": PROTOCOL, "run_id": run_id,
            "bundle_ids": [bundle["bundle_id"] for bundle in bundles], "claims": {}}


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, ValueError, TypeError):
        return True
    return True


def claim_from_state(state, bundles, worker_id, run_id, host, pid, completed_bundle_ids=()):
    """Pure queue transition, shared by the atomic file queue and CPU tests."""
    if state.get("protocol") != PROTOCOL or state.get("bundle_ids") != [b["bundle_id"] for b in bundles]:
        raise ValueError("Shared claim queue does not match this frozen cohort")
    if state.get("run_id") != run_id:
        state["run_id"], state["claims"] = run_id, {}
    claims = state.setdefault("claims", {})
    for key, claim in list(claims.items()):
        if claim.get("run_id") != run_id:
            del claims[key]
        elif claim.get("host") == host and not _pid_alive(claim.get("pid")):
            del claims[key]
    done = set(completed_bundle_ids)
    active = set(claims)
    remaining = [b for b in bundles if b["bundle_id"] not in done]
    available = [b for b in remaining if b["bundle_id"] not in active]
    if not remaining:
        return None, True
    if not available:
        return None, False
    bundle = available[0]
    claims[bundle["bundle_id"]] = {"worker_id": worker_id, "run_id": run_id, "host": host, "pid": pid}
    return bundle, False


def release_claim(out, bundle_id, worker_id, run_id):
    queue_path = Path(out) / "claim_queue.json"
    with _queue_lock(queue_path):
        state = read_json(queue_path)
        claim = state.get("claims", {}).get(bundle_id)
        if claim and claim.get("worker_id") == worker_id and claim.get("run_id") == run_id:
            del state["claims"][bundle_id]
            save_json(queue_path, state)


class _queue_lock:
    def __init__(self, queue_path, timeout=120):
        self.path, self.timeout = Path(str(queue_path) + ".lock"), timeout
        self.fd = None

    def __enter__(self):
        deadline = time.monotonic() + self.timeout
        host = socket.gethostname()
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                with os.fdopen(self.fd, "w", encoding="utf-8") as stream:
                    self.fd = None
                    json.dump({"host": host, "pid": os.getpid(), "time": time.time()}, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                return self
            except FileExistsError:
                try:
                    stale = time.time() - self.path.stat().st_mtime > self.timeout
                    owner = read_json(self.path)
                    if owner.get("host") == host and not _pid_alive(owner.get("pid")):
                        self.path.unlink(missing_ok=True)
                        continue
                    if stale and owner.get("host") != host:
                        self.path.unlink(missing_ok=True)
                        continue
                except (OSError, ValueError, json.JSONDecodeError):
                    try:
                        stale = time.time() - self.path.stat().st_mtime > self.timeout
                    except FileNotFoundError:
                        continue
                    if stale:
                        self.path.unlink(missing_ok=True)
                        continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for queue lock {self.path}")
                time.sleep(0.05)

    def __exit__(self, *_):
        self.path.unlink(missing_ok=True)


def _completed_bundle_ids(out, bundles):
    return {bundle["bundle_id"] for bundle in bundles
            if all(episode_path(out, arm, index).is_file()
                   for arm in ARMS for index in bundle["indices"])}


def claim_next(out, bundles, worker_id, run_id):
    queue_path = Path(out) / "claim_queue.json"
    with _queue_lock(queue_path):
        if queue_path.exists():
            state = read_json(queue_path)
        else:
            state = initial_queue(bundles, run_id)
        bundle, done = claim_from_state(state, bundles, worker_id, run_id,
                                        socket.gethostname(), os.getpid(), _completed_bundle_ids(out, bundles))
        save_json(queue_path, state)
    return bundle, done


def arm_order(worker_id):
    start = worker_id % len(ARMS)
    direction = 1 if worker_id < 2 else -1
    return [ARMS[(start + direction * offset) % len(ARMS)] for offset in range(len(ARMS))]


def worker_completed_slots(completed, worker_id):
    return sum(record.get("worker_id") == worker_id for record in completed.values())


def _worker(worker_id, gpu, protocol, manifest_path, root, out, bundles,
            start_monotonic, soft_deadline, deadline_epoch, run_id):
    _guard_allocation()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    from cosmos_runtime import CosmosRuntime

    worker_out = Path(root) / "artifacts" / run_id / f"worker_{worker_id}"
    worker_out.mkdir(parents=True, exist_ok=True)
    out = Path(out)
    _, rows = load_manifest(manifest_path, protocol)
    completed = load_completed(out, rows, protocol)
    query_counts = collections.Counter()
    for record in completed.values():
        if record.get("worker_id") == worker_id:
            query_counts[record["arm"]] += len(record["query_metrics"])
    run_query_counts = collections.Counter()
    runtime = CosmosRuntime(protocol, Path(root), worker_out, arm="bf16")
    runtime.soft_deadline_epoch = deadline_epoch
    worker_process_id = f"{run_id}/worker_{worker_id}/pid_{os.getpid()}"
    stopped = False
    try:
        while True:
            if time.monotonic() - start_monotonic >= soft_deadline:
                stopped = True
                break
            bundle, done = claim_next(out, bundles, worker_id, run_id)
            if done:
                break
            if bundle is None:
                time.sleep(1.0)
                continue
            try:
                for arm in arm_order(worker_id):
                    missing = [index for index in bundle["indices"] if (arm, index) not in completed]
                    if not missing:
                        continue
                    if time.monotonic() - start_monotonic >= soft_deadline:
                        stopped = True
                        break
                    arm_metadata = runtime.set_arm(arm)
                    evidence = getattr(runtime, "quantization_evidence", None)
                    if evidence is None:
                        evidence = arm_metadata
                    validate_quantization_evidence(evidence, arm)
                    for index in missing:
                        if time.monotonic() - start_monotonic >= soft_deadline:
                            stopped = True
                            break
                        row = rows[index]
                        save_json(worker_out / "progress.json", {
                            "protocol": PROTOCOL, "worker_id": worker_id, "run_id": run_id,
                            "status": "running", "bundle_id": bundle["bundle_id"],
                            "arm": arm, "index": index,
                            "completed_slots": worker_completed_slots(completed, worker_id),
                            "expected_slots": len(ARMS) * len(rows),
                        })
                        record = runtime.episode(row, index)
                        record = dict(record)
                        if record.get("termination") == "soft_deadline":
                            stopped = True
                            break
                        if record.get("complete") is not True or record.get("closed_loop_complete") is not True:
                            raise RuntimeError(
                                f"Cosmos returned an incomplete closed-loop episode {arm}/{index}: "
                                f"{record.get('termination')} {record.get('error')}"
                            )
                        if "query_metrics" not in record:
                            record["query_metrics"] = record.pop("queries", None)
                        episode_evidence = record.get("quantization_evidence")
                        if episode_evidence is not None and episode_evidence != evidence:
                            raise RuntimeError(f"Episode evidence differs from active {arm} encoding")
                        record["quantization_evidence"] = evidence
                        record.update({
                            "protocol": PROTOCOL, "arm": arm, "worker_id": worker_id,
                            "index": index, "variant_id": str(row["variant_id"]),
                            "suite": row["suite"], "dimension": row["dimension"],
                            "task_id": int(row["task_id"]), "state_id": 0,
                            "env_seed": environment_seed(index, protocol),
                            "sampler_seed": sampler_seed(index, protocol), "complete": True,
                            "pbs_job_id": run_id,
                            "execution_scope": "decoded_bf16_numerical_comparison; native low-bit kernel not measured",
                        })
                        metrics = record.get("query_metrics")
                        if isinstance(metrics, list):
                            for replan, metric in enumerate(metrics):
                                metric.update({
                                    "replan": replan,
                                    "query_index_in_worker_arm": query_counts[arm],
                                    "first_query_in_worker_arm": query_counts[arm] == 0,
                                    "first_query_in_run": run_query_counts[arm] == 0,
                                    "pbs_job_id": run_id,
                                    "worker_process_id": worker_process_id,
                                })
                                query_counts[arm] += 1
                                run_query_counts[arm] += 1
                        validate_episode(record, row, protocol, arm, worker_id)
                        atomic_episode(episode_path(out, arm, index), record)
                        completed[(arm, index)] = record
                        save_json(worker_out / "progress.json", {
                            "protocol": PROTOCOL, "worker_id": worker_id, "run_id": run_id,
                            "status": "running", "bundle_id": bundle["bundle_id"],
                            "arm": arm, "index": index,
                            "completed_slots": worker_completed_slots(completed, worker_id),
                            "expected_slots": len(ARMS) * len(rows),
                        })
                        print(f"SLOT_COMPLETE worker={worker_id} arm={arm} index={index} "
                              f"success={record['success']}", flush=True)
                    if stopped:
                        break
            finally:
                release_claim(out, bundle["bundle_id"], worker_id, run_id)
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
        runtime.close()


def _guard_allocation():
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("Must run inside the requested PBS compute allocation")
    hostname = socket.gethostname().lower()
    if "login" in hostname:
        raise RuntimeError("Refusing heavy evaluation work on a login node")
    nodefile_value = os.environ.get("PBS_NODEFILE")
    if not nodefile_value:
        raise RuntimeError("PBS_NODEFILE is required to confirm the compute allocation")
    nodefile = Path(nodefile_value)
    if not nodefile.is_file() or nodefile.stat().st_size > 1_000_000:
        raise RuntimeError("PBS_NODEFILE is missing or unexpectedly large")
    allocated_hosts = {line.strip().lower() for line in nodefile.read_text(encoding="utf-8").splitlines()
                       if line.strip()}
    short_hosts = {host.split(".", 1)[0] for host in allocated_hosts}
    if hostname not in allocated_hosts and hostname.split(".", 1)[0] not in short_hosts:
        raise RuntimeError("This host is not listed in PBS_NODEFILE")


def run(args):
    _guard_allocation()
    protocol = validate_protocol(read_json(args.protocol))
    manifest_path = args.manifest or Path(protocol["cohort"]["manifest"])
    manifest, rows = load_manifest(manifest_path, protocol)
    root = args.root or Path(protocol["remote_root"])
    out = args.out or root / "results" / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    if args.workers != 4:
        raise ValueError("The frozen evaluation requires all four allocated GPUs")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    devices = [device.strip() for device in visible.split(",") if device.strip()]
    if len(devices) != 4:
        raise RuntimeError(f"Expected four visible GPUs, found {len(devices)}")
    bundles = build_bundles(rows, protocol, manifest.get("seed", 20261008))
    run_id = os.environ["PBS_JOBID"]
    soft_deadline = args.soft_deadline_seconds
    if soft_deadline is None:
        soft_deadline = protocol.get("soft_deadline_seconds", DEFAULT_SOFT_DEADLINE)
    start_monotonic = time.monotonic()
    deadline_epoch = time.time() + soft_deadline
    context = mp.get_context("spawn")
    processes = [context.Process(target=_worker, args=(worker, devices[worker], protocol,
                 manifest_path, str(root), str(out), bundles, start_monotonic,
                 soft_deadline, deadline_epoch, run_id),
                 name=f"cosmos-gpu-worker-{worker}") for worker in range(4)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    from aggregate import summarize
    summary = summarize(protocol, rows, load_completed(out, rows, protocol))
    summary["worker_exitcodes"] = {str(p.name): p.exitcode for p in processes}
    save_json(out / "summary.json", summary)
    save_json(out / "progress.json", {
        "protocol": PROTOCOL, "status": summary["status"], "complete": summary["complete"],
        "completed_slots": summary["episode_slots"], "expected_slots": len(ARMS) * len(rows),
        "run_id": run_id,
    })
    if any(p.exitcode != 0 for p in processes):
        raise RuntimeError(f"One or more GPU workers failed: {summary['worker_exitcodes']}")
    print(f"COSMOS_EVALUATION status={summary['status']} slots={summary['episode_slots']}/840", flush=True)
    return summary


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=here / "protocol.json")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", "--worker-count", type=int, default=4)
    parser.add_argument("--soft-deadline-seconds", type=float)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
