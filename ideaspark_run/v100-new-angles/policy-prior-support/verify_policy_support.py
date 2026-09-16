"""CPU-only verifier for the exact policy-prior-support producer contract.

The producer job directory is an explicit input. This file performs no model
construction or model inference: it verifies receipts, checks the saved W4
payload, and delegates the saved MPPI first-update algebra to root's replay.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import re
import sys
from typing import Any


SOURCE_COMMIT = "e9f59321933cbc8e11a002b842adc7d4ffae8ff1"
CHECKPOINT_SHA = "0e2c0eada8f160dd65c8faaa5e4ca2f8f496104e21433e334d716b73257a52c2"
PARENT_SHA = "9240979105b919f6051f27261d00ff33652105f962d039728638caabb042d369"
HELPER_SHA = "3688d97e4c6e2bed4da4ff7aa552605690c3571e49dcf4c026e955b3c642eb11"
RUNNER_SHA = "0fd129c569663c3321485e9ab2d770b4833ad3a01547fb4b2eda0e850cf1eece"
MANIFEST_SHA = "b5101d2a4e400da540ce6ec1ed6e2552439cf3194fc756e8ac9336b1094fde42"
PROTOCOL_SHA = "47ae0816f0f9f64a27aff56174d1c394e92f58a3d85a4dc4913bc0683553d7d9"
INPUT_FREEZE_SHA = "736d30ba2a7a8d32489c69f5dcbcb29faf5cdcf3c583f7e1428cdf08a56a6774"
REPLAY_SHA = "1a69ee65256f70c8656520ff5a4c7948d303e025aebe4ef7a806b975ab35b9fc"
ARMS = ["FP-policy", "W4-policy", "random-replacement"]
SEEDS = list(range(5217, 5225))
RAW_SHAPES = {
    "actions": (8, 3, 3, 512, 1), "values": (8, 3, 512),
    "elite_indices": (8, 3, 64), "weights": (8, 3, 64),
    "mu": (8, 3, 3, 1), "std": (8, 3, 3, 1),
    "mu_values": (8, 3), "masses": (8, 3), "counts": (8, 3),
}
REQUIRED_GATES = (
    "checkpoint_strict_load", "model_eval", "no_grad", "actor_linear_allowlist_complete",
    "fp_noop_exact", "weight_restore_exact", "non_actor_unchanged",
    "rng_pairing_verified", "mu_proxy_num_samples_1", "common_scores_close",
    "quant_readback_exact", "original_config_unchanged",
)


def _json(path: pathlib.Path) -> Any:
    if not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError(f"missing or oversized JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(path: pathlib.Path) -> None:
    if not path.is_file():
        raise ValueError(f"required producer artifact is missing: {path}")


def _write_summary(path: pathlib.Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f"refusing to overwrite verifier summary: {path}")
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    if len(payload.encode("utf-8")) > 65536:
        raise ValueError("verifier summary exceeds 64 KiB")
    temp = pathlib.Path(str(path)+'.writing')
    temp.write_text(payload, encoding="utf-8")
    temp.replace(path)


def _allocation_checks(engineering: dict[str, Any], checks: dict[str, bool]) -> None:
    allocation = engineering.get("allocation")
    checks["allocation_verified"] = isinstance(allocation, dict) and allocation.get("verified") is True
    if not isinstance(allocation, dict):
        for key in ("allocation_job", "allocation_hostname", "allocation_partition", "allocation_scheduler", "allocation_node", "v100_verified"):
            checks[key] = False
        return
    job = str(allocation.get("job_id", ""))
    host = str(allocation.get("hostname", "")).split(".", 1)[0].lower()
    node = str(allocation.get("nodelist", "")).split("[", 1)[0].lower()
    checks["allocation_job"] = job.isdigit() and int(job) > 0
    checks["allocation_hostname"] = bool(re.fullmatch(r"tc1n[0-9]+", host))
    checks["allocation_partition"] = allocation.get("partition") == "UGGPU-TC1"
    checks["allocation_scheduler"] = str(allocation.get("scheduler", "")).lower() == "slurm"
    checks["allocation_node"] = bool(re.fullmatch(r"tc1n[0-9]+", node)) and node == host
    gpu = engineering.get("gpu")
    checks["v100_verified"] = isinstance(gpu, dict) and gpu.get("verified_v100") is True and 'V100' in gpu.get('name','') and gpu.get('compute_capability') == [7,0]


def _input_checks(job: pathlib.Path, engineering: dict[str, Any], checks: dict[str, bool]) -> None:
    manifest_path = job / "input_manifest.json"
    parent_path = job / "parent_manifest.json"
    _require_file(manifest_path)
    _require_file(parent_path)
    manifest = _json(manifest_path)
    parent = _json(parent_path)
    checks["manifest_sha256"] = _sha(manifest_path) == MANIFEST_SHA == engineering.get("manifest_sha256")
    checks["manifest_schema"] = manifest.get("schema") == "policy-prior-support-preparation-v1"
    checks["manifest_task_seeds"] = manifest.get("task") == "cartpole-balance" and manifest.get("seed_order") == SEEDS
    checks["manifest_observation_shape"] = manifest.get("observations", {}).get("shape") == [8, 5]
    checks["manifest_reset_only"] = (manifest.get("inference_ran") is False and manifest.get("full_rollout") is False and
                                       manifest.get("dmcontrol", {}).get("env_steps") == 0 and
                                       manifest.get("dmcontrol", {}).get("render_calls") == 0)
    checks["parent_sha256"] = _sha(parent_path) == PARENT_SHA == engineering.get("parent_manifest_sha256")
    checks["parent_checkpoint"] = parent.get("checkpoint", {}).get("sha256") == CHECKPOINT_SHA
    checks["parent_source"] = parent.get("source", {}).get("commit") == SOURCE_COMMIT
    checks["input_freeze_receipt"] = engineering.get("input_freeze_sha256") == INPUT_FREEZE_SHA
    checks["protocol_receipt"] = engineering.get("protocol_sha256") == PROTOCOL_SHA
    checks["manifest_protocol"] = manifest.get("protocol", {}).get("sha256") == PROTOCOL_SHA
    checks["input_observations_receipt"] = engineering.get("observations_sha256") == manifest['observations']['sha256']
    parent_files = parent['source']['selected_files']
    source_files = engineering['source']['selected_files']
    checks['source_files_match_frozen_parent'] = set(parent_files)==set(source_files) and all(
        source_files[k]['sha256']==v['sha256'] for k,v in parent_files.items())
    freeze_path = job/'policy_support_input_freeze.json'
    checks['actual_input_freeze_hash'] = _sha(freeze_path)==INPUT_FREEZE_SHA
    freeze = _json(freeze_path)
    checks['input_freeze_fields'] = freeze['manifest_sha256']==MANIFEST_SHA and freeze['cpu_job_id']=='64832' and freeze['seed_order']==SEEDS and freeze['protocol_sha256']==PROTOCOL_SHA


def _source_checks(engineering: dict[str, Any], runtime: dict[str, Any], checks: dict[str, bool]) -> None:
    source = engineering.get("source")
    checks["source_commit"] = isinstance(source, dict) and source.get("commit") == SOURCE_COMMIT
    selected = source.get("selected_files") if isinstance(source, dict) else None
    checks["source_files_nonempty"] = isinstance(selected, dict) and len(selected) >= 10
    checks["source_file_receipts"] = isinstance(selected, dict) and all(
        isinstance(row, dict) and isinstance(row.get("sha256"), str) and len(row["sha256"]) == 64
        for row in selected.values()
    )
    imported = runtime.get("source_identity")
    imported_paths = ("common_init", "common_math", "common_layers", "world_model_module", "planner_module", "common_scale")
    checks["actual_import_source_hashes"] = isinstance(imported, dict) and all(
        isinstance(imported.get(key), dict) and isinstance(imported[key].get("sha256"), str) and
        pathlib.Path(str(imported[key].get("path", ""))).is_file() and
        imported[key]["sha256"] == _sha(pathlib.Path(imported[key]["path"])) for key in imported_paths
    )
    selected_paths = {
        "common_math": "tdmpc2/common/math.py", "common_layers": "tdmpc2/common/layers.py",
        "world_model_module": "tdmpc2/common/world_model.py", "planner_module": "tdmpc2/tdmpc2.py",
    }
    checks["selected_import_source_hashes"] = isinstance(selected, dict) and all(
        isinstance(imported.get(key), dict) and isinstance(selected.get(relative), dict) and
        imported[key].get("sha256") == selected[relative].get("sha256") for key, relative in selected_paths.items()
    )
    config_path = runtime.get("config_path")
    config_record = selected.get("tdmpc2/config.yaml", {}) if isinstance(selected, dict) else {}
    checks["actual_config_source_hash"] = (isinstance(config_path, str) and pathlib.Path(config_path).is_file() and
        config_record.get("sha256") == _sha(pathlib.Path(config_path)))


def _runtime_checks(job: pathlib.Path, engineering: dict[str, Any], checks: dict[str, bool]) -> dict[str, Any]:
    runtime_path = job / str(engineering.get("runtime_file", "runtime.json"))
    _require_file(runtime_path)
    runtime = _json(runtime_path)
    checks["runtime_sha256"] = _sha(runtime_path) == engineering.get("runtime_sha256")
    _source_checks(engineering, runtime, checks)
    packages = runtime.get("packages", {})
    checks["runtime_packages"] = (packages.get("torch") == "2.6.0+cu124" and
                                   packages.get("tensordict") == "0.7.2" and packages.get("omegaconf") == "2.3.0")
    cfg = engineering.get("resolved_config")
    expected = {"horizon": 3, "num_pi_trajs": 24, "num_samples": 512, "num_elites": 64,
                "iterations": 6, "temperature": .5, "max_std": 2, "episodic": False,
                "action_dim": 1, "multitask": False, "min_std": .05}
    for key, value in expected.items():
        checks[f"config_{key}"] = isinstance(cfg, dict) and cfg.get(key) == value
    proxy = engineering.get("mu_proxy")
    checks["mu_proxy_num_samples_1"] = isinstance(proxy, dict) and proxy.get("num_samples") == 1 and proxy.get("model_num_samples") == 512
    checks["mu_proxy_output_1x1"] = isinstance(proxy, dict) and proxy.get("output_shape") == [1, 1]
    checkpoint = engineering.get("checkpoint", {})
    checkpoint_path = pathlib.Path(str(checkpoint.get("path", "")))
    checks["checkpoint_receipt"] = (checkpoint.get("sha256") == CHECKPOINT_SHA and checkpoint.get("size_bytes") == 31344610 and
                                     checkpoint_path.is_file() and _sha(checkpoint_path) == CHECKPOINT_SHA)
    return runtime


def _raw_checks(job: pathlib.Path, engineering: dict[str, Any], checks: dict[str, bool]):
    import numpy as np
    raw_path = job / "raw_policy_support.npz"
    _require_file(raw_path)
    with np.load(raw_path, allow_pickle=False) as packed:
        required = set(RAW_SHAPES) | {"arm_names", "reset_seeds", "observations", "completed", "metadata_json"}
        missing = sorted(required - set(packed.files))
        if missing:
            raise ValueError("raw missing required keys: " + ",".join(missing))
        data = {key: np.asarray(packed[key]) for key in packed.files}
    checks["raw_sha256"] = _sha(raw_path) == engineering.get("raw_sha256")
    for key, shape in RAW_SHAPES.items():
        checks[f"raw_{key}_shape"] = data[key].shape == shape
        checks[f"raw_{key}_finite"] = bool(np.isfinite(data[key]).all())
    checks["raw_arm_names"] = data["arm_names"].tolist() == ARMS
    checks["raw_seeds"] = data["reset_seeds"].shape == (8,) and data["reset_seeds"].tolist() == SEEDS
    checks["raw_completed"] = data["completed"].shape == (8,) and data["completed"].dtype == np.bool_ and bool(data["completed"].all())
    metadata = json.loads(str(data["metadata_json"].item()))
    checks["raw_metadata"] = (metadata.get("schema") == "policy-prior-support-raw-v1" and
                               metadata.get("reset_seeds") == SEEDS and metadata.get("arm_names") == ARMS and
                               metadata.get('manifest_sha256') == MANIFEST_SHA and metadata.get('protocol_sha256') == PROTOCOL_SHA)
    manifest = _json(job/'input_manifest.json')
    obs_path = pathlib.Path(manifest['asset_root'])/manifest['observations']['path']
    checks['actual_observations_hash'] = _sha(obs_path)==manifest['observations']['sha256']
    with np.load(obs_path,allow_pickle=False) as original:
        checks['raw_observations_match_input'] = data['observations'].shape==(8,5) and np.array_equal(data['observations'],original['observations'])
    return data, raw_path


def _torch_array(value: Any):
    import numpy as np
    if value is None:
        raise ValueError("quant snapshot field is missing")
    return value.detach().cpu().contiguous().numpy() if hasattr(value, "detach") else np.asarray(value)


def _quant_checks(job: pathlib.Path, engineering: dict[str, Any], checks: dict[str, bool]) -> None:
    import numpy as np
    snapshot_path = job / "quant_snapshot.pt"
    _require_file(snapshot_path)
    checks["quant_snapshot_sha256"] = _sha(snapshot_path) == engineering.get("quant_snapshot_sha256")
    import torch
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    names = {"_pi.0.weight", "_pi.1.weight", "_pi.2.weight"}
    checks["quant_snapshot_names"] = isinstance(snapshot, dict) and set(snapshot) == names
    checks['quant_actor_allowlist'] = set(engineering['actor_weight_names'])==names
    checks['quant_finite_shapes'] = True
    checks['quant_scales_absmax'] = True
    checks["quant_codes_finite"] = checks["quant_codes_integer"] = checks["quant_codes_grid"] = True
    checks["quant_scales_positive"] = checks["quant_dequant_readback"] = True
    for name in sorted(names):
        row = snapshot.get(name, {}) if isinstance(snapshot, dict) else {}
        q = _torch_array(row.get("codes")); scale = _torch_array(row.get("scale"))
        fp = _torch_array(row.get("fp")); dq = _torch_array(row.get("dequant"))
        read_q = _torch_array(row.get("readback_q")); read_fp = _torch_array(row.get("readback_fp"))
        expected_shape = (2,512) if name=='_pi.2.weight' else (512,512)
        checks['quant_finite_shapes'] &= all(x.shape==expected_shape and np.isfinite(x).all() for x in [fp,q,dq,read_q,read_fp])
        maximum = np.abs(fp).max(axis=1,keepdims=True)
        expected_scale = np.where(maximum>0,maximum*np.float32(1.0/7.0),np.ones_like(maximum))
        checks['quant_scales_absmax'] &= bool(np.allclose(scale,expected_scale,atol=1e-10,rtol=1e-6))
        checks["quant_codes_finite"] &= bool(np.isfinite(q).all())
        checks["quant_codes_integer"] &= bool(np.array_equal(q, np.rint(q)))
        checks["quant_codes_grid"] &= bool((q >= -7).all() and (q <= 7).all())
        checks["quant_scales_positive"] &= bool(np.isfinite(scale).all() and (scale > 0).all() and scale.ndim == 2 and scale.shape == (q.shape[0], 1))
        checks["quant_dequant_readback"] &= bool(np.allclose(q.astype(np.float32) * scale.astype(np.float32), dq.astype(np.float32), atol=2e-6, rtol=2e-6))
        checks["quant_dequant_readback"] &= bool(np.array_equal(dq, read_q) and np.array_equal(fp, read_fp) and fp.shape == dq.shape)


def _guard_checks(job: pathlib.Path, engineering: dict[str, Any], raw: dict, checks: dict[str, bool]) -> None:
    gates = engineering.get("gates")
    checks["gates_present"] = isinstance(gates, dict)
    for key in REQUIRED_GATES:
        checks[key] = isinstance(gates, dict) and gates.get(key) is True
    checks["producer_completed_states"] = engineering.get("completed_states") == 8 and bool(raw["completed"].all())
    checks["non_actor_digest_restore"] = isinstance(engineering.get("non_actor_before_sha256"),str) and len(engineering['non_actor_before_sha256'])==64 and engineering.get("non_actor_before_sha256") == engineering.get("non_actor_after_sha256")
    checks["model_digest_restore"] = isinstance(engineering.get("model_before_sha256"),str) and len(engineering['model_before_sha256'])==64 and engineering.get("model_before_sha256") == engineering.get("model_after_sha256")
    checks["rng_receipt_sha256"] = (job / "rng.json").is_file() and _sha(job / "rng.json") == engineering.get("rng_sha256")
    checks["rng_receipt_states"] = False
    if checks["rng_receipt_sha256"]:
        receipts = _json(job / "rng.json")
        checks["rng_receipt_states"] = isinstance(receipts, list) and len(receipts) == 8 and all(
            isinstance(row, dict) and row.get("reset_seed") == SEEDS[i] and
            row.get("policy_seed") == 7501 + i and row.get("common_seed") == 7601 + i and
            row.get("replacement_seed") == 7801 + i and len(row.get("pool_score", [])) == 3 and
            len(row.get("mu_score", [])) == 3 for i, row in enumerate(receipts)
        )
        def valid_state(value):
            return isinstance(value,dict) and set(value)=={'cpu','cuda'} and all(isinstance(x,str) and re.fullmatch('[0-9a-f]{64}',x) for x in value.values())
        def valid_pair(value):
            return isinstance(value,dict) and set(value)=={'before','after'} and all(valid_state(x) for x in value.values())
        checks['rng_actual_pairing'] = checks['rng_receipt_states'] and all(
            row['pool_score_seed']==7701+i and row['mu_score_seed']==7901+i and
            valid_pair(row['fp_proposal']) and row['fp_proposal']==row['q_proposal']==row['fp_noop'] and
            all(valid_pair(x) and x==row['pool_score'][0] for x in row['pool_score']) and
            all(valid_pair(x) and x==row['mu_score'][0] for x in row['mu_score'])
            for i,row in enumerate(receipts))
    else:
        checks['rng_actual_pairing'] = False


def _load_replay(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("policy_support_replay", str(path))
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import replay library: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.replay


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU-only policy support verifier")
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--replay", type=pathlib.Path, required=True)
    args = parser.parse_args()
    # Direct invocation fails closed before NumPy, torch, hashes, or artifact reads.
    from allocation_guard import require_allocation
    allocation = require_allocation()
    import numpy as np
    job = args.input.resolve()
    output = args.output.resolve()
    if not job.name.isdigit() or job.parent.name!='artifacts' or output.name!=str(allocation['job_id']) or output.parent.name!='artifacts' or job==output:
        raise ValueError('Expected separate numeric producer/current allocation job directories')
    if not (job/'engineering.json').is_file():
        _write_summary(output/'verification.json',{'decision':'implementation_inconclusive','allocation':allocation,'reason':'producer engineering missing; no partial science'})
        return 0
    engineering = _json(job / "engineering.json")
    result: dict[str, Any] = {"schema": "policy-prior-support-verification-v2", "allocation": allocation,
                              "producer_status": engineering.get("status")}
    if engineering.get("status") != "complete":
        status = str(engineering.get("status", "")).lower()
        exit_path = job / "exit_status.json"
        exit_code = None
        if exit_path.is_file():
            receipt = _json(exit_path)
            exit_code = receipt.get("exit_code")
        budget = exit_code in (124, 137) or "budget" in status or "timeout" in status
        result.update({"status": "inconclusive", "decision": "inconclusive_budget" if budget else "implementation_inconclusive",
                       "exit_code": exit_code, "reason": "producer incomplete; no partial science replay"})
        _write_summary(output / "verification.json", result)
        return 0
    for path in (job / "tdq_screen.py", job / "runtime.json", job / "input_manifest.json", job / "parent_manifest.json",
                 job / "rng.json", job / "quant_snapshot.pt", args.replay):
        _require_file(path)
    checks: dict[str, bool] = {"runner_sha256_receipt": engineering.get("runner_sha256") == RUNNER_SHA}
    checks['actual_runner_hash'] = _sha(job/'policy_support_screen.py')==RUNNER_SHA
    checks['actual_helper_hash'] = _sha(job/'tdq_screen.py')==HELPER_SHA==engineering['helper_sha256']
    checks['actual_protocol_hash'] = _sha(job/'policy_support_protocol.zh.md')==PROTOCOL_SHA
    checks['cpu_replay_hash'] = _sha(args.replay)==REPLAY_SHA
    checks['producer_job_identity'] = engineering.get('allocation',{}).get('job_id')==job.name
    checks["protocol_sha256"] = engineering.get("protocol_sha256") == PROTOCOL_SHA
    _allocation_checks(engineering, checks)
    _input_checks(job, engineering, checks)
    _runtime_checks(job, engineering, checks)
    raw, raw_path = _raw_checks(job, engineering, checks)
    _quant_checks(job, engineering, checks)
    _guard_checks(job, engineering, raw, checks)
    replay = None
    if all(checks.values()):
        replay = _load_replay(args.replay)(raw, np)
        decision = replay["decision"]
    else:
        decision = "implementation_inconclusive"
    result.update({"status": "verified", "decision": decision, "checks": checks,
                   "raw_sha256": _sha(raw_path), "science": replay,
                   "verifier_sha256":_sha(pathlib.Path(__file__)),"producer_engineering_sha256":_sha(job/'engineering.json'),
                   "model_inference_recomputed_on_cpu": False,
                   "scope": "Saved FP scorer outputs and MPPI first internal update only; no closed-loop return or deployment claim."})
    _write_summary(output / "verification.json", result)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"implementation_inconclusive: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(4)
