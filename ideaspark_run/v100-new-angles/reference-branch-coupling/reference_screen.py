"""Bounded DINO-WM encoder-branch coupling producer; real SLURM V100 only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

TEACHER_SHA = "6e475ff4c75dda269b6d916ac28f8d63879abe6aae6427da6955bb35d36ed559"
MANIFEST_SHA = "6dae677a0e7763896d735495500aea78a9254e59b67fa47817bb9bb9bb1f5c14"
VALID = (124, 125, 126, 127, 128, 129)
TRAJECTORIES = (1035, 1534, 1158, 203, 1837, 1095)
ARMS = ("FP32", "encoder_W4_RTN", "encoder_W4_SR0", "encoder_W4_SR1", "encoder_W4_SR2")
SEEDS = (2701, 2702, 2703)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _json_default(value):
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
    except ImportError:
        pass
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_npz(path: Path, arrays: dict) -> None:
    import numpy as np
    tmp = Path(str(path) + ".tmp")
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def _tensor_hash(value, torch) -> str:
    a = value.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(json.dumps(list(a.shape), separators=(",", ":")).encode())
    h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _state_digest(model, torch, exclude=frozenset()) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name in exclude:
            continue
        h.update(name.encode())
        if torch.is_tensor(value):
            a = value.detach().cpu().contiguous()
            h.update(b"tensor" + str(a.dtype).encode())
            h.update(json.dumps(list(a.shape), separators=(",", ":")).encode())
            h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            h.update(b"metadata" + type(value).__name__.encode() + repr(value).encode())
    return h.hexdigest()


def _storage_token(value) -> int:
    storage = getattr(value, "untyped_storage", None)
    if storage is not None:
        return int(storage().data_ptr())
    return int(value.storage().data_ptr())


def _module_specs(model, smoke, torch) -> list[dict]:
    groups = smoke._linear_groups(model)
    enc = [g for g in groups if str(g.get("family", "")).lower() == "encoder"]
    if len(enc) != 12 or sum(len(g.get("linear", [])) for g in enc) != 48:
        raise RuntimeError("encoder target must contain 12 blocks and 48 Linear weights")
    specs = []
    for group in enc:
        for item in group["linear"]:
            module = smoke._module_for_path(model, group["family"], group["index"], item["relative"])
            weight = module.weight.detach()
            if weight.dtype != torch.float32 or not torch.isfinite(weight).all():
                raise RuntimeError(f"encoder weight is not finite float32: {item['path']}")
            specs.append({"path": str(item["path"]), "family": group["family"],
                          "index": int(group["index"]), "relative": item["relative"],
                          "module": module, "shape": list(weight.shape)})
    specs.sort(key=lambda x: x["path"])
    if len({s["path"] for s in specs}) != 48:
        raise RuntimeError("duplicate encoder module path")
    return specs


def _state_layout(model, specs, torch) -> tuple[list[str], list[str]]:
    targets = {_storage_token(s["module"].weight.detach()): s["path"] for s in specs}
    aliases = []
    for name, value in model.state_dict().items():
        if torch.is_tensor(value) and _storage_token(value) in targets:
            aliases.append(str(name))
    if len(aliases) < 48:
        raise RuntimeError(f"state_dict exposes only {len(aliases)} target aliases")
    return sorted(set(aliases)), aliases


def _quant_records(specs, torch) -> dict:
    records = {}
    generators = [torch.Generator(device="cpu").manual_seed(seed) for seed in SEEDS]
    for spec in specs:
        fp = spec["module"].weight.detach().cpu().clone().contiguous()
        rows = fp.reshape(fp.shape[0], -1)
        maximum = rows.abs().amax(dim=1)
        seven = torch.tensor(7.0, dtype=torch.float32)
        scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / seven)
        norm = rows / scale[:, None]
        lower = torch.floor(norm)
        rtn = torch.round(norm).clamp(-7, 7).to(torch.int8).reshape_as(fp)
        codes = [rtn]
        for generator in generators:
            u = torch.rand(tuple(fp.shape), generator=generator, dtype=torch.float32)
            sr = (torch.floor(norm) + (u.reshape_as(norm) < (norm - lower)).to(torch.float32))
            codes.append(sr.clamp(-7, 7).to(torch.int8).reshape_as(fp))
        records[spec["path"]] = {"fp": fp, "scale": scale.to(torch.float32).contiguous(),
            "codes": torch.stack(codes).contiguous(), "readback_sha256": [None] * 4,
            "restore_sha256": None, "shape": list(fp.shape)}
    return records


def _save_quant(path: Path, records: dict, torch) -> None:
    tmp = Path(str(path) + ".tmp")
    torch.save({"schema": "reference-branch-coupling-quant-v1", "module_keying": "complete_path",
                "arms": ["RTN", "SR0", "SR1", "SR2"], "modules": records}, tmp)
    os.replace(tmp, path)


def _restore(specs, records, torch) -> None:
    with torch.no_grad():
        for spec in specs:
            rec = records[spec["path"]]
            spec["module"].weight.copy_(rec["fp"].to(device=spec["module"].weight.device))
            got = spec["module"].weight.detach().cpu()
            if not torch.equal(got, rec["fp"]):
                raise RuntimeError(f"exact restore failed: {spec['path']}")
            rec["restore_sha256"] = _tensor_hash(got, torch)


def _target_equal(specs, records, torch) -> bool:
    return all(torch.equal(s["module"].weight.detach().cpu(), records[s["path"]]["fp"])
               for s in specs)


def _apply(specs, records, arm: int, torch) -> None:
    with torch.no_grad():
        for spec in specs:
            rec = records[spec["path"]]
            code = rec["codes"][arm - 1].to(torch.float32)
            scale = rec["scale"].reshape((-1,) + (1,) * (code.ndim - 1))
            value = code * scale
            module = spec["module"]
            module.weight.copy_(value.to(device=module.weight.device, dtype=module.weight.dtype))
            got = module.weight.detach().cpu()
            if not torch.equal(got, value):
                raise RuntimeError(f"quantized readback failed: {spec['path']}")
            rec["readback_sha256"][arm - 1] = _tensor_hash(got, torch)


def _case(model, obs, goal, actions, torch, np):
    with torch.no_grad():
        z = model.encode(obs, actions)
        cur = z[..., :384]
        g = model.encode_obs(goal)["visual"]
        seen = []
        def hook(_module, inputs):
            seen.append(inputs[0].detach().clone())
        handle = model.predictor.register_forward_pre_hook(hook)
        try:
            pred = model.predict(z)
        finally:
            handle.remove()
        if len(seen) != 1 or tuple(seen[0].shape) != (1, 196, 404) or not torch.equal(seen[0], z[:, 0]):
            raise RuntimeError('actual predictor input readback mismatch')
    if tuple(z.shape) != (1, 1, 196, 404) or tuple(pred.shape) != (1, 1, 196, 404):
        raise RuntimeError(f"expected encode/predict [1,1,196,404], got {tuple(z.shape)}/{tuple(pred.shape)}")
    if tuple(cur.shape) != (1, 1, 196, 384) or tuple(g.shape) != (1, 1, 196, 384):
        raise RuntimeError("unexpected current/goal visual feature shape")
    values = (pred[0, 0, :, :384], g[0, 0], cur[0, 0], seen[0][0])
    arrays = tuple(np.asarray(v.detach().cpu(), dtype=np.float32).copy() for v in values)
    if not all(np.isfinite(v).all() for v in arrays):
        raise RuntimeError("nonfinite model output")
    return arrays


def run(args, allocation) -> None:
    import numpy as np
    started = time.monotonic()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("engineering.json", "raw.npz", "quant_snapshot.pt")):
        raise RuntimeError("output already contains reference results")
    eng = {"schema": "reference-branch-coupling-engineering-v1", "status": "running",
           "allocation": allocation, "checks": {}, "receipts": {}, "completed": [[False] * 5 for _ in range(6)]}
    raw = None
    def check(name, value):
        eng["checks"][name] = bool(value)
        if not value:
            raise RuntimeError(name)
    def receipt(name, path):
        eng["receipts"][name] = {"path": str(path), "sha256": _sha256(Path(path))}
    def deadline():
        if time.monotonic() - started > int(args.max_seconds):
            raise TimeoutError("producer deadline exceeded")
    try:
        freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))
        check("teacher_helper_pin", _sha256(args.teacher_helper) == TEACHER_SHA)
        check("input_manifest_pin", _sha256(args.input_manifest) == MANIFEST_SHA)
        check("protocol_pin", _sha256(args.protocol) == freeze["protocol_sha256"])
        check("freeze_manifest_pin", freeze["input_manifest_sha256"] == MANIFEST_SHA)
        check("freeze_recipe", tuple(freeze["sr_seeds"]) == SEEDS and freeze["model_horizon"] == 1)
        for n, p in (("producer", Path(__file__)), ("teacher_helper", args.teacher_helper),
                     ("protocol", args.protocol), ("freeze", args.freeze)):
            receipt(n, p)
        helper_spec = importlib.util.spec_from_file_location("reference_teacher_helper", args.teacher_helper)
        helper = importlib.util.module_from_spec(helper_spec)
        helper_spec.loader.exec_module(helper)
        manifest, samples = helper._manifest_samples(args.input_manifest)
        smoke, _screen, helper_hashes = helper._load_helpers(args.helper_dir)
        binding = helper._bind_prepared_runtime(args.root, manifest)
        runtime = smoke._runtime(args.root)
        torch, model, device = runtime["torch"], runtime["model"], runtime["device"]
        check("v100", device.type == "cuda" and "V100" in torch.cuda.get_device_name(device).upper())
        gpu = {"name": torch.cuda.get_device_name(device), "compute_capability": list(torch.cuda.get_device_capability(device))}
        eng["gpu"] = gpu
        check("v100_capability", gpu["compute_capability"] == [7, 0])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        check("eval_frozen", not model.training and not any(p.requires_grad for p in model.parameters()))
        groups, structure = helper._runtime_structure(runtime, smoke)
        specs = _module_specs(model, smoke, torch)
        aliases, _all_aliases = _state_layout(model, specs, torch)
        before = _state_digest(model, torch)
        non_target_before = _state_digest(model, torch, frozenset(aliases))
        identity = helper._source_identity(args.root, runtime, smoke, helper_hashes)
        identity["prepared_runtime_binding"] = binding
        identity["teacher_helper_sha256"] = TEACHER_SHA
        identity["protocol_sha256"] = freeze["protocol_sha256"]
        runtime_info = {"identity": identity, "structure": structure, "target_paths": [s["path"] for s in specs],
                        "target_state_aliases": aliases, "torch_version": str(torch.__version__)}
        _write_json(out / "runtime.json", runtime_info)
        receipt("runtime", out / "runtime.json")
        receipt("smoke_runner.py", Path(args.helper_dir) / "smoke_runner.py")
        receipt("screen_runner.py", Path(args.helper_dir) / "screen_runner.py")
        records = _quant_records(specs, torch)
        eng.update({"input_manifest": {"path": str(args.input_manifest), "sha256": MANIFEST_SHA,
                    "sample_sha256": [s["sample_sha256"] for s in samples]}, "trajectory_ids": list(TRAJECTORIES),
                    "valid_indices": list(VALID), "arm_names": list(ARMS), "target_count": len(specs),
                    "target_paths": [s["path"] for s in specs], "state_before": before,
                    "non_target_state_before": non_target_before, "target_state_aliases": aliases,
                    "quantization": {"bits": 4, "qmin": -7, "qmax": 7, "seeds": list(SEEDS),
                                     "traversal": "complete module path lexicographic; one float32 rand per module/draw"}})
        _write_json(out / "engineering.json", eng)
        raw = {"pred_visual": np.full((6, 5, 196, 384), np.nan, np.float32),
               "goal_visual": np.full((6, 5, 196, 384), np.nan, np.float32),
               "current_visual": np.full((6, 5, 196, 384), np.nan, np.float32),
               "fp_noop_pred": np.full((6, 2, 196, 384), np.nan, np.float32),
               "fp_noop_goal": np.full((6, 2, 196, 384), np.nan, np.float32),
               "fp_noop_current": np.full((6, 2, 196, 384), np.nan, np.float32),
               "predictor_input": np.full((6, 5, 196, 404), np.nan, np.float32),
               "valid_indices": np.asarray(VALID, np.int64), "trajectory_ids": np.asarray(TRAJECTORIES, np.int64),
               "arm_names": np.asarray(ARMS, dtype='<U64'), "completed": np.zeros((6, 5), np.bool_),
               "schema": np.asarray("reference-branch-coupling-raw-v1", dtype='<U64')}
        _write_npz(out / "raw.npz", raw)
        preprocessor = smoke._preprocessor(runtime)
        cases = []
        for sample in samples:
            deadline()
            obs0, targets = helper._make_observations(sample, preprocessor, torch, device)
            actions = torch.as_tensor(sample["action_blocks"][:1], dtype=torch.float32, device=device).unsqueeze(0)
            cases.append((obs0, targets[0], actions))
        fp_values = []
        for si, (obs0, goal, actions) in enumerate(cases):
            deadline()
            fp = _case(model, obs0, goal, actions, torch, np)
            fp_values.append(fp)
            raw["pred_visual"][si, 0], raw["goal_visual"][si, 0], raw["current_visual"][si, 0], raw["predictor_input"][si, 0] = fp
            fp_copy = _case(model, {k:v.clone() for k,v in obs0.items()},
                            {k:v.clone() for k,v in goal.items()}, actions.clone(), torch, np)
            raw["fp_noop_pred"][si, 0], raw["fp_noop_goal"][si, 0], raw["fp_noop_current"][si, 0] = fp_copy[:3]
            check(f"fp_copy_{si}", all(np.array_equal(fp[j], fp_copy[j]) for j in range(4)))
            raw["completed"][si, 0] = True
            eng["completed"][si][0] = True
        _write_npz(out / "raw.npz", raw)
        _write_json(out / "engineering.json", eng)
        for arm in range(1, 5):
            check(f"pristine_before_{arm}", _target_equal(specs, records, torch))
            _apply(specs, records, arm, torch)
            check(f"non_target_{arm}", _state_digest(model, torch, frozenset(aliases)) == non_target_before)
            for si, (obs0, goal, actions) in enumerate(cases):
                deadline()
                value = _case(model, obs0, goal, actions, torch, np)
                raw["pred_visual"][si, arm], raw["goal_visual"][si, arm], raw["current_visual"][si, arm], raw["predictor_input"][si, arm] = value
                raw["completed"][si, arm] = True
                eng["completed"][si][arm] = True
            _restore(specs, records, torch)
            check(f"restore_{arm}", _target_equal(specs, records, torch))
            _write_npz(out / "raw.npz", raw)
            _write_json(out / "engineering.json", eng)
            print(json.dumps({"arm": ARMS[arm], "seconds": time.monotonic() - started}), flush=True)
        _restore(specs, records, torch)
        for si, (obs0, goal, actions) in enumerate(cases):
            deadline()
            restored = _case(model, obs0, goal, actions, torch, np)
            raw["fp_noop_pred"][si, 1], raw["fp_noop_goal"][si, 1], raw["fp_noop_current"][si, 1] = restored[:3]
            check(f"fp_restore_{si}", all(np.array_equal(fp_values[si][j], restored[j]) for j in range(4)))
        check("final_restore", _state_digest(model, torch) == before and _state_digest(model, torch, frozenset(aliases)) == non_target_before)
        check("all_completed", bool(raw["completed"].all()))
        check("finite_raw", all(np.isfinite(v).all() for k, v in raw.items() if k not in {"schema", "arm_names"}))
        check("quant_receipts_complete", all(all(x is not None for x in r["readback_sha256"]) and r["restore_sha256"] is not None for r in records.values()))
        _save_quant(out / "quant_snapshot.pt", records, torch)
        _write_npz(out / 'raw.npz', raw)
        receipt("raw", out / "raw.npz")
        receipt("quant_snapshot", out / "quant_snapshot.pt")
        receipt('contract', out / 'RAW_CONTRACT.json')
        eng['state_after'] = _state_digest(model, torch)
        eng['non_target_state_after'] = _state_digest(model, torch, frozenset(aliases))
        check('predictor_input_readback', True)  # All 42 guarded _case calls returned.
        check('fp32_execution', all(p.dtype == torch.float32 for p in model.parameters()))
        check('nonvisual_input_unchanged', np.array_equal(raw['predictor_input'][...,384:],
            np.broadcast_to(raw['predictor_input'][:,:1,:,384:], (6,5,196,20))))
        deadline()
        eng["status"] = "complete"
        eng["work_seconds"] = time.monotonic() - started
        eng['max_memory_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
        _write_json(out / "engineering.json", eng)
        _write_json(out / "summary.json", {"schema": "reference-branch-coupling-summary-v1", "status": "complete",
            "completed": int(raw["completed"].sum()), "raw": eng["receipts"]["raw"],
            "quant_snapshot": eng["receipts"]["quant_snapshot"], "arms": list(ARMS),
            "valid_indices": list(VALID), "trajectory_ids": list(TRAJECTORIES),
            "note": "CPU verifier computes Shared/Different 3x3 pairing and all scientific gates."})
    except Exception as exc:
        eng["status"] = "failed"
        eng["error"] = f"{type(exc).__name__}: {exc}"[:1200]
        if raw is not None:
            try:
                _write_npz(out / "raw.npz", raw)
                eng["partial_raw_sha256"] = _sha256(out / "raw.npz")
            except Exception:
                pass
        traceback.print_exc()
        raise
    finally:
        _write_json(out / "engineering.json", eng)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    for name in ("root", "input-manifest", "teacher-helper", "helper-dir", "protocol", "freeze", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--max-seconds", type=int, default=180)
    return p.parse_args()


if __name__ == "__main__":
    from allocation_guard import require_allocation
    _allocation = require_allocation()
    run(_args(), _allocation)
