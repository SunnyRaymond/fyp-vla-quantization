"""Bounded PRR CAL smoke and four-arm hard-W4 runner for CCDS.

The only inherited experiment code is the runtime bootstrap, record parser and
SourceHistoryAdapter.  Fitting, donor construction, H=2 evaluation and
quantization are implemented here.  ``engineering`` never reads DEV; ``all``
runs the CAL smoke first and opens DEV only after it passes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = ("q_local", "q_recovery", "l_local", "l_recovery")
SEEDS = (1201, 1202, 1203)
SCHEMA = "prr-ccds-v1"


def _load_base(manifest: Mapping[str, Any]) -> Any:
    candidates = []
    configured = manifest.get("frt_pythonpath") or os.environ.get("PRR_FRT_PYTHONPATH")
    if configured:
        candidates.append(Path(str(configured)))
    candidates += [
        Path("/tc1home/UG/yguo017/frt_ccds/artifacts/64694"),
        Path("/tc1home/UG/yguo017/frt_ccds/control"),
        HERE.parent / "frt-ccds",
    ]
    for candidate in candidates:
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    return importlib.import_module("frt_runner")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    raise TypeError(f"cannot encode {type(value)!r}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    tmp.replace(path)


def _atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    tmp.replace(path)


def _atomic_torch(path: Path, payload: Any, torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        torch.save(payload, stream)
    tmp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not str(payload.get("schema", "")).startswith("prr"):
        raise ValueError("expected a PRR manifest")
    return dict(payload)


def _expected_indices(base: Any, manifest: Mapping[str, Any], split: str) -> set[int]:
    value = base._declared_indices(manifest, split)
    if value is not None:
        return value
    return set(range(72, 78) if split == "cal" else range(78, 84))


def _rows(base: Any, manifest: Mapping[str, Any], manifest_path: Path, split: str) -> list[dict[str, Any]]:
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or split not in splits:
        raise ValueError(f"manifest.splits.{split} is required")
    rows = base._split_source(splits[split], manifest_path.parent, split)
    rows = base._validate_split_rows(rows, split, require_fresh=True)
    expected = _expected_indices(base, manifest, split)
    actual = {int(row["dataset_index"]) for row in rows}
    if actual != expected:
        raise ValueError(f"{split} dataset indices {sorted(actual)} != {sorted(expected)}")
    if len(rows) != 12 or any(sum(int(r["dataset_index"]) == index for r in rows) != 2 for index in expected):
        raise ValueError(f"{split} must contain two windows for each of six indices")
    return rows


def _expand(base: Any, rows: Sequence[Mapping[str, Any]], adapter: Any, torch: Any, device: Any, split: str) -> list[dict[str, Any]]:
    result = base._expand_batched_rows(rows, adapter, torch, device, split)
    if len(result) != 12:
        raise ValueError(f"{split} expansion must contain 12 records, got {len(result)}")
    return result


def _record_metadata(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = ("episode_id", "record_index", "dataset_index", "window_start", "env_seed", "cem_seed", "replay_seed", "target_fingerprint")
    return [{key: row.get(key) for key in fields} for row in rows]


def _state_cpu(model: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _restore_state(model: Any, state: Mapping[str, Any]) -> None:
    model.load_state_dict(state, strict=True)


def _cat(rows: Sequence[Mapping[str, Any]], key: str, device: Any) -> Any:
    torch = __import__("torch")
    return torch.cat([row[key].to(device) for row in rows], dim=0)


def _fp_h2(runtime: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], adapter: Any) -> dict[str, Any]:
    torch = runtime["torch"]
    device = next(runtime["model"].parameters()).device
    history, action, next_action = (_cat(rows, key, device) for key in ("history", "action", "next_action"))
    with torch.no_grad():
        first = adapter.one_step(history, action)
        first_history = adapter.append(history, first)
        second = adapter.one_step(first_history, next_action)
    return {"history": history.detach().cpu(), "action": action.detach().cpu(), "next_action": next_action.detach().cpu(), "first": first.detach().cpu(), "first_history": first_history.detach().cpu(), "second": second.detach().cpu()}


def _source_h2_gate(runtime: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], adapter: Any) -> dict[str, Any]:
    """Compare the source rollout's first two slots where collector bridge data exists."""
    torch, model = runtime["torch"], runtime["model"]
    checked, errors, nulls, episodes = 0, [], [], set()
    seen = set()
    for row in rows:
        episode = str(row.get("episode_id", ""))
        if episode in seen or not isinstance(row.get("obs_0", row.get("source_bridge_obs_0")), Mapping):
            continue
        seen.add(episode)
        obs_value = row.get("source_bridge_obs_0", row.get("obs_0"))
        actions_value = row.get("source_bridge_actions", row.get("actions"))
        if not isinstance(actions_value, (np.ndarray, list, tuple)) and not hasattr(actions_value, "shape"):
            errors.append(f"{episode}:missing_source_actions")
            continue
        try:
            obs = {key: runtime["base"]._tensor(value, torch, next(model.parameters()).device, torch.float32) for key, value in obs_value.items()}
            if obs["visual"].ndim == 4:
                obs = {key: value.unsqueeze(0) for key, value in obs.items()}
            actions = runtime["base"]._tensor(actions_value, torch, next(model.parameters()).device, torch.float32)
            if actions.ndim == 2:
                actions = actions.unsqueeze(0)
            initial_len = int(obs["visual"].shape[1])
            if actions.shape[1] < initial_len + 2:
                raise ValueError("source action sequence is shorter than H=2")
            history = adapter.encode(obs, actions[:, :initial_len])
            first = adapter.one_step(history, actions[:, initial_len : initial_len + 1])
            second = adapter.one_step(adapter.append(history, first), actions[:, initial_len + 1 : initial_len + 2])
            with torch.no_grad():
                _, source_z = model.rollout(obs_0=obs, act=actions)
            actual = source_z[:, initial_len : initial_len + 2]
            expected = torch.cat([first, second], dim=1)
            error = float((expected - actual).abs().max().item())
            checked += 1
            episodes.add(episode)
            if error > 1e-5:
                errors.append(f"{episode}:max_abs={error:g}")
            null = adapter.one_step(history, actions[:, initial_len : initial_len + 1]) - first
            nulls.append(float(null.abs().max().item()))
        except Exception as exc:
            errors.append(f"{episode}:{type(exc).__name__}:{exc}")
    return {"passed": checked > 0 and not errors and max(nulls or [1.0]) == 0.0, "checked": checked, "episodes": sorted(episodes), "errors": errors, "fp_null_max_abs": max(nulls or [float("nan")])}


def _wz(runtime: Mapping[str, Any], fp: Mapping[str, Any], adapter: Any) -> tuple[Any, Any]:
    torch = runtime["torch"]
    values = torch.cat([fp["first"].to(next(runtime["model"].parameters()).device), fp["second"].to(next(runtime["model"].parameters()).device)], dim=0)
    std = values.std(dim=0, unbiased=False)
    wz = (1.0 / std.clamp_min(1e-3)).squeeze(0)
    mask = adapter.action_mask(values[:1])[0, 0]
    wz = wz.masked_fill(mask, 0).detach().cpu()
    return wz, mask.detach().cpu()


def _weighted_mse(pred: Any, target: Any, wz: Any) -> Any:
    weight = wz.to(device=pred.device, dtype=pred.dtype)
    return ((pred - target) * weight).square().mean()


def _load_optional_clean(manifest: Mapping[str, Any], manifest_path: Path, runtime: Mapping[str, Any], modules: Sequence[tuple[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    torch = runtime["torch"]
    donors = manifest.get("donors", {})
    cfg = donors.get("clean_seed_1201") if isinstance(donors, Mapping) else None
    if not isinstance(cfg, Mapping) or not bool(cfg.get("provenance_verified")):
        return None, {"id": "clean_seed_1201", "included": False, "reason": "manifest_provenance_verified_false"}
    if str(cfg.get("method", "clean")) != "clean" or int(cfg.get("seed", 1201)) != 1201 or not bool(cfg.get("cal_only", True)):
        return None, {"id": "clean_seed_1201", "included": False, "reason": "method_seed_or_cal_only_mismatch"}
    path = Path(str(cfg.get("checkpoint", "")))
    if not path.is_absolute():
        path = (manifest_path.parent / path).resolve()
    if not path.is_file():
        return None, {"id": "clean_seed_1201", "included": False, "reason": "checkpoint_missing", "path": str(path)}
    expected = cfg.get("sha256")
    actual = _sha256(path)
    if expected and str(expected) != actual:
        return None, {"id": "clean_seed_1201", "included": False, "reason": "checkpoint_sha256_mismatch", "path": str(path)}
    payload = torch.load(path, map_location="cpu")
    state = payload.get("state_dict", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(state, Mapping):
        raise ValueError("old clean donor has no state_dict")
    info = {"id": "clean_seed_1201", "included": True, "path": str(path), "sha256": actual, "provenance_verified": True}
    return {"state": state, "payload": payload}, info


def _donor_first(runtime: Mapping[str, Any], modules: Sequence[tuple[str, Any]], base_state: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], adapter: Any, clean: Mapping[str, Any] | None) -> tuple[list[str], list[Any]]:
    torch, model = runtime["torch"], runtime["model"]
    history, action = _cat(rows, "history", next(model.parameters()).device), _cat(rows, "action", next(model.parameters()).device)
    names, outputs = [], []
    import prr_quant
    _restore_state(model, base_state)
    prr_quant.materialize_rtn(modules)
    with torch.no_grad():
        outputs.append(adapter.one_step(history, action).detach())
    names.append("q0")
    _restore_state(model, base_state)
    if clean is not None:
        model.load_state_dict(clean["state"], strict=True)
        if isinstance(clean.get("payload"), Mapping) and isinstance(clean["payload"].get("hard_quant"), Mapping):
            prr_quant.restore_hard_map(modules, clean["payload"]["hard_quant"])
        with torch.no_grad():
            outputs.append(adapter.one_step(history, action).detach())
        names.append("clean_seed_1201")
    _restore_state(model, base_state)
    return names, outputs


def _make_split(runtime: Mapping[str, Any], modules: Sequence[tuple[str, Any]], base_state: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], adapter: Any, clean: Mapping[str, Any] | None, wz: Any | None = None) -> dict[str, Any]:
    torch, model = runtime["torch"], runtime["model"]
    fp = _fp_h2(runtime, rows, adapter)
    names, firsts = _donor_first(runtime, modules, base_state, rows, adapter, clean)
    history = fp["history"].to(next(model.parameters()).device)
    next_action = fp["next_action"].to(next(model.parameters()).device)
    donors, local, recovery = [], [], []
    with torch.no_grad():
        for name, first in zip(names, firsts):
            donor_history = adapter.append(history, first)
            local_target = adapter.one_step(donor_history, next_action)
            donors.append(donor_history.detach().cpu())
            local.append(local_target.detach().cpu())
            recovery.append(fp["second"].to(donor_history.device).detach().cpu())
    if wz is None:
        wz, action_mask = _wz(runtime, fp, adapter)
    else:
        action_mask = adapter.action_mask(fp["first"][:1].to(next(model.parameters()).device))[0, 0].detach().cpu()
    donor_history = torch.cat(donors, dim=0)
    local_target = torch.cat(local, dim=0)
    recovery_target = torch.cat(recovery, dim=0)
    donor_count = len(names)
    count = len(rows)
    return {
        "history": fp["history"], "action": fp["action"], "next_action": fp["next_action"], "fp_first": fp["first"], "fp_second": fp["second"], "fp_first_history": fp["first_history"],
        "donor_history": donor_history, "local_target": local_target, "recovery_target": recovery_target, "donor_names": names, "donor_first": torch.stack([x.cpu() for x in firsts], dim=0),
        "donor_local_target": local_target.reshape(donor_count, count, *local_target.shape[1:]), "donor_recovery_target": recovery_target.reshape(donor_count, count, *recovery_target.shape[1:]), "wz": wz, "action_mask": action_mask,
    }


def _indices(n: int, batch: int, seed: int, step: int) -> np.ndarray:
    if batch >= n:
        return np.arange(n, dtype=np.int64)
    return np.sort(np.random.default_rng(int(seed) + int(step) * 1000003).choice(n, size=batch, replace=False).astype(np.int64))


def _fit(runtime: Mapping[str, Any], modules: Sequence[tuple[str, Any]], base_state: Mapping[str, Any], bank: Mapping[str, Any], adapter: Any, arm: str, seed: int, steps: int, batch_size: int, output: Path | None, *, smoke: bool = False) -> dict[str, Any]:
    torch, model = runtime["torch"], runtime["model"]
    import prr_quant
    _restore_state(model, base_state)
    use_lora = arm.startswith("l_")
    handles = prr_quant.attach(modules, use_lora=use_lora, seed=seed, rank=4)
    quant_params = [p for h in handles for p in (h.parametrization.scale_raw, h.parametrization.alpha)]
    lora_params = [p for h in handles for p in (getattr(h.parametrization, "lora_A", None), getattr(h.parametrization, "lora_B", None)) if p is not None]
    groups = [{"params": quant_params, "lr": 0.01}]
    if lora_params:
        groups.append({"params": lora_params, "lr": 0.001})
    optimizer = torch.optim.Adam(groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    data = {key: value.to(next(model.parameters()).device) for key, value in bank.items() if hasattr(value, "to")}
    n = int(data["history"].shape[0])
    donor_n = int(data["donor_history"].shape[0])
    initial_params = {f"{h.path}:{name}": parameter.detach().clone() for h in handles for name, parameter in (("scale_raw", h.parametrization.scale_raw), ("alpha", h.parametrization.alpha), ("lora_A", getattr(h.parametrization, "lora_A", None)), ("lora_B", getattr(h.parametrization, "lora_B", None))) if parameter is not None}
    trace, grads, started = [], [], time.monotonic()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for step in range(int(steps)):
        frac = (step + 1) / max(1, int(steps))
        temperature, reg_weight = 2.0 + (0.1 - 2.0) * frac, 0.01 + (0.1 - 0.01) * frac
        for h in handles:
            h.parametrization.temperature = temperature
        index = torch.as_tensor(_indices(donor_n, min(int(batch_size), donor_n), seed, step), device=next(model.parameters()).device, dtype=torch.long)
        clean_index = torch.remainder(index, n)
        with torch.enable_grad():
            clean_pred = adapter.one_step(data["history"].index_select(0, clean_index), data["action"].index_select(0, clean_index))
            donor_pred = adapter.one_step(data["donor_history"].index_select(0, index), data["next_action"].index_select(0, clean_index))
            clean_loss = _weighted_mse(clean_pred, data["fp_first"].index_select(0, clean_index), data["wz"])
            target_key = "local_target" if arm.endswith("local") else "recovery_target"
            target_loss = _weighted_mse(donor_pred, data[target_key].index_select(0, index), data["wz"])
            rounding = torch.stack([h.parametrization.rounding_regularizer() for h in handles]).mean()
            total = clean_loss + target_loss + reg_weight * rounding
            if not bool(torch.isfinite(total).item()):
                raise FloatingPointError(f"non-finite PRR loss at {arm}:{seed}:{step + 1}")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            grad = prr_quant.grad_norms(handles)
            optimizer.step()
        grads.append(grad)
        if step == 0 or step + 1 == steps or step + 1 in (2, 3) or (step + 1) % max(1, min(100, steps // 5 or 1)) == 0:
            trace.append({"step": step + 1, "clean_loss": float(clean_loss.detach().item()), "target_loss": float(target_loss.detach().item()), "rounding_loss": float(rounding.detach().item()), "total_loss": float(total.detach().item()), "target_norm": float((data[target_key].index_select(0, index) * data["wz"]).reshape(len(index), -1).norm(dim=1).mean().item())})
    update_norms = {name: float((parameter.detach() - initial_params[name]).norm().item()) for h in handles for name, parameter in ((f"{h.path}:scale_raw", h.parametrization.scale_raw), (f"{h.path}:alpha", h.parametrization.alpha), (f"{h.path}:lora_A", getattr(h.parametrization, "lora_A", None)), (f"{h.path}:lora_B", getattr(h.parametrization, "lora_B", None))) if parameter is not None}
    hard_rows = prr_quant.materialize_hard(handles)
    state = _state_cpu(model)
    target_names = {f"{path}.weight" for path, _ in modules}
    non_target_state = {key: value for key, value in state.items() if key not in target_names}
    ledger = [{"path": row["path"], "shape": row["shape"], "q_min": -7, "q_max": 7, "integer_min": row["integer_min"], "integer_max": row["integer_max"], "integer_sha256": hashlib.sha256(row["integer"].numpy().tobytes()).hexdigest(), "scale_sha256": hashlib.sha256(row["scale"].numpy().tobytes()).hexdigest(), "scale_min": float(row["scale"].min()), "scale_max": float(row["scale"].max()), "use_lora": row["use_lora"]} for row in hard_rows]
    payload = {"schema": "prr-hard-checkpoint-v1", "arm": arm, "seed": int(seed), "teacher_removed": True, "lora_removed": True, "alpha_removed": True, "state_dict": non_target_state, "hard_quant": prr_quant.hard_map(hard_rows), "hard_ledger": ledger}
    before = None
    with torch.no_grad():
        before = adapter.one_step(data["history"][:1], data["action"][:1]).detach().cpu()
    _restore_state(model, base_state)
    _load_payload(runtime, modules, payload)
    with torch.no_grad():
        after = adapter.one_step(data["history"][:1], data["action"][:1]).detach().cpu()
    reload_equal = bool(torch.equal(before, after))
    _restore_state(model, base_state)
    peak_alloc = float(torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else 0.0
    peak_reserved = float(torch.cuda.max_memory_reserved() / 2**30) if torch.cuda.is_available() else 0.0
    elapsed = time.monotonic() - started
    non_target_bytes = sum(int(value.numel() * value.element_size()) for key, value in state.items() if key not in target_names)
    integer_bytes = sum(int(row["integer"].numel() * row["integer"].element_size()) for row in hard_rows)
    scale_bytes = sum(int(row["scale"].numel() * row["scale"].element_size()) for row in hard_rows)
    def _grad_max(suffixes: Sequence[str]) -> float:
        return max((float(value) for item in grads[1:] for name, value in item.items() if any(name.endswith(suffix) for suffix in suffixes)), default=0.0)
    gradient_summary = {"scale_raw_max_after_step2": _grad_max((":scale_raw",)), "alpha_max_after_step2": _grad_max((":alpha",)), "lora_A_max_after_step2": _grad_max((":lora_A",)), "lora_B_max_after_step2": _grad_max((":lora_B",))}
    finite_grad = bool(all(np.isfinite(value) for item in grads for value in item.values()))
    quant_grad_ok = gradient_summary["scale_raw_max_after_step2"] > 0 or gradient_summary["alpha_max_after_step2"] > 0
    lora_a_ok = not use_lora or gradient_summary["lora_A_max_after_step2"] > 0
    lora_b_ok = not use_lora or gradient_summary["lora_B_max_after_step2"] > 0
    update_summary = {"quantizer_max": max((value for name, value in update_norms.items() if name.endswith(":scale_raw") or name.endswith(":alpha")), default=0.0), "lora_A_max": max((value for name, value in update_norms.items() if name.endswith(":lora_A")), default=0.0), "lora_B_max": max((value for name, value in update_norms.items() if name.endswith(":lora_B")), default=0.0)}
    result = {"schema": "prr-fit-v1", "arm": arm, "seed": int(seed), "status": "complete", "fit_updates_completed": int(steps), "fit_updates_expected": int(steps), "batch_size": int(batch_size), "gamma": 1.0, "quantizer_lr": 0.01, "lora_lr": 0.001 if use_lora else None, "rank": 4 if use_lora else 0, "batch_schedule_hash": hashlib.sha256(json.dumps([_indices(donor_n, min(int(batch_size), donor_n), seed, step).tolist() for step in range(int(steps))], separators=(",", ":")).encode()).hexdigest(), "fit_trace": trace, "gradient_summary": gradient_summary, "finite_grad": finite_grad, "quant_grad_after_step2": quant_grad_ok, "lora_A_grad_after_step2": lora_a_ok, "lora_B_grad_after_step2": lora_b_ok, "update_summary": update_summary, "update_norms": update_norms, "quant_update_nonzero": update_summary["quantizer_max"] > 0, "lora_A_update_nonzero": update_summary["lora_A_max"] > 0, "lora_B_update_nonzero": update_summary["lora_B_max"] > 0, "reload_equal": reload_equal, "hard_ledger": ledger, "hard_payload": payload, "elapsed_seconds": elapsed, "peak_vram_allocated_gib": peak_alloc, "peak_vram_reserved_gib": peak_reserved, "non_target_state_bytes": non_target_bytes, "logical_w4_integer_bytes": integer_bytes, "scale_bytes": scale_bytes, "smoke": smoke}
    if output is not None:
        checkpoint = output / "methods" / f"{arm}_seed_{seed}.pt"
        _atomic_torch(checkpoint, payload, torch)
        result["checkpoint"] = str(checkpoint.resolve())
        result["checkpoint_sha256"] = _sha256(checkpoint)
        result.pop("hard_payload", None)
        _atomic_json(output / "methods" / f"{arm}_seed_{seed}.json", result)
    return result


def _load_payload(runtime: Mapping[str, Any], modules: Sequence[tuple[str, Any]], payload: Mapping[str, Any]) -> None:
    import prr_quant
    target_names = {f"{path}.weight" for path, _ in modules}
    incompatible = runtime["model"].load_state_dict(payload["state_dict"], strict=False)
    if set(incompatible.missing_keys) != target_names or incompatible.unexpected_keys:
        raise RuntimeError(f"hard checkpoint state coverage mismatch: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}")
    prr_quant.restore_hard_map(modules, payload["hard_quant"])


def _smoke_reload(runtime: Mapping[str, Any], modules: Sequence[tuple[str, Any]], payload: Mapping[str, Any], bank: Mapping[str, Any], adapter: Any) -> bool:
    torch, model = runtime["torch"], runtime["model"]
    with torch.no_grad():
        _load_payload(runtime, modules, payload)
        a = adapter.one_step(bank["history"][:1].to(next(model.parameters()).device), bank["action"][:1].to(next(model.parameters()).device)).detach().cpu()
    buf = io.BytesIO()
    torch.save(payload, buf)
    buf.seek(0)
    loaded = torch.load(buf, map_location=next(model.parameters()).device)
    _load_payload(runtime, modules, loaded)
    with torch.no_grad():
        b = adapter.one_step(bank["history"][:1].to(next(model.parameters()).device), bank["action"][:1].to(next(model.parameters()).device)).detach().cpu()
    return bool(torch.equal(a, b))


def _engineering(runtime: Mapping[str, Any], modules: Sequence[tuple[str, Any]], base_state: Mapping[str, Any], cal: Mapping[str, Any], adapter: Any, manifest: Mapping[str, Any], output: Path) -> dict[str, Any]:
    torch, model = runtime["torch"], runtime["model"]
    source_gate = _source_h2_gate(runtime, cal["rows"], adapter)
    with torch.no_grad():
        h = cal["bank"]["history"].to(next(model.parameters()).device)
        a = cal["bank"]["action"].to(next(model.parameters()).device)
        p1 = adapter.one_step(h, a)
        p2 = adapter.one_step(h, a)
        fp_null = {"passed": bool(torch.equal(p1, p2)), "max_abs": float((p1 - p2).abs().max().item())}
        q0_delta = cal["bank"]["donor_first"][0].to(p1.device) - cal["bank"]["fp_first"].to(p1.device)
        action_mask = cal["bank"]["action_mask"].to(p1.device)
        action_gate = {"passed": bool(torch.equal(q0_delta.masked_select(action_mask), torch.zeros_like(q0_delta.masked_select(action_mask)))), "max_abs_action_delta": float(q0_delta.masked_select(action_mask).abs().max().item())}
        appended = adapter.append(h, p1)
        history_gate = {"passed": bool(h.shape[1] == 1 or torch.equal(appended[:, :-1], h[:, 1:])), "num_hist": int(h.shape[1])}
    cfg = manifest.get("engineering", {}) if isinstance(manifest.get("engineering"), Mapping) else {}
    steps = int(cfg.get("smoke_steps", 3))
    smoke_runs = []
    for arm in ARMS:
        fit = _fit(runtime, modules, base_state, cal["bank"], adapter, arm, 1201, steps, 2, None, smoke=True)
        fit["hard_reload_serialized"] = bool(_smoke_reload(runtime, modules, fit["hard_payload"], cal["bank"], adapter))
        _restore_state(model, base_state)
        full_fit = dict(fit)
        full_fit.pop("hard_payload", None)
        fit.pop("hard_payload", None)
        _atomic_json(output / f"engineering_fit_{arm}_seed_1201.json", full_fit)
        smoke_runs.append({key: fit.get(key) for key in ("arm", "seed", "fit_updates_completed", "batch_size", "elapsed_seconds", "peak_vram_allocated_gib", "peak_vram_reserved_gib", "gradient_summary", "update_summary", "finite_grad", "quant_grad_after_step2", "lora_A_grad_after_step2", "lora_B_grad_after_step2", "quant_update_nonzero", "lora_A_update_nonzero", "lora_B_update_nonzero", "reload_equal", "hard_reload_serialized", "fit_trace")})
    gates = {"source_h2": source_gate, "fp_null": fp_null, "action_invariant": action_gate, "history_invariant": history_gate}
    def arm_ok(item):
        lora = str(item["arm"]).startswith("l_")
        return bool(item["reload_equal"] and item["hard_reload_serialized"] and item["finite_grad"] and item["quant_grad_after_step2"] and item["quant_update_nonzero"] and (not lora or (item["lora_A_grad_after_step2"] and item["lora_B_grad_after_step2"] and item["lora_A_update_nonzero"] and item["lora_B_update_nonzero"])))
    engineering_pass = bool(source_gate["passed"] and fp_null["passed"] and action_gate["passed"] and history_gate["passed"] and all(arm_ok(item) for item in smoke_runs))
    result = {"schema": "prr-engineering-v1", "status": "complete" if engineering_pass else "blocked", "engineering_pass": engineering_pass, "scope": "CAL_only", "dev_opened": False, "steps": steps, "batch_size": 2, "arms": list(ARMS), "gamma": 1.0, "wz_shape": list(cal["bank"]["wz"].shape), "target_norms": {"local": float((cal["bank"]["local_target"] * cal["bank"]["wz"]).reshape(cal["bank"]["local_target"].shape[0], -1).norm(dim=1).mean().item()), "recovery": float((cal["bank"]["recovery_target"] * cal["bank"]["wz"]).reshape(cal["bank"]["recovery_target"].shape[0], -1).norm(dim=1).mean().item())}, "gates": gates, "smoke_runs": smoke_runs, "donors": cal["donor_info"]}
    _atomic_json(output / "engineering_summary.json", result)
    _restore_state(model, base_state)
    return result


def _evaluate(runtime: Mapping[str, Any], modules: Sequence[tuple[str, Any]], payload: Mapping[str, Any], dev: Mapping[str, Any], adapter: Any) -> tuple[Any, Any, Any]:
    torch, model = runtime["torch"], runtime["model"]
    _load_payload(runtime, modules, payload)
    device = next(model.parameters()).device
    with torch.no_grad():
        history, action, next_action = dev["history"].to(device), dev["action"].to(device), dev["next_action"].to(device)
        first = adapter.one_step(history, action)
        terminal = adapter.one_step(adapter.append(history, first), next_action)
        clean_error = (first - dev["fp_first"].to(device)).detach().cpu()
        terminal_error = (terminal - dev["fp_second"].to(device)).detach().cpu()
        donor_errors = []
        for start in range(0, len(dev["donor_names"]) * len(dev["history"]), len(dev["history"])):
            donor_h = dev["donor_history"][start : start + len(dev["history"])].to(device)
            pred = adapter.one_step(donor_h, next_action)
            target = dev["recovery_target"][start : start + len(dev["history"])].to(device)
            donor_errors.append((pred - target).detach().cpu())
    return clean_error, terminal_error, torch.stack(donor_errors, dim=0)


def _formal(runtime: Mapping[str, Any], modules: Sequence[tuple[str, Any]], base_state: Mapping[str, Any], cal: Mapping[str, Any], dev: Mapping[str, Any], adapter: Any, manifest: Mapping[str, Any], output: Path, progress: MutableMapping[str, Any]) -> dict[str, Any]:
    torch = runtime["torch"]
    cal_data = cal.get("bank", cal)
    dev_data = dev.get("bank", dev)
    fit = manifest.get("fit", {}) if isinstance(manifest.get("fit"), Mapping) else {}
    steps, batch = int(fit.get("updates", fit.get("fit_updates", 1000))), int(fit.get("batch_size", 2))
    if steps != 1000 or batch != 2:
        raise ValueError("PRR R1 is frozen to 1000 updates and batch 2")
    results, clean_rows, terminal_rows, donor_rows = [], [], [], []
    device = next(runtime["model"].parameters()).device
    for arm in ARMS:
        for seed in SEEDS:
            key = f"{arm}:{seed}"
            path = output / "methods" / f"{arm}_seed_{seed}.json"
            checkpoint = output / "methods" / f"{arm}_seed_{seed}.pt"
            if key in progress.get("completed", []) and path.is_file() and checkpoint.is_file():
                item = json.loads(path.read_text(encoding="utf-8"))
            else:
                item = _fit(runtime, modules, base_state, cal_data, adapter, arm, seed, steps, batch, output)
                progress.setdefault("completed", []).append(key)
                progress["current"] = key
                _atomic_json(output / "progress.json", progress)
            payload = torch.load(item["checkpoint"], map_location=device)
            clean, terminal, donor = _evaluate(runtime, modules, payload, dev_data, adapter)
            clean_rows.append(clean.numpy().astype(np.float32))
            terminal_rows.append(terminal.numpy().astype(np.float32))
            donor_rows.append(donor.numpy().astype(np.float32))
            del payload
            _restore_state(runtime["model"], base_state)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            results.append(item)
    _restore_state(runtime["model"], base_state)
    arrays = {"arm_ids": np.asarray([f"{arm}:{seed}" for arm in ARMS for seed in SEEDS], dtype="U"), "clean_error": np.stack(clean_rows), "terminal_error": np.stack(terminal_rows), "donor_recovery_error": np.stack(donor_rows), "fp_first": dev_data["fp_first"].numpy().astype(np.float32), "fp_second": dev_data["fp_second"].numpy().astype(np.float32), "fp_donor_local_target": dev_data["donor_local_target"].numpy().astype(np.float32), "fp_donor_recovery_target": dev_data["donor_recovery_target"].numpy().astype(np.float32), "wz": dev_data["wz"].numpy().astype(np.float32), "action_mask": dev_data["action_mask"].numpy().astype(bool)}
    metadata = {"schema": "prr-raw-final-v1", "cal_record_rows": cal_data.get("metadata", []), "dev_record_rows": dev_data["metadata"], "record_rows": dev_data["metadata"], "donors": dev_data["donor_names"], "arm_order": arrays["arm_ids"].tolist(), "target": "fp_second_on_same_full_DEV_batch", "clean_anchor": "fp_first_on_same_full_DEV_batch", "gamma": 1.0, "horizon": 2, "dev_tuning": False, "test_opened": False, "hard_checkpoint_ledger": [{"arm": item["arm"], "seed": item["seed"], "checkpoint": item.get("checkpoint"), "checkpoint_sha256": item.get("checkpoint_sha256") or (_sha256(Path(item["checkpoint"])) if item.get("checkpoint") else None), "hard_ledger": item.get("hard_ledger")} for item in results]}
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, default=_json_default), dtype="U")
    _atomic_npz(output / "raw_final.npz", arrays)
    summary = {"schema": "prr-final-v1", "status": "complete", "formal": True, "scope": "R1_only", "arms": list(ARMS), "seeds": list(SEEDS), "fit_updates": steps, "batch_size": batch, "gamma": 1.0, "horizon": 2, "cal_records": len(cal.get("rows", [])), "dev_records": len(dev.get("rows", [])), "donors": dev_data["donor_names"], "raw_final": str((output / "raw_final.npz").resolve()), "raw_final_sha256": _sha256(output / "raw_final.npz"), "raw_shapes": {key: list(value.shape) for key, value in arrays.items() if hasattr(value, "shape")}, "fit_runs": [{key: item.get(key) for key in ("arm", "seed", "checkpoint", "checkpoint_sha256", "fit_updates_completed", "batch_size", "reload_equal", "elapsed_seconds", "peak_vram_allocated_gib", "peak_vram_reserved_gib")} for item in results], "method_ledger": "methods/<arm>_seed_<seed>.json:hard_ledger", "dev_tuning": False, "r2_opened": False, "test_opened": False, "execution": "fake_quant_emulation_only"}
    _atomic_json(output / "raw_final_summary.json", summary)
    progress.update({"status": "complete", "current": None, "result": "raw_final_summary.json"})
    _atomic_json(output / "progress.json", progress)
    return summary


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--stage", choices=("engineering", "formal", "all"), default="all")
    return parser.parse_args()


def main() -> None:
    args = _parse()
    from allocation_guard import require_allocation
    allocation = require_allocation()
    manifest_path, output = args.manifest.resolve(), args.output.resolve()
    manifest = _load_manifest(manifest_path)
    base = _load_base(manifest)
    gpu = base._verify_gpu(allocation)
    root = base._resolve(args.root or manifest.get("runtime_root") or manifest.get("root"), manifest_path.parent)
    smoke, runtime, identity = base._load_runtime(manifest, root, args.device)
    runtime["base"] = base
    runtime["model"].eval()
    for parameter in runtime["model"].parameters():
        parameter.requires_grad_(False)
    base._check_identity(manifest, identity, gpu)
    modules, groups = base._target_modules(smoke, runtime, manifest)
    if len(modules) != 24:
        raise RuntimeError(f"PRR requires exactly 24 predictor Linear targets, got {len(modules)}")
    adapter = base._core().SourceHistoryAdapter(runtime["model"])
    device = next(runtime["model"].parameters()).device
    cal_rows, cal_expanded = _rows(base, manifest, manifest_path, "cal"), None
    cal_expanded = _expand(base, cal_rows, adapter, runtime["torch"], device, "cal")
    base_state = _state_cpu(runtime["model"])
    clean, clean_info = _load_optional_clean(manifest, manifest_path, runtime, modules)
    cal_fp = _fp_h2(runtime, cal_expanded, adapter)
    cal_wz, cal_mask = _wz(runtime, cal_fp, adapter)
    cal_bank = _make_split(runtime, modules, base_state, cal_expanded, adapter, clean, cal_wz)
    cal_bank.update({"wz": cal_wz, "action_mask": cal_mask})
    cal = {"rows": cal_expanded, "bank": cal_bank, "donor_info": [ {"id": name, "included": True} for name in cal_bank["donor_names"] ]}
    if not any(item["id"] == "clean_seed_1201" for item in cal["donor_info"]):
        cal["donor_info"].append(clean_info)
    output.mkdir(parents=True, exist_ok=True)
    identity_progress = {"schema": "prr-progress-v1", "manifest_sha256": _sha256(manifest_path), "runtime_identity": identity, "target_groups": [g.get("group_id") for g in groups], "arms": list(ARMS), "seeds": list(SEEDS)}
    progress_path = output / "progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != identity_progress:
            raise RuntimeError("existing PRR progress identity differs")
    else:
        progress = {"schema": "prr-progress-v1", "status": "running", "identity": identity_progress, "completed": []}
        _atomic_json(progress_path, progress)
    if args.stage in ("engineering", "all"):
        engineering = _engineering(runtime, modules, base_state, cal, adapter, manifest, output)
    else:
        engineering = json.loads((output / "engineering_summary.json").read_text(encoding="utf-8"))
    if args.stage == "engineering":
        print(json.dumps({key: engineering.get(key) for key in ("schema", "status", "engineering_pass", "dev_opened")}, indent=2), flush=True)
        return
    if not engineering.get("engineering_pass"):
        progress.update({"status": "blocked", "result": "engineering_summary.json"})
        _atomic_json(progress_path, progress)
        print(json.dumps({"schema": "prr-final-v1", "status": "blocked", "reason": "engineering_pass_false", "dev_opened": False}, indent=2), flush=True)
        return
    dev_rows = _rows(base, manifest, manifest_path, "dev")
    dev_expanded = _expand(base, dev_rows, adapter, runtime["torch"], device, "dev")
    dev_bank = _make_split(runtime, modules, base_state, dev_expanded, adapter, clean, cal_wz)
    dev_bank.update({"wz": cal_wz, "action_mask": cal_mask, "rows": dev_expanded, "metadata": _record_metadata(dev_expanded)})
    cal_bank["metadata"] = _record_metadata(cal_expanded)
    summary = _formal(runtime, modules, base_state, {"rows": cal_expanded, "bank": cal_bank}, {"rows": dev_expanded, "bank": dev_bank}, adapter, manifest, output, progress)
    print(json.dumps({key: summary.get(key) for key in ("schema", "status", "formal", "raw_final", "test_opened")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
