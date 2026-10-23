"""Metrics and frozen cohort helpers for the latent diagnostic."""

import math
from collections import OrderedDict
from collections.abc import Mapping
from numbers import Real


def _check_tensor(torch, value, name, ndim):
    if not isinstance(value, torch.Tensor) or value.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-D torch tensor")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain finite floating-point values")


def _as_float_list(tensor):
    return [float(value) for value in tensor.detach().cpu().tolist()]


def _nmse(torch, error, reference):
    error_energy = error.square().mean(dim=(0, 1, 3, 4))
    reference_energy = reference.square().mean(dim=(0, 1, 3, 4)).clamp_min(1e-12)
    return error_energy / reference_energy


def latent_metrics(torch, z_quant, z_teacher):
    """Compare future latents and adjacent temporal changes, excluding frame 0 from future metrics."""
    _check_tensor(torch, z_quant, "z_quant", 5)
    _check_tensor(torch, z_teacher, "z_teacher", 5)
    if z_quant.shape != z_teacher.shape or z_quant.shape[2] < 2:
        raise ValueError("latent tensors must have matching [B,C,T,H,W] shapes with T >= 2")
    if z_quant.device != z_teacher.device:
        raise ValueError("latent tensors must be on the same device")

    quant = z_quant.to(dtype=torch.float32)
    teacher = z_teacher.to(dtype=torch.float32)
    error = quant - teacher
    future_error = error[:, :, 1:]
    future_teacher = teacher[:, :, 1:]
    future_rmse = future_error.square().mean(dim=(0, 1, 3, 4)).sqrt()
    future_global_rmse = future_error.square().mean().sqrt()
    future_global_nmse = future_error.square().sum() / future_teacher.square().sum().clamp_min(1e-12)

    quant_delta = quant[:, :, 1:] - quant[:, :, :-1]
    teacher_delta = teacher[:, :, 1:] - teacher[:, :, :-1]
    delta_error = quant_delta - teacher_delta
    delta_rmse = delta_error.square().mean(dim=(0, 1, 3, 4)).sqrt()
    delta_global_rmse = delta_error.square().mean().sqrt()
    delta_global_nmse = delta_error.square().sum() / teacher_delta.square().sum().clamp_min(1e-12)

    return {
        "latent_shape": [int(size) for size in z_quant.shape],
        "future_latent_rmse": float(future_global_rmse.item()),
        "future_latent_nmse": float(future_global_nmse.item()),
        "future_rmse": _as_float_list(future_rmse),
        "future_nmse": _as_float_list(_nmse(torch, future_error, future_teacher)),
        "temporal_delta_global_rmse": float(delta_global_rmse.item()),
        "temporal_delta_global_nmse": float(delta_global_nmse.item()),
        "temporal_delta_rmse": _as_float_list(delta_rmse),
        "temporal_delta_nmse": _as_float_list(_nmse(torch, delta_error, teacher_delta)),
        "clamp_firstframe_max_abs": float(error[:, :, 0].abs().max().item()),
    }


def action_metrics(torch, action, reference):
    """Return plain-float RMSEs for the frozen action horizons and dimensions."""
    _check_tensor(torch, action, "action", 2)
    _check_tensor(torch, reference, "reference", 2)
    if tuple(action.shape) != (32, 7) or tuple(reference.shape) != (32, 7):
        raise ValueError("action and reference must both have shape [32,7]")
    if action.device != reference.device:
        raise ValueError("action and reference must be on the same device")

    error = action.to(dtype=torch.float32) - reference.to(dtype=torch.float32)

    def rmse(values):
        return float(values.square().mean().sqrt().item())

    return {
        "first10motor_rmse": rmse(error[:10, :6]),
        "full32motor_rmse": rmse(error[:, :6]),
        "late22motor_rmse": rmse(error[10:, :6]),
        "first10gripper_rmse": rmse(error[:10, 6]),
    }


def aggregate(rows):
    """Average seeds within each case, then give every case equal weight per split and arm."""
    required = {"case_id", "seed_index", "split", "arm"}
    if not rows:
        return []

    grouped = OrderedDict()
    seen_contexts = set()
    for row in rows:
        if not isinstance(row, Mapping) or not required.issubset(row):
            raise ValueError("each row must include case_id, seed_index, split, and arm")
        try:
            context = (row["split"], row["arm"], row["case_id"], row["seed_index"])
            hash(context)
        except TypeError:
            raise ValueError("split, arm, case_id, and seed_index must be hashable") from None
        if context in seen_contexts:
            raise ValueError(f"duplicate context: {context!r}")
        seen_contexts.add(context)

        names = set()
        metrics = {}
        for name, value in row.items():
            if name in required or isinstance(value, (str, bytes, list, tuple, dict, set)) or value is None or isinstance(value, bool):
                continue
            if isinstance(value, Real):
                if not math.isfinite(float(value)):
                    raise ValueError(f"metric {name!r} must be finite")
                names.add(name)
                metrics[name] = float(value)
        if not names:
            raise ValueError("each row must include at least one numeric metric")

        group_key = (row["split"], row["arm"])
        group = grouped.setdefault(group_key, {"metric_names": names, "cases": OrderedDict()})
        if names != group["metric_names"]:
            raise ValueError(f"rows in split/arm group {group_key!r} must contain the same numeric metrics")
        case = group["cases"].setdefault(row["case_id"], [])
        case.append(metrics)

    result = []
    for (split, arm), group in grouped.items():
        cases = group["cases"]
        metric_names = group["metric_names"]
        per_case = []
        for case_id, context_metrics in cases.items():
            means = {
                name: sum(item[name] for item in context_metrics) / len(context_metrics)
                for name in metric_names
            }
            per_case.append({
                "case_id": case_id,
                "context_count": len(context_metrics),
                "metrics": means,
            })
        result.append({
            "split": split,
            "arm": arm,
            "case_count": len(cases),
            "context_count": sum(case["context_count"] for case in per_case),
            "metrics": {
                name: sum(case["metrics"][name] for case in per_case) / len(per_case)
                for name in metric_names
            },
            "per_case": per_case,
        })
    return result


def frozen_contexts(protocol):
    """Build the frozen 60-context cohort and parity-balanced two-shard assignment."""
    if not isinstance(protocol, Mapping):
        raise ValueError("protocol must be a mapping")

    cases_by_split = (
        ("calibration", protocol.get("calibration_cases"), [0], 8),
        ("selection", protocol.get("selection_cases"), [0, 1], 4),
        ("test", protocol.get("test_cases"), [0, 1], 22),
    )
    all_cases = set()
    for split, cases, seeds, expected_count in cases_by_split:
        if not isinstance(cases, list) or len(cases) != expected_count:
            raise ValueError(f"{split}_cases must contain exactly {expected_count} cases")
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"{split} seed list contains duplicates")
        for case_id in cases:
            try:
                duplicate = case_id in all_cases
                all_cases.add(case_id)
            except TypeError:
                raise ValueError("case IDs must be hashable") from None
            if duplicate:
                raise ValueError(f"case ID appears in more than one split: {case_id!r}")

    contexts = []
    ordinal = 0
    for split, cases, seeds, _ in cases_by_split:
        for case_id in cases:
            shard = ordinal % 2
            for seed_index in seeds:
                contexts.append({
                    "split": split,
                    "case_id": case_id,
                    "seed_index": seed_index,
                    "shard": shard,
                })
            ordinal += 1

    counts = [sum(context["shard"] == shard for context in contexts) for shard in (0, 1)]
    if len(contexts) != 60 or counts != [30, 30]:
        raise ValueError("frozen cohort must produce 60 contexts split evenly across two shards")
    return contexts


def self_check(torch):
    """Small tensor checks; call only from a script already guarded for PBS compute allocation."""
    teacher = torch.tensor([[[[[0.0]], [[2.0]], [[4.0]]]]])
    quant = torch.tensor([[[[[9.0]], [[3.0]], [[5.0]]]]])
    latent = latent_metrics(torch, quant, teacher)
    assert latent["future_rmse"] == [1.0, 1.0]
    assert latent["future_nmse"] == [0.25, 0.0625]
    assert latent["future_latent_rmse"] == 1.0
    assert abs(latent["future_latent_nmse"] - 0.1) < 1e-6
    assert latent["temporal_delta_rmse"] == [8.0, 0.0]
    assert latent["temporal_delta_nmse"] == [16.0, 0.0]
    assert abs(latent["temporal_delta_global_rmse"] - math.sqrt(32.0)) < 1e-6
    assert latent["temporal_delta_global_nmse"] == 8.0
    assert latent["clamp_firstframe_max_abs"] == 9.0
    action = torch.zeros((32, 7), dtype=torch.float32)
    reference = torch.ones((32, 7), dtype=torch.float32)
    result = action_metrics(torch, action, reference)
    assert set(result) == {
        "first10motor_rmse", "full32motor_rmse", "late22motor_rmse", "first10gripper_rmse"
    }
    assert all(value == 1.0 for value in result.values())
