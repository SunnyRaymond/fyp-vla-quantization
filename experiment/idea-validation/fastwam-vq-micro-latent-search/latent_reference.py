"""Loss, reference-contract, and matched-optimizer helpers for action+future VQ training."""
from __future__ import annotations

import math


def future_nmse_loss(torch, student, teacher):
    """NMSE over generated future frames only; keep the student's full graph connected."""
    if not isinstance(student, torch.Tensor) or not isinstance(teacher, torch.Tensor):
        raise TypeError("student and teacher futures must be torch tensors")
    if student.shape != teacher.shape or student.ndim != 5 or student.shape[2] < 2:
        raise ValueError("future tensors must have matching [B,C,T,H,W] shapes with T >= 2")
    if student.device != teacher.device:
        raise ValueError("student and teacher futures must be on the same device")
    if not student.is_floating_point() or not teacher.is_floating_point():
        raise ValueError("future tensors must be floating point")
    student_future = student.float()[:, :, 1:]
    teacher_future = teacher.float()[:, :, 1:]
    return (student_future - teacher_future).square().sum() / teacher_future.square().sum().clamp_min(1e-12)


def calibration_loss_scale(calibration_rows, calibration_ids, fraction=1.0):
    """Freeze normalization_c from the micro parent calibration contexts alone."""
    expected = list(calibration_ids)
    if not expected or [row.get("case_id") for row in calibration_rows] != expected:
        raise ValueError("Calibration loss scale requires exactly the frozen calibration IDs in order")
    if any(row.get("split") != "calibration" or row.get("seed_index") != 0 for row in calibration_rows):
        raise ValueError("Only seed-0 calibration contexts may determine the latent loss weight")
    action_values = [float(row["action_mse"]) for row in calibration_rows]
    future_values = [float(row["future_nmse"]) for row in calibration_rows]
    if (not all(math.isfinite(value) and value >= 0 for value in action_values)
            or not all(math.isfinite(value) and value > 0 for value in future_values)):
        raise ValueError("Calibration loss values must be finite with positive future NMSE")
    mean_action = sum(action_values) / len(action_values)
    mean_future = sum(future_values) / len(future_values)
    target_fraction = float(fraction)
    normalization_c = mean_action / mean_future
    weight = target_fraction * normalization_c
    if (not math.isfinite(normalization_c) or normalization_c <= 0
            or not math.isfinite(weight) or weight <= 0):
        raise ValueError("Calibration-derived normalization and latent weight must be finite and positive")
    return {
        "policy": "normalization_c = mean_8cal_micro_initial_action_mse / mean_8cal_micro_initial_future_nmse",
        "fraction": target_fraction,
        "calibration_ids": expected,
        "mean_cal_initial_action_mse": mean_action,
        "mean_cal_initial_future_nmse": mean_future,
        "normalization_c": normalization_c,
        "latent_weight": weight,
    }


def resume_reference_contract(reference_summary):
    """Return the frozen reference values used by each candidate and its resumes."""
    if reference_summary.get("status") != "complete":
        raise RuntimeError("Reference preparation is not complete")
    if (reference_summary.get("model_forward_count") != 16
            or reference_summary.get("teacher_forward_count") != 0
            or reference_summary.get("teacher_context_count") != 60
            or reference_summary.get("parent_context_count") != 16
            or reference_summary.get("calibration_context_count") != 8
            or reference_summary.get("selection_context_count") != 8
            or reference_summary.get("target_linears") != 614
            or reference_summary.get("self_check") != "passed"
            or reference_summary.get("selection_cache_parity") is not True):
        raise RuntimeError("Reference preparation does not meet the frozen 60/16/8/8 coverage and parity gates")
    loss_normalization = reference_summary.get("loss_normalization")
    if not isinstance(loss_normalization, dict):
        raise RuntimeError("Reference summary is missing loss_normalization")
    if loss_normalization.get("policy") != (
            "normalization_c = mean_8cal_micro_initial_action_mse / mean_8cal_micro_initial_future_nmse"):
        raise RuntimeError("Reference normalization policy differs from the frozen protocol")
    normalization_c = float(loss_normalization["normalization_c"])
    if not math.isfinite(normalization_c) or normalization_c <= 0:
        raise RuntimeError("Reference normalization_c must be finite and positive")
    return {
        "policy": loss_normalization["policy"],
        "calibration_ids": list(loss_normalization["calibration_ids"]),
        "normalization_c": normalization_c,
        "mean_cal_initial_action_mse": float(loss_normalization["mean_cal_initial_action_mse"]),
        "mean_cal_initial_future_nmse": float(loss_normalization["mean_cal_initial_future_nmse"]),
    }


def reset_adam_after_update1(optimizer, state):
    """Match the action-only baseline's single reset immediately after update one."""
    if state["updates_completed"] == 0 and not state["optimizer_reset_after_update1"]:
        optimizer.state.clear()
        state["optimizer_reset_after_update1"] = True
        return True
    if state["updates_completed"] > 0 and not state["optimizer_reset_after_update1"]:
        raise RuntimeError("Resume state passed update one without the matched Adam reset")
    return False
