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


def calibration_loss_scale(calibration_rows, calibration_ids, fraction):
    """Freeze w from the declared calibration rows alone, in their exact order."""
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
    weight = target_fraction * mean_action / mean_future
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("Calibration-derived latent weight must be finite and positive")
    return {
        "policy": "latent_weight = latent_loss_fraction * mean_cal_initial_action_mse / mean_cal_initial_future_nmse",
        "fraction": target_fraction,
        "calibration_ids": expected,
        "mean_cal_initial_action_mse": mean_action,
        "mean_cal_initial_future_nmse": mean_future,
        "latent_weight": weight,
    }


def resume_reference_contract(reference_summary):
    """Return the immutable reference values that belong in every resume contract."""
    if reference_summary.get("status") != "complete":
        raise RuntimeError("Reference preparation is not complete")
    if reference_summary.get("model_forward_count") != 0:
        raise RuntimeError("Reference preparation must perform zero model forwards")
    if reference_summary.get("context_count") != 60 or reference_summary.get("calibration_context_count") != 8:
        raise RuntimeError("Reference preparation does not cover the frozen 60/8 contexts")
    return {
        "policy": reference_summary["latent_weight_policy"],
        "fraction": float(reference_summary["latent_loss_fraction"]),
        "calibration_ids": list(reference_summary["calibration_ids"]),
        "latent_weight": float(reference_summary["latent_weight"]),
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
