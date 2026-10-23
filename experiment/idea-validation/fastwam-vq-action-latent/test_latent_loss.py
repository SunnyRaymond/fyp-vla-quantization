"""PBS-only checks for future-frame loss, frozen calibration scale, and matched Adam reset."""
from __future__ import annotations

from types import SimpleNamespace


def self_check(torch, protocol=None):
    from latent_reference import (calibration_loss_scale, future_nmse_loss,
                                  reset_adam_after_update1, resume_reference_contract)

    teacher = torch.tensor([100.0, 2.0, 4.0]).reshape(1, 1, 3, 1, 1)
    student_base = torch.tensor([1000.0, 3.0, 6.0], requires_grad=True)
    student = student_base.reshape(1, 1, 3, 1, 1)
    loss = future_nmse_loss(torch, student, teacher)
    assert loss.requires_grad and torch.allclose(loss, torch.tensor(0.25))
    loss.backward()
    gradient = student_base.grad
    assert gradient is not None
    assert gradient[0].abs() == 0
    assert gradient[1:].abs().sum() > 0

    rows = [
        {"case_id": 0, "seed_index": 0, "split": "calibration", "action_mse": 2.0, "future_nmse": 4.0},
        {"case_id": 2, "seed_index": 0, "split": "calibration", "action_mse": 6.0, "future_nmse": 2.0},
    ]
    scale = calibration_loss_scale(rows, [0, 2], 0.1)
    assert scale["latent_weight"] == 0.1 * 4.0 / 3.0
    try:
        calibration_loss_scale(rows + [
            {"case_id": 1004, "seed_index": 0, "split": "test", "action_mse": 1e9, "future_nmse": 1e-9}
        ], [0, 2], 0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("Selection/test rows must not enter calibration weight")

    summary = {"status": "complete", "model_forward_count": 0, "context_count": 60,
               "calibration_context_count": 8, "latent_weight_policy": scale["policy"],
               "latent_loss_fraction": 0.1, "calibration_ids": [0, 2], "latent_weight": scale["latent_weight"]}
    contract = resume_reference_contract(summary)
    changed = resume_reference_contract({**summary, "latent_weight": scale["latent_weight"] * 2})
    assert contract != changed
    if protocol is not None:
        changed_loss = {
            **protocol["latent_loss"], "latent_weight": protocol["latent_loss"]["latent_weight"] * 2,
        }
        assert protocol["latent_loss"] != changed_loss

    optimizer = SimpleNamespace(state={"first_moment": object()})
    state = {"updates_completed": 0, "optimizer_reset_after_update1": False}
    assert reset_adam_after_update1(optimizer, state)
    assert optimizer.state == {} and state["optimizer_reset_after_update1"]
    state["updates_completed"] = 1
    optimizer.state["second_update_moment"] = object()
    assert not reset_adam_after_update1(optimizer, state)
    assert "second_update_moment" in optimizer.state
    print("ACTION_LATENT_LOSS_SELF_CHECK_PASSED", flush=True)
