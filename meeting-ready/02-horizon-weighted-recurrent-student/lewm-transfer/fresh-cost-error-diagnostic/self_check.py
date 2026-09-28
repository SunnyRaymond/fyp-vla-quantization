#!/usr/bin/env python3
"""One synthetic CPU check for the exact latent/score decomposition."""

import torch

from runner import decompose_bank


def main() -> None:
    torch.manual_seed(23)
    candidates, horizon, dims = 300, 5, 192
    target = torch.randn(candidates, horizon, dims) * 0.1
    teacher_terminal = target[:, -1, :]
    teacher_response = teacher_terminal - teacher_terminal.mean(dim=0, keepdim=True)
    orthogonal = torch.randn_like(teacher_response) * 0.02
    orthogonal -= orthogonal.mean(dim=0, keepdim=True)
    orthogonal -= (orthogonal * teacher_response).sum() / teacher_response.square().sum() * teacher_response
    goal = torch.randn(1, 1, dims) * 0.1
    prediction = target.clone()
    prediction[:, -1, :] += 0.03 + (-0.3) * teacher_response + orthogonal
    teacher_cost = (teacher_terminal - goal.reshape(1, dims)).square().sum(dim=-1)
    student_cost = (prediction[:, -1, :] - goal.reshape(1, dims)).square().sum(dim=-1)
    row = {
        "episode_id": 1, "context_id": "synthetic", "stratum": "early", "anchor": 0,
        "goal_emb": goal, "teacher_targets": target.unsqueeze(0), "teacher_objective": teacher_cost.unsqueeze(0),
    }
    result = decompose_bank(row, 0, 2, prediction, student_cost, torch)
    geometry = result["response_geometry"]
    assert abs(geometry["gain_along_teacher_response"] - 0.7) < 1e-6
    assert abs(geometry["parallel_error_relative_energy"] - 0.09) < 1e-6
    assert abs(geometry["parallel_error_relative_energy"] + geometry["orthogonal_error_relative_energy"] - geometry["total_response_error_relative_energy"]) < 1e-8
    assert result["score_terms"]["total_centered_rms_teacher_std"] >= 0
    assert len(result["counterfactuals"]) == 4
    print("fresh cost-error decomposition self-check: PASS")


if __name__ == "__main__":
    main()
