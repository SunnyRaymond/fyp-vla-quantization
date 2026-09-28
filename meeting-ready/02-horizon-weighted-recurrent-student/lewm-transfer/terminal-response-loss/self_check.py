#!/usr/bin/env python3
"""Small CPU-only assertions for per-bank response centering."""

from __future__ import annotations

import torch

from runner import METRICS, bank_centered_terminal_response_loss, paired_episode_deltas


def main() -> None:
    torch.manual_seed(7)
    banks, candidates, horizon, dims = 2, 4, 5, 192
    teacher = torch.randn(banks, candidates, horizon, dims, dtype=torch.float64)

    # A different candidate-constant offset in each bank must disappear after
    # centering, while centering the concatenated eight candidates would not.
    offsets = torch.randn(banks, dims, dtype=torch.float64)
    student = teacher + offsets[:, None, None, :]
    loss, per_bank = bank_centered_terminal_response_loss(
        student.reshape(banks * candidates, horizon, dims),
        teacher.reshape(banks * candidates, horizon, dims),
        banks,
        candidates,
        1e-6,
    )
    assert loss.item() < 1e-24
    assert per_bank.shape == (banks,)
    assert torch.all(per_bank < 1e-24)

    # Alter one candidate's terminal response; the normalized loss must rise.
    changed = student.clone()
    changed[1, 0, -1, 2] += 0.75
    changed_loss, changed_by_bank = bank_centered_terminal_response_loss(
        changed.reshape(banks * candidates, horizon, dims),
        teacher.reshape(banks * candidates, horizon, dims),
        banks,
        candidates,
        1e-6,
    )
    assert changed_loss.item() > loss.item()
    assert changed_by_bank[0].item() < 1e-24
    assert changed_by_bank[1].item() > 0

    # Candidate permutation inside each bank preserves the scalar objective.
    permutation = torch.tensor([2, 0, 3, 1])
    permuted_loss, _ = bank_centered_terminal_response_loss(
        changed[:, permutation].reshape(banks * candidates, horizon, dims),
        teacher[:, permutation].reshape(banks * candidates, horizon, dims),
        banks,
        candidates,
        1e-6,
    )
    assert torch.allclose(changed_loss, permuted_loss, rtol=1e-12, atol=1e-12)

    # Matched deltas and the difference of arm medians need not agree.
    control_blocks, treatment_blocks = [], []
    for episode in range(8):
        for block, (before, after) in enumerate(zip((0, 0, 0, 1, 1, 1), (0.01, 0.01, 0.01, 1.01, 0, 0))):
            key = f"episode={episode}:block={block}"
            common = {"episode_id": episode, "pairing_key": key, **{metric: 0.0 for metric in METRICS}}
            control_blocks.append({**common, "standardized_elite_regret": before})
            treatment_blocks.append({**common, "standardized_elite_regret": after})
    paired = paired_episode_deltas({"blocks": control_blocks}, {"blocks": treatment_blocks})
    assert all(abs(row["standardized_elite_regret"] - 0.01) < 1e-12 for row in paired["per_episode"])
    assert abs(paired["median_delta"]["standardized_elite_regret"] - 0.01) < 1e-12
    print("terminal response loss self-check: PASS")


if __name__ == "__main__":
    main()
