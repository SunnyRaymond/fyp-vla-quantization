"""Derive shortlist M and one-step rank-30/31 swap sensitivity from Stage A."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import torch


FIXED_M = (60, 90, 120, 150, 180, 210, 240, 270)


def describe(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
        "p90_nearest_rank": ordered[max(0, (9 * len(ordered) + 9) // 10 - 1)],
    }


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().item())


def main() -> None:
    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    calls: list[dict[str, object]] = []

    for record in records:
        for row in record["rounds"]:
            cheap = [float(value) for value in row["J_tilde"]]
            full = [float(value) for value in row["J_full"]]
            cheap_order = sorted(range(len(cheap)), key=lambda index: (cheap[index], index))
            full_order = sorted(range(len(full)), key=lambda index: (full[index], index))
            cheap_rank = {candidate: rank + 1 for rank, candidate in enumerate(cheap_order)}
            elite_ranks = sorted(cheap_rank[candidate] for candidate in full_order[:30])

            mu = torch.tensor(row["input_mu"], dtype=torch.float32)
            sigma = torch.tensor(row["input_sigma"], dtype=torch.float32)
            torch.manual_seed(int(row["candidate_population_seed"]))
            actions = torch.randn(300, *mu.shape, dtype=torch.float32) * sigma + mu
            actions[0] = mu

            elite = full_order[:30]
            swapped = full_order[:29] + [full_order[30]]
            base_actions = actions[elite]
            swap_actions = actions[swapped]
            base_mu = base_actions.mean(dim=0)
            swap_mu = swap_actions.mean(dim=0)
            base_sigma = base_actions.std(dim=0)
            swap_sigma = swap_actions.std(dim=0)
            saved_mu = torch.tensor(row["output_mu"], dtype=torch.float32)
            saved_sigma = torch.tensor(row["output_sigma"], dtype=torch.float32)

            calls.append(
                {
                    "observation_id": record["observation_id"],
                    "round_index": int(row["round_index"]),
                    "M_for_30_of_30": elite_ranks[-1],
                    "M_for_29_of_30": elite_ranks[-2],
                    "M_for_28_of_30": elite_ranks[-3],
                    "recall_at_M": {
                        str(m): sum(rank <= m for rank in elite_ranks) / 30.0 for m in FIXED_M
                    },
                    "rank30_rank31_objective_gap": full[full_order[30]] - full[full_order[29]],
                    "swap30_31_mu_max_abs": max_abs(base_mu, swap_mu),
                    "swap30_31_first_action_max_abs": max_abs(base_mu[0], swap_mu[0]),
                    "swap30_31_sigma_max_abs": max_abs(base_sigma, swap_sigma),
                    "reconstruction_output_mu_max_abs": max_abs(base_mu, saved_mu),
                    "reconstruction_output_sigma_max_abs": max_abs(base_sigma, saved_sigma),
                }
            )

    per_episode: list[dict[str, object]] = []
    for observation_id in [record["observation_id"] for record in records]:
        rows = [row for row in calls if row["observation_id"] == observation_id]
        per_episode.append(
            {
                "observation_id": observation_id,
                "M_for_30_of_30_across_all_10_rounds": max(int(row["M_for_30_of_30"]) for row in rows),
                "M_for_30_of_30_round_median": statistics.median(int(row["M_for_30_of_30"]) for row in rows),
                "M_for_29_of_30_across_all_10_rounds": max(int(row["M_for_29_of_30"]) for row in rows),
                "M_for_29_of_30_round_median": statistics.median(int(row["M_for_29_of_30"]) for row in rows),
                "recall_at_M_mean_over_rounds": {
                    str(m): statistics.fmean(float(row["recall_at_M"][str(m)]) for row in rows)
                    for m in FIXED_M
                },
            }
        )

    scalar_metrics = [
        "M_for_30_of_30",
        "M_for_29_of_30",
        "M_for_28_of_30",
        "rank30_rank31_objective_gap",
        "swap30_31_mu_max_abs",
        "swap30_31_first_action_max_abs",
        "swap30_31_sigma_max_abs",
        "reconstruction_output_mu_max_abs",
        "reconstruction_output_sigma_max_abs",
    ]
    output = {
        "source": str(source),
        "n_observations": len(records),
        "n_nested_rounds": len(calls),
        "interpretation_boundary": "M is descriptive for cheap_L4 on baseline full-CEM candidate populations. Swap sensitivity is one-round only and does not establish chained or closed-loop impact.",
        "aggregate": {name: describe([float(row[name]) for row in calls]) for name in scalar_metrics},
        "recall_at_M_all_calls": {
            str(m): describe([float(row["recall_at_M"][str(m)]) for row in calls]) for m in FIXED_M
        },
        "per_episode": per_episode,
        "calls": calls,
    }
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
