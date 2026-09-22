#!/usr/bin/env python3
"""Local, no-HDF5/no-model freeze and pairing preflight."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRANSFER = HERE.parent
FREEZE = HERE / "LEWM_TEMPORAL_BALANCED_TRAIN_FREEZE.json"
PHASE2 = TRANSFER / "state-coverage" / "LEWM_STATE_COVERAGE_FREEZE.json"
MANIFEST = TRANSFER / "state-coverage" / "artifacts" / "24926383.pbs101" / "context_manifest_512.json"
REFERENCE = TRANSFER / "state-coverage" / "artifacts" / "24926383.pbs101" / "lewm_state_coverage_summary.json"
ROWS = TRANSFER / "state-coverage" / "artifacts" / "24926383.pbs101" / "prepared_512" / "prepared_rows.pt"


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def main() -> int:
    freeze = load(FREEZE)
    phase2 = load(PHASE2)
    reference = load(REFERENCE)
    assert freeze["status"] == "frozen"
    assert phase2["status"] == "frozen"
    assert freeze["training_selection"]["source"] == "Phase 2 formal 512 train manifest"
    assert freeze["training_selection"]["episode_ids_reused_verbatim"] is True
    assert all(int(seed) not in (20300907, 20300908, 20300917, 20300918) for seed in freeze["fresh_evaluation"]["action_prefix_seeds"])
    counts = {"early": sum(i % 3 == 0 for i in range(512)), "middle": sum(i % 3 == 1 for i in range(512)), "late": sum(i % 3 == 2 for i in range(512))}
    assert counts == {"early": 171, "middle": 171, "late": 170}
    assert freeze["fresh_evaluation"]["selection_algorithm"].endswith("valid[528:536]")
    assert freeze["scope_boundary"]["official_cem"] == "NOT_RUN_BY_SCOPE"
    assert freeze["scope_boundary"]["planner_viability"] == "NOT_RUN_BY_SCOPE"
    assert freeze["scope_boundary"]["closed_loop"] == "NOT_RUN_BY_SCOPE"
    assert reference["status"] == "PREDICTOR_LEVEL_COMPLETE"
    result = {
        "schema": "lewm-recurrent-student.temporal-balanced-train-preflight",
        "status": "PASS",
        "hdf5_opened": False,
        "model_loaded": False,
        "prepared_rows_read": False,
        "control_prepared_rows_path_present": ROWS.is_file(),
        "training_rows": 512,
        "control_anchor": 0,
        "treatment_anchor_rule": "ordinal mod3 -> early/middle/late",
        "balanced_counts": counts,
        "same_episode_ids": True,
        "manifest_read": False,
        "same_candidate_bank_by_ordinal": "runtime compute-node assertion",
        "fresh_selection": "valid[528:536] after excluding valid[:520] and valid[520:528]",
        "fresh_seeds": freeze["fresh_evaluation"]["action_prefix_seeds"],
        "scope": {"official_cem": "NOT_RUN_BY_SCOPE", "planner_viability": "NOT_RUN_BY_SCOPE", "closed_loop": "NOT_RUN_BY_SCOPE"},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
