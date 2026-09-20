"""Pure static contract checks; no torch/model/GPU/PBS execution."""
from __future__ import annotations

import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FREEZE = json.loads((HERE / "FREEZE.json").read_text(encoding="utf-8"))
SOURCE = (HERE / "run_stage_a.py").read_text(encoding="utf-8")


def test_freeze_shape() -> None:
    assert FREEZE["status"] == "frozen"
    assert FREEZE["scope"]["manifest_schema"] == "jepa-action-prefix-compiler.query-coverage-manifest"
    assert FREEZE["evaluation"]["heldout_contexts"] == 8
    assert FREEZE["evaluation"]["fresh_seeds"] == 2
    assert FREEZE["evaluation"]["candidates_per_block"] == 300
    assert FREEZE["evaluation"]["block_unit"].startswith("context x seed")
    assert FREEZE["evaluation"]["arms"] == ["mean_only_base_repeat", "trained_residual", "shuffled_epsilon_negative_control"]


def test_frozen_cem_and_seeds() -> None:
    cem = FREEZE["training"]["refined_cem"]
    assert (cem["candidates_M"], cem["elite_K"]) == (64, 8)
    assert FREEZE["training"]["distributions"] == ["initial", "refined"]
    assert FREEZE["randomness"]["seeds_fixed_before_output"] is True
    assert len(FREEZE["randomness"]["heldout_seeds"]) == 2


def test_runner_is_static_and_guarded() -> None:
    tree = ast.parse(SOURCE)
    names = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    assert {"CausalResidualStudent", "_mean_stats", "_evaluate", "_latency", "main"} <= names
    for token in ("PBS_JOBID", "PBS_NODEFILE", "cuda.synchronize", "jepa-action-prefix-compiler"):
        assert token in SOURCE
    assert "ssh " not in SOURCE.lower()
    assert "sha256" not in SOURCE.lower()


def test_pbs_guard_and_telemetry() -> None:
    source = (HERE / "stage_a.pbs").read_text(encoding="utf-8")
    for token in ("PBS_JOBID", "PBS_NODEFILE", "nvidia-smi", "sleep 30", "login", "head", "submit"):
        assert token in source


if __name__ == "__main__":
    for fn in (test_freeze_shape, test_frozen_cem_and_seeds, test_runner_is_static_and_guarded, test_pbs_guard_and_telemetry):
        fn()
    print("cem-residual-jepa static contract: PASS")
