"""Pure contract checks; this file never loads the Wall model or starts PBS."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FREEZE = json.loads((HERE / "FREEZE.json").read_text(encoding="utf-8"))


def _runner():
    spec = importlib.util.spec_from_file_location("wall_stage_a", HERE / "run_stage_a.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_required_files_and_syntax():
    for name in ("FREEZE.json", "run_stage_a.py", "stage_a.pbs", "README.md"):
        assert (HERE / name).is_file(), name
    ast.parse((HERE / "run_stage_a.py").read_text(encoding="utf-8"))


def test_freeze_contract():
    assert FREEZE["schema"] == "dino-wm-wall.cem-residual-jepa.stage-a-freeze"
    assert FREEZE["data"]["candidate_shape"] == [300, 5, 10]
    assert FREEZE["data"]["train_episodes"] == [f"development:{i:03d}" for i in range(5)]
    assert FREEZE["data"]["heldout_episodes"] == [f"development:{i:03d}" for i in range(5, 8)]
    assert FREEZE["execution_constraints"]["pbs_compute_node_only"]
    assert FREEZE["execution_constraints"]["gpu_telemetry_seconds"] == 30


def test_frozen_workload_shape_when_present():
    workload = HERE.parent / "world-model-quantization" / "dino-wm-wall" / "artifacts" / "screen" / "artifacts" / "16176810.pbs101" / "evaluation" / "development" / "FP32" / "workload.pkl"
    if not workload.is_file():
        return
    runner = _runner()
    pools = runner._load_cases(workload, workload.parent)
    assert len(pools) == 26
    assert len([p for p in pools if p["episode_number"] <= 4]) == 18
    assert len([p for p in pools if p["episode_number"] >= 5]) == 8
    assert all(p["candidates"].shape == (300, 5, 10) for p in pools)
    assert all(p["mu"].shape == p["sigma"].shape == (5, 10) for p in pools)


def test_student_is_causal():
    import torch

    runner = _runner()
    torch.manual_seed(7)
    student = runner.CausalResidualStudent(6, 3, 2, hidden=16, heads=4)
    base = {"visual": torch.randn(1, 5, 2, 6), "proprio": torch.randn(1, 5, 3)}
    sigma = torch.ones(1, 5, 2)
    first = torch.randn(1, 5, 2); second = torch.randn(1, 5, 2)
    for cut in range(1, 5):
        a, b = first.clone(), first.clone(); b[:, cut:] = second[:, cut:]
        x, y = student(base, a, sigma), student(base, b, sigma)
        assert max(float((x[k][:, :cut] - y[k][:, :cut]).abs().max()) for k in x) <= 1e-6


if __name__ == "__main__":
    test_required_files_and_syntax(); test_freeze_contract(); test_frozen_workload_shape_when_present(); test_student_is_causal(); print("stage-a contract: PASS")
