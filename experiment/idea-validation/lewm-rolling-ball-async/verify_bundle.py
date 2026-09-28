"""Compare full and compact LeWM+CEM planner inputs and one seeded solve."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import traceback
from typing import Any


DEFAULT_TASK_ROOT = Path("/scratch/users/ntu/yguo017/lewm-rolling-ball-async")
DEFAULT_BUNDLE_ROOT = DEFAULT_TASK_ROOT / "bundles" / "rolling-ball-lewm-epoch100"
DEFAULT_FULL_CHECKPOINT = DEFAULT_TASK_ROOT / "runs" / "25578999.pbs101" / "last.ckpt"
ATOL = 1e-6
RTOL = 1e-5


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--full-checkpoint", type=Path, default=DEFAULT_FULL_CHECKPOINT)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--bundle-job", help="Successful CPU export PBS job ID recorded in BUNDLE.json")
    parser.add_argument("--output", type=Path, help="PBS output path for verify_bundle.json")
    parser.add_argument("--self-check", action="store_true", help="Validate CLI/path contract with stdlib only")
    return parser


def stdlib_self_check() -> dict[str, Any]:
    args = make_parser().parse_args(["--self-check"])
    assert args.full_root == DEFAULT_TASK_ROOT
    assert args.full_checkpoint == DEFAULT_FULL_CHECKPOINT
    assert args.bundle_root == DEFAULT_BUNDLE_ROOT
    assert ATOL == 1e-6 and RTOL == 1e-5
    return {
        "status": "PASS",
        "checks": ["default_full_checkpoint", "fixed_bundle_root", "allclose_tolerances"],
        "full_checkpoint": str(args.full_checkpoint),
        "bundle_root": str(args.bundle_root),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _clear_pinned_runtime_modules() -> None:
    """Let each build_runtime import pinned modules from its own root."""
    names = [
        name for name in sys.modules
        if name == "jepa" or name == "utils" or name == "stable_worldmodel"
        or name.startswith("stable_worldmodel.")
    ]
    for name in names:
        del sys.modules[name]


def _module_paths(root: Path) -> dict[str, str]:
    importlib.import_module("jepa")
    importlib.import_module("utils")
    importlib.import_module("stable_worldmodel.solver.cem")
    paths = {
        "jepa": str(Path(sys.modules["jepa"].__file__).resolve()),
        "lewm_utils": str(Path(sys.modules["utils"].__file__).resolve()),
        "cem": str(Path(sys.modules["stable_worldmodel.solver.cem"].__file__).resolve()),
    }
    expected = {
        "jepa": (root / "upstream_lewm").resolve(),
        "lewm_utils": (root / "upstream_lewm").resolve(),
        "cem": (root / "upstream_stablewm").resolve(),
    }
    for key, source_root in expected.items():
        if not Path(paths[key]).is_relative_to(source_root):
            raise RuntimeError(f"{key} imported outside requested pinned root: {paths[key]}")
    return paths


def _same_context(full: dict[str, Any], compact: dict[str, Any], np) -> dict[str, Any]:
    checks = {
        "validation_provenance": full.get("validation_provenance") == compact.get("validation_provenance"),
        "split": full.get("split") == compact.get("split"),
        "goal_refs": full.get("goal_refs") == compact.get("goal_refs"),
        "source_identity": full.get("source") == compact.get("source"),
        "checkpoint_source_identity": full.get("checkpoint_source") == compact.get("checkpoint_source"),
        "checkpoint_epoch": full.get("checkpoint_epoch") == compact.get("checkpoint_epoch") == 100,
        "goal_bank_episode_count": len(full.get("goal_refs", [])) == len(compact.get("goal_refs", [])) == 180,
        "frame_equal": np.array_equal(full.get("frame"), compact.get("frame")),
        "state_available": full.get("state") is not None and compact.get("state") is not None,
        "state_equal": (
            full.get("state") is not None
            and compact.get("state") is not None
            and np.array_equal(full["state"], compact["state"])
        ),
    }
    if not all(checks.values()):
        raise RuntimeError("Full and compact inputs differ: " + json.dumps(checks, sort_keys=True))
    for key in ("validation_episode_id", "validation_dataset_row", "validation_frame_index"):
        if full.get(key) != compact.get(key):
            raise RuntimeError(f"Validation identity differs at {key}")
    if full.get("validation_provenance", {}).get("camera") != "observation.images.fixed_cam":
        raise RuntimeError("Validation probe is not from observation.images.fixed_cam")
    return {
        "checks": checks,
        "validation_episode_id": full["validation_episode_id"],
        "validation_dataset_row": full["validation_dataset_row"],
        "validation_frame_index": full["validation_frame_index"],
        "goal_bank_episode_count": 180,
        "source": full["source"],
    }


def _assert_allclose(name: str, left, right, torch) -> dict[str, Any]:
    if tuple(left.shape) != tuple(right.shape):
        raise RuntimeError(f"{name} shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}")
    if not torch.isfinite(left).all().item() or not torch.isfinite(right).all().item():
        raise RuntimeError(f"{name} contains NaN or infinity")
    close = torch.allclose(left, right, rtol=RTOL, atol=ATOL)
    max_abs = float((left - right).abs().max().item())
    if not close:
        raise RuntimeError(f"{name} is not allclose: max_abs_diff={max_abs}")
    return {"shape": list(left.shape), "allclose": True, "max_abs_diff": max_abs}


def run_comparison(args, job_id: str, host: str, policy_server, torch, np) -> dict[str, Any]:
    bundle_job = str(args.bundle_job)
    export_run = args.full_root / "runs" / bundle_job
    export_report = json.loads((export_run / "EXPORT_SUMMARY.json").read_text(encoding="utf-8"))
    if export_report.get("status") != "PASS" or export_report.get("export_job_id") != bundle_job:
        raise RuntimeError("CPU export report does not verify the requested bundle job")
    for name in ("runner_exit_status.txt", "wrapper_exit_status.txt"):
        if (export_run / name).read_text(encoding="utf-8").strip() != "0":
            raise RuntimeError(f"CPU export {name} is not zero")
    bundle_manifest = json.loads((args.bundle_root / "BUNDLE.json").read_text(encoding="utf-8"))
    if str(bundle_manifest.get("export_job_id", "")) != bundle_job:
        raise RuntimeError(
            f"BUNDLE_JOB {bundle_job!r} does not match manifest export_job_id "
            f"{bundle_manifest.get('export_job_id')!r}"
        )

    _clear_pinned_runtime_modules()
    full_runtime, full_context = policy_server.build_runtime(
        args.full_root, args.full_checkpoint, torch, np
    )
    full_modules = _module_paths(args.full_root)

    _clear_pinned_runtime_modules()
    compact_runtime, compact_context = policy_server.build_runtime(
        args.bundle_root,
        args.bundle_root / "checkpoint" / "last.ckpt",
        torch,
        np,
        compact=True,
    )
    compact_modules = _module_paths(args.bundle_root)

    context_check = _same_context(full_context, compact_context, np)
    if full_modules["jepa"] == compact_modules["jepa"] or full_modules["cem"] == compact_modules["cem"]:
        raise RuntimeError("Runtime module import roots were not isolated between full and compact builds")

    if full_runtime.info() != compact_runtime.info():
        raise RuntimeError("Full and compact /info payloads differ")
    full_request = policy_server._native_smoke_request(full_context)
    compact_request = policy_server._native_smoke_request(compact_context)
    if full_request != compact_request:
        raise RuntimeError("Full and compact native validation requests differ")

    full_reset = full_runtime.reset([0])
    compact_reset = compact_runtime.reset([0])
    if full_reset != {"status": "ok"} or compact_reset != {"status": "ok"}:
        raise RuntimeError("Episode reset did not return the native status")
    full_response = full_runtime.predict(full_request)
    compact_response = compact_runtime.predict(compact_request)

    full_goal = full_runtime.cost_model.goal_bank.detach()
    compact_goal = compact_runtime.cost_model.goal_bank.detach()
    goal_check = _assert_allclose("goal_bank", full_goal, compact_goal, torch)
    full_actions = torch.as_tensor(full_response["actions"], dtype=torch.float32, device="cuda")
    compact_actions = torch.as_tensor(compact_response["actions"], dtype=torch.float32, device="cuda")
    if tuple(full_actions.shape) != (1, 1, 8) or tuple(compact_actions.shape) != (1, 1, 8):
        raise RuntimeError("Planner actions must both have shape [1,1,8]")
    action_check = _assert_allclose("returned_action", full_actions, compact_actions, torch)
    full_cost = full_runtime.cost_model.last_cost.detach()
    compact_cost = compact_runtime.cost_model.last_cost.detach()
    if tuple(full_cost.shape) != (1, 300) or tuple(compact_cost.shape) != (1, 300):
        raise RuntimeError("Final CEM costs must both have shape [1,300]")
    cost_check = _assert_allclose("final_cem_cost", full_cost, compact_cost, torch)

    return {
        "status": "PASS_full_vs_compact_planner_parity",
        "job_id": job_id,
        "host": host,
        "bundle_export_job_id": bundle_job,
        "full_root": str(args.full_root),
        "full_checkpoint": str(args.full_checkpoint),
        "bundle_root": str(args.bundle_root),
        "compact_checkpoint": str(args.bundle_root / "checkpoint" / "last.ckpt"),
        "checkpoint_epoch": 100,
        "module_paths": {"full": full_modules, "compact": compact_modules},
        "validation": context_check,
        "goal_bank": goal_check,
        "action": action_check,
        "final_cem_cost": {
            **cost_check,
            "min": float(full_cost.min().item()),
            "max": float(full_cost.max().item()),
        },
        "cem_latency_s_diagnostic_only": {
            "full": float(full_response["cem_latency_s"]),
            "compact": float(compact_response["cem_latency_s"]),
        },
        "cem_config": {
            "num_samples": 300, "topk": 30, "n_steps": 30, "horizon": 5,
            "action_block": 1, "receding_horizon": 1, "dtype": "float32", "seed": 1234,
        },
        "closed_loop_evaluated": False,
        "task_success": None,
        "observation_age_s": None,
        "rtf": None,
    }


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(stdlib_self_check(), sort_keys=True))
        return 0
    if not args.bundle_job or not args.output:
        parser.error("--bundle-job and --output are required outside --self-check")

    # policy_server only has stdlib imports at module import time; its real PBS
    # and hostname guard runs before NumPy or Torch are imported below.
    policy_server = importlib.import_module("policy_server")
    job_id, host = policy_server.require_gpu_pbs()
    import numpy as np
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is unavailable inside the guarded PBS allocation")
    report: dict[str, Any] = {
        "status": "FAIL",
        "job_id": job_id,
        "host": host,
        "bundle_export_job_id": str(args.bundle_job),
        "closed_loop_evaluated": False,
        "task_success": None,
        "observation_age_s": None,
        "rtf": None,
    }
    try:
        report = run_comparison(args, job_id, host, policy_server, torch, np)
        _write_json(args.output, report)
        print(json.dumps(report, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(args.output, report)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
