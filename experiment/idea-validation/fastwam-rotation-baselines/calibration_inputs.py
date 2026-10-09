"""Render fixed initial observations from the four original LIBERO suites."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys

HERE = Path(__file__).resolve().parent
OLD_HELPERS = (
    HERE.parent / "fastwam-a4-phases12" / "prepare_inputs.py",
    Path("/scratch/users/ntu/yguo017/fastwam-a4-phases12-20261006/prepare_inputs.py"),
)
PROTOCOL = "fastwam-rotation-baselines-v1"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
REQUIRED_OBSERVATIONS = (
    "agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos",
    "robot0_eef_quat", "robot0_gripper_qpos",
)


def legacy_helper():
    path = next((candidate for candidate in OLD_HELPERS if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Trusted A4 prepare_inputs.py is unavailable: {OLD_HELPERS}")
    spec = importlib.util.spec_from_file_location("rotation_legacy_prepare_inputs", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load trusted A4 input helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def allocation_guard() -> None:
    helper = legacy_helper()
    helper.allocation_guard()


def case_specs() -> list[dict]:
    cases = []
    for suite in SUITES:
        for task_id in (0, 1, 2):
            case_id = len(cases)
            cases.append({
                "case_id": case_id,
                "domain": "original",
                "suite": suite,
                "task_id": task_id,
                "state_id": 0,
                "environment_seed": 810100000 + case_id * 1000,
                "sampler_seed": 910100000 + case_id * 1000,
                "split": "calibration" if task_id < 2 else "selection",
                "settling_steps": 30,
                "obsfile": f"inputs/case_{case_id:02d}.npz",
                "metafile": f"inputs/case_{case_id:02d}.json",
            })
    return cases


def validate_plan(plan: dict) -> None:
    rows = plan.get("inputs")
    if plan.get("protocol") != PROTOCOL or not isinstance(rows, list) or len(rows) != 12:
        raise ValueError("Expected the 12-input frozen original-LIBERO plan")
    if [row.get("case_id") for row in rows] != list(range(12)):
        raise ValueError("Input case IDs must be contiguous from 0 through 11")
    if Counter((row.get("suite"), row.get("task_id")) for row in rows) != Counter(
        (suite, task_id) for suite in SUITES for task_id in (0, 1, 2)
    ):
        raise ValueError("Plan must contain original task IDs 0, 1, and 2 for each suite")
    if any(row.get("domain") != "original" or row.get("state_id") != 0
           or row.get("settling_steps") != 30 for row in rows):
        raise ValueError("Every input must use original LIBERO state 0 and 30 settling steps")
    if Counter(row.get("split") for row in rows) != Counter({"calibration": 8, "selection": 4}):
        raise ValueError("Expected eight calibration inputs and four selection inputs")
    for row in rows:
        case_id = row["case_id"]
        if (row.get("environment_seed") != 810100000 + case_id * 1000
                or row.get("sampler_seed") != 910100000 + case_id * 1000):
            raise ValueError(f"Frozen seed mismatch in case {case_id}")
        if not all(isinstance(row.get(key), str) and row[key] for key in
                   ("task_name", "description", "obsfile", "metafile")):
            raise ValueError(f"Missing task or artifact metadata in case {case_id}")
    if plan.get("input_count") != 12 or plan.get("calibration_count") != 8 or plan.get("selection_count") != 4:
        raise ValueError("Plan summary counts do not match its inputs")


def _configure_original_imports(base_root: Path):
    source_root = base_root / "libero" / "libero"
    compat_root = base_root / "libero-compat"
    helper_root = base_root / "FastWAM" / "experiments" / "libero"
    prefix = [
        compat_root,
        source_root,
        base_root / "FastWAM",
        base_root / "FastWAM" / "src",
        helper_root,
    ]
    prefix_strings = [str(path.resolve()) for path in prefix]
    preferred = {str(Path(path).resolve()) for path in prefix}
    sys.path[:] = prefix_strings + [
        path for path in sys.path if str(Path(path or ".").resolve()) not in preferred
    ]
    os.environ["PYTHONPATH"] = os.pathsep.join(prefix_strings)
    os.environ["LIBERO_CONFIG_PATH"] = str((base_root / "libero-config").resolve())
    os.environ["ROOT"] = str(base_root.resolve())
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    return source_root.resolve(), compat_root.resolve(), helper_root.resolve()


def render_worker(out: Path, base_root: Path) -> None:
    allocation_guard()
    source_root, compat_root, helper_root = _configure_original_imports(base_root)
    import numpy as np
    if "float_" not in np.__dict__:
        np.float_ = np.float64
    if np.float_ is not np.float64:
        raise RuntimeError("LIBERO compatibility requires numpy.float_ to alias float64")
    import torch
    import libero
    import libero.libero as libero_core
    from libero.libero import benchmark, envs as envs_module, get_libero_path
    from experiments.libero import libero_utils as ev

    helper = legacy_helper()
    provenance = {
        "libero_outer": str(helper.source_file(libero)),
        "libero_core": str(helper.source_file(libero_core)),
        "benchmark": str(helper.source_file(benchmark)),
        "envs": str(helper.source_file(envs_module)),
        "libero_utils": str(helper.source_file(ev)),
        "compat_prefix": str(compat_root),
        "source_root": str(source_root),
    }
    helper.validate_import_provenance(provenance, source_root, compat_root, helper_root)
    expected_state_root = (base_root / "libero" / "libero" / "init_files").resolve()
    classes = benchmark.get_benchmark_dict()
    out.mkdir(parents=True, exist_ok=False)
    inputs_dir = out / "inputs"
    inputs_dir.mkdir()
    plan_rows = []
    for case in case_specs():
        suite_name = case["suite"]
        suite = classes[suite_name]()
        task_id = case["task_id"]
        task = suite.get_task(task_id)
        task_name = str(task.name)
        description = str(task.language)
        random.seed(case["environment_seed"])
        np.random.seed(case["environment_seed"] % (2**32))
        env = ev.OffScreenRenderEnv(
            bddl_file_name=str(suite.get_task_bddl_file_path(task_id)),
            camera_heights=int(ev.LIBERO_ENV_RESOLUTION),
            camera_widths=int(ev.LIBERO_ENV_RESOLUTION),
        )
        try:
            env.seed(case["environment_seed"])
            env.reset()
            states, state_path = helper.load_original_init_states(
                torch, get_libero_path, suite, task_id, expected_state_root
            )
            if len(states) <= 0:
                raise RuntimeError(f"No official initial state: suite={suite_name} task_id={task_id}")
            obs = env.set_init_state(states[0])
            for _ in range(case["settling_steps"]):
                obs, _reward, done, _info = env.step(ev.get_libero_dummy_action())
                if done:
                    raise RuntimeError(f"Environment terminated while settling case {case['case_id']}")

            arrays = {}
            for key, value in obs.items():
                array = np.asarray(value)
                if array.dtype.kind in "biufc":
                    arrays[str(key)] = array
            missing = [key for key in REQUIRED_OBSERVATIONS if key not in arrays]
            if missing:
                raise RuntimeError(f"Required numeric observations are missing: {missing}")
            resolution = int(ev.LIBERO_ENV_RESOLUTION)
            for key in REQUIRED_OBSERVATIONS[:2]:
                if arrays[key].shape != (resolution, resolution, 3):
                    raise RuntimeError(f"Unexpected {key} shape: {arrays[key].shape}")

            obs_path = out / case["obsfile"]
            with obs_path.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            metadata = {
                **case,
                "task_name": task_name,
                "description": description,
                "statepath": str(state_path),
                "dataset_import_paths": {
                    **provenance,
                    "state_loader": f"{Path(helper.__file__).resolve()}:load_original_init_states",
                },
                "observation_arrays": {
                    key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for key, value in arrays.items()
                },
                "evaluation_episodes": 0,
            }
            helper.write_json_atomic(out / case["metafile"], metadata)
            plan_rows.append({**case, "task_name": task_name, "description": description,
                              "statepath": str(state_path), "dataset_import_paths": metadata["dataset_import_paths"]})
            print(f"CALIBRATION_INPUT_RENDERED case_id={case['case_id']} suite={suite_name} task_id={task_id} split={case['split']}", flush=True)
        finally:
            env.close()

    plan = {
        "protocol": PROTOCOL,
        "source": "original LIBERO benchmark and its official init_states files",
        "suite_order": list(SUITES),
        "case_design": "task IDs 0 and 1 per suite are calibration; task ID 2 is selection",
        "environment_seed_rule": "810100000 + case_id * 1000",
        "sampler_seed_rule": "910100000 + case_id * 1000",
        "state_id": 0,
        "settling_steps": 30,
        "inputs": plan_rows,
        "input_count": 12,
        "calibration_count": 8,
        "selection_count": 4,
        "episode_count": 0,
    }
    validate_plan(plan)
    helper.write_json_atomic(out / "plan.json", plan)


def control_test() -> None:
    rows = case_specs()
    plan = {
        "protocol": PROTOCOL,
        "inputs": [{**row, "task_name": f"{row['suite']}_task_{row['task_id']}",
                    "description": f"synthetic task {row['task_id']}"} for row in rows],
        "input_count": 12,
        "calibration_count": 8,
        "selection_count": 4,
    }
    validate_plan(plan)
    assert [row["case_id"] for row in rows] == list(range(12))
    assert [row["sampler_seed"] for row in rows] == [910100000 + i * 1000 for i in range(12)]
    print("CALIBRATION_INPUT_CONTROL_TEST_OK suites=4 calibration=8 selection=4")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--base-root", type=Path, default=Path(os.environ.get("BASE", "/scratch/users/ntu/yguo017/fastwam-smoke")))
    parser.add_argument("--render-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--control-test", action="store_true")
    args = parser.parse_args()
    if args.control_test:
        control_test()
        return 0
    if args.out is None:
        parser.error("--out is required")
    out = args.out.resolve()
    base_root = args.base_root.resolve()
    if args.render_worker:
        render_worker(out, base_root)
        return 0

    allocation_guard()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing calibration inputs: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    job_id = os.environ["PBS_JOBID"]
    staging = out.with_name(f".{out.name}.staging-{job_id.replace('/', '_')}")
    if staging.exists():
        raise FileExistsError(f"Refusing to overwrite existing staging output: {staging}")
    source_root = base_root / "libero" / "libero"
    compat_root = base_root / "libero-compat"
    prefix = [compat_root, source_root, base_root / "FastWAM", base_root / "FastWAM" / "src",
              base_root / "FastWAM" / "experiments" / "libero"]
    command = [sys.executable, str(Path(__file__).resolve()), "--render-worker",
               "--out", str(staging), "--base-root", str(base_root)]
    env = os.environ.copy()
    # Keep pip --target dependencies while selecting original LIBERO first.
    dependencies = [path for path in env.get("PYTHONPATH", "").split(os.pathsep)
                    if path and Path(path).name == "deps"]
    env["PYTHONPATH"] = os.pathsep.join([*(str(path.resolve()) for path in prefix), *dependencies])
    subprocess.run(command, check=True, env=env)
    os.replace(staging, out)
    print(f"CALIBRATION_INPUTS_COMPLETE path={out} cases=12 calibration=8 selection=4", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
