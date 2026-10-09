#!/usr/bin/env python3
"""Freeze fresh fixed LIBERO observations for action-aware VQ diagnostics.

The coordinator reads only the frozen pilot manifest and literal task maps.
It launches one fresh renderer process per domain so the two ``libero`` trees
cannot contaminate one another. Run the coordinator and both renderers only in
the approved CPU PBS allocation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import socket
import subprocess
import sys
from pathlib import Path


PROTOCOL = "fastwam-vq-action-aware-v1"
SELECTION_SEED = 20261009
SPATIAL_SUITE = "libero_spatial"
PLUS_DIMENSIONS = ("camera", "robot", "layout", "language", "light", "background")
OFFICIAL_DIMENSIONS = {
    "camera": "camera_viewpoints",
    "robot": "robot_initial_states",
    "layout": "objects_layout",
    "language": "language_instructions",
    "light": "light_conditions",
    "background": "background_textures",
}
REQUIRED_OBSERVATIONS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)
DEFAULT_BASE = Path("/scratch/users/ntu/yguo017/fastwam-smoke")
DEFAULT_PILOT = Path("/scratch/users/ntu/yguo017/fastwam-libero-plus-pilot-20261005")


def allocation_guard() -> None:
    """Refuse manifest reads or rendering outside this job's compute nodes."""
    job_id = os.environ.get("PBS_JOBID")
    nodefile = os.environ.get("PBS_NODEFILE")
    if not job_id or not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("Input preparation requires PBS_JOBID and a readable PBS_NODEFILE")
    host = socket.gethostname().split(".", 1)[0]
    allocated = {line.split(".", 1)[0] for line in Path(nodefile).read_text().split()}
    if "login" in host.lower() or host not in allocated:
        raise RuntimeError(f"Input preparation requires an allocated compute host; host={host}")


def load_pilot_helpers(pilot_root: Path):
    path = pilot_root / "prepare_manifest.py"
    if not path.is_file():
        raise FileNotFoundError(f"Pilot manifest helper is missing: {path}")
    spec = importlib.util.spec_from_file_location("fastwam_pilot_prepare_manifest", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load pilot manifest helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_plus_cases(variants: list[dict], seed: int, excluded_variant_ids: set[str]) -> list[dict]:
    """Choose two distinct-task Spatial variants per dimension, excluding prior inputs."""
    rng = random.Random(seed)
    selected: list[dict] = []
    for dimension in PLUS_DIMENSIONS:
        candidates = [
            (index, row) for index, row in enumerate(variants)
            if row.get("suite") == SPATIAL_SUITE
            and row.get("dimension") == OFFICIAL_DIMENSIONS[dimension]
            and isinstance(row.get("variant_id"), str)
            and row["variant_id"] not in excluded_variant_ids
        ]
        rng.shuffle(candidates)
        chosen: list[tuple[int, dict]] = []
        seen_original_tasks: set[str] = set()
        for index, row in candidates:
            original_task = row.get("original_task")
            if not isinstance(original_task, str) or original_task in seen_original_tasks:
                continue
            chosen.append((index, row))
            seen_original_tasks.add(original_task)
            if len(chosen) == 2:
                break
        if len(chosen) != 2:
            raise RuntimeError(f"Could not select two distinct original tasks for {dimension}")
        for source_index, row in chosen:
            selected.append({**row, "source_index": source_index, "dimension": dimension,
                             "official_dimension": OFFICIAL_DIMENSIONS[dimension]})
    if len(selected) != 12:
        raise RuntimeError(f"Expected 12 fresh Plus test cases, got {len(selected)}")
    return selected


def make_plan_inputs(original_tasks: list[str], plus_rows: list[dict], original_selection_ids: list[int],
                     case_id_start: int = 1000) -> list[dict]:
    inputs: list[dict] = []
    def add_original(task_id: int, group: str, state_id: int) -> None:
        case_id = case_id_start + len(inputs)
        task_name = original_tasks[task_id]
        inputs.append({
            "case_id": case_id, "group": group, "domain": "original", "suite": SPATIAL_SUITE,
            "dimension": "unperturbed", "variant": "original", "variant_id": None,
            "original_task": task_name, "task_name": task_name, "task_id": task_id,
            "source_index": None, "state_id": state_id,
            "environment_seed": 820000000 + case_id * 1000,
            "sampler_seeds": [910000000 + case_id * 1000, 910000001 + case_id * 1000],
            "observation_file": f"inputs/case_{case_id:04d}.npz",
            "metadata_file": f"inputs/case_{case_id:04d}.json",
        })

    for task_id in original_selection_ids:
        add_original(task_id, "selection", 2)
    for task_id in range(len(original_tasks)):
        add_original(task_id, "test", 1)
    for row in plus_rows:
        case_id = case_id_start + len(inputs)
        source_index = int(row["source_index"])
        inputs.append({
            "case_id": case_id, "group": "test", "domain": "plus", "suite": SPATIAL_SUITE,
            "dimension": row["dimension"], "variant": row["variant_id"],
            "official_dimension": row["official_dimension"],
            "variant_id": row["variant_id"], "original_task": row["original_task"],
            "task_name": row["task_name"], "task_id": int(row["task_id"]),
            "source_index": source_index, "state_id": 1,
            "environment_seed": 820000000 + case_id * 1000,
            "sampler_seeds": [910000000 + case_id * 1000, 910000001 + case_id * 1000],
            "observation_file": f"inputs/case_{case_id:04d}.npz",
            "metadata_file": f"inputs/case_{case_id:04d}.json",
        })
    expected_ids = list(range(case_id_start, case_id_start + 26))
    if len(inputs) != 26 or [row["case_id"] for row in inputs] != expected_ids:
        raise RuntimeError(f"Expected ordered case IDs 1000..1025, got {len(inputs)} inputs")
    if [row["group"] for row in inputs[:4]] != ["selection"] * 4 or [row["group"] for row in inputs[4:]] != ["test"] * 22:
        raise RuntimeError("Fresh-input selection/test grouping is malformed")
    return inputs


def resolve_state_index(domain: str, requested_state_id: int, available_state_count: int) -> tuple[int, str | None]:
    """Allow only the declared Plus state-1 to state-0 availability fallback."""
    requested_state_id = int(requested_state_id)
    available_state_count = int(available_state_count)
    if available_state_count < 1:
        raise RuntimeError(f"Official initial-state list is empty for domain={domain}")
    if 0 <= requested_state_id < available_state_count:
        return requested_state_id, None
    if domain == "plus" and requested_state_id == 1 and available_state_count == 1:
        return 0, "requested_state_1_unavailable_only_official_state_0_available"
    raise RuntimeError(
        f"Official state_id={requested_state_id} unavailable for domain={domain}; "
        f"count={available_state_count}"
    )


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def source_file(module) -> Path:
    value = getattr(module, "__file__", None)
    if not value:
        raise RuntimeError(f"Imported module has no source file: {module.__name__}")
    return Path(value).resolve()


def is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validate_import_provenance(provenance: dict, source_root: Path, compat_root: Path, helper_root: Path) -> None:
    source_root = source_root.resolve()
    compat_root = compat_root.resolve()
    helper_root = helper_root.resolve()
    outer = Path(provenance["libero_outer"]).resolve()
    core = Path(provenance["libero_core"]).resolve()
    if not (is_under(outer, source_root) or is_under(outer, compat_root)):
        raise RuntimeError(f"Outer LIBERO import escaped source/compat prefixes: {provenance}")
    compat_core_shim = (compat_root / "libero/libero/__init__.py").resolve()
    if not (is_under(core, source_root) or core == compat_core_shim):
        raise RuntimeError(f"LIBERO core import escaped its source root and explicit compat shim: {provenance}")
    for module_name in ("benchmark", "envs"):
        if not is_under(Path(provenance[module_name]).resolve(), source_root):
            raise RuntimeError(f"LIBERO {module_name} import escaped its domain source root: {provenance}")
    if not is_under(Path(provenance["libero_utils"]).resolve(), helper_root):
        raise RuntimeError(f"Fast-WAM LIBERO helper imported outside its expected prefix: {provenance}")


def render_domain(domain: str, base_root: Path, pilot_root: Path, out: Path, cases: list[dict]) -> None:
    allocation_guard()
    if domain == "original":
        source_root = base_root / "libero/libero"
        compat_root = base_root / "libero-compat"
        prefix = [compat_root, source_root, base_root / "FastWAM", base_root / "FastWAM/src",
                  base_root / "FastWAM/experiments/libero"]
        config_path = base_root / "libero-config"
    elif domain == "plus":
        source_root = pilot_root / "LIBERO-plus/libero/libero"
        compat_root = pilot_root / "libero-plus-compat"
        prefix = [compat_root, source_root, pilot_root / "deps", base_root / "FastWAM",
                  base_root / "FastWAM/src", base_root / "FastWAM/experiments/libero"]
        config_path = pilot_root / "libero-config"
    else:
        raise ValueError(f"Unsupported render domain: {domain}")

    os.environ["PYTHONPATH"] = os.pathsep.join(map(str, prefix))
    os.environ["LIBERO_CONFIG_PATH"] = str(config_path)
    os.environ["ROOT"] = str(pilot_root)
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    sys.path[:0] = list(map(str, prefix))

    import numpy as np
    if "float_" not in np.__dict__:
        np.float_ = np.float64
    if np.float_ is not np.float64:
        raise RuntimeError("LIBERO compatibility requires numpy.float_ to alias float64")
    import torch
    import libero
    import libero.libero as libero_core
    from libero.libero import benchmark, get_libero_path
    from libero.libero import envs as envs_module
    from experiments.libero import libero_utils as ev

    source_root = source_root.resolve()
    compat_root = compat_root.resolve()
    base_helper_root = (base_root / "FastWAM/experiments/libero").resolve()
    provenance = {
        "libero_outer": str(source_file(libero)),
        "libero_core": str(source_file(libero_core)),
        "benchmark": str(source_file(benchmark)),
        "envs": str(source_file(envs_module)),
        "libero_utils": str(source_file(ev)),
        "compat_prefix": str(compat_root),
        "source_root": str(source_root),
    }
    validate_import_provenance(provenance, source_root, compat_root, base_helper_root)

    expected_state_root = (base_root / "libero/libero/init_files"
                           if domain == "original"
                           else pilot_root / "LIBERO-plus/libero/libero/init_files")
    if domain == "plus":
        # This module is the pilot's trusted official loader; ROOT was set before importing it.
        loader_path = pilot_root / "libero_state_loader.py"
        loader_spec = importlib.util.spec_from_file_location("libero_state_loader", loader_path)
        if loader_spec is None or loader_spec.loader is None:
            raise RuntimeError(f"Could not load trusted Plus state loader: {loader_path}")
        loader_module = importlib.util.module_from_spec(loader_spec)
        sys.modules["libero_state_loader"] = loader_module
        loader_spec.loader.exec_module(loader_module)
        load_task_init_states = loader_module.load_task_init_states
    else:
        load_task_init_states = None

    classes = benchmark.get_benchmark_dict()
    suite = classes["libero_spatial"]()
    for case in cases:
        task_id = int(case["task_id"])
        task = suite.get_task(task_id)
        if task.name != case["task_name"]:
            raise RuntimeError(f"Task identity mismatch: id={task_id} expected={case['task_name']!r} actual={task.name!r}")
        if domain == "original" and task.name != case["original_task"]:
            raise RuntimeError(f"Original task map identity mismatch for task_id={task_id}")

        environment_seed = int(case["environment_seed"])
        random.seed(environment_seed)
        np.random.seed(environment_seed % (2**32))
        env = ev.OffScreenRenderEnv(
            bddl_file_name=str(suite.get_task_bddl_file_path(task_id)),
            camera_heights=int(ev.LIBERO_ENV_RESOLUTION),
            camera_widths=int(ev.LIBERO_ENV_RESOLUTION),
        )
        try:
            env.seed(environment_seed)
            env.reset()
            requested_state_id = int(case["state_id"])
            if domain == "plus":
                states, state_path = load_task_init_states(suite, task_id)
            else:
                states, state_path = load_original_init_states(torch, get_libero_path, suite, task_id, expected_state_root)
            available_state_count = len(states)
            state_id, state_fallback_reason = resolve_state_index(
                domain, requested_state_id, available_state_count
            )
            obs = env.set_init_state(states[state_id])
            for _ in range(30):
                obs, _reward, done, _info = env.step(ev.get_libero_dummy_action())
                if done:
                    raise RuntimeError(f"Environment terminated during the 30-step settling period: case_id={case['case_id']}")

            arrays: dict[str, object] = {}
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

            observation_path = out / f"case_{int(case['case_id']):04d}.npz"
            temporary_npz = observation_path.with_name(observation_path.name + ".tmp")
            with temporary_npz.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            os.replace(temporary_npz, observation_path)

            domain_provenance = dict(provenance)
            if domain == "plus":
                domain_provenance["state_loader"] = str(source_file(loader_module))
            else:
                domain_provenance["state_loader"] = "prepare_inputs.py:load_original_init_states"
            metadata = {
                "case_id": int(case["case_id"]),
                "group": case["group"],
                "domain": domain,
                "suite": case["suite"],
                "dimension": case["dimension"],
                "official_dimension": case.get("official_dimension"),
                "variant": case["variant"],
                "variant_id": case["variant_id"],
                "original_task": case["original_task"],
                "task_name": task.name,
                "task_id": task_id,
                "source_index": case["source_index"],
                "requested_state_id": requested_state_id,
                "state_id": state_id,
                "available_state_count": available_state_count,
                "state_fallback_reason": state_fallback_reason,
                "environment_seed": environment_seed,
                "sampler_seeds": list(case["sampler_seeds"]),
                "description": str(task.language),
                "statepath": str(state_path),
                "dataset_import_paths": domain_provenance,
                "settling_steps": 30,
                "evaluation_episodes": 0,
                "episode_count": 0,
                "observation_arrays": {
                    key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for key, value in arrays.items()
                },
            }
            write_json_atomic(out / f"case_{int(case['case_id']):04d}.json", metadata)
            print(f"INPUT_RENDERED domain={domain} case_id={case['case_id']} task={task.name}", flush=True)
        finally:
            env.close()


def load_original_init_states(torch, get_libero_path, suite, task_id: int, expected_root: Path):
    """Load official states only through the trusted upstream init_states root."""
    root = Path(get_libero_path("init_states")).resolve()
    expected_root = expected_root.resolve()
    if root != expected_root:
        raise RuntimeError(f"Original init_states root mismatch: expected={expected_root}, actual={root}")
    original_load = torch.load
    loaded_paths: list[str] = []

    def load_official_state(file, *args, **kwargs):
        path = Path(file).resolve()
        path.relative_to(root)
        if not path.is_file():
            raise FileNotFoundError(path)
        loaded_paths.append(str(path))
        kwargs["weights_only"] = False
        return original_load(file, *args, **kwargs)

    torch.load = load_official_state
    try:
        states = suite.get_task_init_states(int(task_id))
    finally:
        torch.load = original_load
    if len(loaded_paths) != 1:
        raise RuntimeError(f"Expected one official initial-state file, got {loaded_paths}")
    return states, loaded_paths[0]


def control_test() -> None:
    synthetic = []
    for dimension in PLUS_DIMENSIONS:
        for task_index in range(50):
            synthetic.append({
                "suite": SPATIAL_SUITE,
                "dimension": OFFICIAL_DIMENSIONS[dimension],
                "original_task": f"task_{task_index % 10}",
                "task_id": task_index,
                "task_name": f"variant_{dimension}_{task_index}",
                "variant_id": f"libero_spatial:{len(synthetic):04d}",
                "success_rate": -999,
            })
    excluded = {"libero_spatial:0000", "libero_spatial:0150"}
    first = select_plus_cases(synthetic, SELECTION_SEED, excluded)
    second = select_plus_cases(synthetic, SELECTION_SEED, excluded)
    assert first == second
    assert len(first) == 12
    assert not ({row["variant_id"] for row in first} & excluded)
    for dimension in PLUS_DIMENSIONS:
        selected = [row for row in first if row["dimension"] == dimension]
        assert len(selected) == 2
        assert len({row["original_task"] for row in selected}) == 2
    plan = make_plan_inputs([f"base_{i}" for i in range(10)], first, [1, 3, 6, 8])
    assert len(plan) == 26 and [row["case_id"] for row in plan] == list(range(1000, 1026))
    assert [row["task_id"] for row in plan[:4]] == [1, 3, 6, 8]
    assert [row["state_id"] for row in plan[:4]] == [2] * 4
    assert [row["task_id"] for row in plan[4:14]] == list(range(10))
    assert [row["state_id"] for row in plan[4:]] == [1] * 22
    for item in plan:
        case_id = item["case_id"]
        assert item["sampler_seeds"] == [910000000 + case_id * 1000,
                                         910000001 + case_id * 1000]
        assert item["environment_seed"] == 820000000 + case_id * 1000

    assert resolve_state_index("plus", 1, 2) == (1, None)
    assert resolve_state_index("plus", 1, 1) == (
        0, "requested_state_1_unavailable_only_official_state_0_available"
    )
    assert resolve_state_index("original", 2, 3) == (2, None)
    for domain, requested, count in (("original", 1, 1), ("original", 2, 1),
                                    ("plus", 2, 1), ("plus", 1, 0)):
        try:
            resolve_state_index(domain, requested, count)
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                f"State fallback control failed to reject domain={domain}, requested={requested}, count={count}"
            )

    root = Path.cwd() / ".phase12-provenance-control"
    source = root / "libero/libero"
    compat = root / "libero-compat"
    helper = root / "FastWAM/experiments/libero"
    provenance = {
        "libero_outer": str(compat / "libero/__init__.py"),
        "libero_core": str(compat / "libero/libero/__init__.py"),
        "benchmark": str(source / "benchmark/__init__.py"),
        "envs": str(source / "envs/__init__.py"),
        "libero_utils": str(helper / "libero_utils.py"),
    }
    validate_import_provenance(provenance, source, compat, helper)
    for key, bad_path in (
        ("libero_core", compat / "other/libero/__init__.py"),
        ("benchmark", compat / "libero/libero/benchmark/__init__.py"),
    ):
        rejected = dict(provenance)
        rejected[key] = str(bad_path)
        try:
            validate_import_provenance(rejected, source, compat, helper)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"Provenance control failed to reject {key}={bad_path}")
    print("CONTROL_TEST_OK cases=26 selection=4 test=22 seeds=stable plus-state-fallback=exact old-variant-exclusion=pass provenance=shim-only", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", "--output", type=Path)
    parser.add_argument("--protocol", type=Path, default=Path(os.environ.get("ROOT", ".")) / "protocol.json")
    parser.add_argument("--base-root", type=Path, default=Path(os.environ.get("BASE", DEFAULT_BASE)))
    parser.add_argument("--pilot-root", type=Path, default=Path(os.environ.get("PILOT_ROOT", DEFAULT_PILOT)))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--render-domain", choices=("original", "plus"), help=argparse.SUPPRESS)
    parser.add_argument("--cases-json", help=argparse.SUPPRESS)
    parser.add_argument("--control-test", action="store_true", help="run pure selection and plan checks")
    args = parser.parse_args()
    if args.control_test:
        control_test()
        return 0

    base_root = args.base_root.resolve()
    pilot_root = args.pilot_root.resolve()
    if args.render_domain:
        if args.out is None or args.cases_json is None:
            parser.error("--render-domain requires --out and --cases-json")
        render_domain(args.render_domain, base_root, pilot_root, args.out, json.loads(args.cases_json))
        return 0

    if args.out is None:
        parser.error("--out is required")
    allocation_guard()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("protocol") != PROTOCOL:
        raise RuntimeError(f"Unexpected protocol name in {args.protocol}: {protocol.get('protocol')!r}")
    roots = protocol["roots"]
    base_root = Path(roots["base"])
    pilot_root = Path(roots["pilot"])
    old_inputs_root = Path(roots["old_inputs"])
    old_banks_root = Path(roots["old_banks"])
    selection = protocol.get("selection", {})
    selection_seed = int(protocol.get("selection_seed", SELECTION_SEED))
    original_selection_ids = list(selection.get("original_task_ids", [1, 3, 6, 8]))
    case_id_start = int(selection.get("case_id_start", 1000))
    if selection.get("state_id", 2) != 2 or selection.get("test_state_id", 1) != 1:
        raise RuntimeError("Protocol state IDs must be selection=2 and test=1")
    if selection.get("test_original_task_ids", list(range(10))) != list(range(10)):
        raise RuntimeError("Fresh test inputs must cover all ten official LIBERO-Spatial tasks")
    declared_dimensions = selection.get("plus_dimensions", list(PLUS_DIMENSIONS))
    if tuple(declared_dimensions) != PLUS_DIMENSIONS or selection.get("plus_per_dimension", 2) != 2:
        raise RuntimeError("Protocol must specify two fresh Plus variants in each declared dimension")
    if not all(path.is_dir() for path in (base_root, pilot_root, old_inputs_root, old_banks_root)):
        raise FileNotFoundError("One or more protocol source roots are unavailable")
    manifest_path = args.manifest or pilot_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    variants = manifest.get("variants")
    if not isinstance(variants, list):
        raise RuntimeError(f"Frozen pilot manifest has no variants array: {manifest_path}")

    helpers = load_pilot_helpers(pilot_root)
    base_map_path = base_root / "libero/libero/benchmark/libero_suite_task_map.py"
    plus_map_path = pilot_root / "LIBERO-plus/libero/libero/benchmark/libero_suite_task_map.py"
    base_map = helpers.load_literal_task_map(base_map_path)
    plus_map = helpers.load_literal_task_map(plus_map_path)
    original_tasks = helpers.task_names_for_suite(base_map, SPATIAL_SUITE, base_map_path)
    plus_task_names = helpers.task_names_for_suite(plus_map, SPATIAL_SUITE, plus_map_path)
    if len(original_tasks) != 10:
        raise RuntimeError(f"Expected ten official LIBERO-Spatial controls, got {len(original_tasks)}")

    old_plan_path = old_inputs_root / "plan.json"
    old_plan = json.loads(old_plan_path.read_text(encoding="utf-8"))
    old_inputs = old_plan.get("inputs")
    if not isinstance(old_inputs, list):
        raise RuntimeError(f"Existing frozen input plan has no inputs array: {old_plan_path}")
    excluded_variant_ids = {
        row["variant_id"] for row in old_inputs
        if row.get("domain") == "plus" and isinstance(row.get("variant_id"), str)
    }
    if not excluded_variant_ids:
        raise RuntimeError("Could not derive prior selected Plus variant IDs from the frozen old input plan")
    plus_rows = select_plus_cases(variants, selection_seed, excluded_variant_ids)
    for row in plus_rows:
        task_id = int(row["task_id"])
        if not 0 <= task_id < len(plus_task_names) or plus_task_names[task_id] != row.get("task_name"):
            raise RuntimeError(f"Plus manifest/task-map identity mismatch at source_index={row['source_index']}")
    if len(original_selection_ids) != 4 or len(set(original_selection_ids)) != 4:
        raise RuntimeError("Exactly four distinct original selection task IDs are required")
    if any(not isinstance(task_id, int) or task_id < 0 or task_id >= len(original_tasks)
           for task_id in original_selection_ids):
        raise RuntimeError(f"Invalid original selection task IDs: {original_selection_ids}")
    plan_inputs = make_plan_inputs(original_tasks, plus_rows, original_selection_ids, case_id_start)
    selected_variant_ids = {row["variant_id"] for row in plus_rows}
    if len(selected_variant_ids) != 12:
        raise RuntimeError("Fresh Plus test variants are not unique")
    if selected_variant_ids & excluded_variant_ids:
        raise RuntimeError("Fresh Plus selection overlaps a previously selected variant ID")
    old_sampler_seeds = {
        int(seed) for row in old_inputs for seed in row.get("sampler_seeds", [])
    }
    fresh_sampler_seeds = {
        int(seed) for row in plan_inputs for seed in row["sampler_seeds"]
    }
    if fresh_sampler_seeds & old_sampler_seeds:
        raise RuntimeError("Fresh sampler seeds overlap with the previous fixed-input plan")

    calibration_case_ids = list(protocol.get("calibration_cases", [0, 2, 4, 6, 10, 12, 14, 16]))
    calibration_seed_indices = list(protocol.get("calibration_seed_indices", [0]))
    if len(calibration_case_ids) != 8 or calibration_seed_indices != [0]:
        raise RuntimeError("Calibration must reuse the existing eight inputs at sampler seed index zero")
    old_by_id = {int(row["case_id"]): row for row in old_inputs}
    calibration_inputs = []
    for old_case_id in calibration_case_ids:
        old = old_by_id[int(old_case_id)]
        observation_file = Path(old["observation_file"])
        metadata_file = Path(old["metadata_file"])
        if observation_file.is_absolute() or ".." in observation_file.parts or metadata_file.is_absolute() or ".." in metadata_file.parts:
            raise RuntimeError(f"Old calibration paths must be relative to the old input root: {old_case_id}")
        if not (old_inputs_root / observation_file).is_file() or not (old_inputs_root / metadata_file).is_file():
            raise FileNotFoundError(f"Old calibration input files are unavailable for case_id={old_case_id}")
        calibration_inputs.append({
            "case_id": int(old_case_id), "seed_index": 0,
            "sampler_seed": int(old["sampler_seeds"][0]),
            "source_root": str(old_inputs_root),
            "observation_file": observation_file.as_posix(),
            "metadata_file": metadata_file.as_posix(),
        })

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    staging = out / f".inputs-staging-{os.environ['PBS_JOBID']}"
    final_inputs = out / "inputs"
    final_plan = out / "plan.json"
    final_summary = out / "prepare_summary.json"
    if any(path.exists() for path in (staging, final_inputs, final_plan, final_summary,
                                      out / ".plan.json.tmp", out / ".prepare_summary.json.tmp")):
        raise FileExistsError(f"Refusing to overwrite existing preparation output: {out}")
    staging.mkdir()

    for domain in ("original", "plus"):
        selected_cases = [row for row in plan_inputs if row["domain"] == domain]
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--render-domain", domain,
            "--out", str(staging),
            "--base-root", str(base_root),
            "--pilot-root", str(pilot_root),
            "--cases-json", json.dumps(selected_cases, separators=(",", ":"), ensure_ascii=False),
        ]
        env = os.environ.copy()
        env["ROOT"] = str(pilot_root)
        subprocess.run(command, check=True, env=env)

    for item in plan_inputs:
        metadata_path = staging / Path(item["metadata_file"]).name
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("requested_state_id") != item["state_id"]:
            raise RuntimeError(f"Case {item['case_id']} rendered a different requested state")
        resolved_state_id, fallback_reason = resolve_state_index(
            item["domain"], metadata["requested_state_id"], metadata["available_state_count"]
        )
        if (metadata.get("state_id") != resolved_state_id
                or metadata.get("state_fallback_reason") != fallback_reason):
            raise RuntimeError(f"Case {item['case_id']} metadata has an invalid state resolution")
        item["state_id"] = metadata["state_id"]
        for key in ("requested_state_id", "available_state_count", "state_fallback_reason"):
            item[key] = metadata[key]
        for key in ("description", "statepath", "dataset_import_paths", "settling_steps", "evaluation_episodes"):
            item[key] = metadata[key]

    state_fallback_cases = [
        {
            "case_id": item["case_id"],
            "domain": item["domain"],
            "requested_state_id": item["requested_state_id"],
            "actual_state_id": item["state_id"],
            "available_state_count": item["available_state_count"],
            "reason": item["state_fallback_reason"],
        }
        for item in plan_inputs if item["state_fallback_reason"] is not None
    ]

    plan = {
        "protocol": PROTOCOL,
        "selection_seed": selection_seed,
        "input_seed": int(protocol.get("input_seed", SELECTION_SEED)),
        "remote_roots": {key: str(value) for key, value in roots.items()},
        "selection_scope": "Four familiar LIBERO-Spatial tasks at new official initial states; not unseen-task evidence.",
        "test_scope": "Ten familiar original tasks and twelve fresh LIBERO-Plus variants; not independent-variant success evidence.",
        "inputs": plan_inputs,
        "calibration_inputs": calibration_inputs,
        "excluded_variant_ids": sorted(excluded_variant_ids),
        "fresh_plus_variant_ids": sorted(selected_variant_ids),
        "case_id_start": case_id_start,
        "case_count": 26,
        "selection_case_count": 4,
        "test_case_count": 22,
        "episode_count": 0,
    }
    summary = {
        "status": "complete",
        "cases": 26,
        "selection_cases": 4,
        "test_cases": 22,
        "excluded_old_variant_count": len(excluded_variant_ids),
        "excluded_old_variant_ids": sorted(excluded_variant_ids),
        "old_variant_exclusion_passed": not bool(selected_variant_ids & excluded_variant_ids),
        "generated_case_ids": [row["case_id"] for row in plan_inputs],
        "calibration_case_ids": calibration_case_ids,
        "state_fallback_count": len(state_fallback_cases),
        "state_fallback_cases": state_fallback_cases,
        "episodes_executed": 0,
    }
    plan_tmp = out / ".plan.json.tmp"
    write_json_atomic(plan_tmp, plan)
    summary_tmp = out / ".prepare_summary.json.tmp"
    write_json_atomic(summary_tmp, summary)
    os.replace(staging, final_inputs)
    os.replace(plan_tmp, final_plan)
    os.replace(summary_tmp, final_summary)
    print(f"INPUTS_READY path={final_plan} cases=26 selection=4 test=22 old_variant_exclusion=pass episodes=0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
