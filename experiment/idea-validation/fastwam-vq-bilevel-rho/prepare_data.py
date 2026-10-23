"""Freeze and render paired outer/confirmation LIBERO contexts on PBS CPU."""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import re
import time


PROTOCOL = "fastwam-vq-bilevel-rho-v1"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SUITE_PHASES = {
    "libero_spatial": ("early", "middle"),
    "libero_object": ("middle", "late"),
    "libero_goal": ("early", "middle"),
    "libero_10": ("middle", "late"),
}
SOURCE_CASE_IDS = set(range(2000, 2080)) | set(range(3000, 3040))
OVERRIDDEN_FIELDS = {
    "case_id", "phase", "demo_id", "frame_index", "demo_length", "sampler_seeds",
    "observation_file", "metadata_file", "environment_seed",
}


def _development_config(protocol: dict) -> dict:
    development = protocol.get("development", {})
    if protocol.get("protocol") != PROTOCOL:
        raise RuntimeError(f"Expected protocol {PROTOCOL}")
    if int(development.get("selection_seed", -1)) != 20261019:
        raise RuntimeError("Development selection seed is frozen at 20261019")
    if development.get("outer_case_ids") != list(range(4000, 4008)):
        raise RuntimeError("Outer case IDs are frozen at 4000..4007")
    if development.get("confirmation_case_ids") != list(range(5000, 5008)):
        raise RuntimeError("Confirmation case IDs are frozen at 5000..5007")
    if development.get("phase_pairs") != [list(SUITE_PHASES[suite]) for suite in SUITES]:
        raise RuntimeError("Development phase pairs differ from the frozen suite allocation")
    return development


def _source_rows(source: dict) -> tuple[list[dict], dict[tuple[str, int], dict]]:
    rows = source.get("inputs")
    if not isinstance(rows, list) or len(rows) != 120:
        raise RuntimeError("Source calibration plan must contain its original 120 inputs")
    case_ids = [int(row["case_id"]) for row in rows]
    if len(set(case_ids)) != 120 or set(case_ids) != SOURCE_CASE_IDS:
        raise RuntimeError("Source plan must contain the frozen 80 calibration and 40 test case IDs")
    by_task: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        by_task.setdefault((str(row["suite"]), int(row["task_id"])), []).append(row)
    if set(by_task) != {(suite, task) for suite in SUITES for task in range(10)}:
        raise RuntimeError("Source plan must cover all ten tasks in each of the four suites")
    bases = {key: min(task_rows, key=lambda row: int(row["case_id"]))
             for key, task_rows in by_task.items()}
    return rows, bases


def _selected_tasks(source: dict, protocol: dict) -> tuple[random.Random, dict[str, list[int]]]:
    development = _development_config(protocol)
    _rows, bases = _source_rows(source)
    rng = random.Random(int(development["selection_seed"]))
    selected = {}
    for suite in SUITES:
        selected[suite] = rng.sample(sorted(task for name, task in bases if name == suite), 2)
    return rng, selected


def _phase_frame(length: int, phase: str, positions: dict) -> int:
    if length < 3 or phase not in positions:
        raise RuntimeError(f"Invalid demonstration length or phase: length={length}, phase={phase}")
    position = float(positions[phase])
    if not 0 <= position <= 1:
        raise RuntimeError(f"Phase position must be within [0,1]: {phase}={position}")
    return int(round((length - 1) * position))


def freeze_development(source, protocol, demo_catalogs):
    """Return deterministic outer and confirmation rows from demo catalogs.

    ``demo_catalogs[(suite, task)]`` maps demonstration IDs to state counts.
    Task and demonstration sampling uses sorted ID populations and never reads
    model errors or other model-derived values.
    """
    development = _development_config(protocol)
    source_rows, bases = _source_rows(source)
    rng, selected_tasks = _selected_tasks(source, protocol)
    excluded = {(str(row["suite"]), int(row["task_id"]), int(row["demo_id"]))
                for row in source_rows}
    outer, confirmation = [], []
    positions = source.get("phase_positions", {})

    for suite_index, suite in enumerate(SUITES):
        for slot, task_id in enumerate(selected_tasks[suite]):
            base = bases[suite, task_id]
            catalog = demo_catalogs.get((suite, task_id))
            if not isinstance(catalog, dict):
                raise RuntimeError(f"Missing demonstration catalog for {suite} task {task_id}")
            lengths = {int(demo_id): int(length) for demo_id, length in catalog.items()}
            candidates = sorted(demo_id for demo_id, length in lengths.items()
                                if length >= 3 and (suite, task_id, demo_id) not in excluded)
            if len(candidates) < 2:
                raise RuntimeError(f"Need two source-disjoint demonstrations for {suite} task {task_id}")
            outer_demo, confirmation_demo = rng.sample(candidates, 2)
            phase = SUITE_PHASES[suite][slot]

            for target, case_id, demo_id in (
                (outer, development["outer_case_ids"][suite_index * 2 + slot], outer_demo),
                (confirmation, development["confirmation_case_ids"][suite_index * 2 + slot], confirmation_demo),
            ):
                row = {
                    **base,
                    "case_id": case_id,
                    "phase": phase,
                    "demo_id": demo_id,
                    "demo_length": lengths[demo_id],
                    "frame_index": _phase_frame(lengths[demo_id], phase, positions),
                    "sampler_seeds": [920_000_000 + case_id * 2, 920_000_000 + case_id * 2 + 1],
                    "observation_file": f"inputs/case_{case_id}.npz",
                    "metadata_file": f"inputs/case_{case_id}.json",
                    "environment_seed": 830_000_000 + case_id * 17,
                }
                target.append(row)

    validate_development(outer + confirmation, source, protocol)
    return outer, confirmation


def validate_development(rows, source, protocol):
    """Validate both 8-row splits together against the frozen source plan."""
    development = _development_config(protocol)
    source_rows, bases = _source_rows(source)
    expected_ids = development["outer_case_ids"] + development["confirmation_case_ids"]
    by_case = {int(row["case_id"]): row for row in rows}
    if len(rows) != 16 or len(by_case) != 16 or set(by_case) != set(expected_ids):
        raise RuntimeError("Development must contain exactly the frozen 8 outer and 8 confirmation cases")
    excluded = {(str(row["suite"]), int(row["task_id"]), int(row["demo_id"]))
                for row in source_rows}
    new_demonstrations = set()

    for suite_index, suite in enumerate(SUITES):
        selected_task_ids = []
        for slot, phase in enumerate(SUITE_PHASES[suite]):
            index = suite_index * 2 + slot
            outer_case = development["outer_case_ids"][index]
            confirmation_case = development["confirmation_case_ids"][index]
            outer_row, confirmation_row = by_case[outer_case], by_case[confirmation_case]
            if (outer_row["suite"], int(outer_row["task_id"]), outer_row["phase"]) != (
                    suite, int(confirmation_row["task_id"]), phase):
                raise RuntimeError(f"Outer case {outer_case} has an invalid suite, task, or phase")
            if (confirmation_row["suite"], confirmation_row["phase"]) != (suite, phase):
                raise RuntimeError(f"Confirmation case {confirmation_case} has an invalid suite or phase")
            if int(outer_row["demo_id"]) == int(confirmation_row["demo_id"]):
                raise RuntimeError(f"Outer and confirmation demos must differ for {suite} slot {slot}")
            selected_task_ids.append(int(outer_row["task_id"]))

            for row, case_id in ((outer_row, outer_case), (confirmation_row, confirmation_case)):
                task_id = int(row["task_id"])
                demo_id = int(row["demo_id"])
                identity = (suite, task_id, demo_id)
                if identity in excluded or identity in new_demonstrations:
                    raise RuntimeError(f"Development demonstration overlaps a frozen input: {identity}")
                new_demonstrations.add(identity)
                base = bases[suite, task_id]
                if any(row.get(key) != value for key, value in base.items()
                       if key not in OVERRIDDEN_FIELDS):
                    raise RuntimeError(f"Source fields changed for development case {case_id}")
                if set(row) != set(base) | OVERRIDDEN_FIELDS:
                    raise RuntimeError(f"Unexpected fields in development case {case_id}")
                if int(row["demo_length"]) < 3 or not 0 <= int(row["frame_index"]) < int(row["demo_length"]):
                    raise RuntimeError(f"Invalid demonstration frame for case {case_id}")
                if row["frame_index"] != _phase_frame(
                        int(row["demo_length"]), phase, source.get("phase_positions", {})):
                    raise RuntimeError(f"Incorrect phase frame for case {case_id}")
                if row["sampler_seeds"] != [920_000_000 + case_id * 2, 920_000_000 + case_id * 2 + 1]:
                    raise RuntimeError(f"Incorrect sampler seeds for case {case_id}")
                if int(row["environment_seed"]) != 830_000_000 + case_id * 17:
                    raise RuntimeError(f"Incorrect environment seed for case {case_id}")
                if (row["observation_file"] != f"inputs/case_{case_id}.npz"
                        or row["metadata_file"] != f"inputs/case_{case_id}.json"):
                    raise RuntimeError(f"Incorrect output paths for case {case_id}")
        if len(set(selected_task_ids)) != 2:
            raise RuntimeError(f"Development must sample two distinct tasks for {suite}")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _catalog_for_tasks(source, selected_tasks, out, receipt, render):
    _rows, bases = _source_rows(source)
    catalogs = {}
    for suite in SUITES:
        for task_id in selected_tasks[suite]:
            with render.open_hdf5(bases[suite, task_id], out, receipt) as h5:
                data = h5.get("data")
                if data is None:
                    raise RuntimeError(f"Original HDF5 lacks data: {suite} task {task_id}")
                demos = {}
                for name, group in data.items():
                    match = re.fullmatch(r"demo_(\d+)", str(name))
                    states = group.get("states") if match else None
                    if states is not None and len(states.shape) == 2 and int(states.shape[0]) >= 3:
                        demos[int(match.group(1))] = int(states.shape[0])
                catalogs[suite, task_id] = demos
    return catalogs


def _valid_render(row, out, render):
    observation = out / row["observation_file"]
    metadata = out / row["metadata_file"]
    if not observation.is_file() or not metadata.is_file():
        return False
    try:
        saved = _read_json(metadata)
    except (OSError, json.JSONDecodeError):
        return False
    return (saved.get("case_id") == row["case_id"]
            and saved.get("input_identity") == render.row_identity(row))


def main() -> int:
    import render_reference as render

    render.allocation_guard()
    root = Path(os.environ["ROOT"])
    protocol = _read_json(root / "protocol.json")
    development = _development_config(protocol)
    calibration_root = Path(protocol["calibration_root"])
    old_data = calibration_root / "data"
    source_path = old_data / "plan.json"
    source = _read_json(source_path)
    receipt = _read_json(old_data / "source_receipt.json")
    if source.get("protocol") != "fastwam-vq-calibration-suites-v1" or source.get("status") != "complete":
        raise RuntimeError("The frozen calibration-suite source plan must be complete")
    if receipt.get("revision") != source.get("source_hf_revision"):
        raise RuntimeError("Source receipt does not match the frozen calibration plan")
    _source_rows(source)

    out = root / "data"
    status_dir = root / "results" / "data_prepare"
    out.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out / "plan.json"

    if plan_path.is_file():
        plan = _read_json(plan_path)
        if plan.get("protocol") != PROTOCOL:
            raise RuntimeError("Existing development plan belongs to another protocol")
        outer = [row for row in plan.get("inputs", [])
                 if int(row.get("case_id", -1)) in development["outer_case_ids"]]
        confirmation = [row for row in plan.get("inputs", [])
                        if int(row.get("case_id", -1)) in development["confirmation_case_ids"]]
        validate_development(outer + confirmation, source, protocol)
        if plan.get("inputs") != source["inputs"] + outer + confirmation:
            raise RuntimeError("Existing plan changed source rows or development row ordering")
        if plan.get("calibration_input_root") != str(old_data):
            raise RuntimeError("Existing plan points to a different calibration input root")
    else:
        _rng, selected_tasks = _selected_tasks(source, protocol)
        catalogs = _catalog_for_tasks(source, selected_tasks, out, receipt, render)
        outer, confirmation = freeze_development(source, protocol, catalogs)
        plan = {
            **source,
            "protocol": PROTOCOL,
            "status": "preparing",
            "selection_seed": development["selection_seed"],
            "inputs": source["inputs"] + outer + confirmation,
            "input_count": 136,
            "development_outer_ids": development["outer_case_ids"],
            "development_confirmation_ids": development["confirmation_case_ids"],
            "calibration_input_root": str(old_data),
            "development_input_root": str(out),
            "source_protocol": source["protocol"],
            "source_plan_file": str(source_path),
            "source_receipt_revision": receipt["revision"],
            "new_episodes": 0,
            "predicted_actions_executed": 0,
        }
        render.write_json_atomic(plan_path, plan)

    validate_development(outer + confirmation, source, protocol)
    fastwam, libero, compat, config, _datasets = render.load_roots(protocol)
    np, benchmark, ev, render_provenance = render.prepare_environment(fastwam, libero, compat, config)
    suites = {name: benchmark.get_benchmark_dict()[name]() for name in SUITES}
    job_id = os.environ["PBS_JOBID"]
    started = float(os.environ.get("FW_JOB_START_EPOCH", time.time()))
    deadline = min(float(protocol.get("soft_deadline_seconds", 6300)), 6300.0)
    completed = []

    for row in outer + confirmation:
        if _valid_render(row, out, render):
            completed.append(row["case_id"])
        else:
            if time.time() - started >= deadline:
                render.write_json_atomic(status_dir / "progress.json", {
                    "phase": "checkpointed", "complete": False, "completed": completed,
                    "planned": 16, "pbs_jobid": job_id,
                })
                return 75
            render.render_one(row, out, (np, ev), suites, render_provenance, fastwam, receipt)
            completed.append(row["case_id"])
        render.write_json_atomic(status_dir / "progress.json", {
            "phase": "rendering", "complete": False, "completed": completed,
            "planned": 16, "pbs_jobid": job_id,
        })
        print(f"DEVELOPMENT_RENDERED {row['case_id']}", flush=True)

    plan["status"] = "complete"
    render.write_json_atomic(plan_path, plan)
    summary = {
        "protocol": PROTOCOL,
        "complete": True,
        "pbs_jobid": job_id,
        "counts": {"development": 16, "outer": 8, "confirmation": 8},
        "development_outer_ids": development["outer_case_ids"],
        "development_confirmation_ids": development["confirmation_case_ids"],
        "demonstration_disjoint_from_source_plan": True,
        "source_plan_input_count": 120,
        "phase_by_suite": {suite: list(SUITE_PHASES[suite]) for suite in SUITES},
        "provenance": {
            "source_plan_file": str(source_path),
            "source_protocol": source["protocol"],
            "source_hf_revision": receipt["revision"],
            "calibration_input_root": str(old_data),
            "development_input_root": str(out),
            "render": render_provenance,
        },
        "new_episodes": 0,
        "predicted_actions_executed": 0,
    }
    render.write_json_atomic(status_dir / "summary.json", summary)
    render.write_json_atomic(status_dir / "progress.json", {
        "phase": "complete", "complete": True, "completed": completed,
        "planned": 16, "pbs_jobid": job_id,
    })
    print("DEVELOPMENT_PREPARATION_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
