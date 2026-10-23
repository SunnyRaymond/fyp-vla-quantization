"""Resumable original-paired280 closed-loop evaluation for trained arms."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import time

from control_common import Deadline, guard, read, save


PROTOCOL = "fastwam-rotation-baselines-v1"
EXPECTED_EPISODES = 280
CHUNK_STEPS = 10
MAX_NORMAL_STEPS = 400
MAX_LONG_STEPS = 700
EXPECTED_MODULES = 614
ENV_SEED_BASE = 800_000_000
SAMPLER_SEED_BASE = 900_000_000
SEED_STRIDE = 1000


class SoftStop(RuntimeError):
    """Deadline reached at an episode/replan boundary; do not save this episode."""


def shard_indices(shard):
    if isinstance(shard, bool) or not isinstance(shard, int) or shard not in (0, 1):
        raise ValueError("Closed-loop shard must be 0 or 1")
    return list(range(shard, EXPECTED_EPISODES, 2))


def episode_cap(row):
    return MAX_LONG_STEPS if row.get("suite") == "libero_10" else MAX_NORMAL_STEPS


def closedloop_manifest_path(parent):
    frozen = parent.get("frozen_evaluation")
    cohort = parent.get("source_closed_protocol", {}).get("cohort")
    if not isinstance(frozen, dict) or not isinstance(cohort, dict):
        raise ValueError("Parent protocol lacks the frozen closed-loop manifest binding")
    frozen_path = frozen.get("closedloop_manifest")
    source_path = cohort.get("reference_manifest")
    if (not isinstance(frozen_path, str) or not frozen_path
            or frozen_path != source_path):
        raise ValueError("Frozen-evaluation and source-cohort manifests must be the same path")
    if frozen.get("closedloop_episodes") != EXPECTED_EPISODES:
        raise ValueError("Parent protocol does not bind all 280 closed-loop episodes")
    return Path(frozen_path)


def validate_manifest_doc(document, parent):
    cohort = parent.get("source_closed_protocol", {}).get("cohort")
    if not isinstance(document, dict) or not isinstance(cohort, dict):
        raise ValueError("Manifest and source cohort must be mappings")
    if (document.get("protocol") != cohort.get("manifest_protocol")
            or document.get("protocol") != PROTOCOL
            or cohort.get("variants_per_arm") != EXPECTED_EPISODES):
        raise ValueError("Closed-loop manifest protocol or cohort size differs")
    rows = document.get("variants")
    if not isinstance(rows, list) or len(rows) != EXPECTED_EPISODES:
        raise ValueError("Expected all original 280 manifest variants")
    seen, cells = set(), {}
    for index, row in enumerate(rows):
        if (not isinstance(row, dict) or type(row.get("index")) is not int
                or row["index"] != index or type(row.get("task_id")) is not int
                or row.get("state_id", 0) != 0):
            raise ValueError(f"Invalid original manifest row {index}")
        variant_id = str(row.get("variant_id", ""))
        suite, dimension = row.get("suite"), row.get("dimension")
        if (not variant_id or variant_id in seen
                or not isinstance(suite, str) or not suite
                or not isinstance(dimension, str) or not dimension):
            raise ValueError(f"Missing or duplicate manifest identity at row {index}")
        seen.add(variant_id)
        cells[(suite, dimension)] = cells.get((suite, dimension), 0) + 1
    if (len({suite for suite, _ in cells}) != 4
            or len({dimension for _, dimension in cells}) != 7
            or len(cells) != 28 or set(cells.values()) != {10}):
        raise ValueError("Original manifest must retain the four-by-seven, ten-per-cell cohort")
    return rows


def load_manifest(parent):
    path = closedloop_manifest_path(parent)
    document = json.loads(path.read_text(encoding="utf-8"))
    return document, validate_manifest_doc(document, parent)


def _seed_values(parent):
    cohort = parent.get("source_closed_protocol", {}).get("cohort", {})
    expected = (cohort.get("env_seed_base"), cohort.get("sampler_seed_base"),
                cohort.get("seed_stride"))
    if expected != (ENV_SEED_BASE, SAMPLER_SEED_BASE, SEED_STRIDE):
        raise ValueError("Parent closed-loop environment/sampler seed rule differs")
    return expected


def validate_closedloop_protocol(parent):
    closed = parent.get("source_closed_protocol")
    inference = closed.get("inference") if isinstance(closed, dict) else None
    cohort = closed.get("cohort") if isinstance(closed, dict) else None
    if (not isinstance(inference, dict) or not isinstance(cohort, dict)
            or inference.get("execute_per_replan") != CHUNK_STEPS
            or inference.get("settling_steps") != 30
            or inference.get("max_steps_normal") != MAX_NORMAL_STEPS
            or inference.get("max_steps_long") != MAX_LONG_STEPS
            or cohort.get("state_id") != 0
            or cohort.get("paired_seeds_across_arms") is not True):
        raise ValueError("Frozen closed-loop step, settling, state, or pairing protocol differs")
    _seed_values(parent)
    closedloop_manifest_path(parent)


def _finite_number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(float(value)))


def _validate_action_chunk(value):
    if (not isinstance(value, list) or len(value) != 32
            or any(not isinstance(row, list) or len(row) != 7
                   or any(not _finite_number(item) for item in row) for row in value)):
        raise ValueError("First actual action chunk must be finite 32x7 values")


def validate_record(record, parent, row, index, arm, binding):
    if (not isinstance(record, dict) or not isinstance(row, dict)
            or not isinstance(binding, dict) or not binding
            or not isinstance(parent.get("protocol"), str) or not parent["protocol"]):
        raise ValueError("Episode record, manifest row, and deployment binding must be mappings")
    if (isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < EXPECTED_EPISODES
            or row.get("index") != index or row.get("state_id", 0) != 0):
        raise ValueError("Episode index/state differs from the original manifest row")
    env_base, sampler_base, stride = _seed_values(parent)
    unit = f"closedloop_{arm}_{index % 2}"
    expected = {
        "protocol": parent.get("protocol"), "unit": unit, "arm": arm,
        "index": index, "row": row,
        "variant_id": str(row.get("variant_id", "")),
        "suite": row.get("suite"), "dimension": row.get("dimension"),
        "task_id": row.get("task_id"), "state_id": 0,
        "environment_seed": env_base + index * stride,
        "sampler_seed": sampler_base + index * stride,
        "deployment_binding": binding, "complete": True, "status": "complete",
    }
    mismatch = [key for key, value in expected.items() if record.get(key) != value]
    if mismatch:
        raise ValueError("Completed episode binding differs: " + ", ".join(mismatch))
    if not isinstance(record.get("producer_pbs_jobid"), str) or not record["producer_pbs_jobid"]:
        raise ValueError("Completed episode lacks its producer PBS job")
    if not isinstance(record.get("description"), str) or not record["description"]:
        raise ValueError("Completed episode lacks the runtime task description")
    success = record.get("success")
    steps, replans = record.get("steps"), record.get("replans")
    cap = episode_cap(row)
    if (not isinstance(success, bool) or type(steps) is not int or not 1 <= steps <= cap
            or type(replans) is not int or not 1 <= replans <= steps
            or steps <= CHUNK_STEPS * (replans - 1)
            or steps > CHUNK_STEPS * replans
            or (not success and steps != cap)
            or record.get("termination") != ("success" if success else "control_step_timeout")):
        raise ValueError("Completed episode success/steps/replans/termination are inconsistent")
    expected_seeds = [sampler_base + index * stride + r for r in range(replans)]
    if record.get("sampler_seeds") != expected_seeds:
        raise ValueError("Episode sampler seeds differ from the original paired schedule")
    if not _finite_number(record.get("episode_seconds")) or record["episode_seconds"] < 0:
        raise ValueError("Episode duration must be finite and nonnegative")
    _validate_action_chunk(record.get("first_action_chunk"))
    return record


def _publish_state(out, state, *, summary=False):
    state["completed"] = len(state["completed_indices"])
    state["this_job"].update({
        "startup_query_attempted": state["startup_query_attempted"],
        "episode_query_attempted": state["episode_query_attempted"],
        "episode_query_completed": state["episode_query_completed"],
        "new_episodes": state["new_episodes_this_job"],
        "reused_episodes": state["reused_episodes_this_job"],
    })
    save(out / "progress.json", state)
    if summary:
        save(out / "summary.json", state)


def _query_count(state, key, publish):
    state[key] += 1
    publish()


def rollout_episode(switching, parent, row, index, arm, binding, unit, job_id,
                    deadline, state, publish=lambda: None):
    """Run one original environment episode; only return a fully finished record."""
    torch, runtime = switching.torch, switching.runtime
    env_base, sampler_base, stride = _seed_values(parent)
    started = time.monotonic()
    observation = runtime.task(row, index)
    cap, steps, replans, success = episode_cap(row), 0, 0, False
    sampler_seeds, first_action_chunk = [], None
    while steps < cap and not success:
        if deadline.near(180):
            raise SoftStop(f"deadline at replan boundary for manifest index {index}")
        seed = sampler_base + index * stride + replans
        datum = switching.datum(observation)
        switching.banks.seen.clear()
        q = runtime.q
        if (not hasattr(q, "_native_int4_gemm_calls")
                or not hasattr(q, "_kv_active")
                or q._native_int4_gemm_calls != 0 or q._kv_active):
            raise RuntimeError("Closed-loop path requires decoded-BF16 non-KV inference")
        state["current_index"], state["current_replan"] = index, replans
        _query_count(state, "episode_query_attempted", publish)
        action, _ = runtime.infer(datum, seed, measure=False)
        if (not torch.is_tensor(action) or tuple(action.shape) != (32, 7)
                or not action.is_floating_point() or not bool(torch.isfinite(action).all())
                or len(switching.banks.seen) != EXPECTED_MODULES
                or q._native_int4_gemm_calls != 0 or q._kv_active):
            raise RuntimeError("Native closed-loop infer missed modules or used a forbidden kernel path")
        if first_action_chunk is None:
            first_action_chunk = action.detach().cpu().tolist()
        commands = runtime.command(action.clone())
        commands_tensor = torch.as_tensor(commands)
        if (tuple(commands_tensor.shape) != (32, 7)
                or not bool(torch.isfinite(commands_tensor).all())):
            raise RuntimeError("Runtime produced invalid 32x7 environment commands")
        sampler_seeds.append(seed)
        _query_count(state, "episode_query_completed", publish)
        for command in commands[:min(CHUNK_STEPS, cap - steps)]:
            observation, _, done, _ = runtime.env.step(command.tolist())
            steps += 1
            if done:
                success = True
                break
        replans += 1
    record = {
        "protocol": parent["protocol"], "unit": unit, "arm": arm,
        "index": index, "row": row,
        "variant_id": str(row.get("variant_id", "")),
        "suite": row["suite"], "dimension": row["dimension"],
        "task_id": row["task_id"], "description": getattr(runtime, "description", None),
        "deployment_binding": binding, "producer_pbs_jobid": job_id,
        "environment_seed": env_base + index * stride, "state_id": 0,
        "sampler_seed": sampler_base + index * stride, "sampler_seeds": sampler_seeds,
        "steps": steps, "replans": replans, "success": success,
        "termination": "success" if success else "control_step_timeout",
        "episode_seconds": time.monotonic() - started,
        "first_action_chunk": first_action_chunk,
        "complete": True, "status": "complete",
    }
    return validate_record(record, parent, row, index, arm, binding)


def save_episode(record, parent, row, index, arm, binding, path):
    """Atomically publish an already complete episode after strict binding checks."""
    validate_record(record, parent, row, index, arm, binding)
    save(path, record)
    return record


def run_and_save_episode(switching, parent, row, index, arm, binding, unit,
                         job_id, deadline, state, path, publish=lambda: None):
    record = rollout_episode(switching, parent, row, index, arm, binding,
                              unit, job_id, deadline, state, publish)
    return save_episode(record, parent, row, index, arm, binding, path)


def _archive_previous_summary(out, current_job):
    path = out / "summary.json"
    if not path.is_file():
        return
    previous = read(path)
    old_job = previous.get("pbs_jobid")
    if not isinstance(old_job, str) or not old_job:
        raise ValueError("Cannot archive prior closed-loop summary without its PBS job id")
    if old_job == current_job:
        return
    if Path(old_job).name != old_job:
        raise ValueError("Unsafe PBS job id in previous summary")
    archive = out / "jobs" / f"{old_job}.json"
    if archive.exists() and read(archive) != previous:
        raise ValueError(f"Prior PBS summary archive conflicts: {archive}")
    save(archive, previous)
    path.unlink()


def _existing_records(episodes_dir, rows, indices, parent, arm, binding):
    if not episodes_dir.exists():
        return {}
    allowed = {f"{index}.json" for index in indices}
    extras = {path.name for path in episodes_dir.glob("*.json")} - allowed
    if extras:
        raise ValueError(f"Episode directory contains out-of-shard records: {sorted(extras)}")
    records = {}
    for index in indices:
        path = episodes_dir / f"{index}.json"
        if path.is_file():
            record = read(path)
            validate_record(record, parent, rows[index], index, arm, binding)
            records[index] = record
    return records


def run(unit, arm, shard):
    guard()
    import torch
    from aggregate import completed
    from diagnostic_runtime import SwitchingRuntime
    import training_evaluation

    here = Path(__file__).resolve().parent
    parent, config = read(here / "protocol.json"), read(here / "training_protocol.json")
    job_id = os.environ["PBS_JOBID"]
    if (not re.fullmatch(r"[a-z][a-z0-9_]{0,25}", arm)
            or arm not in config.get("arms", {})
            or unit != f"closedloop_{arm}_{shard}"):
        raise ValueError("Expected the frozen unit closedloop_<training_arm>_[01]")
    indices = shard_indices(shard)
    root = Path(parent["remote_root"])
    out = root / "results" / unit
    out.mkdir(parents=True, exist_ok=True)
    _archive_previous_summary(out, job_id)
    episodes_dir = out / "episodes"
    state = {
        "protocol": parent["protocol"], "unit": unit, "arm": arm,
        "pbs_jobid": job_id, "status": "running", "complete": False,
        "checkpointed": False, "expected": len(indices), "completed": 0,
        "completed_indices": [], "remaining_indices": list(indices),
        "startup_query_attempted": 0, "episode_query_attempted": 0,
        "episode_query_completed": 0, "new_episodes_this_job": 0,
        "reused_episodes_this_job": 0,
        "training_updates": 0, "new_teacher_queries": 0,
        "this_job": {"pbs_jobid": job_id},
    }
    deadline, switching = None, None
    _publish_state(out, state)
    try:
        validate_closedloop_protocol(parent)
        manifest_doc, rows = load_manifest(parent)
        del manifest_doc
        if any(row.get("state_id", 0) != 0 for row in rows):
            raise ValueError("Original paired280 requires state_id 0")
        completed(root, "train_" + arm)
        deadline = Deadline(6600)
        switching = SwitchingRuntime(parent, out)
        binding = training_evaluation.prepare_runtime(
            switching, parent, config, arm, out, state)
        startup_q = switching.runtime.q
        if (len(switching.banks.seen) != EXPECTED_MODULES
                or not hasattr(startup_q, "_native_int4_gemm_calls")
                or not hasattr(startup_q, "_kv_active")
                or startup_q._native_int4_gemm_calls != 0 or startup_q._kv_active):
            raise RuntimeError("Training deployment startup native-infer gate failed")
        state["deployment_binding"] = binding
        existing = _existing_records(episodes_dir, rows, indices, parent, arm, binding)
        state["completed_indices"] = sorted(existing)
        state["remaining_indices"] = [index for index in indices if index not in existing]
        state["reused_episodes_this_job"] = len(existing)
        _publish_state(out, state)
        recent_seconds = max((record["episode_seconds"] for record in existing.values()), default=0.0)
        for index in state["remaining_indices"][:]:
            reserve = max(1200.0, recent_seconds * 1.3) + 180.0
            if deadline.near(reserve):
                state.update(status="checkpointed", checkpointed=True, complete=False,
                             stop_reason="soft deadline before next full episode",
                             current_index=index)
                _publish_state(out, state, summary=True)
                return 0
            row = rows[index]
            path = episodes_dir / f"{index}.json"
            state["current_index"] = index
            publish = lambda: _publish_state(out, state)
            record = run_and_save_episode(
                switching, parent, row, index, arm, binding, unit,
                job_id, deadline, state, path, publish)
            state["completed_indices"].append(index)
            state["completed_indices"].sort()
            state["remaining_indices"].remove(index)
            state["new_episodes_this_job"] += 1
            recent_seconds = record["episode_seconds"]
            state.pop("current_replan", None)
            state.pop("current_index", None)
            _publish_state(out, state)
        if state["completed_indices"] != indices or len(state["completed_indices"]) != 140:
            raise RuntimeError("Closed-loop shard did not complete exactly its original 140 episodes")
        state.update(status="complete", checkpointed=False, complete=True,
                     expected=140, completed=140, completed_indices=list(indices),
                     remaining_indices=[])
        _publish_state(out, state, summary=True)
        return 0
    except SoftStop as exc:
        state.update(status="checkpointed", checkpointed=True, complete=False,
                     stop_reason=str(exc))
        _publish_state(out, state, summary=True)
        return 0
    except Exception as exc:
        state.update(status="failed", checkpointed=False, complete=False,
                     failure=f"{type(exc).__name__}: {exc}")
        _publish_state(out, state, summary=True)
        raise
    finally:
        if switching is not None:
            try:
                switching.close()
            except Exception as exc:
                state.update(status="failed", checkpointed=False, complete=False,
                             failure=f"close failed: {type(exc).__name__}: {exc}")
                _publish_state(out, state, summary=True)
                raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", required=True)
    args = parser.parse_args(argv)
    match = re.fullmatch(r"closedloop_([a-z][a-z0-9_]{0,25})_([01])", args.unit)
    if not match:
        parser.error("Expected closedloop_<training_arm>_[01]")
    return run(args.unit, match[1], int(match[2]))


if __name__ == "__main__":
    raise SystemExit(main())
