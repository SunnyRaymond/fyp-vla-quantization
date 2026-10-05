"""Fixed-protocol full LIBERO-Goal W4A8 suite evaluator; PBS allocation only."""

import argparse
import contextlib
import itertools
import json
import os
import random
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_ROOT = SCRIPT_DIR.parent
SCREEN_DIR = SCRIPT_DIR if (SCRIPT_DIR / "runner.py").exists() else RUN_ROOT / "mechanism-screen"
PAIRED_DIR = SCRIPT_DIR if (SCRIPT_DIR / "closedloop.py").exists() else RUN_ROOT / "paired-followup"
sys.path.insert(0, str(SCREEN_DIR))
sys.path.insert(0, str(PAIRED_DIR))

import runner as screen_runner  # noqa: E402
import closedloop  # noqa: E402


ARMS = ("bf16", "rtn", "independent", "learned")
QUANT_ARMS = ("rtn", "independent", "learned")
STATE_IDS = tuple(range(50))
EXPECTED_SITE_COUNT = 6446
PROTOCOL_ID = "fastwam-libero-goal-suite-fixed-v1"
ORDER_SEED = 20261003
SAMPLER_SEED_BASE = 1_026_100_000
DRAW_SEED_BASE = 2_026_100_000
ENV_SEED_BASE = 3_026_100_000
TASK_SEED_STRIDE = 100_000
DEFAULT_LOCKED = Path(
    "/scratch/users/ntu/yguo017/wam-phase-screen-20261003/artifacts/"
    "25666715.pbs101/locked_selection.json"
)


class GateError(RuntimeError):
    """A frozen-protocol, identity, or numerical gate failed."""


def save_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def slot_key(arm, state_id):
    return f"{arm}:{state_id:02d}"


def environment_seed(task_id, state_id):
    return ENV_SEED_BASE + task_id * TASK_SEED_STRIDE + state_id


def sampler_seed(task_id, state_id, replan_idx):
    return SAMPLER_SEED_BASE + task_id * TASK_SEED_STRIDE + state_id * 100 + replan_idx


def draw_seed(task_id, state_id, replan_idx):
    return DRAW_SEED_BASE + task_id * TASK_SEED_STRIDE + state_id * 100 + replan_idx


def validate_execution_state_ids(state_ids):
    if (not isinstance(state_ids, list) or not state_ids
            or any(type(state_id) is not int or state_id not in STATE_IDS for state_id in state_ids)
            or state_ids != sorted(set(state_ids))):
        raise ValueError("execution_state_ids must be a nonempty, sorted subset of global states 0..49")
    return tuple(state_ids)


def state_ids_for_range(state_start, state_stop):
    if type(state_start) is not int or type(state_stop) is not int or not 0 <= state_start < state_stop <= len(STATE_IDS):
        raise ValueError("state range must satisfy 0 <= state-start < state-stop <= 50")
    return tuple(range(state_start, state_stop))


def order_schedule(task_id):
    """Balance all six orders, with each arm in each position 16 or 17 times."""
    permutations = list(itertools.permutations(QUANT_ARMS))
    rows = [order for order in permutations for _ in range(8)]
    rows.extend((permutations[0], permutations[3]))
    random.Random(ORDER_SEED + task_id).shuffle(rows)
    return {state_id: rows[state_id] for state_id in STATE_IDS}


def self_check():
    env_seeds, sampler_seeds, draw_seeds = set(), set(), set()
    for task_id in range(10):
        schedule = order_schedule(task_id)
        if len(schedule) != 50 or set(schedule) != set(STATE_IDS):
            raise RuntimeError("Protocol self-check failed: task order coverage")
        positions = {arm: [0, 0, 0] for arm in QUANT_ARMS}
        for order in schedule.values():
            if set(order) != set(QUANT_ARMS):
                raise RuntimeError("Protocol self-check failed: quant arm permutation")
            for pos, arm in enumerate(order):
                positions[arm][pos] += 1
        if any(max(counts) - min(counts) > 1 for counts in positions.values()):
            raise RuntimeError("Protocol self-check failed: quant arm order is unbalanced")
        for state_id in STATE_IDS:
            env_seeds.add(environment_seed(task_id, state_id))
            for replan_idx in range(40):
                sampler_seeds.add(sampler_seed(task_id, state_id, replan_idx))
                draw_seeds.add(draw_seed(task_id, state_id, replan_idx))
    if (len(env_seeds) != 500 or len(sampler_seeds) != 20_000
            or len(draw_seeds) != 20_000 or sampler_seeds & draw_seeds
            or env_seeds & sampler_seeds or env_seeds & draw_seeds):
        raise RuntimeError("Protocol self-check failed: task/state seed collision")

    for start, stop in ((0, 13), (13, 26), (26, 38), (38, 50)):
        state_ids = tuple(range(start, stop))
        records = [
            {"arm": arm, "state_id": state_id, "success": False, "complete": True}
            for arm in ARMS for state_id in state_ids
        ]
        summary = task_summaries(records, 0, state_ids)
        expected_slots = len(state_ids) * len(ARMS)
        if (not summary["complete"] or summary["slots_attempted"] != expected_slots
                or summary["slots_complete"] != expected_slots
                or any(summary["arm_summaries"][arm]["denominator"] != len(state_ids) for arm in ARMS)):
            raise RuntimeError("Protocol self-check failed: state-shard summary/denominator")
        if task_summaries(records[:-1], 0, state_ids)["complete"]:
            raise RuntimeError("Protocol self-check failed: incomplete state shard was accepted")


def switch_task(screen, task_id):
    from libero.libero import benchmark

    close = getattr(screen.env, "close", None)
    if close is not None:
        close()
    task = benchmark.get_benchmark_dict()["libero_goal"]().get_task(task_id)
    env, description = screen.ev.get_libero_env(
        task, screen.ev.LIBERO_ENV_RESOLUTION, 2026 + task_id
    )
    screen.task, screen.description, screen.env = task, description, env
    screen.cfg.EVALUATION.task_suite_name = "libero_goal"
    screen.cfg.EVALUATION.task_id = task_id
    return task, description


def install_action_range_gate(screen):
    """Validate inferred normalized actions without changing the official action path."""
    import types

    original = screen.infer

    def checked_infer(self, *args, **kwargs):
        action, stats, seconds = original(*args, **kwargs)
        if action.ndim != 2 or tuple(action.shape) != (32, 7):
            raise GateError(f"Expected normalized [32,7] action chunk, got {tuple(action.shape)}")
        if not bool(self.torch.isfinite(action).all()):
            raise GateError("Normalized action chunk contains a nonfinite value")
        traced = kwargs.get("trace", args[6] if len(args) > 6 else True)
        expected_count = EXPECTED_SITE_COUNT if traced else 0
        if stats["site_count"] != expected_count or stats["nonfinite_values"] != 0:
            raise GateError(f"Invalid quantization coverage or finite count: {stats}")
        return action, stats, seconds

    screen.infer = types.MethodType(checked_infer, screen)


def establish_task_gates(screen, *, task_id, initial_state, progress, locked_path):
    """Run a BF16 coverage/reference gate on the current task before any W4 conversion."""
    torch = screen.torch
    env = screen.env
    state_id = 0
    env_seed = environment_seed(task_id, state_id)
    env.seed(env_seed)
    env.reset()
    obs = env.set_init_state(initial_state)
    for warmup_idx in range(30):
        obs, _, done, _ = env.step(screen.ev.get_libero_dummy_action())
        if done:
            raise GateError(f"Task {task_id} terminated during warmup step {warmup_idx + 1}")

    datum = closedloop._model_input(screen, obs)
    s_seed = sampler_seed(task_id, state_id, 0)
    d_seed = draw_seed(task_id, state_id, 0)
    reference, stats, _ = screen.infer(datum, "bf16", s_seed, d_seed)
    sites = screen.expected_sites or []
    if len(sites) != EXPECTED_SITE_COUNT or len(set(sites)) != EXPECTED_SITE_COUNT:
        raise GateError(f"Expected {EXPECTED_SITE_COUNT} unique executed sites, got {len(sites)}")
    if stats["site_count"] != EXPECTED_SITE_COUNT:
        raise GateError(f"BF16 traced {stats['site_count']} sites, expected {EXPECTED_SITE_COUNT}")
    learned, locked_data = closedloop.validate_locked(screen, locked_path)
    repeated = screen.infer(datum, "bf16", s_seed, d_seed)[0]
    repeated_error = float((repeated - reference).abs().max())
    unhooked = screen.infer(datum, "bf16", s_seed, d_seed, trace=False)[0]
    identity_error = float((unhooked - reference).abs().max())
    if repeated_error > 1e-6 or identity_error > 1e-6:
        raise GateError(
            f"BF16 reference gate failed: repeat={repeated_error} hook-disabled={identity_error}"
        )

    progress["coverage_validated"] = True
    progress["reference_gate"] = True
    progress["locked_source"] = {
        "path": str(Path(locked_path).resolve()),
        "source_job_id": "25666715.pbs101",
        "selected_keys": len(learned),
        "permutation_nongauge": locked_data.get("permutation_nongauge"),
    }
    closedloop.save_json(Path(progress["out"]) / "coverage.json", {
        "task_id": task_id,
        "task_suite_name": "libero_goal",
        "executed_sites": sites,
        "groups": {site: screen.group(site) for site in sites},
        "selected_modules": list(screen.q.modules),
        "full_path": True,
        "expected_site_count": EXPECTED_SITE_COUNT,
        "action_mode": "first_frame",
        "num_inference_steps": 20,
        "action_horizon": 32,
    })
    save_json(Path(progress["out"]) / "reference_gate.json", {
        "passed": True,
        "task_id": task_id,
        "initial_state_id": state_id,
        "sampler_seed": s_seed,
        "draw_seed": d_seed,
        "repeat_max_abs": repeated_error,
        "hook_disabled_identity_max_abs": identity_error,
        "tolerance": 1e-6,
    })
    return learned


def protocol_for(screen, task_id, description, schedule):
    fixed = screen.cfg.EVALUATION
    return {
        "protocol_id": PROTOCOL_ID,
        "task_suite_name": "libero_goal",
        "task_id": task_id,
        "task_description": str(description),
        "checkpoint": str(Path(screen.cfg.ckpt).resolve()),
        "dataset_stats": str(Path(fixed.dataset_stats_path).resolve()),
        "model_dtype": str(screen.model.torch_dtype),
        "action_mode": "first_frame",
        "num_inference_steps_cfg_and_actual": 20,
        "sigma_shift": 1.0,
        "compile_action_infer": False,
        "action_horizon": 32,
        "warmup_steps": 30,
        "replan_steps": 10,
        "max_control_steps": 400,
        "use_action_ensembler": False,
        "binarize_gripper": True,
        "expected_site_count": EXPECTED_SITE_COUNT,
        "locked_selection_source_job_id": "25666715.pbs101",
        "arms": list(ARMS),
        "initial_state_ids": list(STATE_IDS),
        "quant_arm_order_seed": ORDER_SEED + task_id,
        "quant_arm_order_by_state": {str(k): list(v) for k, v in schedule.items()},
        "seed_rule": {
            "environment": "3026100000 + task_id*100000 + initial_state_id",
            "sampler": "1026100000 + task_id*100000 + initial_state_id*100 + replan_index",
            "draw": "2026100000 + task_id*100000 + initial_state_id*100 + replan_index",
        },
        "score_scope": "fixed-protocol LIBERO-Goal suite score; not a paper-environment/seed reproduction",
    }


def trace_path_for(task_id, arm, state_id):
    return Path("traces") / f"task-{task_id:02d}__{arm}__init-{state_id:02d}.pt"


def resume_completed(resume_dir, out_dir, protocol, execution_state_ids):
    src = Path(resume_dir).resolve()
    if src == Path(out_dir).resolve():
        raise GateError("--resume-dir must be a read-only source different from --out")
    progress = load_json(src / "progress.json")
    if progress.get("protocol") != protocol:
        raise GateError("Resume protocol/task/checkpoint/locked-table identity mismatch")
    if not progress.get("coverage_validated") or not progress.get("reference_gate"):
        raise GateError("Resume source lacks the required BF16 coverage/reference gates")
    coverage = load_json(src / "coverage.json")
    gate = load_json(src / "reference_gate.json")
    if (coverage.get("task_id") != protocol["task_id"]
            or len(coverage.get("executed_sites", [])) != EXPECTED_SITE_COUNT
            or not gate.get("passed") or gate.get("task_id") != protocol["task_id"]):
        raise GateError("Resume source task coverage/reference gate is invalid")

    rows = load_json(src / "episodes.json")
    index = load_json(src / "trace_index.json")
    source_state_ids = validate_execution_state_ids(
        progress.get("execution_state_ids", list(STATE_IDS))
    )
    indexed = index.get("episodes", {})
    if not isinstance(rows, list) or not isinstance(indexed, dict):
        raise GateError("Resume episodes/trace index has an invalid format")
    seen, accepted, copied_index, ignored = set(), [], {}, []
    for row in rows:
        key = slot_key(row.get("arm"), row.get("state_id", -1))
        if key in seen:
            raise GateError(f"Resume source repeats slot {key}")
        seen.add(key)
        if (row.get("task_id") != protocol["task_id"]
                or row.get("arm") not in ARMS or row.get("state_id") not in STATE_IDS):
            raise GateError(f"Resume source contains an out-of-protocol slot: {key}")
        if row["state_id"] not in source_state_ids:
            raise GateError(f"Resume source contains an undeclared execution state: {key}")
        if row["state_id"] not in execution_state_ids:
            continue
        if not row.get("complete"):
            ignored.append(key)
            continue
        state_id = row["state_id"]
        replans = row.get("replans")
        if (not isinstance(row.get("success"), bool) or not isinstance(replans, int)
                or replans < 0 or row.get("environment_seed") != environment_seed(protocol["task_id"], state_id)
                or row.get("initial_state_id") != state_id):
            raise GateError(f"Resume completed slot metadata is invalid: {key}")
        expected_sampler = [sampler_seed(protocol["task_id"], state_id, i) for i in range(replans)]
        expected_draw = [draw_seed(protocol["task_id"], state_id, i) for i in range(replans)]
        if row.get("sampler_seeds") != expected_sampler or row.get("draw_seeds") != expected_draw:
            raise GateError(f"Resume completed slot seeds are invalid: {key}")
        item = indexed.get(key)
        if not item or item.get("trace_file") != row.get("trace_file"):
            raise GateError(f"Resume slot has no matching trace index entry: {key}")
        rel = Path(row["trace_file"])
        source_trace = (src / rel).resolve()
        if rel.is_absolute() or not source_trace.is_relative_to(src) or not source_trace.is_file() or source_trace.stat().st_size == 0:
            raise GateError(f"Resume slot trace is missing or outside its source directory: {key}")
        if item.get("trace_entries") != replans or item.get("complete") is not True:
            raise GateError(f"Resume slot trace metadata is invalid: {key}")
        destination = Path(out_dir) / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_trace, destination)
        accepted.append(row)
        copied_index[key] = item
    return accepted, copied_index, ignored


def task_summaries(records, task_id, execution_state_ids=STATE_IDS):
    state_ids = tuple(execution_state_ids)
    state_set = set(state_ids)
    denominator = len(state_ids)
    by_arm = {arm: [row for row in records if row["arm"] == arm] for arm in ARMS}
    summaries = {}
    for arm, rows in by_arm.items():
        complete = [row for row in rows if row.get("complete") is True]
        successes = sum(row["success"] is True for row in complete)
        all_arm_slots = len(complete) == denominator and {row["state_id"] for row in complete} == state_set
        summaries[arm] = {
            "attempted": len(rows),
            "complete": len(complete),
            "successes": successes,
            "denominator": denominator,
            "success_rate": successes / denominator if all_arm_slots else None,
            "timeout_count": sum(row.get("termination_reason") == "timeout" for row in complete),
            "exception_count": sum(row.get("termination_reason") == "exception" for row in rows),
        }
    expected_slots = {(arm, state_id) for arm in ARMS for state_id in state_ids}
    actual_slots = {(row["arm"], row["state_id"]) for row in records}
    all_slots = len(records) == len(expected_slots) and actual_slots == expected_slots
    all_complete = all_slots and all(summaries[arm]["complete"] == denominator for arm in ARMS)
    comparisons = {}
    for left, right, name in (
        ("bf16", "rtn", "bf16_vs_rtn"),
        ("bf16", "independent", "bf16_vs_independent"),
        ("bf16", "learned", "bf16_vs_learned"),
        ("learned", "rtn", "learned_vs_rtn"),
        ("learned", "independent", "learned_vs_independent"),
    ):
        a = {r["state_id"]: r for r in by_arm[left] if r.get("complete")}
        b = {r["state_id"]: r for r in by_arm[right] if r.get("complete")}
        differences = []
        for state_id in state_ids:
            if state_id not in a or state_id not in b:
                continue
            differences.append({
                "initial_state_id": state_id,
                "left_success": a[state_id]["success"],
                "right_success": b[state_id]["success"],
                "difference_left_minus_right": int(a[state_id]["success"]) - int(b[state_id]["success"]),
            })
        comparisons[name] = {
            "left_arm": left,
            "right_arm": right,
            "paired_slots": len(differences),
            "left_gain": sum(x["difference_left_minus_right"] == 1 for x in differences),
            "right_gain": sum(x["difference_left_minus_right"] == -1 for x in differences),
            "ties": sum(x["difference_left_minus_right"] == 0 for x in differences),
            "per_initial_state": differences,
        }
    return {
        "task_id": task_id,
        "complete": all_complete,
        "slots_attempted": len(records),
        "slots_complete": sum(bool(r.get("complete")) for r in records),
        "arm_summaries": summaries,
        "paired_outcomes": comparisons,
    }


def write_task_result(path, protocol, records, progress):
    execution_state_ids = validate_execution_state_ids(
        progress.get("execution_state_ids", list(STATE_IDS))
    )
    summary = task_summaries(records, protocol["task_id"], execution_state_ids)
    full_task = execution_state_ids == STATE_IDS
    result = {
        "scope": ("full 10-task LIBERO-Goal suite shard; 50 initial states per arm" if full_task
                  else f"LIBERO-Goal state shard for task {protocol['task_id']}; {len(execution_state_ids)} initial states per arm"),
        "score_scope": protocol["score_scope"],
        "execution_state_ids": list(execution_state_ids),
        "protocol": protocol,
        **summary,
        "coverage_validated": progress.get("coverage_validated", False),
        "reference_gate_passed": progress.get("reference_gate", False),
        "records_file": "episodes.json",
        "trace_index_file": "trace_index.json",
        "trace_storage": "one CPU .pt file per episode; normalized action chunks and executed official commands",
    }
    save_json(path, result)
    return result


def run_slot(screen, *, task_id, arm, state_id, initial_state, learned, progress, records, trace_index, out_dir, torch):
    key = slot_key(arm, state_id)
    slot_trace, episode_state = [], {}
    env_seed = environment_seed(task_id, state_id)
    try:
        with open(os.devnull, "w", encoding="utf-8") as quiet, contextlib.redirect_stdout(quiet):
            row = closedloop.run_episode(
                screen,
                arm=arm,
                state_id=state_id,
                initial_state=initial_state,
                locked_phases=learned if arm == "learned" else {},
                env_seed=env_seed,
                progress=progress,
                run_state={"learned_phases": learned},
                episode_state=episode_state,
                trace=slot_trace,
                completed_count=len(records),
                total_slots=progress["total_slots"],
            )
    except (AssertionError, GateError, closedloop.GateError):
        raise
    except Exception as exc:
        row = {
            "arm": arm,
            "state_id": state_id,
            "environment_seed": env_seed,
            "sampler_seeds": list(episode_state.get("sampler_seeds", [])),
            "draw_seeds": list(episode_state.get("draw_seeds", [])),
            "complete": False,
            "success": None,
            "termination_reason": "exception",
            "error_type": type(exc).__name__,
            "error": str(exc)[:400],
            "warmup_steps": episode_state.get("warmup_steps", 0),
            "active_steps": episode_state.get("active_steps", 0),
            "total_env_steps": episode_state.get("warmup_steps", 0) + episode_state.get("active_steps", 0),
            "replans": episode_state.get("replans", len(slot_trace)),
        }

    for entry in slot_trace:
        entry["task_id"] = task_id
        entry["initial_state_id"] = state_id
    row.update({"task_id": task_id, "initial_state_id": state_id})
    relative_trace = trace_path_for(task_id, arm, state_id)
    trace_file = Path(out_dir) / relative_trace
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    temp_trace = trace_file.with_suffix(trace_file.suffix + ".tmp")
    torch.save(slot_trace, temp_trace)
    os.replace(temp_trace, trace_file)
    row["trace_file"] = relative_trace.as_posix()
    trace_index["episodes"][key] = {
        "trace_file": row["trace_file"],
        "trace_entries": len(slot_trace),
        "complete": bool(row.get("complete")),
    }
    records.append(row)
    progress["recorded_by_arm"][arm] = sum(r["arm"] == arm for r in records)
    save_json(Path(out_dir) / "episodes.json", records)
    save_json(Path(out_dir) / "trace_index.json", trace_index)
    progress["completed_slots"] = sum(bool(r.get("complete")) for r in records)
    progress["attempted_slots"] = len(records)
    progress["current"] = None
    save_json(Path(out_dir) / "progress.json", progress)
    write_task_result(Path(out_dir) / "result.json", progress["protocol"], records, progress)
    print(
        f"EPISODE task={task_id} arm={arm} init={state_id} complete={row['complete']} "
        f"success={row['success']} active_steps={row['active_steps']}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, choices=range(10), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--locked", type=Path, default=DEFAULT_LOCKED)
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--state-start", type=int, default=0,
                        help="inclusive global initial-state ID (default: 0)")
    parser.add_argument("--state-stop", type=int, default=50,
                        help="exclusive global initial-state ID (default: 50)")
    args = parser.parse_args()
    try:
        execution_state_ids = state_ids_for_range(args.state_start, args.state_stop)
    except ValueError as exc:
        parser.error(str(exc))

    screen_runner.guard()
    self_check()
    out_dir = args.out.resolve()
    if args.resume_dir and out_dir == args.resume_dir.resolve():
        raise GateError("--out and --resume-dir must be different directories")
    if any((out_dir / name).exists() for name in ("episodes.json", "progress.json", "result.json", "trace_index.json")):
        raise GateError(f"Output already contains suite records: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    locked_path = args.locked.resolve()
    schedule = order_schedule(args.task_id)
    screen = screen_runner.Screen(out_dir)
    records, trace_index = [], {"protocol_id": PROTOCOL_ID, "task_id": args.task_id, "episodes": {}}
    progress = {
        "status": "initializing",
        "out": str(out_dir),
        "locked_path": str(locked_path),
        "coverage_validated": False,
        "reference_gate": False,
        "completed_slots": 0,
        "attempted_slots": 0,
        "total_slots": len(execution_state_ids) * len(ARMS),
        "execution_state_ids": list(execution_state_ids),
        "recorded_by_arm": {arm: 0 for arm in ARMS},
        "current": None,
        "protocol": None,
        "resumed_from": str(args.resume_dir.resolve()) if args.resume_dir else None,
        "resumed_incomplete_slots": [],
        "locked_source": None,
    }
    save_json(out_dir / "episodes.json", records)
    save_json(out_dir / "trace_index.json", trace_index)
    save_json(out_dir / "progress.json", progress)
    try:
        import torch
        from libero.libero import get_libero_path

        if screen.model.torch_dtype != torch.bfloat16 or screen.horizon != 32:
            raise GateError(f"Expected BF16/action_horizon=32, got {screen.model.torch_dtype}/{screen.horizon}")
        inherited_steps = int(screen.cfg.EVALUATION.num_inference_steps)
        screen.cfg.EVALUATION.num_inference_steps = 20
        task, description = switch_task(screen, args.task_id)
        install_action_range_gate(screen)
        fixed = screen.cfg.EVALUATION
        if (str(fixed.action_infer_mode) != "first_frame"
                or int(fixed.num_inference_steps) != 20
                or float(fixed.sigma_shift) != 1.0
                or bool(fixed.compile_action_infer)
                or bool(fixed.use_action_ensembler)
                or not bool(fixed.binarize_gripper)
                or int(fixed.num_steps_wait) != 30
                or int(fixed.replan_steps) != 10
                or int(screen.ev._get_max_steps("libero_goal")) != 400):
            raise GateError("Evaluator config no longer matches the frozen full-suite protocol")
        if not locked_path.is_file():
            raise GateError(f"Locked learned-phase table is missing: {locked_path}")
        initial_path = (
            Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        )
        initial_states = torch.load(initial_path, weights_only=False)
        if len(initial_states) < 50 or any(initial_states[i] is None for i in STATE_IDS):
            raise GateError(f"Task {args.task_id} does not provide 50 nonempty initial states")
        screen.cfg.EVALUATION.num_trials = 50
        closedloop.SAMPLER_SEED_BASE = SAMPLER_SEED_BASE + args.task_id * TASK_SEED_STRIDE
        closedloop.DRAW_SEED_BASE = DRAW_SEED_BASE + args.task_id * TASK_SEED_STRIDE
        protocol = protocol_for(screen, args.task_id, description, schedule)
        progress["protocol"] = protocol
        progress["status"] = "running"
        progress["inherited_num_inference_steps"] = inherited_steps
        progress["task_problem_folder"] = str(task.problem_folder)
        progress["task_init_states_file"] = str(task.init_states_file)

        if args.resume_dir:
            records, accepted_index, ignored = resume_completed(
                args.resume_dir, out_dir, protocol, execution_state_ids
            )
            trace_index["episodes"].update(accepted_index)
            progress["resumed_incomplete_slots"] = ignored
            progress["completed_slots"] = len(records)
            progress["attempted_slots"] = len(records)
            for row in records:
                progress["recorded_by_arm"][row["arm"]] += 1
        learned = establish_task_gates(
            screen, task_id=args.task_id, initial_state=initial_states[0],
            progress=progress, locked_path=locked_path,
        )
        save_json(out_dir / "episodes.json", records)
        save_json(out_dir / "trace_index.json", trace_index)
        save_json(out_dir / "progress.json", progress)
        save_json(out_dir / "runtime_config.json", {
            "checkpoint": str(screen.cfg.ckpt),
            "dataset_stats": str(fixed.dataset_stats_path),
            "task_suite_name": "libero_goal",
            "task_id": args.task_id,
            "task_description": str(description),
            "action_mode": "first_frame",
            "num_inference_steps_cfg_and_actual": 20,
            "inherited_num_inference_steps": inherited_steps,
            "sigma_shift": 1.0,
            "compile_action_infer": False,
            "torch_dtype": str(screen.model.torch_dtype),
            "action_horizon": screen.horizon,
            "warmup_steps": 30,
            "replan_steps": 10,
            "max_control_steps": 400,
            "expected_executed_sites": EXPECTED_SITE_COUNT,
            "pbs_jobid": os.environ.get("PBS_JOBID"),
        })

        # Complete every BF16 slot before the single in-place W4 weight conversion.
        done = {(row["arm"], row["state_id"]) for row in records if row.get("complete")}
        for state_id in execution_state_ids:
            if ("bf16", state_id) in done:
                continue
            progress["current"] = {"arm": "bf16", "state_id": state_id}
            progress["recorded_by_arm"]["bf16"] = sum(r["arm"] == "bf16" for r in records)
            run_slot(
                screen, task_id=args.task_id, arm="bf16", state_id=state_id,
                initial_state=initial_states[state_id], learned=learned, progress=progress,
                records=records, trace_index=trace_index, out_dir=out_dir, torch=torch,
            )
            if records[-1].get("complete"):
                done.add(("bf16", state_id))

        incomplete_bf16 = [s for s in execution_state_ids if ("bf16", s) not in done]
        if incomplete_bf16:
            progress["status"] = "incomplete"
            progress["incomplete_bf16_slots"] = incomplete_bf16
            save_json(out_dir / "progress.json", progress)
            write_task_result(out_dir / "result.json", protocol, records, progress)
            raise RuntimeError("BF16 episodes remain incomplete; W4 conversion was withheld")

        screen.q.weight_quantize()
        done = {(row["arm"], row["state_id"]) for row in records if row.get("complete")}
        for state_id in execution_state_ids:
            for arm in schedule[state_id]:
                if (arm, state_id) in done:
                    continue
                progress["current"] = {"arm": arm, "state_id": state_id}
                progress["recorded_by_arm"][arm] = sum(r["arm"] == arm for r in records)
                run_slot(
                    screen, task_id=args.task_id, arm=arm, state_id=state_id,
                    initial_state=initial_states[state_id], learned=learned, progress=progress,
                    records=records, trace_index=trace_index, out_dir=out_dir, torch=torch,
                )
                if records[-1].get("complete"):
                    done.add((arm, state_id))

        summary = task_summaries(records, args.task_id, execution_state_ids)
        progress["status"] = "complete" if summary["complete"] else "incomplete"
        progress["completed_slots"] = summary["slots_complete"]
        progress["attempted_slots"] = len(records)
        progress["current"] = None
        save_json(out_dir / "progress.json", progress)
        result = write_task_result(out_dir / "result.json", protocol, records, progress)
        print(
            f"TASK_{'COMPLETE' if result['complete'] else 'INCOMPLETE'} task={args.task_id} "
            f"slots={result['slots_complete']}/{progress['total_slots']}",
            flush=True,
        )
        if not result["complete"]:
            raise RuntimeError("Task shard has incomplete episode slots; no suite score is eligible")
    except Exception as exc:
        progress["status"] = "incomplete" if records else "failed_gate"
        progress["current"] = None
        progress["error_type"] = type(exc).__name__
        progress["error"] = "runtime/gate failure; details omitted"
        save_json(out_dir / "progress.json", progress)
        if progress.get("protocol"):
            write_task_result(out_dir / "result.json", progress["protocol"], records, progress)
        raise
    finally:
        close = getattr(screen.env, "close", None)
        if close is not None:
            close()
        screen.q.close()


if __name__ == "__main__":
    main()
