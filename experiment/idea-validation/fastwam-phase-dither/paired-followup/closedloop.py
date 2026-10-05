"""Paired task-0 closed-loop follow-up; run only inside an approved PBS allocation."""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCREEN_DIR = ROOT / "mechanism-screen"
sys.path.insert(0, str(SCREEN_DIR))

import runner as screen_runner  # noqa: E402

sys.path.insert(0, str(screen_runner.BASE / "FastWAM"))


STATE_IDS = tuple(range(16, 24))
ARMS = ("bf16", "rtn", "independent", "learned")
QUANT_ARMS = ("rtn", "independent", "learned")
EXPECTED_SITE_COUNT = 6446
ORDER_SEED = 20261003
SAMPLER_SEED_BASE = 1_026_100_000
DRAW_SEED_BASE = 2_026_100_000
ENV_SEED_BASE = 3_026_100_000


class GateError(RuntimeError):
    """A frozen-protocol or numerical gate failed; stop instead of recording noise."""


def save_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def order_schedule():
    rng = random.Random(ORDER_SEED)
    base = list(QUANT_ARMS)
    rng.shuffle(base)
    rotations = [base[i:] + base[:i] for i in range(3)]
    offsets = [0, 1, 2, 0, 1, 2, 0, 1]
    rng.shuffle(offsets)
    return {state_id: rotations[offsets[i]] for i, state_id in enumerate(STATE_IDS)}


def validate_locked(screen, locked_path):
    if screen.expected_sites is None:
        raise GateError("No traced BF16 call established expected_sites")
    sites = screen.expected_sites
    if len(sites) != EXPECTED_SITE_COUNT or len(set(sites)) != EXPECTED_SITE_COUNT:
        raise GateError(f"Expected {EXPECTED_SITE_COUNT} unique executed sites, got {len(sites)}")
    data = json.loads(Path(locked_path).read_text(encoding="utf-8"))
    learned = data.get("learned")
    if not isinstance(learned, dict) or set(learned) != set(sites):
        missing = len(set(sites) - set(learned or {}))
        extra = len(set(learned or {}) - set(sites))
        raise GateError(f"locked learned keyset mismatch: missing={missing} extra={extra}")
    learned = {site: float(learned[site]) for site in sites}
    if any(not math.isfinite(value) or not 0.0 <= value < 1.0 for value in learned.values()):
        raise GateError("locked learned phases must all be finite and in [0, 1)")
    return learned, data


def _model_input(screen, obs):
    torch = screen.torch
    image, proprio, _ = screen.ev._obs_to_model_input(
        obs,
        cfg=screen.cfg,
        processor=screen.processor,
        width=screen.width,
        height=screen.height,
        device="cuda",
        dtype=screen.model.torch_dtype,
    )
    if not torch.is_tensor(proprio):
        proprio = torch.as_tensor(proprio, dtype=screen.model.torch_dtype)
    return {"image": image, "proprio": proprio}


def _official_action(screen, normalized):
    import numpy as np

    action = screen.ev._denormalize_action(normalized, screen.processor)[0]
    action[..., -1] = action[..., -1] * 2 - 1
    action = screen.ev.invert_gripper_action(action)
    if bool(screen.cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    return action


def run_episode(screen, *, arm, state_id, initial_state, locked_phases, env_seed,
                progress, run_state, episode_state, trace, completed_count, total_slots):
    torch = screen.torch
    env = screen.env
    warmup = int(screen.cfg.EVALUATION.num_steps_wait)
    replan_steps = int(screen.cfg.EVALUATION.replan_steps)
    max_steps = int(screen.ev._get_max_steps("libero_goal"))
    if warmup != 30 or replan_steps != 10 or max_steps != 400:
        raise GateError(f"Frozen evaluator budget changed: warmup={warmup}, replan={replan_steps}, max={max_steps}")

    state = episode_state
    state.update({"active_steps": 0, "warmup_steps": 0, "replans": 0,
                  "sampler_seeds": [], "draw_seeds": []})
    env.seed(env_seed)
    env.reset()
    obs = env.set_init_state(initial_state)
    done = False
    termination = None

    for _ in range(warmup):
        obs, _, done, _ = env.step(screen.ev.get_libero_dummy_action())
        state["warmup_steps"] += 1
        if done:
            termination = "env_done_during_warmup"
            break

    while termination is None and state["active_steps"] < max_steps:
        replan_idx = state["replans"]
        sampler_seed = SAMPLER_SEED_BASE + state_id * 100 + replan_idx
        draw_seed = DRAW_SEED_BASE + state_id * 100 + replan_idx
        datum = _model_input(screen, obs)
        mode = {"bf16": "bf16", "rtn": "rtn", "independent": "independent", "learned": "phase"}[arm]
        phases = locked_phases if arm == "learned" else {}
        normalized, stats, seconds = screen.infer(
            datum, mode, sampler_seed, draw_seed, phases=phases
        )
        if normalized.ndim != 2 or normalized.shape[0] != 32:
            raise GateError(f"Expected normalized [32,D] action chunk, got {tuple(normalized.shape)}")
        if not bool(torch.isfinite(normalized).all()):
            raise GateError(f"Nonfinite normalized action chunk: arm={arm} state_id={state_id} replan={replan_idx}")

        if screen.expected_sites is not None and not progress["coverage_validated"]:
            learned, locked_data = validate_locked(screen, progress["locked_path"])
            run_state["learned_phases"] = learned
            progress["locked_source"] = {
                "path": progress["locked_path"],
                "selected_keys": len(learned),
                "permutation_nongauge": locked_data.get("permutation_nongauge"),
            }
            progress["coverage_validated"] = True
            save_json(Path(progress["out"]) / "coverage.json", {
                "executed_sites": screen.expected_sites,
                "groups": {site: screen.group(site) for site in screen.expected_sites},
                "selected_modules": list(screen.q.modules),
                "full_path": True,
                "expected_site_count": EXPECTED_SITE_COUNT,
                "action_mode": "first_frame",
                "num_inference_steps": 20,
                "action_horizon": 32,
            })

        if arm == "bf16" and not progress["reference_gate"]:
            repeat = screen.infer(datum, "bf16", sampler_seed)[0]
            repeat_error = float((repeat - normalized).abs().max())
            unhooked = screen.infer(datum, "bf16", sampler_seed, trace=False)[0]
            identity_error = float((unhooked - normalized).abs().max())
            if repeat_error > 1e-6 or identity_error > 1e-6:
                raise GateError(f"BF16 reference gate failed: repeat={repeat_error} identity={identity_error}")
            save_json(Path(progress["out"]) / "reference_gate.json", {
                "passed": True,
                "state_id": state_id,
                "replan_index": replan_idx,
                "sampler_seed": sampler_seed,
                "repeat_max_abs": repeat_error,
                "hook_disabled_identity_max_abs": identity_error,
                "tolerance": 1e-6,
            })
            progress["reference_gate"] = True

        command = _official_action(screen, normalized.clone())
        if command.shape != (32, 7) or not bool(torch.isfinite(torch.as_tensor(command)).all()):
            raise GateError(f"Expected finite official [32,7] commands, got {command.shape}")
        executed = min(replan_steps, max_steps - state["active_steps"])
        trace_entry = {
            "arm": arm,
            "state_id": state_id,
            "replan_index": replan_idx,
            "active_step_start": state["active_steps"],
            "sampler_seed": sampler_seed,
            "draw_seed": draw_seed,
            "mode": mode,
            "inference_seconds_fake_quant": seconds,
            "stats": {key: stats[key] for key in (
                "nonfinite_values", "overload_codes", "zero_rows", "quantized_calls", "site_count"
            )},
            "executed_count": 0,
            "normalized_action_chunk": normalized.detach().to(dtype=torch.float32, device="cpu").clone(),
            "official_command_chunk": torch.as_tensor(command, dtype=torch.float32),
        }
        trace.append(trace_entry)
        state["sampler_seeds"].append(sampler_seed)
        state["draw_seeds"].append(draw_seed)
        state["replans"] += 1
        print(f"REPLAN arm={arm} state_id={state_id} index={replan_idx} steps={state['active_steps']} sites={stats['site_count']}", flush=True)
        progress["current"] = {
            "arm": arm, "state_id": state_id, "replan_index": replan_idx,
            "active_steps": state["active_steps"], "sampler_seed": sampler_seed,
            "draw_seed": draw_seed,
        }
        progress["completed_slots"] = completed_count
        progress["total_slots"] = total_slots
        save_json(Path(progress["out"]) / "progress.json", progress)

        executed_commands = []
        try:
            for command_step in command[:executed]:
                obs, _, done, _ = env.step(command_step.tolist())
                executed_commands.append(command_step.copy())
                state["active_steps"] += 1
                if done:
                    termination = "env_done"
                    break
        finally:
            trace_entry["executed_count"] = len(executed_commands)
            trace_entry["executed_commands"] = torch.as_tensor(
                executed_commands, dtype=torch.float32
            ).reshape(-1, 7)

    if termination is None:
        termination = "timeout"
    return {
        "arm": arm,
        "state_id": state_id,
        "environment_seed": env_seed,
        "sampler_seeds": list(state["sampler_seeds"]),
        "draw_seeds": list(state["draw_seeds"]),
        "complete": True,
        "success": bool(done),
        "termination_reason": termination,
        "warmup_steps": state["warmup_steps"],
        "active_steps": state["active_steps"],
        "total_env_steps": state["warmup_steps"] + state["active_steps"],
        "replans": state["replans"],
    }


def paired_outcomes(records):
    by_slot = {(row["arm"], row["state_id"]): row for row in records}
    pairs = {}
    for left, right, name in (
        ("bf16", "rtn", "bf16_vs_rtn"),
        ("bf16", "independent", "bf16_vs_independent"),
        ("bf16", "learned", "bf16_vs_learned"),
        ("learned", "rtn", "learned_vs_rtn"),
        ("learned", "independent", "learned_vs_independent"),
    ):
        rows = []
        for state_id in STATE_IDS:
            a, b = by_slot.get((left, state_id)), by_slot.get((right, state_id))
            valid = bool(a and b and a.get("complete") and b.get("complete"))
            rows.append({
                "state_id": state_id,
                "complete": valid,
                "left_success": a.get("success") if a else None,
                "right_success": b.get("success") if b else None,
                "success_difference_left_minus_right": (
                    int(bool(a["success"])) - int(bool(b["success"])) if valid else None
                ),
            })
        pairs[name] = rows
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--locked", type=Path, required=True)
    args = parser.parse_args()

    screen_runner.guard()
    args.out.mkdir(parents=True, exist_ok=True)
    screen = screen_runner.Screen(args.out)
    # Screen.infer explicitly uses 20; align the inherited train-config value (10).
    inherited_steps = int(screen.cfg.EVALUATION.num_inference_steps)
    screen.cfg.EVALUATION.num_inference_steps = 20
    save_json(args.out / "controller_config.json", {
        "inherited_num_inference_steps": inherited_steps,
        "num_inference_steps": int(screen.cfg.EVALUATION.num_inference_steps),
        "actual_infer_steps": 20,
        "warmup_steps": int(screen.cfg.EVALUATION.num_steps_wait),
        "replan_steps": int(screen.cfg.EVALUATION.replan_steps),
        "binarize_gripper": bool(screen.cfg.EVALUATION.binarize_gripper),
        "use_action_ensembler": bool(screen.cfg.EVALUATION.use_action_ensembler),
        "action_mode": str(screen.cfg.EVALUATION.action_infer_mode),
        "compile_action_infer": bool(screen.cfg.EVALUATION.compile_action_infer),
        "sigma_shift": float(screen.cfg.EVALUATION.sigma_shift),
    })
    records, all_trace = [], []
    schedule = order_schedule()
    progress = {
        "status": "running", "out": str(args.out.resolve()),
        "locked_path": str(args.locked.resolve()), "coverage_validated": False,
        "reference_gate": False, "completed_slots": 0, "total_slots": 32,
        "recorded_by_arm": {arm: 0 for arm in ARMS}, "current": None,
        "order_seed": ORDER_SEED,
        "quant_arm_order_by_state": {str(k): v for k, v in schedule.items()},
        "locked_source": None,
    }
    run_state = {"learned_phases": None}
    try:
        import torch
        from libero.libero import get_libero_path

        if screen.model.torch_dtype != torch.bfloat16 or screen.horizon != 32:
            raise GateError(f"Expected BF16/action_horizon=32, got {screen.model.torch_dtype}/{screen.horizon}")
        fixed = screen.cfg.EVALUATION
        if (str(fixed.action_infer_mode) != "first_frame"
                or int(fixed.num_inference_steps) != 20
                or float(fixed.sigma_shift) != 1.0
                or bool(fixed.compile_action_infer)
                or bool(fixed.use_action_ensembler)
                or not bool(fixed.binarize_gripper)):
            raise GateError("Fast-WAM evaluator config no longer matches the frozen controller settings")
        initial_path = (
            Path(get_libero_path("init_states"))
            / screen.task.problem_folder
            / screen.task.init_states_file
        )
        initial_states = torch.load(initial_path, weights_only=False)
        if len(initial_states) <= max(STATE_IDS):
            raise GateError(f"LIBERO task-0 init states lack all requested IDs 16..23 (count={len(initial_states)})")
        if any(initial_states[state_id] is None for state_id in STATE_IDS):
            raise GateError("One or more requested LIBERO task-0 initial states are empty")

        # Run every BF16 episode before the single in-place W4 weight conversion.
        for arm in ("bf16",):
            for state_id in STATE_IDS:
                slot_trace = []
                row = None
                episode_state = {}
                env_seed = ENV_SEED_BASE + state_id
                try:
                    row = run_episode(
                        screen, arm=arm, state_id=state_id, initial_state=initial_states[state_id],
                        locked_phases={}, env_seed=env_seed, progress=progress, run_state=run_state,
                        episode_state=episode_state, trace=slot_trace,
                        completed_count=len(records), total_slots=32,
                    )
                except AssertionError:
                    raise
                except GateError:
                    raise
                except Exception as exc:
                    row = {
                        "arm": arm, "state_id": state_id, "environment_seed": env_seed,
                        "sampler_seeds": list(episode_state.get("sampler_seeds", [])),
                        "draw_seeds": list(episode_state.get("draw_seeds", [])), "complete": False,
                        "success": None, "termination_reason": "exception",
                        "error_type": type(exc).__name__, "error": str(exc),
                        "warmup_steps": episode_state.get("warmup_steps", 0),
                        "active_steps": episode_state.get("active_steps", 0),
                        "total_env_steps": (episode_state.get("warmup_steps", 0)
                                             + episode_state.get("active_steps", 0)),
                        "replans": episode_state.get("replans", len(slot_trace)),
                    }
                records.append(row)
                all_trace.extend(slot_trace)
                progress["recorded_by_arm"][arm] += 1
                progress["completed_slots"] = len(records)
                progress["current"] = None
                save_json(args.out / "episodes.json", records)
                torch.save(all_trace, args.out / "trace.pt")
                print(f"EPISODE arm={arm} state_id={state_id} success={row['success']} termination={row['termination_reason']} active_steps={row['active_steps']}", flush=True)
                save_json(args.out / "progress.json", progress)

        if not progress["coverage_validated"] or not progress["reference_gate"]:
            raise GateError("BF16 coverage and reference identity gates must pass before W4 conversion")
        screen.q.weight_quantize()

        for state_id in STATE_IDS:
            for arm in schedule[state_id]:
                slot_trace = []
                row = None
                episode_state = {}
                env_seed = ENV_SEED_BASE + state_id
                try:
                    row = run_episode(
                        screen, arm=arm, state_id=state_id, initial_state=initial_states[state_id],
                        locked_phases=run_state["learned_phases"], env_seed=env_seed,
                        progress=progress, run_state=run_state, episode_state=episode_state,
                        trace=slot_trace, completed_count=len(records), total_slots=32,
                    )
                except AssertionError:
                    raise
                except GateError:
                    raise
                except Exception as exc:
                    row = {
                        "arm": arm, "state_id": state_id, "environment_seed": env_seed,
                        "sampler_seeds": list(episode_state.get("sampler_seeds", [])),
                        "draw_seeds": list(episode_state.get("draw_seeds", [])), "complete": False,
                        "success": None, "termination_reason": "exception",
                        "error_type": type(exc).__name__, "error": str(exc),
                        "warmup_steps": episode_state.get("warmup_steps", 0),
                        "active_steps": episode_state.get("active_steps", 0),
                        "total_env_steps": (episode_state.get("warmup_steps", 0)
                                             + episode_state.get("active_steps", 0)),
                        "replans": episode_state.get("replans", len(slot_trace)),
                    }
                records.append(row)
                all_trace.extend(slot_trace)
                progress["recorded_by_arm"][arm] += 1
                progress["completed_slots"] = len(records)
                progress["current"] = None
                save_json(args.out / "episodes.json", records)
                torch.save(all_trace, args.out / "trace.pt")
                print(f"EPISODE arm={arm} state_id={state_id} success={row['success']} termination={row['termination_reason']} active_steps={row['active_steps']}", flush=True)
                save_json(args.out / "progress.json", progress)

        by_arm = {
            arm: [row for row in records if row["arm"] == arm]
            for arm in ARMS
        }
        summaries = {
            arm: {
                "attempted": len(rows),
                "complete": sum(bool(row["complete"]) for row in rows),
                "successes": sum(bool(row["success"]) for row in rows if row["complete"]),
                "success_rate": (
                    sum(bool(row["success"]) for row in rows if row["complete"]) / 8
                    if len(rows) == 8 and all(row["complete"] for row in rows) else None
                ),
                "timeout_count": sum(row["termination_reason"] == "timeout" for row in rows),
                "exception_count": sum(row["termination_reason"] == "exception" for row in rows),
            }
            for arm, rows in by_arm.items()
        }
        all_slots = len(records) == 32 and all(summaries[arm]["attempted"] == 8 for arm in ARMS)
        all_complete = all_slots and all(summaries[arm]["complete"] == 8 for arm in ARMS)
        result = {
            "scope": "single LIBERO-goal task 0 paired closed-loop fake W4A8 follow-up",
            "fixed_model": {
                "task_config": "libero_optional_idm_2cam224_1e-4",
                "checkpoint": str(screen.cfg.ckpt),
                "torch_dtype": str(screen.model.torch_dtype),
                "action_mode": "first_frame",
                "num_inference_steps": 20,
                "sigma_shift": 1.0,
                "compile_action_infer": False,
                "action_horizon": 32,
            },
            "complete": all_complete,
            "all_32_slots_attempted": all_slots,
            "acceptance": {
                "init_state_ids": list(STATE_IDS),
                "episodes_per_arm": {arm: summaries[arm]["attempted"] for arm in ARMS},
                "all_arm_episodes_complete": all_complete,
                "coverage_site_count": len(screen.expected_sites or []),
                "coverage_validated": progress["coverage_validated"],
                "reference_gate_passed": progress["reference_gate"],
                "finite_budget": {"warmup_steps": 30, "max_control_steps": 400, "replan_steps": 10, "action_horizon": 32},
            },
            "official_semantics": {
                "source": "experiments/libero/eval_libero_single.py",
                "max_steps": 400,
                "warmup_steps": 30,
                "success": "env.step done boolean, matching official bool(done)",
                "timeout": "400 control steps without done; success=false, termination_reason=timeout",
                "gripper": "official _denormalize_action, last channel *2-1, invert_gripper_action, then configured np.sign",
            },
            "arm_summaries": summaries,
            "paired_outcomes": paired_outcomes(records),
            "quant_arm_order_by_state": progress["quant_arm_order_by_state"],
            "sampler_and_draw_seed_rule": "base + init_state_id*100 + replan_index; identical per paired arm, freshly initialized by Screen.infer/Quantizer.begin",
            "records_file": "episodes.json",
            "trace_file": "trace.pt",
        }
        save_json(args.out / "result.json", result)
        progress["status"] = "complete" if all_complete else "incomplete"
        progress["result"] = result
        progress["current"] = None
        save_json(args.out / "progress.json", progress)
        print(f"CLOSED_LOOP_{'COMPLETE' if all_complete else 'INCOMPLETE'} records={len(records)}/32", flush=True)
        if not all_complete:
            raise RuntimeError("At least one of the 32 paired episode slots raised an exception")
    finally:
        close = getattr(screen.env, "close", None)
        if close is not None:
            close()
        screen.q.close()


if __name__ == "__main__":
    main()
