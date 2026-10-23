"""Capture real LIBERO-Plus sensor observations and fixed-state diagnostics."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
import os
from pathlib import Path
import re

from control_common import Deadline, guard, read, save
from diagnostic_runtime import (ARMS, NoiseTap, SwitchingRuntime, datum_identity,
                                max_abs_difference, tensor_identity,
                                validate_binding)


CAPTURE_STEPS = (0, 10, 50, 100, 200)
NOISE_CAPTURE_STAGE = "agentview PIL input after wrapper dtype conversion, before the actual sensor transform"


def _sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _noise_module():
    from libero.libero.envs import OffScreenRenderEnv

    module = importlib.import_module(OffScreenRenderEnv.__module__)
    if any(not callable(getattr(module, name, None)) for name in NoiseTap.FUNCTIONS):
        raise RuntimeError(f"Noise transforms are not in {module.__name__}")
    return module


def _assert_frozen_seeds(case):
    index = int(case["index"])
    expected_environment = 800_000_000 + index * 1000
    expected_sampler = 900_000_000 + index * 1000
    if (int(case["environment_seed"]) != expected_environment
            or int(case["sampler_seed"]) != expected_sampler):
        raise RuntimeError(f"Frozen seed binding differs for manifest index {index}")


def _frozen_noise_level(case):
    task_name = case.get("row", {}).get("task_name")
    match = re.search(r"_noise_(\d+)$", task_name or "")
    if match is None:
        raise RuntimeError(f"Frozen task name has no sensor-noise suffix: {task_name!r}")
    level = int(match.group(1))
    if not 1 <= level <= 50:
        raise RuntimeError(f"Frozen sensor-noise level must be in [1,50], got {level}")
    return level


def _noise_identity(env, tap, expected_level=None):
    level = int(getattr(env, "noise", 0))
    if expected_level is not None and level != int(expected_level):
        raise RuntimeError(f"Sensor-noise level differs: plan={expected_level}, env={level}")
    if 1 <= level <= 10:
        name, severity = "motion_blur", level
    elif 11 <= level <= 20:
        name, severity = "gaussian_blur", level - 10
    elif 21 <= level <= 30:
        name, severity = "zoom_blur", level - 20
    elif 31 <= level <= 40:
        name, severity = "fog", level - 30
    elif 41 <= level <= 50:
        name, severity = "glass_blur", level - 40
    else:
        raise RuntimeError(f"Expected LIBERO-Plus sensor noise in [1,50], got {level}")
    if tap.last_clean is None or tap.last_function != name or tap.last_severity != severity:
        raise RuntimeError(
            f"Noise hook mismatch: expected {name}({severity}), got "
            f"{tap.last_function}({tap.last_severity})"
        )
    return {"level": level, "function": name, "severity": severity,
            "hook_calls_since_clear": tap.calls}


def _agent_meta(np, obs):
    value = np.asarray(obs["agentview_image"])
    return {"shape": list(value.shape), "dtype": str(value.dtype)}


def _save_npz(np, path, obs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {str(key): np.asarray(value) for key, value in obs.items()}
    np.savez_compressed(path, **arrays)


def _simstate(np, env):
    state = env.sim.get_state().flatten()
    return np.asarray(state).copy()


def _save_torch(path, torch, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_observation_metadata(path, expected):
    if not path.is_file():
        return False
    saved = _load_json(path)
    keys = ("protocol", "index", "row", "description", "environment_seed",
            "sampler_seed", "input_identity")
    mismatch = [key for key in keys if saved.get(key) != expected.get(key)]
    if mismatch:
        raise ValueError("Saved observation binding differs: " + ", ".join(mismatch))
    return True


def _query_paths(out, index, arm, variant):
    stem = f"{index}_{arm}_{variant}"
    return out / "records" / f"{stem}.pt", out / "records" / f"{stem}.json"


def _check_saved_query(torch, tensor_path, json_path, binding):
    if tensor_path.exists() != json_path.exists():
        raise RuntimeError(f"Partial query output pair exists: {tensor_path.stem}")
    if not tensor_path.exists():
        return None
    saved_json = _load_json(json_path)
    validate_binding(saved_json, binding)
    payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
    validate_binding(payload, binding)
    if (tuple(payload.get("action", torch.empty(0)).shape) != (32, 7)
            or tuple(payload.get("future", torch.empty(0)).shape) != (1, 48, 3, 14, 28)
            or payload.get("counts") != {"video_steps": 10, "action_steps": 10}
            or not isinstance(payload.get("action_initial_noise"), torch.Tensor)
            or not bool(torch.isfinite(payload["action_initial_noise"]).all())):
        raise RuntimeError(f"Saved query tensors or counts are invalid: {tensor_path}")
    if saved_json.get("action_initial_noise_identity") != tensor_identity(
            torch, payload["action_initial_noise"]):
        raise RuntimeError(f"Saved initial-noise identity differs: {tensor_path}")
    if ("commands" in payload and not _valid_command_shape(payload["commands"])):
        raise RuntimeError(f"Saved commands are not 32x7: {tensor_path}")
    if ("commands" in saved_json and not _valid_command_shape(saved_json["commands"])):
        raise RuntimeError(f"Saved JSON commands are not 32x7: {json_path}")
    if ("commands" in payload and "commands" in saved_json
            and payload["commands"] != saved_json["commands"]):
        raise RuntimeError(f"Saved command payload and JSON differ: {tensor_path}")
    return payload


def _valid_command_shape(commands):
    return (isinstance(commands, list) and len(commands) == 32
            and all(isinstance(row, list) and len(row) == 7 for row in commands))


def _tolist(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return value.tolist() if hasattr(value, "tolist") else value


def _input_delta(torch, noisy, clean):
    def pair(a, b):
        difference = (a.detach().float() - b.detach().float()).abs()
        return {"max_abs": float(difference.max()),
                "mean_abs": float(difference.mean()),
                "shape": list(a.shape), "dtype": str(a.dtype)}
    image = pair(noisy["image"], clean["image"])
    proprio = pair(noisy["proprio"], clean["proprio"])
    proprio["exactly_equal"] = bool(torch.equal(noisy["proprio"], clean["proprio"]))
    return {"image": image, "proprio": proprio}


def _rng_changes(torch, before, after):
    """Report native inference effects without restoring any environment RNG."""
    # capture_rng stores NumPy state as a dict of Python scalars and lists.
    numpy_equal = before["numpy"] == after["numpy"]
    cuda_before, cuda_after = before.get("torch_cuda", []), after.get("torch_cuda", [])
    cuda_equal = (len(cuda_before) == len(cuda_after)
                  and all(torch.equal(left, right)
                          for left, right in zip(cuda_before, cuda_after)))
    return {"python_changed": before["python"] != after["python"],
            "numpy_changed": not bool(numpy_equal),
            "torch_cpu_changed": not torch.equal(before["torch_cpu"], after["torch_cpu"]),
            "torch_cuda_changed": not cuda_equal}


def _query_record(runtime, torch, unit, index, arm, variant, seed, description,
                  datum, identity, out, source_action=None):
    tensor_path, json_path = _query_paths(out, index, arm, variant)
    binding = {"protocol": runtime.p["protocol"], "unit": unit, "index": index,
               "arm": arm, "variant": variant, "seed": int(seed),
               "description": description, "input_identity": identity}
    saved = _check_saved_query(torch, tensor_path, json_path, binding)
    if saved is not None:
        saved_json = _load_json(json_path)
        if "commands" not in saved or "commands" not in saved_json:
            commands = runtime.runtime.command(saved["action"].clone()).tolist()
            saved["commands"] = commands
            saved_json["commands"] = commands
            _save_torch(tensor_path, torch, saved)
            save(json_path, saved_json)
        error = None
        if source_action is not None:
            error = max_abs_difference(saved["action"], source_action)
        return {**saved, "query_reused": True,
                "max_abs_vs_source_first_chunk": error}, True

    runtime.activate(arm)
    action, future, counts, noise = runtime.query(datum, seed, description)
    commands = runtime.runtime.command(action.clone()).tolist()
    if not _valid_command_shape(commands):
        raise RuntimeError("Runtime command conversion did not return 32x7 commands")
    source_error = None
    if source_action is not None:
        source_error = max_abs_difference(action, source_action)
    payload = {**binding, "counts": counts, "action": action,
               "future": future, "action_initial_noise": noise, "commands": commands,
               "max_abs_vs_source_first_chunk": source_error}
    _save_torch(tensor_path, torch, payload)
    metadata = {**binding, "counts": counts,
                "max_abs_vs_source_first_chunk": source_error,
                "action_shape": list(action.shape), "action_dtype": str(action.dtype),
                "future_shape": list(future.shape), "future_dtype": str(future.dtype),
                "action_initial_noise_shape": list(noise.shape),
                "action_initial_noise_dtype": str(noise.dtype),
                "action_initial_noise_identity": tensor_identity(torch, noise),
                "commands": commands, "commands_shape": [len(commands), len(commands[0])],
                "query_reused": False}
    save(json_path, metadata)
    return {**payload, "query_reused": False}, False


def _initial_case(runtime, tap, np, torch, unit, case, root, out, source_actions):
    index, row = int(case["index"]), case["row"]
    if any(not _valid_command_shape(source_actions[arm]["action"]) for arm in ARMS):
        raise RuntimeError(f"Frozen source action must be [32,7] at manifest index {index}")
    _assert_frozen_seeds(case)
    tap.clear()
    obs = runtime.runtime.task(row, index)
    description = runtime.runtime.description
    noise = _noise_identity(runtime.runtime.env, tap, _frozen_noise_level(case))
    clean_obs = tap.clean_observation(np, obs)
    clean_image, noisy_image = clean_obs["agentview_image"], obs["agentview_image"]
    if np.asarray(clean_image).shape != np.asarray(noisy_image).shape:
        raise RuntimeError("Clean and noisy agentview image shapes differ")

    noisy_datum, clean_datum = runtime.datum(obs), runtime.datum(clean_obs)
    ids = {"noisy": datum_identity(torch, noisy_datum),
           "clean": datum_identity(torch, clean_datum)}
    initial_dir = root / "data" / "sensor_initial" / str(index)
    noisy_path, clean_path = initial_dir / "observation_noisy.npz", initial_dir / "observation_clean.npz"
    state_path = initial_dir / "state.pt"
    meta_path = initial_dir / "metadata.json"
    metadata_binding = {"protocol": runtime.p["protocol"], "index": index,
                        "row": row, "description": description,
                        "environment_seed": case["environment_seed"],
                        "sampler_seed": case["sampler_seed"], "input_identity": ids}
    reused_obs = _validate_observation_metadata(meta_path, metadata_binding)
    if reused_obs:
        previous = _load_json(meta_path)
        if (not noisy_path.is_file() or not clean_path.is_file() or not state_path.is_file()
                or _sha256_file(noisy_path) != previous["observation_sha256"]["noisy"]
                or _sha256_file(clean_path) != previous["observation_sha256"]["clean"]):
            raise RuntimeError(f"Saved observation files changed for case {index}")
    else:
        _save_npz(np, noisy_path, obs)
        _save_npz(np, clean_path, clean_obs)
        from checkpointing import capture_rng
        _save_torch(state_path, torch, {
            **metadata_binding, "simstate": _simstate(np, runtime.runtime.env),
            "rng": capture_rng(torch), "settling_steps": 30,
            "initial_state_index": 0,
        })

    delta = _input_delta(torch, noisy_datum, clean_datum)
    if reused_obs:
        metadata = _load_json(meta_path)
    else:
        metadata = {**metadata_binding, "unit": unit,
            "original_success": case["original_success"],
            "noise": noise, "noise_capture_stage": NOISE_CAPTURE_STAGE,
            "camera_scope": "agentview_image only; wrist camera unchanged",
            "settling_steps": 30, "initial_state_index": 0,
            "observations": {
                "noisy": {"file": noisy_path.name, "agentview": _agent_meta(np, obs),
                          "input_identity": ids["noisy"]},
                "clean": {"file": clean_path.name, "agentview": _agent_meta(np, clean_obs),
                          "input_identity": ids["clean"]}},
            "simstate_rng_file": state_path.name, "model_input_delta": delta,
            "observation_sha256": {"noisy": _sha256_file(noisy_path),
                                   "clean": _sha256_file(clean_path)},
            "training_use": False}
        save(meta_path, metadata)

    queries, reused, parity = 0, 0, {}
    initial_noise_by_variant = {}
    for variant, datum in (("noisy", noisy_datum), ("clean", clean_datum)):
        identity = ids[variant]
        for arm in ARMS:
            source_action = source_actions[arm]["action"] if variant == "noisy" else None
            result, was_reused = _query_record(
                runtime, torch, unit, index, arm, variant, case["sampler_seed"],
                description, datum, identity, out, source_action)
            reused += int(was_reused)
            queries += int(not was_reused)
            action_initial_noise = result["action_initial_noise"]
            if variant not in initial_noise_by_variant:
                initial_noise_by_variant[variant] = action_initial_noise
            elif max_abs_difference(initial_noise_by_variant[variant], action_initial_noise) != 0:
                raise RuntimeError(
                    f"Sampler initial noise differs across arms for {index}/{variant}"
                )
            if variant == "noisy":
                parity[arm] = result["max_abs_vs_source_first_chunk"]
                if result["max_abs_vs_source_first_chunk"] != 0:
                    raise RuntimeError(
                        f"{arm} noisy source first-chunk parity failed for {index}: "
                        f"{result['max_abs_vs_source_first_chunk']}"
                    )

    repeat_queries, repeat_record = 0, None
    if unit == "sensor_0" and index == int(runtime.sensor_cases[0]["index"]):
        repeat_variant = "noisy_repeat"
        tensor_path = out / "records" / f"{index}_bf16_{repeat_variant}.pt"
        json_path = out / "records" / f"{index}_bf16_{repeat_variant}.json"
        binding = {"protocol": runtime.p["protocol"], "unit": unit, "index": index,
                   "arm": "bf16", "variant": repeat_variant,
                   "seed": int(case["sampler_seed"]), "description": description,
                   "input_identity": ids["noisy"]}
        repeated = _check_saved_query(torch, tensor_path, json_path, binding)
        if repeated is None:
            runtime.activate("bf16")
            action, future, counts, initial_noise = runtime.query(
                noisy_datum, case["sampler_seed"], description)
            commands = runtime.runtime.command(action.clone()).tolist()
            if not _valid_command_shape(commands):
                raise RuntimeError("Runtime command conversion did not return 32x7 commands")
            repeated = {**binding, "counts": counts, "action": action, "future": future,
                        "action_initial_noise": initial_noise, "commands": commands}
            _save_torch(tensor_path, torch, repeated)
            save(json_path, {**binding, "counts": counts, "query_reused": False,
                             "action_initial_noise_shape": list(initial_noise.shape),
                             "action_initial_noise_dtype": str(initial_noise.dtype),
                             "action_initial_noise_identity": tensor_identity(torch, initial_noise),
                             "commands": commands, "commands_shape": [len(commands), len(commands[0])]})
            repeat_queries = 1
        else:
            validate_binding(repeated, binding)
            repeat_json = _load_json(json_path)
            if "commands" not in repeated or "commands" not in repeat_json:
                commands = runtime.runtime.command(repeated["action"].clone()).tolist()
                repeated["commands"] = commands
                repeat_json["commands"] = commands
                save(json_path, repeat_json)
                _save_torch(tensor_path, torch, repeated)
        base = _check_saved_query(*((torch,) + _query_paths(out, index, "bf16", "noisy")),
                                  {"protocol": runtime.p["protocol"], "unit": unit,
                                   "index": index, "arm": "bf16", "variant": "noisy",
                                   "seed": int(case["sampler_seed"]),
                                   "description": description,
                                   "input_identity": ids["noisy"]})
        repeat_error = max_abs_difference(repeated["action"], base["action"])
        repeat_future_error = max_abs_difference(repeated["future"], base["future"])
        repeat_noise_error = max_abs_difference(repeated["action_initial_noise"],
                                                base["action_initial_noise"])
        if repeat_error != 0 or repeat_future_error != 0 or repeat_noise_error != 0:
            raise RuntimeError(
                "Same-input repeat changed output: "
                f"action={repeat_error}, future={repeat_future_error}, noise={repeat_noise_error}"
            )
        repeat_record = {"index": index, "arm": "bf16", "variant": "noisy",
                         "repeat_queries": repeat_queries,
                         "action_max_abs": repeat_error, "future_max_abs": repeat_future_error,
                         "initial_noise_max_abs": repeat_noise_error}

    return {"index": index, "status": "complete", "queries": queries + repeat_queries,
            "base_queries": queries, "repeat_queries": repeat_queries,
            "query_records_reused": reused, "source_first_chunk_max_abs": parity,
            "model_input_delta": delta, "repeat": repeat_record,
            "observation_files_reused": reused_obs,
            "noise_level": noise["level"]}


def _capture_binding(p, unit, case, arm):
    return {"protocol": p["protocol"], "unit": unit, "index": int(case["index"]),
            "arm": arm, "mode": "native_rollout", "row": case["row"],
            "environment_seed": int(case["environment_seed"]),
            "sampler_seed": int(case["sampler_seed"]),
            "capture_before_steps": list(CAPTURE_STEPS),
            "manifest_index": int(case["index"])}


def _capture_state(runtime, tap, np, torch, unit, case, arm, out, obs, datum,
                   steps, replans, sampler_seed, capture_dir, rng_state):
    index = int(case["index"])
    clean_obs = tap.clean_observation(np, obs)
    identity = datum_identity(torch, datum)
    stem = f"{index}_step{steps:03d}"
    noisy_path = capture_dir / f"{stem}_noisy.npz"
    clean_path = capture_dir / f"{stem}_clean.npz"
    sim_rng_path = capture_dir / f"{stem}_state.pt"
    metadata_path = capture_dir / f"{stem}.json"
    metadata = {
        "protocol": runtime.p["protocol"], "unit": unit, "index": index,
        "arm": arm, "row": case["row"], "description": runtime.runtime.description,
        "environment_seed": case["environment_seed"],
        "sampler_seed_base": case["sampler_seed"], "sampler_seed": sampler_seed,
        "step": steps, "replan": replans, "input_identity": identity,
        "noise": _noise_identity(runtime.runtime.env, tap, _frozen_noise_level(case)),
        "noise_capture_stage": NOISE_CAPTURE_STAGE,
        "camera_scope": "agentview_image only; wrist camera unchanged",
        "observations": {
            "noisy": {"file": noisy_path.name, "agentview": _agent_meta(np, obs)},
            "clean": {"file": clean_path.name, "agentview": _agent_meta(np, clean_obs)}},
        "simstate_rng_file": sim_rng_path.name, "training_use": False,
    }
    if metadata_path.exists():
        old = _load_json(metadata_path)
        for key in ("protocol", "unit", "index", "arm", "row", "description",
                    "environment_seed", "sampler_seed_base", "sampler_seed",
                    "step", "replan", "input_identity"):
            if old.get(key) != metadata.get(key):
                raise ValueError(f"Saved capture binding differs for {stem}: {key}")
        if not noisy_path.is_file() or not clean_path.is_file() or not sim_rng_path.is_file():
            raise RuntimeError(f"Saved capture files are incomplete for {stem}")
        return metadata

    _save_npz(np, noisy_path, obs)
    _save_npz(np, clean_path, clean_obs)
    _save_torch(sim_rng_path, torch, {
        **metadata, "simstate": _simstate(np, runtime.runtime.env),
        "rng": rng_state, "settling_steps": 30,
    })
    save(metadata_path, metadata)
    return metadata


def _existing_episode(torch, tensor_path, json_path, binding):
    if tensor_path.exists() != json_path.exists():
        raise RuntimeError(f"Partial completed episode output: {tensor_path}")
    if not tensor_path.exists():
        return None
    metadata = _load_json(json_path)
    for key, value in binding.items():
        if metadata.get(key) != value:
            raise ValueError(f"Saved episode binding differs: {key}")
    payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
    for key, value in binding.items():
        if payload.get(key) != value:
            raise ValueError(f"Saved episode tensor binding differs: {key}")
    return payload, metadata


def _capture_episode(runtime, tap, np, torch, unit, case, arm, source, out,
                     new_query_counter):
    from checkpointing import capture_rng

    index = int(case["index"])
    _assert_frozen_seeds(case)
    if bool(source["success"]) != bool(case["original_success"][arm]):
        raise RuntimeError(f"Source success outcome differs from frozen plan at index {index}")
    if not _valid_command_shape(source["action"]):
        raise RuntimeError(f"Frozen source action must be [32,7] at manifest index {index}")
    binding = _capture_binding(runtime.p, unit, case, arm)
    tensor_path = out / "records" / f"{index}.pt"
    json_path = out / "records" / f"{index}.json"
    previous = _existing_episode(torch, tensor_path, json_path, binding)
    if previous is not None and previous[0].get("episode_complete") is True:
        return {"index": index, "status": previous[0]["status"],
                "episode_complete": True,
                "success": previous[0]["success"], "steps": previous[0]["steps"],
                "source_replay_passed": previous[0]["source_replay_passed"],
                "global_rng_query_effects": previous[0]["global_rng_query_effects"],
                "new_queries": 0, "reused": True,
                "captured_steps": previous[0]["captured_steps"]}

    capture_dir = out / "records" / str(index)
    capture_dir.mkdir(parents=True, exist_ok=True)
    observations, replans = {}, []
    tap.clear()
    obs = runtime.runtime.task(case["row"], index)
    description = runtime.runtime.description
    cap = 700 if case["row"]["suite"] == "libero_10" else 400
    steps, replan, success, first_chunk_error = 0, 0, False, None
    query_start = new_query_counter[0]
    runtime.activate(arm)
    while steps < cap and not success:
        datum = runtime.datum(obs)
        sampler_seed = int(case["sampler_seed"]) + replan
        if steps in CAPTURE_STEPS and steps not in observations:
            metadata = _capture_state(runtime, tap, np, torch, unit, case, arm,
                                     out, obs, datum, steps, replan, sampler_seed,
                                     capture_dir, capture_rng(torch))
            observations[str(steps)] = metadata

        rng_before = capture_rng(torch)
        new_query_counter[0] += 1
        action, _ = runtime.runtime.infer(datum, sampler_seed, measure=False)
        rng_changes = _rng_changes(torch, rng_before, capture_rng(torch))
        action_cpu = action.detach().float().cpu()
        if replan == 0:
            first_chunk_error = max_abs_difference(action_cpu, source["action"])
            if first_chunk_error != 0:
                replans.append({"replan": replan, "step": steps,
                    "sampler_seed": sampler_seed, "action": action_cpu,
                    "commands": None, "executed_commands": [],
                    "source_first_chunk_max_abs": first_chunk_error,
                    "global_rng_changes": rng_changes})
                return _finish_episode(torch, tensor_path, json_path, binding,
                    index, "first_chunk_parity_failed", False, steps, False,
                    first_chunk_error, replans, observations,
                    new_query_counter[0] - query_start, cap, case, arm,
                    source_success=bool(source["success"]),
                    source_steps=int(source["steps"]), source_replay_passed=False,
                    captured_steps=sorted(int(value) for value in observations))

        commands = runtime.runtime.command(action)
        commands_list = _tolist(commands)
        executed, done = [], False
        for command in commands[:min(10, cap - steps)]:
            tap.clear()
            obs, _, done, _ = runtime.runtime.env.step(command.tolist())
            steps += 1
            executed.append(command.tolist())
            if done:
                success = True
                break
        replans.append({"replan": replan, "step": steps - len(executed),
            "sampler_seed": sampler_seed, "action": action_cpu,
            "commands": commands_list,
            "executed_commands": executed, "source_first_chunk_max_abs":
                first_chunk_error if replan == 0 else None,
            "global_rng_changes": rng_changes})
        replan += 1
        if done:
            break

    captured_steps = sorted(int(value) for value in observations)
    source_success = bool(source["success"])
    source_steps = int(source["steps"])
    source_replay_passed = (first_chunk_error == 0 and success == source_success
                            and steps == source_steps)
    status = "complete" if source_replay_passed else "complete_source_replay_mismatch"
    return _finish_episode(torch, tensor_path, json_path, binding, index, status,
        True, steps, success, first_chunk_error, replans, observations,
        new_query_counter[0] - query_start, cap, case, arm,
        source_success=source_success, source_steps=source_steps,
        source_replay_passed=source_replay_passed, captured_steps=captured_steps)


def _finish_episode(torch, tensor_path, json_path, binding, index, status,
                    episode_complete, steps, success, first_chunk_error,
                    replans, observations, new_queries, cap, case, arm,
                    source_success=None, source_steps=None,
                    source_replay_passed=False, captured_steps=None):
    captured_steps = captured_steps if captured_steps is not None else sorted(
        int(value) for value in observations)
    expected_success = case["original_success"].get(arm)
    rng_effect_counts = {
        key: sum(bool(item.get("global_rng_changes", {}).get(key, False)) for item in replans)
        for key in ("python_changed", "numpy_changed", "torch_cpu_changed", "torch_cuda_changed")
    }
    payload = {**binding, "index": index, "arm": arm, "status": status,
        "episode_complete": bool(episode_complete), "success": bool(success),
        "steps": int(steps), "cap": int(cap), "replans": replans,
        "observations": observations, "captured_steps": captured_steps,
        "missing_capture_steps": [step for step in CAPTURE_STEPS if step not in captured_steps],
        "source_first_chunk_max_abs": first_chunk_error,
        "source_success": source_success, "source_steps": source_steps,
        "source_replay_passed": bool(source_replay_passed),
        "source_original_success_by_arm": case["original_success"],
        "collector_success_matches_cpu_source": (None if source_success is None
                                                  else bool(success) == bool(source_success)),
        "collector_steps_match_cpu_source": (None if source_steps is None
                                               else int(steps) == int(source_steps)),
        "source_expected_success_from_plan": expected_success,
        "global_rng_query_effects": rng_effect_counts,
        "new_queries": int(new_queries), "training_updates": 0,
        "training_use": False}
    _save_torch(tensor_path, torch, payload)
    save(json_path, {key: value for key, value in payload.items()
                     if key != "replans"})
    return {"index": index, "status": status,
            "episode_complete": bool(episode_complete),
            "success": bool(success), "steps": int(steps),
            "source_replay_passed": bool(source_replay_passed),
            "global_rng_query_effects": rng_effect_counts,
            "new_queries": int(new_queries), "reused": False,
            "captured_steps": captured_steps}


def _sensor_run(p, unit, plan, root, runtime=None):
    import numpy as np
    import torch

    shard = int(unit[-1])
    cases = sorted((case for case in plan["cases"] if int(case["shard"]) == shard),
                   key=lambda value: int(value["ordinal"]))
    if len(cases) != 20:
        raise RuntimeError(f"{unit} must own exactly 20 sensor cases, got {len(cases)}")
    out = root / "results" / unit
    out.mkdir(parents=True, exist_ok=True)
    owns_runtime = runtime is None
    if owns_runtime:
        runtime = SwitchingRuntime(p, out)
    runtime.sensor_cases = cases
    tap = NoiseTap().install(_noise_module())
    deadline = Deadline(seconds=int(p["resource"]["gpu_soft_deadline_seconds"]))
    completed, source_parities, repeats = [], [], []
    new_queries = 0
    progress = {"protocol": p["protocol"], "unit": unit,
        "pbs_jobid": os.environ["PBS_JOBID"], "complete": False,
        "expected": 20, "completed": 0, "new_queries": 0,
        "training_updates": 0,
        "input_identity": {"sensor_plan_sha256": _sha256_file(root / "data/sensor_plan.json"),
                           "source_closed_job": p["source_closed_job"]},
        "training_use": False}
    save(out / "progress.json", progress)
    try:
        for case in cases:
            if deadline.near(reserve=180):
                break
            source_actions = read(root / "data/source_actions" / f"{case['index']}.json")
            query_start = runtime.query_calls
            try:
                result = _initial_case(runtime, tap, np, torch, unit, case, root, out,
                                       source_actions)
            except Exception as exc:
                new_queries += runtime.query_calls - query_start
                progress.update(new_queries=new_queries, failed_index=case["index"],
                                failure=f"{type(exc).__name__}: {exc}")
                save(out / "progress.json", progress)
                save(out / "summary.json", {**progress, "complete": False,
                     "completed": len(completed), "records": completed,
                     "training_updates": 0, "diagnostic_only": True,
                     "training_use": False})
                raise
            completed.append(result)
            new_queries += result["queries"]
            source_parities.append(result["source_first_chunk_max_abs"])
            if result.get("repeat"):
                repeats.append(result["repeat"])
            progress.update(completed=len(completed), new_queries=new_queries,
                            source_first_chunk_parity=source_parities,
                            same_input_repeats=repeats,
                            strata={"original_bf16_success": sum(
                                bool(row["original_success"]["bf16"]) for row in cases),
                                "original_bf16_failure": sum(
                                not bool(row["original_success"]["bf16"]) for row in cases),
                                "per_channel_success": sum(
                                bool(row["original_success"]["per_channel_clip"]) for row in cases),
                                "relational_success": sum(
                                bool(row["original_success"]["relational"]) for row in cases)})
            save(out / "progress.json", progress)
        complete = len(completed) == len(cases)
        summary = {**progress, "complete": complete,
            "completed": len(completed), "records": completed,
            "new_queries": new_queries, "expected_base_queries": 160,
            "expected_repeat_queries": 1 if unit == "sensor_0" else 0,
            "training_updates": 0,
            "diagnostic_only": True, "training_use": False,
            "sensor_level_counts": dict(Counter(str(row["noise_level"]) for row in completed)),
            "boundary": "same persisted noisy/clean state and sampler seed; no extra environment steps"}
        save(out / "summary.json", summary)
        save(out / "progress.json", summary)
        print("SENSOR_WORKER_SUMMARY " + json.dumps(summary), flush=True)
        return summary
    finally:
        tap.uninstall()
        if owns_runtime:
            runtime.close()


def _sensor_pair_run(p, plan, root):
    from diagnostic_runtime import SwitchingRuntime

    runtime = SwitchingRuntime(p, root/'results/sensor_0')
    summaries = {}
    try:
        for unit in ('sensor_0', 'sensor_1'):
            if summaries and Deadline(p['resource']['gpu_soft_deadline_seconds']).near(180):
                break
            print('SENSOR_BATCH_PHASE '+unit, flush=True)
            summaries[unit] = _sensor_run(p, unit, plan, root, runtime=runtime)
            if summaries[unit]['complete'] is not True:
                break
    finally:
        runtime.close()
    result = {'protocol': p['protocol'], 'pbs_jobid': os.environ['PBS_JOBID'],
              'complete': len(summaries) == 2 and all(s['complete'] for s in summaries.values()),
              'units': {u: {'complete': s['complete'], 'completed': s['completed'], 'new_queries': s.get('new_queries', 0)}
                        for u, s in summaries.items()},
              'expected_endpoint_records': 320, 'extra_repeat_forwards': 1,
              'expected_model_forwards_without_resume': 321,
              'new_queries_this_segment': sum(s.get('new_queries', 0) for s in summaries.values()),
              'scope': 'two original20-case shards sequentially in one allocation; first shard must complete before second; original global deadline unchanged'}
    save(root/'results/sensor_batch/summary.json', result)
    return result


def _capture_run(p, unit, plan, root):
    import numpy as np
    import torch

    match = re.fullmatch(r"capture_(mse|rel)_([01])", unit)
    collector = "per_channel_clip" if match.group(1) == "mse" else "relational"
    shard = int(match.group(2))
    cases = sorted((case for case in plan["discordant_cases"]
                    if int(case["capture_shard"]) == shard),
                   key=lambda value: int(value["ordinal"]))
    expected = sum(int(case["capture_shard"]) == shard for case in plan["discordant_cases"])
    if len(cases) != expected or expected == 0:
        raise RuntimeError(f"{unit} has no frozen discordant cases")
    if any(not bool(case["original_success"]["per_channel_clip"])
           or bool(case["original_success"]["relational"]) for case in cases):
        raise RuntimeError("Capture cases must be the frozen MSE-success/relational-failure stratum")
    out = root / "results" / unit
    out.mkdir(parents=True, exist_ok=True)
    runtime = SwitchingRuntime(p, out)
    tap = NoiseTap().install(_noise_module())
    deadline = Deadline(seconds=int(p["resource"]["gpu_soft_deadline_seconds"]))
    new_queries = [0]
    records, parity = [], []
    progress = {"protocol": p["protocol"], "unit": unit,
        "pbs_jobid": os.environ["PBS_JOBID"], "complete": False,
        "expected": expected, "completed": 0, "new_queries": 0,
        "training_updates": 0,
        "input_identity": {"sensor_plan_sha256": _sha256_file(root / "data/sensor_plan.json"),
                           "source_closed_job": p["source_closed_job"],
                           "collector_arm": collector}, "training_use": False}
    save(out / "progress.json", progress)
    try:
        for case in cases:
            source = read(root / "data/source_actions" / f"{case['index']}.json")[collector]
            # Start only when the allocation can reasonably finish the original episode.
            reserve = 180 + 8*((int(source['steps'])+9)//10) + 2*int(source['steps'])
            if deadline.near(reserve=reserve):
                break
            try:
                result = _capture_episode(runtime, tap, np, torch, unit, case, collector,
                                          source, out, new_queries)
            except Exception as exc:
                progress.update(new_queries=new_queries[0], failed_index=case['index'],
                                failure=f'{type(exc).__name__}: {exc}')
                save(out / 'progress.json', progress)
                save(out / 'summary.json', {**progress, 'records': records,
                                           'source_replay_passed_all': False})
                raise
            records.append(result)
            parity.append({"index": case["index"], "source_replay_passed": result["source_replay_passed"],
                           "success": result["success"], "steps": result["steps"]})
            progress.update(completed=sum(item["episode_complete"] for item in records),
                            attempted=len(records), new_queries=new_queries[0],
                            source_replay=parity)
            save(out / "progress.json", progress)
        complete = len(records) == expected and all(
            item["episode_complete"] and item["source_replay_passed"] for item in records)
        summary = {**progress, "complete": complete,
            "completed": sum(item["episode_complete"] for item in records),
            "attempted": len(records), "new_queries": new_queries[0],
            "records": records, "training_updates": 0,
            "training_use": False,
            "boundary": "only native collector infer advances each episode; no extra checkpoint query or RNG restore",
            "source_replay_passed_all": bool(records) and all(
                item["source_replay_passed"] for item in records)}
        save(out / "summary.json", summary)
        save(out / "progress.json", summary)
        print("CAPTURE_WORKER_SUMMARY " + json.dumps(summary), flush=True)
        return summary
    finally:
        tap.uninstall()
        runtime.close()


def run(unit):
    guard()
    if unit not in ("sensor_0", "sensor_1", "capture_mse_0", "capture_mse_1",
                    "capture_rel_0", "capture_rel_1"):
        raise ValueError(f"Unknown sensor diagnostic unit: {unit}")
    p = read(Path(__file__).with_name("protocol.json"))
    root = Path(p["remote_root"])
    plan = read(root / "data/sensor_plan.json")
    if (plan.get("protocol") != p["protocol"] or plan.get("complete") is not True
            or len(plan.get("sensor_indices", [])) != 40
            or len(plan.get("discordant_cases", [])) != 5
            or plan.get("training_use") is not False):
        raise RuntimeError("Frozen complete 40-case sensor plan and five discordances required")
    for case in (*plan["cases"], *plan["discordant_cases"]):
        _assert_frozen_seeds(case)
        _frozen_noise_level(case)
    if unit == 'sensor_0':
        return _sensor_pair_run(p, plan, root)
    return (_sensor_run(p, unit, plan, root) if unit.startswith("sensor_")
            else _capture_run(p, unit, plan, root))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", required=True,
                        choices=("sensor_0", "sensor_1", "capture_mse_0", "capture_mse_1",
                                 "capture_rel_0", "capture_rel_1"))
    args = parser.parse_args()
    run(args.unit)


if __name__ == "__main__":
    main()
