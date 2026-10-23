"""Offline same-state four-arm queries and fixed-BF16 future-reader checks."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

from control_common import Deadline, guard, read, save


ARMS = ("bf16", "per_channel_clip", "softbin", "relational")
VARIANTS = ("clean", "noisy")
COLLECTOR_UNITS = {
    "per_channel_clip": "capture_mse",
    "relational": "capture_rel",
}
CAPTURE_STEPS = (0, 10, 50, 100, 200)
ACTION_SHAPE = (32, 7)
FUTURE_SHAPE = (1, 48, 3, 14, 28)
NOISE_SHAPE = (1, 32, 7)
MOTOR_DIMS = 6
FIRST_CHUNK = 10
SOFT_DEADLINE_SECONDS = 6600
RESERVE_SECONDS = 180


class DeadlineReached(RuntimeError):
    pass


def _expected_capture_steps(source_steps):
    """Captures occur at loop entry, so a terminal step itself is not captured."""
    return [step for step in CAPTURE_STEPS if step < int(source_steps)]


def _relative(root, path):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def _episode_binding(protocol, capture_unit, case, collector):
    return {
        "protocol": protocol,
        "unit": capture_unit,
        "index": int(case["index"]),
        "arm": collector,
        "mode": "native_rollout",
        "row": case["row"],
        "environment_seed": int(case["environment_seed"]),
        "sampler_seed": int(case["sampler_seed"]),
        "capture_before_steps": list(CAPTURE_STEPS),
        "manifest_index": int(case["index"]),
    }


def _validate_capture_sources(protocol, root, plan, shard):
    if plan.get("protocol") != protocol or plan.get("complete") is not True:
        raise ValueError("sensor_plan.json is incomplete or belongs to another protocol")
    all_cases = plan.get("discordant_cases")
    if not isinstance(all_cases, list) or len(all_cases) != 5:
        raise ValueError("Expected the five frozen discordant capture cases")
    cases = [case for case in all_cases if int(case["capture_shard"]) == shard]
    if not cases or any(int(case["capture_shard"]) not in (0, 1) for case in cases):
        raise ValueError(f"No discordant cases are assigned to trajectory shard {shard}")
    if any(case.get("diagnostic_discordant") is not True for case in cases):
        raise ValueError("Capture case lost its frozen discordant label")
    states = []
    capture_summaries = {}
    for collector, capture_prefix in COLLECTOR_UNITS.items():
        capture_unit = f"{capture_prefix}_{shard}"
        unit_cases = sorted(cases, key=lambda case: int(case["ordinal"]))
        summary_path = root / "results" / capture_unit / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing completed capture summary: {summary_path}")
        summary = read(summary_path)
        expected_indices = [int(case["index"]) for case in unit_cases]
        records = summary.get("records")
        record_map = {int(item["index"]): item for item in records or []}
        if (summary.get("protocol") != protocol or summary.get("unit") != capture_unit
                or summary.get("complete") is not True
                or summary.get("source_replay_passed_all") is not True
                or summary.get("expected") != len(unit_cases)
                or summary.get("completed") != len(unit_cases)
                or summary.get("attempted") != len(unit_cases)
                or sorted(record_map) != sorted(expected_indices)
                or any(item.get("source_replay_passed") is not True
                       for item in record_map.values())):
            raise RuntimeError(
                f"{capture_unit} must be complete with source_replay_passed_all=true "
                "and exactly the frozen shard cases"
            )
        capture_summaries[collector] = {
            "unit": capture_unit,
            "source_summary": _relative(root, summary_path),
            "expected": len(unit_cases),
            "source_replay_passed_all": True,
        }

        for case in unit_cases:
            index = int(case["index"])
            summary_record = record_map[index]
            episode_dir = root / "results" / capture_unit / "records"
            episode_json = episode_dir / f"{index}.json"
            episode_pt = episode_dir / f"{index}.pt"
            state_dir = episode_dir / str(index)
            if not episode_json.is_file() or not episode_pt.is_file():
                raise FileNotFoundError(f"Capture episode files are missing for {capture_unit}/{index}")
            episode = read(episode_json)
            for key, value in _episode_binding(protocol, capture_unit, case, collector).items():
                if episode.get(key) != value:
                    raise ValueError(f"Capture episode binding differs for {capture_unit}/{index}: {key}")
            captured_steps = [int(step) for step in episode.get("captured_steps", [])]
            if (episode.get("episode_complete") is not True
                    or episode.get("source_replay_passed") is not True
                    or episode.get("source_success") != episode.get("success")
                    or episode.get("source_steps") != episode.get("steps")
                    or summary_record.get("captured_steps") != captured_steps
                    or captured_steps != _expected_capture_steps(episode.get("steps", -1))
                    or episode.get("observations") is None):
                raise RuntimeError(f"Capture episode trace is incomplete or inconsistent: {capture_unit}/{index}")

            for step in captured_steps:
                metadata_path = state_dir / f"{index}_step{step:03d}.json"
                if not metadata_path.is_file():
                    raise FileNotFoundError(f"Capture-state metadata is missing: {metadata_path}")
                metadata = read(metadata_path)
                expected_metadata = {
                    "protocol": protocol,
                    "unit": capture_unit,
                    "index": index,
                    "arm": collector,
                    "row": case["row"],
                    "environment_seed": int(case["environment_seed"]),
                    "sampler_seed_base": int(case["sampler_seed"]),
                    "step": step,
                    "description": episode.get("observations", {}).get(str(step), {}).get("description"),
                }
                if expected_metadata["description"] != metadata.get("description"):
                    raise ValueError(f"Capture description binding differs for {capture_unit}/{index}/step{step}")
                for key, value in expected_metadata.items():
                    if metadata.get(key) != value:
                        raise ValueError(f"Capture-state binding differs for {capture_unit}/{index}/step{step}: {key}")
                replan = int(metadata.get("replan", -1))
                seed = int(metadata.get("sampler_seed", -1))
                if seed != int(case["sampler_seed"]) + replan:
                    raise ValueError(f"Capture sampler seed differs for {capture_unit}/{index}/step{step}")
                if episode.get("observations", {}).get(str(step)) != metadata:
                    raise ValueError(f"Capture episode and state metadata differ for {capture_unit}/{index}/step{step}")
                observations = metadata.get("observations", {})
                if set(observations) != set(VARIANTS):
                    raise ValueError(f"Capture must provide clean and noisy NPZs: {capture_unit}/{index}/step{step}")
                observation_paths = {}
                for variant in VARIANTS:
                    filename = observations[variant].get("file")
                    if not isinstance(filename, str) or Path(filename).name != filename:
                        raise ValueError(f"Unsafe or missing capture NPZ filename for {capture_unit}/{index}/step{step}")
                    observation_path = state_dir / filename
                    if not observation_path.is_file():
                        raise FileNotFoundError(f"Capture observation NPZ is missing: {observation_path}")
                    observation_paths[variant] = observation_path
                identity = metadata.get("input_identity")
                if (not isinstance(identity, dict) or set(identity) != {"image", "proprio"}
                        or any(not isinstance(identity.get(key), dict)
                               or not identity[key].get("sha256") for key in ("image", "proprio"))):
                    raise ValueError(f"Capture lacks noisy input identity for {capture_unit}/{index}/step{step}")
                states.append({
                    "collector": collector,
                    "capture_unit": capture_unit,
                    "case": case,
                    "index": index,
                    "step": step,
                    "replan": replan,
                    "seed": seed,
                    "description": metadata["description"],
                    "row": case["row"],
                    "capture_input_identity": identity,
                    "capture_episode_json": _relative(root, episode_json),
                    "capture_state_json": _relative(root, metadata_path),
                    "observation_paths": {
                        variant: _relative(root, observation_paths[variant])
                        for variant in VARIANTS
                    },
                })
    keys = [(state["collector"], state["index"], state["step"]) for state in states]
    if len(keys) != len(set(keys)):
        raise ValueError("Capture source produced duplicate (collector,index,step) states")
    return sorted(states, key=lambda state: (
        state["collector"], int(state["case"]["ordinal"]), state["step"]
    )), capture_summaries


def _state_stem(state):
    return f"{state['collector']}_{state['index']}_step{state['step']:03d}"


def _record_paths(records_dir, stem):
    return records_dir / f"{stem}.pt", records_dir / f"{stem}.json"


def _binding(protocol, unit, state, variant, role, arm):
    return {
        "protocol": protocol,
        "unit": unit,
        "collector": state["collector"],
        "capture_source_unit": state["capture_unit"],
        "index": state["index"],
        "step": state["step"],
        "replan": state["replan"],
        "variant": variant,
        "seed": state["seed"],
        "description": state["description"],
        "row": state["row"],
        "input_identity": state["input_identities"][variant],
        "capture_noisy_input_identity": state["capture_input_identity"],
        "capture_episode_json": state["capture_episode_json"],
        "capture_state_json": state["capture_state_json"],
        "capture_observation_npz": state["observation_paths"][variant],
        **role,
        "arm": arm if role.get("record_type") == "native" else None,
    }


def _validate_record_binding(saved, expected, label):
    mismatch = [key for key, value in expected.items() if saved.get(key) != value]
    if mismatch:
        raise ValueError(f"Saved {label} binding differs: {', '.join(mismatch)}")
    if saved.get("status") != "complete":
        raise ValueError(f"Saved {label} record is not complete")


def _save_record(torch, records_dir, stem, payload, metadata):
    records_dir.mkdir(parents=True, exist_ok=True)
    tensor_path, json_path = _record_paths(records_dir, stem)
    temp_tensor = tensor_path.with_name(tensor_path.name + f".{os.getpid()}.tmp")
    temp_json = json_path.with_name(json_path.name + f".{os.getpid()}.tmp")
    torch.save(payload, temp_tensor)
    temp_json.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp_tensor.replace(tensor_path)
    temp_json.replace(json_path)
    return tensor_path, json_path


def _read_record(torch, records_dir, stem, expected, label):
    tensor_path, json_path = _record_paths(records_dir, stem)
    if not tensor_path.exists() and not json_path.exists():
        return None
    if tensor_path.exists() != json_path.exists():
        partial = tensor_path if tensor_path.exists() else json_path
        if tensor_path.exists():
            payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
            _validate_record_binding(payload, expected, label)
        else:
            _validate_record_binding(read(json_path), expected, label)
        partial.unlink()
        return None
    payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
    metadata = read(json_path)
    _validate_record_binding(payload, expected, label)
    _validate_record_binding(metadata, expected, label)
    if payload.get("record_type") != metadata.get("record_type"):
        raise ValueError(f"Saved {label} PT/JSON record types differ")
    if payload.get("counts") != metadata.get("counts"):
        raise ValueError(f"Saved {label} PT/JSON counts differ")
    if payload.get("metrics") != metadata.get("metrics"):
        raise ValueError(f"Saved {label} PT/JSON metrics differ")
    return payload, metadata


def _load_state_inputs(torch, np, switching, root, state):
    from diagnostic_runtime import datum_identity

    datums, identities = {}, {}
    switching.runtime.description = state["description"]
    for variant in VARIANTS:
        path = root / Path(state["observation_paths"][variant])
        with np.load(path, allow_pickle=False) as archive:
            observation = {name: archive[name].copy() for name in archive.files}
        if "agentview_image" not in observation:
            raise ValueError(f"Captured observation lacks agentview_image: {path}")
        datum = switching.datum(observation)
        identity = datum_identity(torch, datum)
        if variant == "noisy" and identity != state["capture_input_identity"]:
            raise ValueError(
                f"Noisy preprocessed datum identity differs from capture metadata: {path}"
            )
        expected_agent = _capture_agent_meta(state, variant)
        image = observation["agentview_image"]
        if list(image.shape) != expected_agent["shape"] or str(image.dtype) != expected_agent["dtype"]:
            raise ValueError(f"Captured image shape/dtype differs from state metadata: {path}")
        datums[variant] = datum
        identities[variant] = identity
    if not torch.equal(datums["clean"]["proprio"], datums["noisy"]["proprio"]):
        raise ValueError("Clean/noisy captured observations do not share the same proprio state")
    state["input_identities"] = identities
    return datums


def _capture_agent_meta(state, variant):
    source = state.get("capture_observation_meta")
    if not source:
        raise RuntimeError("Capture observation metadata was not loaded")
    return source[variant]["agentview"]


def _load_capture_observation_meta(root, states):
    cache = {}
    for state in states:
        key = (state["collector"], state["index"], state["step"])
        if key not in cache:
            path = root / Path(state["capture_state_json"])
            cache[key] = read(path)["observations"]
        state["capture_observation_meta"] = cache[key]


def _as_cpu_float(torch, value, shape, label):
    tensor = torch.as_tensor(value).detach().float().cpu()
    if tuple(tensor.shape) != tuple(shape) or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{label} must be finite with shape {tuple(shape)}")
    return tensor


def _noise_identity(torch, noise):
    from diagnostic_runtime import tensor_identity

    return tensor_identity(torch, noise)


def _native_binding(protocol, unit, state, variant, arm):
    return _binding(protocol, unit, state, variant,
                    {"record_type": "native"}, arm)


def _reader_binding(protocol, unit, state, variant, donor):
    return _binding(protocol, unit, state, variant,
                    {"record_type": "fixed_bf16_reader",
                     "reader_arm": "bf16", "future_donor_arm": donor}, "bf16")


def _native_record_name(state, variant, arm):
    return f"{_state_stem(state)}_{variant}_native_{arm}"


def _reader_record_name(state, variant, donor):
    return f"{_state_stem(state)}_{variant}_reader_bf16_from_{donor}"


def _ensure_native(torch, switching, records_dir, protocol, unit, state,
                   datum, variant, arm, deadline, counts):
    from reader_worker import _command as checked_command

    binding = _native_binding(protocol, unit, state, variant, arm)
    stem = _native_record_name(state, variant, arm)
    saved = _read_record(torch, records_dir, stem, binding, "native endpoint")
    if saved is not None:
        payload, metadata = saved
        action = _as_cpu_float(torch, payload.get("action"), ACTION_SHAPE, "cached native action")
        future = _as_cpu_float(torch, payload.get("future"), FUTURE_SHAPE, "cached native future")
        noise = _as_cpu_float(torch, payload.get("action_initial_noise"), NOISE_SHAPE,
                              "cached native action noise")
        command = _as_cpu_float(torch, payload.get("commands"), ACTION_SHAPE,
                                "cached native execution commands")
        if metadata.get("action_initial_noise_identity") != _noise_identity(torch, noise):
            raise ValueError(f"Cached native noise identity differs for {stem}")
        if (metadata.get("covered_linears") != 614
                or metadata.get("query_coverage_verified") is not True
                or payload.get("covered_linears") != 614
                or payload.get("query_coverage_verified") is not True):
            raise ValueError(f"Cached native endpoint lacks verified 614-Linear coverage: {stem}")
        return {"action": action, "future": future, "noise": noise,
                "commands": command, "counts": payload["counts"], "metadata": metadata}

    if deadline.near(RESERVE_SECONDS):
        raise DeadlineReached("soft deadline reserve reached before native endpoint")
    switching.activate(arm)
    counts["full_model_queries_attempted_new"] += 1
    started = time.monotonic()
    action, future, native_counts, initial_noise = switching.query(
        datum, int(state["seed"]), description=state["description"]
    )
    counts["full_model_queries_new"] += 1
    counts["native_action_steps_new"] += int(native_counts.get("action_steps", 0))
    if native_counts != {"video_steps": 10, "action_steps": 10}:
        raise RuntimeError(f"Native 10+10 count mismatch for {stem}: {native_counts}")
    action = _as_cpu_float(torch, action, ACTION_SHAPE, "native action")
    future = _as_cpu_float(torch, future, FUTURE_SHAPE, "native future")
    noise = _as_cpu_float(torch, initial_noise, NOISE_SHAPE, "native action initial noise")
    # SwitchingRuntime.query returns its action as CPU float32; mirror the sensor
    # worker's command tap on that exact returned tensor.
    commands = _as_cpu_float(torch, checked_command(switching.runtime, action.clone()),
                             ACTION_SHAPE, "native execution commands")
    from diagnostic_runtime import ARMS as runtime_arms
    if tuple(runtime_arms) != tuple(ARMS) or len(switching.runtime.q.modules) != 614:
        raise RuntimeError("SwitchingRuntime query target coverage differs from 614 frozen Linears")
    noise_id = _noise_identity(torch, noise)
    base = {
        **binding,
        "record_type": "native",
        "status": "complete",
        "counts": dict(native_counts),
        "covered_linears": 614,
        "query_coverage_verified": True,
        "action_initial_noise_identity": noise_id,
        "action_shape": list(action.shape),
        "future_shape": list(future.shape),
        "action_initial_noise_shape": list(noise.shape),
        "commands_shape": list(commands.shape),
        "query_seconds": round(time.monotonic() - started, 6),
        "source_reference": "offline fixed-state native output; no environment stepping",
    }
    payload = {
        **base,
        "action": action,
        "future": future,
        "action_initial_noise": noise,
        "commands": commands,
    }
    metadata = dict(base)
    _save_record(torch, records_dir, stem, payload, metadata)
    return {"action": action, "future": future, "noise": noise,
            "commands": commands, "counts": dict(native_counts), "metadata": metadata}


def _ensure_reader(torch, switching, bf16_reader, stats, records_dir,
                   protocol, unit, state, datum, variant, donor,
                   natives, deadline, counts):
    from reader_worker import _metrics, _run_intervention

    binding = _reader_binding(protocol, unit, state, variant, donor)
    stem = _reader_record_name(state, variant, donor)
    saved = _read_record(torch, records_dir, stem, binding, "fixed BF16 reader")
    if saved is not None:
        payload, metadata = saved
        action = _as_cpu_float(torch, payload.get("action"), ACTION_SHAPE,
                               "cached reader action")
        command = _as_cpu_float(torch, payload.get("commands"), ACTION_SHAPE,
                                "cached reader execution commands")
        noise = _as_cpu_float(torch, payload.get("action_initial_noise"), NOISE_SHAPE,
                              "cached reader action noise")
        noise_error = float((noise - natives[(variant, "bf16")]["noise"]).abs().max())
        if (metadata.get("action_initial_noise_max_abs_vs_native") != 0.0
                or noise_error != 0.0
                or metadata.get("action_initial_noise_identity") != _noise_identity(torch, noise)):
            raise ValueError(f"Cached reader noise parity is not exact for {stem}")
        if donor == "bf16" and float((action - natives[(variant, "bf16")]["action"]).abs().max()) != 0.0:
            raise ValueError(f"Cached BF16 self-reader action parity is not exact for {stem}")
        return {"action": action, "commands": command, "noise": noise,
                "counts": payload["counts"], "metrics": payload["metrics"],
                "metadata": metadata}

    if deadline.near(RESERVE_SECONDS):
        raise DeadlineReached("soft deadline reserve reached before BF16 reader pass")
    donor_record = natives[(variant, donor)]
    teacher = natives[(variant, "bf16")]
    started = time.monotonic()
    counts["reader_passes_attempted_new"] += 1
    action, reader_counts, action_noise = _run_intervention(
        torch, switching, bf16_reader, "bf16", datum, int(state["seed"]),
        donor_record["future"],
    )
    counts["reader_passes_new"] += 1
    counts["reader_action_dit_calls_new"] += int(reader_counts["action_dit_calls"])
    if action.requires_grad:
        raise RuntimeError("Diagnostic fixed reader unexpectedly retained an autograd graph")
    from reader_worker import _command as checked_command
    commands = _as_cpu_float(torch, checked_command(switching.runtime, action.clone()),
                             ACTION_SHAPE, "reader execution commands")
    action = _as_cpu_float(torch, action, ACTION_SHAPE, "reader action")
    action_noise = _as_cpu_float(torch, action_noise, NOISE_SHAPE, "reader action initial noise")
    noise_error = float((action_noise - teacher["noise"]).abs().max())
    if noise_error != 0.0:
        raise RuntimeError(f"Reader action initial noise differs from native BF16 for {stem}")
    if any(not torch.equal(action_noise, natives[(variant, arm)]["noise"]) for arm in ARMS):
        raise RuntimeError(f"Native initial noise differs across arms for {stem}")
    if donor == "bf16":
        parity = float((action - teacher["action"]).abs().max())
        if parity != 0.0:
            raise RuntimeError(f"BF16-reader replay of BF16 future differs from native action: {stem}")
    else:
        parity = None
    metrics = _metrics(torch, action, commands, teacher["action"],
                       teacher["commands"], stats)
    base = {
        **binding,
        "record_type": "fixed_bf16_reader",
        "status": "complete",
        "counts": dict(reader_counts),
        "action_initial_noise_identity": _noise_identity(torch, action_noise),
        "action_initial_noise_max_abs_vs_native": noise_error,
        "self_native_action_max_abs": parity,
        "metrics": metrics,
        "action_shape": list(action.shape),
        "commands_shape": list(commands.shape),
        "action_initial_noise_shape": list(action_noise.shape),
        "reader_checkpointed": False,
        "query_seconds": round(time.monotonic() - started, 6),
        "metric_scope": "first-10 motor KL and execution-command cumulative proxy; fixed input only",
    }
    payload = {
        **base,
        "action": action,
        "commands": commands,
        "action_initial_noise": action_noise,
    }
    metadata = dict(base)
    _save_record(torch, records_dir, stem, payload, metadata)
    return {"action": action, "commands": commands, "noise": action_noise,
            "counts": dict(reader_counts), "metrics": metrics, "metadata": metadata}


def _expected_keys(states):
    native = set()
    reader = set()
    for state in states:
        for variant in VARIANTS:
            for arm in ARMS:
                native.add((state["collector"], state["index"], state["step"], variant, arm))
                reader.add((state["collector"], state["index"], state["step"], variant, arm))
    return native, reader


def _record_counts_valid(native_records, reader_records):
    for record in native_records.values():
        if (record.get("covered_linears") != 614
                or record.get("counts") != {"video_steps": 10, "action_steps": 10}
                or record.get("query_coverage_verified") is not True):
            return False
    for record in reader_records.values():
        counts = record.get("counts", {})
        if (counts.get("video_dit_calls") != 0
                or counts.get("video_prefill_calls") != 1
                or counts.get("video_scheduler_steps") != 10
                or counts.get("action_dit_calls") != 10
                or counts.get("action_scheduler_steps") != 10
                or counts.get("teacher_future_injection_steps") != 10
                or counts.get("teacher_future_injection_verified") is not True):
            return False
    return True


def _initial_noise_parity_valid(native_records, reader_records, states):
    expected_groups = {
        (state["collector"], state["index"], state["step"], variant)
        for state in states for variant in VARIANTS
    }
    native_groups, reader_groups = {}, {}
    for key, record in native_records.items():
        group = key[:4]
        native_groups.setdefault(group, {})[key[4]] = record.get("action_initial_noise_identity")
    for key, record in reader_records.items():
        group = key[:4]
        reader_groups.setdefault(group, {})[key[4]] = record.get("action_initial_noise_identity")
    if set(native_groups) != expected_groups or set(reader_groups) != expected_groups:
        return False
    for group in expected_groups:
        native = native_groups[group]
        reader = reader_groups[group]
        if (set(native) != set(ARMS) or set(reader) != set(ARMS)
                or any(identity is None for identity in native.values())
                or any(identity is None for identity in reader.values())):
            return False
        if len({json.dumps(identity, sort_keys=True) for identity in native.values()}) != 1:
            return False
        bf16_identity = native["bf16"]
        if any(reader[arm] != bf16_identity for arm in ARMS):
            return False
    return True


def _write_summary(path, protocol, unit, shard, states, capture_summaries,
                   native_records, reader_records, counters, status):
    expected_native, expected_reader = _expected_keys(states)
    native_done = set(native_records)
    reader_done = set(reader_records)
    source_gate = all(item.get("source_replay_passed_all") is True
                      for item in capture_summaries.values())
    counts_valid = _record_counts_valid(native_records, reader_records)
    noise_parity_valid = _initial_noise_parity_valid(
        native_records, reader_records, states
    )
    complete = (
        source_gate
        and native_done == expected_native
        and reader_done == expected_reader
        and counts_valid
        and noise_parity_valid
    )
    per_state = []
    for state in states:
        key = (state["collector"], state["index"], state["step"])
        native_keys = {item for item in expected_native if item[:3] == key}
        reader_keys = {item for item in expected_reader if item[:3] == key}
        per_state.append({
            "collector": state["collector"],
            "capture_source_unit": state["capture_unit"],
            "index": state["index"],
            "step": state["step"],
            "seed": state["seed"],
            "native_endpoint_records_saved": len(native_done & native_keys),
            "native_endpoint_records_expected": len(native_keys),
            "fixed_reader_records_saved": len(reader_done & reader_keys),
            "fixed_reader_records_expected": len(reader_keys),
        })
    summary = {
        "protocol": protocol,
        "unit": unit,
        "status": "complete" if complete else status,
        "complete": complete,
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "capture_shard": shard,
        "source_capture_summaries": capture_summaries,
        "source_replay_passed_all": source_gate,
        "discordant_cases_expected": len({state["index"] for state in states}),
        "captured_states_expected": len(states),
        "captured_states_complete": sum(
            item["native_endpoint_records_saved"] == item["native_endpoint_records_expected"]
            and item["fixed_reader_records_saved"] == item["fixed_reader_records_expected"]
            for item in per_state
        ),
        "native_endpoint_records_expected": len(expected_native),
        "native_endpoint_records_saved": len(native_done),
        "fixed_reader_passes_expected": len(expected_reader),
        "fixed_reader_passes_saved": len(reader_done),
        "native_endpoint_loss_comparisons_expected_for_cpu_aggregate": len(states) * len(VARIANTS) * len(ARMS),
        "native_action_response_kl_comparisons_expected_for_cpu_aggregate": len(states) * len(ARMS),
        "native_metric_calculation": "deferred to CPU aggregate using softbin_loss.losses and action_response_kl",
        "full_model_queries_new_this_segment": counters["full_model_queries_new"],
        "full_model_queries_attempted_new_this_segment": counters["full_model_queries_attempted_new"],
        "native_action_steps_new_this_segment": counters["native_action_steps_new"],
        "fixed_bf16_reader_passes_new_this_segment": counters["reader_passes_new"],
        "fixed_reader_passes_attempted_new_this_segment": counters["reader_passes_attempted_new"],
        "fixed_reader_action_dit_calls_new_this_segment": counters["reader_action_dit_calls_new"],
        "new_training_updates": 0,
        "counts_valid": counts_valid,
        "native_and_reader_initial_noise_parity_valid": noise_parity_valid,
        "reader_mode": "BF16Reader checkpointed=False, each future cache rebuilt",
        "source_scope": "saved collector traces only; no env.step, render, or closed-loop rerun",
        "response_scope": "noisy-clean fixed-input response; states are repeated measures within episode",
        "expected_case_indices": sorted({int(state["index"]) for state in states}),
        "states": per_state,
        "updated_at_epoch": time.time(),
    }
    save(path, summary)
    return summary


def _load_stats(protocol_data):
    from reader_worker import _load_stats as load_reader_stats

    return load_reader_stats(protocol_data)


def _run_unit(args):
    protocol_data = read(args.protocol)
    protocol = protocol_data["protocol"]
    shard = int(args.unit[-1])
    root = Path(protocol_data["remote_root"])
    plan = read(root / "data" / "sensor_plan.json")
    states, capture_summaries = _validate_capture_sources(protocol, root, plan, shard)
    _load_capture_observation_meta(root, states)
    out = root / "results" / args.unit
    records_dir = out / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    run_binding = {
        "protocol": protocol,
        "unit": args.unit,
        "capture_shard": shard,
        "capture_units": sorted(item["unit"] for item in capture_summaries.values()),
        "states": [
            {"collector": item["collector"], "index": item["index"], "step": item["step"],
             "seed": item["seed"], "capture_state_json": item["capture_state_json"]}
            for item in states
        ],
        "native_matrix": "4 arms x clean/noisy for every captured state",
        "reader_matrix": "fixed BF16 reader x 4 native futures x clean/noisy for every state",
        "input_source": "capture NPZs; no rerender or env.step",
    }
    binding_path = out / "run_binding.json"
    if binding_path.is_file() and read(binding_path) != run_binding:
        raise ValueError("Existing trajectory output belongs to a different capture binding")
    if not binding_path.is_file():
        save(binding_path, run_binding)

    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    counters = {
        "full_model_queries_attempted_new": 0,
        "full_model_queries_new": 0,
        "native_action_steps_new": 0,
        "reader_passes_attempted_new": 0,
        "reader_passes_new": 0,
        "reader_action_dit_calls_new": 0,
    }
    native_records, reader_records = {}, {}
    status = "running"
    _write_summary(summary_path, protocol, args.unit, shard, states, capture_summaries,
                   native_records, reader_records, counters, status)

    from diagnostic_runtime import SwitchingRuntime
    from frozen_reader import BF16Reader
    import numpy as np
    import torch

    switching = None
    stats, _stats_path = _load_stats(protocol_data)
    deadline = Deadline(SOFT_DEADLINE_SECONDS)
    try:
        switching = SwitchingRuntime(protocol_data, out)
        bf16_reader = BF16Reader(switching.runtime, switching.original_weights)
        for state in states:
            datums = _load_state_inputs(torch, np, switching, root, state)
            natives = {}
            for variant in VARIANTS:
                for arm in ARMS:
                    record = _ensure_native(
                        torch, switching, records_dir, protocol, args.unit, state,
                        datums[variant], variant, arm, deadline, counters,
                    )
                    native_records[(state["collector"], state["index"], state["step"], variant, arm)] = record["metadata"]
                    natives[(variant, arm)] = record
                    _write_summary(summary_path, protocol, args.unit, shard, states,
                                   capture_summaries, native_records, reader_records,
                                   counters, status)
                reference_noise = natives[(variant, "bf16")]["noise"]
                for arm in ARMS[1:]:
                    if not torch.equal(reference_noise, natives[(variant, arm)]["noise"]):
                        raise RuntimeError(
                            f"Native initial action noise differs across arms: "
                            f"{state['collector']}/{state['index']}/step{state['step']}/{variant}"
                        )

            for variant in VARIANTS:
                for donor in ARMS:
                    record = _ensure_reader(
                        torch, switching, bf16_reader, stats, records_dir,
                        protocol, args.unit, state, datums[variant], variant,
                        donor, natives, deadline, counters,
                    )
                    reader_records[(state["collector"], state["index"], state["step"], variant, donor)] = record["metadata"]
                    _write_summary(summary_path, protocol, args.unit, shard, states,
                                   capture_summaries, native_records, reader_records,
                                   counters, status)
            del datums, natives
        status = "complete"
    except DeadlineReached:
        status = "resumable_deadline"
    except Exception:
        status = "failed"
        _write_summary(summary_path, protocol, args.unit, shard, states, capture_summaries,
                       native_records, reader_records, counters, status)
        raise
    finally:
        if switching is not None:
            switching.close()
        summary = _write_summary(summary_path, protocol, args.unit, shard, states,
                                 capture_summaries, native_records, reader_records,
                                 counters, status)
    print("TRAJECTORY_WORKER_SUMMARY " + json.dumps(summary), flush=True)
    return summary


def selfcheck():
    dual = Path(__file__).resolve().parents[1] / "fastwam-vq-dual-kl"
    if dual.is_dir() and str(dual) not in sys.path:
        sys.path.insert(0, str(dual))
    import torch

    assert _expected_capture_steps(60) == [0, 10, 50]
    assert _expected_capture_steps(200) == [0, 10, 50, 100]
    assert _expected_capture_steps(201) == list(CAPTURE_STEPS)
    # Smoke the production deadline API without creating PBS state or querying a model.
    previous_start = os.environ.get("FW_JOB_START_EPOCH")
    os.environ["FW_JOB_START_EPOCH"] = str(int(time.time()))
    try:
        assert isinstance(Deadline(SOFT_DEADLINE_SECONDS).near(RESERVE_SECONDS), bool)
    finally:
        if previous_start is None:
            os.environ.pop("FW_JOB_START_EPOCH", None)
        else:
            os.environ["FW_JOB_START_EPOCH"] = previous_start

    state = {"collector": "per_channel_clip", "index": 7, "step": 0}
    identity = {"shape": list(NOISE_SHAPE), "dtype": "torch.float32", "sha256": "toy"}
    native_records, reader_records = {}, {}
    for variant in VARIANTS:
        for arm in ARMS:
            key = (state["collector"], state["index"], state["step"], variant, arm)
            native_records[key] = {"action_initial_noise_identity": identity}
            reader_records[key] = {"action_initial_noise_identity": identity}
    assert _initial_noise_parity_valid(native_records, reader_records, [state])
    reader_records[(state["collector"], state["index"], state["step"], "clean", "relational")][
        "action_initial_noise_identity"
    ] = {**identity, "sha256": "mismatch"}
    assert not _initial_noise_parity_valid(native_records, reader_records, [state])

    stats = {
        "action": {"mean": [0.0] * 6, "scale": [1.0] * 6},
        "config": {"bins": 16, "interior_min": -3.5, "interior_max": 3.5,
                   "sigma": 0.375, "std_floor": 0.02},
    }
    from reader_worker import _metrics
    action = torch.zeros(ACTION_SHAPE)
    command = torch.zeros_like(action)
    command[:FIRST_CHUNK, :MOTOR_DIMS] = 1.0
    reader_metrics = _metrics(torch, action, command, action, torch.zeros_like(command), stats)
    expected_prefix = math.sqrt(sum(index * index for index in range(1, 11)) / 10)
    assert abs(reader_metrics["first10_command_motor_rmse"] - 1.0) < 1e-7
    assert abs(reader_metrics["command_prefix_first10_rmse"] - expected_prefix) < 1e-6
    assert reader_metrics["command_prefix_first10_max_abs"] == 10.0
    assert "not physical displacement" in reader_metrics["command_prefix_metric_scope"]
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU trajectory selfcheck unexpectedly initialized CUDA")
    print(json.dumps({
        "status": "passed",
        "expected_capture_steps": True,
        "deadline_api_and_noise_parity": True,
        "fixed_reader_command_prefix": True,
        "cuda_initialized": False,
    }))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", choices=("trajectory_0", "trajectory_1"))
    parser.add_argument("--protocol", default=str(Path(__file__).with_name("protocol.json")))
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    if args.unit is None:
        parser.error("--unit is required unless --selfcheck is used")
    guard()
    _run_unit(args)


if __name__ == "__main__":
    main()

