"""Run the frozen BF16 future-to-action reader matrix on a PBS GPU allocation."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import time

from control_common import Deadline, guard, read, save


ARMS = ("bf16", "per_channel_clip", "softbin", "relational")
STUDENT_ARMS = ARMS[1:]
VARIANTS = ("clean", "noisy")
FUTURE_SHAPE = (1, 48, 3, 14, 28)
ACTION_SHAPE = (32, 7)
MOTOR_DIMS = 6
FIRST_CHUNK = 10
TOTAL_CASES = 40
PASSES_PER_PAIR = 7
RESERVE_SECONDS = 180
SOFT_DEADLINE_SECONDS = 6600


class DeadlineReached(RuntimeError):
    pass


def _record_path(out, index, reader, donor, variant, self_check=False):
    if self_check:
        name = f"{index}_{reader}_self_{variant}.json"
    else:
        name = f"{index}_{reader}_from_{donor}_{variant}.json"
    return out / "records" / name


def _identity_from_metadata(metadata, variant):
    try:
        identity = metadata["observations"][variant]["input_identity"]
        if set(identity) != {"image", "proprio"}:
            raise ValueError("input identity must contain image and proprio")
        for key in ("image", "proprio"):
            item = identity[key]
            if not isinstance(item.get("shape"), list) or not item.get("dtype") or not item.get("sha256"):
                raise ValueError(f"invalid {key} input identity")
        return identity
    except (KeyError, TypeError) as exc:
        raise ValueError(f"sensor metadata lacks the {variant} input identity") from exc


def _read_metadata(root, case, protocol):
    index = int(case["index"])
    path = root / "data" / "sensor_initial" / str(index) / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing saved sensor metadata: {path}")
    metadata = read(path)
    expected = {
        "protocol": protocol,
        "index": index,
        "row": case["row"],
        "environment_seed": int(case["environment_seed"]),
        "sampler_seed": int(case["sampler_seed"]),
    }
    mismatch = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatch:
        raise ValueError(f"sensor metadata binding differs for {index}: {mismatch}")
    if not isinstance(metadata.get("description"), str) or not metadata["description"]:
        raise ValueError(f"sensor metadata has no fixed description for {index}")
    for variant in VARIANTS:
        _identity_from_metadata(metadata, variant)
    return metadata


def _binding(protocol, unit, case, metadata, variant, reader, donor):
    return {
        "protocol": protocol,
        "unit": unit,
        "index": int(case["index"]),
        "reader_arm": reader,
        "future_donor_arm": donor,
        "variant": variant,
        "seed": int(case["sampler_seed"]),
        "description": metadata["description"],
        "input_identity": _identity_from_metadata(metadata, variant),
    }


def _validate_saved_reader(record, expected, record_type, self_native=False):
    mismatch = [key for key, value in expected.items() if record.get(key) != value]
    if mismatch:
        raise ValueError("Saved reader output binding differs: " + ", ".join(mismatch))
    if record.get("record_type") != record_type:
        raise ValueError("Saved reader output has a different record type")
    if record.get("status") != "complete":
        raise ValueError("Saved reader output is not complete")
    if tuple(record.get("normalized_action_shape", ())) != ACTION_SHAPE:
        raise ValueError("Saved reader action shape is invalid")
    if tuple(record.get("execution_command_shape", ())) != ACTION_SHAPE:
        raise ValueError("Saved execution command shape is invalid")
    for name in ("normalized_action", "execution_command"):
        values = record.get(name)
        if not isinstance(values, list) or len(values) != ACTION_SHAPE[0]:
            raise ValueError(f"Saved {name} is missing or has the wrong shape")
        if any(not isinstance(row, list) or len(row) != ACTION_SHAPE[1] for row in values):
            raise ValueError(f"Saved {name} is not [32,7]")
    counts = record.get("counts", {})
    required_counts = {
        "video_dit_calls": 0,
        "video_denoise_loop_calls": 10,
        "video_scheduler_steps": 10,
        "action_dit_calls": 10,
        "action_scheduler_steps": 10,
        "video_prefill_calls": 1,
        "teacher_future_injection_steps": 10,
    }
    if any(counts.get(key) != value for key, value in required_counts.items()):
        raise ValueError("Saved reader counts do not show the required 10+10 pass")
    if counts.get("teacher_future_injection_verified") is not True:
        raise ValueError("Saved reader future injection was not verified")
    for key in ("future_tail_matches", "first_frame_clamp_verified",
                "action_conditioning_future_matches", "action_conditioning_first_frame_clamped",
                "future_first_frame_matches_observation"):
        if counts.get(key) is not True:
            raise ValueError(f"Saved reader did not verify {key}")
    noise = record.get("action_noise_parity", {})
    if noise.get("max_abs_vs_bf16_native") != 0.0 or noise.get("max_abs_vs_reader_native") != 0.0:
        raise ValueError("Saved action-initial-noise parity is not exact")
    if self_native and record.get("self_native_action_max_abs") != 0.0:
        raise ValueError("Saved self-action replay parity is not exact")
    return record


def _validate_gate_record(record, binding):
    expected = {key: value for key, value in binding.items()
                if key != "future_donor_arm"}
    expected.setdefault("donor_arm", "relational")
    mismatch = [key for key, value in expected.items() if record.get(key) != value]
    if mismatch:
        raise ValueError("Saved gradient-gate binding differs: " + ", ".join(mismatch))
    counts = record.get("counts", {})
    if (record.get("passed") is not True or record.get("checkpointed") is not True
            or record.get("detached_action_parity_max_abs") != 0.0
            or record.get("action_initial_noise_max_abs_vs_bf16_native") != 0.0
            or record.get("future_gradient_finite") is not True
            or int(record.get("future_gradient_nonzero_elements", 0)) <= 0
            or record.get("frame0_gradient_max_abs") != 0.0
            or record.get("parameters_and_weights_unchanged") is not True
            or record.get("model_and_bank_grads_clear") is not True
            or counts.get("video_dit_calls") != 0
            or counts.get("video_prefill_calls") != 1
            or counts.get("video_scheduler_steps") != 10
            or counts.get("action_dit_calls") != 10
            or counts.get("action_scheduler_steps") != 10):
        raise ValueError("Saved gradient gate does not satisfy the gradient/parity/count checks")
    loss = float(record.get("softbin_first10_motor_kl", float("nan")))
    if not math.isfinite(loss):
        raise ValueError("Saved gradient-gate soft-bin KL is not finite")
    return record


def _expected_core_tasks(cases):
    for case in cases:
        for variant in VARIANTS:
            for donor in ARMS:
                yield case, variant, "bf16", donor
            for reader in STUDENT_ARMS:
                yield case, variant, reader, "bf16"


def _expected_self_tasks(first_case):
    for reader in STUDENT_ARMS:
        yield first_case, "clean", reader, reader


def _complete_artifacts_exist(out, protocol, unit, cases, metadata_by_key, summary):
    if (summary.get("protocol") != protocol or summary.get("unit") != unit
            or summary.get("complete") is not True
            or summary.get("core_records_saved") != 280
            or summary.get("q_self_checks_saved") != 3):
        return False
    def check(case, variant, reader, donor, self_check=False):
        key = (int(case["index"]), variant)
        metadata = metadata_by_key[key]
        expected = _binding(protocol, unit, case, metadata, variant, reader, donor)
        path = _record_path(out, expected["index"], reader, donor, variant, self_check)
        if not path.is_file():
            return False
        record = read(path)
        _validate_saved_reader(
            record, expected, "q_self_check" if self_check else "core",
            self_native=(reader == donor and (reader == "bf16" or self_check)),
        )
        return True

    for case, variant, reader, donor in _expected_core_tasks(cases):
        if not check(case, variant, reader, donor):
            return False
    first_case = min(cases, key=lambda item: int(item["ordinal"]))
    for case, variant, reader, donor in _expected_self_tasks(first_case):
        if not check(case, variant, reader, donor, self_check=True):
            return False
    if unit == "reader_0":
        if summary.get("gradient_gate", {}).get("passed") is not True:
            return False
        gate_path = out / "gradient_gate.json"
        if not gate_path.is_file():
            return False
        gate = read(gate_path)
        expected_gate = {
            "protocol": protocol,
            "unit": unit,
            "index": int(first_case["index"]),
            "reader_arm": "bf16",
            "donor_arm": "relational",
            "variant": "noisy",
            "seed": int(first_case["sampler_seed"]),
            "description": metadata_by_key[(int(first_case["index"]), "noisy")]["description"],
            "input_identity": _identity_from_metadata(
                metadata_by_key[(int(first_case["index"]), "noisy")], "noisy"
            ),
        }
        _validate_gate_record(gate, expected_gate)
    return True


def _validate_plan(plan, protocol, shard):
    if plan.get("protocol") != protocol or plan.get("complete") is not True:
        raise ValueError("sensor_plan.json is incomplete or belongs to another protocol")
    cases = plan.get("cases")
    if not isinstance(cases, list) or len(cases) != TOTAL_CASES:
        raise ValueError(f"Expected all {TOTAL_CASES} frozen sensor cases")
    ordinals = [int(case["ordinal"]) for case in cases]
    if sorted(ordinals) != list(range(TOTAL_CASES)):
        raise ValueError("sensor case ordinals must cover 0 through 39 exactly once")
    if any(int(case["shard"]) != int(case["ordinal"]) % 2 for case in cases):
        raise ValueError("sensor plan shard must equal ordinal modulo 2")
    selected = [case for case in cases if int(case["shard"]) == shard]
    if len(selected) != 20:
        raise ValueError(f"Expected 20 cases for reader shard {shard}, got {len(selected)}")
    if len({int(case["index"]) for case in selected}) != len(selected):
        raise ValueError("sensor plan contains duplicate case indices")
    return sorted(selected, key=lambda item: int(item["ordinal"]))


def _load_npz(np, path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _native_record(torch, root, case, metadata, variant, arm, protocol):
    index = int(case["index"])
    sensor_unit = f"sensor_{int(case['shard'])}"
    stem = f"{index}_{arm}_{variant}"
    directory = root / "results" / sensor_unit / "records"
    tensor_path, json_path = directory / f"{stem}.pt", directory / f"{stem}.json"
    if not tensor_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(f"Missing cached sensor endpoint {stem}")
    expected = {
        "protocol": protocol,
        "unit": sensor_unit,
        "index": index,
        "arm": arm,
        "variant": variant,
        "seed": int(case["sampler_seed"]),
        "description": metadata["description"],
        "input_identity": _identity_from_metadata(metadata, variant),
    }
    payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
    meta = read(json_path)
    for record in (payload, meta):
        mismatch = [key for key, value in expected.items() if record.get(key) != value]
        if mismatch:
            raise ValueError(f"Cached sensor binding differs for {stem}: {mismatch}")
    counts = payload.get("counts")
    if counts != {"video_steps": 10, "action_steps": 10}:
        raise ValueError(f"Cached sensor endpoint {stem} lacks exact 10+10 counts")
    action, future = payload.get("action"), payload.get("future")
    saved_command, json_command = payload.get("commands"), meta.get("commands")
    noise = payload.get("action_initial_noise")
    if (not isinstance(action, torch.Tensor) or action.dtype != torch.float32
            or not isinstance(future, torch.Tensor) or future.dtype != torch.float32
            or tuple(action.shape) != ACTION_SHAPE or tuple(future.shape) != FUTURE_SHAPE):
        raise ValueError(f"Cached endpoint shape differs for {stem}")
    if saved_command is None or json_command is None:
        raise ValueError(f"Cached endpoint has no saved execution commands: {stem}")
    command = torch.as_tensor(saved_command, dtype=torch.float32).detach().cpu()
    command_json = torch.as_tensor(json_command, dtype=torch.float32).detach().cpu()
    if tuple(command.shape) != ACTION_SHAPE or tuple(command_json.shape) != ACTION_SHAPE:
        raise ValueError(f"Cached execution command shape differs for {stem}")
    if not bool(torch.isfinite(command).all() and torch.isfinite(command_json).all()):
        raise ValueError(f"Cached execution command contains non-finite values: {stem}")
    if not torch.equal(command, command_json):
        raise ValueError(f"Cached PT/JSON execution commands differ for {stem}")
    if not isinstance(noise, torch.Tensor) or tuple(noise.shape) != (1, 32, 7):
        raise ValueError(f"Cached endpoint has no captured action noise: {stem}")
    if not bool(torch.isfinite(action).all() and torch.isfinite(future).all()
                and torch.isfinite(noise).all()):
        raise ValueError(f"Cached endpoint contains non-finite tensors: {stem}")
    noise_shape = meta.get("action_initial_noise_shape")
    if noise_shape is not None and tuple(noise_shape) != tuple(noise.shape):
        raise ValueError(f"Cached action-noise shape binding differs for {stem}")
    noise_dtype = meta.get("action_initial_noise_dtype")
    if noise_dtype is not None and noise_dtype != str(noise.dtype):
        raise ValueError(f"Cached action-noise dtype binding differs for {stem}")
    noise_identity = meta.get("action_initial_noise_identity")
    if noise_identity is not None:
        from diagnostic_runtime import tensor_identity
        if tensor_identity(torch, noise) != noise_identity:
            raise ValueError(f"Cached action-noise identity differs for {stem}")
    return {"action": action, "command": command, "future": future,
            "noise": noise.float().cpu(), "counts": counts}


def _load_pair(torch, np, switching, root, case, metadata, variant, protocol):
    index = int(case["index"])
    obs_path = root / "data" / "sensor_initial" / str(index) / f"observation_{variant}.npz"
    if not obs_path.is_file():
        raise FileNotFoundError(f"Missing saved sensor observation: {obs_path}")
    obs = _load_npz(np, obs_path)
    datum = switching.datum(obs)
    from diagnostic_runtime import datum_identity
    identity = datum_identity(torch, datum)
    expected_identity = _identity_from_metadata(metadata, variant)
    if identity != expected_identity:
        raise ValueError(f"Recomputed model-input identity differs for case {index} {variant}")
    switching.runtime.description = metadata["description"]
    natives = {
        arm: _native_record(torch, root, case, metadata, variant, arm, protocol)
        for arm in ARMS
    }
    reference_noise = natives["bf16"]["noise"]
    for arm in ARMS[1:]:
        if not torch.equal(reference_noise, natives[arm]["noise"]):
            raise ValueError(f"Native action initial noise differs across arms for {index} {variant}")
    return datum, natives


@contextmanager
def _count_method(obj, name):
    namespace = getattr(obj, "__dict__", {})
    existed, previous = name in namespace, namespace.get(name)
    original = getattr(obj, name)
    counter = {"calls": 0}

    def counted(*args, **kwargs):
        counter["calls"] += 1
        return original(*args, **kwargs)

    setattr(obj, name, counted)
    try:
        yield counter
    finally:
        if existed:
            setattr(obj, name, previous)
        else:
            delattr(obj, name)


def _compact_counts(torch, diagnostic, prefill_count, action_step_count):
    video_steps = int(diagnostic.get("video_scheduler_steps", -1))
    loop_calls = int(diagnostic.get("video_denoise_loop_calls", -1))
    video_dit = int(diagnostic.get("video_dit_calls", -1))
    action_dit = int(diagnostic.get("action_dit_calls", -1))
    injection_steps = int(diagnostic.get("teacher_future_injection_steps", -1))
    if (video_steps != 10 or loop_calls != 10 or video_dit != 0 or action_dit != 10
            or injection_steps != 10 or prefill_count != 1 or action_step_count != 10):
        raise RuntimeError(
            "Reader call counts differ from the required video-prefill + 10 action steps: "
            f"{diagnostic}, prefill_impl={prefill_count}, action_scheduler={action_step_count}"
        )
    if diagnostic.get("teacher_future_injection_verified") is not True:
        raise RuntimeError("Reader did not verify injected future and observed-frame clamp")
    aliases = (
        ("future_tail_matches", ("future_tail_matches", "teacher_future_latents_unchanged")),
        ("first_frame_clamp_verified", ("first_frame_clamp_verified",)),
        ("action_conditioning_future_matches",
         ("action_conditioning_future_matches", "action_conditioning_future_matches_teacher")),
        ("action_conditioning_first_frame_clamped",
         ("action_conditioning_first_frame_clamped",
          "action_conditioning_first_frame_matches_observation")),
        ("future_first_frame_matches_observation",
         ("future_first_frame_matches_observation", "teacher_first_frame_matches_observation")),
    )
    compact = {
        "video_dit_calls": video_dit,
        "video_denoise_loop_calls": loop_calls,
        "video_scheduler_steps": video_steps,
        "action_dit_calls": action_dit,
        "action_scheduler_steps": int(action_step_count),
        "video_prefill_calls": int(prefill_count),
        "teacher_future_injection_steps": injection_steps,
        "teacher_future_injection_verified": True,
    }
    for target, sources in aliases:
        value = next((diagnostic.get(source) for source in sources
                      if diagnostic.get(source) is not None), None)
        if value is not True:
            raise RuntimeError(f"Reader failed future/clamp check {source}={value}")
        compact[target] = True
    return compact


def _run_intervention(torch, switching, bf16_reader, reader, datum, seed, future):
    switching.activate(reader)
    model = switching.runtime.model
    with _count_method(model.mot, "prefill_video_cache_tensor") as prefill_count, \
            _count_method(model.infer_action_scheduler, "step") as action_step_count:
        if reader == "bf16":
            action, diagnostic = bf16_reader(
                switching.runtime, datum, seed, future, checkpointed=False
            )
        else:
            from future_adapter import run_teacher_future
            output = run_teacher_future(switching.runtime, datum, seed, future)
            action, diagnostic = output["action"], output["_diagnostic"]
    counts = _compact_counts(torch, diagnostic, prefill_count["calls"], action_step_count["calls"])
    noise = diagnostic.get("action_initial_noise")
    if not isinstance(noise, torch.Tensor) or tuple(noise.shape) != (1, 32, 7):
        raise RuntimeError("Reader did not expose its actual action-initial-noise tensor")
    return action, counts, noise.detach().float().cpu()


def _command(runtime, action):
    command = runtime.command(action)
    if not isinstance(command, (list, tuple)) and not hasattr(command, "shape"):
        raise TypeError("runtime.command(action) must return a numeric [32,7] command")
    return command


def _chunk_summary(torch, action, target):
    delta = action[:FIRST_CHUNK, :MOTOR_DIMS].float() - target[:FIRST_CHUNK, :MOTOR_DIMS].float()
    signed = delta.mean(dim=0)
    prefix_delta = delta.cumsum(dim=0)
    return {
        "rmse": float(delta.square().mean().sqrt()),
        "max_abs": float(delta.abs().max()),
        "prefix_cumsum_rmse": float(prefix_delta.square().mean().sqrt()),
        "prefix_cumsum_max_abs": float(prefix_delta.abs().max()),
        "signed_mean_by_motor": signed.tolist(),
        "absolute_chunk_bias_mean": float(signed.abs().mean()),
        "absolute_chunk_bias_by_motor": signed.abs().tolist(),
    }


def _metrics(torch, action, command, teacher_action, teacher_command, stats):
    from control_losses import latent_control_kl

    action = torch.as_tensor(action).detach().float().cpu()
    command = torch.as_tensor(command).detach().float().cpu()
    teacher_action = torch.as_tensor(teacher_action).detach().float().cpu()
    teacher_command = torch.as_tensor(teacher_command).detach().float().cpu()
    if any(tuple(value.shape) != ACTION_SHAPE for value in
           (action, command, teacher_action, teacher_command)):
        raise ValueError("Action and execution command must all have shape [32,7]")
    action_bias = _chunk_summary(torch, action, teacher_action)
    command_bias = _chunk_summary(torch, command, teacher_command)
    kl = latent_control_kl(torch, action, teacher_action, stats)
    return {
        "first10_motor_rmse": action_bias["rmse"],
        "first10_command_motor_rmse": command_bias["rmse"],
        "softbin_first10_motor_kl": float(kl.detach()),
        "normalized_action_chunk_bias": action_bias,
        "execution_command_chunk_bias": command_bias,
        "command_prefix_first10_rmse": command_bias["prefix_cumsum_rmse"],
        "command_prefix_first10_max_abs": command_bias["prefix_cumsum_max_abs"],
        "command_prefix_metric_scope": (
            "cumulative command-delta proxy over first 10 steps; not physical displacement"
        ),
    }


def _execute_pass(torch, switching, bf16_reader, runtime, stats, binding,
                  reader, donor, datum, donor_future, teacher_action,
                  teacher_command, native_reader, bf16_native_noise,
                  record_type, out, started):
    self_check = record_type == "q_self_check"
    path = _record_path(out, binding["index"], reader, donor,
                        binding["variant"], self_check=self_check)
    expected_action = native_reader["action"] if self_check else (
        native_reader["action"] if reader == "bf16" and donor == "bf16" else None
    )
    if path.is_file():
        record = _validate_saved_reader(
            read(path), binding, record_type,
            self_native=(reader == donor and (reader == "bf16" or self_check)),
        )
        return record, False

    action, counts, action_noise = _run_intervention(
        torch, switching, bf16_reader, reader, datum, int(binding["seed"]), donor_future
    )
    if not bool(torch.isfinite(action).all()):
        raise RuntimeError("Reader output contains non-finite values")
    if action.requires_grad:
        raise RuntimeError("Diagnostic reader pass unexpectedly retained an autograd graph")
    noise_vs_bf16 = float((action_noise - bf16_native_noise).abs().max())
    noise_vs_reader = float((action_noise - native_reader["noise"]).abs().max())
    if noise_vs_bf16 != 0.0 or noise_vs_reader != 0.0:
        raise RuntimeError("Reader action-initial-noise differs from native same-seed noise")

    action_cpu = action.detach().float().cpu()
    self_error = None
    if expected_action is not None:
        self_error = float((action_cpu - expected_action).abs().max())
        if self_error != 0.0:
            raise RuntimeError(
                f"Exact self-reader replay failed for {binding['index']} {reader}/{donor} "
                f"{binding['variant']}: max_abs={self_error}"
            )
    command = _command(runtime, action)
    command_cpu = torch.as_tensor(command).detach().float().cpu()
    metrics = _metrics(torch, action_cpu, command_cpu, teacher_action, teacher_command, stats)
    record = {
        **binding,
        "record_type": record_type,
        "status": "complete",
        "reference_scope": "bf16_behavior_reference",
        "query_pbs_jobid": os.environ.get("PBS_JOBID"),
        "query_seconds": round(time.monotonic() - started, 6),
        "action_noise_parity": {
            "max_abs_vs_bf16_native": noise_vs_bf16,
            "max_abs_vs_reader_native": noise_vs_reader,
        },
        "self_native_action_max_abs": self_error,
        "counts": counts,
        "metrics": metrics,
        "normalized_action_shape": list(action_cpu.shape),
        "normalized_action": action_cpu.tolist(),
        "execution_command_shape": list(command_cpu.shape),
        "execution_command": command_cpu.tolist(),
    }
    save(path, record)
    return record, True


def _parameter_snapshot(model, original_weights, banks):
    import torch

    def sig(tensor):
        return (int(tensor.data_ptr()), int(getattr(tensor, "_version", -1)),
                tuple(tensor.shape), str(tensor.dtype), bool(tensor.requires_grad))

    parameters = {
        name: sig(parameter)
        for name, parameter in model.named_parameters()
    }
    weights = {name: sig(weight) for name, weight in original_weights.items()}
    bank_tensors = {}

    def collect(value, label):
        if isinstance(value, torch.Tensor):
            bank_tensors[label] = sig(value)
        elif isinstance(value, dict):
            for key, item in value.items():
                collect(item, f"{label}.{key}")
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                collect(item, f"{label}.{i}")

    for name, value in vars(banks).items():
        collect(value, name)
    grads_clear = all(parameter.grad is None for parameter in model.parameters())
    weight_grads_clear = all(weight.grad is None for weight in original_weights.values())
    bank_grads_clear = all(
        tensor.grad is None for value in vars(banks).values()
        for tensor in _container_tensors(value)
    )
    return {"parameters": parameters, "weights": weights, "bank_tensors": bank_tensors,
            "model_grads_clear": grads_clear, "weight_grads_clear": weight_grads_clear,
            "bank_grads_clear": bank_grads_clear}


def _container_tensors(value):
    import torch
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _container_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _container_tensors(item)


def _run_gradient_gate(torch, switching, bf16_reader, runtime, stats, case,
                       metadata, datum, natives, output_record, out):
    from control_losses import latent_control_kl

    index, variant = int(case["index"]), "noisy"
    reader, donor = "bf16", "relational"
    binding = _binding(
        switching.p["protocol"], f"reader_{int(case['shard'])}", case,
        metadata, variant, reader, donor,
    )
    path = out / "gradient_gate.json"
    if path.is_file():
        saved = read(path)
        return _validate_gate_record(saved, binding), False

    if output_record.get("record_type") != "core" or output_record.get("future_donor_arm") != donor:
        raise ValueError("Gradient gate requires the saved BF16 reader / relational-future action")
    target = natives["bf16"]["action"].to(device=next(runtime.model.parameters()).device)
    baseline = torch.tensor(output_record["normalized_action"], device=target.device,
                            dtype=torch.float32)
    switching.activate("bf16")
    model = runtime.model
    parameter_state_before = _parameter_snapshot(model, switching.original_weights, switching.banks)
    if not (parameter_state_before["model_grads_clear"]
            and parameter_state_before["weight_grads_clear"]
            and parameter_state_before["bank_grads_clear"]):
        raise RuntimeError("Model or channel-bank gradients were already populated before the gate")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Reader weights and model parameters must be frozen")
    device = getattr(model, "device", target.device)
    future = natives[donor]["future"].to(device=device, dtype=getattr(model, "torch_dtype", torch.bfloat16)).clone()
    future.requires_grad_(True)
    started = time.monotonic()
    with _count_method(model.mot, "prefill_video_cache_tensor") as prefill_count, \
            _count_method(model.infer_action_scheduler, "step") as action_step_count:
        action, counts = bf16_reader(runtime, datum, int(case["sampler_seed"]), future,
                                     checkpointed=True)
    if not bool(torch.equal(action.detach().float().cpu(), baseline.float().cpu())):
        parity = float((action.detach().float().cpu() - baseline.float().cpu()).abs().max())
        raise RuntimeError(f"Gradient-enabled reader parity differs from detached action: {parity}")
    # BF16Reader records one cache prefill at the forward boundary. The wrapped
    # implementation can run again during non-reentrant checkpoint recomputation.
    compact_counts = _compact_counts(
        torch, counts, int(counts.get("video_prefill_calls", 0)),
        action_step_count["calls"],
    )
    action_noise = counts.get("action_initial_noise")
    if not isinstance(action_noise, torch.Tensor):
        raise RuntimeError("Gradient gate did not capture actual action initial noise")
    noise_error = float((action_noise.detach().float().cpu() - natives["bf16"]["noise"]).abs().max())
    if noise_error != 0.0:
        raise RuntimeError("Gradient gate initial action noise differs from BF16 native noise")
    loss = latent_control_kl(torch, action, target, stats)
    grad, = torch.autograd.grad(loss, future, retain_graph=False, create_graph=False)
    grad_finite = bool(torch.isfinite(grad).all())
    grad_nonzero = bool(torch.count_nonzero(grad).item() > 0)
    frame0_max = float(grad[:, :, 0].abs().max())
    if not (grad_finite and grad_nonzero and frame0_max == 0.0):
        raise RuntimeError(
            f"Gradient gate failed: finite={grad_finite}, nonzero={grad_nonzero}, "
            f"frame0_max={frame0_max}"
        )
    parameter_state_after = _parameter_snapshot(model, switching.original_weights, switching.banks)
    same_params = (parameter_state_before["parameters"] == parameter_state_after["parameters"]
                   and parameter_state_before["weights"] == parameter_state_after["weights"]
                   and parameter_state_before["bank_tensors"] == parameter_state_after["bank_tensors"])
    grads_clear = (parameter_state_after["model_grads_clear"]
                   and parameter_state_after["weight_grads_clear"]
                   and parameter_state_after["bank_grads_clear"])
    if not same_params or not grads_clear:
        raise RuntimeError("Gradient gate changed reader/model parameters or accumulated parameter grads")
    result = {
        "protocol": binding["protocol"],
        "unit": binding["unit"],
        "index": index,
        "reader_arm": reader,
        "donor_arm": donor,
        "variant": variant,
        "seed": binding["seed"],
        "description": binding["description"],
        "input_identity": binding["input_identity"],
        "passed": True,
        "scope": "one diagnostic gradient gate; no optimizer step or closed-loop claim",
        "detached_action_parity_max_abs": 0.0,
        "action_initial_noise_max_abs_vs_bf16_native": noise_error,
        "softbin_first10_motor_kl": float(loss.detach().float().cpu()),
        "future_gradient_finite": grad_finite,
        "future_gradient_nonzero_elements": int(torch.count_nonzero(grad).item()),
        "future_gradient_max_abs": float(grad.abs().max().detach().cpu()),
        "future_gradient_l2": float(grad.float().norm().detach().cpu()),
        "frame0_gradient_max_abs": frame0_max,
        "checkpointed": True,
        "counts": compact_counts,
        "prefill_implementation_calls_including_recompute": int(prefill_count["calls"]),
        "reader_query_seconds": round(time.monotonic() - started, 6),
        "model_parameter_count_checked": len(parameter_state_before["parameters"]),
        "original_bf16_weight_count_checked": len(parameter_state_before["weights"]),
        "channel_bank_tensor_count_checked": len(parameter_state_before["bank_tensors"]),
        "model_and_bank_grads_clear": grads_clear,
        "parameters_and_weights_unchanged": same_params,
        "query_pbs_jobid": os.environ.get("PBS_JOBID"),
    }
    save(path, result)
    return result, True


def _load_stats(protocol):
    source = protocol["source_relational_protocol"]["source_dual_root"]
    path = Path(source) / "results" / "fit_stats" / "summary.json"
    stats = read(path)
    if (stats.get("protocol") != "fastwam-vq-dual-kl-v1"
            or stats.get("complete") is not True
            or stats.get("heldout_used") is not False
            or stats.get("new_teacher_queries") != 0
            or stats.get("calibration_case_ids") != list(range(2000, 2080))):
        raise ValueError("Frozen soft-bin fit statistics do not match the calibration contract")
    action = stats.get("action", {})
    if len(action.get("mean", [])) != MOTOR_DIMS or len(action.get("scale", [])) != MOTOR_DIMS:
        raise ValueError("Frozen action calibration statistics must cover the six motor channels")
    return stats, str(path)


def _write_summary(path, protocol, unit, cases, core_done, self_done, gate,
                   query_counts, status):
    all_core = len(core_done) == 280
    all_self = len(self_done) == 3
    gate_required = unit == "reader_0"
    core_records = [record for record in core_done.values()]
    self_records = [record for record in self_done.values()]
    all_records = core_records + self_records
    bf16_self_zero = bool(all(
        record.get("self_native_action_max_abs") == 0.0
        for record in core_records
        if record.get("reader_arm") == "bf16" and record.get("future_donor_arm") == "bf16"
    )) and sum(1 for record in core_records
               if record.get("reader_arm") == "bf16" and record.get("future_donor_arm") == "bf16") == 40
    q_self_zero = bool(len(self_records) == 3 and all(
        record.get("self_native_action_max_abs") == 0.0 for record in self_records
    ))
    counts_valid = bool(len(all_records) == 283 and all(
        record.get("counts", {}).get("action_dit_calls") == 10
        and record.get("counts", {}).get("video_scheduler_steps") == 10
        and record.get("counts", {}).get("video_dit_calls") == 0
        and record.get("counts", {}).get("video_prefill_calls") == 1
        for record in all_records
    ))
    complete = (all_core and all_self and bf16_self_zero and q_self_zero
                and counts_valid and (not gate_required or gate.get("passed") is True))
    summary = {
        "protocol": protocol,
        "unit": unit,
        "status": "complete" if complete else status,
        "complete": complete,
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "cases_expected": 20,
        "variants_per_case": 2,
        "core_reader_passes_expected": 280,
        "core_records_saved": len(core_done),
        "q_self_checks_expected": 3,
        "q_self_checks_saved": len(self_done),
        "q_self_check_coverage": "first case in shard, clean variant, three student readers only",
        "teacher_full_forward_count_new": 0,
        "reader_intervention_passes_new_this_segment": query_counts["passes_new"],
        "reader_action_dit_calls_new_this_segment": query_counts["action_dit_calls_new"],
        "reader_intervention_records_saved": len(all_records),
        "reader_action_dit_calls_saved": sum(
            int(record.get("counts", {}).get("action_dit_calls", 0)) for record in all_records
        ) + int(gate.get("counts", {}).get("action_dit_calls", 0)),
        "core_reader_checkpointed": False,
        "gradient_gate": gate if gate else {"required": gate_required, "status": "pending" if gate_required else "not_required"},
        "bf16_self_parity_all_zero": bf16_self_zero,
        "q_self_parity_all_zero": q_self_zero,
        "counts_valid": counts_valid,
        "reference_label": "BF16 behavior reference; not a success oracle",
        "checkpointing_and_resume": "matching completed endpoint records are reused; incomplete summary remains resumable",
        "expected_case_indices": [int(case["index"]) for case in cases],
        "created_at_epoch": time.time(),
    }
    save(path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", required=True, choices=("reader_0", "reader_1"))
    parser.add_argument("--protocol", default=str(Path(__file__).with_name("protocol.json")))
    args = parser.parse_args()

    guard()
    deadline = Deadline(SOFT_DEADLINE_SECONDS)
    protocol_data = read(args.protocol)
    protocol = protocol_data["protocol"]
    shard = int(args.unit[-1])
    root = Path(protocol_data["remote_root"])
    out = root / "results" / args.unit
    records_dir = out / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    plan = read(root / "data" / "sensor_plan.json")
    cases = _validate_plan(plan, protocol, shard)
    first_case = min(cases, key=lambda item: int(item["ordinal"]))
    metadata_by_key = {
        (int(case["index"]), variant): _read_metadata(root, case, protocol)
        for case in cases for variant in VARIANTS
    }
    run_binding_path = out / "run_binding.json"
    run_binding = {
        "protocol": protocol,
        "unit": args.unit,
        "sensor_plan_source_job": plan.get("source_evaluation_job"),
        "case_indices": [int(case["index"]) for case in cases],
        "variants": list(VARIANTS),
        "reader_matrix": "bf16<-4 futures; 3 student readers<-bf16 future",
    }
    if run_binding_path.is_file() and read(run_binding_path) != run_binding:
        raise ValueError("Existing reader output directory belongs to a different frozen binding")
    if not run_binding_path.is_file():
        save(run_binding_path, run_binding)

    summary_path = out / "summary.json"
    previous_summary = read(summary_path) if summary_path.is_file() else {}
    if _complete_artifacts_exist(out, protocol, args.unit, cases,
                                 metadata_by_key, previous_summary):
        print(json.dumps({"unit": args.unit, "status": "already_complete", "complete": True}))
        return

    from diagnostic_runtime import SwitchingRuntime
    from frozen_reader import BF16Reader
    import numpy as np
    import torch

    stats, stats_path = _load_stats(protocol_data)
    artifact_dir = Path(os.environ.get("ARTIFACTS", out))
    switching = None
    query_counts = {"passes_new": 0, "action_dit_calls_new": 0}
    core_records, self_records = {}, {}
    gate = {}
    status = "running"
    try:
        switching = SwitchingRuntime(protocol_data, artifact_dir)
        bf16_reader = BF16Reader(switching.runtime, switching.original_weights)
        # Load and validate any existing endpoint records once; individual input
        # identities are checked again against the recomputed datum before reuse.
        for case, variant, reader, donor in _expected_core_tasks(cases):
            binding = _binding(protocol, args.unit, case,
                               metadata_by_key[(int(case["index"]), variant)],
                               variant, reader, donor)
            path = _record_path(out, binding["index"], reader, donor, variant)
            if path.is_file():
                record = _validate_saved_reader(read(path), binding, "core",
                                                self_native=(reader == "bf16" and donor == "bf16"))
                core_records[(binding["index"], variant, reader, donor, "core")] = record
        for case, variant, reader, donor in _expected_self_tasks(first_case):
            binding = _binding(protocol, args.unit, case,
                               metadata_by_key[(int(case["index"]), variant)],
                               variant, reader, donor)
            path = _record_path(out, binding["index"], reader, donor, variant, self_check=True)
            if path.is_file():
                record = _validate_saved_reader(read(path), binding, "q_self_check", self_native=True)
                self_records[(binding["index"], variant, reader, donor, "q_self_check")] = record

        if args.unit == "reader_0" and (out / "gradient_gate.json").is_file():
            gate = read(out / "gradient_gate.json")
            gate_binding = _binding(
                protocol, args.unit, first_case,
                metadata_by_key[(int(first_case["index"]), "noisy")],
                "noisy", "bf16", "relational",
            )
            _validate_gate_record(gate, gate_binding)

        _write_summary(summary_path, protocol, args.unit, cases,
                       core_records, self_records, gate, query_counts, status)
        stop = False
        for case in cases:
            for variant in VARIANTS:
                metadata = metadata_by_key[(int(case["index"]), variant)]
                datum, natives = _load_pair(torch, np, switching, root, case,
                                            metadata, variant, protocol)
                teacher_action = natives["bf16"]["action"]
                teacher_command = natives["bf16"]["command"]
                for reader, donor in (("bf16", arm) for arm in ARMS):
                    key = (int(case["index"]), variant, reader, donor, "core")
                    if key in core_records:
                        continue
                    if deadline.near(RESERVE_SECONDS):
                        status, stop = "resumable_deadline", True
                        break
                    binding = _binding(protocol, args.unit, case, metadata,
                                       variant, reader, donor)
                    query_counts["passes_new"] += 1
                    query_counts["action_dit_calls_new"] += 10
                    _write_summary(summary_path, protocol, args.unit, cases,
                                   core_records, self_records, gate, query_counts, status)
                    record, _created = _execute_pass(
                        torch, switching, bf16_reader, switching.runtime, stats,
                        binding, reader, donor, datum, natives[donor]["future"],
                        teacher_action, teacher_command,
                        natives[reader], natives["bf16"]["noise"],
                        "core", out, time.monotonic(),
                    )
                    core_records[key] = record
                    _write_summary(summary_path, protocol, args.unit, cases,
                                   core_records, self_records, gate, query_counts, status)
                if stop:
                    break
                for reader in STUDENT_ARMS:
                    donor = "bf16"
                    key = (int(case["index"]), variant, reader, donor, "core")
                    if key in core_records:
                        continue
                    if deadline.near(RESERVE_SECONDS):
                        status, stop = "resumable_deadline", True
                        break
                    binding = _binding(protocol, args.unit, case, metadata,
                                       variant, reader, donor)
                    query_counts["passes_new"] += 1
                    query_counts["action_dit_calls_new"] += 10
                    _write_summary(summary_path, protocol, args.unit, cases,
                                   core_records, self_records, gate, query_counts, status)
                    record, _created = _execute_pass(
                        torch, switching, bf16_reader, switching.runtime, stats,
                        binding, reader, donor, datum, natives[donor]["future"],
                        teacher_action, teacher_command,
                        natives[reader], natives["bf16"]["noise"],
                        "core", out, time.monotonic(),
                    )
                    core_records[key] = record
                    _write_summary(summary_path, protocol, args.unit, cases,
                                   core_records, self_records, gate, query_counts, status)
                if stop:
                    break
            if stop:
                break

        if not stop and len(core_records) == 280:
            metadata = metadata_by_key[(int(first_case["index"]), "clean")]
            datum, natives = _load_pair(torch, np, switching, root, first_case,
                                        metadata, "clean", protocol)
            teacher_action = natives["bf16"]["action"]
            teacher_command = natives["bf16"]["command"]
            for case, variant, reader, donor in _expected_self_tasks(first_case):
                key = (int(case["index"]), variant, reader, donor, "q_self_check")
                if key in self_records:
                    continue
                if deadline.near(RESERVE_SECONDS):
                    status, stop = "resumable_deadline", True
                    break
                binding = _binding(protocol, args.unit, case, metadata,
                                   variant, reader, donor)
                query_counts["passes_new"] += 1
                query_counts["action_dit_calls_new"] += 10
                _write_summary(summary_path, protocol, args.unit, cases,
                               core_records, self_records, gate, query_counts, status)
                record, _created = _execute_pass(
                    torch, switching, bf16_reader, switching.runtime, stats,
                    binding, reader, donor, datum, natives[donor]["future"],
                    teacher_action, teacher_command,
                    natives[reader], natives["bf16"]["noise"],
                    "q_self_check", out, time.monotonic(),
                )
                self_records[key] = record
                _write_summary(summary_path, protocol, args.unit, cases,
                               core_records, self_records, gate, query_counts, status)

        if (not stop and args.unit == "reader_0" and len(core_records) == 280
                and len(self_records) == 3 and not gate):
            if deadline.near(RESERVE_SECONDS):
                status = "resumable_deadline"
            else:
                gate_case = first_case
                gate_metadata = metadata_by_key[(int(gate_case["index"]), "noisy")]
                datum, natives = _load_pair(torch, np, switching, root, gate_case,
                                            gate_metadata, "noisy", protocol)
                key = (int(gate_case["index"]), "noisy", "bf16", "relational", "core")
                output_record = core_records.get(key)
                if output_record is None:
                    raise RuntimeError("Missing BF16-reader relational-noisy detached action")
                query_counts["passes_new"] += 1
                query_counts["action_dit_calls_new"] += 10
                _write_summary(summary_path, protocol, args.unit, cases,
                               core_records, self_records, gate, query_counts, status)
                gate, _created = _run_gradient_gate(
                    torch, switching, bf16_reader, switching.runtime, stats,
                    gate_case, gate_metadata, datum, natives, output_record, out,
                )

        final = _write_summary(summary_path, protocol, args.unit, cases,
                               core_records, self_records, gate, query_counts, status)
        print(json.dumps({
            "unit": args.unit,
            "complete": final["complete"],
            "status": final["status"],
            "core_records": final["core_records_saved"],
            "q_self_checks": final["q_self_checks_saved"],
            "reader_queries_new_this_segment": query_counts["passes_new"],
            "teacher_full_forward_count_new": 0,
            "fit_stats": stats_path,
        }, allow_nan=False))
    except BaseException:
        status = "failed_resumable"
        _write_summary(summary_path, protocol, args.unit, cases,
                       core_records, self_records, gate, query_counts, status)
        raise
    finally:
        if switching is not None:
            switching.close()


if __name__ == "__main__":
    main()
