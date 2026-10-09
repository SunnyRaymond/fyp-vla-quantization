"""Prepare frozen rotations/banks and numerical gates inside one PBS allocation."""
from __future__ import annotations

import argparse
from collections import Counter
import faulthandler
import gc
import inspect
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import types

PROTOCOL = "fastwam-rotation-baselines-v1"
ARMS = ("quarot_adapted_w4a4", "spinquant_adapted_w4a4")
METHODS = {"video": "_denoise_video", "action": "_denoise_action_with_video_cache"}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def move_tree(value, device):
    import torch
    if isinstance(value, torch.Tensor):
        # CPU files are reloaded outside inference_mode before differentiable use.
        return value.detach().to(device=device).clone()
    if isinstance(value, dict):
        return {key: move_tree(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tree(item, device) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported frozen teacher input: {type(value)}")


def normalized_mse(prediction, teacher):
    import torch
    if prediction.shape != teacher.shape or not torch.isfinite(prediction).all():
        raise RuntimeError("Denoiser output is invalid or has changed shape")
    return (prediction.float() - teacher.float()).square().mean() / teacher.float().square().mean().clamp_min(1e-6)


def packet_order(rows, seed, steps):
    """Alternate streams, cycling seeded permutations so all inputs are covered."""
    pools = {stream: [row for row in rows if row["stream"] == stream] for stream in METHODS}
    if not all(pools.values()):
        raise ValueError("Both video and action calibration packets are required")
    generators = {stream: random.Random(seed + index) for index, stream in enumerate(METHODS)}
    queues = {stream: [] for stream in METHODS}
    result = []
    for index in range(steps):
        stream = "video" if index % 2 == 0 else "action"
        if not queues[stream]:
            queues[stream] = list(pools[stream])
            generators[stream].shuffle(queues[stream])
        result.append(queues[stream].pop())
    return result


def validate_protocol(document):
    if (document.get("protocol") != PROTOCOL or document.get("arms") != ["bf16", *ARMS]
            or document.get("rotation", {}).get("block_size") != 128):
        raise ValueError("Unexpected frozen experiment protocol")
    optimization = document["optimization"]
    if (optimization["steps"] != 200 or optimization["selection_interval"] != 40
            or optimization["optimizer"] != "Adam" or optimization["learning_rate"] != 0.001):
        raise ValueError("Unexpected frozen rotation optimization budget")
    if document["calibration"]["denoising_indices"] != [0, 3, 6, 9]:
        raise ValueError("Unexpected teacher trace timetable")


def freeze_controls(out, protocol, source):
    import cohort
    path = out / "protocol.json"
    if path.exists() and load_json(path) != protocol:
        raise RuntimeError("Existing results protocol differs; refusing to mix experiments")
    save(path, protocol)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        cohort.validate_cohort(manifest)
        if manifest["source_manifest"]["path"] != str(source.resolve()):
            raise RuntimeError("Frozen cohort source differs")
    else:
        cohort.build_cohort(source, manifest_path)


def prepare_observations(out, base):
    import calibration_inputs
    destination = out / "calibration_inputs"
    if not (destination / "plan.json").is_file():
        if destination.exists():
            raise RuntimeError("Calibration directory exists without a complete plan")
        subprocess.run([sys.executable, str(Path(calibration_inputs.__file__).resolve()),
                        "--out", str(destination), "--base-root", str(base)], check=True)
    plan = load_json(destination / "plan.json")
    calibration_inputs.validate_plan(plan)
    for case in plan["inputs"]:
        if not (destination / case["obsfile"]).is_file():
            raise FileNotFoundError(destination / case["obsfile"])
    return destination, plan


def datum_for(runtime, inputs, case):
    import numpy as np
    with np.load(inputs / case["obsfile"], allow_pickle=False) as archive:
        observation = {key: archive[key] for key in archive.files}
    runtime.description = case["description"]
    return runtime.datum(observation)


def collect_teacher(runtime, inputs, plan, root, protocol, progress):
    import torch
    root.mkdir(parents=True, exist_ok=True)
    runtime.q.enable("bf16")
    runtime.current_arm = "bf16"
    indices = protocol["calibration"]["denoising_indices"]
    packets, actions = [], {}
    for case in plan["inputs"]:
        case_id = case["case_id"]
        datum = datum_for(runtime, inputs, case)
        counts = Counter()
        originals = {stream: getattr(runtime.model, method) for stream, method in METHODS.items()}
        cache_written = False

        def capture(stream):
            original = originals[stream]
            signature = inspect.signature(original)

            def wrapped(_model, *args, **kwargs):
                nonlocal cache_written
                step = counts[stream]
                counts[stream] += 1
                selected = step in indices
                captured = None
                if selected:
                    bound = signature.bind(*args, **kwargs)
                    bound.apply_defaults()
                    arguments = dict(bound.arguments)
                    if stream == "action":
                        if not cache_written:
                            torch.save(move_tree({key: arguments[key] for key in
                                                  ("video_cache_k", "video_cache_v")}, "cpu"),
                                       root / f"cache_{case_id:02d}.pt")
                            cache_written = True
                        arguments.pop("video_cache_k")
                        arguments.pop("video_cache_v")
                    captured = move_tree(arguments, "cpu")
                prediction = original(*args, **kwargs)
                if selected:
                    filename = f"case_{case_id:02d}_{stream}_{step:02d}.pt"
                    torch.save({"kwargs": captured, "teacher": move_tree(prediction, "cpu"),
                                "proprio": move_tree(datum["proprio"], "cpu"),
                                "case_id": case_id, "stream": stream, "step": step}, root / filename)
                    packets.append({"case_id": case_id, "stream": stream, "step": step,
                                    "split": case["split"], "file": filename,
                                    "cache_file": f"cache_{case_id:02d}.pt" if stream == "action" else None})
                return prediction
            return types.MethodType(wrapped, runtime.model)

        for stream, method in METHODS.items():
            setattr(runtime.model, method, capture(stream))
        try:
            action, _ = runtime.infer(datum, case["sampler_seed"], measure=False)
        finally:
            for stream, method in METHODS.items():
                setattr(runtime.model, method, originals[stream])
        if counts != Counter(video=10, action=10):
            raise RuntimeError(f"Teacher did not execute the official 10+10 denoising calls: {counts}")
        actions[case_id] = action.detach().cpu().clone()
        progress("teacher", case_id=case_id, cases_completed=len(actions), expected_cases=12)
    if len(packets) != 96 or Counter(row["split"] for row in packets) != Counter(calibration=64, selection=32):
        raise RuntimeError("Teacher trace coverage is incomplete")
    torch.save(actions, root / "reference_actions.pt")
    save(root / "receipt.json", {"protocol": PROTOCOL, "complete": True, "packets": packets,
                                 "case_count": 12, "denoising_indices": indices,
                                 "source": "frozen original LIBERO observations", "evaluation_episodes": 0})
    return packets, actions


def replay_packet(runtime, root, row):
    import torch
    packet = torch.load(root / row["file"], map_location="cpu", weights_only=True)
    kwargs = move_tree(packet["kwargs"], "cuda")
    if row["stream"] == "action":
        kwargs.update(move_tree(torch.load(root / row["cache_file"], map_location="cpu", weights_only=True), "cuda"))
    proprio = move_tree(packet["proprio"], "cuda")
    if proprio.ndim == 1:
        proprio = proprio.unsqueeze(0)
    # The official query appended exactly one proprio token to text context.
    kwargs["context"], kwargs["context_mask"] = runtime.model._append_proprio_to_context(
        kwargs["context"][:, :-1], kwargs["context_mask"][:, :-1], proprio)
    prediction = getattr(runtime.model, METHODS[row["stream"]])(**kwargs)
    teacher = move_tree(packet["teacher"], "cuda")
    return normalized_mse(prediction, teacher)


def selection_score(runtime, state, trace, selection):
    import torch
    state.refresh()
    by_stream = {stream: [] for stream in METHODS}
    with torch.no_grad():
        for row in selection:
            value = float(replay_packet(runtime, trace, row))
            if not math.isfinite(value):
                raise RuntimeError("Selection loss is nonfinite")
            by_stream[row["stream"]].append(value)
    state.refresh()
    means = {stream: sum(values) / len(values) for stream, values in by_stream.items()}
    return {"mean": sum(means.values()) / len(means), "streams": means, "packets": len(selection)}


def offload_prompt_components(runtime, device):
    """Move auxiliary components only; preserve model's inference device."""
    # VAE's cudagraph callable captures GPU storage; rebuild after any move.
    if hasattr(runtime.model, "_vae_encode_compiled"):
        del runtime.model._vae_encode_compiled
        runtime.torch._dynamo.reset()
    for attribute in ("text_encoder", "vae"):
        component = getattr(runtime.model, attribute, None)
        if component is not None:
            component.to(device)
    gc.collect()
    runtime.torch.cuda.empty_cache()


def learn_rotation(runtime, state, trace, packets, out, protocol, progress):
    import torch
    import rotation
    optimization = protocol["optimization"]
    calibration = [row for row in packets if row["split"] == "calibration"]
    selection = [row for row in packets if row["split"] == "selection"]
    order = packet_order(calibration, optimization["seed"], optimization["steps"])
    optimizer = rotation.build_rotation_optimizer(state, lr=optimization["learning_rate"])
    handle = rotation.install_training_wrappers(runtime.q.modules, state, quantized=True, checkpointed=True)
    starting = {stream: state.matrix(stream).detach().cpu().clone() for stream in state.streams}
    state.refresh()
    grad_observations = Counter()
    best = None
    selections = []
    try:
        baseline = selection_score(runtime, state, trace, selection)
        save(out / "initial_selection.json", {"protocol": PROTOCOL, "step": 0, "score": baseline})
        for step, row in enumerate(order, 1):
            optimizer.zero_grad(set_to_none=True)
            loss = replay_packet(runtime, trace, row)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite learned rotation loss at step {step}")
            loss.backward()
            norms = {}
            for stream, transform in state.streams.items():
                gradient = transform.raw.grad
                if gradient is None:
                    continue
                if not torch.isfinite(gradient).all():
                    raise RuntimeError(f"Nonfinite {stream} rotation gradient at step {step}")
                norms[stream] = float(gradient.float().norm())
                if norms[stream] > 0:
                    grad_observations[stream] += 1
            torch.nn.utils.clip_grad_norm_([parameter for parameter in state.parameters() if parameter.requires_grad],
                                          optimization["gradient_clip_norm"], error_if_nonfinite=True)
            handle.optimizer_step(optimizer)
            value = float(loss.detach())
            del loss
            with (out / "training.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps({"step": step, "case_id": row["case_id"], "stream": row["stream"],
                                       "denoising_index": row["step"], "loss": value, "gradient_norms": norms}, allow_nan=False) + "\n")
            progress("rotation_learning", step=step, total_steps=optimization["steps"], stream=row["stream"], loss=value)
            if step % optimization["selection_interval"] == 0:
                score = selection_score(runtime, state, trace, selection)
                selections.append({"step": step, **score})
                if best is None or score["mean"] < best["mean"]:
                    best = {"step": step, **score}
                    torch.save({"state_dict": move_tree(state.state_dict(), "cpu"), "protocol": PROTOCOL,
                                "mode": state.mode, "seed": state.seed, "selection": best}, out / "learned_rotation.pt")
                save(out / "selection.json", {"protocol": PROTOCOL, "initial": baseline, "trained": selections, "best": best})
                print(f"ROTATION_SELECTION step={step} score={score['mean']:.8g} best_step={best['step']}", flush=True)
        selected = torch.load(out / "learned_rotation.pt", map_location="cpu", weights_only=True)
        state.load_state_dict(selected["state_dict"])
        state.refresh()
        changes = {stream: float((state.matrix(stream).detach().cpu() - starting[stream]).abs().max())
                   for stream, transform in state.streams.items() if transform.raw.requires_grad}
        trainable_streams = set(changes)
        if not trainable_streams or any(grad_observations[stream] == 0 or changes[stream] <= 1e-7
                                        for stream in trainable_streams):
            raise RuntimeError(f"Learning did not change all trainable rotations: gradients={dict(grad_observations)} changes={changes}")
        receipt = {"protocol": PROTOCOL, "completed_steps": len(order), "selected_step": best["step"],
                   "initial_selection": baseline, "selected_selection": best,
                   "gradient_nonzero_steps": dict(grad_observations), "max_abs_change": changes,
                   "calibration_packets": len(calibration), "selection_packets": len(selection),
                   "optimizer": optimization["optimizer"], "learning_rate": optimization["learning_rate"],
                   "ste_arithmetic": optimization["ste_arithmetic"],
                   "loss": optimization["loss"], "test_queries_used": 0, "base_weights_frozen": True}
        save(out / "training_receipt.json", receipt)
        return receipt
    finally:
        rotation.restore_original_linears(handle)
        state.refresh()
        gc.collect()
        torch.cuda.empty_cache()


def action_error(actual, reference):
    difference = actual.float().cpu() - reference.float().cpu()
    return {"motor_rmse_first10": float(difference[:10, :6].square().mean().sqrt()),
            "gripper_rmse_first10": float(difference[:10, 6].square().mean().sqrt()),
            "action_rmse_32": float(difference.square().mean().sqrt())}


def stage_existing_rotation_checks(out, protocol):
    """Keep original banks/training in place and preserve prior check receipts."""
    old = load_json(out / "protocol.json")
    for key in ("protocol", "arms", "checkpoint", "inference", "cohort", "quantization", "rotation", "calibration"):
        if old[key] != protocol[key]:
            raise RuntimeError(f"Existing rotation scientific controls differ: {key}")
    def normalized_gates(document):
        gates = dict(document["gates"])
        reference_keys = ("native_vs_ste_max_action_rmse_32", "native_vs_grouped_reference_max_action_rmse_32")
        present = [key for key in reference_keys if key in gates]
        if len(present) != 1:
            raise RuntimeError("Exactly one native equivalence RMSE threshold is required")
        gates["native_equivalence_max_action_rmse_32"] = gates.pop(present[0])
        return gates
    if normalized_gates(old) != normalized_gates(protocol):
        raise RuntimeError("Existing rotation numerical gate thresholds differ")
    for key in ("seed", "optimizer", "learning_rate", "steps", "selection_interval", "gradient_clip_norm", "packet_order", "loss", "selection", "boundary"):
        if old["optimization"][key] != protocol["optimization"][key]:
            raise RuntimeError(f"Existing rotation optimization controls differ: {key}")
    history = out / "validation_history" / os.environ["PBS_JOBID"]
    history.mkdir(parents=True, exist_ok=False)
    for name in ("protocol.json", "preparation_progress.json"):
        if (out / name).is_file():
            (history / name).write_text((out / name).read_text(encoding="utf-8"), encoding="utf-8")
    save(history / "reference_correction.json", {
        "previous_gates": old["gates"], "current_gates": protocol["gates"],
        "thresholds_unchanged": True, "training_and_selection_unchanged": True,
        "reference": protocol.get("validation_reference"),
    })
    save(out / "protocol.json", protocol)


def reuse_trained_rotation(state, out, protocol):
    import torch
    path = out / "rotation_training" / "learned_rotation.pt"
    selected = torch.load(path, map_location="cpu", weights_only=True)
    receipt = load_json(out / "rotation_training" / "training_receipt.json")
    if (selected.get("protocol") != PROTOCOL or selected.get("mode") != "learned"
            or selected.get("seed") != protocol["rotation"]["seed"]
            or receipt.get("completed_steps") != 200
            or selected["selection"]["step"] != receipt["selected_step"]):
        raise RuntimeError("Existing learned rotation is incomplete or mismatched")
    state.load_state_dict(selected["state_dict"])
    state.refresh()
    reused = {**receipt, "reused_existing_rotation": True, "source_rotation": str(path),
              "new_learning_steps": 0, "selection_recomputed": False,
              "training_ste_arithmetic": "legacy BF16 dequantized operands and BF16 F.linear; final BF16 output",
              "validation_ste_arithmetic": "dense FP32 dequantized operands/bias/GEMM retained as diagnostic; final BF16 output; no TF32",
              "native_gate_reference": "independent S8 integer dot using A4 codes, G128 FP32 scaled accumulation; final BF16"}
    save(out / "rotation_reuse_receipt.json", reused)
    return reused


def full_query_checks(runtime, state, arm, cases, inputs, references, protocol, progress):
    import torch
    import rotation
    orthogonality = {}
    with torch.no_grad():
        for stream in state.streams:
            matrix = state.matrix(stream)
            orthogonality[stream] = float((matrix.T @ matrix - torch.eye(128, device=matrix.device)).abs().max())
    state.refresh()
    if max(orthogonality.values()) > protocol["gates"]["orthogonality_max_abs_error"]:
        raise RuntimeError(f"Rotation orthogonality check failed: {orthogonality}")
    numeric, quantized_actions = [], {}
    for quantized in (False, True):
        handle = rotation.install_training_wrappers(runtime.q.modules, state, quantized=quantized, checkpointed=False)
        try:
            for case in cases:
                datum = datum_for(runtime, inputs, case)
                action, _ = runtime.infer(datum, case["sampler_seed"], measure=False)
                errors = action_error(action, references[case["case_id"]])
                if quantized:
                    quantized_actions[case["case_id"]] = action.detach().cpu().clone()
                else:
                    if (errors["motor_rmse_first10"] > protocol["gates"]["transformed_bf16_max_motor_rmse_first10"]
                            or errors["gripper_rmse_first10"] > protocol["gates"]["transformed_bf16_max_gripper_rmse_first10"]):
                        raise RuntimeError(f"Transformed BF16 numerical drift failed: {arm} {case['case_id']} {errors}")
                    numeric.append({"arm": arm, "case_id": case["case_id"], **errors})
                progress("full_query_checks", arm=arm, quantized=quantized, case_id=case["case_id"])
        finally:
            rotation.restore_original_linears(handle)
    return {"arm": arm, "passed": True, "orthogonality": orthogonality, "transformed_bf16": numeric}, quantized_actions


def main():
    import runner
    runner.guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, default=runner.ROOT / "manifest.json")
    parser.add_argument("--reuse-trained-rotation", action="store_true",
                        help="Validate existing learned rotation/banks without learning or reselecting")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    faulthandler.enable()
    protocol = load_json(args.protocol)
    validate_protocol(protocol)
    if (args.out / "PREPARED.json").exists():
        raise RuntimeError("Existing PREPARED receipt should be validated and reused by run.pbs")
    started = time.perf_counter()

    def progress(stage, **fields):
        receipt = {"protocol": PROTOCOL, "status": "complete" if stage == "complete" else "running", "stage": stage,
                   "pbs_jobid": os.environ["PBS_JOBID"], "elapsed_seconds": time.perf_counter() - started, **fields}
        save(args.out / "preparation_progress.json", receipt)
        print("PREPARATION_PROGRESS " + json.dumps(receipt, separators=(",", ":")), flush=True)

    runtime = None
    try:
        if args.reuse_trained_rotation:
            stage_existing_rotation_checks(args.out, protocol)
        progress("freeze_cohort")
        freeze_controls(args.out, protocol, args.source_manifest)
        progress("original_calibration_render")
        inputs, plan = prepare_observations(args.out, runner.BASE)
        import torch
        import rotation
        # STE decodes into FP32 to match native integer-dot scaling semantics.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(protocol["optimization"]["seed"])
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
        runner.PROTOCOL = PROTOCOL
        prep_runtime = args.out / "preparation_runtime"
        prep_runtime.mkdir(exist_ok=True)
        progress("model_loading")
        runtime = runner.Runtime(prep_runtime)
        for parameter in runtime.model.parameters():
            parameter.requires_grad_(False)
        trace = args.out / "teacher_trace"
        progress("teacher")
        if (trace / "receipt.json").is_file():
            receipt = load_json(trace / "receipt.json")
            if (receipt.get("protocol") != PROTOCOL or receipt.get("complete") is not True
                    or len(receipt.get("packets", [])) != 96):
                raise RuntimeError("Existing teacher trace is incomplete or mismatched")
            packets = receipt["packets"]
            references = torch.load(trace / "reference_actions.pt", map_location="cpu", weights_only=True)
        else:
            packets, references = collect_teacher(runtime, inputs, plan, trace, protocol, progress)
        runtime.q.enable("bf16")
        runtime.current_arm = "bf16"
        fixed = rotation.build_rotation_state(runtime.q.modules, mode="fixed", seed=protocol["rotation"]["seed"])
        learned = rotation.build_rotation_state(runtime.q.modules, mode="learned", seed=protocol["rotation"]["seed"])
        training = args.out / "rotation_training"
        training.mkdir(exist_ok=True)
        if args.reuse_trained_rotation:
            progress("reuse_trained_rotation")
            training_receipt = reuse_trained_rotation(learned, args.out, protocol)
        else:
            offload_prompt_components(runtime, "cpu")
            training_receipt = learn_rotation(runtime, learned, trace, packets, training, protocol, progress)
            offload_prompt_components(runtime, "cuda")
        selection = [case for case in plan["inputs"] if case["split"] == "selection"]
        numeric, fake_actions, banks = [], {}, {}
        for arm, state in zip(ARMS, (fixed, learned)):
            progress("numerical_gate", arm=arm)
            check, fake_actions[arm] = full_query_checks(runtime, state, arm, selection, inputs, references, protocol, progress)
            numeric.append(check)
            bank = args.out / "banks" / arm
            if args.reuse_trained_rotation:
                bank_receipt = load_json(bank / "manifest.json")
                if bank_receipt.get("arm") != arm or bank_receipt.get("group_size") != 128:
                    raise RuntimeError(f"Existing bank is mismatched: {arm}")
            else:
                rotation.export_native_bank(runtime.q.modules, state, bank, metadata={"arm": arm, "protocol": PROTOCOL,
                                          "base_checkpoint": str(runtime.cfg.ckpt), "training": training_receipt if arm == ARMS[1] else None})
            banks[arm] = str(bank)
        progress("bank_coherence")
        coherence = rotation.validate_native_bank_coherence(
            runtime.q.modules, dict(zip(ARMS, (fixed, learned))), banks
        )
        save(args.out / "bank_coherence_receipt.json", coherence)
        native = []
        for arm in ARMS:
            progress("native_gate", arm=arm)
            rotation.apply_native_bank(runtime.q, banks[arm])
            runtime.q.enable("w4a4")
            runtime.current_arm = arm
            for case in selection:
                before = runtime.q.summary()
                action, _ = runtime.infer(datum_for(runtime, inputs, case), case["sampler_seed"], measure=False)
                after = runtime.q.summary()
                dense_errors = action_error(action, fake_actions[arm][case["case_id"]])
                original_linear = runtime.q._linear
                reference_counters = (runtime.q._integer_gemm_calls, runtime.q._native_int4_gemm_calls)
                try:
                    runtime.q._linear = lambda module, value: rotation.grouped_int8_reference_linear(runtime.q, module, value)
                    reference_action, _ = runtime.infer(datum_for(runtime, inputs, case), case["sampler_seed"], measure=False)
                finally:
                    runtime.q._linear = original_linear
                if reference_counters != (runtime.q._integer_gemm_calls, runtime.q._native_int4_gemm_calls):
                    raise RuntimeError("Independent reference changed native GEMM counters")
                errors = action_error(action, reference_action)
                evidence = {"arm": arm, "case_id": case["case_id"], "errors": errors,
                            "reference": "independent S8 dot with A4 codes; G128 FP32 scaled accumulation",
                            "dense_fp32_ste_diagnostic_errors": dense_errors,
                            "reference_native_counters_unchanged": True,
                            "native_int4_tensorcore": after["native_int4_tensorcore"],
                            "native_int4_ptx_verified": after["native_int4_ptx_verified"],
                            "native_int4_gemm_calls": after["native_int4_gemm_calls"] - before["native_int4_gemm_calls"],
                            "bf16_target_weights_absent": after["bf16_target_linear_weights_absent"],
                            "kv4": after["kv4"]}
                save(args.out / "native_gate_last_receipt.json", evidence)
                if (errors["action_rmse_32"] > protocol["gates"]["native_vs_grouped_reference_max_action_rmse_32"]
                        or not after["native_int4_tensorcore"] or not after["native_int4_ptx_verified"]
                        or after["native_int4_gemm_calls"] <= before["native_int4_gemm_calls"]
                        or not after["bf16_target_linear_weights_absent"] or after["kv4"]):
                    raise RuntimeError(f"Native W4A4 gate failed: {evidence}")
                native.append({"arm": arm, "case_id": case["case_id"], "vs_grouped_integer_reference": errors,
                               "dense_fp32_ste_diagnostic_errors": dense_errors,
                               "reference_native_counters_unchanged": True,
                               "native_int4_gemm_calls": after["native_int4_gemm_calls"] - before["native_int4_gemm_calls"],
                               "native_int4_ptx_line": after["native_int4_ptx_line"],
                               "packed_weight_bytes": after["packed_weight_tensor_bytes"],
                               "bf16_target_weights_absent": True, "kv_cache": "bf16"})
        receipt = {"protocol": PROTOCOL, "ready": True, "banks": banks,
                   "validation_mode": "reuse_trained_rotation" if args.reuse_trained_rotation else "learn_and_validate",
                   "numeric_check": {"passed": True, "arms": numeric},
                   "bank_coherence_check": coherence,
                   "native_int4_check": {"passed": True, "queries": native},
                   "training": training_receipt, "evaluation_episodes": 0,
                   "source_checkpoint": str(runtime.cfg.ckpt), "pipeline_initializations": 1,
                   "pbs_jobid": os.environ["PBS_JOBID"], "elapsed_seconds": time.perf_counter() - started}
        save(args.out / "preparation_receipt.json", receipt)
        save(args.out / "PREPARED.json", receipt)
        progress("complete", ready=True)
        print("ROTATION_PREPARATION_COMPLETE", flush=True)
    except Exception as error:
        save(args.out / "preparation_progress.json", {"protocol": PROTOCOL, "status": "failed",
                                                      "error_type": type(error).__name__, "error": str(error),
                                                      "elapsed_seconds": time.perf_counter() - started})
        raise
    finally:
        if runtime is not None:
            if getattr(runtime.q, "_rotation_native_state", None) is not None:
                rotation.discard_native_bank(runtime.q)
            runtime.q.close()


if __name__ == "__main__":
    main()
