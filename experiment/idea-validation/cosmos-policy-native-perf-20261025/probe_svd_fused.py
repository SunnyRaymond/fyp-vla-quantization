"""A100-only correctness, stream, and CUDA Graph probe for fused SVDQuant."""
from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

from prepare_sources import guard


NUNCHAKU_COMMIT = "302e0e97024ebd68688fe890e5df83731edf7b54"
DEEPCOMPRESSOR_COMMIT = "69f3473f5e1c1504bae35cc50c7858ef900a9b17"
MODULE_NAME = "nunchaku_svdq_fused_native"
BUILD_ARM = "svdquant_w4a4_fused_candidate"
SEED = 777
K = 256
N = 131
RANK = 16
GROUP_SIZE = 128
NRMSE_LIMIT = 0.01
FP16_MAX_ABS_ATOL = 0.125
M_CASES = (129, 1764)
NATIVE_EXECUTIONS = {"default_current_stream": 1,
                     "non_default_current_stream": 1,
                     "cuda_graph_replay": 3}
CAPTURE_ENTRYPOINT_CALLS = 1
FORWARD_ENTRYPOINTS = ("pack_activation", "pack_activation_scales",
                       "quantize_down", "matmul_lora_up")


def _output_path(path: Path) -> Path:
    return path if path.suffix.lower() == ".json" else path / "probe_svd_fused.json"


def _partial_output_path(path: Path) -> Path:
    final = _output_path(path)
    return final.with_name(final.stem + ".partial" + final.suffix)


def _write_partial(path: Path, receipt: dict) -> None:
    raw = json.dumps(receipt, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    if len(raw.encode("utf-8")) > 16 * 1024:
        raise RuntimeError("Fused SVD probe partial receipt exceeded 16 KiB")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _progress_writer(path: Path, receipt: dict):
    def report(case, phase: str, entrypoint: str, state: str) -> None:
        row = {**receipt, "status": "failed" if state in ("call_error", "sync_error") else "running",
               "last_kernel_entry": {"case": case, "phase": phase,
                                      "entrypoint": entrypoint, "state": state,
                                      "timestamp_utc": datetime.now(timezone.utc).isoformat()}}
        _write_partial(path, row)
        print(json.dumps({"status": "running", "case": case, "phase": phase,
                          "entrypoint": entrypoint, "state": state},
                         ensure_ascii=False, separators=(",", ":")), flush=True)
    return report


class _SynchronizedProbeExtension:
    """Sync each diagnostic entrypoint so async device faults name their source."""
    _ENTRYPOINTS = frozenset((*FORWARD_ENTRYPOINTS, "pack_weight", "pack_weight_scales"))

    def __init__(self, extension, torch, progress, current_case, current_phase):
        self._extension = extension
        self._torch = torch
        self._progress = progress
        self._current_case = current_case
        self._current_phase = current_phase

    def __getattr__(self, name):
        value = getattr(self._extension, name)
        if name not in self._ENTRYPOINTS or not callable(value):
            return value

        def invoke(*args, **kwargs):
            case, phase = self._current_case(), self._current_phase()
            self._progress(case, phase, name, "entered")
            try:
                result = value(*args, **kwargs)
            except Exception as exc:
                self._progress(case, phase, name, "call_error")
                raise RuntimeError(
                    f"Fused probe extension entrypoint {name} raised "
                    f"(case={case}, phase={phase})") from exc
            self._progress(case, phase, name, "returned")
            try:
                is_capturing = self._torch.cuda.is_current_stream_capturing()
            except Exception as exc:
                self._progress(case, phase, name, "sync_error")
                raise RuntimeError(
                    f"CUDA state check failed after fused probe entrypoint {name} "
                    f"(case={case}, phase={phase})") from exc
            if is_capturing:
                self._progress(case, phase, name, "capture_recorded")
            else:
                try:
                    self._torch.cuda.current_stream().synchronize()
                except Exception as exc:
                    self._progress(case, phase, name, "sync_error")
                    raise RuntimeError(
                        f"CUDA error immediately after fused probe entrypoint {name} "
                        f"(case={case}, phase={phase})") from exc
                self._progress(case, phase, name, "stream_synchronized")
            return result

        return invoke


def _case_spec(true_m: int, n: int = N, k: int = K) -> dict:
    if true_m <= 0 or n <= 0 or k <= 0 or k % GROUP_SIZE:
        raise ValueError("Probe requires positive M/N and K divisible by G128")
    return {"true_m": true_m,
            "padded_m": ((true_m + 255) // 256) * 256,
            "true_n": n,
            "padded_n": ((n + 127) // 128) * 128,
            "k": k,
            "weight_groups_g128": k // GROUP_SIZE,
            "rank": RANK}


def _load_build_receipt(root: Path) -> tuple[dict, dict]:
    path = root / "results" / "preparation" / "native_svd_fused_build.json"
    if not path.is_file():
        raise FileNotFoundError(f"Fused Nunchaku build receipt is required: {path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    arm = receipt.get("arms", {}).get(BUILD_ARM)
    if (receipt.get("status") != "complete" or not receipt.get("pbs_job_id")
            or receipt.get("arch") != "sm_80" or not isinstance(arm, dict)):
        raise RuntimeError("Fused Nunchaku receipt is incomplete or was not produced by an SM80 PBS build")
    if arm.get("source_commit") != NUNCHAKU_COMMIT or arm.get("module_name") != MODULE_NAME:
        raise RuntimeError("Fused build receipt does not match the pinned Nunchaku module")
    exporter = arm.get("packing_exporter_source", {})
    if (exporter.get("commit") != DEEPCOMPRESSOR_COMMIT
            or exporter.get("function") != "NunchakuWeightPacker.pack_lowrank_weight"):
        raise RuntimeError("Fused build receipt does not bind the reviewed low-rank packing exporter")
    fusion = arm.get("native_kernel_fusion", {})
    if (not fusion.get("sidecar_groupwise_A4_outputs_discarded")
            or fusion.get("checkpoint_codes_scales_factors_requantized") is not False):
        raise RuntimeError("Fused build receipt does not attest the frozen-checkpoint sidecar contract")
    module = Path(arm.get("module", "")).resolve()
    extension_dir = (root / "native_extensions" / "svdquant_w4a4_fused").resolve()
    try:
        module.relative_to(extension_dir)
    except ValueError as exc:
        raise RuntimeError("Fused module path in build receipt escapes its isolated build directory") from exc
    if not module.is_file():
        raise FileNotFoundError(f"Fused Nunchaku module from build receipt is missing: {module}")
    return receipt, arm


def _load_extension(module_path: Path):
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    if not module_path.is_file() or not module_path.name.endswith(suffixes):
        raise FileNotFoundError(f"Built fused kernel binary is missing or has an invalid extension suffix: {module_path}")
    spec = importlib.util.spec_from_file_location(MODULE_NAME, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load the fused native extension: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(MODULE_NAME, None)
        raise
    required = ("pack_activation", "pack_weight", "pack_activation_scales",
                "pack_weight_scales", "quantize_down", "matmul_lora_up")
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError(f"Fused Nunchaku extension is missing required calls: {missing}")
    return module


def _hardware(torch, device) -> dict:
    props = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    uuid = str(getattr(props, "uuid", "") or "")
    if not uuid:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            check=True, capture_output=True, text=True)
        rows = [row.split(",", 1) for row in result.stdout.splitlines() if "," in row]
        visible = [part.strip() for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if part.strip()]
        active = int(torch.cuda.current_device())
        if visible and active < len(visible):
            token = visible[active]
            if token.startswith(("GPU-", "MIG-")):
                uuid = next((row[1].strip() for row in rows if row[1].strip() == token), "")
            else:
                physical = token if token.isdigit() else str(active)
                uuid = next((row[1].strip() for row in rows if row[0].strip() == physical), "")
        elif active < len(rows):
            uuid = rows[active][1].strip()
    if capability != (8, 0) or "A100" not in props.name.upper() or not uuid:
        raise RuntimeError(f"Probe requires an identifiable A100 SM80 GPU; found {props.name} SM{capability[0]}{capability[1]}")
    return {"gpu_uuid": uuid, "gpu_name": props.name,
            "total_memory_bytes": int(props.total_memory),
            "capability": [int(capability[0]), int(capability[1])],
            "device_index": int(torch.cuda.current_device())}


def _pack_checkpoint_codes(weight_codes, torch):
    # Frozen symmetric code q in [-7,7] maps to the baseline bank's offset-8 nibbles.
    encoded = (weight_codes.to(torch.int16) + 8).to(torch.uint8)
    return (encoded[:, 0::2] | (encoded[:, 1::2] << 4)).contiguous()


def _make_bank_and_layouts(torch, functional, extension, device):
    import native_diffusion
    import native_svd_fused
    from types import SimpleNamespace

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    weight_codes = torch.randint(-7, 8, (N, K), device=device,
                                 dtype=torch.int16).to(torch.int8)
    group0 = 0.05 + torch.rand((N,), device=device, dtype=torch.float32) * 0.05
    group1 = 0.18 + torch.rand((N,), device=device, dtype=torch.float32) * 0.12
    weight_scales = torch.stack((group0, group1), dim=1).to(torch.float16).contiguous()
    if bool(torch.any(weight_scales[:, 0] == weight_scales[:, 1]).item()):
        raise AssertionError("G128 test scales must remain distinct across the two groups")
    smoothing = (0.75 + torch.rand((K,), device=device, dtype=torch.float32) * 0.5).to(torch.bfloat16)
    down_bf16 = (torch.randn((RANK, K), device=device, dtype=torch.float32) * 0.10).to(torch.bfloat16)
    up_bf16 = (torch.randn((N, RANK), device=device, dtype=torch.float32) * 0.10).to(torch.bfloat16)
    bank = {
        "encoded": {"shape": (N, K), "codes": _pack_checkpoint_codes(weight_codes, torch),
                    "scales": weight_scales},
        "smoothing": smoothing,
        "down": down_bf16,
        "up": up_bf16,
        "rank": RANK,
    }
    dummy_module = SimpleNamespace(weight=torch.empty((N, K), device=device, dtype=torch.bfloat16))
    core = native_diffusion._nunchaku_bank(
        bank, dummy_module, extension, torch, functional, RANK)
    sidecar_k = ((core["padded_k"] + 127) // 128) * 128
    down_fp16 = native_svd_fused._fp16_factor(down_bf16, torch, "down")
    up_fp16 = native_svd_fused._fp16_factor(up_bf16, torch, "up")
    packed_down = native_svd_fused._pack_lowrank_weight(
        down_fp16, down=True, padded_n=core["padded_n"], padded_k=sidecar_k,
        functional=functional, torch=torch)
    packed_up = native_svd_fused._pack_lowrank_weight(
        up_fp16, down=False, padded_n=core["padded_n"], padded_k=core["padded_k"],
        functional=functional, torch=torch)
    return {"core": core, "bank": bank, "weight_codes": weight_codes,
            "weight_scales": weight_scales, "down_fp16": down_fp16,
            "up_fp16": up_fp16, "packed_down": packed_down,
            "packed_up": packed_up, "sidecar_k": sidecar_k}


def _reference_main(a_codes, a_scale_half, weight_codes, weight_scales, torch):
    groups64 = K // 64
    result = torch.zeros((a_codes.shape[0], weight_codes.shape[0]),
                         device=a_codes.device, dtype=torch.float32)
    a = a_codes.to(torch.float32)
    w = weight_codes.to(torch.float32)
    for group64 in range(groups64):
        begin = group64 * 64
        end = begin + 64
        # K=64 signed-int4 partials are exactly representable in FP32.
        partial = torch.matmul(a[:, begin:end], w[:, begin:end].transpose(0, 1))
        wscale = weight_scales[:, group64 // 2].to(torch.float32)
        result.add_(partial * a_scale_half.to(torch.float32)[:, None] * wscale[None, :])
    return result


def _metrics(output, expected, label: str, torch) -> dict:
    if output.dtype != torch.float16:
        raise RuntimeError(f"{label}: fused kernel must return FP16, got {output.dtype}")
    if tuple(output.shape) != tuple(expected.shape):
        raise RuntimeError(f"{label}: output/reference shapes differ: {tuple(output.shape)} vs {tuple(expected.shape)}")
    if not bool(torch.isfinite(output).all().item()):
        raise AssertionError(f"{label}: fused kernel returned nonfinite values")
    error = output.to(torch.float32) - expected
    max_abs = float(error.abs().max().item())
    rms = float(torch.sqrt(torch.mean(error.square())).item())
    reference_rms = float(torch.sqrt(torch.mean(expected.square())).item())
    nrmse = rms / max(reference_rms, 1e-12)
    if max_abs > FP16_MAX_ABS_ATOL:
        raise AssertionError(f"{label}: max abs error {max_abs:.7g} exceeds fixed FP16 atol {FP16_MAX_ABS_ATOL}")
    if nrmse > NRMSE_LIMIT:
        raise AssertionError(f"{label}: NRMSE {nrmse:.7g} exceeds fixed limit {NRMSE_LIMIT}")
    return {"max_abs_error": max_abs, "rmse": rms,
            "reference_rms": reference_rms, "nrmse": nrmse,
            "max_abs_atol": FP16_MAX_ABS_ATOL,
            "nrmse_limit": NRMSE_LIMIT}


def _probe_case(spec, state, extension, torch, functional, device,
                progress, current_case, current_phase) -> dict:
    import native_diffusion
    import native_svd_fused

    true_m, padded_m = spec["true_m"], spec["padded_m"]
    true_n, padded_n = spec["true_n"], spec["padded_n"]
    core = state["core"]
    native_svd_fused._fusion_shape_contract(
        true_m, true_n, padded_m, padded_n, state["sidecar_k"], RANK)
    torch.manual_seed(SEED + true_m)
    torch.cuda.manual_seed_all(SEED + true_m)
    smooth = state["bank"]["smoothing"].to(torch.float32)
    raw_input = torch.randn((true_m, K), device=device, dtype=torch.float32)
    value = raw_input * smooth[None, :]
    normalized = value.float() / smooth[None, :]

    a_scale = normalized.abs().amax(dim=-1, keepdim=True) / 7
    a_scale_safe = torch.where(a_scale > 0, a_scale, torch.ones_like(a_scale))
    a_codes = normalized.div(a_scale_safe).round().clamp(-7, 7).to(torch.int8)
    a_scale_half = a_scale_safe.to(torch.float16).reshape(true_m)
    expected_main = _reference_main(
        a_codes, a_scale_half, state["weight_codes"], state["weight_scales"], torch)
    sidecar_input = normalized.to(torch.float16).contiguous()
    # Oracle is FP32 using exactly the FP16 activation/factor operands consumed by Nunchaku.
    expected_down = torch.matmul(sidecar_input.float(), state["down_fp16"].float().transpose(0, 1))
    expected_lowrank = torch.matmul(expected_down, state["up_fp16"].float().transpose(0, 1))
    expected = expected_main + expected_lowrank

    current_case["value"] = {"true_m": true_m, "true_n": true_n,
                              "padded_m": padded_m, "padded_n": padded_n}
    current_phase["value"] = "case_reference"
    progress(current_case["value"], "case_reference", "oracle_tensor_setup", "before_sync")
    try:
        torch.cuda.current_stream(device).synchronize()
    except Exception as exc:
        progress(current_case["value"], "case_reference", "oracle_tensor_setup", "sync_error")
        raise RuntimeError(f"CUDA error before native kernels for M={true_m}") from exc
    progress(current_case["value"], "case_reference", "oracle_tensor_setup", "stream_synchronized")

    phases = {}
    call_count = {name: 0 for name in FORWARD_ENTRYPOINTS}

    def native_forward(phase):
        current_phase["value"] = phase
        codes = a_codes
        if padded_m != true_m:
            codes = functional.pad(codes, (0, 0, 0, padded_m - true_m), value=0)
        packed_activation = extension.pack_activation(codes.contiguous())
        call_count["pack_activation"] += 1

        a_scales = a_scale_half.reshape(1, true_m)
        if padded_m != true_m:
            a_scales = functional.pad(a_scales, (0, padded_m - true_m), value=1)
        a_scales = a_scales.expand(K // 64, padded_m).contiguous()
        packed_a_scales = extension.pack_activation_scales(a_scales)
        call_count["pack_activation_scales"] += 1

        lowrank_input = sidecar_input
        if state["sidecar_k"] > K:
            lowrank_input = functional.pad(lowrank_input,
                                           (0, state["sidecar_k"] - K), value=0)
        lowrank_input = lowrank_input.contiguous()
        lora_act = extension.quantize_down(lowrank_input, state["packed_down"])
        call_count["quantize_down"] += 1
        if tuple(lora_act.shape) != (padded_m, RANK):
            raise RuntimeError("Official Nunchaku down helper returned the wrong [padded M, rank] activation")

        output_shape = native_diffusion._nunchaku_output_shape(
            true_m, padded_m, padded_n, true_n)
        output = torch.empty(output_shape, device=device, dtype=torch.float16)
        extension.matmul_lora_up(
            packed_activation, core["packed_weight"], output,
            packed_a_scales, core["packed_wscales"], lora_act, state["packed_up"])
        call_count["matmul_lora_up"] += 1
        return output

    def trim_and_check_padding(output, label: str):
        if padded_n > true_n and bool(torch.count_nonzero(output[:, true_n:]).item()):
            raise AssertionError(f"{label}: padded N output should remain zero for zero-padded weight/LoRA rows")
        return output[:, :true_n].contiguous()

    default = torch.cuda.default_stream(device)
    nondefault = torch.cuda.Stream(device=device)
    nondefault.wait_stream(default)
    for label, stream in (("default_current_stream", default),
                          ("non_default_current_stream", nondefault)):
        before_calls = dict(call_count)
        with torch.cuda.stream(stream):
            current_phase["value"] = label
            native_output = native_forward(label)
        stream.synchronize()
        output = trim_and_check_padding(native_output, label)
        phase_calls = {name: call_count[name] - before_calls[name] for name in call_count}
        phases[label] = {"passed": True, "metrics": _metrics(output, expected, label, torch),
                         "output_shape": list(output.shape),
                         "native_output_shape": list(native_output.shape),
                         "stream_handle": int(stream.cuda_stream),
                         "extension_entrypoint_calls": phase_calls}
        del output, native_output

    graph_stream = torch.cuda.Stream(device=device)
    graph_stream.wait_stream(default)
    graph = torch.cuda.CUDAGraph()
    before_capture_calls = dict(call_count)
    current_phase["value"] = "cuda_graph_capture"
    progress(current_case["value"], "cuda_graph_capture", "CUDAGraph.capture", "entered")
    with torch.cuda.graph(graph, stream=graph_stream):
        graph_native_output = native_forward("cuda_graph_capture")
    progress(current_case["value"], "cuda_graph_capture", "CUDAGraph.capture", "recorded")
    capture_calls = {name: call_count[name] - before_capture_calls[name] for name in call_count}
    current_phase["value"] = "cuda_graph_replay"
    progress(current_case["value"], "cuda_graph_replay", "CUDAGraph.replay",
             f"entering_{NATIVE_EXECUTIONS['cuda_graph_replay']}_replays")
    for _ in range(NATIVE_EXECUTIONS["cuda_graph_replay"]):
        graph.replay()
    try:
        graph_stream.synchronize()
    except Exception as exc:
        progress(current_case["value"], "cuda_graph_replay", "CUDAGraph.replay", "sync_error")
        raise RuntimeError(f"CUDA error after fused graph replay for M={true_m}") from exc
    progress(current_case["value"], "cuda_graph_replay", "CUDAGraph.replay",
             "stream_synchronized")
    graph_output = trim_and_check_padding(graph_native_output, "CUDA graph replay")
    phases["cuda_graph"] = {
        "capture_passed": True,
        "replay_count": NATIVE_EXECUTIONS["cuda_graph_replay"],
        "replay_passed": True,
        "metrics": _metrics(graph_output, expected, "CUDA graph replay", torch),
        "output_shape": list(graph_output.shape),
        "native_output_shape": list(graph_native_output.shape),
        "stream_handle": int(graph_stream.cuda_stream),
        "capture_extension_entrypoint_calls": capture_calls,
        "replay_extension_entrypoint_calls": {
            name: NATIVE_EXECUTIONS["cuda_graph_replay"] for name in FORWARD_ENTRYPOINTS},
        "padded_n_tail_is_zero": True,
    }
    native_executions = sum(NATIVE_EXECUTIONS.values())
    python_calls = 2 + CAPTURE_ENTRYPOINT_CALLS
    executions_by_entrypoint = {name: native_executions for name in FORWARD_ENTRYPOINTS}
    python_calls_by_entrypoint = {name: python_calls for name in FORWARD_ENTRYPOINTS}
    return {
        **spec,
        "weight_scale_semantics": "two distinct frozen FP16 G128 scale groups; adapter duplicates each scale to its two G64 kernel groups",
        "activation_semantics": "whole-K per-token symmetric A4, qmax=7; one row scale repeated over all G64 groups",
        "lora_semantics": "FP16 normalized input and frozen BF16 factors converted to FP16; FP32 down intermediate; official up epilogue scale=1.0",
        "oracle": "FP32 grouped integer GEMM using actual A4/W4 codes and stored FP16 scales, plus FP32 LoRA products from the exact FP16 operands; TF32 disabled; compare before BF16 reference rounding",
        "tolerances": {"max_abs_atol": FP16_MAX_ABS_ATOL, "nrmse_max": NRMSE_LIMIT},
        "stream_tests": phases,
        "extension_entrypoint_calls_per_execution": {name: 1 for name in FORWARD_ENTRYPOINTS},
        "native_kernel_execution_counts": executions_by_entrypoint,
        "native_kernel_executions_per_case": native_executions,
        "extension_entrypoint_python_call_counts": python_calls_by_entrypoint,
        "capture_entrypoint_calls_per_kernel": CAPTURE_ENTRYPOINT_CALLS,
        "policy_forward_calls": 0,
        "training_steps": 0,
        "closed_loop_episodes": 0,
    }


def _run(root: Path, out: Path) -> dict:
    # This guard checks PBS_JOBID, PBS_NODEFILE membership, and rejects login hosts.
    guard()
    import torch
    import torch.nn.functional as functional

    receipt, arm = _load_build_receipt(root)
    extension = _load_extension(Path(arm["module"]).resolve())

    if not torch.cuda.is_available():
        raise RuntimeError("Fused SVDQuant probe requires the approved PBS GPU allocation")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    hardware = _hardware(torch, device)
    import native_diffusion
    import native_svd_fused

    partial_path = _partial_output_path(out)
    current_case = {"value": None}
    current_phase = {"value": "setup"}
    partial_identity = {
        "format": "cosmos_svd_fused_probe_partial_v1",
        "pbs_job_id": os.environ["PBS_JOBID"],
        "probe_host": socket.gethostname(),
        "build_pbs_job_id": receipt["pbs_job_id"],
        "build_host": receipt.get("host"),
        "source_commit": NUNCHAKU_COMMIT,
        "module": arm["module"],
        "hardware": hardware,
        "model_loaded": False,
        "policy_forward_calls": 0,
        "training_steps": 0,
        "closed_loop_episodes": 0,
    }
    progress = _progress_writer(partial_path, partial_identity)
    progress(None, "setup", "probe_start", "entered")
    extension = _SynchronizedProbeExtension(
        extension, torch, progress,
        lambda: current_case["value"], lambda: current_phase["value"])
    state = _make_bank_and_layouts(torch, functional, extension, device)
    setup_calls = {"pack_weight": 1, "pack_weight_scales": 1}
    cases = []
    for m in M_CASES:
        spec = _case_spec(m)
        current_case["value"] = {"true_m": m, "true_n": spec["true_n"],
                                  "padded_m": spec["padded_m"],
                                  "padded_n": spec["padded_n"]}
        cases.append(_probe_case(spec, state, extension, torch, functional, device,
                                 progress, current_case, current_phase))
    aggregate_entrypoints = {name: 0 for name in (*setup_calls, *FORWARD_ENTRYPOINTS)}
    for name, count in setup_calls.items():
        aggregate_entrypoints[name] += count
    for case in cases:
        for name, count in case["native_kernel_execution_counts"].items():
            aggregate_entrypoints[name] += count
    aggregate_python_calls = dict(setup_calls)
    for name in FORWARD_ENTRYPOINTS:
        aggregate_python_calls[name] = sum(
            case["extension_entrypoint_python_call_counts"][name] for case in cases)
    result = {
        "status": "complete",
        "probe": "nunchaku_official_svdquant_rank16_lora_fusion",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "producer": {"probe_pbs_job_id": os.environ["PBS_JOBID"],
                     "probe_host": socket.gethostname(),
                     "build_pbs_job_id": receipt["pbs_job_id"],
                     "build_host": receipt.get("host"),
                     "build_receipt": str((root / "results" / "preparation" /
                                            "native_svd_fused_build.json").resolve())},
        "source": {"nunchaku_commit": NUNCHAKU_COMMIT,
                   "deepcompressor_packing_commit": native_svd_fused.DEEPCOMPRESSOR_COMMIT,
                   "module": arm["module"],
                   "module_name": MODULE_NAME},
        "hardware": hardware,
        "native_kernel_calls": {
            "setup_weight_layout": setup_calls,
            "per_pipeline_execution": {name: 1 for name in FORWARD_ENTRYPOINTS},
            "native_kernel_executions_per_case": NATIVE_EXECUTIONS,
            "cuda_graph_capture_entrypoint_calls_per_kernel": CAPTURE_ENTRYPOINT_CALLS,
            "aggregate_native_kernel_executions": aggregate_entrypoints,
            "aggregate_extension_entrypoint_python_calls": aggregate_python_calls,
            "call_count_scope": "Counts native extension kernel executions; capture records graph nodes without executing them, and graph replays execute the captured nodes",
        },
        "oracle_and_gate": {
            "reference": "FP32 accumulation of groupwise integer GEMM plus FP32 LoRA down/up using FP16 operands; no decoded-BF16 reference rounding",
            "tf32_enabled": False,
            "nrmse_max": NRMSE_LIMIT,
            "fp16_max_abs_atol": FP16_MAX_ABS_ATOL,
            "thresholds_automatically_relaxed": False,
        },
        "diagnostic_synchronization": {
            "enabled": True,
            "scope": "current-stream synchronization after each extension entrypoint in ordinary probe phases; skipped during graph capture and synchronized after replay",
            "partial_receipt": str(partial_path.resolve()),
        },
        "cases": cases,
        "policy_forward_calls": 0,
        "training_steps": 0,
        "closed_loop_episodes": 0,
        "model_loaded": False,
    }
    path = _output_path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if len(encoded.encode("utf-8")) > 256 * 1024:
        raise RuntimeError("Fused SVD probe JSON exceeds the 256 KiB evidence limit")
    path.write_text(encoded, encoding="utf-8")
    partial_path.unlink(missing_ok=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True,
                        help="prepared Cosmos native performance experiment root")
    parser.add_argument("--out", type=Path, required=True,
                        help="JSON output path or directory")
    args = parser.parse_args()
    result = _run(args.root.resolve(), args.out)
    print(json.dumps({"status": result["status"], "out": str(_output_path(args.out).resolve()),
                      "cases": len(result["cases"]), "gpu": result["hardware"]["gpu_name"]}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
