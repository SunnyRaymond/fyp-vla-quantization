"""Compute-allocation-only correctness, stream, graph, and timing probe for native GEMMs."""
from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import socket
from types import SimpleNamespace
from datetime import datetime, timezone

from prepare_sources import guard

QUAROT_SHAPES = ((1, 128), (1, 2048), (1, 8192),
                 (32, 128), (32, 2048), (32, 8192),
                 (128, 128), (128, 2048), (128, 8192),
                 (1764, 128), (1764, 2048), (1764, 8192))
QUAROT_K = 128
SEED = 777
TIMING_REPEATS = 30
FUSED_SCALED_ADD_CASES = ((1, 128, 3, (0, 2)),
                          (17, 257, 5, (0, 2, 4)))
LOW_PRECISION_GEMM_ATOL = 0.125
LOW_PRECISION_GEMM_RTOL = 0.002


def _output_path(path: Path) -> Path:
    if path.suffix.lower() == ".json":
        return path
    return path / "kernel_probe.json"


def _source_commit(root: Path) -> str:
    marker = root / "vendor" / "quarot" / "UPSTREAM_COMMIT"
    if marker.is_file():
        return marker.read_text(encoding="utf-8").strip()
    for name in ("quarot_fused_build.json", "quarot_build.json"):
        receipt = root / "results" / "preparation" / name
        if receipt.is_file():
            value = json.loads(receipt.read_text(encoding="utf-8")).get("source_commit")
            if isinstance(value, str) and value:
                return value
    raise FileNotFoundError("QuaRot source/build receipt is required to identify the probed kernel revision")


def _optional_source_commit(root: Path, vendor_name: str, arm: str):
    marker = root / "vendor" / vendor_name / "UPSTREAM_COMMIT"
    if marker.is_file():
        return marker.read_text(encoding="utf-8").strip()
    receipt = root / "results" / "preparation" / "native_diffusion_build.json"
    if receipt.is_file():
        builds = json.loads(receipt.read_text(encoding="utf-8")).get("arms", {})
        value = builds.get(arm, {}).get("source_commit")
        if isinstance(value, str) and value:
            return value
    return None


def _extension(module_name: str, directory: Path):
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    binaries = [path for path in directory.glob(module_name + "*")
                if path.is_file() and path.name.endswith(suffixes)] if directory.is_dir() else []
    if not binaries:
        raise FileNotFoundError(f"Native extension {module_name} was not built in {directory}")
    sys.path.insert(0, str(directory.resolve()))
    module = importlib.import_module(module_name)
    resolved = Path(module.__file__).resolve()
    if not resolved.is_relative_to(directory.resolve()):
        raise RuntimeError(f"Imported {module_name} from outside its build directory: {resolved}")
    return module


def _select_device_uuid(rows, visible_devices: str | None, device_index: int) -> str:
    parsed = []
    for line in rows.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) == 2 and parts[1]:
            parsed.append((parts[0], parts[1]))
    if not parsed:
        raise RuntimeError("nvidia-smi returned no physical GPU UUID records")
    if visible_devices:
        tokens = [token.strip() for token in visible_devices.split(",") if token.strip()]
        if device_index >= len(tokens):
            raise RuntimeError("CUDA_VISIBLE_DEVICES cannot be mapped to the active CUDA device")
        token = tokens[device_index]
        if token.startswith(("GPU-", "MIG-")):
            match = next((uuid for _, uuid in parsed if uuid == token), None)
            if match is None:
                raise RuntimeError("Visible GPU UUID was not present in nvidia-smi output")
            return match
        if token.isdigit():
            match = next((uuid for index, uuid in parsed if index == token), None)
            if match is not None:
                return match
    if device_index < len(parsed):
        return parsed[device_index][1]
    raise RuntimeError("Could not map the active CUDA device to an nvidia-smi GPU UUID")


def _hardware(torch, device) -> dict:
    props = torch.cuda.get_device_properties(device)
    major, minor = torch.cuda.get_device_capability(device)
    uuid = getattr(props, "uuid", None)
    if uuid is not None:
        uuid = str(uuid)
    if not uuid:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        )
        index = torch.cuda.current_device()
        uuid = _select_device_uuid(result.stdout, os.environ.get("CUDA_VISIBLE_DEVICES"), index)
    if (major, minor) != (8, 0) or "A100" not in props.name.upper():
        raise RuntimeError(f"Probe requires an A100 SM80 allocation; found {props.name} SM{major}{minor}")
    return {
        "gpu_uuid": uuid,
        "gpu_name": props.name,
        "total_memory_bytes": int(props.total_memory),
        "capability": [int(major), int(minor)],
        "device_index": int(torch.cuda.current_device()),
    }


def _pack_signed_i4(codes, torch):
    if bool(torch.any((codes < -8) | (codes > 7)).item()):
        raise ValueError("INT4 source code out of range")
    encoded = codes.to(torch.int16).bitwise_and(15).to(torch.uint8)
    return (encoded[..., 0::2] | (encoded[..., 1::2] << 4)).contiguous()


def _unpack_signed_i4(packed, torch):
    values = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(packed.shape[0], -1).to(torch.int16)
    return torch.where(values >= 8, values - 16, values).to(torch.int8)


def _reference_dot(a_codes, b_codes, torch):
    # The values and K=128 bound make every result exactly representable in FP32.
    return torch.matmul(a_codes.to(torch.float32), b_codes.to(torch.float32).transpose(0, 1))


def _reference_scaled_groups(a_codes, b_codes, a_scales, b_scales, group_width: int, torch):
    groups = a_codes.shape[1] // group_width
    result = torch.zeros((a_codes.shape[0], b_codes.shape[0]),
                         device=a_codes.device, dtype=torch.float32)
    for group in range(groups):
        start = group * group_width
        stop = start + group_width
        partial = torch.matmul(a_codes[:, start:stop].float(),
                               b_codes[:, start:stop].float().transpose(0, 1))
        result.add_(partial * a_scales[group].float()[:, None]
                    * b_scales[group].float()[None, :])
    return result


def _reference_scaled_add(output, accum, a_scales, w_scales, group: int, torch):
    # Match the native sequence: FP32 factor multiply, FP32 accumulator multiply, FP32 add.
    factor = a_scales.to(torch.float32)[:, None] * w_scales[:, group].to(torch.float32)[None, :]
    scaled_accum = accum.to(torch.float32) * factor
    return output + scaled_accum


def _assert_close_low_precision(output, expected, label: str, torch) -> float:
    if not torch.is_tensor(output) or output.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(f"{label}: kernel must return a low-precision tensor")
    if tuple(output.shape) != tuple(expected.shape) or output.device != expected.device:
        raise RuntimeError(f"{label}: native result shape/device does not match its reference")
    error = (output.to(torch.float32) - expected.to(torch.float32)).abs()
    limit = LOW_PRECISION_GEMM_ATOL + expected.to(torch.float32).abs() * LOW_PRECISION_GEMM_RTOL
    max_error = float(error.max().item())
    if bool(torch.any(error > limit).item()):
        raise AssertionError(f"{label}: low-precision GEMM differs from FP32 group reference, max_abs_error={max_error}")
    return max_error


def _assert_exact(output, expected, label: str, torch) -> float:
    if not torch.is_tensor(output) or output.dtype != torch.int32:
        raise RuntimeError(f"{label}: kernel must return int32, got {getattr(output, 'dtype', None)}")
    if tuple(output.shape) != tuple(expected.shape) or output.device != expected.device:
        raise RuntimeError(f"{label}: native result shape/device does not match its reference")
    max_error = float((output.to(torch.float32) - expected).abs().max().item())
    if max_error != 0.0:
        raise AssertionError(f"{label}: exact integer probe failed, max_abs_error={max_error}")
    return max_error


def _time_cuda_events(function, stream, torch, repeats=TIMING_REPEATS) -> dict:
    with torch.cuda.stream(stream):
        for _ in range(5):
            last = function()
    stream.synchronize()
    pairs = []
    with torch.cuda.stream(stream):
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            last = function()
            end.record(stream)
            pairs.append((start, end))
    stream.synchronize()
    samples = [float(start.elapsed_time(end)) for start, end in pairs]
    if not samples or any(value < 0 for value in samples):
        raise RuntimeError("CUDA event timing did not produce valid samples")
    ordered = sorted(samples)
    p90 = ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
    return {
        "repeats": repeats,
        "median_ms": float(statistics.median(samples)),
        "mean_ms": float(statistics.fmean(samples)),
        "p90_ms": float(p90),
        "min_ms": float(ordered[0]),
        "max_ms": float(ordered[-1]),
        "timing_scope": "CUDA events around one extension kernel call on the selected stream",
    }


def _probe_quarot_case(backend, torch, device, m: int, n: int) -> dict:
    k = QUAROT_K
    a_codes = torch.randint(-7, 8, (m, k), device=device, dtype=torch.int16).to(torch.int8)
    b_codes = torch.randint(-7, 8, (n, k), device=device, dtype=torch.int16).to(torch.int8)
    a = _pack_signed_i4(a_codes, torch)
    b = _pack_signed_i4(b_codes, torch)
    expected = _reference_dot(a_codes, b_codes, torch)
    default = torch.cuda.default_stream(device)
    side = torch.cuda.Stream(device=device)
    side.wait_stream(default)

    stream_results = {}
    for stream_name, stream in (("default_current_stream", default),
                                ("non_default_current_stream", side)):
        with torch.cuda.stream(stream):
            output = backend.matmul(a, b)
        stream.synchronize()
        error = _assert_exact(output, expected, f"QuaRot {m}x{n} {stream_name}", torch)
        stream_results[stream_name] = {
            "passed": True,
            "max_abs_error": error,
            "output_dtype": str(output.dtype),
            "output_shape": list(output.shape),
            "stream_handle": int(stream.cuda_stream),
            "correctness_matmul_calls": 1,
            "event_timing": _time_cuda_events(lambda: backend.matmul(a, b), stream, torch),
        }
        del output

    graph_stream = torch.cuda.Stream(device=device)
    graph_stream.wait_stream(default)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=graph_stream):
        graph_output = backend.matmul(a, b)
    with torch.cuda.stream(graph_stream):
        for _ in range(3):
            graph.replay()
    graph_stream.synchronize()
    graph_error = _assert_exact(graph_output, expected, f"QuaRot {m}x{n} CUDA graph replay", torch)
    execution_calls = (sum(row["event_timing"]["repeats"] + 5 + row["correctness_matmul_calls"]
                           for row in stream_results.values()) + 1 + 3)

    return {
        "M": m, "N": n, "K": k,
        "A_packed_shape": list(a.shape), "B_packed_shape": list(b.shape),
        "reference": "FP32 matmul of decoded signed INT4 values; TF32 disabled; exact integer dot",
        "output_exact_within_int32": True,
        "stream_tests": stream_results,
        "cuda_graph": {"capture_passed": True, "replay_count": 3,
                       "replay_passed": True, "max_abs_error": graph_error,
                       "capture_matmul_calls": 1, "graph_replay_matmul_calls": 3,
                       "stream_handle": int(graph_stream.cuda_stream)},
        "native_matmul_calls": execution_calls,
    }


def _probe_fused_quarot_scaled_add(torch, device, root: Path) -> dict:
    module_name = "quarot_official_gemm_fused"
    directory = root / "native_extensions" / module_name
    if not any(path.is_file() and path.name.startswith(module_name)
               for path in directory.glob(module_name + "*") if directory.is_dir()):
        return {"status": "pending_extension_not_built", "extension_dir": str(directory)}
    extension = _extension(module_name, directory)
    if not callable(getattr(extension, "scaled_add", None)):
        raise RuntimeError("Fused QuaRot extension must expose scaled_add(output, accum, a_scales, w_scales, group)")

    default = torch.cuda.default_stream(device)
    side = torch.cuda.Stream(device=device)
    side.wait_stream(default)
    scale_dtypes = (("float16", torch.float16), ("bfloat16", torch.bfloat16))
    dtype_results = []
    native_calls = 0
    for dtype_name, dtype in scale_dtypes:
        case_results = []
        for m, n, groups, tested_groups in FUSED_SCALED_ADD_CASES:
            accum = torch.randint(-4096, 4097, (m, n), device=device, dtype=torch.int32)
            a_scales = torch.rand((m,), device=device, dtype=torch.float32).mul_(0.9).add_(0.05)
            w_scales = torch.rand((n, groups), device=device, dtype=torch.float32).mul_(0.9).add_(0.05)
            w_scales = w_scales.to(dtype).contiguous()
            initial = torch.randn((m, n), device=device, dtype=torch.float32).mul_(2.0)
            group_results = []
            for group in tested_groups:
                expected = _reference_scaled_add(initial, accum, a_scales, w_scales,
                                                 group, torch)
                default_output = initial.clone()
                with torch.cuda.stream(default):
                    extension.scaled_add(default_output, accum, a_scales, w_scales, group)
                default.synchronize()
                if not torch.equal(default_output, expected):
                    max_error = float((default_output - expected).abs().max().item())
                    raise AssertionError(
                        f"Fused QuaRot scaled_add {dtype_name} group={group} default stream "
                        f"violated FP32 mul+add order: max_abs_error={max_error}")

                side_output = initial.clone()
                side.wait_stream(default)
                with torch.cuda.stream(side):
                    extension.scaled_add(side_output, accum, a_scales, w_scales, group)
                side.synchronize()
                if not torch.equal(side_output, expected):
                    max_error = float((side_output - expected).abs().max().item())
                    raise AssertionError(
                        f"Fused QuaRot scaled_add {dtype_name} group={group} non-default stream "
                        f"violated FP32 mul+add order: max_abs_error={max_error}")

                graph_stream = torch.cuda.Stream(device=device)
                graph_stream.wait_stream(default)
                graph = torch.cuda.CUDAGraph()
                graph_output = torch.empty_like(initial)
                with torch.cuda.graph(graph, stream=graph_stream):
                    graph_output.copy_(initial)
                    extension.scaled_add(graph_output, accum, a_scales, w_scales, group)
                with torch.cuda.stream(graph_stream):
                    for _ in range(3):
                        graph.replay()
                graph_stream.synchronize()
                if not torch.equal(graph_output, expected):
                    max_error = float((graph_output - expected).abs().max().item())
                    raise AssertionError(
                        f"Fused QuaRot scaled_add {dtype_name} group={group} CUDA graph "
                        f"violated FP32 mul+add order: max_abs_error={max_error}")

                timing = None
                if group == tested_groups[0]:
                    timing_output = initial.clone()
                    timing = _time_cuda_events(
                        lambda: extension.scaled_add(timing_output, accum, a_scales,
                                                     w_scales, group), default, torch)
                    native_calls += timing["repeats"] + 5
                native_calls += 6
                group_results.append({
                    "group": group,
                    "default_current_stream": {"passed": True, "max_abs_error": 0.0},
                    "non_default_current_stream": {"passed": True, "max_abs_error": 0.0,
                                                    "stream_handle": int(side.cuda_stream)},
                    "cuda_graph": {"capture_passed": True, "replay_count": 3,
                                   "replay_passed": True, "max_abs_error": 0.0,
                                   "stream_handle": int(graph_stream.cuda_stream)},
                    "event_timing": timing,
                    "native_scaled_add_calls": 6 + (0 if timing is None else timing["repeats"] + 5),
                })
            case_results.append({"M": m, "N": n, "groups": groups,
                                 "tested_groups": list(tested_groups),
                                 "group_results": group_results})
        dtype_results.append({"weight_scale_dtype": "torch." + dtype_name,
                              "cases": case_results})

    return {
        "status": "complete",
        "source_commit": _source_commit(root),
        "extension": str(Path(extension.__file__).resolve()),
        "extension_api": "scaled_add(float32[M,N] output, int32[M,N] accum, float32[M] A scales, [N,G] original-dtype W scales, group)",
        "weight_scale_dtypes": ["torch.float16", "torch.bfloat16"],
        "reference": "FP32 factor=ascale*float(wscale[group]); scaled=accum*factor; output=output+scaled, in separate round-to-nearest operations",
        "exact_fp32_order_required": True,
        "stream_and_graph_tests": ["default current stream", "non-default current stream", "CUDA graph capture and 3 replays"],
        "native_scaled_add_calls": native_calls,
        "cases": dtype_results,
    }


def _pack_optional_i4(codes, torch):
    values = codes.to(torch.int16).bitwise_and(15).to(torch.uint8)
    return (values[:, 0::2] | (values[:, 1::2] << 4)).contiguous()


def _pack_offset8_i4(codes, torch):
    values = (codes.to(torch.int16) + 8).to(torch.uint8)
    return (values[:, 0::2] | (values[:, 1::2] << 4)).contiguous()


def _probe_nunchaku(torch, device, root: Path) -> dict:
    module_name = "nunchaku_svdq_native"
    directory = root / "native_extensions" / "svdquant_w4a4"
    if not any(path.is_file() and path.name.startswith(module_name)
               for path in directory.glob(module_name + "*") if directory.is_dir()):
        return {"status": "pending_extension_not_built", "extension_dir": str(directory)}
    extension = _extension(module_name, directory)
    from native_diffusion import _nunchaku_bank, _nunchaku_output_shape
    import torch.nn.functional as functional

    batch, n, k = 129, 131, 128
    padded_m = ((batch + 255) // 256) * 256
    a_codes = torch.randint(-7, 8, (batch, k), device=device, dtype=torch.int16).to(torch.int8)
    b_codes = torch.randint(-7, 8, (n, k), device=device, dtype=torch.int16).to(torch.int8)
    checkpoint_w_scales = (torch.rand((n, k // 128), device=device, dtype=torch.float32)
                           .mul_(0.75).add_(0.125).to(torch.float16))
    bank = {"encoded": {"codes": _pack_offset8_i4(b_codes, torch),
                        "scales": checkpoint_w_scales, "shape": (n, k), "bits": 4},
            "smoothing": torch.ones(k, device=device, dtype=torch.float16),
            "up": None, "down": None}
    module = SimpleNamespace(weight=SimpleNamespace(device=device))
    item = _nunchaku_bank(bank, module, extension, torch, functional, rank=16)
    padded_n = item["padded_n"]
    activation_scales = (torch.rand((batch,), device=device, dtype=torch.float32)
                         .mul_(0.75).add_(0.125).to(torch.float16))
    padded_activation_scales = functional.pad(activation_scales, (0, padded_m - batch), value=1)
    packed_a_scales = padded_activation_scales.reshape(1, padded_m).expand(k // 64, padded_m).contiguous()
    padded_a_codes = functional.pad(a_codes, (0, 0, 0, padded_m - batch), value=0)
    packed_a = extension.pack_activation(padded_a_codes.contiguous())
    packed_as = extension.pack_activation_scales(packed_a_scales)
    packed_b, packed_bs = item["packed_weight"], item["packed_wscales"]
    output_shape = _nunchaku_output_shape(batch, padded_m, padded_n, n)
    reference_b = functional.pad(b_codes, (0, 0, 0, padded_n - n), value=0)
    reference_w_scales = checkpoint_w_scales.transpose(0, 1).contiguous().repeat_interleave(2, dim=0)
    reference_w_scales = functional.pad(reference_w_scales, (0, padded_n - n), value=1)
    reference_a_scales = activation_scales.reshape(1, batch).expand(k // 64, batch)
    expected = _reference_scaled_groups(a_codes, reference_b, reference_a_scales,
                                       reference_w_scales, 64, torch)

    def run():
        out = torch.empty(output_shape, device=device, dtype=torch.float16)
        extension.matmul(packed_a, packed_b, out, packed_as, packed_bs)
        return out

    output = run()
    torch.cuda.current_stream(device).synchronize()
    error = _assert_close_low_precision(output, expected,
                                       "Nunchaku adapter-packed W4A4", torch)
    timing = _time_cuda_events(run, torch.cuda.current_stream(device), torch)
    return {"status": "complete", "source_commit": _optional_source_commit(root, "nunchaku", "svdquant_w4a4"),
            "extension": str(Path(extension.__file__).resolve()),
            "adapter_pack_path": "native_diffusion._nunchaku_bank + official pack_activation/pack_activation_scales",
            "checkpoint_group_semantics": "G128 weight scale repeated into two G64 Nunchaku groups; one per-row A scale repeated across G64 groups",
            "M": batch, "padded_M": padded_m, "N": n, "padded_N": padded_n, "K": k,
            "max_abs_error": error,
            "comparison_tolerance": {"atol": LOW_PRECISION_GEMM_ATOL, "rtol": LOW_PRECISION_GEMM_RTOL},
            "native_matmul_calls": 1 + 5 + TIMING_REPEATS,
            "correctness_matmul_calls": 1,
            "timing_matmul_calls": 5 + TIMING_REPEATS,
            "packed_gpu_resident": True,
            "event_timing": timing,
            "reference": "FP32 signed INT4 G64 group dot with adapter-preserved FP16 scales; low-precision GEMM output tolerance"}


def _probe_viditq(torch, device, root: Path) -> dict:
    module_name = "viditq_atom_per_channel_native"
    directory = root / "native_extensions" / "viditq_w4a4_general"
    if not any(path.is_file() and path.name.startswith(module_name)
               for path in directory.glob(module_name + "*") if directory.is_dir()):
        return {"status": "pending_extension_not_built", "extension_dir": str(directory)}
    from native_diffusion import _extension as diffusion_extension
    from native_diffusion import _viditq_bank, _viditq_activation_scales
    extension = diffusion_extension('viditq_w4a4', root)
    import torch.nn.functional as functional

    batch, n, k = 129, 131, 256
    padded_m = max(128, ((batch + 127) // 128) * 128)
    a_codes = torch.randint(-7, 8, (batch, k), device=device, dtype=torch.int16).to(torch.int8)
    b_codes = torch.randint(-7, 8, (n, k), device=device, dtype=torch.int16).to(torch.int8)
    groups = k // 128
    checkpoint_w_scales = (torch.rand((n, groups), device=device, dtype=torch.float32)
                           .mul_(0.75).add_(0.125).to(torch.float16))
    bank = {"encoded": {"codes": _pack_offset8_i4(b_codes, torch),
                        "scales": checkpoint_w_scales,
                        "zero_points": torch.full((n, groups), 8, device=device, dtype=torch.int16),
                        "shape": (n, k), "bits": 4},
            "smoothing": torch.ones(k, device=device, dtype=torch.float16),
            "signs": None}
    item = _viditq_bank(bank, SimpleNamespace(weight=SimpleNamespace(device=device)),
                        extension, torch, functional)
    packed_b, b_scales, w_offset, _multiplier = item["passes"][0]
    padded_n = item["padded_n"]
    activation_scales = (torch.rand((batch,), device=device, dtype=torch.float32)
                         .mul_(0.75).add_(0.125).to(torch.float16))
    padded_activation_scales = functional.pad(activation_scales, (0, padded_m - batch), value=1)
    scale_rows = padded_activation_scales.reshape(1, padded_m).expand(groups, padded_m)
    a_scales = _viditq_activation_scales(scale_rows, torch)
    padded_a_codes = functional.pad(a_codes, (0, 0, 0, padded_m - batch), value=0)
    packed_a = _pack_optional_i4(padded_a_codes, torch)
    a_keeper = torch.zeros((padded_m, 128), device=device, dtype=torch.int8)
    b_keeper = torch.zeros((128, padded_n), device=device, dtype=torch.int8)
    a_keeper_scales = torch.ones((4 * padded_m,), device=device, dtype=torch.float16)
    b_keeper_scales = torch.ones((padded_n,), device=device, dtype=torch.float16)
    output = torch.empty((padded_m, padded_n), device=device, dtype=torch.float16)
    reference_b = functional.pad(b_codes, (0, 0, 0, padded_n - n), value=0)
    expected = _reference_scaled_groups(padded_a_codes, reference_b,
                                       scale_rows, b_scales, 128, torch)
    if bool(torch.any(w_offset != 0).item()):
        raise RuntimeError("ViDiT-Q zero-point-8 adapter probe must produce zero GEMM offsets")

    def run():
        extension.matmul(packed_a, packed_b, output, a_scales, b_scales,
                         a_keeper, b_keeper, a_keeper_scales, b_keeper_scales)
        return output

    run()
    torch.cuda.current_stream(device).synchronize()
    error = _assert_close_low_precision(output, expected,
                                       "ViDiT-Q adapter-packed Atom W4A4", torch)
    timing = _time_cuda_events(run, torch.cuda.current_stream(device), torch)
    return {"status": "complete", "source_commit": _optional_source_commit(root, "viditq", "viditq_w4a4"),
            "extension": str(Path(extension.__file__).resolve()),
            "adapter_pack_path": "native_diffusion._viditq_bank + official Atom INT4 MMA with task-local independent-column scale epilogue",
            "checkpoint_group_semantics": "W4 G128 checkpoint scales and zero points; A FP16 rows reordered by the official Atom lane layout then replicated fourfold per group",
            "M": batch, "padded_M": padded_m, "N": n, "padded_N": padded_n, "K": k,
            "weight_zero_point": 8, "weight_scale_dtype": "torch.float16",
            "max_abs_error": error,
            "comparison_tolerance": {"atol": LOW_PRECISION_GEMM_ATOL, "rtol": LOW_PRECISION_GEMM_RTOL},
            "native_matmul_calls": 1 + 5 + TIMING_REPEATS,
            "correctness_matmul_calls": 1,
            "timing_matmul_calls": 5 + TIMING_REPEATS,
            "packed_gpu_resident": True,
            "event_timing": timing,
            "reference": "FP32 signed INT4 G128 group dot with adapter-layout scales, zero offset, and low-precision output tolerance"}


def _probe_hadamard(torch, device, root):
    receipt_path = root / 'results/preparation/quarot_hadamard_build.json'
    if not receipt_path.is_file():
        return {'status': 'pending', 'reason': 'Official FHT build receipt absent'}
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('status') != 'complete':
        raise RuntimeError('Official FHT build receipt is incomplete')
    directory = root / 'native_extensions/quarot_official_gemm_fused'
    extension = _extension('quarot_native_hadamard_cuda', directory)
    from native_rotations import _validate_hadamard_build
    _validate_hadamard_build(extension, receipt, directory)
    rows = []
    scale = 1 / math.sqrt(128)
    for m, integer in ((1, True), (17, False), (112896, True)):
        value = (torch.randint(-7, 8, (m, 128), device=device).float() if integer
                 else torch.randn((m, 128), device=device, dtype=torch.float32))
        expected = value
        for half in (1, 2, 4, 8, 16, 32, 64):
            block = expected.reshape(m, 128 // (2*half), 2, half)
            left, right = block[..., 0, :], block[..., 1, :]
            expected = torch.stack((left+right, left-right), -2).reshape(m, 128)
        expected = expected * scale

        def run():
            return extension.fast_hadamard_transform(value, scale)

        def check(actual):
            if actual.dtype != torch.float32 or actual.shape != value.shape:
                raise RuntimeError('Native H128 changed shape/dtype')
            error = float((actual-expected).abs().max())
            if integer:
                if error != 0:
                    raise RuntimeError(f'Integer FP32 H128 must match the exact butterfly: {error}')
            else:
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
            return error

        error = check(run())
        side = torch.cuda.Stream(device=device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            side_output = run()
        torch.cuda.current_stream(device).wait_stream(side)
        error = max(error, check(side_output))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = run()
        for _ in range(3):
            graph.replay()
            error = max(error, check(graph_output))
        timing = _time_cuda_events(run, torch.cuda.current_stream(device), torch)
        rows.append({'M': m, 'K': 128, 'dtype': 'torch.float32', 'integer_input': integer,
                     'max_abs_error': error, 'default_stream': True, 'nondefault_stream': True,
                     'graph_replays': 3, 'event_timing': timing})
    return {'status': 'complete', 'module': str(Path(extension.__file__).resolve()),
            'build_pbs_job_id': receipt['pbs_job_id'], 'source_commit': receipt['source_commit'],
            'scale': scale, 'cases': rows}


def require_all_native(receipt):
    required = {'fused scaled_add': receipt.get('fused_quarot_scaled_add', {}),
                'official FHT': receipt.get('official_hadamard', {}),
                **receipt.get('diffusion_optional_probes', {})}
    if set(required) != {'fused scaled_add', 'official FHT', 'svdquant_w4a4', 'viditq_w4a4'}:
        raise RuntimeError('Required native probe set is incomplete')
    for name, result in required.items():
        if result.get('status') != 'complete':
            raise RuntimeError('Required native probe did not complete: ' + name)


def _write_probe_receipt(path: Path, receipt: dict) -> None:
    content = json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if len(content.encode("utf-8")) > 256 * 1024:
        raise RuntimeError("Probe JSON receipt exceeded the 256 KiB bound")
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(content, encoding='utf-8')
    temporary.replace(path)


def _collect_remaining_probes(receipt, probes, out_path):
    failed = False
    for destination, name, probe in probes:
        try:
            destination[name] = probe()
        except Exception as exc:
            destination[name] = {'status': 'failed', 'error_type': type(exc).__name__,
                                 'error': str(exc)[:2000]}
            failed = True
        receipt['status'] = 'failed' if failed else 'running'
        _write_probe_receipt(out_path, receipt)
    receipt['status'] = 'failed' if failed else 'complete'
    _write_probe_receipt(out_path, receipt)
    return receipt


def run_probe(root: Path, out_path: Path) -> dict:
    # Do not import Torch, load extensions, or touch a GPU until the PBS guard passes.
    guard()
    from startup_trace import startup_trace
    with startup_trace(out_path.parent / 'kernel_probe_startup'):
        import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Native kernel probe requires a real CUDA compute allocation")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda", torch.cuda.current_device())
    hardware = _hardware(torch, device)
    extension_dir = root / "native_extensions" / "quarot_official_gemm"
    backend = _extension("quarot_official_gemm", extension_dir)
    if not callable(getattr(backend, "matmul", None)):
        raise RuntimeError("Official QuaRot extension must expose matmul(A_uint8[M,K/2], B_uint8[N,K/2]) -> int32[M,N]")
    commit = _source_commit(root)
    shapes = [_probe_quarot_case(backend, torch, device, m, n) for m, n in QUAROT_SHAPES]
    total_calls = sum(item["native_matmul_calls"] for item in shapes)
    diffusion = {}
    receipt = {
        "format": "cosmos_native_kernel_probe_v1",
        "status": "running",
        "mode": "random_operator_probe_only_no_policy_or_checkpoint_loaded",
        "pbs_job_id": os.environ["PBS_JOBID"],
        "host": socket.gethostname(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "tf32_disabled": not bool(torch.backends.cuda.matmul.allow_tf32),
        "hardware": hardware,
        "quarot": {
            "source_commit": commit,
            "extension": str(Path(backend.__file__).resolve()),
            "extension_api": "official QuaRot CUTLASS GEMM-only bridge matmul(uint8 A, uint8 B) -> int32",
            "kernel": "signed INT4 x signed INT4, int32 accumulator, SM80",
            "shape_count": len(shapes),
            "native_matmul_calls": total_calls,
            "correctness_matmul_calls": len(shapes) * 2,
            "timing_matmul_calls": len(shapes) * 2 * (5 + TIMING_REPEATS),
            "cuda_graph_replays": len(shapes) * 3,
            "all_max_abs_error_zero": True,
            "shapes": shapes,
        },
        "diffusion_optional_probes": diffusion,
        "training_updates": 0,
        "policy_forward_calls": 0,
        "closed_loop_episodes": 0,
    }
    _write_probe_receipt(out_path, receipt)
    return _collect_remaining_probes(receipt, [
        (diffusion, 'svdquant_w4a4', lambda: _probe_nunchaku(torch, device, root)),
        (diffusion, 'viditq_w4a4', lambda: _probe_viditq(torch, device, root)),
        (receipt, 'fused_quarot_scaled_add', lambda: _probe_fused_quarot_scaled_add(torch, device, root)),
        (receipt, 'official_hadamard', lambda: _probe_hadamard(torch, device, root)),
    ], out_path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path,
                        help="cosmos-policy-native-perf-20261025 experiment directory")
    parser.add_argument("--out", required=True, type=Path,
                        help="JSON receipt path, or output directory (writes kernel_probe.json)")
    parser.add_argument('--require-all', action='store_true')
    args = parser.parse_args(argv)
    output = _output_path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = None
    try:
        receipt = run_probe(args.root.resolve(), output)
        if args.require_all:
            require_all_native(receipt)
        if receipt['status'] == 'failed':
            raise RuntimeError('One or more independent native probes failed; see per-probe results')
    except Exception as exc:
        receipt = {**(receipt or {}),
            "format": "cosmos_native_kernel_probe_v1",
            "status": "failed",
            "pbs_job_id": os.environ.get("PBS_JOBID"),
            "host": socket.gethostname(),
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "training_updates": 0,
            "policy_forward_calls": 0,
            "closed_loop_episodes": 0,
        }
        _write_probe_receipt(output, receipt)
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
        return 2
    _write_probe_receipt(output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
