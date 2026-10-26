"""Allocation-guarded diagnosis of the pinned Nunchaku W4A4 packing path."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import socket
import sys
from datetime import datetime, timezone

from prepare_sources import guard

PINNED_NUNCHAKU = "302e0e97024ebd68688fe890e5df83731edf7b54"
REPRO_SHAPE = (129, 131, 256)
SEED = 777
ATOL = 0.125
RTOL = 0.002


def activation_scale_writes(rows: int, groups: int):
    """Return (destination-half-index, source-row) per official packed half."""
    if rows <= 0 or rows % 256 or groups <= 0:
        raise ValueError("activation scales require positive rows divisible by 256 and groups")
    writes = []
    for block_m in range(rows // 256):
        for group in range(groups):
            for warp_m in range(8):
                for lane in range(16):
                    dst = ((block_m * groups + group) * 8 + warp_m) * 16 + lane
                    row = block_m * 256 + warp_m * 32 + (lane // 8) * 16 + lane % 8
                    writes.append((dst * 2, row, group))
                    writes.append((dst * 2 + 1, row + 8, group))
    return tuple(writes)


def _output_path(path: Path) -> Path:
    return path if path.suffix.lower() == ".json" else path / "diagnose_nunchaku_pack.json"


def _build_receipt(root: Path) -> dict:
    candidates = (root / "results" / "preparation" / "native_diffusion_build.json",
                  root / "retrieved" / "preparation.native_diffusion_build.json")
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError("native_diffusion_build.json is required for the Nunchaku probe")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    arm = receipt.get("arms", {}).get("svdquant_w4a4", {})
    if receipt.get("status") != "complete" or arm.get("source_commit") != PINNED_NUNCHAKU:
        raise RuntimeError("Nunchaku build receipt is incomplete or does not match the pinned source commit")
    if Path(arm.get("module", "")).name != "nunchaku_svdq_native.so":
        raise RuntimeError("Build receipt does not identify nunchaku_svdq_native.so")
    return {"path": str(path), "pbs_job_id": receipt.get("pbs_job_id"),
            "source_commit": arm["source_commit"], "module": arm["module"]}


def _extension(directory: Path):
    import importlib.machinery

    name = "nunchaku_svdq_native"
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    binaries = [item for item in directory.glob(name + "*")
                if item.is_file() and item.name.endswith(suffixes)]
    if len(binaries) != 1:
        raise FileNotFoundError(f"Expected one built {name} module under {directory}, found {len(binaries)}")
    sys.path.insert(0, str(directory.resolve()))
    module = importlib.import_module(name)
    if Path(module.__file__).resolve() != binaries[0].resolve():
        raise RuntimeError(f"Imported {name} from an unexpected path: {module.__file__}")
    return module


def _activation_scale_reference(scales, torch):
    groups, rows = map(int, scales.shape)
    tiled = scales.reshape(groups, rows // 256, 8, 32)
    low = torch.cat((tiled[..., :8], tiled[..., 16:24]), dim=-1)
    high = torch.cat((tiled[..., 8:16], tiled[..., 24:32]), dim=-1)
    return (torch.stack((low, high), dim=-1)
            .permute(1, 0, 2, 3, 4).contiguous().view_as(scales))


def _weight_scale_reference(scales, torch):
    groups, columns = map(int, scales.shape)
    tiled = scales.reshape(groups, columns // 128, 128)
    lane = torch.arange(32, device=scales.device)
    base = (lane // 4) * 16 + (lane % 4) * 2
    packed = torch.stack((tiled[..., base], tiled[..., base + 1],
                          tiled[..., base + 8], tiled[..., base + 9]), dim=-1)
    return packed.permute(1, 0, 2, 3).contiguous().view_as(scales)


def _metrics(actual, expected, torch) -> dict:
    error = (actual.float() - expected.float()).abs()
    max_abs = float(error.max().item())
    rms_error = float(error.square().mean().sqrt().item())
    reference_rms = float(expected.float().square().mean().sqrt().item())
    nrmse = rms_error / reference_rms if reference_rms else rms_error
    bad = torch.nonzero(error > (ATOL + expected.float().abs() * RTOL), as_tuple=False)[:8]
    samples = []
    for row, column in bad.tolist():
        samples.append({"m": int(row), "n": int(column),
                        "actual": float(actual[row, column].float().item()),
                        "expected": float(expected[row, column].float().item()),
                        "abs_error": float(error[row, column].item())})
    return {"max_abs_error": max_abs, "nrmse": nrmse,
            "reference_rms": reference_rms,
            "tolerance": {"atol": ATOL, "rtol": RTOL},
            "within_tolerance": not bool(bad.numel()),
            "first_bad_locations": samples}


def _scale_layout_check(extension, device, torch):
    groups, columns = 4, 256
    w_scales = (torch.arange(groups * columns, device=device, dtype=torch.int32)
                .remainder(2053).reshape(groups, columns).to(torch.float16))
    actual_w = extension.pack_weight_scales(w_scales.contiguous())
    expected_w = _weight_scale_reference(w_scales, torch)
    activation_cases = []
    for rows in (256, 1792):
        a_scales = (torch.arange(groups * rows, device=device, dtype=torch.int32)
                    .remainder(4093).reshape(groups, rows).to(torch.float16))
        actual_a = extension.pack_activation_scales(a_scales.contiguous())
        expected_a = _activation_scale_reference(a_scales, torch)
        activation_cases.append({"rows": rows, "groups": groups,
                                 "output_numel": int(actual_a.numel()),
                                 "official_pinned_layout_match": bool(torch.equal(actual_a, expected_a)),
                                 "index_writes": groups * rows,
                                 "expected_unique_destination_halves": groups * rows})
    torch.cuda.current_stream(device).synchronize()
    return {
        "activation": {"cases": activation_cases},
        "weight": {"shape": [groups, columns], "output_numel": int(actual_w.numel()),
                   "official_pinned_layout_match": bool(torch.equal(actual_w, expected_w))},
        "distinct_scale_values": True,
    }


def _make_bank(b_codes, w_scales_g128, extension, device, torch):
    from kernel_probe import _pack_offset8_i4
    from native_diffusion import _nunchaku_bank

    n, k = map(int, b_codes.shape)
    bank = {"encoded": {"codes": _pack_offset8_i4(b_codes, torch),
                        "scales": w_scales_g128, "shape": (n, k), "bits": 4},
            "smoothing": torch.ones(k, device=device, dtype=torch.float16),
            "up": None, "down": None}
    module = type("WeightOwner", (), {"weight": type("Weight", (), {"device": device})()})()
    return _nunchaku_bank(bank, module, extension, torch, __import__("torch.nn.functional", fromlist=["pad"]), rank=16)


def _run_case(a_codes, b_codes, a_scales, w_scales_g128, extension,
              device, torch, functional, return_output=False):
    batch, k = map(int, a_codes.shape)
    n = int(b_codes.shape[0])
    padded_m = ((batch + 255) // 256) * 256
    bank = _make_bank(b_codes, w_scales_g128, extension, device, torch)
    padded_n = int(bank["padded_n"])
    a_padded = functional.pad(a_codes, (0, 0, 0, padded_m - batch), value=0).contiguous()
    a_scales_padded = functional.pad(a_scales, (0, padded_m - batch), value=1)
    packed_as_input = a_scales_padded.reshape(1, padded_m).expand(k // 64, padded_m).contiguous()
    packed_a = extension.pack_activation(a_padded)
    packed_as = extension.pack_activation_scales(packed_as_input)
    output = torch.empty((batch, padded_n), device=device, dtype=torch.float16)
    extension.matmul(packed_a, bank["packed_weight"], output,
                     packed_as, bank["packed_wscales"])

    padded_b = functional.pad(b_codes, (0, 0, 0, padded_n - n), value=0)
    w_scales_g64 = w_scales_g128.transpose(0, 1).contiguous().repeat_interleave(2, dim=0)
    w_scales_g64 = functional.pad(w_scales_g64, (0, padded_n - n), value=1)
    a_scales_g64 = a_scales.reshape(1, batch).expand(k // 64, batch)
    from kernel_probe import _reference_scaled_groups
    expected = _reference_scaled_groups(a_codes, padded_b, a_scales_g64,
                                        w_scales_g64, 64, torch)
    torch.cuda.current_stream(device).synchronize()
    metrics = _metrics(output, expected, torch)
    metrics.update({"M": batch, "padded_M": padded_m,
                    "N": n, "padded_N": padded_n, "K": k,
                    "native_matmul_calls": 1,
                    "result_shape": list(output.shape)})
    return (metrics, output) if return_output else metrics


def run(root: Path) -> dict:
    # Keep every CUDA/PyTorch action behind the real-allocation guard.
    guard()
    from startup_trace import startup_trace
    with startup_trace(root / 'results/preparation/nunchaku_pack_startup'):
        import torch
    import torch.nn.functional as functional

    build = _build_receipt(root)
    extension_dir = root / "native_extensions" / "svdquant_w4a4"
    extension = _extension(extension_dir)
    from kernel_probe import _hardware

    if not torch.cuda.is_available():
        raise RuntimeError("Nunchaku packing diagnosis requires a CUDA PBS allocation")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda", torch.cuda.current_device())
    hardware = _hardware(torch, device)
    layout = _scale_layout_check(extension, device, torch)
    receipt = {
        "format": "nunchaku_packing_diagnostic_v1", "status": "running",
        "mode": "random_operator_only_no_policy_checkpoint_training_or_closed_loop",
        "pbs_job_id": os.environ["PBS_JOBID"], "host": socket.gethostname(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED, "hardware": hardware,
        "build": build, "extension": str(Path(extension.__file__).resolve()),
        "pinned_scale_layout": {
            "source_commit": PINNED_NUNCHAKU,
            "source": "src/kernels/zgemm/gemm_base.cuh: ASCALES_PACK_SIZE=2, ASCALES_VALID_LANES=16; pack_ascales writes one half2 per valid lane",
            "activation_index": "((block_m*groups+group)*8+warp_m)*16+lane; each packed pair stores input row local then local+8",
            "weight_source": "same gemm_base.cuh pack_wscales, four FP16 values per lane",
        },
        "scale_layout_check": layout,
        "cases": {}, "native_matmul_calls": 0,
        "training_updates": 0, "policy_forward_calls": 0, "closed_loop_episodes": 0,
    }
    activation_layout_matches = all(case["official_pinned_layout_match"]
                                    for case in layout["activation"]["cases"])
    if not (activation_layout_matches
            and layout["weight"]["official_pinned_layout_match"]):
        receipt["status"] = "failed"
        receipt["failure"] = "Extension scale bytes differ from the pinned official lane layout; GEMM skipped to avoid consuming malformed scales."
        return receipt

    # All checkpoint-style weight scales vary by original G128 group and output row.
    m, n, k = REPRO_SHAPE
    a_scale = (0.25 + torch.arange(m, device=device, dtype=torch.float32).remainder(23) * 0.03125).to(torch.float16)
    n_index = torch.arange(n, device=device, dtype=torch.float32)
    w_scale = torch.stack((0.125 + n_index.remainder(17) * 0.03125,
                           0.5 + n_index.remainder(13) * 0.046875), dim=1).to(torch.float16)
    if bool(torch.equal(w_scale[:, 0], w_scale[:, 1])):
        raise AssertionError("G128 diagnostic scales must be distinct")

    constant_a = torch.ones((m, k), device=device, dtype=torch.int8)
    constant_b = torch.ones((n, k), device=device, dtype=torch.int8)
    constant_as = torch.ones((m,), device=device, dtype=torch.float16)
    constant_ws = torch.ones((n, 2), device=device, dtype=torch.float16)
    receipt["cases"]["constant_codes_scales"] = _run_case(
        constant_a, constant_b, constant_as, constant_ws, extension,
        device, torch, functional)

    # Seeded random reproduction of the failed adapter-pack path, now with two G128 groups.
    a_codes = torch.randint(-7, 8, (m, k), device=device, dtype=torch.int16).to(torch.int8)
    b_codes = torch.randint(-7, 8, (n, k), device=device, dtype=torch.int16).to(torch.int8)
    receipt["cases"]["seed777_random_reproduction"] = _run_case(
        a_codes, b_codes, a_scale, w_scale, extension, device, torch, functional)
    receipt["cases"]["seed777_random_reproduction"]["scale_semantics"] = (
        "checkpoint G128 scales are preserved and duplicated into the two official G64 groups; "
        "per-token activation scale is repeated over all G64 groups")

    one_hot_cases = []
    for row, column, k_index in ((0, 0, 0), (0, 130, 63),
                                 (128, 0, 64), (128, 130, 127),
                                 (0, 130, 128), (128, 0, 255)):
        one_a = torch.zeros((m, k), device=device, dtype=torch.int8)
        one_b = torch.zeros((n, k), device=device, dtype=torch.int8)
        one_a[row, k_index] = 1
        one_b[column, k_index] = 1
        measured, selected_output = _run_case(
            one_a, one_b, a_scale, w_scale, extension,
            device, torch, functional, return_output=True)
        expected_value = float((a_scale[row].float() * w_scale[column, k_index // 128].float()).item())
        measured_value = float(selected_output[row, column].float().item())
        one_hot_abs_error = abs(measured_value - expected_value)
        measured["one_hot_selected_value"] = {
            "m": row, "n": column, "k": k_index,
            "g128_group": k_index // 128, "g64_group": k_index // 64,
            "actual": measured_value, "expected": expected_value,
            "max_abs_error": one_hot_abs_error, "atol": 0.001,
            "passed": one_hot_abs_error <= 0.001,
        }
        measured["within_tolerance"] = (measured["within_tolerance"]
                                        and measured["one_hot_selected_value"]["passed"])
        measured.update({"one_hot_m": row, "one_hot_n": column,
                         "one_hot_k": k_index, "g128_group": k_index // 128,
                         "g64_group": k_index // 64})
        one_hot_cases.append(measured)
    receipt["cases"]["single_one_hot_m_n_k"] = one_hot_cases
    all_cases = [receipt["cases"]["constant_codes_scales"],
                 receipt["cases"]["seed777_random_reproduction"], *one_hot_cases]
    receipt["native_matmul_calls"] = sum(case["native_matmul_calls"] for case in all_cases)
    receipt["status"] = ("complete" if all(case["within_tolerance"] for case in all_cases)
                         else "failed")
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    output = _output_path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        receipt = run(args.root.resolve())
    except Exception as exc:
        receipt = {"format": "nunchaku_packing_diagnostic_v1", "status": "failed",
                   "pbs_job_id": os.environ.get("PBS_JOBID"), "host": socket.gethostname(),
                   "error_type": type(exc).__name__, "error": str(exc)[:2000],
                   "training_updates": 0, "policy_forward_calls": 0,
                   "closed_loop_episodes": 0}
    serialized = json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if len(serialized.encode("utf-8")) > 256 * 1024:
        raise RuntimeError("Diagnostic JSON exceeded the 256 KiB limit")
    output.write_text(serialized, encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if receipt.get("status") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
