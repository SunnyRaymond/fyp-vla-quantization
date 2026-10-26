"""Small, allocation-guarded diagnostics for ViDiT-Q Atom W4A4 packing."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import importlib.machinery
import json
import os
from pathlib import Path
import socket
import sys

from prepare_sources import guard

PINNED_VIDITQ = "43657d11b32192ec0f50f0dac8d3a16c3d841196"
M, N, K = 129, 131, 256
GROUP_WIDTH = 128
GROUPS = K // GROUP_WIDTH
PADDED_M = 256
PADDED_N = 256
SEED = 777
ATOL = 0.125
RTOL = 0.002


def _output_path(path: Path) -> Path:
    return path if path.suffix.lower() == ".json" else path / "atom_diagnostic.json"


def _build_receipt(root: Path) -> dict:
    candidates = (root / "results" / "preparation" / "native_diffusion_build.json",
                  root / "retrieved" / "preparation.native_diffusion_build.json")
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError("native_diffusion_build.json is required for the Atom diagnostic")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    arm = receipt.get("arms", {}).get("viditq_w4a4", {})
    if receipt.get("status") != "complete" or arm.get("source_commit") != PINNED_VIDITQ:
        raise RuntimeError("ViDiT-Q build receipt is incomplete or does not match the pinned source commit")
    module_path = Path(arm.get("module", ""))
    if module_path.name != "viditq_atom_native.so":
        raise RuntimeError("Build receipt does not identify viditq_atom_native.so")
    return {"receipt_path": str(path), "pbs_job_id": receipt.get("pbs_job_id"),
            "source_commit": arm["source_commit"], "module": arm["module"]}


def _extension(directory: Path):
    name = "viditq_atom_native"
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    binaries = [path for path in directory.glob(name + "*")
                if path.is_file() and path.name.endswith(suffixes)] if directory.is_dir() else []
    if len(binaries) != 1:
        raise FileNotFoundError(f"Expected one built {name} module under {directory}, found {len(binaries)}")
    resolved_binary = binaries[0].resolve()
    cached = sys.modules.get(name)
    if cached is not None:
        if Path(cached.__file__).resolve() != resolved_binary:
            raise RuntimeError(f"{name} is already loaded from an unexpected path: {cached.__file__}")
        return cached
    sys.path.insert(0, str(directory.resolve()))
    module = importlib.import_module(name)
    if Path(module.__file__).resolve() != resolved_binary:
        raise RuntimeError(f"Imported {name} from an unexpected path: {module.__file__}")
    return module


def _pack_offset8_i4(codes, torch):
    values = (codes.to(torch.int16) + 8).to(torch.uint8)
    return (values[:, 0::2] | (values[:, 1::2] << 4)).contiguous()


def _pack_signed_i4(codes, torch):
    values = codes.to(torch.int16).bitwise_and(15).to(torch.uint8)
    return (values[:, 0::2] | (values[:, 1::2] << 4)).contiguous()


def _activation_scale_source_rows(layout: str, padded_m: int) -> tuple[int, ...]:
    """Source row for each packed half, separated from the CUDA lane oracle."""
    if padded_m < 128 or padded_m % 128:
        raise ValueError("Atom scales require M padded to a 128-row multiple")
    output = []
    if layout == "legacy_row_major":
        order = range(padded_m)
        for row in order:
            output.extend((row,) * 4)
    elif layout in ("atom_v3", "paired_half2_repeat4"):
        for tile in range(0, padded_m, 128):
            for block in range(0, 128, 16):
                for lane in range(8):
                    row = tile + block + lane
                    if layout == "atom_v3":
                        output.extend((row,) * 4)
                        output.extend((row + 8,) * 4)
                    else:
                        for _ in range(4):
                            output.extend((row, row + 8))
    else:
        raise ValueError(f"Unknown A-scale candidate: {layout}")
    if len(output) != 4 * padded_m:
        raise AssertionError("Atom A-scale candidate generated an invalid number of halves")
    return tuple(output)


def _candidate_a_scales(scale_rows, layout: str, torch):
    if layout == "production_packing_v4":
        from native_diffusion import _viditq_activation_scales
        return _viditq_activation_scales(scale_rows, torch)
    rows = _activation_scale_source_rows(layout, int(scale_rows.shape[1]))
    indices = torch.tensor(rows, device=scale_rows.device, dtype=torch.long)
    return scale_rows.index_select(1, indices).contiguous()


def _make_weight_item(b_codes, w_scales_gn, extension, device, torch, functional):
    from native_diffusion import _viditq_bank

    n, k = map(int, b_codes.shape)
    bank = {"encoded": {"codes": _pack_offset8_i4(b_codes, torch),
                        "scales": w_scales_gn.transpose(0, 1).contiguous(),
                        "zero_points": torch.full((n, GROUPS), 8, device=device,
                                                  dtype=torch.int16),
                        "shape": (n, k), "bits": 4},
            "smoothing": torch.ones(k, device=device, dtype=torch.float16),
            "signs": None}
    owner = type("WeightOwner", (), {"weight": type("Weight", (), {"device": device})()})()
    return _viditq_bank(bank, owner, extension, torch, functional,
                        retain_probe_offsets=True)


def _reference(a_codes, b_codes, a_scales_gm, w_scales_gn, padded_m, padded_n, torch):
    """Direct FP32 group dot from source codes and original, un-packed scales."""
    groups = int(a_codes.shape[1]) // GROUP_WIDTH
    result = torch.zeros((padded_m, padded_n), device=a_codes.device, dtype=torch.float32)
    a = torch.zeros((padded_m, int(a_codes.shape[1])), device=a_codes.device, dtype=torch.int8)
    b = torch.zeros((padded_n, int(b_codes.shape[1])), device=b_codes.device, dtype=torch.int8)
    a[:a_codes.shape[0]] = a_codes
    b[:b_codes.shape[0]] = b_codes
    a_scales = torch.ones((groups, padded_m), device=a_codes.device, dtype=torch.float16)
    a_scales[:, :a_scales_gm.shape[1]] = a_scales_gm
    w_scales = torch.ones((groups, padded_n), device=b_codes.device, dtype=torch.float16)
    w_scales[:, :w_scales_gn.shape[1]] = w_scales_gn
    for group in range(groups):
        start = group * GROUP_WIDTH
        stop = start + GROUP_WIDTH
        integer_dot = torch.matmul(a[:, start:stop].float(), b[:, start:stop].float().transpose(0, 1))
        result.add_(integer_dot * a_scales[group].float()[:, None]
                    * w_scales[group].float()[None, :])
    return result


def _metrics(actual, expected, torch, sample_positions=()) -> dict:
    actual_f = actual.float()
    expected_f = expected.float()
    finite_mask = torch.isfinite(actual_f) & torch.isfinite(expected_f)
    finite = bool(finite_mask.all().item())
    samples = []
    for row, column in sample_positions:
        samples.append({"m": int(row), "n": int(column),
                        "actual": float(actual_f[row, column].item())
                        if bool(torch.isfinite(actual_f[row, column]).item()) else None,
                        "expected": float(expected_f[row, column].item())
                        if bool(torch.isfinite(expected_f[row, column]).item()) else None})
    if not finite:
        return {"finite": False, "nonfinite_actual": int((~torch.isfinite(actual_f)).sum().item()),
                "nonfinite_expected": int((~torch.isfinite(expected_f)).sum().item()),
                "max_abs_error": None, "nrmse": None, "reference_rms": None,
                "within_fixed_tolerance": False, "samples": samples}
    error = (actual_f - expected_f).abs()
    max_abs = float(error.max().item())
    rms_error = float(error.square().mean().sqrt().item())
    reference_rms = float(expected_f.square().mean().sqrt().item())
    nrmse = rms_error / reference_rms if reference_rms else rms_error
    limit = ATOL + expected_f.abs() * RTOL
    bad = torch.nonzero(error > limit, as_tuple=False)[:8]
    bad_locations = [{"m": int(row), "n": int(column),
                      "actual": float(actual_f[row, column].item()),
                      "expected": float(expected_f[row, column].item()),
                      "abs_error": float(error[row, column].item())}
                     for row, column in bad.tolist()]
    return {"finite": True, "nonfinite_actual": 0, "nonfinite_expected": 0,
            "max_abs_error": max_abs, "nrmse": nrmse, "reference_rms": reference_rms,
            "within_fixed_tolerance": not bool(bad.numel()),
            "tolerance": {"atol": ATOL, "rtol": RTOL},
            "first_over_tolerance": bad_locations, "samples": samples}


def _candidate_difference(actual, candidate, torch, sample_positions=()) -> dict:
    actual_f = actual.float()
    candidate_f = candidate.float()
    finite = bool((torch.isfinite(actual_f) & torch.isfinite(candidate_f)).all().item())
    if not finite:
        return {"finite": False, "max_abs_output_difference": None,
                "nrmse_between_candidates": None}
    error = (actual_f - candidate_f).abs()
    reference_rms = float(actual_f.square().mean().sqrt().item())
    rms_error = float(error.square().mean().sqrt().item())
    samples = [{"m": int(row), "n": int(column),
                "atom_v3": float(actual_f[row, column].item()),
                "candidate": float(candidate_f[row, column].item())}
               for row, column in sample_positions]
    return {"finite": True, "max_abs_output_difference": float(error.max().item()),
            "nrmse_between_candidates": rms_error / reference_rms if reference_rms else rms_error,
            "samples": samples}


def _run_matmul(a_codes, b_codes, a_scales_gm, w_scales_gn, item, a_layout,
                extension, device, torch, functional, call_counter,
                sample_positions=()):
    padded_a = functional.pad(a_codes, (0, 0, 0, PADDED_M - M), value=0).contiguous()
    a_scale_rows = torch.ones((GROUPS, PADDED_M), device=device, dtype=torch.float16)
    a_scale_rows[:, :M] = a_scales_gm
    w_scale_rows = torch.ones((GROUPS, PADDED_N), device=device, dtype=torch.float16)
    w_scale_rows[:, :N] = w_scales_gn
    packed_a_scales = _candidate_a_scales(a_scale_rows, a_layout, torch)
    packed_a = _pack_signed_i4(padded_a, torch)
    packed_b, packed_b_scales, _offset, _multiplier = item["passes"][0]
    a_keeper = torch.zeros((PADDED_M, GROUP_WIDTH), device=device, dtype=torch.int8)
    b_keeper = torch.zeros((GROUP_WIDTH, PADDED_N), device=device, dtype=torch.int8)
    a_keeper_scales = torch.ones((4 * PADDED_M,), device=device, dtype=torch.float16)
    b_keeper_scales = torch.ones((PADDED_N,), device=device, dtype=torch.float16)
    output = torch.empty((PADDED_M, PADDED_N), device=device, dtype=torch.float16)
    extension.matmul(packed_a, packed_b, output, packed_a_scales, packed_b_scales,
                     a_keeper, b_keeper, a_keeper_scales, b_keeper_scales)
    call_counter[0] += 1
    torch.cuda.current_stream(device).synchronize()
    expected = _reference(a_codes, b_codes, a_scale_rows, w_scale_rows,
                          PADDED_M, PADDED_N, torch)
    result = _metrics(output, expected, torch, sample_positions)
    result.update({"M": M, "padded_M": PADDED_M, "N": N,
                   "padded_N": PADDED_N, "K": K, "groups": GROUPS,
                   "weight_group_width": GROUP_WIDTH, "a_scale_layout": a_layout,
                   "result_shape": list(output.shape)})
    return result, output


def _capture_case(cases: dict, name: str, fn) -> None:
    try:
        cases[name] = fn()
    except Exception as exc:
        cases[name] = {"status": "case_error", "error_type": type(exc).__name__,
                       "error": str(exc)[:1000]}


def run(root: Path, out_path: Path | None = None, *, torch_module=None,
        device=None, allocation_guarded: bool = False, extension=None,
        extension_label: str | None = None) -> dict:
    """Run diagnostic after guard; caller may pass Torch to share one process."""
    if torch_module is not None and not allocation_guarded:
        raise RuntimeError("Pass allocation_guarded=True only after the PBS guard ran before Torch import")
    if not allocation_guarded:
        guard()
    torch = torch_module
    if torch is None:
        import torch as torch_module_local
        torch = torch_module_local
    import torch.nn.functional as functional

    if not torch.cuda.is_available():
        raise RuntimeError("Atom diagnosis requires a real CUDA compute allocation")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())

    from kernel_probe import _hardware
    from native_diffusion import NATIVE_ADAPTER_VERSIONS

    hardware = _hardware(torch, device)
    if extension_label not in (None, "original", "general"):
        raise ValueError("extension_label must be 'original' or 'general'")
    if extension is None:
        if extension_label not in (None, "original"):
            raise ValueError("The general module must be supplied by the caller after receipt validation")
        build = _build_receipt(root)
        extension_dir = Path(build["module"]).resolve().parent
        extension = _extension(extension_dir)
        extension_label = "original"
    elif extension_label == "original":
        build = _build_receipt(root)
    elif extension_label == "general":
        from native_diffusion import _viditq_general_build
        build = _viditq_general_build(root)
    else:
        raise ValueError("Injected diagnostic extensions require extension_label='original' or 'general'")
    if not callable(getattr(extension, "matmul", None)):
        raise RuntimeError(f"Injected {extension_label} Atom module has no callable matmul")
    if extension_label == "general":
        loaded_path = Path(getattr(extension, "__file__", "")).resolve()
        if loaded_path != Path(build["module"]).resolve():
            raise RuntimeError("General Atom module path differs from the strict general-build receipt")
    elif extension_label == "original":
        loaded_path = Path(getattr(extension, "__file__", "")).resolve()
        if loaded_path != Path(build["module"]).resolve():
            raise RuntimeError("Original Atom module path differs from its build receipt")
    cases = {}
    call_counter = [0]
    w_ones = torch.ones((GROUPS, N), device=device, dtype=torch.float16)
    a_ones = torch.ones((GROUPS, M), device=device, dtype=torch.float16)

    constant_a = torch.ones((M, K), device=device, dtype=torch.int8)
    constant_b = torch.ones((N, K), device=device, dtype=torch.int8)
    constant_item = _make_weight_item(constant_b, w_ones, extension, device,
                                      torch, functional)
    _capture_case(cases, "constant_codes_and_scales", lambda: _run_matmul(
        constant_a, constant_b, a_ones, w_ones, constant_item, "atom_v3",
        extension, device, torch, functional, call_counter,
        ((0, 0), (0, 127), (0, 128), (128, 130), (255, 255)))[0])

    random_a = torch.randint(-7, 8, (M, K), device=device, dtype=torch.int16).to(torch.int8)
    random_b = torch.randint(-7, 8, (N, K), device=device, dtype=torch.int16).to(torch.int8)
    random_item = _make_weight_item(random_b, w_ones, extension, device,
                                    torch, functional)
    _capture_case(cases, "seed777_random_codes_unit_scales", lambda: _run_matmul(
        random_a, random_b, a_ones, w_ones, random_item, "atom_v3",
        extension, device, torch, functional, call_counter,
        ((0, 0), (0, 127), (0, 128), (127, 127), (128, 130), (255, 255)))[0])

    one_hot_cases = []
    for index, (m, n, k) in enumerate(((0, 0, 0), (0, 1, 0), (0, 2, 0),
                                       (127, 127, 63),
                                       (128, 128, 64), (128, 130, 127),
                                       (0, 130, 128), (128, 0, 255))):
        a_hot = torch.zeros((M, K), device=device, dtype=torch.int8)
        b_hot = torch.zeros((N, K), device=device, dtype=torch.int8)
        a_hot[m, k] = 1
        b_hot[n, k] = 1
        hot_item = _make_weight_item(b_hot, w_ones, extension, device,
                                     torch, functional)
        case = {"id": index, "one_hot": {"m": m, "n": n, "k": k}}
        _capture_case(case, "result", lambda a_hot=a_hot, b_hot=b_hot, hot_item=hot_item,
                      m=m, n=n: _run_matmul(
                          a_hot, b_hot, a_ones, w_ones, hot_item, "atom_v3",
                          extension, device, torch, functional, call_counter,
                          ((m, n), (0, 0), (127, 127), (128, 128), (128, 130)))[0])
        one_hot_cases.append(case)
    cases["one_hot_boundary_cases"] = one_hot_cases

    unique_a_rows = (0.25 + torch.arange(M, device=device, dtype=torch.float32)
                     .remainder(23) * 0.03125).to(torch.float16)
    unique_a = unique_a_rows.reshape(1, M).expand(GROUPS, M).contiguous()
    unique_a_item = _make_weight_item(constant_b, w_ones, extension, device,
                                      torch, functional)
    unique_a_case = {}
    unique_a_outputs = {}
    a_samples = ((0, 0), (0, 1), (0, 2), (7, 127), (8, 128),
                 (15, 130), (16, 0), (127, 127), (128, 130), (255, 255))
    for layout in ("atom_v3", "legacy_row_major", "paired_half2_repeat4",
                   "production_packing_v4"):
        def run_candidate(layout=layout):
            metrics, output = _run_matmul(
                constant_a, constant_b, unique_a, w_ones, unique_a_item, layout,
                extension, device, torch, functional, call_counter, a_samples)
            unique_a_outputs[layout] = output
            return metrics
        _capture_case(unique_a_case, layout, run_candidate)
    unique_a_case["candidate_output_differences"] = {}
    if "atom_v3" in unique_a_outputs:
        for layout, output in unique_a_outputs.items():
            if layout != "atom_v3":
                unique_a_case["candidate_output_differences"][layout] = _candidate_difference(
                    unique_a_outputs["atom_v3"], output, torch, a_samples)
    unique_a_case["comparison_note"] = (
        "Each candidate is compared directly against FP32 integer dot times original A/W scales; "
        "candidate-to-candidate differences are descriptive only.")
    cases["unique_A_scales_layout_candidates"] = unique_a_case

    for group in range(GROUPS):
        group_a = torch.zeros((M, K), device=device, dtype=torch.int8)
        group_b = torch.zeros((N, K), device=device, dtype=torch.int8)
        start = group * GROUP_WIDTH
        stop = start + GROUP_WIDTH
        group_a[:, start:stop] = 1
        group_b[:, start:stop] = 1
        n_scale = (0.25 + torch.arange(N, device=device, dtype=torch.float32)
                   .remainder(31) * 0.015625 + group * 0.25).to(torch.float16)
        group_w_scales = w_ones.clone()
        group_w_scales[group] = n_scale
        group_item = _make_weight_item(group_b, group_w_scales, extension,
                                       device, torch, functional)
        _capture_case(cases, f"isolated_weight_scale_group_{group}",
                      lambda group_a=group_a, group_b=group_b,
                      group_w_scales=group_w_scales, group_item=group_item: _run_matmul(
                          group_a, group_b, a_ones, group_w_scales, group_item,
                          "atom_v3", extension, device, torch, functional, call_counter,
                          ((0, 0), (0, 1), (0, 2), (0, 127), (0, 128), (0, 130),
                           (128, 0), (128, 1), (128, 2), (128, 130), (255, 255)))[0])

    receipt = {
        "format": "viditq_atom_packing_diagnostic_v1",
        "status": "complete",
        "mode": "native_operator_only_no_policy_load_training_or_closed_loop",
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "host": socket.gethostname(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "hardware": hardware,
        "build": build,
        "extension_label": extension_label,
        "extension": str(Path(extension.__file__).resolve()),
        "adapter_version": NATIVE_ADAPTER_VERSIONS["viditq_w4a4"],
        "shape": {"M": M, "padded_M": PADDED_M, "N": N, "padded_N": PADDED_N,
                  "K": K, "weight_group_width": GROUP_WIDTH, "groups": GROUPS},
        "scale_layout_candidates": ["atom_v3", "legacy_row_major",
                                     "paired_half2_repeat4", "production_packing_v4"],
        "oracle": "FP32 integer dot from source signed code tensors times original FP16 scale tensors; no packed scale reverse-decoding",
        "fixed_tolerance": {"atol": ATOL, "rtol": RTOL,
                            "used_for_reporting_only": True},
        "cases": cases,
        "native_matmul_calls": call_counter[0],
        "training_updates": 0,
        "policy_forward_calls": 0,
        "closed_loop_episodes": 0,
    }
    if out_path is not None:
        _write_receipt(_output_path(out_path), receipt)
    return receipt


def _write_receipt(path: Path, receipt: dict) -> None:
    content = json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if len(content.encode("utf-8")) > 256 * 1024:
        raise RuntimeError("Atom diagnostic JSON exceeded the 256 KiB bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
