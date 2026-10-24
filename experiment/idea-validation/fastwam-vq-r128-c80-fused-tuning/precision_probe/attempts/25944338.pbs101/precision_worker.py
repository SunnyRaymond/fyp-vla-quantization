"""PBS-only numerical attribution for the selected packed VQ GEMM."""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path


SEED = 20261024


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def _dense_kernel_factory():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def dense_gemm(X, W, BIAS, Y,
                   M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                   XS0: tl.constexpr, XS1: tl.constexpr,
                   WS0: tl.constexpr, WS1: tl.constexpr,
                   YS0: tl.constexpr, YS1: tl.constexpr,
                   BS0: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        rows = pid_m * BM + tl.arange(0, BM)
        cols = pid_n * BN + tl.arange(0, BN)
        kk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for kt in range(tl.cdiv(K, BK)):
            k = kt * BK + kk
            a = tl.load(X + rows[:, None] * XS0 + k[None, :] * XS1,
                        (rows[:, None] < M) & (k[None, :] < K), other=0)
            b = tl.load(W + cols[:, None] * WS0 + k[None, :] * WS1,
                        (cols[:, None] < N) & (k[None, :] < K), other=0)
            acc = tl.dot(a, tl.trans(b), acc, out_dtype=tl.float32)
        bias = tl.load(BIAS + cols * BS0, cols < N, other=0).to(tl.float32)
        acc = acc + bias[None, :]
        tl.store(Y + rows[:, None] * YS0 + cols[None, :] * YS1, acc,
                 (rows[:, None] < M) & (cols[None, :] < N))

    return triton, dense_gemm


def _pair_metrics(left, right, torch):
    finite = bool((torch.isfinite(left).all() & torch.isfinite(right).all()).item())
    bit_equal = torch.equal(left.contiguous().view(torch.int16),
                            right.contiguous().view(torch.int16))
    if not finite:
        return {"finite": False, "max_abs": None, "rmse": None,
                "nrmse": None, "raw_bf16_bit_equal": bool(bit_equal)}
    diff = left.float() - right.float()
    max_abs = float(diff.abs().max().item())
    rmse = float(diff.square().mean().sqrt().item())
    reference_rms = right.float().square().mean().sqrt().clamp_min(1e-12)
    nrmse = float((diff.square().mean().sqrt() / reference_rms).item())
    del diff, reference_rms
    return {"finite": True, "max_abs": max_abs, "rmse": rmse,
            "nrmse": nrmse, "raw_bf16_bit_equal": bool(bit_equal)}


def _first_encodings(bundle, wanted):
    found = {}
    for module, encoding in bundle["quantized"].items():
        nk = tuple(map(int, encoding["shape"]))
        if nk in wanted and nk not in found:
            found[nk] = (module, encoding)
    missing = sorted(wanted - found.keys())
    if missing:
        raise ValueError(f"Bundle has no actual quantized encoding for N,K={missing}")
    return found


def main():
    from reference_profile import freeze_numerics, guard

    guard()
    artifacts = Path(os.environ["ARTIFACTS"])
    protocol = json.loads((artifacts / "protocol.json").read_text())
    if protocol.get("new_training_updates") != 0 or protocol.get("new_whole_model_forwards") != 0:
        raise ValueError("Precision probe protocol must forbid training and whole-model forwards")
    shapes = [tuple(map(int, row)) for row in protocol["shapes"]]
    expected_shapes = [
        (294, 3072, 14336), (294, 14336, 3072), (294, 3072, 3072),
        (129, 3072, 3072), (32, 3072, 1024), (32, 1024, 3072),
    ]
    if shapes != expected_shapes:
        raise ValueError(f"Precision probe shapes differ from the frozen six-case panel: {shapes}")

    import torch
    import export_runtime as export
    import fused_vq

    out = Path(protocol["remote_root"]) / "results"
    out.mkdir(parents=True, exist_ok=True)
    shape_dir = out / "precision_shapes"
    progress_path = out / "progress.json"
    dispatch_path = Path(protocol["selected_configs_path"])
    fused_vq.load_dispatch(dispatch_path)
    for m, n, k in shapes:
        if f"{m},{n},{k}" not in fused_vq.SELECTED_CONFIGS:
            raise ValueError(f"Selected dispatch lacks shape {(m, n, k)}")

    freeze_numerics(torch)
    torch.utils.deterministic.fill_uninitialized_memory = True
    original_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = original_reduction
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    header = {
        "protocol": protocol["protocol"], "pbs_jobid": os.environ["PBS_JOBID"],
        "complete": False, "completed_shapes": 0, "expected_shapes": len(shapes),
        "gpu": torch.cuda.get_device_name(device),
        "gpu_uuid": str(getattr(properties, "uuid", "unavailable")),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "numerical_flags": {
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "allow_bf16_reduced_precision_reduction_initial": original_reduction,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "precision_boundary": (
            "FP32-input F.linear rounded to BF16 is a high-precision comparison path, "
            "not an exact mathematical oracle; Triton dense GEMM is an independent "
            "FP32-accumulation implementation, not a proof of cuBLAS equivalence."
        ),
        "new_training_updates": 0, "new_whole_model_forwards": 0,
        "shapes": [],
    }
    _save(out / "precision.json", header)

    _save(progress_path, {"protocol": protocol["protocol"],
                          "pbs_jobid": os.environ["PBS_JOBID"],
                          "phase": "loading_actual_bundle", "completed_shapes": 0,
                          "expected_shapes": len(shapes)})
    bundle = export.load_bundle(protocol["export_reuse"]["bundle_path"])
    encodings = _first_encodings(bundle, {(n, k) for _, n, k in shapes})
    prepared_by_nk = {}
    for nk, (module, cpu_encoding) in encodings.items():
        cuda_encoding = {key: value.to(device=device) if isinstance(value, torch.Tensor) else value
                         for key, value in cpu_encoding.items()}
        decoded = fused_vq.decode_weight(cuda_encoding)
        prepared = fused_vq.prepare(cuda_encoding)
        prepared_by_nk[nk] = (module, decoded, prepared)
    del encodings, bundle
    torch.cuda.synchronize()

    triton, dense_kernel = _dense_kernel_factory()
    all_finite = True
    for index, (m, n, k) in enumerate(shapes, start=1):
        module, weight, prepared = prepared_by_nk[(n, k)]
        config_name = fused_vq.SELECTED_CONFIGS[f"{m},{n},{k}"]
        config = fused_vq.CONFIGS[config_name]
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        x = torch.randn((m, k), generator=generator, dtype=torch.float32).to(
            device=device, dtype=torch.bfloat16)
        zero_bias = torch.zeros((n,), device=device, dtype=torch.bfloat16)
        fp32_bias = zero_bias.float()
        torch.cuda.synchronize()

        outputs = {}
        for flag in (True, False):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag
            outputs[f"cublas_rpr_{str(flag).lower()}"] = torch.nn.functional.linear(
                x, weight, zero_bias)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = original_reduction
        outputs["packed_selected_config"] = fused_vq.prepared_linear(
            x, prepared, zero_bias, config=config_name)
        dense = torch.empty((m, n), device=device, dtype=torch.bfloat16)
        dense_kernel[(triton.cdiv(n, config.BN), triton.cdiv(m, config.BM))](
            x, weight, fp32_bias, dense, m, n, k,
            x.stride(0), x.stride(1), weight.stride(0), weight.stride(1),
            dense.stride(0), dense.stride(1), fp32_bias.stride(0),
            config.BM, config.BN, config.BK,
            num_warps=4, enable_fp_fusion=False)
        outputs["dense_triton_fp32_acc"] = dense
        outputs["fp32_operands_bf16_out"] = torch.nn.functional.linear(
            x.float(), weight.float(), fp32_bias).to(torch.bfloat16)
        torch.cuda.synchronize()

        pairs = {}
        for (left_name, left), (right_name, right) in itertools.combinations(outputs.items(), 2):
            pair = f"{left_name}_vs_{right_name}"
            pairs[pair] = _pair_metrics(left, right, torch)
        shape_finite = all(row["finite"] for row in pairs.values())
        row = {
            "shape": {"M": m, "N": n, "K": k}, "module": module,
            "seed": SEED, "selected_config": config_name,
            "allow_bf16_reduced_precision_reduction_tested": [True, False],
            "outputs_finite": {name: bool(torch.isfinite(value).all().item())
                               for name, value in outputs.items()},
            "pairs": pairs,
        }
        all_finite &= shape_finite and all(row["outputs_finite"].values())

        if (m, n, k) == (32, 1024, 3072):
            bias_generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
            random_bias = torch.randn((n,), generator=bias_generator, dtype=torch.float32).to(
                device=device, dtype=torch.bfloat16)
            packed_bias = fused_vq.prepared_linear(x, prepared, random_bias, config=config_name)
            bias_rows = {}
            for flag in (True, False):
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag
                with_bias = torch.nn.functional.linear(x, weight, random_bias)
                matmul_then_bias = torch.matmul(x, weight.t()) + random_bias
                bias_rows[str(flag).lower()] = _pair_metrics(with_bias, matmul_then_bias, torch)
                bias_rows[str(flag).lower()]["outputs_finite"] = bool(
                    (torch.isfinite(with_bias).all() & torch.isfinite(matmul_then_bias).all()).item())
                bias_rows[str(flag).lower()]["packed_vs_flinear"] = _pair_metrics(packed_bias, with_bias, torch)
                all_finite &= bias_rows[str(flag).lower()]["finite"]
                all_finite &= bias_rows[str(flag).lower()]["packed_vs_flinear"]["finite"]
                del with_bias, matmul_then_bias
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = original_reduction
            row["random_bf16_bias_flinear_vs_matmul_then_bias"] = bias_rows
            del random_bias, packed_bias

        _save(shape_dir / f"shape_{index:02d}_{m}_{n}_{k}.json", row)
        header["shapes"].append(row)
        header["completed_shapes"] = index
        _save(out / "precision.json", header)
        _save(progress_path, {"protocol": protocol["protocol"],
                              "pbs_jobid": os.environ["PBS_JOBID"],
                              "phase": "shape_complete", "completed_shapes": index,
                              "expected_shapes": len(shapes),
                              "last_shape": row["shape"], "all_finite_so_far": all_finite})
        print("PRECISION_SHAPE " + json.dumps({"index": index, "expected": len(shapes),
              "shape": row["shape"], "module": module, "config": config_name,
              "all_finite": shape_finite}, allow_nan=False), flush=True)
        del outputs, dense, fp32_bias, x, zero_bias
        torch.cuda.empty_cache()

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = original_reduction
    header["complete"] = len(header["shapes"]) == len(shapes) and all_finite
    header["all_finite"] = all_finite
    _save(out / "precision.json", header)
    _save(progress_path, {"protocol": protocol["protocol"],
                          "pbs_jobid": os.environ["PBS_JOBID"],
                          "phase": "complete" if header["complete"] else "nonfinite_failure",
                          "completed_shapes": header["completed_shapes"],
                          "expected_shapes": len(shapes), "all_finite": all_finite})
    if not header["complete"]:
        raise RuntimeError("Precision probe did not complete six finite shapes")


if __name__ == "__main__":
    main()
