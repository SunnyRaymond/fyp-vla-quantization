"""One fixed-input comparison of native W4A4 with an independent FP32 reference.

Run only inside an approved PBS compute allocation. This performs inference on
one manifest row and 30 reset-settling no-op steps; it does not run an episode.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from runner import ROOT, Runtime, guard, load_manifest, save, seed_for


GROUP_SIZE = 128


def _float_reference_linear(torch, module, x):
    """Independent W4A4 dequantize + FP32 GEMM; avoids quantization helpers."""
    features = module.in_features
    shape = tuple(x.shape)
    rows = x.reshape(-1, features).float()

    maximum = rows.abs().amax(dim=1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / 7.0, torch.zeros_like(maximum))
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    activation_codes = torch.round(rows / safe_scale).clamp(-7, 7)
    activation = activation_codes * scale

    packed = module._quant_packed_weight.to(torch.int32)
    low = packed & 15
    high = (packed >> 4) & 15
    nibble = torch.stack((low, high), dim=-1).flatten(start_dim=-2)
    signed = torch.where(nibble >= 8, nibble - 16, nibble).float()
    weight_codes = signed[:, :features]
    weight_scales = module._quant_weight_scales.float().repeat_interleave(
        GROUP_SIZE, dim=1
    )[:, :features]
    weight = weight_codes * weight_scales

    output = activation @ weight.T
    if module.bias is not None:
        output = output + module.bias.float()
    return output.reshape(*shape[:-1], module.out_features).to(torch.bfloat16)


def _action_metrics(torch, action, reference):
    delta = action.float() - reference.float()
    first10 = delta[:10]
    return {
        "finite": bool(torch.isfinite(action).all()),
        "shape": list(action.shape),
        "rmse_first10_vs_bf16": float(first10.square().mean().sqrt()),
        "motor_rmse_first10_vs_bf16": float(first10[:, :6].square().mean().sqrt()),
        "gripper_rmse_first10_vs_bf16": float(first10[:, 6].square().mean().sqrt()),
        "rmse_first32_vs_bf16": float(delta.square().mean().sqrt()),
        "max_abs_first10_vs_bf16": float(first10.abs().max()),
        "max_abs_first32_vs_bf16": float(delta.abs().max()),
    }


def main() -> None:
    guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=0, help="frozen manifest row; default: 0")
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifest.json")
    parser.add_argument("--out", type=Path, required=True, help="diagnostic output directory")
    args = parser.parse_args()

    manifest, rows = load_manifest(args.manifest)
    if not 0 <= args.index < len(rows):
        raise ValueError(f"index must be in [0, {len(rows) - 1}]")
    row = rows[args.index]
    args.out.mkdir(parents=True, exist_ok=True)

    runtime = Runtime(args.out)
    previous_tf32 = None
    actions = {}
    query_walls = {}
    query_counters = {}
    gripper_commands = {}
    try:
        if runtime.q.group_size != GROUP_SIZE:
            raise RuntimeError('Independent reference requires the frozen G128 weight recipe')
        observation = runtime.task(row, args.index)
        datum = runtime.datum(observation)
        sampler_seed = seed_for(args.index)

        previous_tf32 = runtime.torch.backends.cuda.matmul.allow_tf32
        runtime.torch.backends.cuda.matmul.allow_tf32 = False

        def counters():
            return {
                "integer_gemm_calls": runtime.q._integer_gemm_calls,
                "native_int4_gemm_calls": runtime.q._native_int4_gemm_calls,
                "kv_packed_prefills": runtime.q._kv_packed_prefills,
                "kv_layer_reads": runtime.q._kv_layer_reads,
            }

        def run_query(label: str, arm: str) -> None:
            runtime.arm(arm)
            before = counters()
            runtime.torch.cuda.synchronize()
            started = time.perf_counter()
            action, _ = runtime.infer(datum, sampler_seed, measure=False)
            runtime.torch.cuda.synchronize()
            query_walls[label] = time.perf_counter() - started
            actions[label] = action.detach().float().cpu()
            commands = runtime.command(action)
            gripper_commands[label] = commands[:10, -1].tolist()
            after = counters()
            query_counters[label] = {
                "before": before,
                "after": after,
                "delta": {key: after[key] - before[key] for key in before},
            }

        try:
            run_query("bf16", "bf16")
            run_query("w4a8", "w4a8")
            run_query("w4a4_native", "w4a4")

            # Mark this arm as a diagnostic-only FP32 reference, not real quant.
            original_linear = runtime.q._linear
            runtime.q._linear = lambda module, x: _float_reference_linear(
                runtime.torch, module, x
            )
            try:
                run_query("diagnostic_reference", "w4a4")
            finally:
                runtime.q._linear = original_linear
        finally:
            runtime.torch.backends.cuda.matmul.allow_tf32 = previous_tf32

        if query_counters['w4a8']['delta']['integer_gemm_calls'] <= 0:
            raise RuntimeError('W4A8 query did not execute integer GEMM')
        if query_counters['w4a4_native']['delta']['native_int4_gemm_calls'] <= 0:
            raise RuntimeError('W4A4 query did not execute native INT4 GEMM')
        if any(query_counters['diagnostic_reference']['delta'].values()):
            raise RuntimeError('Diagnostic floating reference unexpectedly used an integer/cache path')

        bf16 = actions["bf16"]
        comparisons = {
            label: _action_metrics(runtime.torch, action, bf16)
            for label, action in actions.items()
            if label != "bf16"
        }
        native_reference_delta = (
            actions["w4a4_native"] - actions["diagnostic_reference"]
        )
        native_reference_rmse_first10 = float(
            native_reference_delta[:10].square().mean().sqrt()
        )
        native_reference_rmse_first32 = float(
            native_reference_delta.square().mean().sqrt()
        )
        native_reference_max_abs = float(native_reference_delta.abs().max())

        result = {
            "diagnostic": "fixed-input-a4-kernel-vs-fp32-reference-v1",
            "index": args.index,
            "variant_id": str(row["variant_id"]),
            "suite": row["suite"],
            "dimension": row["dimension"],
            "task_id": row["task_id"],
            "environment_seed": seed_for(args.index, environment=True),
            "sampler_seed": sampler_seed,
            "input": "one reset observation after runner.Runtime.task 30 no-op settling steps",
            "query_count": 4,
            "episode_counted": False,
            "post_query_env_steps": 0,
            "normalized_action_comparisons_vs_bf16": comparisons,
            "normalized_actions_32x7": {
                label: action.tolist() for label, action in actions.items()
            },
            "gripper_commands_first10": gripper_commands,
            "gripper_flips_vs_bf16_first10": {
                label: sum(a != b for a, b in zip(commands, gripper_commands["bf16"]))
                for label, commands in gripper_commands.items()
                if label != "bf16"
            },
            "runtime_counter_deltas": query_counters,
            "native_vs_independent_float_reference": {
                "finite": bool(
                    runtime.torch.isfinite(actions["w4a4_native"]).all()
                    and runtime.torch.isfinite(actions["diagnostic_reference"]).all()
                ),
                "rmse_first10": native_reference_rmse_first10,
                "rmse_first32": native_reference_rmse_first32,
                "max_abs_first32": native_reference_max_abs,
                "classification": "diagnostic_reference_not_real_quant",
            },
            "query_wall_seconds_diagnostic_only": query_walls,
            "timing_claim": "none; measure=False and FP32 reference may be slower",
            "reference": {
                "activation": "per-row FP32 absmax/7, round, clamp[-7,7], dequantize",
                "weight": "decode packed low/high nibbles independently; signed two's complement; BF16 G128 scales",
                "gemm": "FP32 matmul, TF32 disabled, FP32 bias, BF16 output",
            },
            "interpretation_limit": (
                "A nonzero full-query delta alone is not a kernel bug: FP32 accumulation "
                "and per-layer BF16 rounding differ from native integer accumulation."
            ),
            "manifest_population_counts": manifest.get("population_counts"),
        }
        save(args.out / "diagnose_a4.json", result)
        print(f"DIAGNOSTIC_COMPLETE index={args.index} output={args.out / 'diagnose_a4.json'}", flush=True)
    finally:
        if runtime.env is not None:
            runtime.env.close()
        runtime.q.close()


if __name__ == "__main__":
    main()
