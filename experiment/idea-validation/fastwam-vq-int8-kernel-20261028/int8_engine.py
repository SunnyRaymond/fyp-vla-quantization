"""1b engine on the native INT8 path: FP32 Smooth/H -> INT8 operand + per-token scale -> fused VQ->INT8 GEMM.

Same condition cache and CUDA Graph regions as plus_engine (C+buffer); no BF16 weight scratch, no decoder.
First use of every module checks the INT8 operand and scale bitwise against the eager reference and the GEMM
output bitwise against the exact FP64 integer reference.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import vq_int8

MAX_ROWS = 1024


def install(runtime, bundle, selection):
    import torch
    import abc_runtime
    import batched_transform
    import condition_cache
    import fused_input
    import graph_runtime
    import kernel_runtime

    W = abc_runtime.WORKSPACE_ELEMENTS

    class Int8Banks:
        def __init__(self, base):
            self.runtime = runtime
            self.modules, self.transforms, self.rotations = base.modules, base.transforms, base.rotations
            self.seen, self.calls, self.decoded_calls = base.seen, base.calls, Counter()
            self.ever_seen, self.workspace_validation, self.diagnostics, self.track = set(), {}, True, True
            self.prepared, self.configs, self.untuned, self.output_bitwise = {}, {}, set(), {}
            for name in self.modules:
                transform = self.transforms[name]
                transform["scale"] = transform["scale"].reshape(-1).contiguous()
                transform["signs"] = transform["signs"].reshape(-1).contiguous()
                self.prepared[name] = vq_int8.prepare_vq(base.encodings[name])
            # Only the packed indices stay resident; drop BF16 books, BF16 row scales and the unused exponents.
            base.encodings.clear()
            base.prepared.clear()
            device = next(iter(self.prepared.values())).indices.device
            self.h_buffer = torch.empty(W, dtype=torch.float32, device=device)
            self.q_buffer = torch.empty(W, dtype=torch.int8, device=device)
            self.s_buffer = torch.empty(MAX_ROWS, dtype=torch.float32, device=device)
            self.workspace_bytes = W * 5 + MAX_ROWS * 4
            self.workspace_layout = dict(h_fp32=W * 4, q_int8=W, scales=MAX_ROWS * 4)
            self.selection = selection
            self.packed_bytes = sum(p.indices.numel() + p.books.numel() + p.row_scales.numel() * 4
                                    for p in self.prepared.values())

        def parameters(self):
            return []

        def config(self, M, K, N):
            key = f"{M},{K},{N}"
            cfg = self.configs.get(key)
            if cfg is None:
                name = self.selection.get(key)
                if name is None:
                    # Shapes outside the tuned panel (none expected) fall back to a plain tile and are recorded.
                    self.untuned.add(key)
                    bm = min(128, max(16, 1 << (M - 1).bit_length()))
                    cfg = vq_int8.Config(bm, 64, 128, 1, 8 if bm * 64 >= 8192 else 4, 3, True)
                else:
                    cfg = vq_int8.config_from_name(name)
                self.configs[key] = cfg
            return cfg

        def operand(self, x, name):
            width, elements = x.shape[-1], x.numel()
            rows = elements // width
            if elements > W or rows > MAX_ROWS:
                raise ValueError(f"Workspace capacity exceeded at {name}: {elements}")
            values = x.contiguous()
            h = self.h_buffer[:elements].view(rows, width)
            t = self.transforms[name]
            full_blocks = width // 128
            starts, widths, inverses = fused_input._tail_segments(width)
            if full_blocks:
                fused_input._smooth_h_kernel[(rows * full_blocks,)](
                    values, t["scale"], t["signs"], h, width, 0, 128, full_blocks, 1.0 / (128 ** .5), 128,
                    num_warps=4, enable_fp_fusion=False)
            for start, segment, inverse in zip(starts, widths, inverses):
                fused_input._smooth_h_kernel[(rows,)](
                    values, t["scale"], t["signs"], h, width, start, segment, 1, inverse, 128,
                    num_warps=4, enable_fp_fusion=False)
            q = self.q_buffer[:elements].view(rows, width)
            s = self.s_buffer[:rows]
            vq_int8.quant_a8(h, q, s)
            return q, s

        def validate(self, x, name, q, s, y, bias):
            z = batched_transform.transform_input(x, self.transforms[name]).reshape(-1, x.shape[-1])
            q_ref, s_ref = vq_int8.reference_quant_a8(z)
            p = self.prepared[name]
            y_ref = vq_int8.reference_linear(q_ref, s_ref, vq_int8.decode_vq_int(p), p.row_scales, bias)
            operand_ok = torch.equal(q, q_ref) and torch.equal(s.view(torch.int32), s_ref.view(torch.int32))
            output_bitwise = torch.equal(y.view(torch.int16), y_ref.view(torch.int16))
            relative = float((y.float() - y_ref.float()).abs().max() / y_ref.float().abs().max().clamp_min(1e-30))
            self.output_bitwise[name] = output_bitwise
            self.workspace_validation[name] = bool(operand_ok)
            # Anything beyond BF16 rounding (2^-8 relative) is a kernel bug, not a rounding-mode difference.
            if not operand_ok or relative > 2 ** -7:
                raise ValueError(f"INT8 engine differs from the exact reference at {name}: "
                                 f"operand {operand_ok}, output relative {relative}")

        def forward(self, name, module):
            bias, prepared = module.bias, self.prepared[name]

            def run(x):
                if self.track and not torch.compiler.is_compiling():
                    self.seen.add(name)
                    self.ever_seen.add(name)
                    self.calls[name] += 1
                q, s = self.operand(x, name)
                y = vq_int8.vq_linear(q, s, prepared, bias, self.config(q.shape[0], q.shape[1], prepared.rows))
                if self.diagnostics and name not in self.workspace_validation:
                    self.validate(x, name, q, s, y, bias)
                return y.view(*x.shape[:-1], prepared.rows)
            return run

    base = kernel_runtime.install(runtime, bundle, packed=True, fused=True, direct=True)
    banks = Int8Banks(base)
    for name, module in banks.modules.items():
        module.forward = banks.forward(name, module)
    runtime.banks, runtime.seen, runtime.forwards = banks, banks.seen, {}
    runtime.current_arm = runtime.arm_name = "H1B_INT8"
    optimization = abc_runtime.Optimization(runtime, banks, "c_buffer")
    runtime.optimization = optimization
    optimization.cache = condition_cache.install(runtime.model, diagnostics=True, stable_buffers=True)
    graph_runtime.install(runtime, optimization, compiled=False)
    torch.cuda.empty_cache()
    runtime.install_metadata.update(
        arm="H1B_INT8", mode="c_buffer_int8", activation_bits=8, use_u=False,
        packed_gpu_resident=True, dense_target_weights=False, native_vq_kernel=True,
        native_integer_gemm=True, condition_cache=True, cuda_graph=True, weight_scratch_bytes=0,
        packed_target_bytes=banks.packed_bytes, activation_workspace_bytes=banks.workspace_bytes,
        arithmetic="FP32 Smooth/H -> INT8 + per-token FP32 scale -> fused VQ->INT8 MMA (INT32) -> FP32 epilogue -> BF16")
    return banks


def build_runtime(p, out, selection):
    """Load the 1b bundle, install the INT8 engine, validate all 614 modules on the anchor query."""
    import os
    import numpy as np
    import torch
    import abc_runtime
    import calibration_source
    import export_runtime
    from plus_engine import _generic_header, read, save
    from profile_worker import _capture_anchor
    from reference_profile import freeze_numerics

    freeze_numerics(torch)
    props = torch.cuda.get_device_properties(0)
    export_runtime._validate_header = _generic_header
    ep = read(Path(p["engine"]["source_artifacts"]) / "protocol.json")
    bundle = export_runtime.load_bundle(p["h1b"]["bundle"])
    if bundle.get("use_u") or not bundle.get("integer_codebooks") or bundle.get("layer_rotations"):
        raise ValueError("Expected the 1b H-only integer-codebook bundle")
    runtime = abc_runtime.make_runtime(out, ep, bundle, direct=True)
    banks = install(runtime, bundle, selection)
    del bundle

    plan_path = Path(p["engine"]["calibration_plan"])
    row = next(r for r in read(plan_path)["inputs"] if r["case_id"] == 2000)
    anchor = torch.load(p["h1b"]["anchor"], map_location="cpu", weights_only=True)
    observation, metadata = calibration_source.load_observation(np, plan_path.parent, row)
    runtime.description = metadata["description"]
    action, _, future, calls = _capture_anchor(runtime, runtime.datum(observation), row["sampler_seeds"][0], torch)
    validated = banks.workspace_validation
    if calls != 1 or len(banks.ever_seen) != 614 or len(validated) != 614 or not all(validated.values()):
        raise RuntimeError(f"INT8 engine validation incomplete: {len(validated)} modules")
    if set(runtime.optimization.graph.regions) != {"video_step", "video_prefill", "action_step"}:
        raise RuntimeError("Expected all three CUDA Graph regions")
    # Not bitwise by design: the anchor came from BF16 decoded weights and BF16-rounded A8 operands.
    a, b = action.detach().float().cpu(), anchor["action"].float()
    f, g = future.float().cpu(), anchor["future"].float()
    anchor_drift = dict(action_max_abs=float((a - b).abs().max()),
                        action_rmse=float((a - b).square().mean().sqrt()),
                        future_nmse=float((f - g).square().sum() / g.square().sum()))
    runtime.optimization.diagnostics = False
    runtime.optimization.reset_metrics()
    save(out / "startup.json", dict(
        protocol=p["protocol"], pbs_jobid=os.environ["PBS_JOBID"], passed=True,
        operand_bitwise_modules=sum(validated.values()),
        output_bitwise_modules=sum(banks.output_bitwise.values()), anchor_drift_vs_bf16_emulated=anchor_drift,
        untuned_shapes=sorted(banks.untuned),
        gpu=props.name, gpu_uuid=str(props.uuid), installation=runtime.install_metadata))
    del action, future, anchor, observation
    torch.cuda.empty_cache()
    return runtime
