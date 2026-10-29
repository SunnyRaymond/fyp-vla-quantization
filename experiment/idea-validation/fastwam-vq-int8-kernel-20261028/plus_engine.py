"""C+decoder+buffer engine with configurable activation bits and optional U, plus the Plus episode loop."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


@triton.jit
def _a8_kernel(x_ptr, out_ptr, K: tl.constexpr, BLOCK: tl.constexpr):
    # fused_input._a4_kernel with qmax 7 -> 127. The reciprocal-multiply scale matches PyTorch's
    # tensor / python-scalar path; fp32(1/127) == 1.0f/127.0f.
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    mask = lane < K
    x = tl.load(x_ptr + row * K + lane, mask, other=0.0).to(tl.float32)
    row_max = tl.max(tl.abs(x), axis=0)
    scale = row_max * (1.0 / 127.0)
    safe_scale = tl.where(scale > 0.0, scale, 1.0)
    scaled = tl.div_rn(x, safe_scale)
    lower = tl.floor(scaled)
    fraction = scaled - lower
    lower_i = lower.to(tl.int32)
    round_up = (fraction > 0.5) | ((fraction == 0.5) & ((lower_i & 1) != 0))
    rounded = lower + round_up.to(tl.float32)
    rounded = libdevice.copysign(rounded, scaled)
    quantized = tl.minimum(tl.maximum(rounded, -127.0), 127.0)
    decoded = quantized * scale
    tl.store(out_ptr + row * K + lane, decoded.to(tl.bfloat16), mask)


def reference_operand(x, transform, rotations, stream, use_u, qmax):
    """Eager arithmetic of the calibration/fixed-input path (A4: export_runtime.a4; A8: same with 127)."""
    import torch
    import batched_transform
    z = batched_transform.transform_input(x, transform)
    if use_u:
        z = rotations.apply_after_h(z, stream)
    z = z.float()
    scale = z.abs().amax(-1, keepdim=True) / qmax
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    return (torch.round(z / safe).clamp(-qmax, qmax) * scale).to(torch.bfloat16)


def _generic_header(bundle):
    """Non-Q0 bundles (e.g. 1b integer codebooks): same container, any source job."""
    import export_runtime
    if (bundle.get("format") != export_runtime.FORMAT
            or len(bundle.get("quantized", {})) != export_runtime.MODULE_COUNT
            or set(bundle.get("rotations", {})) != {"action", "video"}):
        raise ValueError("Bundle is not a 614-module MXVQ inference bundle with action/video rotations")
    excluded = set(bundle["excluded_model_state_keys"])
    if not excluded or excluded.intersection(bundle["model_state"]):
        raise ValueError("Bundle retains a quantized dense weight or lacks its binding")


def install(runtime, bundle, arm, activation_bits, use_u):
    """abc_runtime.install(mode='c_buffer') with the activation kernel and U step made configurable."""
    import torch
    import torch.nn.functional as F
    import abc_runtime
    import condition_cache
    import fused_input
    import graph_runtime
    import kernel_runtime
    import layer_runtime
    from rotation_bank import stream_for

    W = abc_runtime.WORKSPACE_ELEMENTS
    qmax = {4: 7, 8: 127}[activation_bits]
    kernel = fused_input._a4_kernel if activation_bits == 4 else _a8_kernel

    class Banks(layer_runtime.LayerDecodeBanks):
        def __init__(self, base):
            super().__init__(runtime, base, "layer_decode_fp32")
            # H/U are dead before weight decode, so they borrow the weight scratch; A stays separate.
            self.workspace_layout = abc_runtime.workspace_layout(self.scratch_bytes, True)
            scratch = self.scratch.view(torch.float32)
            self.h_buffer, self.u_buffer = scratch[:W], scratch[W:2 * W]
            self.a_buffer = torch.empty(W, dtype=torch.bfloat16, device=self.scratch.device)
            self.workspace_bytes = self.workspace_layout["independent_bytes"]
            self.decoder = layer_runtime.decode_prepared
            self.workspace_validation, self.ever_seen = {}, set()
            self.diagnostics, self.track = True, True
            self.streams = {name: stream_for(name) for name in self.modules}

        def operand(self, x, name):
            width, elements = x.shape[-1], x.numel()
            if elements > W:
                raise ValueError(f"Workspace capacity exceeded at {name}: {elements}")
            values = x.contiguous()
            h = self.h_buffer[:elements].view(values.shape)
            out = self.a_buffer[:elements].view(values.shape)
            rows = elements // width
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
            z = h
            if use_u and full_blocks:
                z = self.u_buffer[:elements].view(values.shape)
                full = full_blocks * 128
                torch.matmul(h[..., :full].reshape(*h.shape[:-1], full_blocks, 128),
                             self.rotations.matrix(self.streams[name]),
                             out=z[..., :full].view(*z.shape[:-1], full_blocks, 128))
                if full < width:
                    z[..., full:].copy_(h[..., full:])
            block = triton.next_power_of_2(width)
            kernel[(rows,)](z, out, width, block, num_warps=8 if block >= 8192 else 4, enable_fp_fusion=False)
            if self.diagnostics and name not in self.workspace_validation:
                expected = reference_operand(x, self.reference_transforms[name], self.rotations,
                                             self.streams[name], use_u, qmax)
                equal = torch.equal(out.view(torch.int16), expected.view(torch.int16))
                self.workspace_validation[name] = bool(equal)
                if not equal:
                    raise ValueError(f"A{activation_bits} engine operand differs from eager at {name}")
            return out

        def forward(self, name, module):
            bias, prepared, scratch = module.bias, self.prepared[name], self.scratch

            def run(x):
                if self.track and not torch.compiler.is_compiling():
                    self.seen.add(name)
                    self.ever_seen.add(name)
                    self.calls[name] += 1
                    self.decoded_calls[name] += 1
                return F.linear(self.operand(x, name), self.decoder(prepared, scratch), bias)
            return run

    base = kernel_runtime.install(runtime, bundle, packed=True, fused=True, direct=True)
    banks = Banks(base)
    for name, module in banks.modules.items():
        module.forward = banks.forward(name, module)
    runtime.banks, runtime.seen, runtime.forwards = banks, banks.seen, {}
    runtime.current_arm = runtime.arm_name = arm
    optimization = abc_runtime.Optimization(runtime, banks, "c_buffer")
    runtime.optimization = optimization
    optimization.cache = condition_cache.install(runtime.model, diagnostics=True, stable_buffers=True)
    graph_runtime.install(runtime, optimization, compiled=False)
    runtime.install_metadata.update(
        arm=arm, mode="c_buffer", activation_bits=activation_bits, use_u=use_u,
        packed_gpu_resident=True, dense_target_weights=False, native_vq_kernel=False,
        native_integer_gemm=False, condition_cache=True, cuda_graph=True,
        arithmetic=f"FP32 Smooth/H{'/U' if use_u else ''} + A{activation_bits} + layer BF16 decode + BF16 cuBLAS")
    return banks


def build_runtime(p, arm, cfg, out):
    """Load the bundle once, install the engine, reuse decoder choices, then pass the eager anchor bitwise."""
    import numpy as np
    import torch
    import abc_runtime
    import calibration_source
    import decoder_runtime
    import export_runtime
    import fused_vq
    from profile_worker import _capture_anchor
    from reference_profile import freeze_numerics

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible GPU per worker is required")
    props = torch.cuda.get_device_properties(0)
    freeze_numerics(torch)
    if cfg["bundle"] != p["engine"]["default_bundle"]:
        export_runtime._validate_header = _generic_header
    ep = read(Path(p["engine"]["source_artifacts"]) / "protocol.json")
    bundle = export_runtime.load_bundle(cfg["bundle"])
    runtime = abc_runtime.make_runtime(out, ep, bundle, direct=True)
    banks = install(runtime, bundle, arm, cfg["activation_bits"], cfg["use_u"])
    del bundle
    # Decoder configs only change speed; every candidate was bitwise-checked against the reference decoder.
    tuning = read(Path(p["engine"]["source_results"]) / "tuning.json")
    decisions = {tuple(r["shape"]): r["selected"] for r in tuning["shapes"]}
    missing = {(q.rows, q.width) for q in banks.prepared.values()} - set(decisions)
    if missing:
        raise RuntimeError(f"No decoder decision for shapes {sorted(missing)}")
    banks.decoder = decoder_runtime.DecoderDispatcher(
        decisions, tuning, out / "reused_decoder.json", fused_vq._decode_kernel)

    plan_path = Path(p["engine"]["calibration_plan"])
    row = next(r for r in read(plan_path)["inputs"] if r["case_id"] == 2000)
    anchor = torch.load(cfg["anchor"], map_location="cpu", weights_only=True)
    seed = row["sampler_seeds"][0]
    if anchor.get("sampler_seed", seed) != seed:
        raise RuntimeError("Anchor sampler seed differs from the case-2000 plan row")
    observation, metadata = calibration_source.load_observation(np, plan_path.parent, row)
    runtime.description = metadata["description"]
    action, _, future, calls = _capture_anchor(runtime, runtime.datum(observation), seed, torch)
    action_error = float((action.detach().float().cpu() - anchor["action"].float()).abs().max())
    future_error = float((future.float().cpu() - anchor["future"].float()).abs().max())
    validated = banks.workspace_validation
    if (action_error != 0 or future_error != 0 or calls != 1 or len(banks.ever_seen) != 614
            or len(validated) != 614 or not all(validated.values())):
        raise RuntimeError(f"Eager anchor parity failed: action {action_error}, future {future_error}")
    if set(runtime.optimization.graph.regions) != {"video_step", "video_prefill", "action_step"}:
        raise RuntimeError("Expected all three CUDA Graph regions")
    runtime.optimization.diagnostics = False
    runtime.optimization.reset_metrics()
    save(out / "startup.json", dict(
        protocol=p["protocol"], arm=arm, pbs_jobid=os.environ["PBS_JOBID"], passed=True,
        anchor=cfg["anchor"], action_max_abs=action_error, future_max_abs=future_error,
        operand_bitwise_modules=sum(validated.values()), gpu=props.name, gpu_uuid=str(props.uuid),
        reused_decoder=tuning["pbs_jobid"], installation=runtime.install_metadata))
    del action, future, anchor, observation
    torch.cuda.empty_cache()
    return runtime


def episode(runtime, row, index, arm):
    """Same seeds, settling, caps and 10-action replans as the historical decoded-dense Plus episodes."""
    import numpy as np
    import torch
    import runner as pilot
    started = time.monotonic()
    obs = runtime.task(row, index)
    cap = 700 if row["suite"] == "libero_10" else 400
    steps, replans, success, call_seconds = 0, 0, False, []
    while steps < cap and not success:
        datum = runtime.datum(obs)
        torch.cuda.synchronize()
        begin = time.monotonic()
        action, _ = runtime.infer(datum, pilot.seed_for(index, replans), measure=False)
        torch.cuda.synchronize()
        call_seconds.append(time.monotonic() - begin)
        commands = runtime.command(action)
        if not np.isfinite(commands).all():
            raise RuntimeError("Non-finite controller commands")
        for command in commands[:min(10, cap - steps)]:
            obs, _, done, _ = runtime.env.step(command.tolist())
            steps += 1
            if done:
                success = True
                break
        replans += 1
    runtime.env.close()
    runtime.env = None
    return dict(
        arm=arm, variant_id=str(row["variant_id"]), index=index, suite=row["suite"],
        dimension=row["dimension"], task_id=int(row["task_id"]), success=success,
        termination="success" if success else "control_step_timeout", steps=steps, replans=replans,
        env_seed=pilot.seed_for(index, environment=True), sampler_seed=pilot.seed_for(index),
        episode_seconds=time.monotonic() - started, policy_call_seconds=call_seconds,
        pbs_jobid=os.environ["PBS_JOBID"])
