"""Frozen arithmetic with shared activation workspace and optional execution reuse."""
from __future__ import annotations

from collections import Counter


MODES = ("runtime_a", "runtime_b", "runtime_c", "runtime_compile")
# The frozen panel has at most 294 rows and 14336 input features.
WORKSPACE_ELEMENTS = 294 * 14336


def make_runtime(out, p, bundle, direct=True):
    import layer_runtime

    return layer_runtime.make_runtime(out, p, bundle, direct=direct)


class Optimization:
    def __init__(self, runtime, banks, mode):
        self.runtime, self.banks, self.mode = runtime, banks, mode
        self.cache = self.graph = None
        self.compiled = {"enabled": False, "regions": {}}
        self._diagnostics = True

    @property
    def diagnostics(self):
        return self._diagnostics

    @diagnostics.setter
    def diagnostics(self, value):
        self._diagnostics = bool(value)
        self.banks.diagnostics = bool(value)
        if self.cache is not None:
            self.cache.diagnostics = bool(value)
            if not value:
                for slot in self.cache._slots.values():
                    slot["input"] = None

    def reset_metrics(self):
        self.banks.calls.clear()
        self.banks.seen.clear()
        self.banks.decoded_calls.clear()
        if self.graph is not None:
            self.graph.reset_metrics()

    def report(self):
        graph = self.graph.report() if self.graph is not None else {}
        logical_replays = graph.get("logical_replayed_target_calls", 0)
        modules = set(self.banks.ever_seen)
        return dict(
            mode=self.mode, workspace_bytes=self.banks.workspace_bytes,
            operand_validation={
                "checked_modules": len(self.banks.workspace_validation),
                "bitwise_passed": sum(self.banks.workspace_validation.values()),
                "failed": [k for k, v in self.banks.workspace_validation.items() if not v],
            },
            condition_cache=self.cache.metadata() if self.cache is not None else {},
            graph=graph, compiled=self.compiled,
            logical_coverage=dict(modules=len(modules),
                target_linear_calls=sum(self.banks.calls.values()) + logical_replays,
                python_target_linear_calls=sum(self.banks.calls.values()),
                graph_replayed_target_linear_calls=logical_replays),
        )


def install(runtime, bundle, mode="runtime_a"):
    if mode not in MODES:
        raise ValueError(mode)
    import torch
    import torch.nn.functional as F
    import triton
    import fused_input
    import kernel_runtime
    import layer_runtime
    from rotation_bank import stream_for

    class WorkspaceBanks(layer_runtime.LayerDecodeBanks):
        def __init__(self, base):
            super().__init__(runtime, base, "layer_decode_fp32")
            device = self.scratch.device
            self.h_buffer = torch.empty(WORKSPACE_ELEMENTS, dtype=torch.float32, device=device)
            self.u_buffer = torch.empty_like(self.h_buffer)
            self.a_buffer = torch.empty(WORKSPACE_ELEMENTS, dtype=torch.bfloat16, device=device)
            self.workspace_bytes = WORKSPACE_ELEMENTS * 10
            self.workspace_validation = {}
            self.ever_seen = set()
            self.diagnostics = True
            self.track = True
            self.streams = {name: stream_for(name) for name in self.modules}

        def operand(self, x, name):
            width = x.shape[-1]
            elements = x.numel()
            if elements > WORKSPACE_ELEMENTS:
                raise ValueError(f"Frozen workspace capacity exceeded at {name}: {elements}")
            values = x.contiguous()
            h = self.h_buffer[:elements].view(values.shape)
            z = self.u_buffer[:elements].view(values.shape)
            operand = self.a_buffer[:elements].view(values.shape)
            rows = elements // width
            transform = self.transforms[name]
            full_blocks = width // 128
            starts, widths, inverses = fused_input._tail_segments(width)
            if full_blocks:
                fused_input._smooth_h_kernel[(rows * full_blocks,)](
                    values, transform["scale"], transform["signs"], h, width,
                    0, 128, full_blocks, 1.0 / (128 ** .5), 128,
                    num_warps=4, enable_fp_fusion=False,
                )
            for start, segment_width, inv in zip(starts, widths, inverses):
                fused_input._smooth_h_kernel[(rows,)](
                    values, transform["scale"], transform["signs"], h, width,
                    start, segment_width, 1, inv, 128,
                    num_warps=4, enable_fp_fusion=False,
                )
            full = full_blocks * 128
            if full:
                head = h[..., :full].reshape(*h.shape[:-1], full_blocks, 128)
                out = z[..., :full].view(*z.shape[:-1], full_blocks, 128)
                torch.matmul(head, self.rotations.matrix(self.streams[name]), out=out)
                if full < width:
                    z[..., full:].copy_(h[..., full:])
            else:
                z.copy_(h)
            block = triton.next_power_of_2(width)
            fused_input._a4_kernel[(rows,)](
                z, operand, width, block,
                num_warps=8 if block >= 8192 else 4, enable_fp_fusion=False,
            )
            if self.diagnostics and name not in self.workspace_validation:
                expected = super()._operand(x, name)
                equal = torch.equal(operand.view(torch.int16), expected.view(torch.int16))
                self.workspace_validation[name] = bool(equal)
                if not equal:
                    raise ValueError("Workspace A4 operand differs at " + name)
            return operand

        def forward(self, name, module):
            bias = module.bias
            operand, prepared, scratch = self.operand, self.prepared[name], self.scratch

            def run(x):
                if self.track and not torch.compiler.is_compiling():
                    self.seen.add(name)
                    self.ever_seen.add(name)
                    self.calls[name] += 1
                    self.decoded_calls[name] += 1
                qx = operand(x, name)
                weight = layer_runtime.decode_prepared(prepared, scratch)
                return F.linear(qx, weight, bias)

            return run

        def coverage_summary(self):
            row = super().coverage_summary()
            row.update(seen_count=len(self.ever_seen), workspace_bytes=self.workspace_bytes,
                       workspace_validated_modules=len(self.workspace_validation))
            return row

    base = kernel_runtime.install(runtime, bundle, packed=True, fused=True, direct=True)
    banks = WorkspaceBanks(base)
    for name, module in banks.modules.items():
        module.forward = banks.forward(name, module)
    runtime.banks, runtime.seen, runtime.forwards = banks, banks.seen, {}
    runtime.current_arm = runtime.arm_name = "reader_kl"
    optimization = Optimization(runtime, banks, mode)
    runtime.optimization = optimization
    if mode != "runtime_a":
        import condition_cache

        optimization.cache = condition_cache.install(runtime.model, diagnostics=True, stable_buffers=True)
    if mode in ("runtime_c", "runtime_compile"):
        import graph_runtime

        graph_runtime.install(runtime, optimization, compiled=mode == "runtime_compile")
    runtime.install_metadata.update(
        mode=mode, packed_gpu_resident=True, dense_target_weights=False,
        native_vq_kernel=False, layer_decode=True, native_integer_gemm=False,
        decoded_weight_scratch_bytes=banks.scratch_bytes,
        activation_workspace_bytes=banks.workspace_bytes,
        arithmetic="original FP32 Smooth/H/U/A4 + BF16 cuBLAS Linear",
        native_bf16_linear_gemm=True, condition_cache=mode != "runtime_a",
        cuda_graph=mode == "runtime_c", compiled=mode == "runtime_compile",
    )
    return banks
