"""The INT4 tensor-core path of the integer LAQ runtime: int4_gemm.cu (PTX from nvcc) loaded through the CUDA driver.

CudaOps replaces the quantize / unpack / INT8 GEMM / epilogue kernels of TritonOps by three kernels: plan4 (row layout of
the operand), quant4 (packed 4-bit codes) and gemm4 (INT4 GEMM with the dequantization fused). The weights stay packed;
there is no INT8 buffer. Kernels launch on torch's current stream, so they are captured by CUDA graphs like any other.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch

import laq

BM, BN, KCODES = 32, 128, 512


class Module:
    def __init__(self, ptx):
        torch.empty(1, device="cuda")
        self.cuda = ctypes.CDLL("libcuda.so.1")
        self.cuda.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p, ctypes.c_void_p,
                                                                                      ctypes.c_void_p]
        self.module = ctypes.c_void_p()
        self._check(self.cuda.cuModuleLoadData(ctypes.byref(self.module), Path(ptx).read_bytes() + b"\0"), "load")
        self.functions = {}

    @staticmethod
    def _check(code, what):
        if code:
            raise RuntimeError(f"CUDA driver error {code} ({what})")

    def launch(self, name, grid, block, args):
        f = self.functions.get(name)
        if f is None:
            f = self.functions[name] = ctypes.c_void_p()
            self._check(self.cuda.cuModuleGetFunction(ctypes.byref(f), self.module, name.encode()), name)
        values = [ctypes.c_void_p(a.data_ptr()) if torch.is_tensor(a) else
                  ctypes.c_float(a) if isinstance(a, float) else ctypes.c_int32(a) for a in args]
        params = (ctypes.c_void_p * len(values))(*[ctypes.addressof(v) for v in values])
        gx, gy = grid
        self._check(self.cuda.cuLaunchKernel(f, gx, gy, 1, block, 1, 1, 0, torch.cuda.current_stream().cuda_stream,
                                             params, None), name)


class CudaOps:
    name = "int4"

    def __init__(self, ptx=None):
        self.m = Module(ptx or os.environ.get("INT4_PTX") or Path(__file__).with_name("int4_gemm.ptx"))
        self.identity = {}

    def operand(self, r, idx):
        """r FP32 [T, K] contiguous -> (codes uint8 [Mp, K / 2], scales [Mp], dst int32 [Mp], paired tiles, T)."""
        T, K = r.shape
        if K % KCODES or T > 1024:
            raise ValueError(f"No INT4 operand for {T} x {K}")
        R = 0 if idx is None else int(idx.numel())
        p16 = (R + 7) // 8
        mp = -(-(16 * p16 + T - R) // BM) * BM
        if R:
            plan = torch.empty((3, mp), dtype=torch.int32, device=r.device)
            self.m.launch("plan4", (1, 1), 1024, [idx, R, T, plan[0], plan[1], plan[2], mp])
        else:
            plan = self.identity.get((T, r.device))
            if plan is None:
                plan = torch.full((3, mp), -1, dtype=torch.int32, device=r.device)
                plan[1] = 0
                plan[0, :T] = plan[2, :T] = torch.arange(T, dtype=torch.int32, device=r.device)
                self.identity[(T, r.device)] = plan
        codes = torch.empty((mp, K // 2), dtype=torch.uint8, device=r.device)
        scales = torch.empty(mp, dtype=torch.float32, device=r.device)
        self.m.launch("quant4", (mp, 1), 256, [r, plan[0], plan[1], codes, scales, K, float(laq.A_CLIP)])
        return codes, scales, plan[2], p16, T

    def gemm(self, codes, scales, dst, p16, T, layer, scratch, dtype):
        if dtype != torch.bfloat16:
            raise TypeError("The kernel writes BF16")
        if not layer.twos or layer.N % BN or layer.K % KCODES:
            raise ValueError(f"{layer.name}: no INT4 GEMM for {layer.N} x {layer.K}")
        y = torch.empty((T, layer.N), dtype=torch.bfloat16, device=codes.device)
        bias = layer.bias if layer.bias is not None else layer.rs
        self.m.launch("gemm4", (layer.N // BN, codes.shape[0] // BM), 256,
                      [codes, layer.packed, scales, layer.rs, bias, dst, y, layer.N, layer.K, p16,
                       int(layer.bias is not None)])
        return y

    def kvq(self, r, T, heads, kind, idx):
        import int4_kernels
        flags = torch.zeros(T, dtype=torch.int8, device=r.device)
        if idx is not None:
            flags.index_fill_(0, idx, 1)
        return int4_kernels.kvq(r, flags, heads, laq.KV_CLIP, kind == "K")
