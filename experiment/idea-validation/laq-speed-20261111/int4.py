"""Integer runtime of LAQ on Fast-WAM: the same quantizers as laq.py, executed on 4-bit operands.

  weights      stay packed at 4 bits (two codes per byte) with one FP32 scale per output row; the dense BF16 copies are
               freed. A GEMM unpacks its weight into one shared INT8 scratch and runs the cuBLASLt INT8 GEMM
               (torch._int_mm, INT32 accumulation, exact). `resident8` keeps the unpacked INT8 weights instead.
  activations  rotation as two or three small matrix products (Kronecker factors of the Hadamard transforms), then one
               kernel writes the 4-bit codes and the per-token scales. A refined token contributes two rows of codes
               (first pass in place, second pass appended), so the second 4-bit pass is the same GEMM on more rows.
  output       one epilogue kernel: (acc * token scale + acc of the second pass * its scale) * row scale + bias -> BF16.
  keys/values  head-dimension Hadamard as a matrix product, one quantize-dequantize kernel, Hadamard back.
  text sites   the context is the same at every step of a query: computed at the first video / action step and reused.

Engine.install replaces State.linear and State.kv; everything else of laq.py (selection, gaze, phases, graphs) is
unchanged. TorchOps is the reference arithmetic (any device); TritonOps runs the kernels of int4_kernels.py.
"""
from __future__ import annotations

from pathlib import Path

import torch

import laq
import methods as mt
import qops as q


# ---------------------------------------------------------------- rotations as small matrix products
class FastXf:
    """A rotation of the last dimension written as products along the axes of a view; plan rows are (pre, a, post, R):
    the operand is viewed as [T * pre, a, post] and every length-a vector v along the middle axis becomes v @ R."""

    def __init__(self, n, signs, plan):
        self.n, self.signs, self.plan = n, signs, plan

    @staticmethod
    def _fht(w, device):
        return q.fht(torch.eye(w, device=device))

    @classmethod
    def hadamard(cls, n, signs, device):
        k = 28 if n % 28 == 0 else 12 if n % 12 == 0 else 1
        m = n // k
        if m & (m - 1):
            raise ValueError(f"Width {n} has no Hadamard of the QuaRot family")
        b = 1 << ((m.bit_length() - 1 + 1) // 2)
        a = m // b
        plan = [(k * a, b, 1, cls._fht(b, device))]
        if a > 1:
            plan.append((k, a, b, cls._fht(a, device)))
        if k > 1:
            plan.append((1, k, m, q._base(k, device, torch.float32).t().contiguous()))
        return cls(n, signs, plan)

    @classmethod
    def from_xf(cls, xf, n, device):
        if isinstance(xf, mt.Had):
            return cls.hadamard(n, xf.s, device)
        if isinstance(xf, mt.QuarotO):
            eye = torch.eye(xf.h, device=device)
            return cls(n, None, [(xf.h, xf.d, 1, q.had(torch.eye(xf.d, device=device))), (1, xf.h, xf.d, q.had(eye))])
        raise TypeError(f"No fast form for {type(xf).__name__}")

    def apply(self, x):
        """x [T, n] any float dtype -> FP32 [T, n] contiguous."""
        T = x.shape[0]
        y = x.float()
        if self.signs is not None:
            y = y * self.signs
        for pre, a, post, r in self.plan:
            if post == 1:
                y = y.reshape(T * pre, a) @ r
            else:
                y = torch.matmul(r.t(), y.reshape(T * pre, a, post))
        return y.reshape(T, self.n).contiguous()


# ---------------------------------------------------------------- weights
class Layer:
    def __init__(self, name, bank, bias, resident8):
        if bank["kind"] != "sym" or bank["bits"] != 4 or bank["group"] != bank["shape"][1]:
            raise ValueError(f"{name}: expected per-row symmetric INT4 codes")
        self.name = name
        self.N, self.K = (int(v) for v in bank["shape"])
        self.packed = bank["packed"].contiguous()
        self.rs = bank["scales"].float().reshape(self.N).contiguous()
        self.bias = None if bias is None else bias.detach().float().contiguous()
        self.w8 = self.dense() if resident8 or self.K % 2 else None
        if self.w8 is not None and self.K % 2 == 0 and resident8:
            self.packed = None

    def dense(self):
        """INT8 [N, K] codes in [-8, 7] (reference unpack)."""
        return (q._unpack4(self.packed, self.K).to(torch.int16) - 8).to(torch.int8).contiguous()

    def bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.packed, self.w8, self.rs, self.bias) if t is not None)


# ---------------------------------------------------------------- arithmetic
class TorchOps:
    """Reference arithmetic: the quantizers of laq.py and an exact integer GEMM, on any device."""

    name = "torch"

    def quant(self, r, idx):
        m = r.abs().amax(1) * laq.A_CLIP
        s = torch.where(m == 0, torch.ones_like(m), m / 7)
        codes = torch.round(r / s[:, None]).clamp(-8, 7)
        if idx is None:
            return codes.to(torch.int8), s
        x = r.index_select(0, idx)
        m1 = x.abs().amax(1)
        s1 = torch.where(m1 == 0, torch.ones_like(m1), m1 / 7)
        q1 = torch.round(x / s1[:, None]).clamp(-8, 7)
        res = x - q1 * s1[:, None]
        m2 = res.abs().amax(1)
        s2 = torch.where(m2 == 0, torch.ones_like(m2), m2 / 7)
        q2 = torch.round(res / s2[:, None]).clamp(-8, 7)
        return (torch.cat((codes.index_copy(0, idx, q1), q2)).to(torch.int8),
                torch.cat((s.index_copy(0, idx, s1), s2)))

    def gemm(self, codes, scales, idx, layer, scratch, dtype):
        w = layer.w8 if layer.w8 is not None else layer.dense()
        acc = (codes.double() @ w.double().t()).float()
        T = codes.shape[0] - (0 if idx is None else idx.numel())
        t = acc[:T] * scales[:T, None]
        if idx is not None:
            t = t.index_add(0, idx, acc[T:] * scales[T:, None])
        y = t * layer.rs[None, :]
        if layer.bias is not None:
            y = y + layer.bias[None, :]
        return y.to(dtype)

    def kvq(self, r, T, heads, kind, idx):
        base, fine = (laq.k_base, laq.k_fine) if kind == "K" else (laq.v_base, laq.v_fine)
        return laq.refine(r.reshape(1, T, heads, -1), idx, base, fine).reshape(r.shape)


class TritonOps:
    name = "triton"

    def __init__(self):
        import int4_kernels
        self.k = int4_kernels

    def quant(self, r, idx):
        return self.k.quant(r, idx, laq.A_CLIP)

    def gemm(self, codes, scales, idx, layer, scratch, dtype):
        if dtype != torch.bfloat16:
            raise TypeError("The kernel epilogue writes BF16")
        rows, K = codes.shape
        N = layer.N
        if layer.w8 is not None:
            w = layer.w8
        else:
            w = self.k.unpack(layer.packed, scratch[:N * K].view(N, K))
        if rows > 16 and K % 8 == 0 and N % 8 == 0:
            acc = torch._int_mm(codes, w.t())
        else:  # FP32 holds these integer sums exactly (|sum| <= 64 K < 2^24)
            acc = (codes.float() @ w.float().t()).to(torch.int32)
        part = None
        T = rows
        if idx is not None:
            T = rows - idx.numel()
            part = torch.full((T,), -1, dtype=torch.int32, device=codes.device)
            part.index_copy_(0, idx, torch.arange(idx.numel(), dtype=torch.int32, device=codes.device))
        return self.k.epilogue(acc.contiguous(), scales, layer.rs, layer.bias, part, T)

    def kvq(self, r, T, heads, kind, idx):
        flags = torch.zeros(T, dtype=torch.int8, device=r.device)
        if idx is not None:
            flags.index_fill_(0, idx, 1)
        return self.k.kvq(r, flags, heads, laq.KV_CLIP, kind == "K")


# ---------------------------------------------------------------- engine
class Engine:
    def __init__(self, st, ops):
        self.st, self.ops = st, ops
        self.layers, self.xfs, self.text = {}, {}, {}
        self.scratch = None
        self.fd = q.fht(torch.eye(st.d, device=st.device))
        self.gemms = 0

    def install(self, modules, index_of, bank_dir, resident8=False):
        """Load the packed codes, free the dense weights and take over State.linear / State.kv."""
        st, largest = self.st, 0
        for name, module in modules.items():
            if mt.parse(name) is None:
                continue
            bank = torch.load(Path(bank_dir) / f"{index_of[name]:04d}.pt", map_location=st.device, weights_only=True)
            if bank["module"] != name:
                raise RuntimeError(f"Bank does not match {name}")
            layer = Layer(name, bank, module.bias, resident8)
            self.layers[name] = layer
            if layer.w8 is None:
                largest = max(largest, layer.N * layer.K)
            module.weight = torch.nn.Parameter(torch.empty(0, dtype=module.weight.dtype, device=st.device),
                                               requires_grad=False)
        if largest:
            self.scratch = torch.empty(largest, dtype=torch.int8, device=st.device)
        st.quant, st.inject = True, None
        st.shared, st.frozen = None, {}
        st.linear, st.kv, st.native = self.linear, self.kv, self
        return len(self.layers)

    def reset(self):
        self.text = {}
        self.st.shared = None

    def weight_bytes(self):
        return sum(layer.bytes() for layer in self.layers.values())

    # ---- text sites
    def _text_hit(self, key, x):
        st = self.st
        if st.graphing:
            if st.phase is not None and st.phase[0] in ("video", "action") and st.phase[1] == 0:
                return None
            return self.text[key][1]
        hit = self.text.get(key)
        if hit is not None and (hit[0] is x or (hit[0].shape == x.shape and torch.equal(hit[0], x))):
            return hit[1]
        return None

    # ---- operands
    def xf(self, name, xf, n):
        fast = self.xfs.get(name)
        if fast is None:
            fast = self.xfs[name] = FastXf.from_xf(xf, n, self.st.device)
        return fast

    def linear(self, x, module, xf, segs, name, stream=None):
        st = self.st
        st.seen.add(name)
        if x.shape[0] != 1:
            raise RuntimeError("The integer runtime serves one query at a time")
        T = x.shape[1]
        text = segs[0][0].startswith("text.")
        if text:
            hit = self._text_hit(name, x)
            if hit is not None:
                return hit
        if stream is not None and st.shared is not None and st.shared[0] is x and st.shared[1] == stream:
            operand = st.shared[2]
        else:
            idx = st.index(segs, T)
            if idx is not None and idx.numel() == 0:
                idx = None
            codes, scales = self.ops.quant(self.xf(name, xf, x.shape[2]).apply(x[0]), idx)
            operand = (codes, scales, idx)
            if stream is not None:
                st.shared = (x, stream, operand)
        self.gemms += 1
        out = self.ops.gemm(*operand, self.layers[name], self.scratch, x.dtype).unsqueeze(0)
        if text:
            self.text[name] = (x, out)
        return out

    def kv(self, x, kind, segs, key, site):
        st = self.st
        st.counts[site] += 1
        T = x.shape[1]
        if site == "text":
            hit = self._text_hit(key, x)
            if hit is not None:
                return hit
        idx = st.index(segs, T)
        if idx is not None and idx.numel() == 0:
            idx = None
        r = (x[0].float().reshape(T * st.h, st.d) @ self.fd).contiguous()
        out = (self.ops.kvq(r, T, st.h, kind, idx) @ self.fd).reshape(x.shape).to(x.dtype)
        if site == "text":
            self.text[key] = (x, out)
        return out


def cache_bytes(st, blocks, refined):
    """Size of the cached keys and values of one query at 16 bits and as 4-bit codes. Per (token, head): head_dim
    codes plus an FP16 scale and a 4-bit zero point; a refined token stores a second pass of the same size.
    refined: {site: refined tokens}; sites: video self-attention (the observed and future tokens) and the two text
    contexts."""
    values = st.h * st.d
    tokens = {"video": st.video, "text.video": st.shapes["context_tokens"], "text.action": st.shapes["context_tokens"]}
    per_token = (values * 4 + st.h * (16 + 4)) / 8
    out = dict(bf16=0.0, int4=0.0)
    for site, n in tokens.items():
        out["bf16"] += 2 * blocks * n * values * 2
        out["int4"] += 2 * blocks * (n + refined.get(site, 0)) * per_token
    return out
