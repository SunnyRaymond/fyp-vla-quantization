"""Fixed-context cache experiments for the pinned FastWAM action expert.

This is an inference-only experiment helper. It temporarily patches module
``forward`` methods and must not be used for concurrent calls on one model.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import socket
from time import perf_counter
from typing import Iterator

import torch
import torch.nn.functional as F
from torch import nn

from fastwam.models.wan22.wan_video_dit import flash_attention


_MISSING = object()


def _tensor_bytes(tensor: torch.Tensor | None) -> int:
    return 0 if tensor is None else tensor.numel() * tensor.element_size()


def _sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


class ContextCache:
    """Reuse one batch-1 action context during a denoising chunk.

    Args:
        model: FastWAM model, or its ``action_expert``.
        mode: ``embedding`` caches only action text embedding; ``kv`` also
            caches each block's cross-attention K/V; ``batched_kv`` computes
            all blocks' fixed-context K/V with stacked per-layer parameters.

    Call ``prepare(context)`` once per chunk, inside or outside ``activate()``.
    ``context`` must be the fixed ``[1, sequence, text_dim]`` input. Queries
    may have batch W; cached context/K/V are expanded as views for that batch.
    A3's packed K/V parameters are allocated in ``__init__`` and survive
    ``clear()``. ``cache_bytes`` counts owned prepared tensors, while
    ``packed_weight_bytes`` counts those persistent A3 parameter copies.
    ``prepare_latency_ms`` reports CPU enqueue time only; synchronize at the
    benchmark boundary for device-complete timing.
    """

    MODES = ("embedding", "kv", "batched_kv")

    def __init__(self, model: nn.Module, mode: str):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.mode = mode
        self.expert = getattr(model, "action_expert", model)
        self.text_embedding = self.expert.text_embedding
        self.attentions = tuple(block.cross_attn for block in self.expert.blocks)
        if not self.attentions:
            raise ValueError("action expert has no cross-attention blocks")
        self._text_embedding_forward = self.text_embedding.forward
        self._active = False

        self._raw_context: torch.Tensor | None = None
        self._raw_signature: tuple | None = None
        self._context_embedding: torch.Tensor | None = None
        self._kv_keys: tuple[torch.Tensor, ...] | torch.Tensor | None = None
        self._kv_values: tuple[torch.Tensor, ...] | torch.Tensor | None = None
        self.prepare_latency_ms: float | None = None

        self._packed: dict[str, torch.Tensor | None | tuple[float, ...]] = {}
        self.packed_weight_build_ms = 0.0
        if mode == "batched_kv":
            device = self.attentions[0].k.weight.device
            _sync(device)
            started = perf_counter()
            with torch.no_grad():
                self._packed = self._pack_cross_attention_weights()
            _sync(device)
            self.packed_weight_build_ms = (perf_counter() - started) * 1000.0

    def _pack_cross_attention_weights(
        self,
    ) -> dict[str, torch.Tensor | None | tuple[float, ...]]:
        def stack_linear(name: str) -> tuple[torch.Tensor, torch.Tensor | None]:
            layers = [getattr(attn, name) for attn in self.attentions]
            weights = torch.stack([layer.weight.detach() for layer in layers])
            biases = [layer.bias for layer in layers]
            if all(bias is None for bias in biases):
                bias = None
            elif any(bias is None for bias in biases):
                raise ValueError(f"cross-attention {name} bias presence differs by layer")
            else:
                bias = torch.stack([item.detach() for item in biases if item is not None])
            return weights, bias

        key_weight, key_bias = stack_linear("k")
        value_weight, value_bias = stack_linear("v")
        norm_weights = torch.stack([attn.norm_k.weight.detach() for attn in self.attentions])
        norm_eps = tuple(float(attn.norm_k.eps) for attn in self.attentions)
        return {
            "k_weight": key_weight,
            "k_bias": key_bias,
            "v_weight": value_weight,
            "v_bias": value_bias,
            "norm_weight": norm_weights,
            "norm_eps": norm_eps,
        }

    @property
    def packed_weight_bytes(self) -> int:
        return sum(
            _tensor_bytes(value)
            for value in self._packed.values()
            if isinstance(value, torch.Tensor)
        )

    @property
    def cache_bytes(self) -> int:
        total = _tensor_bytes(self._context_embedding)
        for cached in (self._kv_keys, self._kv_values):
            if isinstance(cached, tuple):
                total += sum(_tensor_bytes(tensor) for tensor in cached)
            else:
                total += _tensor_bytes(cached)
        return total

    @property
    def prepared(self) -> bool:
        return self._context_embedding is not None

    @staticmethod
    def _signature(context: torch.Tensor) -> tuple:
        try:
            version = context._version
        except RuntimeError:  # inference-mode tensors do not track versions
            version = None
        return (
            context.untyped_storage().data_ptr(),
            context.storage_offset(),
            version,
            tuple(context.shape[1:]),
            context.dtype,
            context.device,
        )

    def _check_raw_context(self, context: torch.Tensor) -> None:
        if self._raw_signature is None:
            raise RuntimeError("ContextCache.prepare(context) must run before cached inference")
        current = self._signature(context)
        expected = self._raw_signature
        if current[:2] != expected[:2] or current[3:] != expected[3:]:
            raise ValueError("cached context changed storage, sequence shape, dtype, or device")
        if expected[2] is not None and current[2] != expected[2]:
            raise ValueError("cached context was modified in place after prepare()")
        if context.ndim != 3 or context.shape[0] < 1:
            raise ValueError("context must have shape [1 or expanded batch, sequence, text_dim]")

    def _check_cached_embedding(self, context: torch.Tensor) -> None:
        cached = self._context_embedding
        if cached is None:
            raise RuntimeError("ContextCache.prepare(context) must run before cached inference")
        if (
            context.ndim != 3
            or context.shape[1:] != cached.shape[1:]
            or context.untyped_storage().data_ptr() != cached.untyped_storage().data_ptr()
        ):
            raise ValueError("cross-attention received context outside the prepared cache")

    def _cached_embedding_forward(self, context: torch.Tensor) -> torch.Tensor:
        self._check_raw_context(context)
        cached = self._context_embedding
        assert cached is not None
        if context.shape[0] == cached.shape[0]:
            return cached
        if cached.shape[0] == 1:
            return cached.expand(context.shape[0], -1, -1)
        raise ValueError("only a singleton prepared context can expand to query batch W")

    @staticmethod
    def _expand_batch(tensor: torch.Tensor, batch: int) -> torch.Tensor:
        if tensor.shape[0] == batch:
            return tensor
        if tensor.shape[0] == 1:
            return tensor.expand(batch, -1, -1)
        raise ValueError(
            f"cached context batch {tensor.shape[0]} cannot serve query batch {batch}"
        )

    def _cached_cross_forward(self, layer_idx: int, attn: nn.Module):
        def forward(x: torch.Tensor, context: torch.Tensor, ctx_mask=None) -> torch.Tensor:
            self._check_cached_embedding(context)
            keys = self._kv_keys
            values = self._kv_values
            if keys is None or values is None:
                raise RuntimeError("K/V cache is empty; call prepare(context) first")
            key = keys[layer_idx] if isinstance(keys, tuple) else keys[layer_idx]
            value = values[layer_idx] if isinstance(values, tuple) else values[layer_idx]
            key = self._expand_batch(key, x.shape[0])
            value = self._expand_batch(value, x.shape[0])
            query = attn.norm_q(attn.q(x))
            attended = flash_attention(
                q=query,
                k=key,
                v=value,
                num_heads=attn.num_heads,
                ctx_mask=ctx_mask,
            )
            return attn.o(attended)

        return forward

    def prepare(self, context: torch.Tensor) -> torch.Tensor:
        """Build this chunk's embedding and optional K/V cache; return embedding."""
        self.clear()
        if context.ndim != 3 or context.shape[0] != 1:
            raise ValueError(
                "this experiment requires one fixed context with shape [1, sequence, text_dim]"
            )
        started = perf_counter()
        with torch.no_grad():
            embedded = self._text_embedding_forward(context).detach()
            keys = values = None
            if self.mode == "kv":
                keys = tuple(
                    attn.norm_k(attn.k(embedded)).detach() for attn in self.attentions
                )
                values = tuple(attn.v(embedded).detach() for attn in self.attentions)
            elif self.mode == "batched_kv":
                packed = self._packed
                layer_count, out_dim, in_dim = packed["k_weight"].shape
                batch, sequence, _ = embedded.shape
                key_bias = packed["k_bias"]
                value_bias = packed["v_bias"]
                key_flat = F.linear(
                    embedded,
                    packed["k_weight"].reshape(layer_count * out_dim, in_dim),
                    None if key_bias is None else key_bias.reshape(-1),
                )
                value_flat = F.linear(
                    embedded,
                    packed["v_weight"].reshape(layer_count * out_dim, in_dim),
                    None if value_bias is None else value_bias.reshape(-1),
                )
                key_raw = key_flat.view(batch, sequence, layer_count, out_dim)
                key_raw = key_raw.permute(2, 0, 1, 3).contiguous()
                value = value_flat.view(batch, sequence, layer_count, out_dim)
                value = value.permute(2, 0, 1, 3).contiguous()

                key_float = key_raw.float()
                eps = torch.tensor(
                    packed["norm_eps"], dtype=torch.float32, device=key_raw.device
                )[:, None, None, None]
                normed = key_float * torch.rsqrt(
                    key_float.square().mean(dim=-1, keepdim=True)
                    + eps
                )
                keys = (
                    normed.to(key_raw.dtype)
                    * packed["norm_weight"][:, None, None, :]
                ).detach()
                values = value.detach()

        self.prepare_latency_ms = (perf_counter() - started) * 1000.0
        self._raw_context = context
        self._raw_signature = self._signature(context)
        self._context_embedding = embedded
        self._kv_keys = keys
        self._kv_values = values
        return embedded

    @contextmanager
    def activate(self) -> Iterator["ContextCache"]:
        """Temporarily use this cache and always restore original forwards."""
        if self._active:
            raise RuntimeError("ContextCache.activate() cannot be nested")
        patched: list[tuple[nn.Module, object]] = []

        def replace_forward(module: nn.Module, forward) -> None:
            previous = module.__dict__.get("forward", _MISSING)
            module.forward = forward
            patched.append((module, previous))

        try:
            replace_forward(self.text_embedding, self._cached_embedding_forward)
            if self.mode in ("kv", "batched_kv"):
                for index, attn in enumerate(self.attentions):
                    replace_forward(attn, self._cached_cross_forward(index, attn))
            self._active = True
            yield self
        finally:
            self._active = False
            for module, previous in reversed(patched):
                if previous is _MISSING:
                    module.__dict__.pop("forward", None)
                else:
                    module.forward = previous

    def clear(self) -> None:
        """Release per-chunk input/embedding/KV tensors, retaining A3 weights."""
        self._raw_context = None
        self._raw_signature = None
        self._context_embedding = None
        self._kv_keys = None
        self._kv_values = None


def _require_pbs_compute() -> None:
    """Reject local/login execution; do not infer allocation from PBS_JOBID alone."""
    job_id = os.environ.get("PBS_JOBID", "").strip()
    nodefile = os.environ.get("PBS_NODEFILE", "").strip()
    hostname = socket.gethostname().strip().lower().rstrip(".")
    if not job_id or not nodefile:
        raise RuntimeError("self_check requires an active PBS_JOBID and PBS_NODEFILE")
    if "login" in hostname:
        raise RuntimeError(f"self_check is forbidden on a login node: {hostname}")
    nodefile_path = Path(nodefile)
    if not nodefile_path.is_file():
        raise RuntimeError(f"PBS_NODEFILE is missing: {nodefile}")
    allocated = {
        line.strip().lower().rstrip(".")
        for line in nodefile_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    host_aliases = {hostname, hostname.split(".", 1)[0]}
    node_aliases = {name for entry in allocated for name in (entry, entry.split(".", 1)[0])}
    if not host_aliases.intersection(node_aliases):
        raise RuntimeError(
            f"current host {hostname} is not listed in PBS_NODEFILE for job {job_id}"
        )


def self_check() -> dict[str, dict[str, float | int]]:
    """Run a small parity/restoration check on a verified PBS compute node."""
    _require_pbs_compute()
    from fastwam.models.wan22.wan_video_dit import CrossAttention

    class ToyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.cross_attn = CrossAttention(hidden_dim=8, attn_head_dim=4, num_heads=2)

    class ToyActionExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_embedding = nn.Sequential(
                nn.Linear(6, 8), nn.GELU(approximate="tanh"), nn.Linear(8, 8)
            )
            self.blocks = nn.ModuleList([ToyBlock() for _ in range(3)])

    def run(expert: ToyActionExpert, action: torch.Tensor, context: torch.Tensor):
        hidden = action
        embedded = expert.text_embedding(context)
        for block in expert.blocks:
            hidden = hidden + block.cross_attn(hidden, embedded)
        return hidden

    torch.manual_seed(17)
    expert = ToyActionExpert().eval()
    action = torch.randn(3, 4, 8)
    context = torch.randn(1, 5, 6)
    with torch.no_grad():
        reference = run(expert, action, context)

    report: dict[str, dict[str, float | int]] = {}
    for mode in ContextCache.MODES:
        cache = ContextCache(expert, mode)
        with torch.no_grad(), cache.activate():
            cache.prepare(context)
            actual = run(expert, action, context)
        torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-5)
        max_abs = float((actual - reference).abs().max().item())

        modules = [expert.text_embedding, *(block.cross_attn for block in expert.blocks)]
        original_slots = [module.__dict__.get("forward", _MISSING) for module in modules]
        try:
            with cache.activate():
                cache.prepare(context)
                raise RuntimeError("intentional restoration check")
        except RuntimeError as error:
            if str(error) != "intentional restoration check":
                raise
        for module, previous in zip(modules, original_slots):
            current = module.__dict__.get("forward", _MISSING)
            if current is not previous:
                raise AssertionError("module forward was not restored after an exception")

        report[mode] = {
            "max_abs_error": max_abs,
            "cache_bytes": cache.cache_bytes,
            "packed_weight_bytes": cache.packed_weight_bytes,
        }
        cache.clear()
        if cache.cache_bytes != 0:
            raise AssertionError("clear() did not release per-chunk cache tensors")
    return report


toy_self_check = self_check
