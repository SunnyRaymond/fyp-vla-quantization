"""Query-local reuse for fixed FastWAM text and cross-attention context."""

from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from typing import Any, Callable


class CacheManager:
    """Cache expert text embeddings and per-block cross-attention K/V per query."""

    def __init__(self, model: Any, *, diagnostics: bool = False, stable_buffers: bool = True):
        self.model = model
        self.diagnostics = diagnostics
        self.stable_buffers = stable_buffers
        self.active = False
        self._capturing = False
        self._query = 0
        self._slots: dict[str, dict[str, Any]] = {}
        self._slot_counts: dict[str, dict[str, int]] = {}
        self._modules: dict[str, tuple[Any, Callable[..., Any]]] = {}
        self._counts = {"hits": 0, "misses": 0, "diagnostic_mismatches": 0}
        self._by_kind = {
            kind: {"hits": 0, "misses": 0}
            for kind in ("text", "k", "v")
        }
        self._experts: dict[str, Any] = {}
        self._text_linear_counts: dict[str, int] = {}
        self._install_expert("video", model.video_expert)
        self._install_expert("action", model.action_expert)

        original_infer_action = model.infer_action

        @wraps(original_infer_action)
        def query_scoped_infer_action(*args: Any, **kwargs: Any) -> Any:
            self.begin_query()
            try:
                return original_infer_action(*args, **kwargs)
            finally:
                self.end_query()

        model.infer_action = query_scoped_infer_action

    def _install_expert(self, name: str, expert: Any) -> None:
        self._experts[name] = expert
        self._text_linear_counts[name] = _linear_count(expert.text_embedding)
        self._wrap(f"text:{name}", "text", expert.text_embedding)
        for index, block in enumerate(expert.blocks):
            attention = block.cross_attn
            self._wrap(f"k:{name}:{index}", "k", attention.k)
            self._wrap(f"v:{name}:{index}", "v", attention.v)

    def _wrap(self, key: str, kind: str, module: Any) -> None:
        original = module.forward
        self._modules[key] = (module, original)
        self._slots[key] = {"buffer": None, "valid": False, "input": None}
        self._slot_counts[key] = {"hits": 0, "misses": 0}

        def cached_forward(value: Any, *args: Any, **kwargs: Any) -> Any:
            if not self.active:
                return original(value, *args, **kwargs)
            slot = self._slots[key]
            equal = True
            if slot["valid"] and self.diagnostics and not self._capturing:
                equal = _same_value(value, slot["input"])
            if slot["valid"] and equal:
                self._count(key, kind, "hits")
                return slot["buffer"]
            if slot["valid"] and not equal:
                self._counts["diagnostic_mismatches"] += 1
                raise ValueError(f"query-local cache invariant violated for {key}")
            if self._capturing:
                raise RuntimeError(f"cache miss for {key} during graph capture; call prepare_expert first")
            result = original(value, *args, **kwargs)
            self._save(key, kind, value, result)
            return self._slots[key]["buffer"]

        module.forward = cached_forward

    def begin_query(self) -> None:
        if self.active:
            raise RuntimeError("a cache query is already active")
        self._query += 1
        self.active = True
        for slot in self._slots.values():
            slot["valid"] = False
            slot["input"] = None

    def end_query(self) -> None:
        self.active = False
        self._capturing = False

    @contextmanager
    def graph_capture(self):
        """Mark graph capture so no diagnostic equality checks or buffer writes occur."""
        if not self.active:
            raise RuntimeError("graph capture requires an active query")
        if self._capturing:
            raise RuntimeError("graph capture is already active")
        self._capturing = True
        try:
            yield
        finally:
            self._capturing = False

    def prepare_expert(self, expert_name: str, raw_context: Any) -> Any:
        """Eagerly fill stable cache buffers before capturing/replaying an expert graph."""
        if not self.active:
            raise RuntimeError("prepare_expert requires an active query")
        if self._capturing:
            raise RuntimeError("prepare_expert must run before graph capture")
        if expert_name not in self._experts:
            raise ValueError(f"unknown expert {expert_name!r}; expected one of {tuple(self._experts)}")

        context = self._apply(f"text:{expert_name}", raw_context)
        expert = self._experts[expert_name]
        for index, block in enumerate(expert.blocks):
            self._apply(f"k:{expert_name}:{index}", context)
            self._apply(f"v:{expert_name}:{index}", context)
        return context

    def _apply(self, key: str, value: Any) -> Any:
        slot = self._slots[key]
        kind = key.split(":", 1)[0]
        equal = True
        if slot["valid"] and self.diagnostics:
            equal = _same_value(value, slot["input"])
        if slot["valid"] and equal:
            self._count(key, kind, "hits")
            return slot["buffer"]
        if slot["valid"] and not equal:
            self._counts["diagnostic_mismatches"] += 1
            raise ValueError(f"query-local cache invariant violated for {key}")
        if self._capturing:
            raise RuntimeError(f"cache miss for {key} during graph capture; call prepare_expert first")
        module, original = self._modules[key]
        result = original(value)
        self._save(key, kind, value, result)
        return self._slots[key]["buffer"]

    def _save(self, key: str, kind: str, value: Any, result: Any) -> None:
        slot = self._slots[key]
        buffer = slot["buffer"]
        if self.stable_buffers and buffer is not None and _compatible(buffer, result):
            copier = getattr(buffer, "copy_", None)
            if callable(copier):
                copier(result)
            else:
                slot["buffer"] = result
        elif self.stable_buffers:
            slot["buffer"] = _clone_value(result)
        else:
            slot["buffer"] = result
        slot["valid"] = True
        slot["input"] = _clone_value(value) if self.diagnostics else None
        self._count(key, kind, "misses")

    def _count(self, key: str, kind: str, event: str) -> None:
        if self._capturing:
            return
        self._counts[event] += 1
        self._slot_counts[key][event] += 1
        if kind in self._by_kind:
            self._by_kind[kind][event] += 1

    def reset_metrics(self) -> None:
        """Reset query and hit/miss counters without changing cached buffers or validity."""
        if self.active:
            raise RuntimeError("cannot reset cache metrics during an active query")
        self._query = 0
        for event in self._counts:
            self._counts[event] = 0
        for counts in self._by_kind.values():
            for event in counts:
                counts[event] = 0
        for counts in self._slot_counts.values():
            for event in counts:
                counts[event] = 0

    def metadata(self) -> dict[str, Any]:
        """Return JSON-safe observed counters; graph replays bypass Python wrappers."""
        bytes_by_key = {key: _tensor_bytes(slot["buffer"]) for key, slot in self._slots.items()}
        layers = {
            key[2:]
            for key in self._slots
            if key.startswith("k:")
        }
        kv_hits = sum(
            min(self._slot_counts[f"k:{layer}"]["hits"], self._slot_counts[f"v:{layer}"]["hits"])
            for layer in layers
        )
        kv_misses = sum(
            self._slots[f"k:{layer}"]["buffer"] is not None
            and self._slots[f"v:{layer}"]["buffer"] is not None
            for layer in layers
        )
        text_embedding_hits = self._by_kind["text"]["hits"]
        text_linear_hits = sum(
            self._slot_counts[f"text:{expert}"]["hits"] * linears
            for expert, linears in self._text_linear_counts.items()
        )
        return {
            "queries": self._query,
            "active": self.active,
            "hits": self._counts["hits"],
            "misses": self._counts["misses"],
            "diagnostic_mismatches": self._counts["diagnostic_mismatches"],
            "by_kind": {kind: dict(counts) for kind, counts in self._by_kind.items()},
            "cross_kv_pair_hits": kv_hits,
            "cross_kv_linear_hits": self._by_kind["k"]["hits"] + self._by_kind["v"]["hits"],
            "cross_kv_layer_buffers": kv_misses,
            "text_embedding_hits": text_embedding_hits,
            "text_linear_hits": text_linear_hits,
            "buffer_bytes": sum(bytes_by_key.values()),
            "buffer_bytes_by_key": bytes_by_key,
        }


def install(model: Any, *, diagnostics: bool = False, stable_buffers: bool = True) -> CacheManager:
    """Wrap one model's infer_action and install query-local expert caches."""
    return CacheManager(model, diagnostics=diagnostics, stable_buffers=stable_buffers)


def _clone_value(value: Any) -> Any:
    clone = getattr(value, "clone", None)
    if callable(clone):
        return clone()
    try:
        return deepcopy(value)
    except Exception:
        return value


def _compatible(left: Any, right: Any) -> bool:
    for attribute in ("shape", "dtype", "device"):
        a, b = getattr(left, attribute, None), getattr(right, attribute, None)
        if a is not None and b is not None and a != b:
            return False
    return True


def _same_value(left: Any, right: Any) -> bool:
    if left is right:
        return True
    if not _compatible(left, right):
        return False
    equal = getattr(left, "equal", None)
    if callable(equal):
        try:
            return bool(equal(right))
        except Exception:
            return False
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_same_value(a, b) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same_value(left[k], right[k]) for k in left)
    if isinstance(left, (str, bytes, int, float, bool, type(None))):
        return left == right
    return False


def _tensor_bytes(value: Any) -> int:
    if value is None:
        return 0
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    numel, element_size = getattr(value, "numel", None), getattr(value, "element_size", None)
    if callable(numel) and callable(element_size):
        try:
            return int(numel()) * int(element_size())
        except Exception:
            return 0
    return 0


def _linear_count(module: Any) -> int:
    modules = getattr(module, "modules", None)
    if callable(modules):
        try:
            count = sum(type(child).__name__ == "Linear" for child in modules())
            if count:
                return count
        except Exception:
            pass
    return 1
