"""Small crash-safe checkpoint helpers; importing this module does not load torch."""
from __future__ import annotations

import importlib
import os
import random
import tempfile
from pathlib import Path


def atomic_torch_save(torch, path: Path, payload) -> None:
    """Write a checkpoint atomically, leaving any previous file intact on failure."""
    path = Path(path)
    temporary = None
    try:
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        except OSError:
            pass
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def to_cpu(value):
    """Detach and clone tensors onto CPU throughout ordinary containers."""
    torch = importlib.import_module("torch")
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(item) for item in value)
    return value


def capture_rng(torch) -> dict:
    """Capture Torch, Python, and NumPy RNG state using weights_only-safe values."""
    try:
        numpy = importlib.import_module("numpy")
        kind, keys, position, has_gauss, cached_gaussian = numpy.random.get_state()
        numpy_state = {"kind": str(kind), "keys": keys.tolist(), "position": int(position),
                       "has_gauss": int(has_gauss), "cached_gaussian": float(cached_gaussian)}
    except ImportError:
        numpy_state = None
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return {"torch_cpu": torch.get_rng_state().clone(), "torch_cuda": [state.clone() for state in cuda_states],
            "python": random.getstate(), "numpy": numpy_state}


def restore_rng(torch, state: dict) -> None:
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    random.setstate(state["python"])
    if state["numpy"] is not None:
        numpy = importlib.import_module("numpy")
        saved = state["numpy"]
        numpy.random.set_state((saved["kind"], numpy.asarray(saved["keys"], dtype=numpy.uint32),
                                saved["position"], saved["has_gauss"], saved["cached_gaussian"]))


def optimizer_to_device(torch, optimizer, device):
    """Move optimizer tensors with parameters; keep uncapturable Adam steps on CPU."""
    adam_types = tuple(cls for cls in (getattr(torch.optim, "Adam", None),
                                       getattr(torch.optim, "AdamW", None)) if cls is not None)
    groups = {parameter: group for group in optimizer.param_groups for parameter in group["params"]}
    for parameter, state in optimizer.state.items():
        group = groups.get(parameter, {})
        keep_step_on_cpu = (isinstance(optimizer, adam_types) and not group.get("capturable", False)
                            and not group.get("fused", False))
        for key, value in state.items():
            if torch.is_tensor(value):
                if key == "step" and value.numel() == 1 and keep_step_on_cpu:
                    state[key] = value.to("cpu")
                else:
                    state[key] = value.to(device)
    return optimizer
