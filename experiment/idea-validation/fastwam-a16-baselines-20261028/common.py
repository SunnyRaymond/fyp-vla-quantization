"""Shared pieces: JSON I/O, PBS guard, BF16 runtime (optionally without T5), T5 cache, case-2000 anchor."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket

TARGETS = 614


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def torch_save(obj, path):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def guard():
    host = socket.gethostname().split(".")[0]
    nodefile = os.environ.get("PBS_NODEFILE")
    if (not os.environ.get("PBS_JOBID") or "login" in host.lower() or not nodefile
            or host not in {x.split(".")[0] for x in Path(nodefile).read_text().split()}):
        raise RuntimeError("Approved PBS compute allocation required")


def results(p):
    return Path(p["remote_root"]) / "results"


def bf16_runtime(out, text_encoder=True):
    """runner.Runtime on the BF16 arm; without T5 the model is built with model.load_text_encoder=false."""
    import torch
    import hydra
    from reference_profile import freeze_numerics
    import runner
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible GPU per process is required")
    freeze_numerics(torch)
    Path(out).mkdir(parents=True, exist_ok=True)
    original = hydra.compose
    if not text_encoder:
        def compose(*args, overrides=(), **kwargs):
            return original(*args, overrides=[*overrides, "model.load_text_encoder=false"], **kwargs)
        hydra.compose = compose
    try:
        runtime = runner.Runtime(Path(out))
    finally:
        hydra.compose = original
    if (runtime.model.text_encoder is None) == text_encoder:
        raise RuntimeError("Text-encoder loading differs from the request")
    runtime.arm("bf16")
    for parameter in runtime.model.parameters():
        parameter.requires_grad_(False)
    runtime.model.eval()
    if len(runtime.q.modules) != TARGETS:
        raise RuntimeError(f"Expected {TARGETS} target Linears, got {len(runtime.q.modules)}")
    return runtime


def install_t5_cache(runtime, cache):
    """Serve model.encode_prompt from precomputed (embedding, mask) pairs; unknown prompts fail loudly."""
    model = runtime.model

    def encode_prompt(prompt):
        if prompt not in cache:
            raise KeyError(f"Prompt missing from the T5 cache: {prompt!r}")
        emb, mask = cache[prompt]
        return emb.to(model.device), mask.to(model.device)
    model.encode_prompt = encode_prompt


def anchor_datum(p, runtime):
    """Case-2000 observation and its sampler seed (seed index 0), with runtime.description set."""
    import numpy as np
    import calibration_source
    plan_path = Path(p["panel"]["calibration_plan"])
    row = next(r for r in read(plan_path)["inputs"] if r["case_id"] == p["panel"]["anchor_case"])
    observation, metadata = calibration_source.load_observation(np, plan_path.parent, row)
    runtime.description = metadata["description"]
    return runtime.datum(observation), row["sampler_seeds"][0]


def capture(p, runtime):
    """Native action and future latent for the anchor query."""
    import torch
    from profile_worker import _capture_anchor
    datum, seed = anchor_datum(p, runtime)
    action, _, future, calls = _capture_anchor(runtime, datum, seed, torch)
    if calls != 1:
        raise RuntimeError("Anchor query called infer_action more than once")
    return action.detach().float().cpu(), future.float().cpu()


def max_abs(a, b):
    return float((a.float() - b.float()).abs().max())
