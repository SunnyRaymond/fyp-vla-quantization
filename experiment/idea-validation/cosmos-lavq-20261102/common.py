"""Shared pieces: JSON/tensor I/O, PBS guard, panels, the BF16 Cosmos runtime, calibration inputs, anchor, weight install.

The A16 baselines' common.py with this experiment's roots: the BF16 masters, C80 Hessians, merged T5 cache and BF16
anchor come from the baselines' prep (baseline_root/results/prep); arms live under results/quantized/<arm>.
Stdlib at import time (sched.py / analysis.py run on host python3.11); torch is imported inside the functions.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket

TARGETS = 280


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


def baseline(p):
    return Path(p["baseline_root"]) / "results"


def prep(p):
    return baseline(p) / "prep"


def merged_cache(p):
    return prep(p) / "text_embeddings.pkl"


def panel_rows(p, name):
    """Frozen rows of panel A, B or C (slot indices offset .. offset + 279)."""
    cfg = p["panels"][name]
    doc = read(cfg["manifest"])
    rows = doc["variants"]
    if doc.get("protocol") != cfg["protocol"] or sorted(int(r["index"]) for r in rows) != list(
            range(cfg["offset"], cfg["offset"] + 280)):
        raise RuntimeError(f"Panel {name} manifest does not match {cfg['protocol']} / offset {cfg['offset']}")
    return rows


def arm_panels(p, arm):
    return p["arms"][arm]["panels"]


def load_pickle(path):
    import pickle
    with open(path, "rb") as stream:
        return pickle.load(stream)


def dump_pickle(obj, path):
    import pickle
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with open(tmp, "wb") as stream:
        pickle.dump(obj, stream, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def runtime(p, out, cache):
    """The c8-w4a4 CosmosRuntime (arm bf16) on the given T5 cache; all parameters frozen."""
    import torch
    from cosmos_runtime import CosmosRuntime
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rt = CosmosRuntime(read(p["cosmos"]["runtime_protocol"]), Path(p["cosmos"]["asset_root"]), Path(out),
                       arm="bf16", text_embeddings_path=str(cache))
    for parameter in rt.model.parameters():
        parameter.requires_grad_(False)
    rt.model.eval()
    if len(rt.target_linears) != TARGETS:
        raise RuntimeError(f"Expected {TARGETS} target Linears, got {len(rt.target_linears)}")
    return rt


def calibration_input(p, case):
    """(raw LIBERO observation, metadata, plan row) of a frozen original-domain case (C80 or held-out)."""
    import numpy as np
    plan_path = Path(p["calibration"]["plan"])
    row = next(r for r in read(plan_path)["inputs"] if int(r["case_id"]) == int(case))
    data = plan_path.parent
    with np.load(data / row["observation_file"], allow_pickle=False) as archive:
        observation = {k: archive[k].copy() for k in archive.files}
    metadata = read(data / row["metadata_file"])
    if int(metadata["case_id"]) != int(case) or metadata["suite"] != row["suite"]:
        raise RuntimeError(f"Calibration metadata differs for case {case}")
    return observation, metadata, row


def calibration_query(p, rt, case, capture=False):
    observation, metadata, row = calibration_input(p, case)
    seed = int(row["sampler_seeds"][p["calibration"]["seed_index"]])
    return rt.query(observation, metadata["description"], seed, suite=str(metadata["suite"]), capture=capture)


def anchor(p, rt):
    """Anchor case (seed index 0): action chunk [16, 7] and the generated future-image / future-wrist latents."""
    import torch
    import endpoint as legacy
    out = calibration_query(p, rt, p["calibration"]["anchor_case"], capture=True)
    captured, batch = out.get("captured", {}), out["data_batch"]
    generated = out.get("generated_latent", captured.get("generated_latent"))
    indices = out.get("latent_indices", captured.get("latent_indices", batch.get("latent_indices")))
    if not torch.is_tensor(generated) or generated.ndim != 5:
        raise RuntimeError("get_action did not expose the generated latent")
    size = generated.shape[0]
    future = {}
    for key, field in (("future_image", "future_image_latent_idx"), ("future_wrist", "future_wrist_image_latent_idx")):
        idx = legacy._tensor_index(batch, field, size, generated.device, torch, indices)
        future[key] = legacy._gather_slot(generated, idx, size, torch).float().cpu()
    return dict(action=torch.as_tensor(out["actions"]).float(), **future)


def anchor_errors(a, ref):
    """Max-abs action difference and future NMSE of anchor a against the reference anchor."""
    errors = dict(action_max_abs=float((a["action"] - ref["action"]).abs().max()),
                  action_rel_rmse=float((a["action"] - ref["action"]).square().mean().sqrt()
                                        / ref["action"].square().mean().sqrt()))
    for key in ("future_image", "future_wrist"):
        errors[f"{key}_nmse"] = float((a[key] - ref[key]).square().sum() / ref[key].square().sum())
    return errors


def same_anchor(a, b):
    import torch
    return all(torch.equal(a[k], b[k]) for k in ("action", "future_image", "future_wrist"))


def install(p, rt, arm):
    """Copy the arm's dense BF16 weights (results/quantized/<arm>/weights, files or links) into the target Linears."""
    import torch
    src = results(p) / "quantized" / arm
    if not (src / "COMPLETE.json").exists():
        raise RuntimeError(f"{arm} is not assembled: {src / 'COMPLETE.json'} missing")
    with torch.no_grad():
        for i, (name, module) in enumerate(rt.target_linears.items()):
            saved = torch.load(src / "weights" / f"{i:04d}.pt", map_location="cpu", weights_only=True)
            if saved["module"] != name or tuple(saved["weight"].shape) != tuple(module.weight.shape):
                raise RuntimeError(f"{arm} weight {i} does not match {name}")
            module.weight.copy_(saved["weight"].to(module.weight.device, module.weight.dtype))
            if not torch.equal(module.weight, saved["weight"].to(module.weight.device, module.weight.dtype)):
                raise RuntimeError(f"{arm} weight {i} was not installed exactly")
    return TARGETS
