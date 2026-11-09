"""AgileX fine-tuning hooks for FastWAM.

create_model: builds the Optional-IDM model and loads the LIBERO checkpoint *before* the trainer hands the model to
DeepSpeed. The trainer's own `resume=<file.pt>` path loads weights after `accelerator.prepare`, when ZeRO has already
copied the random init into its fp32 master partitions; the first optimizer step would then write that random init
back. Loading here also checks that every checkpoint tensor lands in the model (the stock loader uses strict=False).

GroupedAdamW: splits the flat trainable-parameter list into contiguous groups of similar size. ZeRO-2 steps one group
at a time and materialises only that group's fp32 gradient, which keeps the 6B-parameter step inside 40 GB.

Layer checkpointing: ZeRO-2 keeps ~28 GiB of weights + fp32 master + Adam per A100-40GB, and one sample's activations
through the 30 joint MoT layers exceed the remaining ~8 GiB (measured: OOM at batch 1). Upstream MoT refuses its own
checkpointing because of the torch.compile(fullgraph) path, which is off here, so each `_forward_joint_layer` call
(the only training path: training_loss -> forward_joint_core) is wrapped in non-reentrant torch.utils.checkpoint.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

_TorchAdamW = torch.optim.AdamW


def split_by_numel(params: list, groups: int) -> list[list]:
    total = sum(p.numel() for p in params)
    target = total / groups
    out, cur, size = [], [], 0
    for p in params:
        cur.append(p)
        size += p.numel()
        if size >= target and len(out) < groups - 1:
            out.append(cur)
            cur, size = [], 0
    if cur:
        out.append(cur)
    return out


def _grouped(params):
    params = list(params)
    groups = int(os.environ.get("FW_OPT_GROUPS", "16"))
    if groups > 1 and params and isinstance(params[0], torch.Tensor):
        params = [{"params": chunk} for chunk in split_by_numel(params, groups)]
    return params


class GroupedAdamW(_TorchAdamW):
    def __init__(self, params, *args, **kwargs):
        super().__init__(_grouped(params), *args, **kwargs)


class OffloadAdamW(_TorchAdamW):
    """FW_OFFLOAD=1 (ZeRO-2 offload_optimizer=cpu): build DeepSpeedCPUAdam (JIT-compiled cpu_adam, AdamW mode, same
    betas/eps/weight decay as the trainer's AdamW) on the grouped parameters; if the op cannot be built, fall back to
    torch AdamW, which ZeRO-Offload also steps on the CPU partitions (zero_force_ds_cpu_optimizer=false)."""

    def __new__(cls, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, **kwargs):
        groups = _grouped(params)
        try:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            opt = DeepSpeedCPUAdam(groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, adamw_mode=True)
            print("FW_OPTIMIZER DeepSpeedCPUAdam", flush=True)
        except Exception as err:  # noqa: BLE001  (compiler/ninja missing in the container)
            print(f"FW_OPTIMIZER torch AdamW fallback ({type(err).__name__}: {str(err)[:300]})", flush=True)
            opt = GroupedAdamW(groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        return opt


def install_layer_checkpointing() -> None:
    from torch.utils.checkpoint import checkpoint
    from fastwam.models.wan22.mot import MoT

    if getattr(MoT, "_fw_agilex_checkpointed", False):
        return
    original = MoT._forward_joint_layer

    def _forward_joint_layer(self, *args, **kwargs):
        if self.training and torch.is_grad_enabled() and not getattr(self, "compile_training_layers", False):
            return checkpoint(original, self, *args, use_reentrant=False, **kwargs)
        return original(self, *args, **kwargs)

    MoT._forward_joint_layer = _forward_joint_layer
    MoT._fw_agilex_checkpointed = True
    print("FW_LAYER_CHECKPOINT on: MoT._forward_joint_layer wrapped in non-reentrant checkpoint", flush=True)


def install_memprobe(out_dir: str) -> None:
    """FW_MEMPROBE=<dir>: per-rank JSON lines of CUDA memory and live gradients inside the ZeRO gradient hook and
    around the forward; on OOM a summary of the largest live allocations by Python stack (memory history on)."""
    from deepspeed.runtime.zero import stage_1_and_2 as s12
    from fastwam.trainer import Wan22Trainer

    rank = int(os.environ.get("RANK", "0"))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log = open(Path(out_dir) / f"memprobe_rank{rank}.jsonl", "a", encoding="utf-8")
    gib = 2 ** 30
    torch.cuda.memory._record_memory_history(max_entries=300000)

    def emit(**row):
        row.update(alloc=round(torch.cuda.memory_allocated() / gib, 3),
                   max_alloc=round(torch.cuda.max_memory_allocated() / gib, 3))
        log.write(json.dumps(row) + "\n")
        log.flush()

    def live_grads(opt):
        sizes, dtypes, count = 0, set(), 0
        for group in opt.bit16_groups:
            for p in group:
                if p.grad is not None:
                    count += 1
                    sizes += p.grad.numel() * p.grad.element_size()
                    dtypes.add(str(p.grad.dtype))
        return dict(live_grads=count, live_grad_gib=round(sizes / gib, 3), grad_dtypes=sorted(dtypes))

    orig_process = s12.DeepSpeedZeroOptimizer.process_gradients
    calls = {"n": 0}

    def process_gradients(self, param, i):
        calls["n"] += 1
        if calls["n"] == 1:
            emit(event="ds_config", partition_gradients=self.partition_gradients,
                 contiguous_gradients=self.contiguous_gradients, overlap_comm=self.overlap_comm,
                 reduce_bucket_size=self.reduce_bucket_size, use_grad_accum_attribute=self.use_grad_accum_attribute,
                 dtype=str(self.dtype), groups=len(self.bit16_groups),
                 params=sum(len(g) for g in self.bit16_groups))
        if calls["n"] % 40 == 1:
            g = param.grad
            emit(event="hook", n=calls["n"], param_shape=list(param.shape),
                 grad_dtype=None if g is None else str(g.dtype), **live_grads(self))
        return orig_process(self, param, i)

    s12.DeepSpeedZeroOptimizer.process_gradients = process_gradients

    def summarize_snapshot(tag):
        snap = torch.cuda.memory._snapshot()
        by_site, total = {}, 0
        for seg in snap["segments"]:
            for blk in seg["blocks"]:
                if blk["state"] != "active_allocated":
                    continue
                total += blk["size"]
                frames = [f for f in blk.get("frames", []) if "/torch/" not in f["filename"]][:4]
                key = " <- ".join(f"{Path(f['filename']).name}:{f['line']}:{f['name']}" for f in frames) or "?"
                by_site[key] = by_site.get(key, 0) + blk["size"]
        top = sorted(by_site.items(), key=lambda kv: -kv[1])[:30]
        emit(event="snapshot", tag=tag, active_gib=round(total / gib, 3),
             top=[dict(gib=round(v / gib, 3), site=k) for k, v in top])

    orig_train = Wan22Trainer.train

    def train(self):
        model = self.accelerator.unwrap_model(self.model)
        orig_loss = model.training_loss

        def training_loss(sample, *a, **k):
            emit(event="forward_start", batch=int(sample["video"].shape[0]))
            out = orig_loss(sample, *a, **k)
            emit(event="forward_end")
            summarize_snapshot("after_forward") if calls["n"] == 0 else None
            return out

        model.training_loss = training_loss
        self.save_checkpoint = lambda: {"weights_path": None, "state_path": None}
        try:
            return orig_train(self)
        except torch.OutOfMemoryError:
            emit(event="oom", hooks=calls["n"])
            summarize_snapshot("oom")
            raise
        finally:
            emit(event="end", hooks=calls["n"])

    Wan22Trainer.train = train


def install() -> None:
    torch.optim.AdamW = OffloadAdamW if os.environ.get("FW_OFFLOAD") == "1" else GroupedAdamW
    if os.environ.get("FW_LAYER_CHECKPOINT", "1") == "1":
        install_layer_checkpointing()
    if os.environ.get("FW_MEMPROBE"):
        install_memprobe(os.environ["FW_MEMPROBE"])


def _rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def create_model(init_checkpoint: str, **kwargs):
    from fastwam.runtime import create_fastwam_optional_idm

    start = time.time()
    model = create_fastwam_optional_idm(**kwargs)
    built = time.time()
    payload = torch.load(init_checkpoint, map_location="cpu", mmap=True, weights_only=False)
    if "mot" not in payload:
        raise RuntimeError(f"{init_checkpoint} has no `mot` state dict (keys: {sorted(payload)[:10]})")
    result = model.mot.load_state_dict(payload["mot"], strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint/model key mismatch: {len(result.missing_keys)} missing (first {result.missing_keys[:5]}), "
            f"{len(result.unexpected_keys)} unexpected (first {result.unexpected_keys[:5]})"
        )
    if model.proprio_encoder is None or "proprio_encoder" not in payload:
        raise RuntimeError("Expected a proprio encoder in both the model and the checkpoint")
    model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    report = dict(
        init_checkpoint=str(init_checkpoint),
        mot_tensors=len(payload["mot"]),
        mot_params=sum(t.numel() for t in payload["mot"].values()),
        dit_params=sum(p.numel() for p in model.dit.parameters()),
        proprio_weight_shape=list(model.proprio_encoder.weight.shape),
        build_seconds=round(built - start, 1),
        load_seconds=round(time.time() - built, 1),
        payload_keys=sorted(k for k in payload if k not in ("mot", "proprio_encoder")),
    )
    del payload
    out = os.environ.get("FW_INIT_REPORT")
    if out and _rank() == 0:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("FW_INIT " + json.dumps(report), flush=True)
    return model
