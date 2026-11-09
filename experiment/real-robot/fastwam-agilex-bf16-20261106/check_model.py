"""CPU check of the training model setup, run by prep.pbs inside the FastWAM container.

  check_model.py --config <composed training config> --out <results>/prep

Builds the model exactly as training does (hydra instantiate of cfg.model -> fw_agilex.create_model) on the CPU,
which fails on any checkpoint/model key mismatch; then checks that the trainer's trainable set (dit + proprio encoder)
is fully covered by the GroupedAdamW split, and writes the per-GPU ZeRO-2 memory budget for 4 A100-40GB.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import fw_agilex
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from fastwam.utils.config_resolvers import register_default_resolvers

    fw_agilex.install()
    register_default_resolvers()
    cfg = OmegaConf.load(args.config)
    os.environ["FW_INIT_REPORT"] = str(args.out / "init_load.json")
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cpu")

    model.requires_grad_(False)
    model.dit.requires_grad_(True)
    model.proprio_encoder.requires_grad_(True)
    trainable = list(model.dit.parameters()) + list(model.proprio_encoder.parameters())
    n = sum(p.numel() for p in trainable)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    groups = int(os.environ.get("FW_OPT_GROUPS", "16"))
    opt = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-2, betas=(0.9, 0.95))
    sizes = [sum(p.numel() for p in g["params"]) for g in opt.param_groups]
    covered = {id(p) for g in opt.param_groups for p in g["params"]}
    if len(covered) != len(trainable) or sum(sizes) != n:
        raise SystemExit("GroupedAdamW split does not cover the trainable parameters exactly once")

    gib = 1024 ** 3
    world = 4
    budget = dict(
        bf16_params=2 * n / gib, own_bf16_grads=2 * n / world / gib, fp32_master=4 * n / world / gib,
        adam_moments=8 * n / world / gib, fp32_grad_one_group=4 * max(sizes) / world / gib,
        fp32_grad_single_group=4 * n / world / gib)
    budget["states_peak_grouped"] = sum(budget[k] for k in ("bf16_params", "own_bf16_grads", "fp32_master",
                                                              "adam_moments", "fp32_grad_one_group"))
    budget["states_peak_single_group"] = budget["states_peak_grouped"] - budget["fp32_grad_one_group"] \
        + budget["fp32_grad_single_group"]
    report = dict(
        trainable_params=n, frozen_params=frozen, optimizer_groups=len(sizes), group_numel_max=max(sizes),
        group_numel_min=min(sizes), type=type(opt).__name__,
        zero2_per_gpu_gib={k: round(v, 2) for k, v in budget.items()},
        init_load=json.loads((args.out / "init_load.json").read_text()),
        gradient_checkpointing=bool(cfg.model.mot_checkpoint_mixed_attn))
    (args.out / "check_model.json").write_text(json.dumps(report, indent=2) + "\n")
    print("CHECK_MODEL " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
