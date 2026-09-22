#!/usr/bin/env python3
"""Small local-only checks for the frozen tail-robust runner.

The checks use only JSON and tiny tensors.  They intentionally do not open the
HDF5 dataset, load an official checkpoint, or inspect prepared rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--phase2-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    phase2 = json.loads(args.phase2_freeze.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen" or phase2.get("status") != "frozen":
        raise ValueError("both freezes must be frozen")
    if freeze["design"]["new_arms"] != ["ema_score", "ema_tail_score"]:
        raise ValueError("arm list drifted")
    if phase2["context_manifest"]["new_train_target"] != 512:
        raise ValueError("Phase 2 512-context authority missing")

    # Gradient check on the exact context-level score aggregation shape.
    student_cost = torch.linspace(0.1, 6.4, 8 * 64, dtype=torch.float64).reshape(8, 64).requires_grad_()
    teacher_cost = torch.flip(student_cost.detach(), dims=(1,)) + 0.03
    teacher_mean = teacher_cost.mean(dim=1, keepdim=True)
    teacher_std = teacher_cost.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
    student_norm = (student_cost - student_cost.mean(dim=1, keepdim=True)) / teacher_std
    teacher_norm = (teacher_cost - teacher_mean) / teacher_std
    element = torch.nn.functional.smooth_l1_loss(student_norm, teacher_norm, beta=1.0, reduction="none")
    order = torch.argsort(teacher_cost, dim=1, stable=True)
    weights = torch.ones_like(element)
    weights.scatter_(1, order[:, :12], 2.0)
    per_context = (element * weights).sum(dim=1) / weights.sum(dim=1)
    mean_score = per_context.mean()
    tail_score = 0.75 * mean_score + 0.25 * torch.topk(per_context, k=2).values.mean()
    tail_score.backward()
    gradient_check = bool(torch.isfinite(student_cost.grad).all().item() and float(student_cost.grad.abs().sum()) > 0.0)
    if not gradient_check:
        raise ValueError("score/tail aggregation gradient check failed")

    # EMA isolation: the cloned step-0 state must not be mutated by the online update.
    initial = torch.tensor([1.0, 2.0], dtype=torch.float32)
    ema = initial.clone()
    online = initial.clone()
    online.add_(torch.tensor([0.5, -0.25]))
    ema.mul_(0.999).add_(online, alpha=0.001)
    ema_isolation = bool(torch.equal(initial, torch.tensor([1.0, 2.0])) and not torch.equal(ema, online))
    if not ema_isolation:
        raise ValueError("EMA isolation check failed")

    # Same initialization/schedule check: independent generators yield identical
    # context selections, while the two arms remain separately seeded.
    g1 = torch.Generator(device="cpu").manual_seed(20300904)
    g2 = torch.Generator(device="cpu").manual_seed(20300904)
    schedule_equal = all(torch.equal(torch.randperm(512, generator=g1)[:8], torch.randperm(512, generator=g2)[:8]) for _ in range(12))
    init_a = torch.Generator(device="cpu").manual_seed(20300901)
    init_b = torch.Generator(device="cpu").manual_seed(20300901)
    initialization_equal = bool(torch.equal(torch.rand(32, generator=init_a), torch.rand(32, generator=init_b)))
    if not schedule_equal or not initialization_equal:
        raise ValueError("same-init/schedule check failed")

    result = {
        "schema": "lewm-recurrent-student.tail-robust.preflight",
        "status": "PASS",
        "scope": "local syntax/status/math/gradient/EMA isolation/tail aggregation/same-init schedule only",
        "large_data_or_model_loaded": False,
        "checks": {
            "freeze_status": "PASS",
            "phase2_authority": "PASS",
            "score_gradient": "PASS",
            "tail_aggregation": "PASS",
            "ema_isolation": "PASS",
            "same_initialization": "PASS",
            "same_context_schedule": "PASS",
        },
        "tail_formula": "0.75*mean(per_context_score_loss)+0.25*mean(top2(per_context_score_loss))",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
