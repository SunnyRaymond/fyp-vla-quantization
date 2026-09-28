#!/usr/bin/env python3
"""Bounded cost-only wrapper around the pinned official LpWM training path."""
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO = Path(os.environ["LPWM_REPO"])
CAL_ROOT = Path(os.environ["LPWM_CALIBRATION_ROOT"])
ARM = os.environ["LPWM_CALIBRATION_ARM"]
assert ARM in {"sparse", "dense"}
WARMUP_UPDATES = 10
TIMED_UPDATES = 100
TOTAL_UPDATES = WARMUP_UPDATES + TIMED_UPDATES
ARM_DIR = CAL_ROOT / ARM
ARM_DIR.mkdir(parents=True, exist_ok=True)
UPDATES_PATH = ARM_DIR / "updates.jsonl"
SUMMARY_PATH = ARM_DIR / "training_summary.json"


class CalibrationComplete(Exception):
    """Intentional stop after the fixed cost-calibration update count."""


def write_json(path: Path, payload: dict) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    partial.replace(path)


class BoundedTimedLoader:
    """Preserve the real prepared loader and stop after exactly 110 official updates."""
    def __init__(self, loader):
        self.loader = loader
        self.completed_updates = 0
        self.warmup_peak_allocated = None
        self.warmup_peak_reserved = None

    def __len__(self):
        return TOTAL_UPDATES

    def __iter__(self):
        iterator = iter(self.loader)
        with UPDATES_PATH.open("w", encoding="utf-8", buffering=1) as timings:
            while True:
                update_index = self.completed_updates + 1
                started = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    return
                batch_ready = time.perf_counter()
                yield batch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                finished = time.perf_counter()
                self.completed_updates += 1
                phase = "warmup" if update_index <= WARMUP_UPDATES else "timed"
                record = {
                    "update": update_index,
                    "phase": phase,
                    "batch_wait_seconds": batch_ready - started,
                    "batch_to_update_complete_seconds": finished - started,
                }
                timings.write(json.dumps(record) + "\n")
                timings.flush()
                if update_index == WARMUP_UPDATES and torch.cuda.is_available():
                    self.warmup_peak_allocated = torch.cuda.max_memory_allocated()
                    self.warmup_peak_reserved = torch.cuda.max_memory_reserved()
                    torch.cuda.reset_peak_memory_stats()
                if self.completed_updates == TOTAL_UPDATES:
                    raise CalibrationComplete("fixed 110-update cost calibration reached")


import train as official_train

original_init = official_train.Trainer.__init__


def timed_init(self, cfg):
    expected = {
        "env": "pusht",
        "frameskip": 5,
        "num_hist": 3,
        "epochs": 2,
        "batch_size": 64,
        "num_workers": 20,
        "predictor": "mlp_var",
        "proj_dim": 384,
        "action_emb_dim": 384,
        "mup": True,
        "seed": 0,
        "link": "reprelu" if ARM == "sparse" else "identity",
        "target_p": 1 if ARM == "sparse" else 2,
        "reg_weight": 0.1 if ARM == "sparse" else 0.01,
        "mup_lr": 5e-4 if ARM == "sparse" else 5e-5,
    }
    actual = {
        "env": str(cfg.env.name),
        "frameskip": int(cfg.frameskip),
        "num_hist": int(cfg.num_hist),
        "epochs": int(cfg.training.epochs),
        "batch_size": int(cfg.training.batch_size),
        "num_workers": int(cfg.env.num_workers),
        "predictor": str(cfg.predictor.mode),
        "proj_dim": int(cfg.encoder.proj_dim),
        "action_emb_dim": int(cfg.action_emb_dim),
        "mup": bool(cfg.mup),
        "seed": int(cfg.training.seed),
        "link": str(cfg.link.kind),
        "target_p": int(cfg.target_p),
        "reg_weight": float(cfg.reg_weight),
        "mup_lr": float(cfg.training.mup_lr),
        "mu_resolved": float(cfg.mu),
        "mu_override_applied": ARM == "sparse",
        "regularizer_config": str(cfg.regularizer._target_),
    }
    exact = all(actual[key] == value for key, value in expected.items())
    composed = {
        "status": "COMPOSED_CONFIG_PASS" if exact else "COMPOSED_CONFIG_MISMATCH",
        "hydra_config_name": "train_rdmreg.yaml",
        "hydra_config_path": str(REPO / "conf"),
        "arm": ARM,
        "expected": expected,
        "actual": actual,
        "full_epochs_executed": False,
        "note": "Calibration wrapper stops after the fixed 110-update cost sample; composed formal epoch count remains 2.",
    }
    write_json(ARM_DIR / "composed_config_identity.json", composed)
    if not exact:
        raise RuntimeError("Hydra-composed configuration does not match the frozen calibration cell")
    started = time.perf_counter()
    original_init(self, cfg)
    self._calibration_init_seconds = time.perf_counter() - started


def bounded_run(self):
    full_train_loader = self.dataloaders["train"]
    full_train_batches_per_epoch = len(full_train_loader)
    measured_loader = BoundedTimedLoader(full_train_loader)
    self.dataloaders["train"] = measured_loader
    self.epoch = 1
    train_started = time.perf_counter()
    try:
        # This is the unchanged official Trainer.train() forward/loss/backward/
        # optimizer path. The iterator raises only after update 110 completes.
        self.train()
    except CalibrationComplete:
        pass
    train_seconds = time.perf_counter() - train_started
    if measured_loader.completed_updates != TOTAL_UPDATES:
        raise RuntimeError(f"expected {TOTAL_UPDATES} updates; got {measured_loader.completed_updates}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        timed_peak_allocated = torch.cuda.max_memory_allocated()
        timed_peak_reserved = torch.cuda.max_memory_reserved()
        torch.cuda.reset_peak_memory_stats()
    validation_started = time.perf_counter()
    self.val()  # cost only; validation metrics are intentionally not exported
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        validation_peak_allocated = torch.cuda.max_memory_allocated()
        validation_peak_reserved = torch.cuda.max_memory_reserved()
    else:
        validation_peak_allocated = None
        validation_peak_reserved = None
    validation_seconds = time.perf_counter() - validation_started

    checkpoint_started = time.perf_counter()
    self.accelerator.wait_for_everyone()
    ckpt_path, model_name, model_epoch = self.save_ckpt()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    checkpoint_seconds = time.perf_counter() - checkpoint_started

    rows = [json.loads(line) for line in UPDATES_PATH.read_text(encoding="utf-8").splitlines()]
    timed = [row["batch_to_update_complete_seconds"] for row in rows if row["phase"] == "timed"]
    batch_wait = [row["batch_wait_seconds"] for row in rows if row["phase"] == "timed"]
    summary = {
        "status": "CALIBRATION_COMPLETE",
        "arm": ARM,
        "run_name": os.environ["LPWM_CALIBRATION_RUN_NAME"],
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch_label": model_epoch,
        "training_config_epochs_unchanged": int(self.cfg.training.epochs),
        "batch_size": int(self.cfg.training.batch_size),
        "num_workers": int(self.cfg.env.num_workers),
        "official_full_train_batches_per_epoch": full_train_batches_per_epoch,
        "calibration_updates_total": measured_loader.completed_updates,
        "warmup_updates": WARMUP_UPDATES,
        "timed_updates": len(timed),
        "timed_update_seconds_mean": statistics.mean(timed),
        "timed_update_seconds_median": statistics.median(timed),
        "timed_update_seconds_p90": sorted(timed)[int(0.9 * (len(timed) - 1))],
        "timed_batch_wait_seconds_mean": statistics.mean(batch_wait),
        "trainer_init_seconds": self._calibration_init_seconds,
        "110_updates_loop_seconds": train_seconds,
        "one_full_official_validation_pass_seconds": validation_seconds,
        "one_official_checkpoint_save_seconds": checkpoint_seconds,
        "warmup_peak_gpu_allocated_bytes": measured_loader.warmup_peak_allocated,
        "warmup_peak_gpu_reserved_bytes": measured_loader.warmup_peak_reserved,
        "timed_peak_gpu_allocated_bytes": timed_peak_allocated if torch.cuda.is_available() else None,
        "timed_peak_gpu_reserved_bytes": timed_peak_reserved if torch.cuda.is_available() else None,
        "validation_peak_gpu_allocated_bytes": validation_peak_allocated,
        "validation_peak_gpu_reserved_bytes": validation_peak_reserved,
        "formal_training_updates_per_arm_2_epochs": full_train_batches_per_epoch * int(self.cfg.training.epochs),
        "metric_values_exported": False,
        "calibration_checkpoint_is_official_reproduction_checkpoint": False,
    }
    write_json(SUMMARY_PATH, summary)
    complete_path = ARM_DIR / "CALIBRATION_COMPLETE"
    complete_path.write_text("CALIBRATION_COMPLETE\n", encoding="ascii")
    print(f"CALIBRATION_COMPLETE arm={ARM} updates={measured_loader.completed_updates} summary={SUMMARY_PATH}")


official_train.Trainer.__init__ = timed_init
official_train.Trainer.run = bounded_run

link, target_p, reg_weight, mup_lr = (
    ("reprelu", "1", "0.1", "5e-4") if ARM == "sparse"
    else ("identity", "2", "0.01", "5e-5")
)
run_name = os.environ["LPWM_CALIBRATION_RUN_NAME"]
run_dir = Path(os.environ["CKPT_BASE"]) / "outputs" / run_name
sys.argv = [
    str(REPO / "train.py"), "--config-path", str(REPO / "conf"),
    "--config-name", "train_rdmreg.yaml",
    "env=pusht", "frameskip=5", "num_hist=3", "encoder=vit_scratch",
    f"link={link}", "regularizer=rdmreg", f"target_p={target_p}", "agg=b",
    "training.epochs=2", "training.batch_size=64", "env.num_workers=20",
    f"ckpt_base_path={os.environ['CKPT_BASE']}", f"hydra.run.dir={run_dir}",
    "hydra.job.chdir=true", "predictor=mlp_var", f"training.mup_lr={mup_lr}",
    f"reg_weight={reg_weight}", "mup=true", "training.seed=0",
    "encoder.proj_dim=384", "action_emb_dim=384",
]
if ARM == "sparse":
    sys.argv.append("mu=0")

official_train.main()
