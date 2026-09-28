"""Train the pinned, vanilla LeWM objective on prepared Rolling Ball episodes.

This entry point deliberately refuses to import Torch/LeWM outside a real PBS
compute allocation. It does not run the Isaac benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time


LEWM_COMMIT = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"
STABLE_WM_COMMIT = "10c26dbd5677083fa31dba69eb738b973845e9a4"
DATASET_COMMIT = "9295b6e9878609a992047f0b8b65421a493299e7"
HISTORY_SIZE = 3
NUM_PREDS = 1
FRAMESKIP = 1
FORMAL_EPOCHS = 100


def require_pbs_compute_node() -> tuple[str, str]:
    """Fail before importing numerical/model packages unless on allocated node."""
    job_id = os.environ.get("PBS_JOBID", "").strip()
    nodefile_name = os.environ.get("PBS_NODEFILE", "").strip()
    if not job_id or not nodefile_name:
        raise RuntimeError("PBS_JOBID and PBS_NODEFILE are required")
    nodefile = Path(nodefile_name)
    if not nodefile.is_file():
        raise RuntimeError("PBS_NODEFILE does not name a readable allocation file")

    host = socket.gethostname().split(".")[0].lower()
    if any(tag in host for tag in ("login", "head", "submit")):
        raise RuntimeError(f"Refusing non-compute host: {host}")
    allocated_hosts = {
        line.strip().split(".")[0].lower()
        for line in nodefile.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if host not in allocated_hosts:
        raise RuntimeError(f"Current host {host} is absent from PBS_NODEFILE")
    return job_id, host


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path,
                        help="Task root containing data/, upstream_lewm/, and source manifests")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-epochs", type=int, default=FORMAL_EPOCHS)
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--limit-val-batches", type=int)
    parser.add_argument("--alignment-check-only", action="store_true",
                        help="Validate one dataset sample on this PBS allocation, then exit")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def main() -> int:
    args = parse_args()
    job_id, host = require_pbs_compute_node()

    # Imports that can load numerical runtimes or model code stay below the PBS guard.
    import importlib.metadata
    import numpy as np
    import torch
    import lightning as pl
    import stable_pretraining as spt
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf, open_dict
    from torch.utils.data import DataLoader

    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    data_dir = root / "data"
    lewm_dir = root / "upstream_lewm"
    source_id = read_json(lewm_dir / "PINNED.json")
    stable_id_path = root / "upstream_stablewm" / "PINNED.json"
    stable_id = read_json(stable_id_path)
    if source_id.get("commit") != LEWM_COMMIT:
        raise RuntimeError("upstream_lewm/PINNED.json does not match the frozen vanilla LeWM commit")
    if stable_id.get("commit") != STABLE_WM_COMMIT:
        raise RuntimeError("upstream_stablewm/PINNED.json does not match the frozen stable-worldmodel commit")
    if source_id.get("repository") != "lucas-maes/le-wm":
        raise RuntimeError("Unexpected LeWM source repository")
    if stable_id.get("repository") != "galilai-group/stable-worldmodel":
        raise RuntimeError("Unexpected stable-worldmodel source repository")
    for required_source in ("train.py", "utils.py", "jepa.py", "module.py",
                            "config/train/lewm.yaml", "config/train/model/lewm.yaml"):
        if not (lewm_dir / required_source).is_file():
            raise RuntimeError(f"Pinned LeWM source is missing {required_source}")

    frozen = read_json(root / "source" / "TRAIN_FREEZE.json")
    if frozen.get("source", {}).get("lewm") != f"lucas-maes/le-wm@{LEWM_COMMIT}":
        raise RuntimeError("TRAIN_FREEZE.json has a different LeWM identity")
    if frozen.get("source", {}).get("dataset") != f"cyx337/ReflexBench_dataset@{DATASET_COMMIT}":
        raise RuntimeError("TRAIN_FREEZE.json has a different dataset identity")
    summary = read_json(data_dir / "summary.json")
    if summary.get("status") != "PASS":
        raise RuntimeError("Prepared dataset summary is not PASS")
    split = read_json(data_dir / "split.json")
    train_episode_ids = [int(x) for x in split["train_episode_ids"]]
    val_episode_ids = [int(x) for x in split["validation_episode_ids"]]
    if len(train_episode_ids) != 180 or len(val_episode_ids) != 20:
        raise RuntimeError("Episode split differs from the frozen 180/20 split")
    if set(train_episode_ids) & set(val_episode_ids):
        raise RuntimeError("Training and validation episode IDs overlap")

    source_dir = root / "upstream_lewm"
    sys.path.insert(0, str(source_dir))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data import RollingDataset
    from utils import get_img_preprocessor

    image_transform = get_img_preprocessor(source="pixels", target="pixels", img_size=224)
    common = dict(history_size=HISTORY_SIZE, num_preds=NUM_PREDS, frameskip=FRAMESKIP,
                  transform=image_transform)
    train_set = RollingDataset(root, split="train", **common)
    action_stats = train_set.action_stats
    val_set = RollingDataset(root, split="validation", action_stats=action_stats, **common)

    def alignment_check() -> dict:
        sample = train_set[0]
        pixels = sample["pixels"]
        action = sample["action"]
        if tuple(pixels.shape) != (HISTORY_SIZE + NUM_PREDS, 3, 224, 224):
            raise RuntimeError(f"Unexpected sample pixels shape: {tuple(pixels.shape)}")
        if tuple(action.shape) != (HISTORY_SIZE, 8):
            raise RuntimeError(f"Unexpected sample action shape: {tuple(action.shape)}")
        if not torch.isfinite(action).all():
            raise RuntimeError("Sample actions contain NaN or infinity")

        meta = train_set.window_metadata(0)
        frame_indices = torch.as_tensor(meta["frame_indices"]).reshape(-1)
        action_indices = torch.as_tensor(meta["action_indices"]).reshape(-1)
        row_indices = torch.as_tensor(meta["row_indices"]).reshape(-1)
        action_row_indices = torch.as_tensor(meta["action_row_indices"]).reshape(-1)
        if frame_indices.numel() != HISTORY_SIZE + NUM_PREDS:
            raise RuntimeError("window_metadata returned the wrong number of frames")
        if action_indices.numel() != HISTORY_SIZE:
            raise RuntimeError("window_metadata returned the wrong number of actions")
        if row_indices.numel() != HISTORY_SIZE + NUM_PREDS or action_row_indices.numel() != HISTORY_SIZE:
            raise RuntimeError("window_metadata returned the wrong number of source rows")
        if not torch.equal(frame_indices[1:] - frame_indices[:-1], torch.ones_like(frame_indices[:-1])):
            raise RuntimeError("Sample frames are not contiguous at frameskip=1")
        if not torch.equal(action_indices, frame_indices[:-1]):
            raise RuntimeError("Actions are not aligned to frame[k] -> frame[k+1]")
        if not torch.equal(action_row_indices, row_indices[:-1]):
            raise RuntimeError("Action source rows do not align with their source frames")
        episode_id = int(meta["episode_id"])
        if episode_id not in train_episode_ids:
            raise RuntimeError("The training window belongs to an episode outside the train split")
        source_episode_ids = np.load(data_dir / "episode_ids.npy", mmap_mode="r")
        if any(int(source_episode_ids[int(row)]) != episode_id for row in row_indices.tolist()):
            raise RuntimeError("Source image rows cross an episode boundary")
        raw_actions = np.load(data_dir / "actions.npy", mmap_mode="r")
        raw_window_actions = np.asarray(raw_actions[action_row_indices.numpy()], dtype=np.float32)
        expected_actions = (raw_window_actions - action_stats["mean"]) / action_stats["std"]
        if not np.allclose(action.numpy(), expected_actions, rtol=1e-5, atol=1e-6):
            raise RuntimeError("Returned action values do not match train-only normalized source actions")
        return {"status": "PASS", "pixels_shape": list(pixels.shape),
                "actions_shape": list(action.shape),
                "source_row_alignment_checked": True,
                "episode_boundary_checked": True,
                "train_only_normalization_checked": True,
                "episode_id": episode_id}

    train_window_validation = train_set.validate_windows()
    val_window_validation = val_set.validate_windows()
    for name, result in (("train", train_window_validation), ("validation", val_window_validation)):
        dataset = train_set if name == "train" else val_set
        if (not isinstance(result, dict) or int(result.get("windows_checked", 0)) != len(dataset)
                or result.get("transition_alignment") is not True
                or result.get("episode_boundaries") is not True):
            raise RuntimeError(f"RollingDataset.validate_windows() did not confirm {name} alignment")
    alignment = alignment_check()
    alignment["validate_windows"] = {
        "train": train_window_validation,
        "validation": val_window_validation,
    }
    if args.alignment_check_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "alignment_check.json").write_text(
            json.dumps({**alignment, "job_id": job_id, "host": host}, indent=2) + "\n",
            encoding="utf-8")
        print(json.dumps({**alignment, "job_id": job_id, "host": host}), flush=True)
        return 0

    smoke = (args.max_epochs != FORMAL_EPOCHS or args.limit_train_batches is not None
             or args.limit_val_batches is not None)
    if args.max_epochs <= 0:
        raise ValueError("--max-epochs must be positive")
    if not smoke and (args.limit_train_batches is not None or args.limit_val_batches is not None):
        raise RuntimeError("Batch limits are smoke-only; use the frozen 100-epoch formal run")
    if not torch.cuda.is_available():
        raise RuntimeError("LeWM training requires the allocated CUDA GPU")

    pl.seed_everything(0, workers=True)
    torch.backends.cudnn.benchmark = False
    with initialize_config_dir(version_base=None, config_dir=str(source_dir / "config" / "train")):
        cfg = compose(config_name="lewm")
    with open_dict(cfg):
        cfg.seed = 0
        cfg.history_size = HISTORY_SIZE
        cfg.num_preds = NUM_PREDS
        cfg.img_size = 224
        cfg.num_workers = 2
        cfg.data.dataset.frameskip = FRAMESKIP
        cfg.data.dataset.num_steps = HISTORY_SIZE + NUM_PREDS
        cfg.data.dataset.name = "prepared_rolling_ball_arrays"
        cfg.data.dataset.keys_to_load = ["pixels", "action"]
        cfg.trainer.max_epochs = args.max_epochs
        cfg.subdir = output_dir.name
        cfg.loader.batch_size = 128
        cfg.loader.num_workers = 2
        cfg.model.action_encoder.input_dim = 8

    generator = torch.Generator().manual_seed(0)
    loader_args = dict(batch_size=128, num_workers=2, pin_memory=True,
                       persistent_workers=True, prefetch_factor=3)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True,
                              generator=generator, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    # Match vanilla train.py: Hydra-instantiated cfg.model, its LeJEPA forward,
    # official SIGReg, AdamW and epoch scheduler.
    import importlib.util
    from functools import partial

    spec = importlib.util.spec_from_file_location("vanilla_lewm_train", source_dir / "train.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the pinned upstream LeWM train.py")
    upstream_train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream_train)
    model = instantiate(cfg.model)
    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        }
    }
    module = spt.Module(
        model=model,
        sigreg=upstream_train.SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(upstream_train.lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep Manager's logs local to this PBS run; raw checkpoints are saved below.
    spt.set(cache_dir=str(output_dir / "spt_cache"), requeue_checkpoint=False)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    (output_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    terminal_frames = train_terminal_indices(data_dir, train_episode_ids, np)
    (output_dir / "train_goal_bank_indices.json").write_text(
        json.dumps({"source": "train split terminal frames only", "indices": terminal_frames}, indent=2) + "\n",
        encoding="utf-8")
    action_bounds = training_action_bounds(data_dir, train_episode_ids, np)
    metadata = {
        "lewm": source_id,
        "stable_worldmodel": stable_id,
        "dataset": f"cyx337/ReflexBench_dataset@{DATASET_COMMIT}",
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "stable-pretraining", "stable-worldmodel", "lightning", "hydra-core")
        },
        "job_id": job_id,
        "host": host,
        "smoke_only": smoke,
    }
    checkpoint_action_stats = {
        key: np.asarray(value, dtype=np.float32).reshape(8).tolist()
        for key, value in action_stats.items()
    }
    checkpoint_action_bounds = {
        "min": np.asarray(action_bounds["min"], dtype=np.float32).reshape(8).tolist(),
        "max": np.asarray(action_bounds["max"], dtype=np.float32).reshape(8).tolist(),
        "meaning": action_bounds["meaning"],
    }
    callback = EpochArtifacts(output_dir, smoke, cfg_dict, checkpoint_action_stats, split,
                              metadata, terminal_frames, checkpoint_action_bounds, torch)
    callbacks = [callback]
    trainer_kwargs = dict(cfg.trainer)
    trainer_kwargs.update(default_root_dir=str(output_dir), callbacks=callbacks,
                          num_sanity_val_steps=1, logger=None,
                          enable_checkpointing=False)
    if args.limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = args.limit_val_batches
    trainer = pl.Trainer(**trainer_kwargs)
    data_module = spt.data.DataModule(train=train_loader, val=val_loader)
    started = time.time()
    manager = spt.Manager(trainer=trainer, module=module, data=data_module, seed=0, ckpt_path=None)
    manager()
    completed_epochs = len(callback.history)
    expected_epochs = list(range(1, completed_epochs + 1))
    formal_complete = (not smoke and expected_epochs == list(range(1, FORMAL_EPOCHS + 1))
                       and callback.last_checkpoint_epoch == FORMAL_EPOCHS)
    status = "SMOKE_ONLY" if smoke else ("PASS_epoch100" if formal_complete else "INCOMPLETE")
    final = {
        "status": status,
        "job_id": job_id,
        "host": host,
        "smoke_only": smoke,
        "completed_epochs": completed_epochs,
        "max_epochs": args.max_epochs,
        "elapsed_seconds": time.time() - started,
        "best_val_prediction_mse": callback.best_val_prediction_mse,
        "best_val_is_diagnostic_only": True,
        "baseline_checkpoint": "last.ckpt (epoch 100)" if formal_complete else None,
        "alignment": alignment,
        "closed_loop_evaluated": False,
        "task_success_claim": False,
        "source": metadata,
    }
    (output_dir / "summary.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(final), flush=True)
    if not smoke and status != "PASS_epoch100":
        return 2
    return 0


def train_terminal_indices(data_dir: Path, train_episode_ids: list[int], np) -> list[dict]:
    episode_ids = np.load(data_dir / "episode_ids.npy", mmap_mode="r")
    frame_indices = np.load(data_dir / "frame_indices.npy", mmap_mode="r")
    train_ids = set(train_episode_ids)
    terminal = {}
    for row, (episode_id, frame_index) in enumerate(zip(episode_ids, frame_indices)):
        episode_id = int(episode_id)
        if episode_id in train_ids:
            old = terminal.get(episode_id)
            if old is None or int(frame_index) > old[0]:
                terminal[episode_id] = (int(frame_index), row)
    if set(terminal) != train_ids:
        raise RuntimeError("Could not derive one terminal frame for each training episode")
    return [{"episode_id": episode_id, "dataset_row": terminal[episode_id][1]}
            for episode_id in sorted(train_ids)]


def training_action_bounds(data_dir: Path, train_episode_ids: list[int], np) -> dict:
    episode_ids = np.load(data_dir / "episode_ids.npy", mmap_mode="r")
    actions = np.load(data_dir / "actions.npy", mmap_mode="r")
    mask = np.isin(episode_ids, np.asarray(train_episode_ids, dtype=episode_ids.dtype))
    rows = np.asarray(actions[mask], dtype=np.float32)
    rows = rows[np.isfinite(rows).all(axis=1)]
    if rows.ndim != 2 or rows.shape[1] != 8 or not len(rows):
        raise RuntimeError("Could not derive finite 8-D action bounds from training episodes")
    return {"min": rows.min(axis=0), "max": rows.max(axis=0),
            "meaning": "raw demonstrated absolute joint targets; not validated safe CEM limits"}


class EpochArtifacts:
    """Defined lazily after the PBS guard so importing this file stays lightweight."""

    def __new__(cls, *args, **kwargs):
        import lightning as pl
        import math

        class _Callback(pl.Callback):
            def __init__(self, output_dir, smoke, cfg, action_stats, split, source,
                         terminal_frames, action_bounds, torch):
                super().__init__()
                self.output_dir = output_dir
                self.smoke = smoke
                self.cfg = cfg
                self.action_stats = action_stats
                self.split = split
                self.source = source
                self.terminal_frames = terminal_frames
                self.action_bounds = action_bounds
                self.torch = torch
                self.best_val_prediction_mse = None
                self.last_checkpoint_epoch = None
                self.history = []
                self.values = {"train": {}, "val": {}}

            def on_train_epoch_start(self, trainer, pl_module):
                self.values = {"train": {}, "val": {}}

            def capture(self, stage, outputs, batch):
                if not isinstance(outputs, dict):
                    return
                weight = int(batch["pixels"].shape[0])
                for key in ("pred_loss", "sigreg_loss", "loss"):
                    value = outputs.get(key)
                    if value is None or not self.torch.is_tensor(value):
                        continue
                    scalar = float(value.detach().float().cpu())
                    if not math.isfinite(scalar):
                        raise FloatingPointError(f"Nonfinite {stage}/{key} captured during training")
                    pair = self.values[stage].setdefault(key, [0.0, 0])
                    pair[0] += scalar * weight
                    pair[1] += weight

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                self.capture("train", outputs, batch)

            def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                        batch_idx, dataloader_idx=0):
                if not trainer.sanity_checking:
                    self.capture("val", outputs, batch)

            def on_validation_epoch_end(self, trainer, pl_module):
                if trainer.sanity_checking:
                    return
                epoch = int(trainer.current_epoch) + 1
                if epoch != len(self.history) + 1:
                    raise RuntimeError(
                        f"Validation epoch sequence is not continuous: expected {len(self.history) + 1}, got {epoch}")
                metrics = {"epoch": epoch,
                           "train": self.means("train"), "validation": self.means("val")}
                for stage in ("train", "val"):
                    for key, value in self.values[stage].items():
                        if value[1] and not math.isfinite(value[0] / value[1]):
                            raise FloatingPointError(f"Nonfinite {stage}/{key} captured in epoch {epoch}")
                    for name, value in trainer.callback_metrics.items():
                        if str(name).startswith(stage + "/") and "loss" in str(name):
                            scalar = (float(value.detach().float().cpu())
                                      if hasattr(value, "detach") else float(value))
                            if not math.isfinite(scalar):
                                raise FloatingPointError(f"Nonfinite {name} captured in epoch {epoch}")
                pred = metrics["validation"].get("pred_loss")
                if pred is None:
                    pred = self.logged_metric(trainer, "val/pred_loss", "val/pred_loss_epoch")
                    if pred is not None:
                        metrics["validation"]["pred_loss"] = pred
                if pred is None or not math.isfinite(pred):
                    raise RuntimeError("Vanilla LeWM val/pred_loss was not available for checkpoint selection")
                metrics["val_prediction_mse"] = pred
                self.history.append(metrics)
                with (self.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(metrics) + "\n")
                is_best = self.best_val_prediction_mse is None or pred < self.best_val_prediction_mse
                if is_best:
                    self.best_val_prediction_mse = pred
                if self.smoke:
                    return
                model_state = {key: value.detach().cpu() for key, value in pl_module.model.state_dict().items()}
                artifact = {
                    "format": "vanilla_lewm_raw_model_state_v1",
                    "model_state_dict": model_state,
                    "model_config": self.cfg["model"],
                    "resolved_training_config": self.cfg,
                    "action_stats": self.action_stats,
                    "action_bounds": self.action_bounds,
                    "split": self.split,
                    "source": self.source,
                    "train_goal_bank_terminal_frames": self.terminal_frames,
                    "selection_metric": "last completed epoch; formal baseline is epoch 100",
                    "epoch": metrics["epoch"],
                }
                self.torch.save(artifact, self.output_dir / "last.ckpt")
                self.last_checkpoint_epoch = metrics["epoch"]
                if is_best:
                    diagnostic = dict(artifact)
                    diagnostic["selection_metric"] = "validation prediction MSE (diagnostic only)"
                    self.torch.save(diagnostic, self.output_dir / "best_val_pred.ckpt")

            def means(self, stage):
                return {key: (value[0] / value[1]) for key, value in self.values[stage].items()
                        if value[1] > 0}

            @staticmethod
            def logged_metric(trainer, *names):
                for name in names:
                    value = trainer.callback_metrics.get(name)
                    if value is not None:
                        return (float(value.detach().float().cpu())
                                if hasattr(value, "detach") else float(value))
                return None

        return _Callback(*args, **kwargs)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"train_baseline failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
