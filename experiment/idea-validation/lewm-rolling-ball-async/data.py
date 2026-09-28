"""Small, episode-safe dataset adapter for the frozen Rolling Ball LeWM run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset


class RollingDataset(Dataset):
    """Return four consecutive RGB frames and their first three transitions.

    The frozen task uses history_size=3, num_preds=1, frameskip=1.  Thus
    action[k] is the recorded absolute joint target associated with frame[k]
    and is the input for predicting frame[k + 1].
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        history_size: int = 3,
        num_preds: int = 1,
        frameskip: int = 1,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        action_stats: dict[str, np.ndarray] | None = None,
    ) -> None:
        if split not in ("train", "validation"):
            raise ValueError("split must be 'train' or 'validation'")
        if (history_size, num_preds, frameskip) != (3, 1, 1):
            raise ValueError("This adapter implements only the frozen (3, 1, 1) window")

        self.root = Path(root)
        self.data_dir = self.root / "data"
        self.split = split
        self.history_size = history_size
        self.num_preds = num_preds
        self.transform = transform

        with (self.data_dir / "summary.json").open(encoding="utf-8") as stream:
            summary = json.load(stream)
        if summary.get("status") != "PASS" or summary.get("task_index") != 3:
            raise RuntimeError("Rolling Ball data summary must be PASS for task_index=3")
        with (self.data_dir / "split.json").open(encoding="utf-8") as stream:
            split_info = json.load(stream)
        self.train_episode_ids = tuple(int(x) for x in split_info["train_episode_ids"])
        self.validation_episode_ids = tuple(int(x) for x in split_info["validation_episode_ids"])
        if len(self.train_episode_ids) != 180 or len(self.validation_episode_ids) != 20:
            raise RuntimeError("Expected the frozen 180/20 episode split")
        if set(self.train_episode_ids) & set(self.validation_episode_ids):
            raise RuntimeError("Training and validation episodes overlap")
        selected = set(self.train_episode_ids if split == "train" else self.validation_episode_ids)

        self.frames = np.load(self.data_dir / "frames.npy", mmap_mode="r")
        self.actions = np.load(self.data_dir / "actions.npy", mmap_mode="r")
        self.episode_ids = np.load(self.data_dir / "episode_ids.npy", mmap_mode="r")
        self.frame_indices = np.load(self.data_dir / "frame_indices.npy", mmap_mode="r")
        if self.frames.dtype != np.uint8 or self.frames.ndim != 4 or self.frames.shape[1:] != (224, 224, 3):
            raise RuntimeError(f"Unexpected frame array shape/dtype: {self.frames.shape}/{self.frames.dtype}")
        n = len(self.frames)
        if self.actions.shape != (n, 8) or self.actions.dtype != np.float32:
            raise RuntimeError(f"Unexpected action array shape/dtype: {self.actions.shape}/{self.actions.dtype}")
        if self.episode_ids.shape != (n,) or self.frame_indices.shape != (n,):
            raise RuntimeError("Episode/frame index arrays do not match image count")

        rows_by_episode: dict[int, list[int]] = {}
        for row, (episode_id, local_frame) in enumerate(zip(self.episode_ids, self.frame_indices)):
            eid = int(episode_id)
            if eid in selected:
                rows_by_episode.setdefault(eid, []).append(row)
        expected_ids = set(self.train_episode_ids if split == "train" else self.validation_episode_ids)
        if set(rows_by_episode) != expected_ids:
            raise RuntimeError(f"Split episodes and array episodes disagree for {split}")
        self.rows_by_episode: dict[int, np.ndarray] = {}
        self.windows: list[tuple[int, int]] = []
        for eid in sorted(expected_ids):
            rows = np.asarray(rows_by_episode[eid], dtype=np.int64)
            order = np.argsort(np.asarray(self.frame_indices[rows]), kind="stable")
            rows = rows[order]
            local = np.asarray(self.frame_indices[rows], dtype=np.int64)
            if not np.array_equal(local, np.arange(26, dtype=np.int64)):
                raise RuntimeError(f"Episode {eid} does not have ordered local frames 0..25")
            self.rows_by_episode[eid] = rows
            self.windows.extend((eid, start) for start in range(26 - (history_size + num_preds) + 1))

        if action_stats is None:
            if split != "train":
                raise ValueError("Validation must reuse training action_stats")
            train_rows = np.flatnonzero(np.isin(self.episode_ids, self.train_episode_ids))
            train_actions = np.asarray(self.actions[train_rows], dtype=np.float64)
            mean = train_actions.mean(axis=0)
            std = np.maximum(train_actions.std(axis=0, ddof=1), 1e-6)
            action_stats = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
        self.action_stats = {
            "mean": np.asarray(action_stats["mean"], dtype=np.float32).reshape(8).copy(),
            "std": np.asarray(action_stats["std"], dtype=np.float32).reshape(8).copy(),
        }
        if not np.isfinite(self.action_stats["mean"]).all() or not np.isfinite(self.action_stats["std"]).all():
            raise ValueError("Action normalization statistics must be finite")
        if np.any(self.action_stats["std"] < 1e-6):
            raise ValueError("Action standard deviations must be floored at 1e-6")

    def __len__(self) -> int:
        return len(self.windows)

    def window_metadata(self, index: int) -> dict[str, Any]:
        """Indices are episode-local except the explicit source-row fields."""
        eid, start = self.windows[index]
        rows = self.rows_by_episode[eid][start : start + self.history_size + self.num_preds]
        local_frames = np.asarray(self.frame_indices[rows], dtype=np.int64)
        action_rows = rows[: self.history_size].copy()
        return {
            "episode_id": eid,
            "frame_indices": local_frames.copy(),
            "action_indices": local_frames[:-1].copy(),
            "row_indices": rows.copy(),
            "action_row_indices": action_rows,
        }

    def validate_windows(self) -> dict[str, int | bool]:
        """Check transition alignment and episode boundaries without reading images."""
        for index in range(len(self)):
            meta = self.window_metadata(index)
            if not np.all(np.diff(meta["frame_indices"]) == 1):
                raise RuntimeError("A sample contains nonconsecutive episode-local frames")
            if not np.array_equal(meta["action_indices"], meta["frame_indices"][:-1]):
                raise RuntimeError("Action local indices do not map frame[k] to frame[k+1]")
            if not np.array_equal(meta["action_row_indices"], meta["row_indices"][:-1]):
                raise RuntimeError("Action array rows do not map frame[k] to frame[k+1]")
            if not np.all(np.asarray(self.episode_ids[meta["row_indices"]]) == meta["episode_id"]):
                raise RuntimeError("A window crosses an episode boundary")
        return {"windows_checked": len(self), "transition_alignment": True, "episode_boundaries": True}

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        meta = self.window_metadata(index)
        rows = meta["row_indices"]
        image_array = np.asarray(self.frames[rows], dtype=np.uint8)
        pixels = torch.from_numpy(np.transpose(image_array, (0, 3, 1, 2)).copy())

        raw_actions = np.asarray(self.actions[meta["action_row_indices"]], dtype=np.float32)
        normalized = (raw_actions - self.action_stats["mean"]) / self.action_stats["std"]
        sample = {
            "pixels": pixels,
            "action": torch.from_numpy(np.asarray(normalized, dtype=np.float32).copy()),
        }
        if self.transform is not None:
            sample = self.transform(sample)
            if not isinstance(sample, dict) or "pixels" not in sample or "action" not in sample:
                raise TypeError("transform must return a sample dict containing pixels and action")
        if tuple(sample["pixels"].shape) != (4, 3, 224, 224):
            raise RuntimeError(f"Transform changed sample pixel shape unexpectedly: {sample['pixels'].shape}")
        if tuple(sample["action"].shape) != (3, 8):
            raise RuntimeError(f"Sample action shape is not [3, 8]: {sample['action'].shape}")
        return sample
