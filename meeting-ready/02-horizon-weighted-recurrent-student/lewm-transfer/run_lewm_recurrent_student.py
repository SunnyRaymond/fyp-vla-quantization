#!/usr/bin/env python3
"""LeWM PushT horizon-weighted recurrent student runner.

The runner is deliberately contract-first.  It refuses checkpoint/HDF5/model
work until the GPU interface probe has produced an authoritative JSON.  The
probe supplies the runtime observation history and official CEM rollout shape;
the latter is never inferred from the frozen horizon.

The small module and the data/solver adapters in this file are intended to be
used by the eventual PBS run.  ``--mode status`` is safe on a login node and is
also useful while the probe is pending.  ``--mode run`` requires a PBS compute
allocation and a validated probe.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "lewm-recurrent-student.runner"
LATENT_DIM = 192
HISTORY_LENGTH = 3
ACTION_DIM = 10
HORIZON = 5
HIDDEN_DIM = 256
HORIZON_WEIGHTS = (1.0 / 3.0, 2.0 / 3.0, 1.0, 4.0 / 3.0, 5.0 / 3.0)
SNAPSHOT_STEPS = (500, 1000, 1500)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lewm-root", type=Path, required=True)
    parser.add_argument("--stablewm-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--interface-probe", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--prepared-rows", type=Path, default=None)
    parser.add_argument("--action-low", type=float, nargs=2, default=None, metavar=("X", "Y"))
    parser.add_argument("--action-high", type=float, nargs=2, default=None, metavar=("X", "Y"))
    parser.add_argument("--gaussian-std", type=float, default=None)
    parser.add_argument("--mode", choices=("status", "run"), default="status")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    return value


def require_compute_node() -> None:
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("PBS_JOBID is required for model/HDF5 work")
    host = platform.node().lower()
    if any(token in host for token in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login host: {host}")


def validate_freeze(freeze: Mapping[str, Any]) -> dict[str, Any]:
    if freeze.get("schema") != "jepa-action-prefix-compiler.lewm-horizon-weighted-recurrent-transfer-freeze":
        raise ValueError("unexpected LeWM recurrent-student freeze schema")
    if int(freeze.get("schema_version", -1)) != 1:
        raise ValueError("unsupported LeWM recurrent-student freeze schema version")
    scope = freeze["scope"]
    student = freeze["student"]
    schedule = freeze["shared_data_and_schedule"]
    training = schedule["training"]
    if int(scope["latent_dim"]) != LATENT_DIM or int(scope["packed_action_token_dim"]) != ACTION_DIM:
        raise ValueError("frozen LeWM dimensions are not 192-D/10-D")
    if int(scope["horizon"]) != HORIZON or int(scope["policy_initial_history_length"]) != 1 or int(scope["predictor_max_history_length"]) != HISTORY_LENGTH:
        raise ValueError("frozen LeWM policy/predictor history contract changed")
    if int(student["hidden_dim"]) != HIDDEN_DIM:
        raise ValueError("frozen student hidden_dim changed")
    if bool(scope["student_receives_goal"]) or bool(scope["student_encode_obs_calls"]):
        raise ValueError("student must remain goal-free and encoder-free")
    if bool(scope["student_contains_teacher_or_source_parameters"]):
        raise ValueError("student must not contain teacher/source parameters")
    if bool(scope["teacher_forcing"]) or not bool(scope["free_running_feedback"]):
        raise ValueError("student must use predicted-latent free-running feedback")
    forbidden = {str(item).lower() for item in student["forbidden"]}
    if "teacher forcing" not in forbidden or "goal input" not in forbidden:
        raise ValueError("student forbidden list is incomplete")
    if [float(x) for x in training["loss"]["horizon_weights"]] != list(HORIZON_WEIGHTS):
        raise ValueError("horizon weights drifted from the frozen treatment")
    if int(training["steps"]) != 1500 or int(training["batch_contexts"]) != 8:
        raise ValueError("training budget/batch changed")
    return {
        "latent_dim": LATENT_DIM,
        "history_length": HISTORY_LENGTH,
        "action_dim": ACTION_DIM,
        "horizon": HORIZON,
        "hidden_dim": HIDDEN_DIM,
        "weights": list(HORIZON_WEIGHTS),
        "steps": int(training["steps"]),
        "batch_contexts": int(training["batch_contexts"]),
        "snapshot_steps": list(SNAPSHOT_STEPS),
        "seeds": dict(schedule["seeds"]),
    }


@dataclass(frozen=True)
class InterfaceContract:
    """Runtime facts observed from the official LeWM implementation."""

    observation_history_h: int
    official_future_prediction_count: int
    official_predicted_emb_shape: tuple[int, ...]
    student_probe_future_count: int
    candidate_shape: tuple[int, ...]
    semantics_equal: bool
    raw_action_dim: int | None

    @property
    def probe_shape_count_match(self) -> bool:
        """Metadata only; no planner/CEM path consumes this value."""
        return self.semantics_equal and self.official_future_prediction_count == HORIZON


def load_interface_contract(path: Path) -> InterfaceContract:
    if not path.is_file():
        raise FileNotFoundError(path)
    probe = load_json(path)
    if probe.get("schema") != "lewm-recurrent-student.interface-probe":
        raise ValueError("interface probe schema is not authoritative LeWM probe output")
    if probe.get("status") != "PASS":
        raise ValueError(f"interface probe status is {probe.get('status')!r}")
    rollout = probe.get("official_jepa_rollout", {})
    rolling = probe.get("rolling_latest3_five_targets", {})
    cem = probe.get("cem_candidate", {})
    mapping = probe.get("training_and_stage_b_mapping", {})
    shape = rollout.get("predicted_emb_shape")
    candidate_shape = cem.get("shape")
    if not isinstance(shape, list) or not isinstance(candidate_shape, list):
        raise ValueError("probe is missing official predicted/candidate shapes")
    h = int(rollout["H"])
    future_count = int(rollout["future_predictions_beyond_initial_H"])
    student_count = int(rolling["future_target_count"])
    semantics_equal = bool(mapping.get("semantics_equal_to_official_cem_rollout", False))
    values = probe.get("solver", {})
    raw_action_dim = values.get("raw_action_dim")
    return InterfaceContract(
        observation_history_h=h,
        official_future_prediction_count=future_count,
        official_predicted_emb_shape=tuple(int(x) for x in shape),
        student_probe_future_count=student_count,
        candidate_shape=tuple(int(x) for x in candidate_shape),
        semantics_equal=semantics_equal,
        raw_action_dim=None if raw_action_dim is None else int(raw_action_dim),
    )


def waiting_status(freeze_path: Path, protocol_path: Path, probe_path: Path) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "WAITING_FOR_INTERFACE_PROBE",
        "freeze": str(freeze_path.resolve()),
        "protocol": str(protocol_path.resolve()),
        "interface_probe": str(probe_path.resolve()),
        "wait_for": [
            "authoritative interface_probe.json with status=PASS",
            "runtime H, official future prediction count, predicted_emb shape, and action preprocessing",
        ],
        "model_work_started": False,
        "stage_b": {
            "status": "PENDING_INTERFACE_PROBE",
            "prediction_count_assumption": None,
            "shape_assumption": None,
        },
        "claim_boundary": "No training, predictor metric, CEM metric, or GO/NO-GO conclusion is produced while the probe is absent.",
    }


def _left_pad_history(history: Any, length: int = HISTORY_LENGTH) -> Any:
    """Left-pad H=1/2 official histories without changing the current latent."""
    import torch

    if history.ndim != 3:
        raise ValueError(f"latent history must be [B,L,D], got {tuple(history.shape)}")
    if history.shape[-1] != LATENT_DIM:
        raise ValueError("latent history width must be 192")
    if history.shape[1] < 1:
        raise ValueError("latent history cannot be empty")
    if history.shape[1] > length:
        return history[:, -length:]
    if history.shape[1] == length:
        return history
    pad = history[:, :1].expand(-1, length - history.shape[1], -1)
    return torch.cat((pad, history), dim=1)


class LeWMCompactRecurrentTransitionStudent:  # constructed lazily to keep status mode dependency-free
    """Shared h256 residual recurrent student on 192-D LeWM compact latents."""

    def __new__(cls) -> Any:
        import torch.nn as nn

        class _Impl(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.history_adapter = nn.Linear(HISTORY_LENGTH * LATENT_DIM, HIDDEN_DIM)
                self.latent_projection = nn.Linear(LATENT_DIM, HIDDEN_DIM)
                self.action_projection = nn.Linear(ACTION_DIM, HIDDEN_DIM)
                self.shared_transition = nn.Sequential(
                    nn.LayerNorm(HIDDEN_DIM),
                    nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 4),
                    nn.GELU(),
                    nn.Linear(HIDDEN_DIM * 4, HIDDEN_DIM),
                )
                self.output_norm = nn.LayerNorm(HIDDEN_DIM)
                self.latent_update = nn.Linear(HIDDEN_DIM, LATENT_DIM)

            def forward(self, latent_history: Any, packed_actions: Any) -> Any:
                import torch

                if packed_actions.ndim != 3 or packed_actions.shape[-1] != ACTION_DIM:
                    raise ValueError("packed actions must have shape [B,T,10]")
                history = _left_pad_history(latent_history)
                state = history[:, -1]
                outputs = []
                for step in range(int(packed_actions.shape[1])):
                    history_condition = self.history_adapter(history.reshape(history.shape[0], -1))
                    latent_hidden = self.latent_projection(state)
                    action_hidden = self.action_projection(packed_actions[:, step])
                    transition = self.shared_transition(history_condition + latent_hidden + action_hidden)
                    state = state + self.latent_update(self.output_norm(latent_hidden + transition))
                    outputs.append(state)
                    history = torch.cat((history[:, 1:], state[:, None]), dim=1)
                return torch.stack(outputs, dim=1)

        return _Impl()


def student_parameter_count() -> int:
    import torch

    torch.manual_seed(0)
    model = LeWMCompactRecurrentTransitionStudent()
    return int(sum(parameter.numel() for parameter in model.parameters()))


def official_teacher_targets(model: Any, latent_history: Any, future_actions: Any) -> Any:
    """Generate detached z(t+1)..z(t+5) through official ``predict`` calls.

    The official predictor accepts length 1/2/3 windows.  We therefore retain
    the actual available window instead of padding it to three for the teacher;
    only the student uses its explicitly documented left-padding adapter.
    """
    import torch

    history = latent_history.detach().clone()
    # Official H=1 rollout aligns the first candidate action token with the
    # single current latent.  There is no extra pre-candidate action token.
    actions = future_actions[:, :0].detach().clone()
    outputs = []
    with torch.no_grad():
        for step in range(int(future_actions.shape[1])):
            actions = torch.cat((actions, future_actions[:, step : step + 1]), dim=1)
            window = min(HISTORY_LENGTH, int(history.shape[1]))
            pred = model.predict(
                history[:, -window:], model.action_encoder(actions[:, -window:])
            )[:, -1:]
            outputs.append(pred)
            history = torch.cat((history, pred), dim=1)
    return torch.cat(outputs, dim=1).detach()


def recurrent_loss(prediction: Any, target: Any, weights: Sequence[float] = HORIZON_WEIGHTS) -> Any:
    import torch

    if prediction.shape != target.shape or prediction.shape[1] != HORIZON:
        raise ValueError(f"student/target shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}")
    per_horizon = (prediction - target).square().mean(dim=(0, 2))
    weight_tensor = torch.as_tensor(weights, dtype=per_horizon.dtype, device=per_horizon.device)
    if not torch.isclose(weight_tensor.mean(), torch.ones((), device=weight_tensor.device, dtype=weight_tensor.dtype)):
        raise ValueError("horizon weights must be mean-normalized")
    return (per_horizon * weight_tensor).mean(), per_horizon


def _tensor(value: Any, dtype: Any = None) -> Any:
    import torch

    result = value if torch.is_tensor(value) else torch.as_tensor(value)
    return result if dtype is None else result.to(dtype=dtype)


def _rank_values(values: Any) -> Any:
    import torch

    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float64, device=values.device)
    return ranks


def _spearman(first: Any, second: Any) -> float:
    import torch

    a = _rank_values(first.flatten().to(dtype=torch.float64))
    b = _rank_values(second.flatten().to(dtype=torch.float64))
    a = a - a.mean()
    b = b - b.mean()
    denominator = torch.sqrt((a.square().sum() * b.square().sum()).clamp_min(1e-24))
    return float((a.mul(b).sum() / denominator).detach().cpu())


def _topk_overlap(first: Any, second: Any, k: int = 30) -> float:
    a = set(int(x) for x in first.flatten().topk(min(k, first.numel()), largest=False).indices.detach().cpu())
    b = set(int(x) for x in second.flatten().topk(min(k, second.numel()), largest=False).indices.detach().cpu())
    return float(len(a & b) / max(1, min(k, first.numel())))


def validate_prepared_rows(rows: Sequence[Mapping[str, Any]], freeze: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the small detached-row exchange format used by Stage A."""
    context_spec = freeze["shared_data_and_schedule"]["context_manifest"]
    train = [row for row in rows if row.get("split") == "train"]
    heldout = [row for row in rows if row.get("split") == "heldout"]
    if len(train) != int(context_spec["train_context_target"]):
        raise ValueError("prepared train rows do not match the frozen 256 contexts")
    if len(heldout) != int(context_spec["heldout_contexts"]):
        raise ValueError("prepared held-out rows do not match the frozen 8 contexts")
    for split, group in (("train", train), ("heldout", heldout)):
        for row in group:
            context = _tensor(row["latent_history"])
            actions = _tensor(row["future_actions"])
            targets = _tensor(row["teacher_targets"])
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context.shape != (1, 1, LATENT_DIM):
                raise ValueError(f"{split} latent_history must be official policy H=1: [1,1,192]")
            expected_shape = (4, HORIZON, ACTION_DIM) if split == "train" else (2, 300, HORIZON, ACTION_DIM)
            expected_target_shape = (4, HORIZON, LATENT_DIM) if split == "train" else (2, 300, HORIZON, LATENT_DIM)
            if tuple(actions.shape) != expected_shape:
                raise ValueError(f"{split} future_actions must be {expected_shape}")
            if tuple(targets.shape) != expected_target_shape:
                raise ValueError(f"{split} teacher_targets shape mismatch")
            if split == "heldout":
                objective = row.get("teacher_objective")
                if objective is None or tuple(_tensor(objective).shape) != (2, 300):
                    raise ValueError("held-out rows need official teacher_objective [2,300]")
                if "goal_emb" not in row:
                    raise ValueError("held-out rows need goal_emb for the official external criterion")
                if "action_history" not in row:
                    raise ValueError("held-out rows need the cached official action history for teacher timing")
                history_actions = _tensor(row["action_history"])
                if history_actions.ndim != 3 or history_actions.shape[0] != 1 or history_actions.shape[1] < 1 or history_actions.shape[1] > HISTORY_LENGTH or history_actions.shape[2] != ACTION_DIM:
                    raise ValueError("held-out action_history must be [1,1..3,10]")
    return {"train_contexts": len(train), "heldout_contexts": len(heldout), "heldout_blocks": len(heldout) * 2}


def validate_manifest(path: Path, freeze: Mapping[str, Any]) -> dict[str, Any]:
    manifest = load_json(path)
    if manifest.get("schema") != "lewm-recurrent-student.context-manifest":
        raise ValueError("unexpected LeWM context manifest schema")
    if manifest.get("episode_disjoint") is not True:
        raise ValueError("context manifest is not episode-disjoint")
    train = manifest.get("splits", {}).get("train", [])
    heldout = manifest.get("splits", {}).get("heldout", [])
    expected_train = int(freeze["shared_data_and_schedule"]["context_manifest"]["train_context_target"])
    expected_heldout = int(freeze["shared_data_and_schedule"]["context_manifest"]["heldout_contexts"])
    if len(train) != expected_train or len(heldout) != expected_heldout:
        raise ValueError(f"manifest counts must be {expected_train}/{expected_heldout}")
    train_episodes = {int(row["episode_id"]) for row in train}
    heldout_episodes = {int(row["episode_id"]) for row in heldout}
    if train_episodes & heldout_episodes:
        raise ValueError("train and heldout episodes overlap")
    for split_name, rows in (("train", train), ("heldout", heldout)):
        for row in rows:
            if len(row.get("history_steps", [])) != 1:
                raise ValueError(f"{split_name} row lacks the explicit official H=1 current frame")
            if len(row.get("history_action_starts", [])) != 1:
                raise ValueError(f"{split_name} row lacks the explicit H=1 action start")
            if "future_action_start" not in row:
                raise ValueError(f"{split_name} row lacks future_action_start")
    return manifest


class HDF5EpisodeSliceReader:
    """Read only manifest-selected episode slices from official LeWM HDF5."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Any = None

    def __enter__(self) -> "HDF5EpisodeSliceReader":
        # The official PushT HDF5 uses an external compression filter.  Import
        # the already-installed plugin package on the compute node so it
        # registers its bundled filters with h5py; no install or download.
        import hdf5plugin  # noqa: F401
        import h5py

        self._handle = h5py.File(self.path, "r")
        required = {"pixels", "action", "episode_idx", "step_idx", "ep_len", "ep_offset"}
        missing = sorted(required - set(self._handle.keys()))
        if missing:
            raise ValueError(f"official HDF5 is missing datasets: {missing}")
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def episode_slice(self, episode_id: int) -> dict[str, Any]:
        if self._handle is None:
            raise RuntimeError("HDF5 reader must be used inside a context manager")
        ep = int(episode_id)
        offset = int(self._handle["ep_offset"][ep])
        length = int(self._handle["ep_len"][ep])
        end = offset + length
        return {
            "pixels": self._handle["pixels"][offset:end],
            "action": self._handle["action"][offset:end],
            "episode_idx": self._handle["episode_idx"][offset:end],
            "step_idx": self._handle["step_idx"][offset:end],
            "episode_id": ep,
            "offset": offset,
            "length": length,
        }

    def read_manifest_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Return an explicit row; no cross-episode or silent index fallback."""
        episode = self.episode_slice(int(row["episode_id"]))
        history_steps = [int(x) for x in row["history_steps"]]
        history_action_starts = [int(x) for x in row["history_action_starts"]]
        future_start = int(row["future_action_start"])
        required_action_end = future_start + HORIZON * (ACTION_DIM // 2)
        all_steps = history_steps + [future_start, required_action_end - 1]
        if min(all_steps) < 0 or max(all_steps) >= episode["length"]:
            raise IndexError(f"manifest row exceeds episode {episode['episode_id']} bounds")
        if any(int(episode["episode_idx"][step]) != episode["episode_id"] for step in history_steps):
            raise ValueError("episode_idx disagrees with manifest episode_id")
        raw_actions = episode["action"]
        return {
            "episode_id": episode["episode_id"],
            "history_steps": history_steps,
            "history_action_starts": history_action_starts,
            "future_action_start": future_start,
            "pixels": episode["pixels"][history_steps],
            "history_actions_raw": [raw_actions[start : start + ACTION_DIM // 2] for start in history_action_starts],
            "future_actions_raw": raw_actions[future_start:required_action_end],
        }


def build_context_manifest(dataset_path: Path, output_path: Path, freeze: Mapping[str, Any]) -> dict[str, Any]:
    """Select deterministic episode-disjoint H=1 anchors from official HDF5."""
    import h5py

    spec = freeze["shared_data_and_schedule"]["context_manifest"]
    train_target = int(spec["train_context_target"])
    heldout_target = int(spec["heldout_contexts"])
    seed = int(freeze["shared_data_and_schedule"]["seeds"]["context_manifest"])
    with h5py.File(dataset_path, "r") as handle:
        lengths = [int(x) for x in handle["ep_len"][:]]
    valid = [episode for episode, length in enumerate(lengths) if length >= HORIZON * 5 + 1]
    rng = random.Random(seed)
    rng.shuffle(valid)
    if len(valid) < train_target + heldout_target:
        raise RuntimeError("pinned HDF5 cannot provide the requested episode-disjoint contexts")
    heldout_episodes = valid[:heldout_target]
    train_episodes = valid[heldout_target:heldout_target + train_target]

    def row(episode: int, split: str, ordinal: int) -> dict[str, Any]:
        # A row records the current frame and the exact 25-primitive action span;
        # no hidden stride or duplicate context is introduced.
        return {
            "split": split,
            "context_id": f"{split}_{ordinal:04d}",
            "episode_id": int(episode),
            "history_steps": [0],
            "history_action_starts": [0],
            "future_action_start": 0,
            "goal_step_offset": HORIZON * 5,
        }

    manifest = {
        "schema": "lewm-recurrent-student.context-manifest",
        "schema_version": 1,
        "episode_disjoint": True,
        "selection_seed": seed,
        "history_contract": {"policy_initial_history_length": 1, "predictor_max_history_length": 3},
        "splits": {
            "train": [row(episode, "train", index) for index, episode in enumerate(train_episodes)],
            "heldout": [row(episode, "heldout", index) for index, episode in enumerate(heldout_episodes)],
        },
    }
    write_json(output_path, manifest)
    return manifest


def pack_raw_actions(raw_actions: Any) -> Any:
    """Pack five primitive 2-D actions into the official 10-D token shape."""
    import numpy as np

    value = np.asarray(raw_actions)
    if value.ndim != 2 or value.shape[-1] != 2 or value.shape[0] % 5:
        raise ValueError(f"raw PushT actions must be [N,2] with N divisible by 5, got {value.shape}")
    return value.reshape(value.shape[0] // 5, 10).astype("float32", copy=False)


def _normalise_pixels(raw: Any) -> Any:
    import torch

    pixels = torch.as_tensor(raw, dtype=torch.float32)
    if pixels.ndim == 3:
        pixels = pixels.unsqueeze(0)
    if pixels.shape[-1] == 3:
        pixels = pixels.permute(0, 3, 1, 2)
    if pixels.ndim != 4 or pixels.shape[1] != 3:
        raise ValueError(f"HDF5 pixels must convert to [B,3,H,W], got {tuple(pixels.shape)}")
    pixels = pixels / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=pixels.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=pixels.dtype).view(1, 3, 1, 1)
    return (pixels - mean) / std


def _parse_action_bounds(args: argparse.Namespace, freeze: Mapping[str, Any]) -> tuple[Any, Any, float]:
    import torch

    if args.action_low is None or args.action_high is None or args.gaussian_std is None:
        raise RuntimeError("action range and gaussian std must be explicit after probe; no normalization defaults are guessed")
    low = torch.tensor(args.action_low, dtype=torch.float32).reshape(1, 1, 2)
    high = torch.tensor(args.action_high, dtype=torch.float32).reshape(1, 1, 2)
    if bool((high <= low).any()) or args.gaussian_std <= 0:
        raise ValueError("action-high must exceed action-low and gaussian-std must be positive")
    slate = freeze["shared_data_and_schedule"]["action_query_slate"]
    expected_low = torch.tensor(slate["official_action_low"], dtype=torch.float32).reshape(1, 1, 2)
    expected_high = torch.tensor(slate["official_action_high"], dtype=torch.float32).reshape(1, 1, 2)
    expected_std = float(slate["gaussian_std"])
    if not torch.equal(low, expected_low) or not torch.equal(high, expected_high):
        raise ValueError("action bounds drifted from the frozen official PushT range")
    if not math.isclose(float(args.gaussian_std), expected_std, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("gaussian std drifted from the frozen query-slate value")
    return low, high, float(args.gaussian_std)


def _slate_actions(logged: Any, generator: Any, low: Any, high: Any, gaussian_std: float, count: int) -> Any:
    import torch

    primitive = logged.reshape(-1, 2)
    samples = primitive.unsqueeze(0) + gaussian_std * torch.randn((count, primitive.shape[0], 2), generator=generator)
    samples = samples.clamp(low, high)
    return samples.reshape(count, HORIZON, ACTION_DIM)


def prepare_detached_rows(model: Any, dataset_path: Path, manifest: Mapping[str, Any], freeze: Mapping[str, Any], args: argparse.Namespace, output: Path) -> list[dict[str, Any]]:
    """Encode the manifest and materialize frozen slates/teacher targets on GPU."""
    import torch

    low, high, gaussian_std = _parse_action_bounds(args, freeze)
    seeds = freeze["shared_data_and_schedule"]["seeds"]
    slate_seed = int(seeds["action_slate"])
    heldout_seeds = [int(x) for x in seeds["heldout_action_prefix"]]
    cpu_generator = torch.Generator(device="cpu").manual_seed(slate_seed)
    rows: list[dict[str, Any]] = []
    with HDF5EpisodeSliceReader(dataset_path) as reader:
        for row in list(manifest["splits"]["train"]) + list(manifest["splits"]["heldout"]):
            raw = reader.read_manifest_row(row)
            current = _normalise_pixels(raw["pixels"][:1]).unsqueeze(1).to("cuda")
            goal_index = int(row.get("goal_step_offset", HORIZON * 5))
            episode = reader.episode_slice(int(row["episode_id"]))
            goal_pixels = _normalise_pixels(
                episode["pixels"][goal_index : goal_index + 1]
            ).unsqueeze(1).to("cuda")
            # Keep slate generation on the frozen CPU generator; transfer only
            # tensors that enter the official teacher.
            logged = torch.from_numpy(pack_raw_actions(raw["future_actions_raw"]))
            encode_action = logged[:1].unsqueeze(0).to("cuda")
            with torch.no_grad():
                latent = model.encode({"pixels": current, "action": encode_action})["emb"]
                goal_emb = model.encode({"pixels": goal_pixels})["emb"]
            if tuple(latent.shape) != (1, 1, LATENT_DIM):
                raise RuntimeError(f"official H=1 encoder output is not [1,1,192]: {tuple(latent.shape)}")
            history_actions = encode_action
            if row["split"] == "train":
                gaussian_a = _slate_actions(logged, cpu_generator, low, high, gaussian_std, 1)[0]
                gaussian_b = _slate_actions(logged, cpu_generator, low, high, gaussian_std, 1)[0]
                candidate64 = _slate_actions(logged, cpu_generator, low, high, gaussian_std, 64).to("cuda")
                context64 = latent.expand(64, -1, -1)
                targets64 = official_teacher_targets(model, context64, candidate64)
                score_row = {"latent_history": latent, "goal_emb": goal_emb}
                objective64 = _official_objective(model, score_row, targets64)
                elite_rows = candidate64[objective64.topk(8, largest=False).indices]
                elite_mean = elite_rows.mean(dim=0).detach().cpu()
                elite_variance = elite_rows.var(dim=0, unbiased=False).clamp_min(
                    float(freeze["shared_data_and_schedule"]["action_query_slate"]["variance_floor"])
                ).detach().cpu()
                cem_noise = torch.randn(elite_mean.shape, generator=cpu_generator)
                cem_resample = elite_mean + elite_variance.sqrt() * cem_noise
                cem_resample = cem_resample.reshape(-1, 2).clamp(
                    low.reshape(2), high.reshape(2)
                ).reshape(HORIZON, ACTION_DIM)
                actions = torch.stack((logged.cpu(), gaussian_a.cpu(), gaussian_b.cpu(), cem_resample), dim=0)
                context = latent.detach().cpu()
                target = official_teacher_targets(model, context.to("cuda").expand(4, -1, -1), actions.to("cuda"))
                rows.append({"split": "train", "context_id": row["context_id"], "latent_history": context, "action_history": history_actions.detach().cpu(), "future_actions": actions, "teacher_targets": target.detach().cpu(), "goal_emb": goal_emb.detach().cpu()})
            else:
                banks = []
                targets = []
                objectives = []
                for seed in heldout_seeds:
                    generator = torch.Generator(device="cpu").manual_seed(seed)
                    actions = _slate_actions(logged, generator, low, high, gaussian_std, 300).to("cuda")
                    context = latent.expand(300, -1, -1)
                    target = official_teacher_targets(model, context, actions)
                    score_row = {"latent_history": latent, "goal_emb": goal_emb}
                    objective = _official_objective(model, score_row, target)
                    banks.append(actions.detach().cpu()); targets.append(target.detach().cpu()); objectives.append(objective.detach().cpu())
                rows.append({"split": "heldout", "context_id": row["context_id"], "latent_history": latent.detach().cpu(), "action_history": history_actions.detach().cpu(), "future_actions": torch.stack(banks), "teacher_targets": torch.stack(targets), "teacher_objective": torch.stack(objectives), "goal_emb": goal_emb.detach().cpu()})
    torch.save(rows, output / "prepared_rows.pt")
    return rows


def validate_query_slate(rows: Sequence[Mapping[str, Any]]) -> None:
    """Validate a precomputed slate without inventing action normalization."""
    if not rows:
        raise ValueError("query slate is empty")
    for row in rows:
        actions = _tensor(row["future_actions"])
        if tuple(actions.shape) != (4, HORIZON, ACTION_DIM):
            raise ValueError("training query slate action shape must be [4,5,10]")


def train_student(rows: Sequence[Mapping[str, Any]], settings: Mapping[str, Any], output: Path) -> dict[str, Any]:
    """Train the fixed student on precomputed detached teacher rows."""
    import torch

    train_rows = [row for row in rows if row.get("split") == "train"]
    if len(train_rows) < settings["batch_contexts"]:
        raise ValueError("prepared rows are smaller than one training batch")
    torch.manual_seed(int(settings["seeds"]["initialization"]))
    student = LeWMCompactRecurrentTransitionStudent().to("cuda")
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    generator = torch.Generator(device="cpu").manual_seed(int(settings["seeds"]["context_schedule"]))
    histories: list[dict[str, Any]] = []
    snapshots: dict[str, Any] = {}
    for step in range(1, int(settings["steps"]) + 1):
        indices = torch.randperm(len(train_rows), generator=generator)[: settings["batch_contexts"]]
        contexts = torch.cat([
            _tensor(train_rows[int(i)]["latent_history"]).expand(4, -1, -1)
            for i in indices
        ], dim=0).to("cuda")
        actions = torch.cat([_tensor(train_rows[int(i)]["future_actions"]) for i in indices], dim=0).to("cuda")
        targets = torch.cat([_tensor(train_rows[int(i)]["teacher_targets"]) for i in indices], dim=0).to("cuda")
        prediction = student(contexts, actions)
        loss, per_horizon = recurrent_loss(prediction, targets)
        if not torch.isfinite(loss) or not bool(torch.isfinite(per_horizon).all()):
            raise FloatingPointError(f"non-finite training loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        histories.append({"step": step, "weighted_mse": float(loss.detach().cpu()), "per_horizon_mse": [float(x) for x in per_horizon.detach().cpu()]})
        if step in SNAPSHOT_STEPS:
            snapshots[f"step_{step}"] = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}
            torch.save(snapshots[f"step_{step}"], output / f"lewm_recurrent_student_step{step:04d}.pt")
    first = histories[0]["weighted_mse"]
    last10 = statistics.median(item["weighted_mse"] for item in histories[-10:])
    return {
        "student": student,
        "snapshots": snapshots,
        "parameter_count": int(sum(parameter.numel() for parameter in student.parameters())),
        "per_step": histories,
        "last10_to_first_ratio": float(last10 / max(first, 1e-12)),
    }


def load_official_checkpoint(stablewm_home: Path) -> Any:
    import torch

    path = stablewm_home / "pusht" / "lewm_object.ckpt"
    if not path.is_file():
        raise FileNotFoundError(path)
    model = torch.load(path, map_location="cpu", weights_only=False)
    if not hasattr(model, "predict") or not hasattr(model, "criterion"):
        raise TypeError("staged checkpoint is not an official LeWM predictor/criterion object")
    return model.to("cuda").eval()


def _official_objective(model: Any, row: Mapping[str, Any], prediction: Any) -> Any:
    """Score student future latents with the frozen external official criterion."""
    import torch

    initial = _tensor(row["latent_history"], dtype=prediction.dtype).to(prediction.device)
    if initial.ndim == 2:
        initial = initial.unsqueeze(0)
    initial = initial.expand(prediction.shape[0], -1, -1)
    predicted_emb = torch.cat((initial, prediction), dim=1).unsqueeze(0)
    goal = _tensor(row["goal_emb"], dtype=prediction.dtype).to(prediction.device)
    if goal.ndim == 2:
        goal = goal.unsqueeze(0)
    value = model.criterion({"predicted_emb": predicted_emb, "goal_emb": goal})
    if not torch.is_tensor(value):
        raise TypeError("official criterion did not return a tensor")
    return value.reshape(prediction.shape[0], -1)[:, 0]


def _stage_a_metrics_for_block(model: Any, student: Any, row: Mapping[str, Any], block: int) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    actions = _tensor(row["future_actions"][block], dtype=torch.float32).to("cuda")
    target = _tensor(row["teacher_targets"][block], dtype=torch.float32).to("cuda")
    context = _tensor(row["latent_history"], dtype=torch.float32).to("cuda")
    context = context.expand(actions.shape[0], -1, -1)
    with torch.no_grad():
        prediction = student(context, actions)
        student_objective = _official_objective(model, row, prediction)
    teacher_objective = _tensor(row["teacher_objective"][block], dtype=student_objective.dtype).to(student_objective.device)
    per_horizon_mse = (prediction - target).square().mean(dim=(0, 2))
    per_horizon_cosine = [float(F.cosine_similarity(prediction[:, h], target[:, h], dim=1).mean().detach().cpu()) for h in range(HORIZON)]
    return {
        "block": block,
        "candidate_count": int(actions.shape[0]),
        "spearman": _spearman(teacher_objective, student_objective),
        "top30_overlap": _topk_overlap(teacher_objective, student_objective, 30),
        "per_horizon_mse": [float(x.detach().cpu()) for x in per_horizon_mse],
        "per_horizon_cosine": per_horizon_cosine,
        "relative_latent_mse": float(((prediction - target).square().mean() / target.square().mean().clamp_min(1e-8)).detach().cpu()),
        "finite": bool(torch.isfinite(prediction).all().item() and torch.isfinite(student_objective).all().item()),
    }


def stage_a_evaluate(model: Any, snapshots: Mapping[str, Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], settings: Mapping[str, Any]) -> dict[str, Any]:
    heldout = [row for row in rows if row.get("split") == "heldout"]
    evaluation: dict[str, Any] = {}
    for snapshot_name in ("step_500", "step_1000", "step_1500"):
        student = LeWMCompactRecurrentTransitionStudent().to("cuda")
        student.load_state_dict(snapshots[snapshot_name], strict=True)
        student.eval()
        blocks = []
        for row in heldout:
            for block in range(2):
                item = _stage_a_metrics_for_block(model, student, row, block)
                item["context_id"] = row.get("context_id")
                item["pairing_key"] = f"seed_index={block}:context={row.get('context_id')}"
                blocks.append(item)
        spearman = [float(item["spearman"]) for item in blocks]
        top30 = [float(item["top30_overlap"]) for item in blocks]
        evaluation[snapshot_name] = {
            "per_block": blocks,
            "terminal_ranking": {
                "spearman_median": float(statistics.median(spearman)),
                "spearman_minimum": min(spearman),
                "top30_overlap_median": float(statistics.median(top30)),
                "top30_overlap_minimum": min(top30),
            },
            "relative_latent_mse_median": float(statistics.median(float(item["relative_latent_mse"]) for item in blocks)),
            "finite": all(bool(item["finite"]) for item in blocks),
        }
    return evaluation


def causality_test(student: Any, row: Mapping[str, Any], seed: int, tolerance: float) -> dict[str, Any]:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    context = _tensor(row["latent_history"], dtype=torch.float32).to("cuda")
    results: dict[str, Any] = {}
    for prefix_length in range(1, HORIZON):
        first = torch.randn((1, HORIZON, ACTION_DIM), generator=generator)
        second = torch.randn((1, HORIZON, ACTION_DIM), generator=generator)
        second[:, :prefix_length] = first[:, :prefix_length]
        with torch.no_grad():
            out_a = student(context, first.to("cuda"))
            out_b = student(context, second.to("cuda"))
        difference = float((out_a[:, :prefix_length] - out_b[:, :prefix_length]).abs().max().cpu())
        results[str(prefix_length)] = {"max_abs": difference, "threshold": tolerance, "passed": difference <= tolerance}
    return results


def predictor_latency(model: Any, student: Any, row: Mapping[str, Any], warmup: int = 3, repeats: int = 10) -> dict[str, Any]:
    import torch

    actions = _tensor(row["future_actions"][0], dtype=torch.float32).to("cuda")
    context = _tensor(row["latent_history"], dtype=torch.float32).to("cuda").expand(actions.shape[0], -1, -1)
    history_actions = _tensor(row["action_history"], dtype=torch.float32).to("cuda").expand(actions.shape[0], -1, -1)
    for _ in range(warmup):
        with torch.no_grad():
            student(context, actions)
            official_teacher_targets(model, context, actions)
    student_ms: list[float] = []
    teacher_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            student(context, actions)
        end.record(); torch.cuda.synchronize(); student_ms.append(float(start.elapsed_time(end)))
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True); start.record()
        with torch.no_grad():
            official_teacher_targets(model, context, actions)
        end.record(); torch.cuda.synchronize(); teacher_ms.append(float(start.elapsed_time(end)))
    student_median = float(statistics.median(student_ms))
    teacher_median = float(statistics.median(teacher_ms))
    return {"student_ms": student_median, "teacher_ms": teacher_median, "reduction": 1.0 - student_median / max(teacher_median, 1e-12), "warmup": warmup, "repeats": repeats, "boundary": "cached H=1 latent + normalized packed action prefix -> five-step predictor; excludes encoder/CEM/environment"}


def predictor_gates(freeze: Mapping[str, Any], training: Mapping[str, Any], evaluation: Mapping[str, Any], causality: Mapping[str, Any], latency: Mapping[str, Any]) -> dict[str, Any]:
    final = evaluation["step_1500"]
    gate = freeze["gates"]["predictor_feasibility"]
    integrity = bool(training["last10_to_first_ratio"] <= float(freeze["gates"]["convergence"]["last10_to_first_training_mse_ratio_max"]) and final["finite"] and all(item["passed"] for item in causality.values()))
    ranking = final["terminal_ranking"]
    positive_spearman = sum(float(item["spearman"]) > 0 for item in final["per_block"])
    positive_top30 = sum(float(item["top30_overlap"]) > 0 for item in final["per_block"])
    fidelity = bool(ranking["spearman_median"] >= float(gate["median_spearman_min"]) and ranking["spearman_minimum"] >= float(gate["minimum_spearman_min"]) and ranking["top30_overlap_median"] >= float(gate["median_top30_min"]) and ranking["top30_overlap_minimum"] >= float(gate["minimum_top30_min"]) and final["relative_latent_mse_median"] <= float(gate["median_relative_latent_mse_max"]) and positive_spearman >= int(gate["positive_spearman_blocks_min"]) and positive_top30 >= int(gate["positive_top30_blocks_min"]))
    latency_ok = float(latency["reduction"]) >= float(gate["predictor_only_latency_reduction_min"])
    passed = bool(integrity and fidelity and latency_ok)
    return {"integrity_and_convergence": {"status": "PASS" if integrity else "FAIL"}, "predictor_fidelity_and_ranking": {"status": "PASS" if fidelity else "FAIL", "metrics": final["terminal_ranking"], "relative_latent_mse_median": final["relative_latent_mse_median"], "positive_spearman_blocks": positive_spearman, "positive_top30_blocks": positive_top30}, "causality": {"status": "PASS" if all(item["passed"] for item in causality.values()) else "FAIL", "per_prefix": dict(causality)}, "predictor_latency": {"status": "PASS" if latency_ok else "FAIL", "metrics": latency}, "predictor_feasibility": "GO" if passed else "NO-GO", "full_cem_viability": "NOT_RUN_BY_SCOPE"}


def run(args: argparse.Namespace, freeze: Mapping[str, Any], settings: Mapping[str, Any], contract: InterfaceContract) -> dict[str, Any]:
    """Compute-node predictor-level run: manifest, targets, train, Stage A gates."""
    require_compute_node()
    import torch

    dataset_path = (args.dataset or (args.stablewm_home / "pusht_expert_train.h5")).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    manifest_path = (args.manifest or (args.output / "context_manifest.json")).resolve()
    if manifest_path.is_file():
        manifest = validate_manifest(manifest_path, freeze)
    else:
        manifest = build_context_manifest(dataset_path, manifest_path, freeze)
        manifest = validate_manifest(manifest_path, freeze)
    official_model = load_official_checkpoint(args.stablewm_home.resolve())
    rows_path = (args.prepared_rows or (args.output / "prepared_rows.pt")).resolve()
    if rows_path.is_file():
        rows = torch.load(rows_path, map_location="cpu", weights_only=False)
    else:
        rows = prepare_detached_rows(official_model, dataset_path, manifest, freeze, args, args.output.resolve())
    if not isinstance(rows, list):
        raise ValueError("prepared rows must be a list")
    row_meta = validate_prepared_rows(rows, freeze)
    validate_query_slate([item for item in rows if item.get("split") == "train"])
    result = train_student(rows, settings, args.output.resolve())
    evaluation = stage_a_evaluate(official_model, result["snapshots"], rows, settings)
    final_student = LeWMCompactRecurrentTransitionStudent().to("cuda")
    final_student.load_state_dict(result["snapshots"]["step_1500"], strict=True)
    final_student.eval()
    heldout = [row for row in rows if row.get("split") == "heldout"]
    causality = causality_test(final_student, heldout[0], int(settings["seeds"]["heldout_action_prefix"][0]), 1e-6)
    latency = predictor_latency(official_model, final_student, heldout[0])
    gates = predictor_gates(freeze, {"last10_to_first_ratio": result["last10_to_first_ratio"]}, evaluation, causality, latency)
    summary = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "PREDICTOR_LEVEL_COMPLETE",
        "freeze": str(args.freeze.resolve()),
        "protocol": str(args.protocol.resolve()),
        "interface_probe": str(args.interface_probe.resolve()),
        "interface_contract": jsonable(contract.__dict__),
        "source": {"lewm_commit": freeze["evidence_boundary"]["source_commit"], "stable_worldmodel_cem_commit": freeze["evidence_boundary"]["stable_worldmodel_cem_commit"], "checkpoint": str((args.stablewm_home / "pusht" / "lewm_object.ckpt").resolve()), "dataset": str(dataset_path)},
        "manifest": str(manifest_path),
        "manifest_counts": {"train": len(manifest["splits"]["train"]), "heldout": len(manifest["splits"]["heldout"])},
        "prepared_row_counts": row_meta,
        "student": {"class": "LeWMCompactRecurrentTransitionStudent", "latent_dim": LATENT_DIM, "history_length": HISTORY_LENGTH, "official_policy_history_h": contract.observation_history_h, "action_dim": ACTION_DIM, "hidden_dim": HIDDEN_DIM, "parameter_count": result["parameter_count"], "teacher_forcing": False, "goal_input": False, "encode_obs_calls": False},
        "training": {"steps": settings["steps"], "snapshot_steps": list(SNAPSHOT_STEPS), "horizon_weights": list(HORIZON_WEIGHTS), "last10_to_first_ratio": result["last10_to_first_ratio"], "per_step": result["per_step"]},
        "pairing": {"initialization_seed": settings["seeds"]["initialization"], "training_seed": settings["seeds"]["training"], "context_manifest_seed": settings["seeds"]["context_manifest"], "context_schedule_seed": settings["seeds"]["context_schedule"], "action_slate_seed": settings["seeds"]["action_slate"], "heldout_action_prefix_seeds": settings["seeds"]["heldout_action_prefix"], "same_manifest_and_teacher_targets": True, "same_heldout_candidate_bank": True},
        "stage_a": {"status": "COMPLETE", "candidate_count": 300, "blocks": 16, "topk": 30, "evaluation": evaluation, "causality": causality, "predictor_latency": latency},
        "stage_b": {"status": "NOT_RUN_BY_SCOPE", "official_prediction_count": contract.official_future_prediction_count, "shape": list(contract.official_predicted_emb_shape), "reason": "predictor-level task scope explicitly disables official CEM execution; probe count/shape are retained for interface documentation only"},
        "gates": gates,
        "decisions": {"predictor_feasibility": gates["predictor_feasibility"], "strict_predictor_replacement": "NOT_COMPUTED", "full_cem_viability": "NOT_RUN_BY_SCOPE"},
        "claim_boundary": "This summary is predictor-level only. It does not claim official CEM viability, encode_obs speedup, closed-loop PushT success, or Fast-LeWM comparison.",
    }
    write_json(args.output.resolve() / "lewm_recurrent_student_summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    freeze = load_json(args.freeze.resolve())
    settings = validate_freeze(freeze)
    # The protocol is Markdown, not a machine schema; retain the path and
    # only check its title before status/run handling.
    text = args.protocol.read_text(encoding="utf-8")
    if "LeWM PushT" not in text or "Horizon-Weighted Recurrent Student" not in text:
        raise ValueError("protocol does not look like the frozen LeWM recurrent protocol")
    if args.mode == "status":
        if not args.interface_probe.is_file():
            write_json(args.output / "run_status.json", waiting_status(args.freeze, args.protocol, args.interface_probe))
            print(json.dumps({"status": "WAITING_FOR_INTERFACE_PROBE", "output": str(args.output / 'run_status.json')}))
            return 3
        contract = load_interface_contract(args.interface_probe.resolve())
        value = {"schema": SCHEMA, "schema_version": 1, "status": "PROBE_READY", "interface_contract": jsonable(contract.__dict__), "probe_shape_count_match": contract.probe_shape_count_match, "stage_b": "NOT_RUN_BY_SCOPE"}
        write_json(args.output / "run_status.json", value)
        print(json.dumps(value))
        return 0
    if not args.interface_probe.is_file():
        write_json(args.output / "run_status.json", waiting_status(args.freeze, args.protocol, args.interface_probe))
        print(json.dumps({"status": "WAITING_FOR_INTERFACE_PROBE"}))
        return 3
    contract = load_interface_contract(args.interface_probe.resolve())
    summary = run(args, freeze, settings, contract)
    print(json.dumps({"status": summary["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
