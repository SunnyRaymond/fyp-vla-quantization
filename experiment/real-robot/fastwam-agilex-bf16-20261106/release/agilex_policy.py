"""Real-robot inference wrapper for the AgileX Piper BF16 Fast-WAM (Optional IDM) fine-tune.

    from agilex_policy import AgileXFastWAMPolicy
    policy = AgileXFastWAMPolicy("/path/to/agilex_fastwam_bf16")          # bundle directory
    out = policy.predict(fixed_rgb, wrist_rgb, state, "place the lemon to the box")
    out["actions"]   # (32, 7) float32: joint_1..joint_6 absolute targets in rad, gripper opening in m

Inputs follow the recorded HDF5 exactly:
  fixed_rgb  observation/images/realsense_fixed  (240, 320, 3) uint8, same channel order as the recorder stored
  wrist_rgb  observation/images/realsense_wrist  (240, 320, 3) uint8
  state      observation/state                   (7,) joints 1..6 in rad + gripper opening in m
Output row k is the target for observation t+k+1 (the recorded action[t] == state[t+1]), 10 Hz.

The preprocessing mirrors training (FastWAM RobotVideoDataset + FastWAMProcessor with the AgileX config):
each camera uint8 -> /255 -> bilinear antialiased resize to 224x224 -> concat [fixed | wrist] on width -> (x-0.5)/0.5;
state min/max-normalised with dataset_stats.json and zero-padded to 8 dims; text = cached umT5 context (128 tokens) of
DEFAULT_PROMPT.format(task=instruction); joints are predicted relative to the current state, gripper absolute.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as TF

BUNDLE_FILES = ("policy_config.json", "dataset_stats.json", "prompts.json")


class AgileXFastWAMPolicy:
    def __init__(self, bundle_dir: str | os.PathLike, device: str = "cuda", checkpoint: str | None = None,
                 mode: str | None = None, num_inference_steps: int | None = None, sigma_shift: float | None = None,
                 seed: int | None = 0):
        self.bundle = Path(bundle_dir).resolve()
        for name in BUNDLE_FILES:
            if not (self.bundle / name).is_file():
                raise FileNotFoundError(f"bundle file missing: {self.bundle / name}")
        cfg = json.loads((self.bundle / "policy_config.json").read_text(encoding="utf-8"))
        self.cfg = cfg
        infer = cfg["inference"]
        self.mode = mode or infer["action_infer_mode"]
        if self.mode not in ("idm", "first_frame"):
            raise ValueError(f"mode must be idm or first_frame, got {self.mode}")
        self.num_inference_steps = int(num_inference_steps or infer["num_inference_steps"])
        self.sigma_shift = infer["sigma_shift"] if sigma_shift is None else float(sigma_shift)
        self.seed = seed
        self.action_horizon = int(infer["action_horizon"])
        self.num_video_frames = int(infer["num_video_frames"])
        self.camera_hw = tuple(cfg["camera_input_hw"])        # (240, 320) as recorded
        self.camera_resize_hw = tuple(cfg["camera_resize_hw"])  # (224, 224) per camera
        self.joint_dims = int(cfg["joint_dims"])
        self.proprio_dim = int(cfg["model"]["proprio_dim"])
        self.device = torch.device(device)

        # Only the Wan2.2 VAE is read from the model cache (DiT weights come from the checkpoint, text from the cache).
        os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(self.bundle / "model-cache"))
        os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
        os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from fastwam.runtime import create_fastwam_optional_idm
        from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
        from fastwam.datasets.lerobot.utils.normalizer import LinearNormalizer, load_dataset_stats_from_json

        t0 = time.time()
        self.model = create_fastwam_optional_idm(**cfg["model"], model_dtype=torch.bfloat16, device=str(self.device))
        ckpt = Path(checkpoint) if checkpoint else self.bundle / cfg["checkpoint"]
        payload = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)
        missing, unexpected = self.model.mot.load_state_dict(payload["mot"], strict=False)
        if missing or unexpected:
            raise RuntimeError(f"checkpoint/model key mismatch: missing {list(missing)[:5]} ({len(missing)}), "
                               f"unexpected {list(unexpected)[:5]} ({len(unexpected)})")
        self.model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        self.checkpoint_step = payload.get("step")
        del payload
        self.model = self.model.to(self.device).eval()
        self.load_seconds = time.time() - t0

        stats = load_dataset_stats_from_json(str(self.bundle / "dataset_stats.json"))
        self.normalizer = LinearNormalizer(shape_meta=cfg["shape_meta"], use_stepwise_action_norm=False,
                                           default_mode="min/max", exception_mode=None, stats=stats)

        self.prompt_template = DEFAULT_PROMPT
        self.contexts: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for instruction, name in json.loads((self.bundle / "prompts.json").read_text(encoding="utf-8")).items():
            prompt = DEFAULT_PROMPT.format(task=instruction)
            expected = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if not name.startswith(expected):
                raise RuntimeError(f"text cache {name} does not belong to prompt {prompt!r}")
            cache = torch.load(self.bundle / "text_embeds" / name, map_location="cpu")
            context, mask = cache["context"], cache["mask"].bool()
            context = context.clone()
            context[~mask] = 0.0  # as RobotVideoDataset does in training
            self.contexts[instruction] = (context.unsqueeze(0).to(self.device, torch.bfloat16),
                                          torch.ones_like(mask).unsqueeze(0).to(self.device))

    @property
    def instructions(self) -> list[str]:
        return list(self.contexts)

    def _camera(self, rgb: np.ndarray) -> torch.Tensor:
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"camera frame must be (H, W, 3) uint8, got {rgb.shape} {rgb.dtype}")
        if tuple(rgb.shape[:2]) != self.camera_hw:
            raise ValueError(f"camera frame must be {self.camera_hw} like the recorded data, got {rgb.shape[:2]}")
        x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return TF.resize(x, list(self.camera_resize_hw), interpolation=TF.InterpolationMode.BILINEAR, antialias=True)

    def preprocess(self, fixed_rgb: np.ndarray, wrist_rgb: np.ndarray, state: np.ndarray):
        image = torch.cat([self._camera(fixed_rgb), self._camera(wrist_rgb)], dim=-1)  # [1, 3, 224, 448]
        image = (image - 0.5) / 0.5
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape != (self.joint_dims + 1,) or not np.isfinite(state).all():
            raise ValueError(f"state must be {self.joint_dims + 1} finite values (joints rad + gripper m), got {state}")
        batch = {"state": {"joints": torch.from_numpy(state[None, :self.joint_dims].copy()),
                           "gripper": torch.from_numpy(state[None, self.joint_dims:].copy())}}
        batch = self.normalizer.forward(batch)
        proprio = torch.cat([batch["state"]["joints"], batch["state"]["gripper"]], dim=-1)
        proprio = torch.nn.functional.pad(proprio, (0, self.proprio_dim - proprio.shape[-1]))  # [1, 8]
        return image, proprio

    def postprocess(self, action: torch.Tensor, state: np.ndarray) -> np.ndarray:
        action = action.float().cpu()
        if action.ndim == 3:
            action = action[0]
        batch = {"action": {"joints": action[:, :self.joint_dims].clone(),
                            "gripper": action[:, self.joint_dims:self.joint_dims + 1].clone()},
                 "state": {"joints": torch.zeros(1, self.joint_dims), "gripper": torch.zeros(1, 1)}}
        batch = self.normalizer.backward(batch)
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        joints = batch["action"]["joints"].numpy() + state[None, :self.joint_dims]  # relative -> absolute
        gripper = batch["action"]["gripper"].numpy()
        return np.concatenate([joints, gripper], axis=1).astype(np.float32)

    @torch.no_grad()
    def predict(self, fixed_rgb: np.ndarray, wrist_rgb: np.ndarray, state: np.ndarray, instruction: str,
                seed: int | None = None) -> dict[str, Any]:
        if instruction not in self.contexts:
            raise KeyError(f"no cached text context for {instruction!r}; supported: {self.instructions}")
        t0 = time.perf_counter()
        image, proprio = self.preprocess(fixed_rgb, wrist_rgb, state)
        context, context_mask = self.contexts[instruction]
        kwargs = dict(prompt=None, input_image=image.to(self.device, torch.bfloat16), action_horizon=self.action_horizon,
                      proprio=proprio.to(self.device), context=context, context_mask=context_mask,
                      num_inference_steps=self.num_inference_steps, sigma_shift=self.sigma_shift,
                      seed=self.seed if seed is None else seed, rand_device="cpu", tiled=False,
                      compile_action_infer=False, action_infer_mode=self.mode)
        if self.mode == "idm":
            kwargs["num_video_frames"] = self.num_video_frames
        pred = self.model.infer_action(**kwargs)
        actions = self.postprocess(pred["action"], state)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if not np.isfinite(actions).all():
            raise FloatingPointError("non-finite action predicted")
        return dict(actions=actions, latency_s=time.perf_counter() - t0)
