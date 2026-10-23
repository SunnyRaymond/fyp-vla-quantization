"""Pure-standard-library self-checks for the future intervention patches."""
from __future__ import annotations

from contextlib import nullcontext
import random
import unittest

import future_adapter


class FakeTensor:
    def __init__(self, shape, values=None, *, device="cuda", dtype="bf16"):
        self.shape = tuple(shape)
        self.values = list(values) if values is not None else [0.0] * (
            self.shape[2] if len(self.shape) > 2 else 1
        )
        self.device = device
        self.dtype = dtype

    def clone(self):
        return FakeTensor(self.shape, self.values, device=self.device, dtype=self.dtype)

    def detach(self):
        return self

    def to(self, *, device=None, dtype=None):
        return FakeTensor(
            self.shape,
            self.values,
            device=self.device if device is None else device,
            dtype=self.dtype if dtype is None else dtype,
        )

    def __getitem__(self, key):
        start, stop, _step = key[2].indices(self.shape[2])
        shape = (self.shape[0], self.shape[1], stop - start, *self.shape[3:])
        return FakeTensor(shape, self.values[start:stop], device=self.device, dtype=self.dtype)

    def __setitem__(self, key, value):
        start, stop, _step = key[2].indices(self.shape[2])
        self.values[start:stop] = value.values


class FakeTorch:
    float32 = "float32"

    @staticmethod
    def inference_mode():
        return nullcontext()

    @staticmethod
    def equal(left, right):
        return left.shape == right.shape and left.values == right.values


class FakeScheduler:
    def step(self, prediction, _delta, sample):
        return FakeTensor(
            sample.shape,
            [value + update for value, update in zip(sample.values, prediction.values)],
            device=sample.device,
            dtype=sample.dtype,
        )


class FakeVideoExpert:
    def __init__(self):
        self.prepare_calls = 0

    def prepare(self, *, x, **_kwargs):
        self.prepare_calls += 1
        return x


class FakeModel:
    def __init__(self):
        self.device = "cuda"
        self.torch_dtype = "bf16"
        self.vae = type("VAE", (), {
            "temporal_downsample_factor": 4,
            "upsampling_factor": 16,
            "model": type("VAECore", (), {"z_dim": 1})(),
        })()
        self.video_expert = FakeVideoExpert()
        self.infer_video_scheduler = FakeScheduler()
        self.native_video_calls = 0
        self.action_calls = 0
        self.action_noise = None
        self.action_condition = None
        self.fail_action = False
        self.last_kwargs = None

    def _denoise_video(self, *, latents_video, **_kwargs):
        self.native_video_calls += 1
        return FakeTensor(latents_video.shape, [1.0] * latents_video.shape[2])

    def _denoise_action_with_video_cache(self, *, latents_action, **_kwargs):
        self.action_calls += 1
        if self.action_noise is None:
            self.action_noise = latents_action.clone()
        if self.fail_action:
            raise RuntimeError("mock action failure")
        return latents_action

    def infer_action(self, **kwargs):
        self.last_kwargs = kwargs
        video_rng = random.Random(kwargs["seed"])
        action_rng = random.Random(kwargs["seed"])
        video = FakeTensor((1, 1, 3, 1, 1), [video_rng.random() for _ in range(3)])
        first_frame = FakeTensor((1, 1, 1, 1, 1), [0.25])
        video[:, :, 0:1] = first_frame
        action = FakeTensor(
            (1, 32, 7),
            [action_rng.random() for _ in range(32 * 7)],
        )
        for _ in range(future_adapter.INFERENCE_STEPS):
            prediction = self._denoise_video(
                latents_video=video,
                timestep_video=None,
                context=None,
                context_mask=None,
                video_self_attn_mask=None,
                fuse_vae_embedding_in_latents=False,
            )
            video = self.infer_video_scheduler.step(prediction, None, video)
            video[:, :, 0:1] = first_frame.clone()
        self.video_expert.prepare(x=video, timestep=None)
        for _ in range(future_adapter.INFERENCE_STEPS):
            self._denoise_action_with_video_cache(latents_action=action)
        return {"video_latents": video, "action": action}


class FutureAdapterTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.runtime = type("Runtime", (), {
            "model": self.model,
            "torch": FakeTorch(),
            "ev": type("Eval", (), {"DEFAULT_PROMPT": "do {task}"})(),
            "description": "the task",
        })()
        self.datum = {
            "image": FakeTensor((1, 3, 16, 16)),
            "proprio": FakeTensor((1, 7)),
        }

    def test_native_capture_keeps_inference_and_reports_same_action_noise(self):
        output = future_adapter.run_capture(self.runtime, self.datum, 17)
        diagnostic = output["_diagnostic"]
        self.assertEqual(self.model.native_video_calls, 10)
        self.assertEqual(diagnostic["video_dit_calls"], 10)
        self.assertEqual(diagnostic["action_dit_calls"], 10)
        self.assertEqual(diagnostic["action_initial_noise"].shape, (1, 32, 7))
        self.assertEqual(diagnostic["action_initial_noise"].device, "cpu")
        self.assertEqual(self.model.last_kwargs["action_infer_mode"], "idm")
        self.assertEqual(self.model.last_kwargs["num_video_frames"], 9)
        self.assertEqual(self.model.last_kwargs["num_inference_steps"], 10)
        self.assertFalse("_denoise_video" in self.model.__dict__)
        self.assertFalse("step" in self.model.infer_video_scheduler.__dict__)

    def test_teacher_future_skips_video_dit_and_changes_action_conditioning(self):
        native = future_adapter.run_capture(self.runtime, self.datum, 17)
        teacher = native["video_latents"].clone()
        teacher.values[1:] = [100.0, 200.0]
        self.model.native_video_calls = 0
        self.model.action_calls = 0
        self.model.action_noise = None

        intervened = future_adapter.run_teacher_future(self.runtime, self.datum, 17, teacher)
        diagnostic = intervened["_diagnostic"]
        self.assertEqual(self.model.native_video_calls, 0)
        self.assertEqual(diagnostic["video_dit_calls"], 0)
        self.assertEqual(diagnostic["action_dit_calls"], 10)
        self.assertEqual(diagnostic["teacher_future_injection_steps"], 10)
        self.assertTrue(diagnostic["teacher_future_injection_verified"])
        self.assertTrue(diagnostic["action_conditioning_future_matches_teacher"])
        self.assertTrue(diagnostic["action_conditioning_first_frame_matches_observation"])
        self.assertEqual(
            diagnostic["action_initial_noise"].values,
            native["_diagnostic"]["action_initial_noise"].values,
        )
        self.assertEqual(diagnostic["action_conditioning_latents"].values, teacher.values)
        self.assertEqual(intervened["video_latents"].values, teacher.values)

    def test_patches_restore_when_action_inference_fails(self):
        teacher = FakeTensor((1, 1, 3, 1, 1), [0.25, 0.5, 0.75])
        self.model.fail_action = True
        with self.assertRaisesRegex(RuntimeError, "mock action failure"):
            future_adapter.run_teacher_future(self.runtime, self.datum, 17, teacher)
        self.assertNotIn("_denoise_video", self.model.__dict__)
        self.assertNotIn("_denoise_action_with_video_cache", self.model.__dict__)
        self.assertNotIn("step", self.model.infer_video_scheduler.__dict__)
        self.assertNotIn("prepare", self.model.video_expert.__dict__)

    def test_wrong_teacher_shape_is_rejected_before_patching(self):
        wrong_shape = FakeTensor((1, 1, 2, 1, 1), [0.5, 0.75])
        with self.assertRaisesRegex(ValueError, "teacher_latents must have shape"):
            future_adapter.run_teacher_future(self.runtime, self.datum, 17, wrong_shape)
        self.assertEqual(self.model.action_calls, 0)
        self.assertNotIn("_denoise_video", self.model.__dict__)


if __name__ == "__main__":
    unittest.main()
