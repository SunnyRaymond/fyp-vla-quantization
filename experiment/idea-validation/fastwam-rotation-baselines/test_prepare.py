"""Small CPU controls for teacher capture and balanced optimization inputs."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import patch

import torch
import prepare


class PreparationControls(unittest.TestCase):
    def test_training_order_covers_all_cases_and_timesteps(self):
        packets = [{"case_id": case, "stream": stream, "step": step}
                   for stream in ("video", "action") for case in range(8) for step in (0, 3, 6, 9)]
        order = prepare.packet_order(packets, 20261008, 200)
        self.assertEqual(order, prepare.packet_order(packets, 20261008, 200))
        self.assertEqual(len(order), 200)
        self.assertEqual({(row["stream"], row["case_id"], row["step"]) for row in order[:64]},
                         {(row["stream"], row["case_id"], row["step"]) for row in packets})
        self.assertTrue(all(row["stream"] == ("video" if index % 2 == 0 else "action")
                            for index, row in enumerate(order)))

    def test_teacher_inputs_are_saved_before_inplace_sampler_change(self):
        class Model:
            def _denoise_video(self, latents_video):
                return latents_video + 1

            def _denoise_action_with_video_cache(self, latents_action, video_cache_k, video_cache_v):
                return latents_action + 1

        class Runtime:
            def __init__(self):
                self.model = Model()
                self.q = types.SimpleNamespace(enable=lambda _arm: None)

            def infer(self, datum, seed, measure=False):
                with torch.inference_mode():
                    latents = torch.zeros(1, 2)
                    for _ in range(10):
                        self.model._denoise_video(latents_video=latents)
                        latents.add_(1)
                    latents = torch.zeros(1, 2)
                    for _ in range(10):
                        self.model._denoise_action_with_video_cache(latents_action=latents,
                            video_cache_k=[torch.ones(1, 2)], video_cache_v=[torch.ones(1, 2)])
                        latents.add_(1)
                    return torch.zeros(32, 7), {}

        plan = {"inputs": [{"case_id": case, "split": "selection" if case % 3 == 2 else "calibration",
                             "sampler_seed": case} for case in range(12)]}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(prepare, "datum_for", return_value={"proprio": torch.ones(8)}):
                packets, actions = prepare.collect_teacher(Runtime(), root, plan, root,
                    {"calibration": {"denoising_indices": [0, 3, 6, 9]}}, lambda *_args, **_kwargs: None)
            self.assertEqual(len(packets), 96)
            self.assertEqual(len(actions), 12)
            for stream in ("video", "action"):
                for step in (0, 3, 6, 9):
                    packet = torch.load(root / f"case_00_{stream}_{step:02d}.pt", weights_only=True)
                    value = packet["kwargs"][f"latents_{stream}"]
                    self.assertTrue(torch.equal(value, torch.full((1, 2), float(step))))
                    self.assertFalse(value.is_inference())
            self.assertTrue(json.loads((root / "receipt.json").read_text())["complete"])


if __name__ == "__main__":
    unittest.main()
