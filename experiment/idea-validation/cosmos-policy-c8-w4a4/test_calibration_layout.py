"""Regression for the official permuted-video layout across C8 capture and replay."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

import calibrate
import cosmos_runtime
import endpoint


class CalibrationLayoutTests(unittest.TestCase):
    def test_official_video_layout_survives_capture_serialization_and_replay(self):
        # Same BTHWC -> BCTHW permutation as pinned get_action, at tiny size.
        images = torch.arange(1 * 5 * 7 * 11 * 3).reshape(1, 5, 7, 11, 3)
        for dtype in (torch.uint8, torch.bfloat16):
            with self.subTest(dtype=dtype):
                official = images.to(dtype).permute(0, 4, 1, 2, 3)
                self.assertTrue(official.is_contiguous(memory_format=torch.channels_last_3d))
                self.assertFalse(official.is_contiguous())
                captured = cosmos_runtime._cpu_tree({'video': official}, torch)
                context = calibrate._cpu_tree(captured, torch)
                with TemporaryDirectory() as directory:
                    path = Path(directory) / 'context.pt'
                    torch.save(context, path)
                    saved = torch.load(path, map_location='cpu', weights_only=False)
                replay = endpoint._tree_copy_to_device(saved, torch.device('cpu'), torch)
                for stage in (captured, context, saved, replay):
                    video = stage['video']
                    self.assertTrue(torch.equal(video, official))
                    self.assertEqual(video.dtype, official.dtype)
                    self.assertEqual(video.stride(), official.stride())
                    self.assertTrue(video.is_contiguous(memory_format=torch.channels_last_3d))
                    self.assertFalse(video.is_contiguous())


if __name__ == '__main__':
    unittest.main()
