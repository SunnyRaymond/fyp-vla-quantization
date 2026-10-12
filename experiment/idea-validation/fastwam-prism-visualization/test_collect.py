"""CPU-only checks for collector window, stage and selection helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


COLLECT_PATH = Path(__file__).with_name("collect.py")
SPEC = importlib.util.spec_from_file_location("prism_collect", COLLECT_PATH)
collect = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(collect)


class CollectorHelperTests(unittest.TestCase):
    def test_window_bounds_clamp_without_pooling_or_negative_rows(self):
        self.assertEqual(collect.window_bounds(32, 0, 128), (0, 32))
        self.assertEqual(collect.window_bounds(32, 28, 10), (28, 32))
        self.assertEqual(collect.window_bounds(32, 40, 10), (32, 32))
        with self.assertRaises(ValueError):
            collect.window_bounds(32, -1, 1)

    def test_alignment_uses_only_first_batch_branch(self):
        self.assertEqual(collect.aligned_window((2, 32, 256), (2, 32, 64), 0, 128, 256, 64), (0, 32))
        self.assertEqual(collect.aligned_window((32, 256), (32, 64), 4, 12, 256, 64), (4, 16))
        with self.assertRaisesRegex(ValueError, "leading dimensions"):
            collect.aligned_window((2, 32, 256), (2, 31, 64), 0, 128)
        with self.assertRaisesRegex(ValueError, "batch index"):
            collect.aligned_window((2, 32, 256), (2, 32, 64), 0, 128, batch_index=2)

    def test_capture_respects_real_stage_and_frozen_steps(self):
        steps = [0, 4, 9]
        self.assertTrue(collect.should_capture("video_expert", "video", 4, steps))
        self.assertTrue(collect.should_capture("action_expert", "action", 9, steps))
        self.assertTrue(collect.should_capture("video_expert", "video_conditioning_prefill", -1, steps))
        self.assertFalse(collect.should_capture("video_expert", "action", 4, steps))
        self.assertFalse(collect.should_capture("action_expert", "action", 8, steps))
        self.assertFalse(collect.should_capture("action_expert", "conditioning", 0, steps))

    def test_stage_tagger_keeps_conditioning_separate(self):
        tagger = collect.CollectorStageTagger()
        self.assertEqual((tagger.stage, tagger.step), ("conditioning", -1))
        tagger.set_stage("video_conditioning_prefill", -1)
        self.assertEqual((tagger.stage, tagger.step), ("video_conditioning_prefill", -1))
        tagger.set_stage("video", 4)
        self.assertEqual((tagger.stage, tagger.step), ("video", 4))
        with self.assertRaises(ValueError):
            tagger.set_stage("action", -1)

    def test_selected_modules_match_exact_suffix_and_block_coverage(self):
        blocks, streams, suffixes = [0, 10, 20, 29], ["video_expert", "action_expert"], ["self_attn.q", "ffn.2"]
        names = [f"{stream}.blocks.{block}.{suffix}"
                 for stream in streams for block in blocks for suffix in suffixes]
        result = collect.selected_modules(names, blocks, streams, suffixes)
        self.assertEqual(len(result), 16)
        self.assertEqual({(stream, block) for _index, _name, stream, block in result},
                         {(stream, block) for stream in streams for block in blocks})
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            collect.selected_modules(names[:-1], blocks, streams, suffixes)


if __name__ == "__main__":
    unittest.main()
