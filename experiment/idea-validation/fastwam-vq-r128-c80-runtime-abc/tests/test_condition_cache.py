import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from condition_cache import install


class FakeTensor:
    equality_checks = 0

    def __init__(self, values):
        self.values = tuple(values)
        self.shape = (len(self.values),)
        self.dtype = "fake32"
        self.device = "cpu"
        self.nbytes = len(self.values) * 4

    def clone(self):
        return FakeTensor(self.values)

    def copy_(self, other):
        self.values = tuple(other.values)
        return self

    def equal(self, other):
        type(self).equality_checks += 1
        return self.values == other.values


class FakeProjection:
    def __init__(self, factor):
        self.factor = factor
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        return FakeTensor([item * self.factor for item in value.values])

    def __call__(self, value):
        return self.forward(value)


class Linear(FakeProjection):
    pass


class FakeTextEmbedding(FakeProjection):
    def modules(self):
        return [self, Linear(1), Linear(1)]


class FakeAttention:
    def __init__(self, index):
        self.k = FakeProjection(index + 2)
        self.v = FakeProjection(index + 5)


class FakeBlock:
    def __init__(self, index):
        self.cross_attn = FakeAttention(index)


class FakeExpert:
    def __init__(self, offset):
        self.text_embedding = FakeTextEmbedding(offset)
        self.blocks = [FakeBlock(0), FakeBlock(1)]


class FakeModel:
    def __init__(self):
        self.video_expert = FakeExpert(2)
        self.action_expert = FakeExpert(3)
        self.fail = False

    def infer_action(self, raw_context):
        if self.fail:
            raise ValueError("fake inference failure")
        outputs = []
        for expert in (self.video_expert, self.action_expert):
            for _ in range(3):
                context = expert.text_embedding(raw_context)
                outputs.append(context.values)
                for block in expert.blocks:
                    outputs.append(block.cross_attn.k(context).values)
                    outputs.append(block.cross_attn.v(context).values)
        return outputs


class QueryCacheTests(unittest.TestCase):
    def setUp(self):
        FakeTensor.equality_checks = 0

    def test_query_hits_are_separate_and_buffers_refill_without_stale_values(self):
        model = FakeModel()
        cache = install(model)

        first = FakeTensor([1, 2])
        model.infer_action(first)
        first_meta = cache.metadata()
        self.assertEqual(first_meta["queries"], 1)
        self.assertEqual(first_meta["text_embedding_hits"], 4)
        self.assertEqual(first_meta["text_linear_hits"], 8)
        self.assertEqual(first_meta["cross_kv_pair_hits"], 8)
        self.assertEqual(first_meta["cross_kv_linear_hits"], 16)
        self.assertEqual(first_meta["misses"], 10)
        self.assertFalse(first_meta["active"])
        first_buffers = {key: id(slot["buffer"]) for key, slot in cache._slots.items()}
        self.assertEqual(len(set(first_buffers.values())), len(first_buffers))

        second = FakeTensor([7, 11])
        model.infer_action(second)
        second_meta = cache.metadata()
        self.assertEqual(second_meta["queries"], 2)
        self.assertEqual(
            {key: id(slot["buffer"]) for key, slot in cache._slots.items()},
            first_buffers,
        )
        self.assertEqual(cache._slots["text:video"]["buffer"].values, (14, 22))
        self.assertEqual(cache._slots["text:action"]["buffer"].values, (21, 33))
        self.assertEqual(second_meta["buffer_bytes"], 10 * 2 * 4)

    def test_eager_prepare_makes_capture_reads_stable_buffers_without_diagnostics(self):
        model = FakeModel()
        cache = install(model, diagnostics=True)
        cache.begin_query()
        context = FakeTensor([2, 4])
        embedded = cache.prepare_expert("video", context)
        buffers = {key: id(slot["buffer"]) for key, slot in cache._slots.items()}
        calls = model.video_expert.blocks[0].cross_attn.k.calls
        checks_before_capture = FakeTensor.equality_checks
        hits_before_capture = cache.metadata()["hits"]
        with cache.graph_capture():
            self.assertIs(model.video_expert.text_embedding(context), embedded)
            for block in model.video_expert.blocks:
                block.cross_attn.k(embedded)
                block.cross_attn.v(embedded)
        self.assertEqual(FakeTensor.equality_checks, checks_before_capture)
        self.assertEqual(cache.metadata()["hits"], hits_before_capture)
        self.assertEqual(model.video_expert.blocks[0].cross_attn.k.calls, calls)
        self.assertEqual(
            {key: id(slot["buffer"]) for key, slot in cache._slots.items()},
            buffers,
        )
        cache.end_query()

    def test_diagnostic_mismatch_fails_closed_within_query(self):
        model = FakeModel()
        cache = install(model, diagnostics=True)
        cache.begin_query()
        cache.prepare_expert("video", FakeTensor([1, 3]))
        with self.assertRaisesRegex(ValueError, "invariant violated for text:video"):
            cache.prepare_expert("video", FakeTensor([5, 8]))
        self.assertEqual(
            cache._slots["k:video:0"]["buffer"].values,
            (4, 12),
        )
        self.assertGreater(cache.metadata()["diagnostic_mismatches"], 0)
        cache.end_query()

    def test_reset_metrics_keeps_cached_buffer_and_validity(self):
        model = FakeModel()
        cache = install(model)
        model.infer_action(FakeTensor([2]))
        video_context = cache._slots["text:video"]["buffer"]
        cache.reset_metrics()
        self.assertEqual(cache.metadata()["queries"], 0)
        self.assertEqual(cache.metadata()["hits"], 0)
        self.assertIs(cache._slots["text:video"]["buffer"], video_context)
        self.assertTrue(cache._slots["text:video"]["valid"])

    def test_inference_exception_closes_query(self):
        model = FakeModel()
        cache = install(model)
        model.fail = True
        with self.assertRaisesRegex(ValueError, "fake inference failure"):
            model.infer_action(FakeTensor([1]))
        self.assertFalse(cache.active)


if __name__ == "__main__":
    unittest.main()
