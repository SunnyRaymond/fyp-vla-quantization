"""Control-flow checks without loading CUDA or Torch locally."""
import unittest
from collections import Counter
from types import SimpleNamespace

from graph_runtime import Regions, copy_inputs, tree_map


class Tensor:
    def __init__(self, value, shape=(2,)):
        self.value, self.shape, self.dtype, self.device = value, shape, "bf16", "cuda"

    def data_ptr(self):
        return id(self)

    def copy_(self, source):
        self.value = source.value


class Torch:
    Tensor = Tensor


class GraphInputs(unittest.TestCase):
    def test_nested_changed_inputs_are_refilled(self):
        target = {"x": Tensor(1), "cache": [Tensor(2)], "flag": True}
        source = {"x": Tensor(9), "cache": [Tensor(8)], "flag": True}
        copy_inputs(Torch, target, source)
        self.assertEqual((target["x"].value, target["cache"][0].value), (9, 8))

    def test_changed_shape_and_scalar_rejected(self):
        with self.assertRaises(ValueError):
            copy_inputs(Torch, Tensor(0), Tensor(1, (3,)))
        with self.assertRaises(ValueError):
            copy_inputs(Torch, {"flag": True}, {"flag": False})

    def test_tree_structure_preserved(self):
        self.assertEqual(tree_map(lambda x: x + 1, {"a": (1, [2])}), {"a": (2, [3])})

    def test_module_frequencies_include_replays_and_eager_calls(self):
        regions = Regions.__new__(Regions)
        regions.banks = SimpleNamespace(calls=Counter({'query_condition': 80}))
        regions.regions = {
            'step': {'replays': 800, 'capture_calls_by_module': {'linear': 2}},
            'prefill': {'replays': 80, 'capture_calls_by_module': {'linear': 1}},
        }
        self.assertEqual(regions.module_call_counts(), {'query_condition': 80, 'linear': 1680})


if __name__ == "__main__":
    unittest.main()
