"""Storage accounting contracts without importing Torch or loading tensors."""
import unittest
from types import SimpleNamespace

from export_bundle import _tensor_inventory, _cpu_shared_state


class StorageInventoryTests(unittest.TestCase):
    def test_nontensor_module_extra_state_is_preserved(self):
        extra = {'version': 1}
        state = {'layer._extra_state': None, 'other._extra_state': extra}
        result = _cpu_shared_state(state, SimpleNamespace(is_tensor=lambda _value: False))
        self.assertEqual(result, state)
        self.assertIs(result['other._extra_state'], extra)

    def test_empty_and_shared_storage_count_logical_references_once_per_backing(self):
        storage = SimpleNamespace(data_ptr=lambda: 10, nbytes=lambda: 16)

        class Tensor:
            device = 'cuda:0'

            def __init__(self, elements):
                self.numel = lambda: elements
                self.element_size = lambda: 2
                self.untyped_storage = lambda: storage

        torch = SimpleNamespace(is_tensor=lambda value: isinstance(value, Tensor))
        a, b = Tensor(4), Tensor(2)
        self.assertEqual(_tensor_inventory({'a': a, 'aliases': [a, b]}, torch), {
            'tensor_references': 3, 'logical_tensor_bytes': 20,
            'unique_storage_count': 1, 'unique_storage_bytes': 16})
        self.assertEqual(_tensor_inventory({}, torch), {
            'tensor_references': 0, 'logical_tensor_bytes': 0,
            'unique_storage_count': 0, 'unique_storage_bytes': 0})


if __name__ == '__main__':
    unittest.main()
