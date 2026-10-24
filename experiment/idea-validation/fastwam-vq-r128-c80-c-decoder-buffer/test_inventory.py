"""stdlib checks for storage alias and component accounting."""
import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from inventory import (finalize_runtime_probe, install_runtime_probe,
                       inventory_from_descriptors, write_runtime_probe)


class FakeHandle:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class FakeModule:
    def __init__(self, parameters=()):
        self.parameters = list(parameters)
        self.hooks = []

    def named_parameters(self, recurse=False, remove_duplicate=False):
        return iter(self.parameters)

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return FakeHandle()

    def call_existing_forward(self):
        for hook in self.hooks:
            hook(self, (), None)


class FakeModel:
    def __init__(self, text_module, target_module):
        self.text_module = text_module
        self.target_module = target_module

    def named_parameters(self, recurse=True, remove_duplicate=False):
        if recurse:
            return iter([('text_encoder.norm.weight', object()), ('action_head.weight', object())])
        return iter(())

    def named_modules(self, remove_duplicate=False):
        return iter([('', self), ('text_encoder.norm', self.text_module), ('action_head', self.target_module)])


class InventoryTests(unittest.TestCase):
    def test_aliases_count_backing_storage_once_and_attribute_to_one_component(self):
        report = inventory_from_descriptors([
            dict(state_key='model.text_encoder.embed_tokens.weight', storage_key='shared',
                 storage_bytes=16, logical_bytes=16, dtype='torch.bfloat16',
                 shape=[4, 2], component='text_encoder', module='model.text_encoder.embed_tokens'),
            dict(state_key='model.lm_head.weight', storage_key='shared',
                 storage_bytes=16, logical_bytes=16, dtype='torch.bfloat16',
                 shape=[4, 2], component='text_encoder', module='model.lm_head'),
            dict(state_key='model.vision_encoder.patch.weight', storage_key='vision',
                 storage_bytes=8, logical_bytes=8, dtype='torch.bfloat16',
                 shape=[2, 2], component='vision_encoder', module='model.vision_encoder.patch'),
        ], expected_unique_bytes=24)

        self.assertEqual(report['tensor_key_count'], 3)
        self.assertEqual(report['unique_storage_count'], 2)
        self.assertEqual(report['unique_storage_bytes'], 24)
        self.assertEqual(report['alias_extra_reference_count'], 1)
        self.assertEqual(sum(row['unique_storage_bytes'] for row in report['components']), 24)
        shared = next(row for row in report['storages'] if row['tensor_reference_count'] == 2)
        self.assertEqual(shared['tensor_reference_count'], 2)
        self.assertEqual(shared['storage_bytes'], 16)

    def test_tensor_view_reports_logical_bytes_separately_from_storage(self):
        report = inventory_from_descriptors([
            dict(state_key='model.layers.0.proj.weight', storage_key='base',
                 storage_bytes=32, logical_bytes=16, dtype='torch.bfloat16',
                 shape=[2, 4], stride=[8, 1], storage_offset=0, is_contiguous=False),
        ])
        self.assertEqual(report['logical_tensor_bytes'], 16)
        self.assertEqual(report['unique_storage_bytes'], 32)
        self.assertTrue(report['storages'][0]['has_tensor_view'])

    def test_runtime_probe_counts_existing_calls_excludes_quantized_target_and_removes_hooks(self):
        text = FakeModule([('weight', object())])
        target = FakeModule([('weight', object())])
        runtime = SimpleNamespace(model=FakeModel(text, target), q=SimpleNamespace(modules={'target': target}))
        probe = install_runtime_probe(runtime)

        text.call_existing_forward()
        target.call_existing_forward()
        report = finalize_runtime_probe(probe)

        self.assertEqual(report['executed_owner_module_count'], 1)
        self.assertEqual(report['total_owner_module_calls'], 1)
        self.assertEqual(report['quantized_target_modules_excluded'], 1)
        self.assertEqual(len(probe['handles']), 1)
        self.assertTrue(probe['handles'][0].removed)
        self.assertIn('not evidence', report['interpretation'])
        json.dumps(report)

    def test_runtime_probe_shards_roundtrip_all_owners_below_250000_bytes(self):
        executed = [dict(module_paths=[f'text_encoder.layer_{i}.norm'],
                          direct_parameter_aliases=[f'text_encoder.layer_{i}.norm.weight:' + 'x' * 2600],
                          components=['text_encoder'], calls=i + 1) for i in range(100)]
        unobserved = [dict(module_paths=[f'vision_encoder.block_{i}.norm'],
                           direct_parameter_aliases=[f'vision_encoder.block_{i}.norm.weight:' + 'y' * 2600],
                           components=['vision_encoder'], calls=0) for i in range(20)]
        report = dict(status='captured', evidence='same forward hooks', coverage_scope='anchor+warmups',
            graph_replay_note='graph replay is not observed', interpretation='not unused',
            quantized_target_modules_excluded=614, owner_module_count=120,
            executed_owner_module_count=100, total_owner_module_calls=sum(row['calls'] for row in executed),
            by_component={'text_encoder': {'owner_module_count': 100},
                          'vision_encoder': {'owner_module_count': 20}},
            executed_modules=executed, unobserved_owner_modules=unobserved)

        with tempfile.TemporaryDirectory() as temp_dir:
            compact = write_runtime_probe(temp_dir, report)
            restored = []
            for entry in compact['detail_files']:
                path = Path(temp_dir) / entry['path']
                self.assertLess(path.stat().st_size, 250_000)
                restored.extend(json.loads(path.read_text(encoding='utf-8'))['owners'])
            summary_path = Path(temp_dir) / 'parameter_execution.json'
            self.assertLess(summary_path.stat().st_size, 250_000)
            self.assertGreater(len(compact['detail_files']), 1)
            self.assertEqual(len(restored), len(executed) + len(unobserved))
            expected = ([dict(status='executed', **row) for row in executed]
                        + [dict(status='unobserved', **row) for row in unobserved])
            key = lambda row: (tuple(row['module_paths']), row['status'])
            self.assertEqual(sorted(restored, key=key), sorted(expected, key=key))
            self.assertEqual(sum(row['status'] == 'executed' for row in restored), len(executed))
            self.assertEqual(sum(row['status'] == 'unobserved' for row in restored), len(unobserved))
            self.assertEqual({row['calls'] for row in restored if row['status'] == 'unobserved'}, {0})


if __name__ == '__main__':
    unittest.main()
