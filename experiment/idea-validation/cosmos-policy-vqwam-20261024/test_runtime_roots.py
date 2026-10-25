"""Separate legacy asset-root validation from this task's output root."""
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import common


class RuntimeRootTests(unittest.TestCase):
    def test_legacy_constructor_observes_assets_and_new_root_is_restored(self):
        observed = []
        model = SimpleNamespace(parameters=lambda: [])
        def constructor(protocol, assets, out, **kwargs):
            observed.append(os.environ['ROOT'])
            return SimpleNamespace(model=model, controller=SimpleNamespace(),
                                   target_linears={}, query=lambda: None)
        protocol = {'protocol': 'test', 'asset_root': '/assets', 'arms': ['bf16']}
        with patch.object(common, 'guard'), patch.object(common, 'ensure_imports'), \
                patch.object(common, 'read_json', return_value={'protocol': 'test', 'status': 'complete'}), \
                patch.object(Path, 'is_file', return_value=True), \
                patch.dict('sys.modules', {'cosmos_runtime': SimpleNamespace(CosmosRuntime=constructor)}), \
                patch.dict(os.environ, {'ROOT': '/new_task'}):
            result = common.prepare_runtime(protocol, '/new_task', '/output')
            self.assertEqual(observed, [str(Path('/assets'))])
            self.assertEqual(os.environ['ROOT'], '/new_task')
            self.assertEqual(result.task_root, Path('/new_task'))

    def test_constructor_failure_restores_task_environment(self):
        with patch.object(common, 'guard'), patch.object(common, 'ensure_imports'), \
                patch.object(common, 'read_json', return_value={'protocol': 'test', 'status': 'complete'}), \
                patch.object(Path, 'is_file', return_value=True), \
                patch.dict('sys.modules', {'cosmos_runtime': SimpleNamespace(
                    CosmosRuntime=lambda *a, **k: (_ for _ in ()).throw(RuntimeError('load failed')))}), \
                patch.dict(os.environ, {'ROOT': '/new_task'}):
            with self.assertRaisesRegex(RuntimeError, 'load failed'):
                common.prepare_runtime({'protocol': 'test', 'asset_root': '/assets'}, '/new_task', '/output')
            self.assertEqual(os.environ['ROOT'], '/new_task')


if __name__ == '__main__':
    unittest.main()
