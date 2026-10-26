"""Lifecycle regression without importing Torch or loading checkpoints."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import engine_optimized


class OptimizedLifecycleTests(unittest.TestCase):
    def test_restore_removes_stale_closure_and_is_idempotent(self):
        previous_end = Mock()
        runtime = SimpleNamespace(install_metadata={}, end_query=previous_end)
        banks = SimpleNamespace(graph_workspaces={}, default_workspace=object())
        cache = SimpleNamespace(_forward=Mock(), slots={})
        original_cache = cache._forward
        engine = SimpleNamespace(runtime=runtime, banks=banks, cache=cache,
                                 graphs=SimpleNamespace(restore=Mock(), records={}), restore=Mock())
        runtime.ours_vqwam_engine = engine
        graphs = SimpleNamespace(metadata=lambda: {}, restore=Mock())
        modules = {'common': SimpleNamespace(torch_module=lambda: SimpleNamespace(
                       cuda=SimpleNamespace(empty_cache=Mock()))),
                   'engine_profile': SimpleNamespace(_install_arm=Mock())}
        with patch.dict(sys.modules, modules), patch.object(engine_optimized, 'AllNfeGraphs', return_value=graphs):
            engine_optimized.install(runtime, {}, Path('/old'), Path('/new'))
        restore = runtime.restore_optimized_engine
        restore()
        restore()
        self.assertFalse(hasattr(runtime, 'restore_optimized_engine'))
        self.assertFalse(hasattr(runtime, 'optimized_vq_graphs'))
        self.assertIs(runtime.end_query, previous_end)
        self.assertIs(cache._forward, original_cache)
        engine.restore.assert_called_once()
        graphs.restore.assert_called_once()

    def test_restored_none_engine_must_be_reinstalled(self):
        install = Mock(side_effect=RuntimeError('reinstall reached'))
        modules = {'common': SimpleNamespace(torch_module=lambda: object()),
                   'engine_profile': SimpleNamespace(_install_arm=install)}
        runtime = SimpleNamespace(ours_vqwam_engine=None)
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(RuntimeError, 'reinstall reached'):
                engine_optimized.install(runtime, {}, Path('/old'), Path('/new'))
        install.assert_called_once_with(runtime, {}, Path('/old'), Path('/new'),
                                        'ours_vqwam', {})


if __name__ == '__main__':
    unittest.main()
