"""Small stdlib-only contract tests for the runtime profile worker."""
import ast
import runpy
from pathlib import Path
import unittest


WORKER = Path(__file__).with_name('profile_worker.py')
ns = runpy.run_path(str(WORKER))


class ProfileWorkerContractTests(unittest.TestCase):
    def test_full_panel_and_geometry_are_required(self):
        settings = dict(case_ids=list(range(3000, 3040)), seed_indices=[0, 1],
                        warmup_queries=3, queries_per_mode=80, steps_video=10,
                        steps_action=10, batch_size=1, action_horizon=32, video_frames=9)
        self.assertEqual(ns['validate_panel'](settings), list(range(3000, 3040)))
        for key, value in (('case_ids', list(range(3000, 3010))),
                           ('seed_indices', [0]), ('action_horizon', 16)):
            changed = dict(settings, **{key: value})
            with self.assertRaises(ValueError):
                ns['validate_panel'](changed)

    def test_all_target_modules_must_be_hit_without_a_call_count_gate(self):
        ns['validate_module_coverage']({'module_count': 614, 'seen_count': 614})
        with self.assertRaises(ValueError):
            ns['validate_module_coverage']({'module_count': 614, 'seen_count': 613})
        self.assertNotIn('514880', WORKER.read_text(encoding='utf-8'))

    def test_guard_precedes_torch_import_and_modes_do_not_fallback(self):
        tree = ast.parse(WORKER.read_text(encoding='utf-8'))
        module_torch_imports = [node for node in tree.body if isinstance(node, ast.Import)
                                and any(alias.name == 'torch' for alias in node.names)]
        self.assertEqual(module_torch_imports, [])
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'main')
        guard = next(i for i, node in enumerate(main.body)
                     if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                     and isinstance(node.value.func, ast.Name) and node.value.func.id == 'guard')
        self.assertEqual(ns['MODES'], ('runtime_a', 'runtime_b', 'runtime_c', 'runtime_compile'))
        self.assertLess(guard, next(i for i, node in enumerate(main.body)
                                    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                                    and isinstance(node.value.func, ast.Name)
                                    and node.value.func.id == '_freeze_numerics'))


if __name__ == '__main__':
    unittest.main()
