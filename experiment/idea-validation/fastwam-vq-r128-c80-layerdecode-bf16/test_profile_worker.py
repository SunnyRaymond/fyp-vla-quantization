"""Small stdlib contract tests for the frozen profile worker."""
import ast
import runpy
from pathlib import Path
import unittest


WORKER = Path(__file__).with_name('profile_worker.py')
ns = runpy.run_path(str(WORKER))


class ProfileWorkerContractTests(unittest.TestCase):
    def test_full_panel_is_required(self):
        validate_panel = ns['validate_panel']
        settings = dict(case_ids=list(range(3000, 3040)), seed_indices=[0, 1],
                        warmup_queries=3, queries_per_mode=80)
        self.assertEqual(validate_panel(settings), list(range(3000, 3040)))
        settings['case_ids'] = list(range(3000, 3010))
        with self.assertRaises(ValueError):
            validate_panel(settings)

    def test_formal_coverage_is_exact(self):
        validate_coverage = ns['validate_coverage']
        validate_coverage(614, 514880)
        with self.assertRaises(ValueError):
            validate_coverage(613, 514880)
        with self.assertRaises(ValueError):
            validate_coverage(614, 514879)

    def test_torch_is_not_imported_at_module_scope_and_guard_precedes_it(self):
        tree = ast.parse(WORKER.read_text(encoding='utf-8'))
        module_torch_imports = [node for node in tree.body if isinstance(node, ast.Import)
                                and any(alias.name == 'torch' for alias in node.names)]
        self.assertEqual(module_torch_imports, [])
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'main')
        guard_call = next(i for i, node in enumerate(main.body)
                          if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                          and isinstance(node.value.func, ast.Name) and node.value.func.id == 'guard')
        torch_import = next(i for i, node in enumerate(main.body)
                            if isinstance(node, ast.Import) and any(a.name == 'torch' for a in node.names))
        self.assertLess(guard_call, torch_import)


if __name__ == '__main__':
    unittest.main()
