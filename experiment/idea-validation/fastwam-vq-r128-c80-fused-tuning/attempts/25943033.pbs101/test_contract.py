"""Small stdlib checks for full-panel scope and actual packed kernel dispatch."""
import ast
import importlib.util
import json
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent


class ContractTests(unittest.TestCase):
    def test_full_frozen_panel_and_long_allocation(self):
        p=json.loads((HERE/'protocol.json').read_text())
        self.assertEqual(p['profiling']['case_ids'],list(range(3000,3040)))
        self.assertEqual(p['profiling']['seed_indices'],[0,1])
        self.assertEqual(p['profiling']['queries_per_mode'],80)
        self.assertGreater(int(p['allocation']['walltime'].split(':')[0]),2)
        self.assertEqual(p['export']['new_training_updates'],0)

    def test_all_source_parses(self):
        for path in HERE.glob('*.py'): ast.parse(path.read_text(),filename=str(path))

    def test_fused_consumer_has_no_dense_decode(self):
        tree=ast.parse((HERE/'fused_vq.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepared_linear')
        self.assertFalse(any(isinstance(n,ast.Name) and n.id=='decode_weight' for n in ast.walk(fn)))
        self.assertIn('prepared_linear',(HERE/'kernel_runtime.py').read_text())

    def test_completion_gate_rejects_missing_kernel_validation(self):
        spec=importlib.util.spec_from_file_location('kernel_control',HERE/'control.py')
        control=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(control)
        self.assertFalse(control.complete_gate({}))
        self.assertIn('kernel_validation_complete',(HERE/'control.py').read_text())


if __name__=='__main__': unittest.main()
