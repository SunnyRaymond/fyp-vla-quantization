"""Check sampler return arity at its C8 callers without importing Torch."""
import ast
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent


class SamplerContractTests(unittest.TestCase):
    def test_callers_match_sampler_return_arity(self):
        source = HERE.parent / 'fastwam-vq-micro-latent-search' / 'sampler.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        result = next(node.value for node in ast.walk(tree)
                      if isinstance(node, ast.Return) and isinstance(node.value, ast.IfExp)
                      and isinstance(node.value.test, ast.Name) and node.value.test.id == 'return_future')
        arity = {True: len(result.body.elts), False: len(result.orelse.elts)}
        checked = 0
        for name in ('prepare_c8.py', 'train_c8.py'):
            callers = ast.parse((HERE / name).read_text(encoding='utf-8'))
            for node in ast.walk(callers):
                if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                    continue
                call = node.value
                if not (isinstance(call.func, ast.Attribute) and call.func.attr == 'differentiable_action'):
                    continue
                requested = next(keyword.value.value for keyword in call.keywords
                                 if keyword.arg == 'return_future' and isinstance(keyword.value, ast.Constant))
                with self.subTest(file=name, line=node.lineno):
                    self.assertEqual(len(node.targets[0].elts), arity[requested])
                checked += 1
        self.assertEqual(checked, 4)


if __name__ == '__main__':
    unittest.main()
