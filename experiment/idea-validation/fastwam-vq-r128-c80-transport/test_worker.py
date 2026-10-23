import ast
import copy
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('base_worker', HERE.parent / 'fastwam-vq-r128-c80-dual/worker.py')
base = importlib.util.module_from_spec(spec)
sys.modules['base_worker'] = base
spec.loader.exec_module(base)
spec = importlib.util.spec_from_file_location('transport_worker_tested', HERE / 'worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class WorkerMetadataTests(unittest.TestCase):
    def test_single_gpu_amendment_cannot_launch_uncomp_training(self):
        w = object.__new__(worker.Work)
        w.execution, w.arm = {'mode': 'comp_only_reuse_joint8'}, 'Uncomp8'
        with self.assertRaisesRegex(RuntimeError, 'only Comp8 may resume'):
            w.train()
        driver = (HERE / 'run.pbs').read_text(encoding='utf-8')
        self.assertIn('select=1:ncpus=16:mem=110gb:ngpus=1', driver)
        self.assertIn('walltime=04:00:00', driver)
        self.assertNotIn('--arm Uncomp8', driver)
        self.assertNotIn('GPU_UUID1', driver)

    def test_refresh_releases_endpoint_cache_before_decoding(self):
        tree = ast.parse((HERE / 'transport_bank.py').read_text(encoding='utf-8'))
        bank = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'TransportBanks')
        refresh = next(node for node in bank.body if isinstance(node, ast.FunctionDef) and node.name == 'refresh')
        case = self

        class Decoder:
            def refresh(self):
                case.assertFalse(self.dense_cache, 'old full weight copy survives into replacement decode')
                case.assertFalse(self.relative_cache)
                self.decoded = True

        fixture = ast.ClassDef(name='TransportBanks', bases=[ast.Name(id='Decoder', ctx=ast.Load())],
            keywords=[], body=[refresh], decorator_list=[])
        module = ast.fix_missing_locations(ast.Module(body=[fixture], type_ignores=[]))
        namespace = {'Decoder': Decoder}
        exec(compile(module, 'transport_bank.py:refresh', 'exec'), namespace)
        instance = namespace['TransportBanks']()
        instance.dense_cache, instance.relative_cache = {'old': object()}, {'old': object()}
        instance.refresh()
        self.assertTrue(instance.decoded)

    def test_query_primes_compensation_before_sampler_checkpoint(self):
        for compensated in (False, True):
            with self.subTest(compensated=compensated):
                events, seen = [], set()
                rotations = SimpleNamespace(raw={'action': None, 'video': None},
                    refresh=lambda: events.append('refresh'), matrix=lambda s: events.append('matrix/' + s))
                bank = SimpleNamespace(rotations=rotations, seen=seen, compensated=compensated,
                    query_reset=lambda: events.append('reset'), relative=lambda s: events.append('relative/' + s))
                w = object.__new__(worker.Work)
                w.student_queries = w.gradient_queries = 0
                w.torch = SimpleNamespace(set_grad_enabled=lambda grad: nullcontext())
                w.bank, w.runtime, w.arm = bank, SimpleNamespace(), 'Comp8'
                w.enough = lambda: None
                w.datum = lambda case: ('input', 'description')
                w.rows = {2000: {'sampler_seeds': [123]}}

                def sample(runtime, value, seed, **kwargs):
                    events.append('sampler')
                    self.assertEqual((value, seed), ('input', 123))
                    self.assertEqual(kwargs, dict(checkpointed=True, return_future=True))
                    seen.update(range(614))
                    return 'action', 'counts', 'future'

                w.sampler = SimpleNamespace(differentiable_action=sample)
                self.assertEqual(w.query(2000, grad=True), ('action', 'future', 'counts'))
                expected = ['reset', 'refresh']
                for stream in rotations.raw:
                    expected.append('matrix/' + stream)
                    if compensated:
                        expected.append('relative/' + stream)
                self.assertEqual(events, expected + ['sampler'])
                self.assertEqual((w.student_queries, w.gradient_queries), (1, 1))

    def test_guarded_import_has_no_numerical_runtime(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('numpy', sys.modules)

    def test_full_context_partition_keeps_both_seeds_together(self):
        left, right = worker.units(0), worker.units(1)
        self.assertEqual(len(left), 80)
        self.assertEqual(len(right), 80)
        self.assertEqual(set(left) | set(right), set(worker.units()))
        self.assertFalse(set(left) & set(right))
        for shard in (left, right):
            for split, case, seed in shard:
                if split == 'heldout':
                    self.assertIn((split, case, 1 - seed), shard)

    def test_frozen_design_rejects_changed_training_or_cohort(self):
        p = json.loads((HERE / 'protocol.json').read_text(encoding='utf-8'))
        worker.validate(p)
        for section, field, bad in [('training', 'rotation_learning_rate', 1e-4),
                                    ('training', 'extra_losses', ['gripper']),
                                    ('cohorts', 'heldout', {'case_ids': [3000], 'seed_indices': [0, 1]}),
                                    ('source_dual', 'job', 'wrong')]:
            q = copy.deepcopy(p)
            q[section][field] = bad
            with self.assertRaises(RuntimeError):
                worker.validate(q)

    def test_reused_endpoint_preserves_and_checks_query_binding(self):
        w = object.__new__(worker.Work)
        w.rows = {2000: dict(suite='libero_spatial', sampler_seeds=[123])}
        w.job = 'test.pbs101'
        values = {k: .1 for k in worker.METRICS}
        with tempfile.TemporaryDirectory() as directory:
            w.out = Path(directory)
            result = w.endpoint('Seq_A4', 'calibration', 2000, 0, values=values, source_binding={'job': 'source'})
            self.assertEqual(result['source_binding'], {'job': 'source'})
            self.assertEqual(w.endpoint('Seq_A4', 'calibration', 2000, 0), result)
            path = w.out / 'endpoints/Seq_A4/calibration/2000_0.json'
            result['sampler_seed'] = 999
            worker.save(path, result)
            with self.assertRaises(RuntimeError):
                w.endpoint('Seq_A4', 'calibration', 2000, 0)


if __name__ == '__main__':
    unittest.main()
