"""Pure-stdlib protocol and cohort checks; no model or tensor computation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from worker import (CASES, HELDOUT, endpoint_units, prepare_output_directories, validate_protocol,
                    validate_schedule, validate_teacher_binding)


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.p = json.loads(Path(__file__).with_name('protocol.json').read_text(encoding='utf-8'))

    def test_frozen_scope(self):
        validate_protocol(self.p)

    def test_tensor_parents_exist_before_first_atomic_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / 'results'
            for shard in (0, 1):
                unit = out / f'stages/moments_{shard}'
                prepare_output_directories(out, unit)
                self.assertTrue(unit.is_dir())
                for folder, name in [('moments', f'{2000 + shard}.pt'), ('banks', f'{shard:04d}.pt')]:
                    with tempfile.NamedTemporaryFile(dir=out / folder, prefix='.' + name, delete=True) as stream:
                        stream.write(b'fixture')

    def test_original_c80_teacher_schema_keeps_plan_and_actual_seed_binding(self):
        row = {'case_id': 2001, 'sampler_seeds': [910004002, 910004003]}
        legacy = {'plan_row': row, 'sampler_seed': row['sampler_seeds'][0]}
        validate_teacher_binding(legacy, row, 0, calibration=True)
        with self.assertRaises(RuntimeError):
            validate_teacher_binding(legacy, row, 0, calibration=False)
        for changed in [dict(legacy, seed_index=1), dict(legacy, sampler_seed=910004003),
                        dict(legacy, plan_row=dict(row, case_id=2002))]:
            with self.assertRaises(RuntimeError):
                validate_teacher_binding(changed, row, 0, calibration=True)
        validate_teacher_binding(dict(legacy, seed_index=0), row, 0, calibration=False)

    def test_training_or_cohort_changes_rejected(self):
        for section, key, value in [('training', 'vq_updates', 16), ('training', 'contexts_per_update', 1),
            ('training', 'rho', 1), ('training', 'rotation_learning_rate', .0001),
            ('allocation', 'gpus', 1), ('quantization', 'D', 2), ('calibration', 'case_ids', CASES[:8])]:
            changed = copy.deepcopy(self.p)
            changed[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(RuntimeError):
                validate_protocol(changed)

    def test_shards_are_disjoint_and_complete(self):
        a, b = endpoint_units(self.p, 0), endpoint_units(self.p, 1)
        self.assertEqual(len(a), 80)
        self.assertEqual(len(b), 80)
        self.assertFalse(set(a) & set(b))
        self.assertEqual(set(a) | set(b), set(endpoint_units(self.p)))

    def test_schedule_rejects_missing_case_and_unbalanced_batches(self):
        suites = self.p['suites']
        rows = {c: {'suite': suites[(c - 2000) // 20]} for c in CASES}
        pools = [[c for c in CASES if rows[c]['suite'] == s] for s in suites]
        schedule = []
        for i in range(8):
            batch = []
            for j, pool in enumerate(pools):
                n = 3 if (j - i) % 4 < 2 else 2
                batch.extend(pool[:n])
                del pool[:n]
            schedule.append(batch)
        validate_schedule(schedule, rows, suites)
        changed = copy.deepcopy(schedule)
        changed[0][0] = changed[1][0]
        with self.assertRaises(RuntimeError):
            validate_schedule(changed, rows, suites)


if __name__ == '__main__':
    unittest.main()
