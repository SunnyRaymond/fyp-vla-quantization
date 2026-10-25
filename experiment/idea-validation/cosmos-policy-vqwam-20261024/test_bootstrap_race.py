"""PBS race trust boundaries, without SSH, model work, or real queue mutations."""
from datetime import datetime, timedelta
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import bootstrap_race as race
from common import save_json


class BootstrapRaceTests(unittest.TestCase):
    def test_actual_start_beats_queue_position_and_same_second_tie(self):
        start = datetime(2026, 10, 24, 18)
        self.assertEqual(race.winner_group(start, None), 'single')
        self.assertEqual(race.winner_group(None, start), 'bundle')
        self.assertEqual(race.winner_group(start + timedelta(seconds=1), start), 'bundle')
        self.assertEqual(race.winner_group(start, start), 'single')
        self.assertIsNone(race.started({'job_state': 'Q', 'stime': str(start)}))

    def test_wrong_dependency_is_never_rebound(self):
        ready = {'single': [{'stage': 'teacher', 'job_id': '1.pbs'}],
                 'bundle': {'job_id': '4.pbs'},
                 'dependents': [{'job_id': '5.pbs', 'job_name': 'child'}]}
        with tempfile.TemporaryDirectory() as directory, patch.object(race, 'status', return_value={
                'job_state': 'H', 'depend': 'afterok:99.pbs'}), patch.object(race, 'command') as command:
            with self.assertRaisesRegex(RuntimeError, 'old teacher'):
                race.migrate_dependencies(ready, Path(directory))
            command.assert_not_called()

    def test_bundle_migrates_dependencies_before_canceling_originals(self):
        singles = [{'stage': stage, 'job_id': str(i) + '.pbs', 'job_name': stage}
                   for i, stage in enumerate(['teacher', 'perf_bf16', 'perf_spinquant'], 1)]
        ready = {'registration': 'r1', 'single': singles,
                 'bundle': {'job_id': '4.pbs', 'job_name': 'bundle'}, 'dependents': []}
        events = []
        def status(row):
            return ({'job_state': 'R', 'stime': 'Sat Oct 24 18:00:00 2026', 'exec_host': 'node/0'}
                    if row['job_id'] == '4.pbs' else {'job_state': 'Q'})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_json(root / 'control/bootstrap_race/submissions_ready.json', ready)
            with patch.object(race, 'guard'), patch.object(race, 'status', side_effect=status), \
                    patch.dict(os.environ, {'PBS_JOBID': '4.pbs'}), \
                    patch.object(race, 'migrate_dependencies', side_effect=lambda *_: events.append('migrate') or []), \
                    patch.object(race, 'cancel', side_effect=lambda row: events.append(row['job_id']) or row):
                self.assertEqual(race.claim(root, 'bundle'), 0)
            self.assertEqual(events, ['migrate', '1.pbs', '2.pbs', '3.pbs'])
            self.assertTrue((root / 'control/bootstrap_race/ready_for_work.json').is_file())

    def test_baseline_followup_single_winner_cancels_only_bundle(self):
        singles = [
            {'stage': 'perf_svdquant', 'job_id': '11.pbs', 'job_name': 'cpvq24psvd'},
            {'stage': 'viditq', 'job_id': '12.pbs', 'job_name': 'cpvq24vid'},
        ]
        bundle = {'job_id': '13.pbs', 'job_name': 'cpvq24c2'}
        ready = {'registration': 'c2-r1', 'single': singles, 'bundle': bundle, 'dependents': []}
        starts = {'11.pbs': None, '12.pbs': 'Sat Oct 24 18:00:00 2026',
                  '13.pbs': 'Sat Oct 24 18:00:01 2026'}
        events = []

        def status(row):
            value = starts[row['job_id']]
            return ({'job_state': 'R', 'stime': value, 'exec_host': 'node/0'}
                    if value else {'job_state': 'Q'})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            race_dir = root / 'control/baseline_followup_race'
            save_json(race_dir / 'submissions_ready.json', ready)
            with patch.object(race, 'guard'), patch.object(race, 'status', side_effect=status), \
                    patch.dict(os.environ, {'PBS_JOBID': '12.pbs'}), \
                    patch.object(race, 'cancel', side_effect=lambda row: events.append(row['job_id']) or row):
                self.assertEqual(race.claim(root, 'single', 'viditq', 'control/baseline_followup_race'), 0)
            self.assertEqual(events, ['13.pbs'])
            winner = race.read_json(race_dir / 'ready_for_work.json')
            self.assertEqual(winner['candidate'], 'single')
            self.assertEqual(winner['coordinator_job'], '12.pbs')
            self.assertFalse((root / 'control/bootstrap_race/ready_for_work.json').exists())

    def test_baseline_followup_bundle_cancels_only_registered_two_singles(self):
        singles = [
            {'stage': 'perf_svdquant', 'job_id': '21.pbs', 'job_name': 'cpvq24psvd'},
            {'stage': 'viditq', 'job_id': '22.pbs', 'job_name': 'cpvq24vid'},
        ]
        bundle = {'job_id': '23.pbs', 'job_name': 'cpvq24c2'}
        ready = {'registration': 'c2-r2', 'single': singles, 'bundle': bundle, 'dependents': []}
        events = []

        def status(row):
            if row['job_id'] == bundle['job_id']:
                return {'job_state': 'R', 'stime': 'Sat Oct 24 18:00:00 2026', 'exec_host': 'node/1'}
            return {'job_state': 'Q'}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            race_dir = root / 'control/baseline_followup_race'
            save_json(race_dir / 'submissions_ready.json', ready)
            with patch.object(race, 'guard'), patch.object(race, 'status', side_effect=status), \
                    patch.dict(os.environ, {'PBS_JOBID': bundle['job_id']}), \
                    patch.object(race, 'migrate_dependencies', return_value=[]), \
                    patch.object(race, 'cancel', side_effect=lambda row: events.append(row['job_id']) or row):
                self.assertEqual(race.claim(root, 'bundle', race_directory='control/baseline_followup_race'), 0)
            self.assertEqual(events, ['21.pbs', '22.pbs'])
            self.assertEqual(race.read_json(race_dir / 'ready_for_work.json')['candidate'], 'bundle')

    def test_race_directory_cannot_escape_task_root(self):
        with patch.object(race, 'guard'):
            with self.assertRaisesRegex(RuntimeError, 'stay inside'):
                race.claim('.', 'bundle', race_directory='../ours_race')

    def test_bundle_with_no_registered_dependents_needs_no_teacher_single(self):
        ready = {'single': [{'stage': 'perf_svdquant', 'job_id': '31.pbs'},
                            {'stage': 'viditq', 'job_id': '32.pbs'}],
                 'bundle': {'job_id': '33.pbs'}, 'dependents': []}
        with patch.object(race, 'status') as status, patch.object(race, 'command') as command:
            self.assertEqual(race.migrate_dependencies(ready, Path('.')), [])
            status.assert_not_called()
            command.assert_not_called()

    def test_custom_single_race_fails_closed_until_exact_registration_exists(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(race, 'guard'), \
                patch.dict(os.environ, {'PBS_JOBID': '31.pbs'}):
            with self.assertRaisesRegex(RuntimeError, 'registered counterpart identities'):
                race.claim(Path(directory), 'single', 'perf_svdquant',
                           'control/baseline_followup_race')

    def test_baseline_registration_rejects_unregistered_perf_viditq_single(self):
        ready = {'registration': 'c2-r3',
                 'single': [{'stage': 'perf_svdquant', 'job_id': '31.pbs', 'job_name': 'cpvq24psvd'},
                            {'stage': 'viditq', 'job_id': '32.pbs', 'job_name': 'cpvq24vid'}],
                 'bundle': {'job_id': '33.pbs', 'job_name': 'cpvq24c2'}, 'dependents': []}
        with tempfile.TemporaryDirectory() as directory:
            race_dir = Path(directory) / 'control/baseline_followup_race'
            save_json(race_dir / 'submissions_ready.json', ready)
            with patch.object(race, 'guard'), patch.dict(os.environ, {'PBS_JOBID': '34.pbs'}), \
                    patch.object(race, 'status') as status:
                with self.assertRaisesRegex(RuntimeError, 'registered race candidate'):
                    race.claim(Path(directory), 'single', 'perf_viditq',
                               'control/baseline_followup_race')
                status.assert_not_called()

    def test_baseline_registration_rejects_extra_single_before_any_status_or_cancel(self):
        ready = {'registration': 'c2-r4',
                 'single': [{'stage': 'perf_svdquant', 'job_id': '41.pbs', 'job_name': 'cpvq24psvd'},
                            {'stage': 'viditq', 'job_id': '42.pbs', 'job_name': 'cpvq24vid'},
                            {'stage': 'teacher', 'job_id': '43.pbs', 'job_name': 'cpvq24teach'}],
                 'bundle': {'job_id': '44.pbs', 'job_name': 'cpvq24c2'}, 'dependents': []}
        with tempfile.TemporaryDirectory() as directory:
            race_dir = Path(directory) / 'control/baseline_followup_race'
            save_json(race_dir / 'submissions_ready.json', ready)
            with patch.object(race, 'guard'), patch.dict(os.environ, {'PBS_JOBID': '41.pbs'}), \
                    patch.object(race, 'status') as status, patch.object(race, 'cancel') as cancel:
                with self.assertRaisesRegex(RuntimeError, 'exactly two singles'):
                    race.claim(Path(directory), 'single', 'perf_svdquant',
                               'control/baseline_followup_race')
                status.assert_not_called()
                cancel.assert_not_called()


if __name__ == '__main__':
    unittest.main()
