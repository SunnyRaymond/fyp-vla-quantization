import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import control_native as control


class DependencyTests(unittest.TestCase):
    def test_repack_stages_reuse_four_cpu_cuda12_build_templates(self):
        expected = {
            'diffusion_repack': ('cpnp25drepack', 'diffusion_native.pbs'),
            'svd_fused_repack': ('cpnp25sfrepack', 'svd_fused_build_native.pbs'),
        }
        self.assertEqual({name: control.STAGES[name] for name in expected}, expected)
        for _, script in expected.values():
            source = (control.HERE / script).read_text(encoding='utf-8')
            self.assertIn('#PBS -l select=1:ncpus=4:mem=32gb', source)
            self.assertIn('#PBS -l walltime=00:59:00', source)

    def test_cascade_retry_requires_a_verified_failed_ancestor(self):
        root = 'job_state = F\ndepend = afterok:11.pbs101@pbs101:\n\t12.pbs101@pbs101\nsubstate = 92\n'
        middle = 'job_state = F\ndepend = afterok:13.pbs101@pbs101\nsubstate = 92\n'
        statuses = {'11.pbs101': 'job_state = Q\n', '12.pbs101': middle,
                    '13.pbs101': 'job_state = F\nExit_status = 1\n'}
        self.assertEqual(control.afterok_jobs(root), ['11.pbs101', '12.pbs101'])
        self.assertTrue(control.proven_failed_chain(root, lambda job: (0, statuses[job])))
        statuses['13.pbs101'] = 'job_state = F\nExit_status = 0\n'
        self.assertFalse(control.proven_failed_chain(root, lambda job: (0, statuses[job])))
        self.assertFalse(control.proven_failed_chain(root, lambda job: (1, 'timeout')))

    def test_dependency_failure_requires_terminal_parent_and_no_execution(self):
        cancelled = 'job_state = F\nsubstate = 92\ndepend = afterok:123.pbs\n'
        failed = 'job_state = F\nExit_status = 1\n'
        self.assertTrue(control.failed_dependency_without_execution(cancelled, [failed]))
        self.assertFalse(control.failed_dependency_without_execution(cancelled, ['job_state = Q\n']))
        self.assertFalse(control.failed_dependency_without_execution(cancelled + 'exec_host = node/0\n', [failed]))
        self.assertFalse(control.terminal_failure('job_state = F\nExit_status = 0\n'))

    def test_dependencies_only_use_registered_handles(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'capture_handle.json').write_text(json.dumps({'job_id': '123.pbs101'}))
            with patch.object(control, 'HERE', root):
                self.assertEqual(control.dependency_handles(['capture', 'capture'], 'gates'),
                                 {'capture': '123.pbs101'})
                with self.assertRaisesRegex(ValueError, 'itself'):
                    control.dependency_handles(['gates'], 'gates')
                (root / 'capture_handle.json').write_text(json.dumps({'job_id': '123; bad'}))
                with self.assertRaisesRegex(ValueError, 'Invalid'):
                    control.dependency_handles(['capture'], 'gates')


if __name__ == '__main__':
    unittest.main()
