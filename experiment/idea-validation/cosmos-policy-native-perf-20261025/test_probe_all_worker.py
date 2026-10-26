import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import probe_all_worker as worker


class ProbeAllWorkerTests(unittest.TestCase):
    def test_diagnostics_share_the_process_and_native_failure_stays_failure(self):
        original, general, torch, device = object(), object(), object(), object()
        labels = []
        def diagnostic(root, path, **kwargs):
            self.assertIs(kwargs['torch_module'], torch)
            self.assertIs(kwargs['device'], device)
            self.assertIs(kwargs['allocation_guarded'], True)
            label = kwargs['extension_label']
            self.assertIs(kwargs['extension'], original if label == 'original' else general)
            labels.append(label)
            return {'status': 'complete'}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(worker.diagnose_nunchaku_pack, 'run', return_value={'status': 'complete'}), \
                 patch.object(worker.kernel_probe, '_extension', return_value=original), \
                 patch.object(worker.native_diffusion, '_extension', return_value=general), \
                 patch.object(worker.diagnose_atom, 'run', side_effect=diagnostic), \
                 patch.object(worker.kernel_probe, 'main', return_value=2) as native:
                self.assertEqual(worker._run_stages(root, torch, device), 2)
                native.assert_called_once()
            report = json.loads((root / 'results/preparation/probe_all_driver.json').read_text())
        self.assertEqual(labels, ['original', 'general'])
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['policy_forward_calls'], 0)

    def test_bad_nunchaku_layout_stops_before_loading_native_operands(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(worker.diagnose_nunchaku_pack, 'run', return_value={'status': 'failed'}), \
                 patch.object(worker.kernel_probe, '_extension') as load:
                with self.assertRaisesRegex(RuntimeError, 'malformed operands'):
                    worker._run_stages(Path(temporary), object(), object())
                load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
