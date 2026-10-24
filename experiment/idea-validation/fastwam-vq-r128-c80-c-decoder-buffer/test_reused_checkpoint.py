"""Producer identity regression using the actual completed GPU receipts."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from finalize import complete_modes, decoder_checkpoint_matches


class ReusedCheckpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).parent / 'pulled/25960283.pbs101'
        cls.decoder = json.loads((root / 'c_decoder/tuning.json').read_text())
        cls.buffer = json.loads((root / 'c_buffer/tuning.json').read_text())

    def test_real_distinct_jobs_preserve_checkpoint_and_tuning_origin(self):
        self.assertNotEqual(self.decoder['pbs_jobid'], self.buffer['pbs_jobid'])
        self.assertTrue(decoder_checkpoint_matches(self.decoder, self.buffer))

    def test_wrong_source_job_is_rejected(self):
        changed = deepcopy(self.buffer)
        changed['reuse_source_pbs_jobid'] = 'another.pbs101'
        self.assertFalse(decoder_checkpoint_matches(self.decoder, changed))

    def test_changed_checkpoint_is_rejected(self):
        changed = deepcopy(self.buffer)
        changed['checkpoint_identity']['source_job'] = 'another.pbs101'
        self.assertFalse(decoder_checkpoint_matches(self.decoder, changed))

    def test_real_report_has_only_the_repaired_producer_rule_failure(self):
        root = Path(__file__).parent
        report = json.loads((root / 'pulled/25960283.pbs101/summary.json').read_text())
        self.assertFalse(report['complete'])
        checks = report['modes']['c_buffer']['validation_checks']
        self.assertEqual([key for key, passed in checks['checks'].items() if not passed],
                         ['decoder_checkpoint_matches_c_decoder'])
        checks['checks']['decoder_checkpoint_matches_c_decoder'] = decoder_checkpoint_matches(
            self.decoder, self.buffer)
        checks['passed'] = all(checks['checks'].values())
        protocol = json.loads((root / 'long/protocol.json').read_text())
        sources = {}
        for key in ('candidate_reuse', 'completed_candidate_reuse'):
            for mode in protocol[key]['modes']:
                sources[mode] = {field: protocol[key][field]
                                 for field in ('source_job', 'source_protocol')}
        self.assertTrue(complete_modes(report['modes'], report['stage_gates'], set(sources), sources))


if __name__ == '__main__':
    unittest.main()
