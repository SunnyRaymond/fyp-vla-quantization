"""Exercise actual adapter metadata code without loading a model or Torch."""
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


class AdapterMetadataTests(unittest.TestCase):
    def test_real_module_loads_frozen_c8_rows(self):
        runner = types.ModuleType('runner')
        runner.Runtime = object
        spec = importlib.util.spec_from_file_location('adapter_metadata_test', HERE / 'vq_adapter.py')
        adapter = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'runner': runner}):
            spec.loader.exec_module(adapter)
        protocol = json.loads((HERE / 'protocol.json').read_text(encoding='utf-8'))
        suites = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
        ids = protocol['calibration']['case_ids']
        plan = {'protocol': 'fastwam-vq-calibration-suites-v1', 'status': 'complete',
                'calibration_cohorts': {'c8': ids},
                'inputs': [{'case_id': case, 'suite': suites[index // 2]}
                           for index, case in enumerate(ids)]}
        with patch.object(adapter, 'read_json', return_value=plan):
            self.assertEqual(set(adapter._c8_rows(protocol)), set(ids))


if __name__ == '__main__':
    unittest.main()
