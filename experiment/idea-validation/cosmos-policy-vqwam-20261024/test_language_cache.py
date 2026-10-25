"""CPU queue and exact-key cache contracts without tensor/pickle work."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import common
import prepare_language_cache as cache


class LanguageCacheTests(unittest.TestCase):
    def test_actual_routed_cpu_queue_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            nodefile = Path(directory) / 'nodes'
            nodefile.write_text('compute-node\n')
            environment = {'COSMOS_CPU_ONLY': '1', 'CUDA_VISIBLE_DEVICES': '',
                           'COSMOS_EXECUTION_QUEUE': 'qdev', 'PBS_JOBID': 'unit.pbs',
                           'PBS_NODEFILE': str(nodefile)}
            with patch.dict('os.environ', environment), \
                    patch.object(cache.socket, 'gethostname', return_value='compute-node'), \
                    patch.object(common, 'torch_module', return_value=SimpleNamespace(__version__='mock')):
                self.assertEqual(cache.require_cpu_allocation()['queue'], 'qdev')

    def test_merge_preserves_exact_strings_and_existing_merged_values(self):
        result = cache.combine_caches({'Exact key': 'old'}, {'Exact key': 'new', 'exact key': 'official'})
        self.assertEqual(result, {'Exact key': 'old', 'exact key': 'official'})


if __name__ == '__main__':
    unittest.main()
