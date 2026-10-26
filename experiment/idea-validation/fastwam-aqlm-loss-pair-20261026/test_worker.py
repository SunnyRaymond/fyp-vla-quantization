"""Small protocol/aggregation checks; no model loading or cluster operations."""
import copy
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock
from pathlib import Path

from worker import validate, read, save, beta_from_means, load_capture_case, official_quantize, PROTOCOL
from analysis import aggregate, paired


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.p=read(Path(__file__).with_name('protocol.json'))

    def test_closed_loop_is_rejected(self):
        validate(self.p)
        bad=copy.deepcopy(self.p)
        bad['boundary']['closed_loop_authorized']=True
        with self.assertRaises(ValueError):
            validate(bad)

    def test_full_heldout_is_required(self):
        bad=copy.deepcopy(self.p)
        bad['heldout']['case_ids'].pop()
        with self.assertRaises(ValueError):
            validate(bad)

    def test_requested_minimal_fit_cap_and_budget(self):
        fit=self.p['fitting']
        self.assertEqual((1<<self.p['quantization']['nbits_per_codebook'])*fit['init_max_points_per_centroid'],4096)
        self.assertEqual(fit['init_max_iter'],5)
        self.assertEqual(fit['max_epochs']*fit['steps_per_epoch'],5)
        bad=copy.deepcopy(self.p)
        bad['fitting']['relative_mse_tolerance']=.001
        with self.assertRaises(ValueError):
            validate(bad)

    def test_beta_uses_frozen_initial_rule(self):
        value=beta_from_means(self.p,2,4)
        self.assertAlmostEqual(value,.1*self.p['training']['latent_weight']*.5)
        with self.assertRaises(ValueError):
            beta_from_means(self.p,1,0)

    def test_seeds_then_suites_not_record_mean(self):
        rows=[dict(case_id=1,suite='a',metrics={'error':v}) for v in (0.,2.)]
        rows += [dict(case_id=2,suite='a',metrics={'error':3.})]
        rows += [dict(case_id=3,suite='b',metrics={'error':10.})]
        a=aggregate(rows,['a','b'])
        self.assertEqual(a['mean']['error'],6.)
        b=copy.deepcopy(a)
        for row in b['by_context'].values():
            row['metrics']['error']-=1
        b['mean']['error']-=1
        self.assertEqual(paired(a,b)['error']['wins'],3)
        self.assertEqual(paired(a,b)['error']['delta'],-1.)

    def test_capture_archive_and_legacy_keep_same_target_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            names=['layer_a','layer_b']
            values=[object(),object()]
            receipt=dict(protocol=PROTOCOL,case_id=2000,calls={name:1 for name in names})
            save(root/'cases/2000.json',receipt)
            legacy=SimpleNamespace(load=Mock(side_effect=values))
            a=load_capture_case(legacy,root,2000,names)
            self.assertEqual(legacy.load.call_count,2)
            receipt['layout']='context_archive_v2'
            save(root/'cases/2000.json',receipt)
            archive=SimpleNamespace(load=Mock(return_value=dict(protocol=PROTOCOL,case_id=2000,rows=dict(zip(names,values)))))
            b=load_capture_case(archive,root,2000,names)
            self.assertEqual(archive.load.call_count,1)
            self.assertTrue(all(a[n] is b[n] for n in names))
            archive.load.return_value['case_id']=2001
            with self.assertRaises(ValueError):
                load_capture_case(archive,root,2000,names)

    def test_official_fit_restores_determinism_even_on_error(self):
        settings={'enabled':True,'warn_only':False}
        def change(enabled,warn_only=False):
            settings.update(enabled=enabled,warn_only=warn_only)
        torch=SimpleNamespace(are_deterministic_algorithms_enabled=lambda:settings['enabled'],
            is_deterministic_algorithms_warn_only_enabled=lambda:settings['warn_only'],
            use_deterministic_algorithms=change)
        def failing(**kwargs):
            self.assertFalse(settings['enabled'])
            raise RuntimeError('test official fit error')
        with self.assertRaises(RuntimeError):
            official_quantize(torch,SimpleNamespace(quantize=failing),object())
        self.assertEqual(settings,{'enabled':True,'warn_only':False})


if __name__=='__main__':
    unittest.main()
