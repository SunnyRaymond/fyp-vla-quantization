import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import worker


class TrainingPolicy(unittest.TestCase):
    def test_both_fixed_controls_start_from_common_update_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for arm,rho in (('low',.1),('current',1.5)):
                out=root/arm;out.mkdir();steps=[];loaded=[];restored=[];finished=[]
                def load(path,**kwargs):
                    loaded.append(Path(path));return {'updates':0}
                e=SimpleNamespace(p={'protocol':'test','training':{'fixed_rhos':{'low':.1,'current':1.5}},
                                     'calibration':{'case_ids':list(range(8))}},root=root,out=out,
                    torch=SimpleNamespace(load=load),install=lambda:None,restore=restored.append,
                    snapshot=lambda updates:{'updates':updates},checkpoint=lambda *args:None,
                    train_step=lambda case,rho,index,kind:steps.append((case,rho,index)) or {'update':index+1},
                    complete=lambda **value:finished.append(value))
                worker.fixed_training(e,arm)
                self.assertEqual(loaded,[root/'results/initial_state.pt'])
                self.assertEqual(restored,[{'updates':0}])
                self.assertEqual(steps,[(i%8,rho,i) for i in range(16)])
                self.assertEqual(finished[0]['new_committed_updates'],16)
                self.assertEqual(finished[0]['reused_updates'],0)

    def test_old_numerical_endpoint_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'results/train_low';out.mkdir(parents=True)
            path=out/'summary.json';path.write_text(json.dumps({'complete':True,'updates':16,'source_format':'legacy'}))
            with self.assertRaises(RuntimeError):worker.training_info(root,'low')
            path.write_text(json.dumps({'complete':True,'updates':16,'numerics':worker.NUMERICS}))
            self.assertEqual(worker.training_info(root,'low')['numerics'],worker.NUMERICS)


    def test_same_rho_reuses_endpoint_without_loading_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'results/train_current';out.mkdir(parents=True)
            (out/'summary.json').write_text(json.dumps({'complete':True,'updates':16,'numerics':worker.NUMERICS,
                'rho':1.5,'checkpoint':'same-endpoint.pt','source_format':'bilevel'}))
            selected=root/'results/selected_rho.json'
            selected.write_text(json.dumps({'complete':True,'protocol':'test','numerics':worker.NUMERICS,'rho':1.5}))
            p={'remote_root':str(root),'protocol':'test','training':{'fixed_rhos':{'low':.1,'current':1.5}}}
            self.assertTrue(worker.reuse_selected(p,'metadata-test'))
            result=json.loads((root/'results/train_selected/summary.json').read_text())
            self.assertEqual(result['checkpoint'],'same-endpoint.pt')
            self.assertFalse(result['model_loaded']);self.assertEqual(result['new_committed_updates'],0)
            selected.write_text(json.dumps({'complete':True,'protocol':'test','numerics':worker.NUMERICS,'rho':1.0}))
            self.assertFalse(worker.reuse_selected(p,'metadata-test'))


if __name__=='__main__':unittest.main()
