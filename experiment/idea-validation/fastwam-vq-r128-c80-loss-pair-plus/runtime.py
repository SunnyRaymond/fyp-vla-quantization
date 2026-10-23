"""Restore calibrated MX-VQ and fixed R0 in the proven paired episode runtime."""
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path

ARMS = ('az', 'reader_kl')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def expected_install_metadata(protocol, arm):
    source = protocol['arm_sources'][arm]
    return dict(protocol=protocol['protocol'], arm=arm,
        source_protocol=source['source_protocol'], source_job=source['source_job'],
        source_deployment=source['deployment'], source_bank=source['source_bank'],
        weight_bits=4, activation_bits=4, execution='decoded_bf16_dense_vq', vq_enabled=True,
        kv_cache_dtype='bf16', native_low_bit_kernel=False, native_int4_gemm_calls=0, module_count=614,
        transform_implementation='signed_h128_cayley_mxvq_v1', smooth_alpha=.5, hadamard_block=128,
        D=1, clipping_ratio=1, rotation_shared_streams=['action','video'], rotation_shape=[128,128],
        rotation_updates=8, vq_updates=8, training_exposures=80,
        new_training_updates=0, all_parameters_frozen=True)


def validate_source_documents(protocol, arm, source_protocol, summary, anchor):
    source = protocol['arm_sources'][arm]
    p = source_protocol
    if (p.get('protocol') != source['source_protocol'] or p.get('remote_root') != source['source_root']
            or p.get('arms') != list(ARMS) or p.get('target_linears') != 614
            or p['source_initialization']['bank_directory'] != source['source_bank']
            or p['source_initialization']['protocol'] != source['source_bank_id']
            or p['training']['vq_updates'] != 8 or p['training']['contexts_per_update'] != 10
            or p['training']['training_exposures'] != 80 or p['training']['rotation_updates'] != 0
            or p['calibration']['case_ids'] != list(range(2000,2080))
            or p['calibration']['plan_file'] != protocol['sources']['calibration_plan']
            or p['quantization']['D'] != 1 or p['quantization']['clipping_ratio'] != 1):
        raise ValueError('Frozen paired calibration source differs')
    expected = dict(protocol=source['source_protocol'],arm=arm,pbs_jobid=source['source_job'],
        updates=8,exposures=80,target_linears=614,frozen_rotation=True)
    if (any(summary.get(k) != v for k,v in {**expected,'complete':True,'status':'complete','final_anchor':True}.items())
            or any(anchor.get(k) != v for k,v in expected.items())
            or summary.get('new_teacher_queries') != 0
            or anchor.get('case_id') != 2000 or anchor.get('seed_index') != 0
            or anchor.get('sampler_seed') != 910004000
            or anchor.get('counts') != {'video_steps':10,'action_steps':10}):
        raise ValueError('Calibration summary or final anchor is incomplete/differently bound')
    beta=summary.get('beta')
    if (type(beta) not in (int,float) or not math.isfinite(beta)
            or (arm=='az' and beta!=0) or (arm=='reader_kl' and beta<=0)):
        raise ValueError('Calibrated loss identity differs')
    for key in ('initial_parity_max_abs','formal_parity_max_abs'):
        value=summary.get(key)
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1e-6:
            raise ValueError('Calibration source parity gate failed')
    return anchor


def _runtime_type():
    import runner as pilot
    spec=importlib.util.spec_from_file_location('loss_pair_episode_base',Path(__file__).with_name('previous_vq_adapter.py'))
    previous=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    base=previous._runtime_type()

    class Runtime(base):
        def __init__(self,out,protocol,worker_id):
            import torch
            from rotation_bank import RotationBanks,VQBanks
            pilot.guard()
            if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
                raise RuntimeError('Match calibration CUBLAS_WORKSPACE_CONFIG=:4096:8 before model loading')
            if type(worker_id) is not int or worker_id not in range(4):
                raise ValueError('Expected one of four fixed workers')
            arm=ARMS[worker_id//2]
            source=protocol['arm_sources'][arm]
            source_protocol=read_json(source['source_protocol_file'])
            summary,anchor=read_json(source['source_summary']),read_json(source['endpoint'])
            validate_source_documents(protocol,arm,source_protocol,summary,anchor)
            whole=read_json(Path(source['source_root'])/'results/summary.json')
            if (not whole.get('complete') or whole.get('pbs_jobid')!=source['source_job']
                    or whole.get('arms',{}).get(arm)!=summary):
                raise ValueError('Whole paired calibration is not complete')
            plan=read_json(protocol['sources']['calibration_plan'])
            rows={r['case_id']:r for r in plan['inputs']}
            if plan['calibration_cohorts']['c80']!=list(range(2000,2080)) or rows[2000]!=anchor['plan_row']:
                raise ValueError('Final anchor and original C80 observation plan differ')
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False
            torch.backends.cudnn.deterministic=True
            torch.use_deterministic_algorithms(True)
            pilot.Runtime.__init__(self,out)
            self.model.eval()
            self.arm('bf16')
            self.protocol,self.arm_name,self.current_arm=protocol,arm,arm
            self.out,self.worker_id=Path(out),worker_id
            self.run_queries=self.linear_calls=0
            self.transform_implementation=protocol['quantization']['transform_implementation']
            rb=RotationBanks(self,source_protocol['calibration']['bank_directory'])
            deployment=torch.load(source['deployment'],map_location='cpu',weights_only=True)
            if (deployment.get('protocol')!=source_protocol or deployment.get('arm')!=arm
                    or deployment.get('pbs_jobid')!=source['source_job'] or deployment.get('updates')!=8
                    or deployment.get('exposures')!=80 or deployment.get('target_linears')!=614
                    or deployment.get('bank_directory')!=source['source_bank']
                    or deployment.get('bank_protocol')!=source['source_bank_id']
                    or deployment.get('frozen_rotation') is not True or deployment.get('beta')!=summary['beta']):
                raise ValueError('Actual final VQ deployment differs')
            original_r=torch.load(source_protocol['source_rotation']['deployment'],map_location='cpu',weights_only=True)
            if (set(deployment['rotations'])!={'action','video'}
                    or any(not torch.equal(v,original_r['rotations'][s]) for s,v in deployment['rotations'].items())):
                raise ValueError('Saved calibration R0 is not the fixed source R0')
            rb.rotations.restore(deployment['rotations'])
            rb.rotations.freeze()
            rb.close()
            self.banks=VQBanks(self,source['source_bank'],rb.rotations,rb.transforms,source['source_bank_id'])
            self.banks.restore(deployment['module_masters'])
            for parameter in self.banks.parameters():
                parameter.requires_grad_(False)
            self.seen,self.forwards=self.banks.seen,self.banks.forwards
            for mod in self.q.modules.values():
                forward=mod.forward
                def counted(value,_forward=forward):
                    self.linear_calls+=1
                    return _forward(value)
                mod.forward=counted
            with torch.no_grad():
                orth=self.banks.rotations.orthogonality()
                movement={s:float((self.banks.rotations.matrix(s)-torch.eye(128,device='cuda')).norm())
                          for s in ('action','video')}
            if (any(not math.isfinite(v) for v in (*orth.values(),*movement.values()))
                    or max(orth.values())>5e-5 or min(movement.values())<=0
                    or any(v.requires_grad for v in self.model.parameters())):
                raise ValueError('Frozen R0/model installation gate failed')
            self.install_metadata={**expected_install_metadata(protocol,arm),
                                   'orthogonality':orth,'rotation_movement':movement}
            self.source_protocol,self.anchor,self.reference_row=source_protocol,anchor,rows[2000]
            del deployment,original_r,rb
            torch.cuda.empty_cache()

        def prepare(self,results_root):
            import numpy as np
            import torch
            import sampler
            import calibration_source
            source=self.protocol['arm_sources'][self.current_arm]
            saved=torch.load(source['anchor_tensor'],map_location='cpu',weights_only=True)
            if any(saved.get(k)!=v for k,v in self.anchor.items()):
                raise ValueError('Saved tensor anchor metadata differs from JSON')
            obs,meta=calibration_source.load_observation(np,Path(self.protocol['sources']['calibration_plan']).parent,self.reference_row)
            self.description=meta['description']
            datum,seed=self.datum(obs),self.anchor['sampler_seed']
            self.banks.seen.clear()
            with torch.no_grad():
                action,counts,future=sampler.differentiable_action(self,datum,seed,checkpointed=False,return_future=True)
            if (counts!={'video_steps':10,'action_steps':10} or len(self.banks.seen)!=614
                    or tuple(action.shape)!=(32,7) or tuple(future.shape)!=(1,48,3,14,28)
                    or not bool(torch.isfinite(action).all()) or not bool(torch.isfinite(future).all())):
                raise ValueError('Full-query anchor shape/coverage gate failed')
            aerr=float((action.float().cpu()-saved['action'].float()).abs().max())
            zerr=float((future.float().cpu()-saved['future'].float()).abs().max())
            actual,_=self.infer(datum,seed,measure=False)
            err=float((actual.float().cpu()-action.float().cpu()).abs().max())
            if (aerr!=0 or zerr!=0 or not math.isfinite(err) or err>1e-6
                    or self.q._native_int4_gemm_calls!=0 or self.q._kv_active):
                raise ValueError(f'Full tensor anchor/formal infer parity failed: {aerr},{zerr},{err}')
            self.run_queries=0
            return dict(protocol=self.protocol['protocol'],worker_id=self.worker_id,arm=self.current_arm,
                pbs_job_id=os.environ['PBS_JOBID'],source_job=source['source_job'],source_endpoint=source['endpoint'],
                passed=True,observed_modules=614,case_id=2000,seed_index=0,sampler_seed=seed,
                source_anchor_passed=True,source_action_max_abs=aerr,source_future_max_abs=zerr,
                actual_infer_action_max_abs=err,startup_queries=2,new_training_updates=0,
                install_metadata=self.install_metadata)

    return Runtime


class R128Runtime:
    def __new__(cls,*args,**kwargs):
        return _runtime_type()(*args,**kwargs)
