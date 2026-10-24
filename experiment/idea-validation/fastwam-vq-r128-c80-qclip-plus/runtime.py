"""Frozen final Q0 weights/R plus the source Q-Clip ratios and operand implementation."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

ARMS = ('qclip',)
METRICS = ('first10motor_rmse','first10motor_mse','future_latent_nmse','mixed_az_loss',
           'full32motor_rmse','late22motor_rmse','first10gripper_rmse','first10gripper_disagreement')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def expected_install_metadata(protocol,arm):
    if arm!='qclip':
        raise ValueError('Only the fixed Q-Clip arm is authorized')
    source=protocol['arm_sources'][arm]
    return dict(protocol=protocol['protocol'],arm=arm,source_protocol=source['source_protocol'],
        source_job=source['source_job'],source_deployment=source['deployment'],
        source_q0_checkpoint=source['source_checkpoint'],source_bank=protocol['sources']['shared_bank_source'],
        weight_bits=4,activation_bits=4,execution='decoded_bf16_dense_vq',vq_enabled=True,
        kv_cache_dtype='bf16',native_low_bit_kernel=False,native_int4_gemm_calls=0,module_count=614,
        transform_implementation='signed_h128_cayley_mxvq_qclip_v1',smooth_alpha=.5,hadamard_block=128,D=1,
        clipping_mode='learned_scalar_per_linear_after_Smooth_H_R',clipping_updates=80,
        rotation_shared_streams=['action','video'],rotation_shape=[128,128],rotation_updates=8,
        vq_updates=8,training_exposures=80,new_training_updates=0,all_parameters_frozen=True)


def validate_source_documents(protocol,source_protocol,summary,endpoint):
    source=protocol['arm_sources']['qclip']
    if (source_protocol.get('protocol')!=source['source_protocol']
            or source_protocol.get('remote_root')!=source['source_root']
            or source_protocol.get('source_checkpoint')!=source['source_checkpoint']
            or source_protocol.get('source_job')!='25921795.pbs101'
            or source_protocol.get('source_arm')!='reader_kl'
            or source_protocol['training']['updates']!=80
            or source_protocol['training']['exposures']!=80
            or source_protocol['training']['latent_weight']!=protocol['training']['latent_weight']
            or source_protocol['training']['beta']!=protocol['training']['beta']
            or source_protocol['clipping']!=protocol['clipping']):
        raise ValueError('Frozen Q-Clip protocol or Q0 initialization differs')
    expected=dict(protocol=source['source_protocol'],pbs_jobid=source['source_job'],status='complete',
                  complete=True,updates=80,exposures=80,target_linears=614,weights_frozen=True,
                  rotation_frozen=True,new_teacher_queries=0,closedloop_episodes=0)
    if any(summary.get(k)!=v for k,v in expected.items()):
        raise ValueError('Q-Clip source is incomplete or differently bound')
    for key in ('initial_parity_max_abs','formal_parity_max_abs'):
        value=summary.get(key)
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1e-6:
            raise ValueError('Q-Clip source parity gate failed')
    expected_endpoint=dict(protocol=source['source_protocol'],pbs_jobid=source['source_job'],
        arm='qclip',split='calibration',case_id=2000,seed_index=0,sampler_seed=910004000,
        source_checkpoint=source['source_checkpoint'])
    if any(endpoint.get(k)!=v for k,v in expected_endpoint.items()):
        raise ValueError('Q-Clip fixed final endpoint binding differs')
    for key in METRICS:
        value=endpoint.get(key)
        if type(value) not in (int,float) or not math.isfinite(value) or value<0:
            raise ValueError('Q-Clip reference metric is missing/nonfinite')
    return endpoint


def compare_source_metrics(actual,expected,tolerance):
    errors={}
    for key in METRICS:
        a,b=actual[key],expected[key]
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in (a,b)):
            raise ValueError('Nonfinite Q-Clip metric: '+key)
        errors[key]=abs(a-b)
    if max(errors.values())>tolerance:
        raise ValueError('Q-Clip final checkpoint metrics do not replay: '+str(errors))
    return errors


def _runtime_type():
    import reference_pair_runtime as paired
    base=paired._runtime_type()

    class Runtime(base):
        def __init__(self,out,protocol,worker_id):
            if type(worker_id) is not int or worker_id not in range(4):
                raise ValueError('Expected one of four Q-Clip workers')
            source=protocol['arm_sources']['qclip']
            cfg=read_json(source['source_protocol_file'])
            summary=read_json(source['source_summary'])
            endpoint=read_json(source['endpoint'])
            validate_source_documents(protocol,cfg,summary,endpoint)
            pair_protocol=read_json(protocol['sources']['q0_protocol_file'])
            if (pair_protocol['protocol']!='fastwam-vq-r128-c80-loss-pair-plus-v1'
                    or pair_protocol['arm_sources']['reader_kl']['deployment']!=source['source_checkpoint']):
                raise ValueError('Pinned Q0 runtime does not restore the declared parent checkpoint')
            self.target_protocol,self.target_worker=protocol,worker_id
            self.clip_source_protocol,self.clip_source_summary,self.clip_endpoint=cfg,summary,endpoint
            # Reuse the proven Q0 loader and full tensor anchor before installing clipping.
            base.__init__(self,out,pair_protocol,2+worker_id%2)

        def prepare(self,results_root):
            import torch
            import sampler
            import latent_reference
            import numpy as np
            import calibration_source
            from clip_bank import ClipBank
            q0_receipt=base.prepare(self,results_root)
            protocol=self.target_protocol
            source=protocol['arm_sources']['qclip']
            self.protocol,self.worker_id=protocol,self.target_worker
            self.arm_name=self.current_arm='qclip'
            self.transform_implementation=protocol['quantization']['transform_implementation']
            payload=torch.load(source['deployment'],map_location='cpu',weights_only=True)
            if (payload.get('protocol')!=self.clip_source_protocol or payload.get('updates')!=80
                    or payload.get('exposures')!=80 or payload.get('weights_frozen') is not True
                    or payload.get('rotation_frozen') is not True
                    or payload.get('source_checkpoint')!=source['source_checkpoint']):
                raise ValueError('Actual final80 clipping deployment differs')
            self.clips=ClipBank(self.banks)
            self.clips.restore(payload['ratios'])
            for parameter in self.clips.parameters():
                parameter.requires_grad_(False)
            clipping=self.clips.summary()
            if clipping!=self.clip_source_summary['clipping']:
                raise ValueError('Restored 614 clipping ratios differ from the source summary')
            if (len(self.clips.parameters())!=614 or any(v.requires_grad for v in self.clips.parameters())
                    or any(v.requires_grad for v in self.banks.parameters())
                    or any(v.requires_grad for v in self.model.parameters())
                    or any(v.requires_grad for v in self.banks.rotations.parameters())):
                raise ValueError('Q-Clip inference must freeze every weight, VQ, ratio and rotation')
            # ClipBank installs its exact source forward, so restore query-call accounting afterward.
            for module in self.q.modules.values():
                forward=module.forward
                def counted(value,_forward=forward):
                    self.linear_calls+=1
                    return _forward(value)
                module.forward=counted
            self.install_metadata={**expected_install_metadata(protocol,'qclip'),
                'orthogonality':q0_receipt['install_metadata']['orthogonality'],
                'rotation_movement':q0_receipt['install_metadata']['rotation_movement'],
                'clipping':clipping}
            row=self.reference_row
            if self.clip_endpoint['plan_row']!=row:
                raise ValueError('Q-Clip reference uses a different original observation')
            observation,metadata=calibration_source.load_observation(
                np,Path(protocol['sources']['calibration_plan']).parent,row)
            self.description=metadata['description']
            datum,seed=self.datum(observation),row['sampler_seeds'][0]
            self.banks.seen.clear()
            with torch.no_grad():
                action,counts,future=sampler.differentiable_action(self,datum,seed,checkpointed=False,return_future=True)
            if counts!={'video_steps':10,'action_steps':10} or len(self.banks.seen)!=614:
                raise ValueError('Q-Clip source query missed target Linears or sampler steps')
            teacher=torch.load(Path(self.source_protocol['c80_teacher_directory'])/'2000_0.pt',
                               map_location='cpu',weights_only=True)
            if teacher['plan_row']!=row or teacher['sampler_seed']!=seed:
                raise ValueError('Cached BF16 teacher binding differs')
            delta=action.float()-teacher['action'].to('cuda').float()
            a=delta[:10,:6].square().mean()
            z=latent_reference.future_nmse_loss(torch,future,teacher['future'].to('cuda'))
            commands=self.command(action.detach().clone())
            teacher_commands=self.command(teacher['action'].detach().clone())
            metrics=dict(first10motor_rmse=float(a.sqrt()),first10motor_mse=float(a),
                future_latent_nmse=float(z),mixed_az_loss=float(a+protocol['training']['latent_weight']*z),
                full32motor_rmse=float(delta[:,:6].square().mean().sqrt()),
                late22motor_rmse=float(delta[10:,:6].square().mean().sqrt()),
                first10gripper_rmse=float(delta[:10,6].square().mean().sqrt()),
                first10gripper_disagreement=float((commands[:10,6]!=teacher_commands[:10,6]).mean()))
            errors=compare_source_metrics(metrics,self.clip_endpoint,protocol['startup_checks']['source_metric_max_abs_tolerance'])
            actual,_=self.infer(datum,seed,measure=False)
            formal=float((actual.float().cpu()-action.float().cpu()).abs().max())
            if (not math.isfinite(formal) or formal>1e-6 or len(self.banks.seen)!=614
                    or self.q._native_int4_gemm_calls!=0 or self.q._kv_active):
                raise ValueError('Q-Clip formal action query parity/coverage failed')
            self.run_queries=0
            return dict(protocol=protocol['protocol'],worker_id=self.worker_id,arm='qclip',
                pbs_job_id=os.environ['PBS_JOBID'],source_job=source['source_job'],source_endpoint=source['endpoint'],
                passed=True,observed_modules=614,case_id=2000,seed_index=0,sampler_seed=seed,
                source_metrics_passed=True,source_metric_errors=errors,q0_anchor_passed=True,
                q0_action_max_abs=q0_receipt['source_action_max_abs'],q0_future_max_abs=q0_receipt['source_future_max_abs'],
                actual_infer_action_max_abs=formal,startup_queries=4,new_training_updates=0,
                install_metadata=self.install_metadata)

    return Runtime


class R128Runtime:
    def __new__(cls,*args,**kwargs):
        return _runtime_type()(*args,**kwargs)
