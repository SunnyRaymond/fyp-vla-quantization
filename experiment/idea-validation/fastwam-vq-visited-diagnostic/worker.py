"""Frozen same-input visited-state diagnostic; model work is PBS-only."""
import json
import math
import os
from pathlib import Path
import time

import metrics

PROTOCOL = 'fastwam-vq-visited-diagnostic-v1'
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
VARIANTS = ('base', 'image_plus', 'image_minus', 'proprio_plus', 'proprio_minus')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def validate_protocol(p):
    if (p.get('protocol') != PROTOCOL or p.get('case_ids') != list(range(2001, 2080, 2))
            or p.get('arms') != ['bf16', 'clip_only', 'joint']
            or p.get('variants') != list(VARIANTS) or p.get('seed_index') != 0
            or p.get('executed_steps') != 10 or p.get('expected_records') != 600
            or p.get('training_updates') != 0 or p.get('new_rollouts') != 0
            or p.get('channel_scaling') is not False):
        raise ValueError('Frozen 40-state same-input diagnostic contract differs')
    if (p['perturbations']['image']['raw_uint8_step'] != 1
            or p['perturbations']['image']['epsilon'] != 1 / 255
            or p['perturbations']['proprio']['epsilon'] != 0.005
            or p['perturbations']['proprio']['direction'] != [1/math.sqrt(3)]*3):
        raise ValueError('Predeclared perturbations differ')
    return p


def binding(p, row, arm, variant):
    return {'protocol': p['protocol'], 'case_id': row['case_id'], 'suite': row['suite'],
            'arm': arm, 'variant': variant, 'seed_index': 0,
            'sampler_seed': row['sampler_seeds'][0], 'plan_row': row,
            'source_checkpoint': p['checkpoints'].get(arm, 'original BF16 runtime checkpoint')}


def validate_record(record, p, row, arm, variant):
    if any(record.get(k) != v for k, v in binding(p, row, arm, variant).items()):
        raise ValueError('Endpoint input/seed/checkpoint binding differs')
    metrics.endpoint_metrics(record, record)
    if record.get('training_updates') != 0 or not record.get('pbs_jobid'):
        raise ValueError('Endpoint must be inference-only and PBS-bound')
    if any(v[6] not in (-1, 0, 1) for v in record['commands']):
        raise ValueError('Actual gripper commands must be -1/0/+1')


def require_panel(records, p):
    expected = {(c, a, v) for c in p['case_ids'] for a in p['arms'] for v in p['variants']}
    keys = [(r['case_id'], r['arm'], r['variant']) for r in records]
    if len(keys) != 600 or len(set(keys)) != 600 or set(keys) != expected:
        raise ValueError('Exactly 40 states x 3 arms x 5 variants required')
    for case in p['case_ids']:
        for variant in VARIANTS:
            paired = [r for r in records if r['case_id']==case and r['variant']==variant]
            if any(r['sampler_seed']!=paired[0]['sampler_seed']
                   or r['input_perturbation']!=paired[0]['input_perturbation'] for r in paired):
                raise ValueError('All arms must use identical cached input and seed')


def sensor_selfcheck(np, p):
    obs={'agentview_image':np.array([[[0,127,255]]],dtype=np.uint8),
         'robot0_eye_in_hand_image':np.array([[[255,127,0]]],dtype=np.uint8),
         'robot0_eef_pos':np.array([0.1,0.2,0.3]),
         'robot0_eef_quat':np.array([0.,0.,0.,1.]),'robot0_gripper_qpos':np.array([0.01,0.01])}
    positive,_=perturb_observation(np,obs,'image_plus',p)
    negative,_=perturb_observation(np,obs,'image_minus',p)
    assert positive['agentview_image'].tolist()==[[[1,128,255]]]
    assert negative['agentview_image'].tolist()==[[[0,126,254]]]
    assert obs['agentview_image'].tolist()==[[[0,127,255]]]
    moved,_=perturb_observation(np,obs,'proprio_plus',p)
    assert abs(float(np.linalg.norm(moved['robot0_eef_pos']-obs['robot0_eef_pos']))-0.005)<1e-12
    assert np.array_equal(moved['robot0_eef_quat'],obs['robot0_eef_quat'])
    assert np.array_equal(moved['robot0_gripper_qpos'],obs['robot0_gripper_qpos'])
    assert np.array_equal(moved['agentview_image'],obs['agentview_image'])
    return {'passed':True,'brightness_bounds_no_wrap':True,'raw_eef_delta_l2_m':0.005}


def perturb_observation(np, obs, variant, p):
    """Perturb raw sensors before the unchanged deployed preprocessing."""
    result = dict(obs)
    info = {'variant': variant, 'raw_eef_xyz_delta_m': [0.0]*3, 'raw_image_delta': {}}
    if variant == 'base':
        return result, info
    sign = 1 if variant.endswith('_plus') else -1
    if variant.startswith('image_'):
        for key in ('agentview_image', 'robot0_eye_in_hand_image'):
            original = np.asarray(obs[key])
            if original.dtype != np.uint8 or original.ndim != 3 or original.shape[-1] != 3:
                raise ValueError('Expected two raw uint8 RGB cameras')
            integer = original.astype(np.int16) + sign
            result[key] = np.clip(integer, 0, 255).astype(np.uint8)
            difference = (result[key].astype(np.float64) - original) / 255
            info['raw_image_delta'][key] = {
                'rms_intensity': float(np.sqrt(np.mean(difference**2))),
                'saturated_fraction': float(np.mean((integer < 0) | (integer > 255))),
                'changed_fraction': float(np.mean(difference != 0)),
            }
    elif variant.startswith('proprio_'):
        epsilon = p['perturbations']['proprio']['epsilon']
        delta = sign * epsilon * np.asarray(p['perturbations']['proprio']['direction'])
        original = np.asarray(obs['robot0_eef_pos'])
        if original.shape != (3,) or not np.isfinite(original).all():
            raise ValueError('Expected raw EEF xyz')
        result['robot0_eef_pos'] = original.astype(np.float64) + delta
        info['raw_eef_xyz_delta_m'] = delta.tolist()
    else:
        raise ValueError('Unknown perturbation variant')
    return result, info


def geometry(base, plus, minus, teacher_base, teacher_plus, teacher_minus, epsilon):
    del base, teacher_base
    student = [(plus['action'][t][d]-minus['action'][t][d])/(2*epsilon)
               for t in range(10) for d in range(6)]
    teacher = [(teacher_plus['action'][t][d]-teacher_minus['action'][t][d])/(2*epsilon)
               for t in range(10) for d in range(6)]
    ss, tt = sum(v*v for v in student), sum(v*v for v in teacher)
    return {'student_rms': math.sqrt(ss/60), 'teacher_rms': math.sqrt(tt/60),
            'error_rms': math.sqrt(sum((s-t)**2 for s,t in zip(student,teacher))/60),
            'gain': math.sqrt(ss/tt) if tt else None,
            'cosine': sum(s*t for s,t in zip(student,teacher))/math.sqrt(ss*tt) if ss and tt else None}


def summarize(root, p, records):
    require_panel(records, p)
    result = metrics.summarize(records)
    if not result['coverage']['complete_600_record_panel']:
        raise RuntimeError('Metrics summary does not cover the entire paired panel')
    for item in result.pop('case_metrics'):
        save(root/'results/case_metrics'/f"{item['case_id']}_{item['variant']}.json", item)
    indexed = {(r['case_id'], r['arm'], r['variant']): r for r in records}
    sensitivity = []
    for case in p['case_ids']:
        for arm in ('clip_only', 'joint'):
            for family in ('image', 'proprio'):
                variants = ('base', family+'_plus', family+'_minus')
                samples = [indexed[case, a, v] for a in (arm, 'bf16') for v in variants]
                epsilon = p['perturbations'][family]['epsilon']
                item = {'case_id': case, 'suite': samples[0]['suite'], 'arm': arm,
                        'family': family, 'epsilon': epsilon,
                        'normalized': metrics.sensitivity_metrics(*samples, epsilon),
                        'executed_command': metrics.sensitivity_metrics(
                            *[{**r, 'action': r['commands']} for r in samples], epsilon),
                        'geometry': geometry(*samples, epsilon)}
                sensitivity.append(item)
                save(root/'results/sensitivity_cases'/f'{case}_{arm}_{family}.json', item)
    result['sensitivity'] = metrics.aggregate_sensitivity(sensitivity)
    if not result['sensitivity']['coverage']['complete_160_row_panel']:
        raise RuntimeError('Both modalities and arms must cover all 40 sensitivity cases')
    result.update(protocol=p['protocol'], complete=True, training_updates=0, new_rollouts=0,
                  pbs_jobid=os.environ['PBS_JOBID'], scope=p['scope'],
                  motor_dimensions=p['motor_dimensions'], gripper_command_labels=p['gripper_command_labels'])
    save(root/'results/comparison.json', result)
    return result


def render_report(result):
    base = result['endpoint']['base']['overall_case_weighted']
    lines = ['# 40 visited states：同输入动作诊断', '',
             '40个固定states、原seed_index0。每个state的5个输入变体在三组间完全配对。无训练、无新增rollout。', '',
             '| Motor dimension | clip-only normalized RMSE | joint normalized RMSE |', '|---|---:|---:|']
    for d, name in enumerate(result['motor_dimensions']):
        lines.append(f'| {name} | {base["clip_only_rmse_by_dimension"][d]:.9f} | {base["joint_rmse_by_dimension"][d]:.9f} |')
    lines += ['', '| Base指标 | clip-only | joint |', '|---|---:|---:|']
    for label, getter in (
        ('First10 motor RMSE', lambda e:e['motor']['normalized_action_overall_rmse_case_weighted']),
        ('Gripper normalized RMSE', lambda e:e['gripper']['normalized_rmse']),
        ('Gripper command disagreement /400', lambda e:e['gripper']['command_sign_disagreement_count']),
        ('Gripper switch timing mismatch /360', lambda e:e['gripper']['switch_timing_mismatch_count'])):
        values = [getter(base[a+'_endpoint_case_weighted']) for a in ('clip_only', 'joint')]
        lines.append(f'| {label} | {values[0]:.9g} | {values[1]:.9g} |')
    lines += ['', '局部敏感性：图像为两路相机brightness ±1 uint8，proprio为原始EEF xyz沿单位对角方向±5mm。',
              '响应与BF16的误差及各suite/维度统计见comparison.json；逐case完整动作和开合序列另存records/case_metrics/sensitivity_cases。',
              '不以响应越小判更好；两条有限差分方向不能代表完整Jacobian或闭环稳健性。',
              '实际gripper command -1=open、+1=close、0为阈值零指令；单位和反归一化路径与实际部署一致。']
    return '\n'.join(lines)+'\n'


def main():
    import visited_common as vc
    vc.guard()
    import numpy as np
    import torch
    import runner
    import c80_adapter
    import c80_contract
    from clip_banks import ClipBanks
    artifact = Path(os.environ['ARTIFACTS'])
    p = validate_protocol(read(artifact/'protocol.json'))
    root = Path(p['remote_root'])
    save(root/'results/sensor_selfcheck.json',sensor_selfcheck(np,p))
    rp = vc.validate(read(Path(p['visited_root'])/'protocol.json'))
    cp = read(Path(p['c80_root'])/'protocol.json')
    c80_contract.validate_protocol(cp)
    if rp['protocol'] != p['visited_protocol'] or cp['protocol'] != p['c80_protocol']:
        raise RuntimeError('Source protocol differs')
    if cp['source_c80_protocol'] != rp['source_c80_protocol']:
        raise RuntimeError('Visited source and C80 checkpoints use different original pools')
    prep = read(Path(p['visited_root'])/'results/prepare/summary.json')
    if not (prep['complete'] and prep['pbs_jobid']==p['visited_prepare_job']
            and prep['visited_contexts']==40 and prep['teacher_parity_max_abs']==0
            and prep['render_parity_passed']):
        raise RuntimeError('Existing visited states/teacher are incomplete')
    rows = vc.expected_rows(rp)
    plan = read(Path(p['visited_root'])/'data/plan.json')
    if plan.get('status')!='complete' or {r['case_id']:r for r in plan['inputs']} != rows:
        raise RuntimeError('Visited plan changed')
    if {s:sum(rows[c]['suite']==s for c in p['case_ids']) for s in SUITES} != dict.fromkeys(SUITES,10):
        raise RuntimeError('All four suites need 10 visited states')
    preflight = {a:c80_adapter.validate_training_artifacts(cp,a,Path(p['checkpoints'][a]).parent)
                 for a in ('clip_only','joint')}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261018)
    started = int(os.environ['FW_JOB_START_EPOCH'])
    soft = int(os.environ['FW_SOFT_DEADLINE_SECONDS'])
    state = {'protocol':p['protocol'],'pbs_jobid':os.environ['PBS_JOBID'],'complete':False,
             'phase':'loading_runtime','records_completed':0,'expected_records':600,
             'new_model_queries':0,'training_updates':0,'new_rollouts':0}

    def progress(**items):
        state.update(items)
        save(root/'results/progress.json',state)
        print('VISITED_DIAGNOSTIC_PROGRESS '+json.dumps(state),flush=True)

    def deadline():
        if time.time()-started > soft-120:
            raise TimeoutError('Soft deadline; complete records retained without reducing scope')

    runtime = None
    try:
        progress()
        runtime = runner.Runtime(root/'results')
        runtime.model.eval()
        runtime.arm('bf16')
        for value in runtime.model.parameters(): value.requires_grad_(False)
        datums, input_info, teachers = {}, {}, {}
        for case in p['case_ids']:
            obs, meta = vc.load_observation(np,rp,rows[case])
            saved = torch.load(vc.teacher_path(rp,rows[case],0),map_location='cpu',weights_only=True)
            if saved['plan_row']!=rows[case] or saved['sampler_seed']!=rows[case]['sampler_seeds'][0]:
                raise RuntimeError('Cached BF16 teacher input/seed binding differs')
            teachers[case] = saved['action'].float()
            for variant in VARIANTS:
                perturbed, info = perturb_observation(np,obs,variant,p)
                datums[case,variant] = runtime.datum(perturbed)
                info['description'] = meta['description']
                input_info[case,variant] = info
            for variant in VARIANTS:
                info = input_info[case,variant]
                for key in ('image','proprio'):
                    diff=datums[case,variant][key].float()-datums[case,'base'][key].float()
                    info['model_'+key+'_delta_rms']=float(diff.square().mean().sqrt())
                    info['model_'+key+'_delta_max_abs']=float(diff.abs().max())
                info['normalized_proprio_bf16'] = datums[case,variant]['proprio'].float().cpu().tolist()
        records=[]
        parities={}
        banks=None

        def query(case,variant):
            deadline()
            if banks is not None: banks.seen.clear()
            runtime.description=input_info[case,variant]['description']
            before=time.monotonic()
            action,_=runtime.infer(datums[case,variant],rows[case]['sampler_seeds'][0],measure=False)
            if banks is not None and len(banks.seen)!=614:
                raise RuntimeError('All 614 deployed Linears must execute')
            state['new_model_queries']+=1
            return action.detach().float().cpu(),time.monotonic()-before

        def make_record(case,arm,variant,action,seconds,origin):
            commands=runtime.command(action.clone())
            continuous=runtime.ev._denormalize_action(action.clone(),runtime.processor)[0].copy()
            continuous[:,-1]=-(2*continuous[:,-1]-1)
            if not np.isfinite(continuous).all() or not np.array_equal(np.sign(continuous[:,6]),commands[:,6]):
                raise RuntimeError('Actual gripper command conversion differs')
            return {**binding(p,rows[case],arm,variant),'pbs_jobid':os.environ['PBS_JOBID'],
                    'action':action.tolist(),'commands':commands.tolist(),
                    'continuous_commands':continuous.tolist(),'query_seconds':seconds,
                    'origin':origin,'training_updates':0,'input_perturbation':input_info[case,variant]}

        for arm in p['arms']:
            progress(phase='evaluating',arm=arm)
            if arm!='bf16':
                deployment=torch.load(p['checkpoints'][arm],map_location='cpu',weights_only=True)
                c80_contract.validate_deployment(deployment,cp,arm)
                c80_adapter._runtime_type()._validate_deployment(torch,deployment,cp,arm,preflight[arm])
                if banks is None:
                    banks=ClipBanks(runtime,p['bank_directory'],deployment['module_masters'],'clip_only',cp['clipping'])
                else:
                    banks.restore(deployment['module_masters'])
                banks.restore_clips(deployment['ratios'])
                for group in (banks.books,banks.logs,banks.ratios):
                    for value in group.values():value.requires_grad_(False)
                runtime.current_arm=arm
                ref=preflight[arm]['reference_action']
                obs,meta=vc.load_observation(np,rp,rows[2000])
                runtime.description=meta['description'];banks.seen.clear();deadline()
                check,_=runtime.infer(runtime.datum(obs),ref['sampler_seed'],measure=False)
                state['new_model_queries']+=1
                error=float((check.float().cpu()-torch.tensor(ref['action'])).abs().max())
                if error>1e-6 or len(banks.seen)!=614:raise RuntimeError('C80 deployment reference replay failed')
                parities[arm+'_reference_max_abs']=error
                del deployment
                torch.cuda.empty_cache()
            if any(v.requires_grad for v in runtime.model.parameters()) or (banks is not None and
                    any(v.requires_grad for group in (banks.books,banks.logs,banks.ratios) for v in group.values())):
                raise RuntimeError('Evaluation must freeze all model and bank parameters')
            for case in p['case_ids']:
                for variant in VARIANTS:
                    if arm=='bf16' and variant=='base' and case!=p['case_ids'][0]:
                        action,seconds,origin=teachers[case],0.0,'cached_bf16_teacher'
                    else:
                        action,seconds=query(case,variant);origin='fresh_query'
                    if arm=='bf16' and variant=='base':
                        error=float((action-teachers[case]).abs().max())
                        if error>1e-6:raise RuntimeError('BF16 visited-state cached teacher parity failed')
                        if case==p['case_ids'][0]:parities['bf16_cached_teacher_max_abs']=error
                    record=make_record(case,arm,variant,action,seconds,origin)
                    validate_record(record,p,rows[case],arm,variant)
                    save(root/'results/records'/arm/f'{case}_{variant}.json',record)
                    records.append(record)
                    progress(records_completed=len(records),case_id=case,variant=variant,
                             last_query_seconds=seconds)
            repeated,_=query(p['case_ids'][0],'base')
            first=next(r for r in records if r['case_id']==p['case_ids'][0] and r['arm']==arm and r['variant']=='base')
            error=float((repeated-torch.tensor(first['action'])).abs().max())
            parities[arm+'_same_input_repeat_max_abs']=error
            if error>1e-6:raise RuntimeError('Same-input same-seed inference repeat exceeds tolerance')
        if runtime.q._native_int4_gemm_calls or runtime.q._kv_active:
            raise RuntimeError('Unexpected native INT4 or KV4 path')
        if state['new_model_queries']!=p['expected_model_queries']:
            raise RuntimeError('Actual query count differs from frozen budget')
        progress(phase='aggregating',parities=parities,all_parameters_frozen=True)
        result=summarize(root,p,records)
        (root/'results/RESULTS.zh.md').write_text(render_report(result))
        save(root/'results/input_perturbations.json',{'cases':[{'case_id':c,'variants':{v:input_info[c,v] for v in VARIANTS}} for c in p['case_ids']]})
        progress(phase='complete',complete=True,parities=parities,all_parameters_frozen=True,
                 cached_bf16_base_records=39,new_perturbed_bf16_teacher_queries=160)
        save(root/'results/summary.json',state)
        save(artifact/'summary.json',state)
    except Exception as exc:
        progress(phase='failed',error=type(exc).__name__+': '+str(exc))
        save(root/'results/summary.json',state)
        raise
    finally:
        if runtime is not None and runtime.env is not None:runtime.env.close()


if __name__=='__main__':main()
