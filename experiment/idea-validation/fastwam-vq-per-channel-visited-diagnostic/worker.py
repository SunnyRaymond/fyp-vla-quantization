"""Frozen visited-panel extension; model and data work require real PBS compute."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import metrics
import source_worker as source

ARM = 'per_channel_clip'
STUDENTS = ('clip_only', 'joint', ARM)
VARIANTS = source.VARIANTS
read, save = source.read, source.save


def validate_protocol(p):
    old = p['source_diagnostic_protocol']
    source.validate_protocol(old)
    for key in ('case_ids', 'seed_index', 'variants', 'perturbations', 'executed_steps', 'output_action_horizon'):
        if p[key] != old[key]: raise ValueError('Frozen diagnostic differs: ' + key)
    if (p['arms'] != ['bf16', *STUDENTS] or p['expected_records'] != 800
            or p['new_endpoint_records'] != 200 or p['reused_endpoint_records'] != 600
            or p['expected_model_queries'] != 205 or p['training_updates'] != 0 or p['new_rollouts'] != 0):
        raise ValueError('Full per-channel panel or inference-only budget differs')
    return p


def binding(p, row, variant):
    return {'protocol': p['protocol'], 'case_id': row['case_id'], 'suite': row['suite'],
            'arm': ARM, 'variant': variant, 'seed_index': 0,
            'sampler_seed': row['sampler_seeds'][0], 'plan_row': row,
            'source_checkpoint': p['channel']['deployment']}


def validate_new(record, p, row, variant):
    if any(record.get(k) != v for k, v in binding(p, row, variant).items()):
        raise ValueError('Per-channel endpoint input/seed/checkpoint differs')
    if record.get('training_updates') != 0 or not record.get('pbs_jobid'):
        raise ValueError('Inference-only PBS record required')
    metrics.endpoint_metrics(record, record)
    if any(r[6] not in (-1, 0, 1) for r in record['commands']):
        raise ValueError('Gripper must use actual -1/0/+1 commands')


def source_records(p, rows):
    old, root = p['source_diagnostic_protocol'], Path(p['source_diagnostic_root'])
    summary = read(root / 'results/summary.json')
    if (not summary['complete'] or summary['records_completed'] != 600
            or summary['pbs_jobid'] != p['source_diagnostic_job']):
        raise RuntimeError('Completed frozen diagnostic required')
    records = []
    for case in p['case_ids']:
        for arm in old['arms']:
            for variant in VARIANTS:
                record = read(root / 'results/records' / arm / f'{case}_{variant}.json')
                source.validate_record(record, old, rows[case], arm, variant)
                records.append(record)
    source.require_panel(records, old)
    return records


def transitions(student, teacher):
    def events(record):
        signs = [int(r[6]) for r in record['commands'][:10]]
        return [{'step': i, 'from': signs[i-1], 'to': signs[i]} for i in range(1, 10) if signs[i] != signs[i-1]]
    a, b = events(student), events(teacher)
    same = [(e['from'], e['to']) for e in a] == [(e['from'], e['to']) for e in b]
    delays = [x['step']-y['step'] for x,y in zip(a,b)] if same else []
    return {'student': a, 'teacher': b, 'direction_sequence_equal': same,
            'ordered_event_delays_when_sequence_equal': delays,
            'missed_open_events': max(0, sum(e['to']==-1 for e in b)-sum(e['to']==-1 for e in a)),
            'extra_open_events': max(0, sum(e['to']==-1 for e in a)-sum(e['to']==-1 for e in b)),
            'missed_close_events': max(0, sum(e['to']==1 for e in b)-sum(e['to']==1 for e in a)),
            'extra_close_events': max(0, sum(e['to']==1 for e in a)-sum(e['to']==1 for e in b))}


def run(p):
    import visited_common as vc
    vc.guard()
    import numpy as np
    import torch
    import runner
    import sampler
    import calibration_source
    import latent_reference
    from metrics_reference import metrics as endpoint_metrics
    from channel_banks import ChannelBanks
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261018)
    root = Path(p['remote_root']); artifact = Path(os.environ['ARTIFACTS'])
    rp = vc.validate(read(Path(p['visited_root'])/'protocol.json'))
    rows = vc.expected_rows(rp)
    if {r['case_id']: r for r in read(Path(p['visited_root'])/'data/plan.json')['inputs']} != rows:
        raise RuntimeError('Visited observation plan changed')
    old = source_records(p, rows)
    indexed = {(r['case_id'], r['arm'], r['variant']): r for r in old}
    src_protocol = read(artifact/'source_channel_protocol.json')
    if read(Path(p['channel']['root'])/'protocol.json') != src_protocol:
        raise RuntimeError('Original per-channel protocol changed')
    receipt = read(Path(p['channel']['results'])/'training_receipt.json')
    if (receipt['pbs_jobid'] != p['channel']['training_job'] or not receipt['complete']
            or receipt['updates'] != 80 or receipt['scaling']['group_size'] != 1
            or receipt['scaling']['channels'] != 1806863
            or any(v != 0 for v in receipt['cached_query_parity'].values())):
        raise RuntimeError('Source final80 per-channel receipt differs')
    state = {'protocol': p['protocol'], 'pbs_jobid': os.environ['PBS_JOBID'], 'complete': False,
             'phase': 'loading', 'records_completed': 0, 'reused_endpoint_records': 600,
             'new_model_queries': 0, 'new_teacher_queries': 0, 'training_updates': 0, 'new_rollouts': 0}
    started = int(os.environ['FW_JOB_START_EPOCH'])
    def deadline():
        if time.time()-started > p['budget']['soft_deadline_seconds']-120:
            raise TimeoutError('Soft deadline; full scope retained without automatic GPU resubmission')
    def progress(**values):
        state.update(values); save(root/'results/progress.json', state)
        print('PC_VISITED_PROGRESS '+json.dumps(state), flush=True)
    runtime = None; banks = None
    try:
        progress()
        runtime = runner.Runtime(root/'results'); runtime.model.eval(); runtime.arm('bf16')
        for param in runtime.model.parameters(): param.requires_grad_(False)
        datums = {}; info = {}; parities = {}
        for case in p['case_ids']:
            deadline()
            obs, meta = vc.load_observation(np, rp, rows[case])
            for variant in VARIANTS:
                perturbed, details = source.perturb_observation(np, obs, variant, p)
                datums[case,variant] = runtime.datum(perturbed)
                details['description'] = meta['description']; info[case,variant] = details
            for variant in VARIANTS:
                for field in ('image', 'proprio'):
                    delta = datums[case,variant][field].float()-datums[case,'base'][field].float()
                    info[case,variant]['model_'+field+'_delta_rms'] = float(delta.square().mean().sqrt())
                    info[case,variant]['model_'+field+'_delta_max_abs'] = float(delta.abs().max())
                info[case,variant]['normalized_proprio_bf16'] = datums[case,variant]['proprio'].float().cpu().tolist()
                if info[case,variant] != indexed[case,'bf16',variant]['input_perturbation']:
                    raise RuntimeError(f'Reconstructed frozen sensor/preprocessing metadata differs: {case}/{variant}')
        save(root/'results/sensor_selfcheck.json', source.sensor_selfcheck(np,p))
        save(root/'results/input_reuse.json', {'complete': True, 'matched_inputs': 200,
              'same_plan_rows_seeds_perturbations': True, 'same_model_input_metadata': True,
              'image_tensor_scope': 'frozen raw observations and preprocessing; old full image tensor not persisted'})
        first = p['case_ids'][0]
        runtime.description = info[first,'base']['description']
        deadline(); fresh,_ = runtime.infer(datums[first,'base'], rows[first]['sampler_seeds'][0], measure=False)
        state['new_model_queries'] += 1
        parities['bf16_visited_action_max_abs'] = float((fresh.float().cpu()-torch.tensor(indexed[first,'bf16','base']['action'])).abs().max())
        if parities['bf16_visited_action_max_abs'] != 0: raise RuntimeError('Fresh BF16 visited replay differs')
        del fresh
        deployment = torch.load(p['channel']['deployment'], map_location='cpu', weights_only=True)
        baseline = torch.load(p['channel']['baseline_deployment'], map_location='cpu', weights_only=True)
        if (deployment['protocol'] != src_protocol or deployment['updates'] != 80
                or deployment['baseline_checkpoint'] != p['channel']['baseline_deployment']
                or baseline['arm'] != 'rho1.5' or baseline['updates'] != 8):
            raise RuntimeError('Deployment does not identify the completed baseline/final80')
        banks = ChannelBanks(runtime, p['bank_directory'], baseline['module_masters'],
                             src_protocol['clipping'], src_protocol['scaling'])
        banks.restore_clips(deployment['ratios']); banks.restore_scales(deployment['log2_scales'])
        for group in (banks.books,banks.logs,banks.ratios,banks.shadows):
            for param in group.values(): param.requires_grad_(False)
        if any(param.requires_grad for param in runtime.model.parameters()): raise RuntimeError('Frozen model required')
        del deployment,baseline
        runtime.current_arm = ARM
        # Reproduce the source final80 endpoint before caching the same BF16 compensation.
        original_plan = Path(src_protocol['calibration']['plan_file'])
        original_rows = {r['case_id']:r for r in read(original_plan)['inputs']}
        obs,meta = calibration_source.load_observation(np, original_plan.parent, original_rows[2000])
        source_datum = runtime.datum(obs); runtime.description = meta['description']
        reference = read(p['channel']['reference_endpoint'])
        teacher = torch.load(Path(p['channel']['teacher_directory'])/'2000_0.pt', map_location='cpu', weights_only=True)
        if teacher['plan_row'] != original_rows[2000] or teacher['sampler_seed'] != reference['sampler_seed']:
            raise RuntimeError('Source endpoint teacher binding differs')
        def source_query():
            deadline(); banks.seen.clear()
            with torch.no_grad():
                action,counts,future = sampler.differentiable_action(runtime,source_datum,reference['sampler_seed'],checkpointed=False,return_future=True)
            state['new_model_queries'] += 1
            if len(banks.seen) != 614 or counts != {'video_steps':10,'action_steps':10}: raise RuntimeError('Source query coverage differs')
            return action.detach().cpu(),future.detach().cpu()
        a,f = source_query()
        observed = endpoint_metrics(torch, latent_reference, a, f, teacher, src_protocol['training'])
        if any(observed[k] != reference[k] for k in observed): raise RuntimeError('Source endpoint metrics do not reproduce exactly')
        rounding = banks.cache_compensated_weights(); b,g = source_query()
        parities['source_uncached_cached_action_max_abs'] = float((a.float()-b.float()).abs().max())
        parities['source_uncached_cached_future_max_abs'] = float((f.float()-g.float()).abs().max())
        deadline(); actual,_ = runtime.infer(source_datum,reference['sampler_seed'],measure=False); state['new_model_queries'] += 1
        parities['source_actual_infer_action_max_abs'] = float((actual.float().cpu()-b.float()).abs().max())
        if any(v != 0 for v in parities.values()): raise RuntimeError('Per-channel source/cache/infer parity differs')
        save(root/'results/source_parity.json', {'passed':True,'parities':parities,'source_metrics':observed,
             'frozen':banks.verify_frozen(torch.load(p['channel']['baseline_deployment'],map_location='cpu',weights_only=True)['module_masters']),
             'scaling':banks.scaling_summary(),'clipping':banks.clip_summary(),'weight_rounding':{k:v for k,v in rounding.items() if k!='per_module'}})
        del a,b,f,g,actual,teacher,source_datum
        torch.cuda.empty_cache(); new = []
        for case in p['case_ids']:
            for variant in VARIANTS:
                deadline(); runtime.description = info[case,variant]['description']; banks.seen.clear()
                before = time.monotonic()
                action,_ = runtime.infer(datums[case,variant],rows[case]['sampler_seeds'][0],measure=False)
                state['new_model_queries'] += 1
                if len(banks.seen) != 614: raise RuntimeError('All614 per-channel Linears required')
                action = action.detach().float().cpu()
                commands = runtime.command(action.clone())
                continuous = runtime.ev._denormalize_action(action.clone(),runtime.processor)[0].copy()
                continuous[:,-1] = -(2*continuous[:,-1]-1)
                if not np.array_equal(np.sign(continuous[:,6]),commands[:,6]): raise RuntimeError('Actual gripper conversion differs')
                record = {**binding(p,rows[case],variant),'pbs_jobid':os.environ['PBS_JOBID'],'training_updates':0,
                    'action':action.tolist(),'commands':commands.tolist(),'continuous_commands':continuous.tolist(),
                    'input_perturbation':info[case,variant],'query_seconds':time.monotonic()-before,'origin':'fresh_per_channel_query'}
                validate_new(record,p,rows[case],variant)
                save(root/'results/records'/ARM/f'{case}_{variant}.json',record); new.append(record)
                progress(phase='evaluating',records_completed=len(new),case_id=case,variant=variant,parities=parities)
        deadline(); runtime.description = info[first,'base']['description']
        repeated,_ = runtime.infer(datums[first,'base'],rows[first]['sampler_seeds'][0],measure=False); state['new_model_queries'] += 1
        parities['same_input_repeat_max_abs'] = float((repeated.float().cpu()-torch.tensor(new[0]['action'])).abs().max())
        if parities['same_input_repeat_max_abs'] != 0 or state['new_model_queries'] != 205: raise RuntimeError('Repeat or full query count differs')
        if runtime.q._native_int4_gemm_calls or runtime.q._kv_active: raise RuntimeError('Unexpected INT4/KV4 execution')
        progress(phase='complete',complete=True,records_completed=200,parities=parities,
                 all_parameters_frozen=True,matched_input_metadata=200)
        save(root/'results/summary.json',state)
    except Exception as exc:
        progress(phase='failed',error=type(exc).__name__+': '+str(exc)); raise
    finally:
        if runtime is not None and runtime.env is not None: runtime.env.close()


def aggregate(p):
    import visited_common as vc
    vc.guard(); root = Path(p['remote_root'])
    rows = vc.expected_rows(read(Path(p['visited_root'])/'protocol.json'))
    summary = read(root/'results/summary.json')
    if not summary['complete'] or summary['records_completed'] != 200 or summary['new_model_queries'] != 205:
        raise RuntimeError('Full GPU endpoint panel required')
    records = source_records(p,rows)
    for case in p['case_ids']:
        for variant in VARIANTS:
            record = read(root/'results/records'/ARM/f'{case}_{variant}.json')
            validate_new(record,p,rows[case],variant); records.append(record)
    indexed = {(r['case_id'],r['arm'],r['variant']):r for r in records}
    if len(indexed) != 800 or len(records) != 800: raise RuntimeError('Exactly800 endpoints required')
    endpoints = []; sensitivities = []
    for case in p['case_ids']:
        for variant in VARIANTS:
            teacher = indexed[case,'bf16',variant]
            for arm in STUDENTS:
                record = indexed[case,arm,variant]
                if any(record[k] != teacher[k] for k in ('plan_row','sampler_seed','input_perturbation')):
                    raise RuntimeError('Paired inputs or seeds differ')
                measured = metrics.endpoint_metrics(record,teacher)
                item = {'case_id':case,'suite':record['suite'],'arm':arm,'variant':variant,
                        'endpoint':measured,'transitions':transitions(record,teacher)}
                endpoints.append(item); save(root/'results/case_metrics'/f'{case}_{arm}_{variant}.json',item)
        for arm in STUDENTS:
            for family in ('image','proprio'):
                samples = [indexed[case,a,v] for a in (arm,'bf16') for v in ('base',family+'_plus',family+'_minus')]
                epsilon = p['perturbations'][family]['epsilon']
                item = {'case_id':case,'suite':samples[0]['suite'],'arm':arm,'family':family,'epsilon':epsilon,
                    'normalized':metrics.sensitivity_metrics(*samples,epsilon),
                    'executed_command':metrics.sensitivity_metrics(*[{**r,'action':r['commands']} for r in samples],epsilon),
                    'geometry':source.geometry(*samples,epsilon)}
                sensitivities.append(item);save(root/'results/sensitivity_cases'/f'{case}_{arm}_{family}.json',item)
    if len(endpoints) != 600 or len(sensitivities) != 240: raise RuntimeError('Endpoint/sensitivity coverage differs')
    def endpoint_group(selected):
        result = metrics._mean_endpoint(selected,'endpoint')
        result['cases'] = len(selected)
        result['mean_absolute_case_bias'] = {field:[sum(abs(r['endpoint']['motor'][field]['per_dimension'][d]['signed_bias']) for r in selected)/len(selected) for d in range(6)] for field in ('normalized_action','executed_command')}
        result['gripper']['mismatch_cases'] = sum(r['endpoint']['gripper']['command_sign_disagreement']['count']>0 for r in selected)
        delays = [v for r in selected for v in r['transitions']['ordered_event_delays_when_sequence_equal']]
        result['transitions'] = {k:sum(r['transitions'][k] for r in selected) for k in ('missed_open_events','extra_open_events','missed_close_events','extra_close_events')}
        result['transitions'].update(direction_sequence_mismatch_cases=sum(not r['transitions']['direction_sequence_equal'] for r in selected),
            matched_event_delay_count=len(delays),mean_abs_event_delay=sum(abs(v) for v in delays)/len(delays) if delays else None,
            max_abs_event_delay=max(map(abs,delays)) if delays else None)
        return result
    def parsed_sensitivity(row):
        # Reuse the frozen numeric validators; the old whole-panel validator only accepts two arm names.
        return {'case_id':str(row['case_id']),'suite':row['suite'],'arm':row['arm'],'family':row['family'],
            'central_normalized':metrics._sensitivity_central(row['normalized'],'normalized'),
            'central_executed':metrics._sensitivity_central(row['executed_command'],'executed_command'),
            'geometry':metrics._sensitivity_geometry(row),
            'sides':{side:{'normalized_error':metrics._sensitivity_side(row['normalized'],'normalized',side),
                          'executed_error':metrics._sensitivity_side(row['executed_command'],'executed_command',side),
                          'gripper':metrics._sensitivity_gripper(row,side)} for side in ('plus','minus')}}
    parsed = [parsed_sensitivity(r) for r in sensitivities]
    def paired_counts(pairs):
        valid = [(a,b) for a,b in pairs if a is not None and b is not None]
        tied = sum(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-15) for a,b in valid)
        return {'defined_cases':len(valid), 'per_channel_lower':sum(a<b and not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-15) for a,b in valid),
                'per_channel_higher':sum(a>b and not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-15) for a,b in valid),'tied':tied}
    def by_scope(suite=None):
        e = [r for r in endpoints if suite is None or r['suite']==suite]
        s = [r for r in parsed if suite is None or r['suite']==suite]
        bases = {(r['case_id'],r['arm']):r for r in e if r['variant']=='base'}
        responses = {(r['case_id'],r['arm'],r['family']):r for r in s}
        cases = sorted({r['case_id'] for r in e})
        return {'endpoint':{variant:{arm:endpoint_group([r for r in e if r['variant']==variant and r['arm']==arm]) for arm in STUDENTS} for variant in VARIANTS},
                'sensitivity':{family:{arm:metrics._summarize_sensitivity_arm([r for r in s if r['family']==family and r['arm']==arm]) for arm in STUDENTS} for family in ('image','proprio')},
                'paired_base_motor':{ref:{field:paired_counts([(bases[c,ARM]['endpoint']['motor'][field]['overall_rmse'],bases[c,ref]['endpoint']['motor'][field]['overall_rmse']) for c in cases]) for field in ('normalized_action','executed_command')} for ref in ('clip_only','joint')},
                'paired_response_error':{family:{ref:paired_counts([(responses[str(c),ARM,family]['geometry']['error_rms'],responses[str(c),ref,family]['geometry']['error_rms']) for c in cases]) for ref in ('clip_only','joint')} for family in ('image','proprio')}}
    overview = {'protocol':p['protocol'],'complete':True,'pbs_jobid':os.environ['PBS_JOBID'],
        'source_gpu_job':summary['pbs_jobid'],'records':800,'new_endpoint_records':200,'reused_endpoint_records':600,
        'sensitivity_rows':240,'case_count':40,'training_updates':0,'new_rollouts':0,'new_teacher_queries':0,
        'new_model_queries':205,'overall':by_scope(),'by_suite':{suite:by_scope(suite) for suite in source.SUITES}}
    save(root/'results/comparison.json',overview)
    details = {'complete':True,'case_count':40,'base_cases':[r for r in endpoints if r['variant']=='base'],
        'wrong_gripper_commands':{arm:[{'case_id':case,'step':step,'teacher_command':indexed[case,'bf16','base']['commands'][step][6],
             'student_command':indexed[case,arm,'base']['commands'][step][6],
             'teacher_margin':abs(indexed[case,'bf16','base']['continuous_commands'][step][6])}
             for case in p['case_ids'] for step in range(10) if indexed[case,arm,'base']['commands'][step][6] != indexed[case,'bf16','base']['commands'][step][6]] for arm in STUDENTS}}
    save(root/'results/diagnostic_details.json',details)
    # Each report piece is bounded; the complete four-arm report exceeds one small control file.
    compact_details = {**details, 'base_cases':[{'case_id':r['case_id'],'suite':r['suite'],'arm':r['arm'],
        'motor_rmse':r['endpoint']['motor']['normalized_action']['overall_rmse'],
        'command_motor_rmse':r['endpoint']['motor']['executed_command']['overall_rmse'],
        'normalized_dimension_rmse':[d['rmse'] for d in r['endpoint']['motor']['normalized_action']['per_dimension']],
        'normalized_bias':[d['signed_bias'] for d in r['endpoint']['motor']['normalized_action']['per_dimension']],
        'command_bias':[d['signed_bias'] for d in r['endpoint']['motor']['executed_command']['per_dimension']],
        'gripper':r['endpoint']['gripper'],'transitions':r['transitions']} for r in details['base_cases']]}
    pieces = {'comparison.index': {k:v for k,v in overview.items() if k not in ('overall','by_suite')},
              'comparison.overall': overview['overall'],
              'diagnostic_details.index': {k:v for k,v in compact_details.items() if k!='base_cases'}}
    pieces['comparison.index']['suite_names'] = list(source.SUITES)
    for suite in source.SUITES:
        pieces['comparison.'+suite] = overview['by_suite'][suite]
        pieces['diagnostic_details.'+suite] = [r for r in compact_details['base_cases'] if r['suite']==suite]
    for name,value in pieces.items():
        path = root/'results'/f'{name}.compact.json'
        path.write_text(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n')
        if path.stat().st_size > 262144: raise RuntimeError('Report piece exceeds bounded pull size: '+name)
    save(root/'results/aggregate_summary.json',{'complete':True,'pbs_jobid':os.environ['PBS_JOBID'],'records':800,'sensitivity_rows':240,'new_model_queries':0})
    print('PC_VISITED_AGGREGATE_COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--unit',choices=('run','aggregate'),required=True)
    parser.add_argument('--protocol',required=True);args=parser.parse_args()
    p=validate_protocol(read(args.protocol))
    return run(p) if args.unit=='run' else aggregate(p)


if __name__=='__main__':main()
