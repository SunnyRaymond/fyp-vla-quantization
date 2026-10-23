"""CPU-only aggregation reusing the frozen endpoint, event and sensitivity metrics."""
import copy
import json
import math
import os
from pathlib import Path
import metrics
import c80_reference as c80
import visited_reference as vref
from worker import validate_record

source=vref.source
read,save=vref.read,vref.save
transitions=vref.transitions
ARM='per_channel_gripper'
STUDENTS=('clip_only','joint','per_channel_clip',ARM)
VARIANTS=vref.VARIANTS


def validate_new(record,p,row,variant):
    expected={'protocol':p['protocol'],'case_id':row['case_id'],'suite':row['suite'],'arm':ARM,
        'variant':variant,'seed_index':0,'sampler_seed':row['sampler_seeds'][0],'plan_row':row,
        'source_checkpoint':str(Path(p['remote_root'])/'results/train_eval/deployment.pt'),
        'training_updates':80,'lambda_g':1/6}
    if any(record.get(k)!=v for k,v in expected.items()) or not record.get('pbs_jobid'):
        raise RuntimeError('Gripper visited input/seed/checkpoint differs')
    metrics.endpoint_metrics(record,record)


def source_records(p,rows):
    old=p['experiment']['visited_source_protocol']
    records=vref.source_records(old,rows)
    root=Path(p['experiment']['visited_source_root'])
    for case in old['case_ids']:
        for variant in VARIANTS:
            record=read(root/'results/records/per_channel_clip'/f'{case}_{variant}.json')
            vref.validate_new(record,old,rows[case],variant);records.append(record)
    if len(records)!=800:raise RuntimeError('All800 reference endpoints required')
    return records


def endpoint_group(rows):
    result=metrics._mean_endpoint(rows,'endpoint');result['queries']=len(rows)
    result['cases']=len({r['case_id'] for r in rows})
    result['mean_absolute_query_bias']={field:[sum(abs(r['endpoint']['motor'][field]['per_dimension'][d]['signed_bias']) for r in rows)/len(rows) for d in range(6)] for field in ('normalized_action','executed_command')}
    result['gripper']['mismatch_queries']=sum(r['endpoint']['gripper']['command_sign_disagreement']['count']>0 for r in rows)
    result['transitions']={key:sum(r['transitions'][key] for r in rows) for key in ('missed_open_events','extra_open_events','missed_close_events','extra_close_events')}
    result['transitions']['direction_sequence_mismatch_queries']=sum(not r['transitions']['direction_sequence_equal'] for r in rows)
    return result


def aggregate_original(p):
    root=Path(p['remote_root']);plan=read(p['calibration']['plan_file']);rows={r['case_id']:r for r in plan['inputs']}
    source_protocol=read(Path(os.environ['ARTIFACTS'])/'source_per_channel_protocol.json')
    receipt=read(root/'results/train_eval/training_receipt.json')
    checks=('codebooks_unchanged','indices_unchanged','row_gains_unchanged','original_exponents_unchanged','continuous_scale_metadata_separate','gradient_coupling_complete','input_absmax_chain','loss_only_change')
    if receipt['updates']!=80 or receipt['lambda_g']!=1/6 or not all(receipt[k] is True for k in checks):raise RuntimeError('Frozen single-factor update80 training required')
    if [h['case_id'] for h in receipt['history']]!=p['calibration']['update_order'] or any(h['lambda_g']!=1/6 for h in receipt['history']):raise RuntimeError('Training order/lambda differs')
    if receipt['initial_parity_max_abs']!=0 or receipt['actual_infer_parity_max_abs']!=0 or any(v!=0 for v in receipt['cached_query_parity'].values()):raise RuntimeError('Deployment parity differs')
    result={'complete':True,'shared_teacher_targets_reused':True,'splits':{}}
    for split in ('calibration','heldout'):
        stats={};records_by_arm={};executed=[]
        for arm in p['arms']:
            origin=root if arm==ARM else Path(p['source_per_channel_root']) if arm=='per_channel_clip' else Path(p['control_root'])
            rr=[read(origin/'results'/split/arm/'queries'/f'{case}_{seed}.json') for case,seed in c80.queries(p,split)]
            for record in rr:
                row=rows[record['case_id']];seed=record['seed_index']
                if arm in (ARM,'per_channel_clip'):validate_record(record,p if arm==ARM else source_protocol,row,split,seed)
                else:c80.validate_binding(record,p['source_control_protocol'],row,split,arm,seed)
                if arm==ARM:
                    executed.append({'case_id':row['case_id'],'suite':row['suite'],'seed_index':seed,
                        'endpoint':metrics.endpoint_metrics(record,record['teacher']),
                        'transitions':transitions(record,record['teacher'])})
            stats[arm]=c80.aggregate_records(rr,p,split);records_by_arm[arm]=rr
            saved=read(origin/'results'/split/arm/'summary.json')
            if not saved['complete'] or any(saved[k]!=v for k,v in stats[arm].items()):raise RuntimeError('Original endpoint summary differs')
        delta={control:{metric:100*(stats[ARM]['overall'][metric]/stats[control]['overall'][metric]-1) for metric in c80.METRICS} for control in ('baseline','clip_only','per_channel_clip')}
        paired={}
        for control in ('baseline','clip_only','per_channel_clip'):
            paired[control]={}
            for metric in ('first10motor_rmse','first10gripper_rmse','future_latent_nmse'):
                means={arm:{case:sum(r[metric] for r in rr if r['case_id']==case)/sum(r['case_id']==case for r in rr) for case in sorted({r['case_id'] for r in rr})} for arm,rr in records_by_arm.items() if arm in (ARM,control)}
                diffs=[means[ARM][case]-v for case,v in means[control].items()]
                paired[control][metric]={'cases':len(diffs),'improved':sum(v<0 for v in diffs),'worsened':sum(v>0 for v in diffs),'tied':sum(v==0 for v in diffs)}
        result['splits'][split]={'arms':stats,'percent_change':delta,'paired_after_seed_average':paired,
            'new_arm_executed':{'overall':endpoint_group(executed),'by_suite':{suite:endpoint_group([r for r in executed if r['suite']==suite]) for suite in source.SUITES}},
            'executed_control_scope':'old original80 controls retain scalar motor/gripper/latent metrics only; actual commands compared across all arms on visited panel'}
        for suite in source.SUITES:
            save(root/'results'/f'{split}_details.{suite}.compact.json',[r for r in executed if r['suite']==suite])
    return result


def aggregate_visited(p):
    import visited_common as vc
    vc.guard(); root = Path(p['remote_root'])
    rows = vc.expected_rows(read(Path(p['visited_root'])/'protocol.json'))
    summary = read(root/'results/train_eval/summary.json')
    if not summary['complete'] or summary['updates'] != 80 or summary['total_model_queries'] != 445:
        raise RuntimeError('Full GPU endpoint panel required')
    records = source_records(p,rows)
    for case in p['case_ids']:
        for variant in VARIANTS:
            record = read(root/'results/records'/ARM/f'{case}_{variant}.json')
            validate_new(record,p,rows[case],variant); records.append(record)
    indexed = {(r['case_id'],r['arm'],r['variant']):r for r in records}
    if len(indexed) != 1000 or len(records) != 1000: raise RuntimeError('Exactly1000 endpoints required')
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
    if len(endpoints) != 800 or len(sensitivities) != 320: raise RuntimeError('Endpoint/sensitivity coverage differs')
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
        return {'defined_cases':len(valid), 'gripper_lower':sum(a<b and not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-15) for a,b in valid),
                'gripper_higher':sum(a>b and not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-15) for a,b in valid),'tied':tied}
    def by_scope(suite=None):
        e = [r for r in endpoints if suite is None or r['suite']==suite]
        s = [r for r in parsed if suite is None or r['suite']==suite]
        bases = {(r['case_id'],r['arm']):r for r in e if r['variant']=='base'}
        responses = {(r['case_id'],r['arm'],r['family']):r for r in s}
        cases = sorted({r['case_id'] for r in e})
        return {'endpoint':{variant:{arm:endpoint_group([r for r in e if r['variant']==variant and r['arm']==arm]) for arm in STUDENTS} for variant in VARIANTS},
                'sensitivity':{family:{arm:metrics._summarize_sensitivity_arm([r for r in s if r['family']==family and r['arm']==arm]) for arm in STUDENTS} for family in ('image','proprio')},
                'paired_base_motor':{ref:{field:paired_counts([(bases[c,ARM]['endpoint']['motor'][field]['overall_rmse'],bases[c,ref]['endpoint']['motor'][field]['overall_rmse']) for c in cases]) for field in ('normalized_action','executed_command')} for ref in ('clip_only','joint','per_channel_clip')},
                'paired_response_error':{family:{ref:paired_counts([(responses[str(c),ARM,family]['geometry']['error_rms'],responses[str(c),ref,family]['geometry']['error_rms']) for c in cases]) for ref in ('clip_only','joint','per_channel_clip')} for family in ('image','proprio')}}
    overview = {'protocol':p['protocol'],'complete':True,'pbs_jobid':os.environ['PBS_JOBID'],
        'source_gpu_job':summary['pbs_jobid'],'records':1000,'new_endpoint_records':200,'reused_endpoint_records':800,
        'sensitivity_rows':320,'case_count':40,'training_updates':80,'new_rollouts':0,'new_teacher_queries':0,
        'new_model_queries':445,'overall':by_scope(),'by_suite':{suite:by_scope(suite) for suite in source.SUITES}}
    overview['original']=aggregate_original(p['experiment'])
    overview['scope']='visited40x5 and original heldout80 are separate panels; existing teacher targets reused'
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
    save(root/'results/aggregate_summary.json',{'complete':True,'pbs_jobid':os.environ['PBS_JOBID'],'records':1000,'sensitivity_rows':320,'original_records':640,'original_new_records':160,'original_reused_records':480,'training_updates':80,'new_model_queries':0})
    print('PC_VISITED_AGGREGATE_COMPLETE',flush=True)


def aggregate(p):
    vp=copy.deepcopy(p['visited_source_protocol'])
    vp.update(protocol=p['protocol'],remote_root=p['remote_root'],experiment=p)
    aggregate_visited(vp)
