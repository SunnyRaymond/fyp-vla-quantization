"""Prepare independent demonstration development observations on PBS CPU only."""
from collections import Counter
import json
import os
from pathlib import Path
import random
import time

import render_reference as render


def freeze_development(source, protocol, demo_catalogs):
    rng=random.Random(protocol['development']['selection_seed'])
    excluded={(r['suite'],r['task_id'],r['demo_id']) for r in source['inputs']}
    rows=[]
    for suite_index,suite in enumerate(render.SUITES):
        task_order=list(range(10));rng.shuffle(task_order)
        phases=dict(zip(task_order,protocol['development']['phase_order']))
        for task in range(10):
            base=next(r for r in source['inputs'] if r['suite']==suite and r['task_id']==task)
            demos=demo_catalogs[suite,task]
            candidates=sorted(d for d,n in demos.items() if n>=3 and (suite,task,d) not in excluded)
            if not candidates:raise RuntimeError('No disjoint development demonstration')
            demo=rng.choice(candidates);case=4000+suite_index*10+task
            row={**base,'case_id':case,'phase':phases[task],'demo_id':demo,'demo_length':demos[demo],
                 'frame_index':render.phase_frame(demos[demo],phases[task],source['phase_positions']),
                 'sampler_seeds':[protocol['development']['sampler_seed_base']+case*2+i for i in (0,1)],
                 'observation_file':f'inputs/case_{case}.npz','metadata_file':f'inputs/case_{case}.json',
                 'environment_seed':830000000+case*17}
            rows.append(row)
    validate_development(rows,source,protocol)
    return rows


def validate_development(rows,source,protocol):
    excluded={(r['suite'],r['task_id'],r['demo_id']) for r in source['inputs']}
    if [r['case_id'] for r in rows]!=protocol['development']['case_ids']:
        raise RuntimeError('The exact40 development contexts are required')
    if len({(r['suite'],r['task_id']) for r in rows})!=40:
        raise RuntimeError('Development must cover all40 tasks')
    if any((r['suite'],r['task_id'],r['demo_id']) in excluded for r in rows):
        raise RuntimeError('Development demonstration leakage')
    for suite in render.SUITES:
        ss=[r for r in rows if r['suite']==suite]
        if Counter(r['phase'] for r in ss)!=Counter({'early':3,'middle':4,'late':3}):
            raise RuntimeError('Development phase balance differs')


def main():
    render.allocation_guard()
    root=Path(os.environ['ROOT'])
    protocol=json.loads((root/'protocol.json').read_text())
    old_data=Path(protocol['calibration_root'])/'data'
    source=json.loads((old_data/'plan.json').read_text())
    if source['status']!='complete' or source['calibration_cohorts']['c80']!=protocol['calibration']['case_ids']:
        raise RuntimeError('Frozen C80 source inputs are incomplete')
    out=root/'data';out.mkdir(parents=True,exist_ok=True)
    status=root/'results/data_prepare';status.mkdir(parents=True,exist_ok=True)
    receipt=json.loads((old_data/'source_receipt.json').read_text())
    if receipt['revision']!=source['source_hf_revision']:
        raise RuntimeError('Pinned dataset receipt differs')
    render.write_json_atomic(out/'source_receipt.json',receipt)
    plan_path=out/'plan.json'
    if plan_path.exists():
        plan=json.loads(plan_path.read_text())
        if plan['protocol']!=protocol['protocol']:raise RuntimeError('Existing development plan differs')
        dev=[r for r in plan['inputs'] if r['case_id'] in protocol['development']['case_ids']]
    else:
        catalogs={}
        for suite in render.SUITES:
            for task in range(10):
                row=next(r for r in source['inputs'] if r['suite']==suite and r['task_id']==task)
                with render.open_hdf5(row,out,receipt) as h5:
                    catalogs[suite,task]={int(name.removeprefix('demo_')):int(group['states'].shape[0])
                        for name,group in h5['data'].items() if name.startswith('demo_')}
        dev=freeze_development(source,protocol,catalogs)
        cal=[r for r in source['inputs'] if r['case_id'] in protocol['calibration']['case_ids']]
        plan={'protocol':protocol['protocol'],'status':'preparing','selection_seed':protocol['development']['selection_seed'],
              'inputs':cal+dev,'calibration_case_ids':protocol['calibration']['case_ids'],
              'development_case_ids':protocol['development']['case_ids'],
              'calibration_input_root':str(old_data),'development_input_root':str(out),
              'excluded_previous_inputs':[{k:r[k] for k in ('case_id','suite','task_id','demo_id')} for r in source['inputs']],
              'source_hf_revision':source['source_hf_revision'],'new_episodes':0,'predicted_actions_executed':0}
        render.write_json_atomic(plan_path,plan)
    validate_development(dev,source,protocol)
    fastwam,libero,compat,config,_=render.load_roots(protocol)
    np,benchmark,ev,provenance=render.prepare_environment(fastwam,libero,compat,config)
    suites={name:benchmark.get_benchmark_dict()[name]() for name in render.SUITES}
    started=float(os.environ['FW_JOB_START_EPOCH'])
    completed=[]
    for row in dev:
        obs,meta=out/row['observation_file'],out/row['metadata_file']
        valid=obs.is_file() and meta.is_file() and json.loads(meta.read_text()).get('input_identity')==render.row_identity(row)
        if not valid:
            if time.time()-started>6000:
                render.write_json_atomic(status/'progress.json',{'phase':'checkpointed','complete':False,'completed':completed,'pbs_jobid':os.environ['PBS_JOBID']})
                return 75
            render.render_one(row,out,(np,ev),suites,provenance,fastwam,receipt)
        completed.append(row['case_id'])
        render.write_json_atomic(status/'progress.json',{'phase':'rendering','complete':False,'completed':completed,'pbs_jobid':os.environ['PBS_JOBID']})
        print('DEVELOPMENT_RENDERED '+str(row['case_id']),flush=True)
    plan['status']='complete';render.write_json_atomic(plan_path,plan)
    summary={'protocol':protocol['protocol'],'complete':True,'pbs_jobid':os.environ['PBS_JOBID'],
             'development_contexts':40,'calibration_contexts':80,'distinct_development_tasks':40,
             'demonstration_disjoint_from_calibration_and_previous_test':True,
             'phase_counts_per_suite':{'early':3,'middle':4,'late':3},'new_episodes':0,'predicted_actions_executed':0}
    render.write_json_atomic(status/'summary.json',summary)
    print('DEVELOPMENT_PREPARATION_COMPLETE',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
