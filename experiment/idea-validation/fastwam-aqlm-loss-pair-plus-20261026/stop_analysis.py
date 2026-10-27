"""Read-only cached-evidence analysis on a verified CPU PBS allocation."""
from __future__ import annotations

import collections
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def distribution(values):
    values=sorted(float(v) for v in values)
    return dict(n=len(values),minimum=values[0] if values else None,
                median=statistics.median(values) if values else None,
                p90=values[math.ceil(.9*len(values))-1] if values else None,
                maximum=values[-1] if values else None)


def main():
    import runner
    runner.guard()
    import torch
    torch.set_num_threads(4)
    import aggregate
    import evaluate

    root=Path(os.environ['NEW_ROOT'])
    out=Path(sys.argv[1])
    protocol=read(root/'protocol.json')
    evaluate.validate_protocol(protocol)
    fixed=Path(protocol['sources']['parent_root'])
    source_protocol=read(protocol['sources']['parent_protocol_file'])
    _,rows=evaluate.load_manifest(protocol['cohort']['reference_manifest'])
    working=out/'working'
    tail_recovery={}
    for worker in range(4):
        src=root/'results/workers'/f'worker_{worker}'
        dst=working/'workers'/f'worker_{worker}'
        dst.mkdir(parents=True,exist_ok=True)
        for name in ('setup_receipt.json','protocol.json','summary.json','progress.json','failure.json','episodes.jsonl'):
            if (src/name).exists():shutil.copy2(src/name,dst/name)
        tail_recovery[str(worker)]={'source_bytes':(src/'episodes.jsonl').stat().st_size}
    arms,workers=aggregate.load_workers(working,rows,protocol)
    for worker in range(4):
        tail_recovery[str(worker)]['validated_copy_bytes']=(working/'workers'/f'worker_{worker}'/'episodes.jsonl').stat().st_size
    common=sorted(set(arms['reconstruction'])&set(arms['ours']))
    closed=dict(controlled_user_stop=True,full_cohort_per_arm=280,
        reconstruction_full=aggregate.rate(arms['reconstruction'],range(280)),
        ours_observed=aggregate.rate(arms['ours'],range(280)),
        matched_episodes=len(common),
        matched_rates={arm:aggregate.rate(values,common) for arm,values in arms.items()},
        ours_vs_reconstruction=aggregate.paired(arms['ours'],arms['reconstruction'],common),
        by_suite={suite:dict(indices=len([i for i in common if rows[i]['suite']==suite]),
            rates={arm:aggregate.rate(values,[i for i in common if rows[i]['suite']==suite])
                   for arm,values in arms.items()},
            paired=aggregate.paired(arms['ours'],arms['reconstruction'],
                                    [i for i in common if rows[i]['suite']==suite]))
            for suite in sorted({r['suite'] for r in rows})},
        workers=workers,tail_recovery=tail_recovery,
        source_episode_pbs_jobs=sorted({r['pbs_job_id'] for values in arms.values() for r in values.values()}))
    save(out/'closedloop_stopped.json',closed)

    initial={}
    initial_kl={}
    case_ids=list(range(2000,2080))
    for case in case_ids:
        cached=torch.load(fixed/f'results/reader_preparation/inputs/{case}.pt',map_location='cpu',weights_only=True)
        initial[case]=cached['metrics']
        initial_kl[case]=read(fixed/f'results/reader_preparation/cases/{case}.json')['reader_kl']
    metric_keys=tuple(initial[case_ids[0]])
    initial_mean={key:statistics.mean(initial[c][key] for c in case_ids) for key in metric_keys}
    calibration={'initial_shared':initial_mean,'arms':{}}
    for arm in protocol['arms']:
        records=[read(fixed/f'results/arms/{arm}/evaluation/calibration/{c}_0.json')['metrics'] for c in case_ids]
        final={key:statistics.mean(row[key] for row in records) for key in metric_keys}
        calibration['arms'][arm]=dict(final=final,
            percent_vs_initial={key:100*(final[key]-initial_mean[key])/initial_mean[key]
                                if initial_mean[key] else None for key in metric_keys},
            contexts_worse_vs_initial={key:sum(row[key]>initial[c][key]+1e-12 for c,row in zip(case_ids,records))
                                       for key in metric_keys})
    ours_history=read(fixed/'results/arms/ours/training.json')['history']
    beta=read(fixed/'results/reader_preparation/summary.json')['beta']
    drift=[]
    for update in ours_history:
        cases=update['cases']
        base={key:statistics.mean(initial[c][key] for c in cases) for key in metric_keys}
        kl0=statistics.mean(initial_kl[c] for c in cases)
        before=update['before']
        total0=base['mixed_loss']+beta*kl0
        total=before['mixed_loss']+beta*update['reader_kl']
        drift.append(dict(update=update['update'],cases=cases,initial_same_cases=base,
            before_update=before,initial_reader_kl=kl0,before_reader_kl=update['reader_kl'],
            initial_total=total0,before_total=total,total_ratio=total/total0,
            mixed_ratio=before['mixed_loss']/base['mixed_loss'],
            motor_mse_ratio=before['first10motor_mse']/base['first10motor_mse'],
            gradient_norm=update['gradient_norm']))
    save(out/'calibration_drift.json',dict(calibration=calibration,ours_same_case_drift=drift,beta=beta))

    checkpoints={}
    deployments={}
    optimizer_steps={}
    for arm in protocol['arms']:
        checkpoint=torch.load(fixed/f'results/arms/{arm}/checkpoint.pt',map_location='cpu',weights_only=False)
        deployment=torch.load(fixed/f'results/arms/{arm}/deployment.pt',map_location='cpu',weights_only=True)
        if checkpoint['protocol']!=source_protocol or checkpoint['arm']!=arm or checkpoint['state']['updates']!=8:
            raise ValueError('Final checkpoint binding differs')
        if deployment['protocol']!=source_protocol or deployment['arm']!=arm or deployment['updates']!=8:
            raise ValueError('Final deployment binding differs')
        checkpoints[arm]=checkpoint['parameters']
        deployments[arm]=deployment['parameters']
        steps=[float(value['step']) for value in checkpoint['optimizer']['state'].values() if 'step' in value]
        optimizer_steps[arm]=dict(values=sorted(set(steps)),states=len(steps),updates=checkpoint['state']['updates'],
                                  exposures=checkpoint['state']['exposures'],lineage=checkpoint['state']['lineage'])
        del checkpoint,deployment
    totals={arm:{kind:collections.defaultdict(float) for kind in ('books','scales')} for arm in protocol['arms']}
    layers=[]
    for index in range(614):
        saved=torch.load(fixed/f'results/banks/{index:04d}.pt',map_location='cpu',weights_only=False)
        name=saved['module'];q=saved['quantized_weight']
        for kind,original in (('books',q.codebooks.detach().float()),('scales',q.scales.detach().float())):
            initial2=float(original.square().sum())
            for arm in protocol['arms']:
                master=checkpoints[arm][name][kind].float()
                stored=deployments[arm][name][kind].float()
                if original.shape!=master.shape or original.shape!=stored.shape:
                    raise ValueError('Parameter shape differs')
                delta2=float((master-original).square().sum())
                stored2=float((stored-original).square().sum())
                rounded2=float((stored-master).square().sum())
                n=original.numel();changed=int((stored!=original).sum())
                row=dict(module=name,arm=arm,kind=kind,elements=n,initial_rms=math.sqrt(initial2/n),
                    delta_rms=math.sqrt(delta2/n),relative_l2=math.sqrt(delta2/initial2) if initial2 else None,
                    stored_relative_l2=math.sqrt(stored2/initial2) if initial2 else None,
                    deployed_changed_fraction=changed/n,
                    sign_flips=int(((original>0)&(stored<0) | (original<0)&(stored>0)).sum()))
                if kind=='scales':
                    relative=((master-original).abs()/original.abs().clamp_min(1e-12)).reshape(-1)
                    row.update(relative_abs_median=float(relative.median()),
                               relative_abs_p90=float(torch.quantile(relative,.9)),
                               relative_abs_max=float(relative.max()),
                               fraction_change_gt25pct=float((relative>.25).float().mean()),
                               fraction_change_gt50pct=float((relative>.5).float().mean()))
                layers.append(row)
                total=totals[arm][kind]
                for key,value in dict(initial2=initial2,delta2=delta2,stored2=stored2,
                                      rounded2=rounded2,elements=n,changed=changed,sign_flips=row['sign_flips']).items():
                    total[key]+=value
        del saved,q
    parameters={}
    for arm in protocol['arms']:
        parameters[arm]={}
        for kind,total in totals[arm].items():
            selected=[r for r in layers if r['arm']==arm and r['kind']==kind]
            parameters[arm][kind]=dict(elements=int(total['elements']),
                initial_rms=math.sqrt(total['initial2']/total['elements']),
                delta_rms=math.sqrt(total['delta2']/total['elements']),
                relative_l2=math.sqrt(total['delta2']/total['initial2']),
                stored_relative_l2=math.sqrt(total['stored2']/total['initial2']),
                master_to_deployed_rounding_rms=math.sqrt(total['rounded2']/total['elements']),
                deployed_changed_fraction=total['changed']/total['elements'],
                sign_flips=int(total['sign_flips']),
                layer_relative_l2=distribution(r['relative_l2'] for r in selected),
                top_relative_layers=sorted(selected,key=lambda r:r['relative_l2'],reverse=True)[:8])
    save(out/'parameter_layers.json',layers)
    save(out/'parameter_summary.json',dict(optimizer_steps=optimizer_steps,parameters=parameters))
    result=dict(status='analysis_complete',complete=True,checked_utc=datetime.now(timezone.utc).isoformat(),
        analysis_pbs_job=os.environ['PBS_JOBID'],stopped_episode_job='26010239.pbs101',
        new_training_updates=0,new_model_forwards=0,closed_loop_full_goal_complete=False,
        closed_loop=closed,calibration=calibration,ours_same_case_drift=drift,beta=beta,
        parameter_summary=parameters,optimizer_steps=optimizer_steps)
    encoded=json.dumps(result,allow_nan=False).encode()
    if len(encoded)>256*1024:raise ValueError('Analysis compact summary exceeds limit')
    save(out/'summary.json',result)
    print(json.dumps(dict(status='analysis_complete',counts={arm:len(v) for arm,v in arms.items()},
                          matched=len(common),new_model_forwards=0)),flush=True)


if __name__=='__main__':
    main()
