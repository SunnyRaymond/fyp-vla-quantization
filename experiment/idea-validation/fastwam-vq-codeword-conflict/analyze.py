"""CPU-only paired diagnostic summaries and calibration-only intervention selection."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
from common import PROTOCOL, ARMS, guard, save, stem, units


def load_units(torch,root,p,phase):
    results=[]
    for shard in range(4):
        directory=root/'results'/f'{phase}-{shard}'
        summary=json.loads((directory/'summary.json').read_text())
        expected=units(p,shard)
        if phase=='exact': expected=[v for v in expected if v[0]=='action_only']
        if summary['status']!='complete' or summary['protocol']!=PROTOCOL or summary['context_count']!=len(expected):
            raise RuntimeError('Cannot summarize an incomplete gradient shard')
        for arm,case in expected:
            r=torch.load(directory/f'{stem(arm,case)}.pt',map_location='cpu',weights_only=True,mmap=True)
            if r['protocol']!=PROTOCOL or r['phase']!=phase or r['arm']!=arm or r['case_id']!=case or r['seed_index']!=0:
                raise RuntimeError('Gradient record identity mismatch')
            results.append(r)
    return results


def weighted(torch,values,weights,valid):
    w=weights*valid
    return float((values*w).sum()/w.sum().clamp_min(1))


def analyze_capture(torch,root,p):
    from stats import FIELDS
    rows=load_units(torch,root,p,'capture')
    if len(rows)!=24: raise RuntimeError('Expected24 checkpoint-context gradient records')
    names=rows[0]['module_names']
    if len(names)!=614 or any(r['module_names']!=names or tuple(r['fields'])!=FIELDS for r in rows):
        raise RuntimeError('Gradient table module/schema coverage differs')
    f={name:i for i,name in enumerate(FIELDS)}
    cases=[]
    for r in rows:
        t=r['tables'].double()
        pop=t[...,f['population']]
        valid=(t[...,f['latent_active_count']]>1)&(pop>1)
        both=(t[...,f['exact_action_gradient_sq']]>0)&(t[...,f['exact_latent_gradient_sq']]>0)
        values={k:weighted(torch,t[...,f[k]],pop,valid) for k in
                ('latent_cancellation','latent_mean_pair_cosine','latent_shared_harm_fraction',
                 'joint_shared_harm_fraction','random_group_cancellation','random_group_pair_cosine')}
        values['occupied_codewords_with_active_latent_sample']=int(valid.sum())
        values['action_latent_conflicting_codeword_fraction']=float(((t[...,f['exact_action_latent_cosine']]<0)&both).sum()/both.sum().clamp_min(1))
        values['exact_action_step_latent_derivative']=float(t[...,f['exact_action_step_latent_derivative']].sum())
        values['exact_joint_step_latent_derivative']=float(t[...,f['exact_joint_step_latent_derivative']].sum())
        cases.append(dict(arm=r['arm'],case_id=r['case_id'],action_mse=r['action_mse'],future_nmse=r['future_nmse'],**values))
    aggregate={a:{k:statistics.mean(r[k] for r in cases if r['arm']==a) for k in cases[0] if k not in ('arm','case_id')}
               for a in ARMS}
    for arm in ARMS:
        subset=[r for r in rows if r['arm']==arm]
        az=torch.stack([r['exact_action'].double() for r in subset]).mean(0)
        zz=torch.stack([r['exact_latent'].double() for r in subset])
        z=zz.mean(0)
        both=(az.square().sum(-1)>0)&(z.square().sum(-1)>0)
        dot=(az*z).sum(-1)
        aggregate[arm]['pooled_cal_action_latent_conflicting_codeword_fraction']=float(((dot<0)&both).sum()/both.sum().clamp_min(1))
        aggregate[arm]['pooled_cal_action_step_latent_derivative']=-float(dot.sum())
        denominator=zz.norm(dim=-1).sum(0)
        between=1-zz.sum(0).norm(dim=-1)/denominator.clamp_min(1e-30)
        active=denominator>0
        aggregate[arm]['between_context_codeword_cancellation']=float(between[active].mean())
    # Selection uses action-only calibration only. It never consults test rows.
    arows=[r for r in rows if r['arm']=='action_only']
    mean=torch.stack([r['tables'].double() for r in arows]).mean(0)
    pop=mean[:,0,:,f['population']]
    energy=mean[:,0,:,f['sample_latent_gradient_sq']]/mean[:,0,:,f['sample_count']].clamp_min(1)*pop
    score=energy*mean[:,0,:,f['latent_cancellation']].square()
    score=torch.where((pop>1)&(mean[:,0,:,f['latent_active_count']]>1),score,0)
    topvals,topids=score.topk(8,dim=1)
    ranking=topvals.sum(1)
    selected_indices=ranking.argsort(descending=True)[:4].tolist()
    if len(selected_indices)!=4 or any(float(ranking[i])<=0 for i in selected_indices):
        raise RuntimeError('Fewer than4 modules show measurable latent sharing constraints; inspect before further work')
    selected=[dict(module=names[i],module_index=i,book=0,codewords=topids[i].tolist(),
                   score=float(ranking[i]),population_members=int(pop[i,topids[i]].sum()),
                   sampled_cancellation=mean[i,0,topids[i],f['latent_cancellation']].tolist(),
                   null_cancellation=mean[i,0,topids[i],f['random_group_cancellation']].tolist()) for i in selected_indices]
    selection=dict(protocol=PROTOCOL,source='all8 action_only calibration contexts; sample statistics only',
                   module_names=[v['module'] for v in selected],groups=selected,
                   rule='Four modules ranked by sum of top8 book0 population-weighted member-gradient-energy times squared sampled cancellation; only occupied multi-member latent-active codewords')
    out=root/'results/analysis_capture'; out.mkdir(exist_ok=True)
    save(out/'selection.json',selection)
    result=dict(protocol=PROTOCOL,phase='analyze_capture',status='complete',job_id=os.environ['PBS_JOBID'],
                records=24,calibration_cases=8,target_linears=614,member_metrics='fixed stratified sample; not full population rates',
                exact_gradients='full codeword gradients across all repeated calls',aggregate=aggregate,per_case=cases,selection=selection)
    save(out/'analysis.json',result); save(Path(os.environ['ARTIFACTS'])/'summary.json',result)
    fields=list(cases[0])
    with (out/'paired_cases.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(cases)
    lines=['# Codeword gradient conflict: sampled screen','','All8 calibration contexts; all3 checkpoints; all614 Linears.',
           'Within-codeword metrics use fixed stratified samples. Exact shared-codeword gradients use all members/calls.', '',
           '| Checkpoint | Latent cancellation | Random grouping | Mean pair cosine | Latent shared-step opposed member fraction | Action/latent conflicting codeword fraction |',
           '|---|---:|---:|---:|---:|---:|']
    for arm,v in aggregate.items():
        lines.append(f"| {arm} | {v['latent_cancellation']:.6f} | {v['random_group_cancellation']:.6f} | {v['latent_mean_pair_cosine']:.6f} | {v['latent_shared_harm_fraction']:.6f} | {v['action_latent_conflicting_codeword_fraction']:.6f} |")
    lines+=['','Sampling cancellation alone is not evidence of harmful assignment. Full-member and finite-forward checks remain required.', '', 'Selected modules:']
    lines += [f"- {v['module']}, book0, codewords={v['codewords']}" for v in selected]
    (out/'REPORT.zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def analyze_exact(torch,root,p):
    from common import weight_vectors, vector_mask
    from stats import cancellation
    rows=load_units(torch,root,p,'exact')
    if len(rows)!=8: raise RuntimeError('All8 calibration contexts required for exact direction construction')
    selection=json.loads((root/'results/analysis_capture/selection.json').read_text())
    source=Path(p['source_root'])/'results/full/banks'/p['initial_arm']
    overlay=torch.load(Path(p['distill_result'])/'selected_codebook_overlay.pt',weights_only=True,map_location='cpu')
    modules,report,full_checks={},[],[]
    for group in selection['groups']:
        name=group['module']
        gradients=torch.stack([r['decoded_latent_gradients'][name].float() for r in rows]).mean(0)
        bank=torch.load(source/f"{group['module_index']:04d}.pt",weights_only=True,map_location='cpu')
        enc=bank['encoded']; idx=enc['indices'].reshape(-1,2).long()
        codes=torch.tensor(group['codewords'])
        mask=(idx[:,0,None]==codes[None,:]).any(1)
        ids=mask.nonzero().flatten()
        g=weight_vectors(torch,gradients)[ids]
        rows_index=ids//((gradients.shape[1]+3)//4)
        logs=overlay['master_state'][name]['log_scales']
        # Reuse captured deployed BF16 scales for one identical CPU-built probe across GPU shards.
        modules[name]=dict(flat_ids=ids,indices=idx[ids],gradient=g,row_ids=rows_index,
                           books=overlay['master_state'][name]['books'],log_scales=logs,
                           row_scales=rows[0]['stored_row_scales'][name],
                           active=enc['row_scales']>0,codes=codes,shape=list(gradients.shape),
                           valid_coordinates=vector_mask(torch,gradients.shape,ids))
        valid=modules[name]['valid_coordinates']
        for row in rows:
            members=weight_vectors(torch,row['decoded_latent_gradients'][name])[ids].double()
            scales=row['stored_row_scales'][name][rows_index].double()
            for code in codes.tolist():
                pick=idx[ids,0]==code
                raw=members[pick]
                s=scales[pick]
                u=raw*s[:,None]
                total=u.sum(0)
                if not torch.allclose(total.float(),row['exact_latent_books'][name][0,code],atol=1e-7,rtol=1e-3):
                    raise RuntimeError('Full-member sum differs from exact autograd codeword gradient')
                c,pair,active=cancellation(torch,u[None],torch.ones((1,len(u)),dtype=torch.bool))
                live=u.norm(dim=1)>0
                harm=((u*total).sum(1)<0)&live
                divisor=(s.square()[:,None]*valid[pick]).sum(0)
                energy=raw.square().sum()
                retained=(total.square()/divisor.clamp_min(1e-30)).sum()/energy.clamp_min(1e-30)
                if not bool(torch.isfinite(retained)) or float(retained)>1+1e-6:
                    raise RuntimeError('Invalid full-member projection energy')
                full_checks.append(dict(module=name,case_id=row['case_id'],book=0,codeword=code,
                    members=len(u),active_members=int(active[0]),cancellation=float(c[0]),
                    mean_pair_cosine=float(pair[0]),shared_step_harm_fraction=float(harm.sum()/live.sum().clamp_min(1)),
                    retained_descent_fraction=float(retained),untied_gradient_sq=float(energy),
                    zero_scale_members=int((s==0).sum()),
                    zero_scale_gradient_energy_fraction=float(raw[s==0].square().sum()/energy.clamp_min(1e-30))))
        report.append(dict(module=name,book=0,codeword_count=len(codes),full_member_count=len(ids),
                           gradient_sq=float(g.double().square().sum()),case_ids=[r['case_id'] for r in rows],
                           zero_scale_support_members=int((modules[name]['row_scales'][rows_index]==0).sum())))
    out=root/'results/analysis_exact'; out.mkdir(exist_ok=True)
    torch.save(dict(protocol=PROTOCOL,module_names=list(modules),modules=modules),out/'directions.pt')
    from intervene import make_updates
    directions=dict(protocol=PROTOCOL,module_names=list(modules),modules=modules)
    updates,norms=make_updates(torch,directions,None,p,device='cpu')
    baselines={}
    for name,d in modules.items():
        if not all(torch.equal(r['stored_row_scales'][name],d['row_scales']) for r in rows):
            raise RuntimeError('Stored BF16 scales changed across exact captures')
        books=d['books'].to(torch.bfloat16).float(); idx=d['indices']; s=d['row_scales'].float()[d['row_ids']]
        baselines[name]=(((books[0][idx[:,0]]+books[1][idx[:,1]])*s[:,None])*d['valid_coordinates']).to(torch.bfloat16)
    torch.save(dict(protocol=PROTOCOL,flat_ids={n:d['flat_ids'] for n,d in modules.items()},baseline=baselines,
                    values=updates,norms=norms),out/'updates.pt')
    result=dict(protocol=PROTOCOL,phase='analyze_exact',status='complete',job_id=os.environ['PBS_JOBID'],
                calibration_cases=8,targeted_modules=len(modules),groups=report,full_member_metrics=full_checks,
                intervention_step_norms=p['intervention_step_norms'],selection_used_test=False,norms=norms,
                note='Untied directions relax weight tying; finite probes are not a compressed-model improvement')
    save(out/'analysis.json',result); save(Path(os.environ['ARTIFACTS'])/'summary.json',result)
    lines=['# Full-member codeword checks','','All8 action-only calibration contexts; selected4 modules and32 book0 codewords.',
           'Gradients are summed per vector over sampler calls before grouping. Book1 is fixed.',
           'Full-member sums match exact autograd codeword gradients. Zero-active rows are reported explicitly.', '',
           '| Module | Codeword | Members | Full cancellation | Sample cancellation | Full pair cosine | Shared-step opposed members | Retained descent fraction | Zero-scale gradient energy |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for group in selection['groups']:
        for pos,code in enumerate(group['codewords']):
            records=[r for r in full_checks if r['module']==group['module'] and r['codeword']==code]
            if len(records)!=8:raise RuntimeError('Full-member check must cover all8 calibration contexts')
            mean=lambda field:statistics.mean(r[field] for r in records)
            lines.append(f"| {group['module']} | {code} | {records[0]['members']} | {mean('cancellation'):.6f} | {group['sampled_cancellation'][pos]:.6f} | {mean('mean_pair_cosine'):.6f} | {mean('shared_step_harm_fraction'):.6f} | {mean('retained_descent_fraction'):.6f} | {mean('zero_scale_gradient_energy_fraction'):.6f} |")
    lines+=['','Pooled-calibration projection and actual BF16 perturbation norms are recorded in analysis.json.',
            'Nonzero zero-scale gradient energy would mean untied updates also revive rows held fixed by zero scales; their benefit could not be attributed solely to codeword sharing.',
            'Untied directions increase degrees of freedom; no fixed-bit assignment improvement is established by this check.']
    (out/'REPORT.zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def analyze_interventions(torch,root,p):
    from common import contexts
    from intervene import variants,key
    roster=contexts(p)
    rows=[]
    for shard in range(4):
        d=root/'results'/f'intervene-{shard}'
        s=json.loads((d/'summary.json').read_text())
        expected=[c for c in roster if c['shard']==shard]
        if s['status']!='complete' or s['protocol']!=PROTOCOL or s['row_count']!=len(expected)*6:
            raise RuntimeError('Complete360-row intervention cohort required')
        for c in expected:
            for label,kind,eps in variants(p):
                r=json.loads((d/f'{key(c,label)}.json').read_text())
                if r['context']!=c or r['label']!=label or r['protocol']!=PROTOCOL or not r['teacher_reader_unquantized'] or not r['future_injection_verified']:
                    raise RuntimeError('Intervention context/reader mismatch')
                rows.append(r)
    if len(rows)!=360: raise RuntimeError('Incomplete intervention row count')
    metric_names=('future_nmse','teacher_first10','teacher_full32','teacher_late22','teacher_gripper','native_first10')

    def values(r,baseline=False):
        a=r['baseline_teacher' if baseline else 'teacher']; b=r['baseline_native' if baseline else 'native']
        return dict(future_nmse=r['baseline_future_nmse' if baseline else 'future_nmse'],
                    teacher_first10=a['first10motor_rmse'],teacher_full32=a['full32motor_rmse'],
                    teacher_late22=a['late22motor_rmse'],teacher_gripper=a['first10gripper_rmse'],
                    native_first10=b['first10motor_rmse'])

    def aggregate(subset,baseline=False):
        groups={}
        for r in subset: groups.setdefault(r['case_id'],[]).append(values(r,baseline))
        means={str(c):{m:statistics.mean(v[m] for v in records) for m in metric_names} for c,records in groups.items()}
        return dict(case_count=len(means),context_count=len(subset),mean={m:statistics.mean(v[m] for v in means.values()) for m in metric_names},per_case=means)

    tables,contrasts={},{}
    for split in ('calibration','selection','test'):
        for level,eps in enumerate(p['intervention_step_norms']):
            arms={}
            for kind in ('shared','untied','random'):
                subset=[r for r in rows if r['split']==split and r['label']==f'{kind}_{level}']
                v=aggregate(subset)
                expected={'calibration':(8,8),'selection':(4,8),'test':(22,44)}[split]
                if (v['case_count'],v['context_count'])!=expected: raise RuntimeError('Split denominator changed')
                arms[kind]=v; tables[f'{split}/{kind}_{level}']=v
                if kind=='shared': arms['baseline']=aggregate(subset,True);tables[f'{split}/baseline_{level}']=arms['baseline']
            for left,right in (('untied','shared'),('untied','random'),('shared','baseline'),('untied','baseline'),('random','baseline')):
                metrics={}
                for m in metric_names:
                    delta={c:arms[left]['per_case'][c][m]-arms[right]['per_case'][c][m] for c in arms[left]['per_case']}
                    metrics[m]=dict(mean_delta=statistics.mean(delta.values()),median_delta=statistics.median(delta.values()),
                                    improved=sum(v<0 for v in delta.values()),worsened=sum(v>0 for v in delta.values()),
                                    tied=sum(v==0 for v in delta.values()),per_case_delta=delta)
                contrasts[f'{split}/{level}/{left}-minus-{right}']=metrics
    checks=[]
    for level in range(2):
        cal=contrasts[f'calibration/{level}/untied-minus-shared']['future_nmse']
        test=contrasts[f'test/{level}/untied-minus-shared']
        rand=contrasts[f'test/{level}/untied-minus-random']['future_nmse']
        checks.append(dict(level=level,cal_future_mean_lower=cal['mean_delta']<0,
            test_future_mean_lower=test['future_nmse']['mean_delta']<0,
            test_future_majority=test['future_nmse']['improved']>=12,
            teacher_motor_mean_no_worse=test['teacher_first10']['mean_delta']<=0,
            better_than_random_future=rand['mean_delta']<0))
    gate=all(all(v for k,v in check.items() if k!='level') for check in checks)
    quadrants={}
    for level,eps in enumerate(p['intervention_step_norms']):
        contrast=contrasts[f'test/{level}/untied-minus-shared']
        f=contrast['future_nmse']['per_case_delta']; m=contrast['teacher_first10']['per_case_delta']
        quadrants[str(level)]=dict(epsilon=eps,case_count=len(f),
            both_improve=sum(f[c]<0 and m[c]<0 for c in f),
            future_improves_motor_worsens=sum(f[c]<0 and m[c]>0 for c in f),
            future_worsens_motor_improves=sum(f[c]>0 and m[c]<0 for c in f),
            both_worsen=sum(f[c]>0 and m[c]>0 for c in f),
            any_exact_tie=sum(f[c]==0 or m[c]==0 for c in f))
    out=root/'results/analysis';out.mkdir(exist_ok=True)
    # Descriptive dimension reference only; it does not alter the frozen gate.
    exact=json.loads((root/'results/analysis_exact/analysis.json').read_text())
    for shard in range(4):
        norms=json.loads((root/'results'/f'intervene-{shard}'/'step_norms.json').read_text())
        if norms!=exact['norms']: raise RuntimeError('Perturbation norm receipts differ across shards')
    directions=torch.load(root/'results/analysis_exact/directions.pt',weights_only=True,map_location='cpu',mmap=True)
    geometry={}
    for name,d in directions['modules'].items():
        valid=d['valid_coordinates']
        live=d['row_scales'][d['row_ids']]>0
        occupied=torch.zeros((256,4),dtype=torch.float64)
        occupied.index_add_(0,d['indices'][:,0].long(),(valid&live[:,None]).double())
        rank,dimension=int((occupied>0).sum()),int(valid.sum())
        if not 0<rank<=dimension: raise RuntimeError('Invalid shared-tangent dimension')
        reference=rank/dimension
        retained=exact['norms']['projection'][name]['retained_descent_fraction']
        geometry[name]=dict(shared_tangent_rank=rank,decoded_support_dimension=dimension,
            isotropic_expected_energy_fraction=reference,actual_retained_energy_fraction=retained,
            actual_over_isotropic_reference=retained/reference)
    result=dict(protocol=PROTOCOL,phase='analyze',status='complete',job_id=os.environ['PBS_JOBID'],
                row_count=360,total_contexts=60,test_case_count=22,test_context_count=44,
                tables=tables,contrasts=contrasts,engineering_progression_checks=checks,assignment_research_supported=gate,
                perturbation_norm_receipts_match=True,
                test_untied_shared_future_motor_quadrants=quadrants,
                shared_tangent_dimension_reference=geometry,
                claim_scope=p['claim_scope'])
    save(out/'analysis.json',result)
    # Bounded terminal receipt; full per-case tables stay in the result artifact.
    save(Path(os.environ['ARTIFACTS'])/'summary.json',{k:v for k,v in result.items() if k not in ('tables','contrasts')})
    flat=[]
    for r in rows:
        flat.append(dict(split=r['split'],case_id=r['case_id'],seed_index=r['seed_index'],label=r['label'],**values(r)))
    with (out/'paired_contexts.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    lines=['# Shared-codeword finite intervention','','完整60contexts；test22cases/44contexts。所有方向仅由8个calibration contexts决定。',
        'Untied是增加自由度的容量诊断，不是同bits可部署方法。', '',
        '| Split | Arm | Future NMSE | Teacher first10 | Teacher full32 | Teacher late22 | Teacher gripper | Native first10 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for name,v in tables.items(): lines.append('| '+name.replace('/',' | ')+' | '+' | '.join(f'{v["mean"][m]:.9g}' for m in metric_names)+' |')
    lines+=['','| Test contrast | Future mean delta | Future improved | Teacher first10 delta | Motor improved |',
            '|---|---:|---:|---:|---:|']
    for name,v in contrasts.items():
        if name.startswith('test/'):
            lines.append(f"| {name} | {v['future_nmse']['mean_delta']:.9g} | {v['future_nmse']['improved']}/22 | {v['teacher_first10']['mean_delta']:.9g} | {v['teacher_first10']['improved']}/22 |")
    lines+=['','| Untied versus shared epsilon | Both improve | Future improves / motor worsens | Future worsens / motor improves | Both worsen | Any exact tie |',
            '|---|---:|---:|---:|---:|---:|']
    for v in quadrants.values():
        lines.append(f"| {v['epsilon']} | {v['both_improve']} | {v['future_improves_motor_worsens']} | {v['future_worsens_motor_improves']} | {v['both_worsen']} | {v['any_exact_tie']} |")
    lines+=['','Quadrants are descriptive signs for22 held-out cases after averaging paired seeds, without an uncertainty or significance claim.']
    lines+=['','| Module | Shared tangent rank | Support dimension | Isotropic rank/D | Actual retained energy | Actual/reference |',
            '|---|---:|---:|---:|---:|---:|']
    for name,v in geometry.items():
        lines.append(f"| {name} | {v['shared_tangent_rank']} | {v['decoded_support_dimension']} | {v['isotropic_expected_energy_fraction']:.9g} | {v['actual_retained_energy_fraction']:.9g} | {v['actual_over_isotropic_reference']:.6g} |")
    lines+=['','The isotropic rank/D reference explains dimensional restriction only. It is not a matched alternative assignment, empirical gradient null, significance test, or finite-BF16 performance prediction.']
    lines += ['',f'Predefined engineering progression gate={gate}; not a statistical significance test.',p['claim_scope']]
    (out/'REPORT.zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    guard()
    import torch
    torch.set_num_threads(4)
    a=argparse.ArgumentParser(); a.add_argument('--phase',choices=('analyze_capture','analyze_exact','analyze'),required=True)
    args=a.parse_args(); root=Path(os.environ['ROOT']); p=json.loads((root/'protocol.json').read_text())
    if args.phase=='analyze_capture': analyze_capture(torch,root,p)
    elif args.phase=='analyze_exact': analyze_exact(torch,root,p)
    else: analyze_interventions(torch,root,p)


if __name__=='__main__': main()
