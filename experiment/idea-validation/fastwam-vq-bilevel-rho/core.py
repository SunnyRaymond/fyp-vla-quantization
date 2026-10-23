"""Pure controls for scalar finite-difference bilevel updates and paired metrics."""
import math

SUITES=('libero_spatial','libero_object','libero_goal','libero_10')
METRICS=('first10motor_rmse','full32motor_rmse','late22motor_rmse','first10gripper_rmse','future_latent_nmse')

def rho_step(rho,minus,plus,config):
    if not all(math.isfinite(v) and v>=0 for v in (rho,minus,plus)) or rho<=0:
        raise ValueError('Invalid rho or development objective')
    lo,hi=math.log(config['minimum_rho']),math.log(config['maximum_rho'])
    s=math.log(rho);delta=math.log(config['probe_factor'])
    sm,sp=max(lo,s-delta),min(hi,s+delta)
    gradient=(plus-minus)/(sp-sm)
    contrast=abs(plus-minus)/max((plus+minus)/2,1e-12)
    active=contrast>config['relative_deadband']
    updated=min(hi,max(lo,s-math.log(config['outer_step_factor'])*((gradient>0)-(gradient<0)))) if active else s
    return {'rho_before':rho,'rho_minus':math.exp(sm),'rho_plus':math.exp(sp),
            'minus_rmse':minus,'plus_rmse':plus,'gradient_log_rho':gradient,
            'relative_contrast':contrast,'signal_above_deadband':active,'rho_after':math.exp(updated)}

def aggregate(records,cases,seeds,metrics=METRICS):
    keys=[(r['case_id'],r['seed_index']) for r in records]
    expected={(c,s) for c in cases for s in seeds}
    if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('Incomplete or duplicate paired cohort')
    by_suite={}
    for suite in SUITES:
        rows=[r for r in records if r['suite']==suite];ids=sorted({r['case_id'] for r in rows})
        if not ids:raise ValueError('Missing suite')
        by_suite[suite]={m:sum(sum(r[m] for r in rows if r['case_id']==c)/len(seeds) for c in ids)/len(ids) for m in metrics}
    overall={m:sum(r[m] for r in by_suite.values())/4 for m in metrics}
    if any(not math.isfinite(v) or v<0 for v in overall.values()):raise ValueError('Invalid metric')
    return {'queries':len(records),'cases':len(cases),'overall':overall,'by_suite':by_suite}
