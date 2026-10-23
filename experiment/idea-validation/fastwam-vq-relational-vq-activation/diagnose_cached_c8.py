"""C8 fit diagnosis using saved outputs only, without model/teacher forwards."""
import json
import math
import os
from pathlib import Path

from relational_selfcheck import require_allocation


def main():
    job,host=require_allocation()
    import torch
    import relational_reference as prior
    from softbin_loss import losses
    from relational_loss import relational_losses
    from diagnose_vq import read,save
    torch.set_num_threads(4)
    root=Path(os.environ['NEW_ROOT']);p=read(root/'protocol.json');cases=p['vq_training']['case_ids']
    stats=read(Path(p['source_dual_root'])/'results/fit_stats/summary.json')
    scale=read(Path(p['source_dual_root'])/'results/prepare/loss_scaling.json')
    prep=read(Path(p['source_relational_root'])/'results/prepare/summary.json')
    raw=torch.load(Path(p['baseline_root'])/'results/reference/reference_cache.pt',map_location='cpu',weights_only=True)
    result={'complete':False,'pbs_jobid':job,'hostname':host,'new_model_queries':0,
            'new_teacher_queries':0,'new_training_updates':0,'uses_heldout':False,'case_ids':cases,'arms':{}}
    for arm in ('raw','kl8'):
        records=[]
        for case in cases:
            target=torch.load(Path(p['control_root'])/'results/teacher/calibration'/f'{case}_0.pt',map_location='cpu',weights_only=True)
            source=raw['parent'][str(case)] if arm=='raw' else torch.load(root/'results/vq_kl/tensors/calibration'/f'{case}_0.pt',map_location='cpu',weights_only=True)
            assert source['sampler_seed']==target['sampler_seed']
            v=prior._terms(torch,losses,relational_losses,source['action'],source['future'],target,stats,p,prep['relational_config'])
            soft,total=prior._objectives(v,p,scale,prep['coefficients'])
            row={'case_id':case,**{k:float(v) for k,v in v.items()},'softbin_objective':float(soft),'relational_objective':float(total)}
            row['first10motor_rmse']=math.sqrt(row['action_mse'])
            records.append(row)
        result['arms'][arm]={'records':records,'mean':{k:sum(r[k] for r in records)/8 for k in records[0] if k!='case_id'}}
    records=[]
    for case in cases:
        old=read(Path(p['control_root'])/'results/calibration/baseline/queries'/f'{case}_0.json')
        records.append({'case_id':case,'first10motor_rmse':old['first10motor_rmse'],'action_mse':old['first10motor_mse'],'latent_nmse':old['future_latent_nmse']})
    result['arms']['mse8']={'records':records,'mean':{k:sum(r[k] for r in records)/8 for k in records[0] if k!='case_id'}}
    result['original_kl_history']=read(root/'results/vq_kl/training_receipt.json')['vq_history']
    result['complete']=True
    save(root/'results/vq_diagnosis/cached_summary.json',result)
    print('CACHED_C8_DIAGNOSIS_COMPLETE',flush=True)


if __name__=='__main__':main()
