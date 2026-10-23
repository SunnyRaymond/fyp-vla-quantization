"""CPU validation of raw-nullspace projections and all eight BF16 repair pairs."""
import os
from pathlib import Path
from anchor_selfcheck import require_allocation
from prepare import read,save,load_reference
from mechanism_math import matched_repairs,selfcheck


def main():
    require_allocation()
    import torch
    torch.set_num_threads(4)
    p=read(Path(__file__).with_name('protocol.json'))
    config=read(Path(__file__).with_name('mechanism_protocol.json'))
    root=Path(p['remote_root']);prep=read(root/'results/prepare/summary.json')
    cache,_,rows=load_reference(torch,p)
    result={'protocol':config['protocol'],'unit':'mechanism_preflight','pbs_jobid':os.environ['PBS_JOBID'],
        'complete':False,'new_model_queries':0,'training_updates':0,'heldout_used':False,
        'selfcheck':selfcheck(torch,prep,p['anchor']),'cases':[]}
    for case in config['case_ids']:
        z=torch.load(root/'results/anchor_kl/tensors/calibration'/f'{case}_0.pt',map_location='cpu',weights_only=True)
        if z['plan_row']!=rows[case] or z['updates']!=8:raise ValueError('C8 endpoint binding differs')
        _,check=matched_repairs(torch,z['future'],cache['teacher'][str(case)]['future'],prep,p['anchor'],config['repair_nmse_relative_tolerance'])
        result['cases'].append({'case_id':case,**check})
    result['complete']=True
    save(root/'results/mechanism_preflight/summary.json',result)
    print('MECHANISM_PREFLIGHT_PASSED',flush=True)


if __name__=='__main__':main()
