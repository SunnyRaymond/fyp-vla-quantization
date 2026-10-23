"""Fit fixed soft-bin coordinates from the original C80 teacher, on PBS CPU."""
import json
import os
from pathlib import Path
import socket


def guard():
    host = socket.gethostname().split('.')[0]
    if not os.environ.get('PBS_JOBID') or 'login' in host:
        raise RuntimeError('A real PBS compute allocation is required')
    if host not in {n.split('.')[0] for n in Path(os.environ['PBS_NODEFILE']).read_text().split()}:
        raise RuntimeError('Host outside allocation')


def main():
    guard()
    import torch
    torch.set_num_threads(4)
    root = Path(os.environ['NEW_ROOT'])
    p = json.loads((Path(os.environ['ARTIFACTS'])/'protocol.json').read_text())
    teacher = Path(p['control_root'])/'results/teacher/calibration'
    plan = json.loads(Path(p['calibration']['plan_file']).read_text())
    if plan['calibration_cohorts']['c80'] != p['calibration']['case_ids']:
        raise RuntimeError('Original C80 differs')
    rows = {r['case_id']:r for r in plan['inputs']}
    moments = {}
    for case in p['calibration']['case_ids']:
        saved = torch.load(teacher/f'{case}_0.pt',map_location='cpu',weights_only=True)
        if saved['plan_row'] != rows[case] or saved['sampler_seed'] != rows[case]['sampler_seeds'][0]:
            raise RuntimeError('Shared teacher binding differs')
        values = {'action':saved['action'][:10,:6].double().T,
                  'latent':saved['future'][:,:,1:].double().movedim(1,0)}
        for name,value in values.items():
            x = value.reshape(value.shape[0],-1)
            if not bool(torch.isfinite(x).all()):
                raise RuntimeError('Nonfinite teacher')
            if name not in moments:
                moments[name] = [torch.zeros(x.shape[0],dtype=torch.float64),torch.zeros(x.shape[0],dtype=torch.float64),0]
            moments[name][0] += x.sum(1)
            moments[name][1] += x.square().sum(1)
            moments[name][2] += x.shape[1]
        print('STATS_CASE',case,flush=True)
    result = {'protocol':p['protocol'],'complete':True,'pbs_jobid':os.environ['PBS_JOBID'],
              'calibration_case_ids':p['calibration']['case_ids'],'seed_indices':[0],
              'new_teacher_queries':0,'heldout_used':False,'config':p['softbin']}
    for name,(s,ss,n) in moments.items():
        mean = s/n
        scale = (ss/n-mean.square()).clamp_min(0).sqrt().clamp_min(p['softbin']['std_floor'])
        result[name] = {'mean':mean.tolist(),'scale':scale.tolist(),'count_per_coordinate':n}
    out = root/'results/fit_stats'
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('DUAL_KL_STATS_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
