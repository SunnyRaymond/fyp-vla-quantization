"""Select tiles using saved calibration scores before any new model outcome."""
import json
from pathlib import Path
from common import guard,protocol,save,receipt
from design import SCORE_KEYS,select_local,conditions


def main():
    guard()
    import torch
    from test_intervention import self_check
    self_check()
    p=protocol();root=Path(p['remote_root']);parent=Path(p['parent_root'])
    layout=json.loads((parent/'layout.json').read_text());ms={m['name']:m for m in layout['modules']}
    data=[]
    for i,case in enumerate(p['calibration_cases']):
        row=torch.load(parent/f'results/score-{i%4}/case_{case}.pt',map_location='cpu',weights_only=True)
        if row['protocol']!='fastwam-vq-residual-capacity-v1' or row['case_id']!=case or row['seed_index']!=0:raise RuntimeError('Source score identity differs')
        data.append(row)
    means={k:torch.stack([d['scores'][k] for d in data]).mean(0) for k in SCORE_KEYS}
    tiles=[]
    for mi,name in enumerate(p['selected_modules']):
        m=ms[name]
        for label,local in select_local(torch,means,m,p['selection_seed']+mi):
            global_index=m['offset']+local
            tiles.append(dict(id=f't{len(tiles):02d}',module=name,module_index=mi,local_tile=local,global_tile=global_index,label=label,
                vector_count=p['tile_vectors'],weight_count=p['tile_vectors']*4,
                mean_scores={k:float(means[k][global_index]) for k in SCORE_KEYS},
                predictions={str(d['case_id']):{k:float(d['scores'][k][global_index]) for k in SCORE_KEYS} for d in data}))
    cs=conditions(tiles,p['selection_seed'])
    plan=dict(protocol=p['protocol'],case_ids=p['calibration_cases'],selected_modules=p['selected_modules'],tiles=tiles,conditions=cs,
              scope='Selection from calibration means only; selected endpoints recorded even if their label has no matching sign. All singles precede interpretation; multi conditions fixed before new outcomes.')
    save(root/'plan.json',plan)
    result=receipt('prepare',status='complete',tile_count=28,condition_count=54,intervention_count=52,case_count=8,row_count=432,model_forward_count=0,
        selected_tiles=[dict(id=t['id'],module=t['module'],local_tile=t['local_tile'],label=t['label'],mean_scores=t['mean_scores']) for t in tiles])
    save(root/'prepare_summary.json',result)


if __name__=='__main__':main()
