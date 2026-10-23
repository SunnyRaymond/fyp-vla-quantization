"""CPU-only construction of all60 matched controls from frozen donor caches."""
import json
from pathlib import Path
from common import guard,protocol,contexts,stem,save,receipt
from matching import match,nmse


def main():
    guard()
    import torch
    from checkpointing import atomic_torch_save
    p=protocol();root=Path(p['remote_root']);prior=Path(p['future_reader_root'])
    oldp=json.loads((prior/'protocol.json').read_text())
    for key in ('calibration_cases','selection_cases','test_cases','evaluation_seed_indices','source_root','initial_arm','distill_result','protected_result','roots'):
        if p[key]!=oldp[key]:raise RuntimeError('Frozen source changed: '+key)
    manifest=json.loads((prior/'input_manifest.json').read_text())
    if manifest['protocol']!=oldp['protocol'] or len(manifest['contexts'])!=60:raise RuntimeError('Wrong prior manifest')
    for phase in ('teacher','cross_a','cross_z'):
        for shard in (0,1):
            s=json.loads((prior/f'results/{phase}-{shard}/summary.json').read_text())
            if s['protocol']!=oldp['protocol'] or s['status']!='complete' or s['context_count']!=30 or s['row_count']!=(120 if phase=='teacher' else 60):
                raise RuntimeError('Historical full cohort is incomplete')
    table={(e['case_id'],e['seed_index']):e for e in manifest['contexts']}
    result=[];prepared=root/'prepared';prepared.mkdir(exist_ok=True)
    for c in contexts(p):
        e=table[c['case_id'],c['seed_index']]
        if any(e[k]!=v for k,v in c.items()):raise RuntimeError('Paired cohort assignment changed')
        old=torch.load(e['cache_path'],map_location='cpu',weights_only=True)
        protected=torch.load(prior/'futures'/f'{stem(c)}.pt',map_location='cpu',weights_only=True)
        if old['context']!=c or protected['context']!=c or old['sampler_seed']!=e['sampler_seed'] or protected['sampler_seed']!=e['sampler_seed'] or protected['protocol']!=oldp['protocol']:
            raise RuntimeError('Donor context/seed differs')
        caps=old['captures'];teacher=caps['bf16'];a_raw=caps['distilled_self']['video_latents'];b_raw=protected['output']['video_latents'];t=teacher['video_latents']
        a=a_raw.to(torch.bfloat16);b=b_raw.to(torch.bfloat16)
        if tuple(t.shape)!=(1,48,3,14,28):raise RuntimeError('Unexpected future layout')
        matched,check=match(torch,a,b,t,p,seed=e['sampler_seed'])
        check.update(teacher_dtype=str(t.dtype),source_a_dtype=str(a_raw.dtype),source_b_dtype=str(b_raw.dtype),
            a_native_nmse=nmse(torch,a_raw,t),b_native_nmse=nmse(torch,b_raw,t))
        refs={};rows={}
        for name in ('action_only','action_latent'):
            payload=torch.load(prior/f"results/teacher-{c['shard']}"/f'{stem(c)}_{name}.pt',map_location='cpu',weights_only=True)
            r=payload['row']
            if r['protocol']!=oldp['protocol'] or r['context']!=c or r['sampler_seed']!=e['sampler_seed'] or r['reader']!='bf16' or r['donor']!=name or not r['future_injection_verified']:
                raise RuntimeError('Cached BF16-reader reference differs')
            refs[name]=payload['action'];rows[name]=r
        path=prepared/f'{stem(c)}.pt'
        atomic_torch_save(torch,path,dict(protocol=p['protocol'],context=c,sampler_seed=e['sampler_seed'],
            futures=dict(action_only=a,action_latent=b,matched=matched),teacher_future=t,teacher_action=teacher['action'],
            action_initial_noise=teacher['_diagnostic']['action_initial_noise'],reader_actions=refs,reader_rows=rows,matching=check))
        result.append(dict(**c,sampler_seed=e['sampler_seed'],prepared_path=str(path),domain=rows['action_only']['domain'],dimension=rows['action_only']['dimension'],**check))
        print('MATCH_PREPARED',stem(c),check['relative_nmse_gap'],flush=True)
        del old,protected,caps,teacher,a_raw,b_raw,a,b,t,matched,refs,rows
    save(root/'input_manifest.json',dict(protocol=p['protocol'],contexts=result))
    summary=receipt('prepare',status='complete',context_count=60,test_case_count=22,matched_count=60,cached_reader_rows=120,model_forward_count=0,
        max_absolute_nmse_gap=max(r['absolute_nmse_gap'] for r in result),max_relative_nmse_gap=max(r['relative_nmse_gap'] for r in result),
        min_direction_cosine=min(r['direction_cosine'] for r in result if r['direction_cosine'] is not None),
        amplification_contexts=sum(r['alpha']>1 for r in result),matching_rows=result)
    save(root/'prepare_summary.json',summary)


if __name__=='__main__':main()
