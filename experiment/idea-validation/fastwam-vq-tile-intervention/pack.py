"""Decode frozen source on CUDA once, preserving its exact BF16 decoder path."""
import json
import os
from pathlib import Path
import signal
import time
from common import guard,protocol,save,receipt,contract


def main():
    guard()
    import torch
    import codebook
    from checkpointing import atomic_torch_save
    p=protocol();root=Path(p['remote_root']);out=root/'packed';out.mkdir(exist_ok=True)
    contract(out/'protocol.json',p)
    plan=json.loads((root/'plan.json').read_text());wanted={}
    for t in plan['tiles']:wanted.setdefault(t['module'],[]).append(t)
    overlay=torch.load(Path(p['distill_result'])/'selected_codebook_overlay.pt',map_location='cpu',weights_only=True)
    names=overlay['module_names'];n=p['pack_chunk_modules'];expected=(len(names)+n-1)//n
    if len(names)!=614:raise RuntimeError('Source targets changed')
    stop=[];signal.signal(signal.SIGTERM,lambda *_:stop.append(True))
    def finish(status):
        done=sum((out/f'chunk_{i:02d}.pt').is_file() for i in range(expected))
        s=receipt('pack',status=status,chunk_count=done,expected_chunks=expected,target_linears=614,selected_tiles=28,model_forward_count=0)
        save(out/'summary.json',s)
    for chunk in range(expected):
        path=out/f'chunk_{chunk:02d}.pt'
        if path.is_file():continue
        if stop or time.time()+600>float(os.environ['FW_JOB_START_EPOCH'])+p['walltime_seconds']:
            finish('resumable');return
        weights={};transforms={};tile_data={}
        for i in range(chunk*n,min((chunk+1)*n,len(names))):
            name=names[i]
            b=torch.load(Path(p['source_root'])/'results/full/banks'/p['initial_arm']/f'{i:04d}.pt',map_location='cpu',weights_only=True)
            if b['module']!=name:raise RuntimeError('Source bank order differs')
            enc={k:v.to('cuda') if torch.is_tensor(v) else v for k,v in b['encoded'].items()}
            state=overlay['master_state'][name]
            enc['books']=state['books'].to('cuda',dtype=torch.bfloat16)
            enc['row_scales']=(state['log_scales'].to('cuda').exp()*(enc['row_scales']>0)).to(torch.bfloat16)
            dense=codebook.decode_vq(enc)
            weights[name]=dense.cpu();transforms[name]=b['transform']
            if name in wanted:
                if dense.shape[1]%4:raise RuntimeError('Selected module has padding')
                idx=enc['indices'].long()
                base=(enc['books'][0].float()[idx[...,0]]*enc['row_scales'].float()[:,None,None]).to(torch.bfloat16).reshape(dense.shape)
                for t in wanted[name]:
                    a=t['local_tile']*p['tile_vectors']*4;z=a+p['tile_vectors']*4
                    tile_data[t['id']]=dict(full=dense.flatten()[a:z].cpu().clone(),base=base.flatten()[a:z].cpu().clone())
                del base,idx
            del enc,dense,b
        atomic_torch_save(torch,path,dict(protocol=p['protocol'],chunk=chunk,weights=weights,transforms=transforms,tiles=tile_data))
        print('TILE_PACK_CHUNK',chunk,expected,flush=True)
        del weights,transforms,tile_data
        finish('running')
    finish('complete')


if __name__=='__main__':main()
