"""Validate reusable references and byte layout in a CPU allocation."""
import json
import os
from pathlib import Path
from common import guard,protocol,contexts,receipt,save


def main():
    guard()
    import torch
    from test_capacity import self_check
    self_check()
    p=protocol(); root=Path(p['remote_root']); prior=Path(p['future_reader_root'])
    for phase in ('teacher','cross'):
        s=json.loads((prior/f'results/analysis_{phase}/analysis.json').read_text())
        if s['status']!='complete' or s['row_count']!=240 or s['test_case_count']!=22 or s['replay_max_abs']!=0:
            raise RuntimeError('Reusable full-cohort references are not verified')
    manifest=json.loads((prior/'input_manifest.json').read_text())
    if len(manifest['contexts'])!=60: raise RuntimeError('Missing reference contexts')
    for c in manifest['contexts']:
        if not Path(c['cache_path']).is_file(): raise RuntimeError('Missing paired source capture')
    overlay=torch.load(Path(p['distill_result'])/'selected_codebook_overlay.pt',map_location='cpu',weights_only=True)
    layout=[]; offset=0; base_bytes=0; scales_bytes=0; books_bytes=0; transform_bytes=0
    for i,name in enumerate(overlay['module_names']):
        bank=torch.load(Path(p['source_root'])/'results/full/banks'/p['initial_arm']/f'{i:04d}.pt',map_location='cpu',weights_only=True)
        enc=bank['encoded']
        if bank['module']!=name: raise RuntimeError('Bank identity differs')
        shape=list(enc['shape']); n=enc['indices'].numel()//2; tiles=(n+p['tile_vectors']-1)//p['tile_vectors']
        layout.append(dict(name=name,shape=shape,vectors=n,tiles=tiles,offset=offset,
                           transform_metadata={k:v for k,v in bank['transform'].items() if not torch.is_tensor(v)}))
        offset+=tiles; base_bytes+=n
        scales_bytes+=enc['row_scales'].numel()*2
        books_bytes+=enc['books'].numel()*2
        transform_bytes+=sum(v.numel()*v.element_size() for v in bank['transform'].values() if torch.is_tensor(v))
        if i%100==0: print('PREPARE_BANKS',i,614,flush=True)
    if len(layout)!=614: raise RuntimeError('Missing target Linears')
    ref=json.loads((Path(p['latent_root'])/'reference_summary.json').read_text())
    # A common explicit byte schema includes fields and per-module padding metadata.
    metadata=dict(format=p['protocol'],tile_vectors=p['tile_vectors'],vector_dim=4,books=[2,256,4],
                  index_dtype='uint8',book_scale_dtype='bfloat16',tag_bits=2,tag_meaning={'0':'none','1':'even slots','2':'all slots'},layout=layout)
    metadata_raw=json.dumps(metadata,separators=(',',':')).encode()
    (root/'encoding_metadata.json').write_bytes(metadata_raw)
    metadata_bytes=len(metadata_raw)
    budget=dict(base_index_bytes=base_bytes,residual_index_bytes=offset*p['tile_vectors']//2,
                allocation_tag_bytes=(2*offset+7)//8,codebook_bytes=books_bytes,scale_bytes=scales_bytes,
                transform_tensor_bytes=transform_bytes,layout_metadata_bytes=metadata_bytes)
    budget['total_bytes']=sum(budget.values())
    save(root/'layout.json',dict(protocol=p['protocol'],modules=layout,total_tiles=offset,budget=budget))
    result=receipt('prepare',status='complete',target_linears=614,context_count=60,total_tiles=offset,
                   model_forward_count=0,latent_weight=ref['latent_weight'],budget=budget)
    save(root/'prepare_summary.json',result)


if __name__=='__main__': main()
