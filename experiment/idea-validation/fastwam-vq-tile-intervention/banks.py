"""Patch only the chosen decoded residual slices; keep all other weights exact."""
import json
from pathlib import Path


def apply_tiles(torch,modules,tiles,data,previous,current):
    for tid in previous:
        t=tiles[tid];w=modules[t['module']].weight
        a=t['local_tile']*t['weight_count'];b=a+t['weight_count']
        w.data.view(-1)[a:b].copy_(data[tid]['full'].to(w.device))
    for tid in current:
        t=tiles[tid];w=modules[t['module']].weight
        a=t['local_tile']*t['weight_count'];b=a+t['weight_count']
        w.data.view(-1)[a:b].copy_(data[tid]['base'].to(w.device))


class Banks:
    def __init__(self,runtime,p,plan):
        import torch
        import smooth_vq as core
        from train_banks import A4Input
        self.torch=torch;self.modules=runtime.q.modules;runtime.q.disable()
        if len(self.modules)!=614:raise RuntimeError('614 targets required')
        for v in runtime.model.parameters():v.requires_grad_(False)
        self.original={name:m.weight.detach().cpu().clone() for name,m in self.modules.items()}
        self.forwards={name:m.forward for name,m in self.modules.items()}
        self.full={};self.transforms={};self.data={};self.active=[];self.mode='bf16';self.seen=set()
        total=(614+p['pack_chunk_modules']-1)//p['pack_chunk_modules']
        for i in range(total):
            d=torch.load(Path(p['remote_root'])/f'packed/chunk_{i:02d}.pt',map_location='cpu',weights_only=True,mmap=True)
            if d['protocol']!=p['protocol'] or d['chunk']!=i:raise RuntimeError('Packed source identity differs')
            self.full.update(d['weights']);self.data.update(d['tiles'])
            self.transforms.update({k:{a:b.to('cuda') if torch.is_tensor(b) else b for a,b in v.items()} for k,v in d['transforms'].items()})
        if list(self.full)!=list(self.modules):raise RuntimeError('Packed target names/order differ')
        self.tiles={t['id']:t for t in plan['tiles']}
        if set(self.tiles)!=set(self.data):raise RuntimeError('Selected tile payload differs')
        for name,m in self.modules.items():
            def forward(x,_name=name,_mod=m):
                self.seen.add(_name)
                if self.mode=='bf16':return self.forwards[_name](x)
                return torch.nn.functional.linear(A4Input.apply(x,self.transforms[_name]),_mod.weight,_mod.bias)
            m.forward=forward
        self.activate('full')

    def activate(self,mode):
        self.mode=mode;self.active=[]
        for name,m in self.modules.items():m.weight.data=(self.original if mode=='bf16' else self.full)[name].to('cuda')
        self.seen.clear()

    def patch(self,tile_ids):
        if self.mode!='full':raise RuntimeError('Patch requires full student mode')
        apply_tiles(self.torch,self.modules,self.tiles,self.data,self.active,tile_ids)
        self.active=list(tile_ids);self.seen.clear()

    def close(self):
        for name,m in self.modules.items():m.forward=self.forwards[name]
