"""Original BF16 reader plus frozen, selectively decoded residual VQ weights."""
from pathlib import Path
from allocation import vectors, tile_sum, keep_mask


def observed_weight(torch,owner):
    class ObservedWeight(torch.autograd.Function):
        @staticmethod
        def forward(ctx,weight,name):
            ctx.name=name
            return weight.detach().view_as(weight)

        @staticmethod
        def backward(ctx,grad):
            # Accumulate calls before squaring: checkpoints may revisit each Linear.
            if owner.objective is not None:
                value=grad.detach().float().cpu()
                if ctx.name not in owner.buffers: owner.buffers[ctx.name]=torch.zeros_like(value)
                owner.buffers[ctx.name].add_(value)
            return None,None
    return ObservedWeight


class Banks:
    def __init__(self,runtime,p,scoring=False):
        import torch
        import codebook
        import smooth_vq as core
        from train_banks import A4Input
        self.torch,self.runtime,self.p=torch,runtime,p
        runtime.q.disable()
        self.modules=runtime.q.modules
        if len(self.modules)!=614: raise RuntimeError('614 target Linears required')
        for parameter in runtime.model.parameters(): parameter.requires_grad_(False)
        overlay=torch.load(Path(p['distill_result'])/'selected_codebook_overlay.pt',map_location='cpu',weights_only=True)
        if list(self.modules)!=overlay['module_names']: raise RuntimeError('Selected overlay module order differs')
        self.mode='bf16'
        self.original,self.full,self.base,self.transforms,self.layout={},{},{},{},{}
        self.forwards={n:m.forward for n,m in self.modules.items()}
        self.buffers,self.stats,self.seen={}, {},set()
        self.objective=None
        self.collect=False
        static={'magnitude':[],'reconstruction':[]}
        offset=0
        self.observer=observed_weight(torch,self)
        for i,(name,mod) in enumerate(self.modules.items()):
            self.original[name]=mod.weight.detach().cpu().clone()
            b=torch.load(Path(p['source_root'])/'results/full/banks'/p['initial_arm']/f'{i:04d}.pt',map_location='cpu',weights_only=True)
            if b['module']!=name or tuple(b['encoded']['shape'])!=tuple(mod.weight.shape): raise RuntimeError('Source bank differs')
            enc={k:v.to('cuda') if torch.is_tensor(v) else v for k,v in b['encoded'].items()}
            state=overlay['master_state'][name]
            enc['books']=state['books'].to('cuda',dtype=torch.bfloat16)
            enc['row_scales']=(state['log_scales'].to('cuda').exp()*(enc['row_scales']>0)).to(torch.bfloat16)
            dense=codebook.decode_vq(enc)
            full=vectors(torch,dense).cpu()
            idx=enc['indices'].long()
            base=(enc['books'][0].float()[idx[...,0]]*enc['row_scales'].float()[:,None,None]).to(torch.bfloat16)
            base=base.reshape(mod.weight.shape[0],-1)[:,:mod.weight.shape[1]]
            base=vectors(torch,base).cpu()
            self.full[name],self.base[name]=full,base
            self.transforms[name]={k:v.to('cuda') if torch.is_tensor(v) else v for k,v in b['transform'].items()}
            if scoring:
                target=vectors(torch,core.transform_weight(mod.weight.detach(),self.transforms[name]).to(torch.bfloat16)).float().cpu()
                static['magnitude'].append(tile_sum(torch,target.square(),p['tile_vectors']))
                gain=(base.float()-target).square()-(full.float()-target).square()
                static['reconstruction'].append(tile_sum(torch,gain,p['tile_vectors']))
                del target,gain
            n=(len(full)+p['tile_vectors']-1)//p['tile_vectors']
            self.layout[name]=dict(shape=list(mod.weight.shape),vectors=len(full),offset=offset,tiles=n)
            offset+=n

            def forward(x,_name=name,_mod=mod):
                self.seen.add(_name)
                if self.mode=='bf16': return self.forwards[_name](x)
                if self.collect:
                    z=core.transform_input(x.detach(),self.transforms[_name]).float()
                    z=z.reshape(-1,z.shape[-1])
                    if _name not in self.stats:
                        self.stats[_name]=dict(square=torch.zeros(z.shape[-1],device=z.device),q_square=torch.zeros(z.shape[-1],device=z.device),maximum=torch.zeros(z.shape[-1],device=z.device),rows=0)
                    s=self.stats[_name]
                    s['square'].add_(z.square().sum(0)); s['maximum']=torch.maximum(s['maximum'],z.abs().amax(0)); s['rows']+=len(z)
                qx=A4Input.apply(x,self.transforms[_name])
                if self.collect:
                    q=qx.detach().float().reshape(-1,qx.shape[-1])
                    self.stats[_name]['q_square'].add_(q.square().sum(0))
                w=self.observer.apply(_mod.weight,_name) if _mod.weight.requires_grad else _mod.weight
                return torch.nn.functional.linear(qx,w,_mod.bias)
            mod.forward=forward
            if i%100==0: print('CAPACITY_BANKS',i,614,flush=True)
            del b,enc,dense
        self.static={k:torch.cat(v) for k,v in static.items() if v}
        self.total_tiles=offset
        del overlay

    def activate(self,arm,levels=None,grad=False):
        torch=self.torch
        self.mode=arm
        for name,mod in self.modules.items():
            mod.weight.requires_grad_(False)
            if arm=='bf16': w=self.original[name]
            else:
                v=self.full[name]
                if levels is not None:
                    l=self.layout[name]
                    mask=keep_mask(torch,levels[l['offset']:l['offset']+l['tiles']],len(v),self.p['tile_vectors'])
                    v=torch.where(mask[:,None],v,self.base[name])
                shape=self.layout[name]['shape']
                w=v.reshape(shape[0],-1)[:,:shape[1]].contiguous()
            mod.weight.data=w.to('cuda')
            mod.weight.requires_grad_(grad and arm!='bf16')
            mod.weight.grad=None
        self.seen.clear()

    def backward(self,loss,label,retain=False):
        self.buffers={}; self.objective=label
        loss.backward(retain_graph=retain)
        self.objective=None
        result=self.buffers
        self.buffers={}
        return result

    def scores(self,ga,gz,latent_weight):
        torch=self.torch
        out={k:[] for k in ('action_fisher','action_damage','joint_fisher','latent_damage')}
        for name in self.modules:
            delta=self.base[name].float()-self.full[name].float()
            def gradient(data):
                return vectors(torch,data[name]) if name in data else torch.zeros_like(delta)
            a,z=gradient(ga),gradient(gz)
            out['action_fisher'].append(tile_sum(torch,(a*delta).square(),self.p['tile_vectors']))
            out['action_damage'].append(tile_sum(torch,a*delta,self.p['tile_vectors']))
            out['joint_fisher'].append(tile_sum(torch,((a+latent_weight*z)*delta).square(),self.p['tile_vectors']))
            out['latent_damage'].append(tile_sum(torch,z*delta,self.p['tile_vectors']))
        return {k:torch.cat(v) for k,v in out.items()}

    def reader_scores(self,gr):
        torch=self.torch; values=[]
        for name in self.modules:
            if name in gr:
                delta=self.base[name].float()-self.full[name].float()
                values.append(tile_sum(torch,vectors(torch,gr[name])*delta,self.p['tile_vectors']))
            else: values.append(torch.zeros(self.layout[name]['tiles']))
        return torch.cat(values)

    def outliers(self):
        torch=self.torch
        import smooth_vq as core
        output=[]; local=[]
        for name in self.modules:
            l=self.layout[name]
            if name not in self.stats: raise RuntimeError('Missing calibration activation statistics')
            s=self.stats[name]
            ratio=(s['maximum']/(s['square']/s['rows']).sqrt().clamp_min(1e-12)).cpu()
            groups=(l['shape'][1]+3)//4
            q=torch.nn.functional.pad(ratio,(0,(-len(ratio))%4)).reshape(-1,4).mean(1)
            output.append(tile_sum(torch,q.repeat(l['shape'][0]),self.p['tile_vectors']))
            h=(s['q_square']/s['rows']).cpu()
            h=torch.nn.functional.pad(h,(0,(-len(h))%4)).reshape(-1,4).repeat(l['shape'][0],1)
            # Recompute one module at a time instead of retaining24GB of gain tensors.
            target=vectors(torch,core.transform_weight(self.original[name].to('cuda'),self.transforms[name]).to(torch.bfloat16)).float().cpu()
            gain=(self.base[name].float()-target).square()-(self.full[name].float()-target).square()
            local.append(tile_sum(torch,h*gain,self.p['tile_vectors']))
            del target,gain
        self.stats={}
        return dict(activation_outlier=torch.cat(output),local_output=torch.cat(local))

    def close(self):
        for name,mod in self.modules.items(): mod.forward=self.forwards[name]


def differentiable_reader(runtime,datum,seed,future):
    """Use the same raw sampler/checkpoints; injected future stays connected."""
    import torch
    import sampler
    import future_adapter
    model=runtime.model
    patches=[]
    def skip_video(*,latents_video,**_): return torch.zeros_like(latents_video)
    def inject(_prediction,_delta,sample):
        return torch.cat((sample[:,:,:1].detach(),future[:,:,1:]),dim=2)
    future_adapter._patch(model,'_denoise_video',skip_video,patches)
    future_adapter._patch(model.infer_video_scheduler,'step',inject,patches)
    try:
        return sampler.differentiable_action(runtime,datum,seed,checkpointed=True,return_future=True)
    finally: future_adapter._restore(patches)


def parity(torch,actual,expected,p,label):
    d=actual.detach().float().cpu()-expected.float().cpu()
    result=dict(max_abs=float(d.abs().max()),rmse=float(d.square().mean().sqrt()))
    if not bool(torch.isfinite(d).all()) or result['max_abs']>p['parity_max_abs_tolerance'] or result['rmse']>p['parity_rmse_tolerance']:
        raise RuntimeError(f'{label} parity failed: {result}')
    return result
