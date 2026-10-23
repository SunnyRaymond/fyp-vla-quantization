"""B32 power-of-two preconditioning with frozen VQ content."""
import math

import torch
import batched_transform
import micro_codebook as micro
from clip_banks import ClipBanks


def exponent_values(shadow, lower, upper):
    return torch.minimum(torch.maximum(shadow.detach().round(), lower), upper).to(torch.int16)


class GroupedA4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ratio, shadow, transform, lower, upper, compensate):
        z=batched_transform.transform_input(x,transform).float()
        k=exponent_values(shadow,lower,upper)
        d=torch.exp2(k.float()).repeat_interleave(32)[:z.shape[-1]]
        v=z/d
        maximum=v.abs().amax(dim=-1,keepdim=True)
        scale=maximum*ratio/7
        safe=torch.where(scale>0,scale,torch.ones_like(scale))
        u=v/safe;q=u.round().clamp(-7,7)
        operand=(q*scale).to(torch.bfloat16)
        if any(ctx.needs_input_grad[:3]):
            inside=v.abs()<=maximum*ratio
            derivative=torch.where(inside,q-u,q)
            ctx.save_for_backward(inside,derivative.to(torch.bfloat16),maximum/7,d,ratio)
            ctx.normalizer=math.sqrt(z.numel()*7)
        ctx.transform,ctx.dtype,ctx.compensate=transform,x.dtype,compensate
        return (operand.float()*d).to(torch.bfloat16) if compensate else operand

    @staticmethod
    def backward(ctx, grad):
        if not ctx.compensate:raise RuntimeError('Folded weights are evaluation-only')
        inside,derivative,base_scale,d,ratio=ctx.saved_tensors
        g=grad.float();term=g*d*derivative.float()*base_scale
        ratio_grad=term.sum()/ctx.normalizer
        width=g.shape[-1];rows=g.numel()//width;groups=(width+31)//32
        # LSQ/STE surrogate: dynamic absmax is detached, as in clipping-only.
        per_channel=term*ratio*math.log(2)
        grouped=torch.nn.functional.pad(per_channel.reshape(rows,width),(0,groups*32-width))
        group_grad=grouped.reshape(rows,groups,32).sum((0,2))
        counts=torch.full((groups,),32,device=g.device,dtype=torch.float32)
        counts[-1]=width-(groups-1)*32
        group_grad=group_grad/(rows*counts*7).sqrt()
        # Uniform scaling is redundant with a dynamic per-token quantizer.
        group_grad=group_grad-group_grad.mean()
        input_grad=batched_transform.inverse_right_r(
            g*inside,ctx.transform['signs'],ctx.transform['hadamard'],ctx.transform['block'])
        return (input_grad/ctx.transform['scale']).to(ctx.dtype),ratio_grad,group_grad,None,None,None,None


class GroupBanks(ClipBanks):
    def __init__(self,runtime,bank_directory,masters,clipping,scaling):
        super().__init__(runtime,bank_directory,masters,'clip_only',clipping)
        self.shadows={};self.bounds={};self.folded=False;self.static={}
        for name,mod in self.modules.items():
            exponents=self.encodings[name]['scale_exponents'].to(torch.int16)
            lower=torch.maximum(torch.full_like(exponents[0],scaling['minimum_exponent']),-exponents.amin(0))
            upper=torch.minimum(torch.full_like(exponents[0],scaling['maximum_exponent']),254-exponents.amax(0))
            if not bool((lower<=0).all() and (upper>=0).all()):raise RuntimeError('Invalid original exponent bounds')
            self.bounds[name]=(lower.float(),upper.float())
            self.shadows[name]=torch.nn.Parameter(torch.zeros(exponents.shape[1],device='cuda'))
            index=self.encodings[name]['indices']
            self.static[name]=(index.data_ptr(),index._version)

            def forward(x,_name=name,_mod=mod):
                self.seen.add(_name)
                qx=GroupedA4.apply(x,self.ratios[_name],self.shadows[_name],self.transforms[_name],
                    *self.bounds[_name],not self.folded)
                return torch.nn.functional.linear(qx,_mod.weight,_mod.bias)
            mod.forward=forward

    def optimizer_groups(self,training,scaling=None):
        if scaling is None:raise RuntimeError('Explicit grouped scaling configuration required')
        return [{'params':list(self.ratios.values()),'lr':training['clip_learning_rate']},
                {'params':list(self.shadows.values()),'lr':scaling['shadow_learning_rate']}]

    @torch.no_grad()
    def project(self):
        super().project()
        for name,value in self.shadows.items():
            lo,hi=self.bounds[name];value.copy_(torch.minimum(torch.maximum(value,lo),hi))

    def scale_snapshot(self):
        return {name:value.detach().cpu().clone() for name,value in self.shadows.items()}

    @torch.no_grad()
    def restore_scales(self,values):
        if set(values)!=set(self.shadows):raise RuntimeError('Grouped scaling module coverage differs')
        for name,value in values.items():
            lo,hi=self.bounds[name]
            if value.shape!=self.shadows[name].shape or not bool(torch.isfinite(value).all()):
                raise RuntimeError('Invalid grouped scaling shadow')
            v=value.to('cuda')
            if not bool(((v>=lo)&(v<=hi)).all()):raise RuntimeError('Scaling checkpoint outside bounds')
            self.shadows[name].copy_(v)

    def actual_exponents(self):
        return {name:exponent_values(value,*self.bounds[name]) for name,value in self.shadows.items()}

    def scaling_summary(self):
        values=torch.cat(list(self.actual_exponents().values()))
        shadow=torch.cat([v.detach() for v in self.shadows.values()])
        return {'modules':len(self.shadows),'groups':values.numel(),'changed_groups':int((values!=0).sum()),
            'negative_groups':int((values<0).sum()),'positive_groups':int((values>0).sum()),
            'changed_modules':sum(bool((v!=0).any()) for v in self.actual_exponents().values()),
            'shadow_abs_max':float(shadow.abs().max()),'folded_weight_evaluation':self.folded}

    def coverage(self):
        result=super().coverage();values=list(self.shadows.values())
        result['group']={'connected':sum(v.grad is not None for v in values),
            'finite':sum(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in values),
            'nonzero':sum(v.grad is not None and bool((v.grad!=0).any()) for v in values)}
        return result

    @torch.no_grad()
    def fold(self):
        if self.folded:raise RuntimeError('Do not fold a grouped bank twice')
        for value in list(self.ratios.values())+list(self.shadows.values()):value.requires_grad_(False)
        for name,k in self.actual_exponents().items():
            encoding=self.stored_encoding(name)
            adjusted=encoding['scale_exponents'].to(torch.int16)+k[None,:]
            if not bool(((adjusted>=0)&(adjusted<=254)).all()):raise RuntimeError('B32 exponent overflow')
            encoding['scale_exponents']=adjusted.to(torch.uint8)
            self.modules[name].weight.data=micro.decode_vq(encoding)
        self.folded=True

    def verify_frozen(self,masters):
        for name,master in masters.items():
            if not torch.equal(self.books[name].detach().cpu(),master['books']) or not torch.equal(
                    self.logs[name].detach().cpu(),master['log_scales']):
                raise RuntimeError('Frozen codebook or row gain changed')
            index=self.encodings[name]['indices']
            if (index.data_ptr(),index._version)!=self.static[name]:raise RuntimeError('Frozen VQ indices changed')
        return {'codebooks_unchanged':True,'indices_unchanged':True,'row_gains_unchanged':True,
                'deterministic_exponent_compensation_only':True}
