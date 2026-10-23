"""Same continuous B32 forward, with the dynamic-absmax chain in backward."""
import math

import torch
import batched_transform
from continuous_banks import ContinuousBanks, channel_scales, compensated_weight, quantized_input


def coupled_surrogate(g,inside,derivative,base_scale,d,ratio,max_sign,tie_count):
    """Return z, ratio and normalized log2-D gradients for the declared LSQ STE."""
    width=g.shape[-1];rows=g.numel()//width;groups=(width+31)//32
    term=g*derivative.float()*base_scale
    ratio_grad=term.sum()/math.sqrt(g.numel()*7)
    support=max_sign.float()/tie_count
    dm=(g*derivative.float()).sum(-1,keepdim=True)*ratio/7
    z_grad=(g*inside+dm*support)/d
    per_channel=term*ratio*math.log(2)
    # D changes the token maximum; assign its scale effect to the actual max group(s).
    per_channel=per_channel-per_channel.sum(-1,keepdim=True)*support.abs()
    padded=torch.nn.functional.pad(per_channel.reshape(rows,width),(0,groups*32-width))
    group_grad=padded.reshape(rows,groups,32).sum((0,2))
    counts=torch.full((groups,),32,device=g.device,dtype=torch.float32)
    counts[-1]=width-(groups-1)*32
    group_grad=group_grad/(rows*counts*7).sqrt();group_grad-=group_grad.mean()
    return z_grad,ratio_grad,group_grad


class CoupledLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,ratio,log2_scales,weight,bias,transform):
        z=batched_transform.transform_input(x,transform).float()
        d=channel_scales(log2_scales,z.shape[-1]);v=z/d
        maximum=v.abs().amax(-1,keepdim=True);scale=maximum*ratio/7
        safe=torch.where(scale>0,scale,torch.ones_like(scale));u=v/safe;q=u.round().clamp(-7,7)
        result=torch.nn.functional.linear((q*scale).to(torch.bfloat16),compensated_weight(weight,log2_scales),bias)
        if any(ctx.needs_input_grad[:3]):
            inside=v.abs()<=maximum*ratio
            derivative=torch.where(inside,q-u,q).to(torch.bfloat16)
            maxima=v.abs()==maximum
            max_sign=torch.where(maxima,v.sign(),torch.zeros_like(v)).to(torch.int8)
            tie_count=maxima.sum(-1,keepdim=True)
            ctx.save_for_backward(inside,derivative,maximum/7,d,ratio,log2_scales,weight,max_sign,tie_count)
        ctx.transform,ctx.dtype=transform,x.dtype
        return result

    @staticmethod
    def backward(ctx,grad):
        inside,derivative,base_scale,d,ratio,log2_scales,weight,max_sign,tie_count=ctx.saved_tensors
        scaled=compensated_weight(weight,log2_scales)
        g=torch.nn.functional.linear(grad,scaled.t()).float()
        z_grad,ratio_grad,group_grad=coupled_surrogate(g,inside,derivative,base_scale,d,ratio,max_sign,tie_count)
        x_grad=batched_transform.inverse_right_r(z_grad,ctx.transform['signs'],
            ctx.transform['hadamard'],ctx.transform['block'])/ctx.transform['scale']
        return x_grad.to(ctx.dtype),ratio_grad,group_grad,None,None,None


class CoupledBanks(ContinuousBanks):
    def __init__(self,runtime,bank_directory,masters,clipping,scaling):
        super().__init__(runtime,bank_directory,masters,clipping,scaling)
        for name,mod in self.modules.items():
            def forward(x,_name=name,_mod=mod):
                self.seen.add(_name)
                if self.cached:
                    operand=quantized_input(x,self.ratios[_name],self.shadows[_name],self.transforms[_name])[0]
                    return torch.nn.functional.linear(operand,_mod.weight,_mod.bias)
                return CoupledLinear.apply(x,self.ratios[_name],self.shadows[_name],_mod.weight,_mod.bias,self.transforms[_name])
            mod.forward=forward

    def scaling_summary(self):
        return {**super().scaling_summary(),'dynamic_absmax_coupling':True,
                'input_absmax_chain':True,'amax_ties':'equal_share','max_support_storage':'int8'}
