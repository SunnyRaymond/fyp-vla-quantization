"""Run mathematical and cohort checks only within the CPU preparation allocation."""
from common import guard,protocol,contexts
from matching import match,nmse,gate


def main():
    guard()
    import torch
    p=protocol();cs=contexts(p)
    assert len(cs)==60 and sum(c['split']=='test' for c in cs)==44
    torch.manual_seed(20261017)
    t=torch.randn(1,48,3,14,28).bfloat16()
    a=(t.float()+0.2*torch.randn_like(t.float())).bfloat16();a[:,:,:1]=t[:,:,:1]
    for scale in (0.,0.6,1.,1.4):
        b=a.clone();b[:,:,1:]=(t[:,:,1:].double()+scale*(a[:,:,1:].double()-t[:,:,1:].double())).bfloat16()
        c,r=match(torch,a,b,t,p)
        assert abs(nmse(torch,c,t)-nmse(torch,b,t))<=r['matching_tolerance']
        assert torch.equal(c[:,:,:1],t[:,:,:1])
        if scale==0:assert torch.equal(c,t)
        if scale==1:assert torch.equal(c,a)
    b=(t.float()+0.17*torch.randn_like(t.float())).bfloat16();b[:,:,:1]=t[:,:,:1]
    rounded,check=match(torch,a,b,t,p)
    repeated,again=match(torch,a,b,t,p)
    assert check['rounding_correction_count']>0
    assert check['rounding_correction_count']<=p['rounding_correction_max_coordinates']
    assert torch.equal(rounded,repeated) and check==again
    try:match(torch,t,b,t,p)
    except RuntimeError:pass
    else:raise AssertionError('Degenerate nonzero target must fail')
    assert gate({str(i):-1. for i in range(22)})['passed']
    assert not gate({str(i):(-100. if i==0 else 0.01) for i in range(22)})['passed']
    print('MATCH_SELF_CHECK passed',flush=True)


if __name__=='__main__':main()
