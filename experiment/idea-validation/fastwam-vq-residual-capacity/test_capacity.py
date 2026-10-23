"""Small mathematical/gradient checks; executed only in CPU allocations."""
from common import guard


def self_check():
    guard()
    import torch
    from torch.utils.checkpoint import checkpoint
    from allocation import ranked_levels,keep_mask,vectors,tile_sum,normalized_positive,module_matched
    for n in (1,2,3,8,9):
        levels=ranked_levels(torch,torch.arange(n,dtype=torch.float32))
        assert int(levels.long().sum())==n
        assert len(keep_mask(torch,levels,n*8,8))==n*8
        assert int(keep_mask(torch,levels,n*8,8).sum())==n*4
    padded=vectors(torch,torch.arange(21,dtype=torch.float32).reshape(3,7))
    assert padded.shape==(6,4) and bool((padded[[1,3,5],3]==0).all())
    assert torch.equal(tile_sum(torch,padded,4),torch.tensor([91.,119.]))
    assert torch.equal(normalized_positive(torch,torch.tensor([-1.,1.,3.])),torch.tensor([0.,.25,.75]))
    template=torch.tensor([2,0,1,0,2,2],dtype=torch.uint8)
    matched=module_matched(torch,template,torch.arange(6,dtype=torch.float32),[{'offset':0,'tiles':3},{'offset':3,'tiles':3}])
    assert torch.equal(matched,torch.tensor([0,1,2,0,2,2],dtype=torch.uint8))
    # The production observer must accumulate repeated calls and multiple backwards.
    from capacity import observed_weight
    class Dummy: pass
    owner=Dummy();owner.objective=None;owner.buffers={}
    Observed=observed_weight(torch,owner)
    w=torch.arange(6,dtype=torch.float32).reshape(2,3).requires_grad_()
    x=torch.tensor([[1.,2.,3.]],requires_grad=True)
    def call(a):return torch.nn.functional.linear(a,Observed.apply(w,'m'))
    y=checkpoint(call,x,use_reentrant=False)+checkpoint(call,2*x,use_reentrant=False)
    for coefficient in (1.,2.):
        owner.buffers={};owner.objective='test'
        (coefficient*y.sum()).backward(retain_graph=True)
        assert torch.equal(owner.buffers['m'],3*coefficient*x.detach().expand(2,3))
        assert w.grad is None
    print('CAPACITY_MATH_AND_OBSERVER_TESTS_PASSED',flush=True)


if __name__=='__main__':self_check()
