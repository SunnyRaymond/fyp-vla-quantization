"""Small mathematical tests, executed with torch only inside a PBS allocation."""
from common import guard


def main():
    guard()
    import torch
    from common import weight_vectors, vector_mask
    narrow=torch.arange(14).reshape(2,7)
    packed=weight_vectors(torch,narrow)
    assert packed.shape==(4,4) and torch.equal(packed.reshape(2,8)[:,:7],narrow)
    assert torch.equal(vector_mask(torch,narrow.shape,torch.arange(4)),packed.new_tensor([[1,1,1,1],[1,1,1,0],[1,1,1,1],[1,1,1,0]],dtype=torch.bool))
    from stats import cancellation
    v = torch.tensor([[[1., 0.], [-1., 0.]], [[1., 0.], [1., 0.]], [[0., 0.], [0., 0.]]])
    valid = torch.ones((3, 2), dtype=torch.bool)
    c, pair, n = cancellation(torch, v, valid)
    assert torch.equal(c, torch.tensor([1., 0., 0.]))
    assert torch.equal(pair, torch.tensor([-1., 1., 0.]))
    assert torch.equal(n, torch.tensor([2, 2, 0]))
    # Calls must be summed per weight BEFORE measuring cancellation.
    call1, call2 = torch.tensor([[1., 0.], [1., 0.]]), torch.tensor([[0., 0.], [-2., 0.]])
    c, _, _ = cancellation(torch, (call1 + call2)[None], valid[:1])
    assert c.item() == 1
    # Unequal row scales: decoded-space projection onto one shared centroid.
    s, g = torch.tensor([1., 2.]), torch.tensor([[1., 0.], [-1., 0.]])
    dc = -(s[:, None] * g).sum(0) / s.square().sum()
    projected = s[:, None] * dc
    assert torch.allclose((g + projected).T @ s, torch.zeros(2))
    assert projected.square().sum() <= g.square().sum()
    from intervene import make_updates
    books=torch.full((2,256,4),0.125);books[0]=0.5
    module=dict(flat_ids=torch.tensor([0,1]),indices=torch.tensor([[0,0],[0,0]]),
                gradient=torch.tensor([[1.,0,0,0],[-0.5,0,0,0]]),row_ids=torch.tensor([0,1]),
                books=books,row_scales=torch.ones(2,dtype=torch.bfloat16))
    p=dict(sample_seed=17,intervention_step_norms=[0.25,0.5],step_norm_matching_relative_tolerance=0.05)
    updates,receipt=make_updates(torch,{'modules':{'example':module}},None,p)
    assert len(updates)==6
    assert all(v['relative_matching_error']<=0.05 for k,v in receipt.items() if k!='projection')
    assert abs(receipt['projection']['example']['retained_descent_fraction']-0.1)<1e-6
    assert torch.equal(updates['shared_0']['example'][0],updates['shared_0']['example'][1])
    assert not torch.equal(updates['untied_0']['example'][0],updates['untied_0']['example'][1])
    module['valid_coordinates']=torch.tensor([[1,1,1,1],[1,1,1,0]],dtype=torch.bool)
    module['gradient'][:,3]=torch.tensor([1.,0.])
    padded,receipt=make_updates(torch,{'modules':{'example':module}},None,p)
    assert all(v['example'][1,3]==0 for v in padded.values())
    # Exercise the production observer with real decoder backward and checkpoint recomputation.
    from capture import observed_decoder
    from train_banks import DecodedWeight
    from torch.utils.checkpoint import checkpoint
    books=torch.full((2,256,4),0.125,requires_grad=True)
    logs=torch.zeros(2,requires_grad=True)
    active=torch.ones(2)
    indices=torch.tensor([[[0,0],[1,1]],[[0,0],[1,1]]],dtype=torch.uint8)
    dense=torch.full((2,7),0.25,dtype=torch.bfloat16)
    for sampled in (False,True):
        state={'objective':None,'buffers':{},'book_sums':{},'calls':{}}
        ids={'example':torch.tensor([0,2])}
        observed=observed_decoder(torch,DecodedWeight,{books.data_ptr():'example'},state,{'example'},ids,sampled)
        def repeated():
            first=observed.apply(books,logs,active,indices,dense)
            second=observed.apply(books,logs,active,indices,dense)
            return first+2*second
        result=checkpoint(repeated,use_reentrant=False)
        for objective,scale in (('action',1.),('latent',2.)):
            books.grad=logs.grad=None
            state.update(objective=objective,buffers={},book_sums={},calls={})
            (result.float().sum()*scale).backward(retain_graph=objective=='action')
            expected=weight_vectors(torch,torch.full_like(dense,3*scale)).float()
            if sampled:expected=expected[ids['example']]
            assert state['calls']['example']==2
            assert torch.equal(state['buffers']['example'],expected)
            assert torch.equal(state['book_sums']['example'],books.grad)
    print('CHECKPOINT_OBSERVER_TESTS_PASSED',flush=True)
    from fast_a4 import A4Input as BatchedInput
    from train_banks import A4Input as SourceInput
    generator=torch.Generator().manual_seed(161022031)
    for width in (7,129,1536,5120,13824):
        t=dict(input_dim=width,hadamard=True,block=128,scale=torch.linspace(1/16,16,width),
               signs=(torch.arange(width)%2*2-1).float())
        for dtype in (torch.float32,torch.bfloat16):
            data=torch.randn(2,3,width*2,generator=generator,dtype=dtype)[...,::2]
            first=data.detach().requires_grad_();second=data.detach().clone().requires_grad_()
            expected,actual=SourceInput.apply(first,t),BatchedInput.apply(second,t)
            assert torch.equal(expected,actual)
            probe=torch.randn(actual.shape,generator=generator)
            (expected.float()*probe).sum().backward();(actual.float()*probe).sum().backward()
            assert torch.equal(first.grad,second.grad)
    print('BATCHED_A4_STE_TESTS_PASSED',flush=True)
    print('CONFLICT_MATH_TESTS_PASSED', flush=True)


if __name__ == '__main__':
    main()
