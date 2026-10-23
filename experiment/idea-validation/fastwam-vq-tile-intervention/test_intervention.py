"""Small production-helper checks, run inside the CPU prepare allocation."""


def self_check():
    import torch
    from types import SimpleNamespace
    from design import SCORE_KEYS,select_local,conditions,interaction
    from banks import apply_tiles
    from analyze import rank,predictive
    assert rank([3,1,1])==[2,0.5,0.5]
    stat=predictive([dict(prediction=1,actual_delta=2,noise_floor=1e-12),dict(prediction=-1,actual_delta=1,noise_floor=1e-12)])
    assert stat['informative_sign_n']==2 and stat['sign_correct']==1
    means={k:torch.linspace(-2,2,48) for k in SCORE_KEYS}
    means['action_fisher']=means['action_fisher'].square();means['joint_fisher']=means['joint_fisher'].square()
    tiles=[]
    for i in range(4):
        m=dict(name=('video_' if i<2 else 'action_')+str(i),vectors=12*4096,offset=i*12,shape=[16384,12])
        choices=select_local(torch,means,m,20+i)
        assert len(choices)==7 and len({j for _,j in choices})==7
        for label,j in choices:tiles.append(dict(id=f't{len(tiles):02d}',module=m['name'],local_tile=j))
    cs=conditions(tiles,12)
    assert len(cs)==54 and sum(c['size']==1 for c in cs)==28 and sum(c['size']>1 for c in cs)==24
    assert cs[0]['id']=='baseline_start' and cs[-1]['id']=='baseline_repeat'
    original=torch.arange(32,dtype=torch.bfloat16).reshape(4,8)
    mod=SimpleNamespace(weight=torch.nn.Parameter(original.clone(),requires_grad=False))
    ts={'a':dict(module='m',local_tile=0,weight_count=8),'b':dict(module='m',local_tile=2,weight_count=8)}
    data={k:dict(full=original.flatten()[t['local_tile']*8:t['local_tile']*8+8].clone(),base=torch.full((8,),-1,dtype=torch.bfloat16)) for k,t in ts.items()}
    apply_tiles(torch,{'m':mod},ts,data,[],['a']);assert bool((mod.weight.flatten()[:8]==-1).all())
    apply_tiles(torch,{'m':mod},ts,data,['a'],['b']);assert torch.equal(mod.weight.flatten()[:8],original.flatten()[:8])
    apply_tiles(torch,{'m':mod},ts,data,['b'],[]);assert torch.equal(mod.weight,original)
    q=interaction(16,1,[4,9],16);assert q['interaction']==4 and q['response_nonadditivity_loss']==0
    q=interaction(25,1,[4,9],16);assert q['interaction']==13 and q['quadratic_loss_cross_term']+q['response_nonadditivity_loss']==13
    print('TILE_SELF_CHECK passed',flush=True)


if __name__=='__main__':
    from common import guard
    guard();self_check()
