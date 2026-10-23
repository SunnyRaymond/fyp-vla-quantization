"""Equal-byte residual layouts; tensor operations run only inside PBS."""


def vectors(torch, w):
    return torch.nn.functional.pad(w,(0,(-w.shape[1])%4)).reshape(-1,4)


def tile_sum(torch, x, tile):
    if x.ndim==2: x=x.sum(1)
    return torch.nn.functional.pad(x,(0,(-len(x))%tile)).reshape(-1,tile).sum(1)


def ranked_levels(torch, scores):
    if scores.ndim!=1 or not bool(torch.isfinite(scores).all()):
        raise ValueError('Finite one-dimensional scores required')
    n=len(scores)
    order=torch.argsort(scores,descending=True,stable=True)
    levels=torch.zeros(n,dtype=torch.uint8)
    levels[order[:n//2]]=2
    if n%2: levels[order[n//2]]=1
    if int(levels.long().sum())!=n:
        raise RuntimeError('Residual byte budget changed')
    return levels


def keep_mask(torch, levels, vector_count, tile):
    ids=torch.arange(vector_count,device=levels.device)
    tags=levels[ids//tile]
    return (tags==2)|((tags==1)&((ids%2)==0))


def normalized_positive(torch, scores):
    value=scores.clamp_min(0)
    return value/value.sum().clamp_min(torch.finfo(value.dtype).tiny)


def module_matched(torch,template,scores,modules):
    """Change positions while preserving every module's full/half/zero tag counts."""
    result=torch.zeros_like(template)
    for m in modules:
        start,end=m['offset'],m['offset']+m['tiles']
        tags=torch.sort(template[start:end],descending=True).values
        order=torch.argsort(scores[start:end],descending=True,stable=True)
        result[start:end][order]=tags
        if not torch.equal(torch.sort(result[start:end]).values,torch.sort(template[start:end]).values):
            raise RuntimeError('Per-module capacity quota changed')
    return result
