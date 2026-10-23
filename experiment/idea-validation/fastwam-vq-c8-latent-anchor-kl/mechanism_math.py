"""Tensor-only geometry and reversible Adam proposals; run inside an allocation."""
import copy
import math
import statistics as st

TERMS = ('action', 'weighted_nmse', 'weighted_kl')


def dot(a, b, indices=None):
    selected = range(len(a)) if indices is None else indices
    return sum(float((a[i].double()*b[i].double()).sum()) for i in selected)


def alignment(a, b, indices=None):
    aa, bb, ab = dot(a,a,indices), dot(b,b,indices), dot(a,b,indices)
    return {'left_norm': math.sqrt(aa), 'right_norm': math.sqrt(bb), 'dot': ab,
            'cosine': ab/math.sqrt(aa*bb) if aa and bb else None}


def group_alignment(a, b):
    return {name: alignment(a,b,indices) for name,indices in
            [('all',None),('books',range(0,len(a),2)),('log_row_gains',range(1,len(a),2))]}


def mean_gradients(records):
    n=len(records)
    return {term:[sum(r['gradients'][term][i] for r in records)/n
                  for i in range(len(records[0]['gradients'][term]))] for term in TERMS}


def mix(gradients, fraction):
    return [a+z+fraction*k for a,z,k in zip(*(gradients[t] for t in TERMS))]


def adam_delta(torch, values, gradients, lr, saved=None):
    parameters=[torch.nn.Parameter(v.clone()) for v in values]
    optimizer=torch.optim.Adam(parameters,lr=lr)
    if saved is not None: optimizer.load_state_dict(copy.deepcopy(saved))
    for p,g in zip(parameters,gradients): p.grad=g.clone()
    torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
    optimizer.step()
    result=[p.detach()-v for p,v in zip(parameters,values)]
    if not all(bool(torch.isfinite(v).all()) for v in result): raise ValueError('Nonfinite Adam delta')
    return result


def case_geometry(records):
    average=mean_gradients(records)
    result={'mean_gradient_alignments':{},'per_case':[],'cross_case':{}}
    pairs=[('action','weighted_nmse'),('action','weighted_kl'),('weighted_nmse','weighted_kl')]
    for left,right in pairs:
        result['mean_gradient_alignments'][left+'__'+right]=group_alignment(average[left],average[right])
    result['mean_gradient_alignments']['base__weighted_kl']=group_alignment(mix(average,0),average['weighted_kl'])
    for r in records:
        row={'case_id':r['case_id'],'alignments':{}}
        for left,right in pairs: row['alignments'][left+'__'+right]=group_alignment(r['gradients'][left],r['gradients'][right])
        row['alignments']['base__weighted_kl']=group_alignment(mix(r['gradients'],0),r['gradients']['weighted_kl'])
        result['per_case'].append(row)
    for term in TERMS:
        pairwise=[alignment(a['gradients'][term],b['gradients'][term])['cosine']
                  for i,a in enumerate(records) for b in records[i+1:]]
        norms=[math.sqrt(dot(r['gradients'][term],r['gradients'][term])) for r in records]
        result['cross_case'][term]={'pairwise_cosines':pairwise,
            'negative_pairs':sum(c is not None and c<0 for c in pairwise),
            'mean_pairwise_cosine':st.mean(c for c in pairwise if c is not None),
            'mean_gradient_norm_over_mean_case_norm':math.sqrt(dot(average[term],average[term]))/st.mean(norms)}
    return average,result


def visible_projection(torch, residual, teacher, prep, config):
    """Raw-Euclidean rowspace of the standardized pooled teacher-anchor map."""
    from anchor_loss import latent_tokens,neighbor_mask
    anchors,ids=latent_tokens(torch,teacher,prep['stats'],config)
    anchors,ids=anchors[0].double(),ids.cpu()
    mask=neighbor_mask(torch,ids)
    scale=torch.tensor(prep['stats']['latent']['scale'],dtype=torch.float64)
    visible=torch.zeros_like(residual,dtype=torch.float64)
    pool=config['pool_size']; offsets=[0]
    for width in config['view_widths']: offsets.append(offsets[-1]+width)
    ranks=[]
    for i,(t,v,r,c) in enumerate(ids.tolist()):
        rows=anchors[mask[i]]
        raw_rows=(rows[1:]-rows[0])/scale
        _,singular,vh=torch.linalg.svd(raw_rows,full_matrices=False)
        rank=int((singular>singular.max()*1e-7).sum());ranks.append(rank)
        ys=slice(r*pool,(r+1)*pool);xs=slice(offsets[v]+c*pool,offsets[v]+(c+1)*pool)
        mean=residual[0,:,t,ys,xs].double().mean((-2,-1))
        projection=(mean@vh[:rank].T)@vh[:rank]
        visible[0,:,t,ys,xs]=projection[:,None,None]
    return visible.float(),ranks


def raw_nmse(torch, future, teacher):
    return float((future[:,:,1:].float()-teacher[:,:,1:].float()).square().sum()/teacher[:,:,1:].float().square().sum())


def matched_repairs(torch, future, teacher, prep, config, tolerance):
    from anchor_loss import teacher_anchor_kl
    residual=future[:,:,1:].float()-teacher[:,:,1:].float()
    visible,ranks=visible_projection(torch,residual,teacher,prep,config)
    null=residual-visible
    energies={name:float(v.double().square().sum()) for name,v in [('visible',visible),('null',null)]}
    total=float(residual.double().square().sum())
    orthogonal=float((visible.double()*null.double()).sum())
    if abs(orthogonal)>1e-5*total: raise ValueError('Raw projection is not orthogonal')
    target_energy=total-.5*min(energies.values())
    denominator=float(teacher[:,:,1:].float().square().sum())
    target_nmse=target_energy/denominator
    repairs={};checks={}
    for name,component in [('visible',visible),('null',null)]:
        lo,hi=0.,1.;best=None
        for _ in range(64):
            alpha=(lo+hi)/2
            candidate=future.clone().float()
            candidate[:,:,1:]-=alpha*component
            candidate=candidate.to(torch.bfloat16).float()
            actual=raw_nmse(torch,candidate,teacher);error=abs(actual-target_nmse)
            if best is None or error<best[0]:best=(error,candidate,actual,alpha)
            if actual>target_nmse:lo=alpha
            else:hi=alpha
        relative=best[0]/target_nmse
        if relative>tolerance:raise ValueError('BF16 NMSE matching exceeded frozen tolerance: '+str(relative))
        repairs[name]=best[1]
        checks[name]={'actual_nmse':best[2],'target_nmse':target_nmse,'relative_error':relative,'alpha':best[3],
            'actual_anchor_kl':float(teacher_anchor_kl(torch,best[1],teacher,prep['stats'],config,prep['temperature']))}
    return repairs,{'raw_total_error_energy':total,'raw_component_energies':energies,
        'raw_null_fraction':energies['null']/total,'orthogonality_relative':abs(orthogonal)/total,
        'local_rank_min':min(ranks),'local_rank_max':max(ranks),'repairs':checks}


def selfcheck(torch,prep,config):
    from anchor_loss import teacher_anchor_kl
    generator=torch.Generator().manual_seed(20261021)
    teacher=torch.randn(1,48,3,14,28,generator=generator)
    error=torch.randn(1,48,2,14,28,generator=generator)*.1
    visible,_=visible_projection(torch,error,teacher,prep,config);null=error-visible
    candidate=teacher.clone();candidate[:,:,1:]+=error
    only_visible=teacher.clone();only_visible[:,:,1:]+=visible
    only_null=teacher.clone();only_null[:,:,1:]+=null
    kl=lambda z:float(teacher_anchor_kl(torch,z,teacher,prep['stats'],config,prep['temperature']))
    assert abs(kl(candidate)-kl(only_visible))<2e-6 and kl(only_null)<1e-6
    assert abs(float((visible.double()*null.double()).sum()))<1e-5*float(error.square().sum())
    values=[torch.tensor([.25,-.5])];gradients=[torch.tensor([.2,-.3])]
    delta=adam_delta(torch,values,gradients,.0005)
    torch.testing.assert_close(delta[0],torch.tensor([-.0005,.0005]),atol=1e-7,rtol=1e-5)
    return {'passed':True,'checks':['raw_projection_orthogonality','exact_anchor_nullspace','fresh_adam_delta']}
