"""Frozen within-module sampling and finite intervention identities (PBS only)."""
import random


SCORE_KEYS=('action_damage','latent_damage','reader_damage','action_fisher','joint_fisher')


def select_local(torch,means,module,seed):
    n=module['vectors']//4096
    if n<7 or module['shape'][1]%4: raise ValueError('Need seven complete, unpadded tiles')
    start=module['offset']; x={k:v[start:start+n] for k,v in means.items()}
    used=set(); result=[]
    def pick(label,values,descending):
        order=torch.argsort(values,descending=descending,stable=True).tolist()
        i=next(j for j in order if j not in used)
        used.add(i);result.append((label,i))
    if module['name'].startswith('video_'):
        for key in ('latent_damage','reader_damage'):
            pick(key+'_positive',x[key],True);pick(key+'_negative',x[key],False)
        near=sum(x[k].abs()/x[k].abs().max().clamp_min(1e-30) for k in ('latent_damage','reader_damage'))
        pick('near_zero_latent_reader',near,False)
        pick('action_fisher_high',x['action_fisher'],True)
    else:
        pick('action_positive',x['action_damage'],True);pick('action_negative',x['action_damage'],False)
        pick('near_zero_action',x['action_damage'].abs(),False)
        pick('action_fisher_high',x['action_fisher'],True)
        pick('joint_fisher_next',x['joint_fisher'],True)
    rng=random.Random(seed); order=list(range(n));rng.shuffle(order)
    for i in order:
        if i not in used:
            used.add(i);result.append(('random_'+str(len(result)),i))
        if len(result)==7:break
    return result


def conditions(tiles,seed):
    if len(tiles)!=28 or len({t['id'] for t in tiles})!=28:raise ValueError('28 distinct tiles required')
    out=[dict(id='single_'+t['id'],kind='single',tile_ids=[t['id']],size=1) for t in tiles]
    for m in range(4):
        ids=[t['id'] for t in tiles[m*7:m*7+7]]
        pairs=((0,1),(2,3),(0,5),(4,6)) if m<2 else ((0,1),(0,3),(1,4),(5,6))
        for j,(a,b) in enumerate(pairs):out.append(dict(id=f'pair_m{m}_{j}',kind='within_pair',tile_ids=[ids[a],ids[b]],size=2))
        out.append(dict(id=f'quartet_m{m}',kind='within_quartet',tile_ids=ids[:4],size=4))
    for name,idx in [('cross_video_pair',[0,9]),('cross_action_video_pair',[14,2]),('cross_fisher_four',[5,12,17,24]),('cross_extreme_eight',[0,1,7,8,14,15,21,22])]:
        out.append(dict(id=name,kind='cross_module',tile_ids=[tiles[i]['id'] for i in idx],size=len(idx)))
    if len(out)!=52 or any(len(set(c['tile_ids']))!=c['size'] for c in out):raise RuntimeError('Intervention design differs')
    if len({tuple(sorted(c['tile_ids'])) for c in out})!=52:raise RuntimeError('Duplicate intervention')
    random.Random(seed+100).shuffle(out)
    return [dict(id='baseline_start',kind='baseline',tile_ids=[],size=0)]+out+[dict(id='baseline_repeat',kind='baseline',tile_ids=[],size=0)]


def interaction(actual,baseline,singles,additive_output):
    linear=sum(s-baseline for s in singles)
    joint=actual-baseline
    curvature=additive_output-baseline-linear
    response=actual-additive_output
    return dict(joint_delta=joint,sum_single_delta=linear,interaction=joint-linear,
                quadratic_loss_cross_term=curvature,response_nonadditivity_loss=response)
