"""One bounded actor-only PTQ proposal screen; real SLURM/V100 only."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

HELPER_SHA = '3688d97e4c6e2bed4da4ff7aa552605690c3571e49dcf4c026e955b3c642eb11'
PROTOCOL_SHA = '47ae0816f0f9f64a27aff56174d1c394e92f58a3d85a4dc4913bc0683553d7d9'
MANIFEST_SHA = 'b5101d2a4e400da540ce6ec1ed6e2552439cf3194fc756e8ac9336b1094fde42'
PARENT_SHA = '9240979105b919f6051f27261d00ff33652105f962d039728638caabb042d369'
ARMS = ['FP-policy', 'W4-policy', 'random-replacement']
SEEDS = list(range(5217,5225))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    payload = json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    if len(payload.encode('utf-8'))>65536: raise ValueError('JSON exceeds controller limit')
    temp = Path(str(path)+'.writing')
    temp.write_text(payload,encoding='utf-8')
    os.replace(temp,path)


def require(value, message):
    if not value: raise RuntimeError(message)


def run(args, allocation, eng):
    import numpy as np
    import torch
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = args.output
    start = time.monotonic()
    helper_path = out/'tdq_screen.py'
    require(sha(helper_path)==HELPER_SHA,'helper hash')
    spec = importlib.util.spec_from_file_location('tdq_support_helper',helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    gpu = helper._gpu_evidence(torch)
    require(sha(args.protocol)==PROTOCOL_SHA,'protocol hash')
    freeze = read_json(args.input_freeze)
    require(freeze['schema']=='policy-support-input-freeze-v1' and freeze['cpu_job_id']=='64832','input freeze identity')
    require(freeze['manifest_sha256']==MANIFEST_SHA and sha(args.manifest)==MANIFEST_SHA,'manifest freeze')
    require(freeze['protocol_sha256']==PROTOCOL_SHA and freeze['seed_order']==SEEDS,'freeze protocol/seeds')
    manifest = read_json(args.manifest)
    require(manifest['schema']=='policy-prior-support-preparation-v1' and manifest['seed_order']==SEEDS,'input schema/seeds')
    require(manifest['allocation']['verified'] is True and manifest['allocation']['job_id']=='64832','preparation allocation')
    require(manifest['observations']['shape']==[8,5] and manifest['protocol']['sha256']==PROTOCOL_SHA,'input shape/protocol')
    require(manifest['dmcontrol']['env_steps']==0 and manifest['dmcontrol']['render_calls']==0 and manifest['inference_ran'] is False,'reset-only input')
    parent_path = Path(manifest['parent_manifest']['path'])
    require(sha(parent_path)==PARENT_SHA and manifest['parent_manifest']['sha256']==PARENT_SHA,'parent manifest')
    parent = read_json(parent_path)
    obs_path = args.manifest.parent/manifest['observations']['path']
    require(sha(obs_path)==manifest['observations']['sha256'],'input array hash')
    with np.load(obs_path,allow_pickle=False) as packed:
        observations = packed['observations'].copy()
        require(packed['seeds'].tolist()==SEEDS,'raw reset seeds')
        ometa = json.loads(str(packed['metadata_json'].item()))
    require(observations.shape==(8,5) and np.isfinite(observations).all(),'finite observations')
    require(ometa['protocol_sha256']==PROTOCOL_SHA and ometa['env_steps']==0,'raw metadata')
    source = helper._source_identity(args.source_root)
    require(parent['source']['commit']==helper.SOURCE_COMMIT and len(parent['source']['selected_files'])>=10,'source parent contract')
    helper._assert_manifest_source(parent,source)
    checkpoint = helper._checkpoint_identity(args.checkpoint,parent)
    eng.update(gpu=gpu,source=source,checkpoint=checkpoint,
               manifest_sha256=MANIFEST_SHA,parent_manifest_sha256=PARENT_SHA,
               protocol_sha256=PROTOCOL_SHA,helper_sha256=HELPER_SHA,
               input_freeze_sha256=sha(args.input_freeze),observations_sha256=sha(obs_path))
    write_json(out/'engineering.json',eng)
    model, td_math, runtime, cfg = helper._load_model(args.source_root,args.checkpoint,args.config,out,torch)
    # Import the pinned class without constructing its training optimizers.
    import tdmpc2 as td_agent
    require(Path(td_agent.__file__).resolve()==(args.source_root/'tdmpc2/tdmpc2.py').resolve(),'official scorer module path')
    runtime['source_identity']['planner_module'] = helper._module_identity(td_agent)
    runtime['source_identity']['common_scale'] = helper._module_identity(__import__('common.scale',fromlist=['RunningScale']))
    runtime['source_files_hashed_but_not_imported_by_runner'].remove('tdmpc2/tdmpc2.py')
    expected = dict(horizon=3,num_pi_trajs=24,num_samples=512,num_elites=64,iterations=6,
                    temperature=.5,max_std=2,episodic=False,action_dim=1,multitask=False,min_std=.05)
    require(all(getattr(cfg,k)==v for k,v in expected.items()),'resolved planner configuration')
    proxy = SimpleNamespace(cfg=cfg,model=model)
    proxy.discount = td_agent.TDMPC2._get_discount(proxy,cfg.episode_length)
    mu_cfg = copy.copy(cfg)
    mu_cfg.num_samples = 1
    mu_proxy = SimpleNamespace(cfg=mu_cfg,model=model,discount=proxy.discount)
    require(model.cfg is cfg and model.cfg.num_samples==512,'original model configuration unchanged')
    write_json(out/'runtime.json',runtime)
    write_json(out/'input_manifest.json',manifest)
    write_json(out/'parent_manifest.json',parent)
    # Copy exact manifest bytes too: JSON pretty-printing is not an identity copy.
    (out/'input_manifest.json').write_bytes(args.manifest.read_bytes())
    (out/'parent_manifest.json').write_bytes(parent_path.read_bytes())
    modules = {n:m for n,m in model.named_modules() if n.startswith('_pi.') and isinstance(m,torch.nn.Linear)}
    require(set(modules)=={'_pi.0','_pi.1','_pi.2'},'all three actor Linear modules')
    actor_names = sorted(n+'.weight' for n in modules)
    state = model.state_dict()
    other_names = sorted(set(state)-set(actor_names))
    other_before = helper._state_digest(state,other_names,torch)
    all_before = helper._state_digest(state,sorted(state),torch)
    quant = {}
    for name,module in sorted(modules.items()):
        fp = module.weight.detach().clone()
        maximum = fp.abs().amax(dim=1,keepdim=True)
        scale = torch.where(maximum>0,maximum*(1.0/7.0),torch.ones_like(maximum))
        codes = torch.round(fp/scale).clamp(-7,7).to(torch.int8)
        dequant = codes.to(torch.float32)*scale
        require(torch.isfinite(dequant).all().item(),'finite quantized weight')
        quant[name+'.weight'] = dict(fp=fp,scale=scale,codes=codes,dequant=dequant)
    required_gates = ['checkpoint_strict_load','model_eval','no_grad','actor_linear_allowlist_complete',
                      'fp_noop_exact','weight_restore_exact','non_actor_unchanged',
                      'rng_pairing_verified','mu_proxy_num_samples_1','common_scores_close',
                      'quant_readback_exact','original_config_unchanged']
    gates = {k:True for k in required_gates}
    gates['checkpoint_strict_load'] = runtime['load']['strict'] and not runtime['load']['missing_keys'] and not runtime['load']['unexpected_keys']
    gates['model_eval'] = not model.training
    gates['no_grad'] = not torch.is_grad_enabled()
    rng_receipts = []

    def seed(value):
        torch.manual_seed(value)
        torch.cuda.manual_seed_all(value)

    def rng():
        return {'cpu':hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                'cuda':hashlib.sha256(torch.cuda.get_rng_state(0).cpu().numpy().tobytes()).hexdigest()}

    def bind(use_q):
        for name,module in modules.items():
            q = quant[name+'.weight']
            target = q['dequant' if use_q else 'fp']
            module.weight.copy_(target)
            exact = torch.equal(module.weight,target)
            gates['quant_readback_exact' if use_q else 'weight_restore_exact'] &= exact
            q['readback_q' if use_q else 'readback_fp'] = module.weight.detach().cpu().clone()
        gates['non_actor_unchanged'] &= helper._state_digest(model.state_dict(),other_names,torch)==other_before

    def propose(z,random_seed):
        seed(random_seed)
        before = rng()
        zz = z.repeat(24,1)
        acts = []
        for t in range(3):
            action,_ = model.pi(zz,None)
            acts.append(action)
            if t<2: zz=model.next(zz,action,None)
        return torch.stack(acts),{'before':before,'after':rng()}

    def score(z,actions,random_seed,small=False):
        require(all(torch.equal(modules[n[:-7]].weight,record['fp']) for n,record in quant.items()),'FP actor restored before scoring')
        seed(random_seed)
        before = rng()
        count = 1 if small else 512
        value = td_agent.TDMPC2._estimate_value(mu_proxy if small else proxy,z.repeat(count,1),actions,None)
        require(tuple(value.shape)==(count,1),'official scorer output shape')
        require(torch.isfinite(value).all().item(),'nonfinite scorer output; no nan_to_num masking')
        return value,{'before':before,'after':rng()}

    shapes = dict(actions=(8,3,3,512,1),values=(8,3,512),elite_indices=(8,3,64),weights=(8,3,64),
                  mu=(8,3,3,1),std=(8,3,3,1),mu_values=(8,3),masses=(8,3),counts=(8,3))
    raw = {k:np.zeros(v,dtype=np.int64 if k in {'elite_indices','counts'} else np.float32) for k,v in shapes.items()}
    raw.update(observations=observations,completed=np.zeros((8,),dtype=np.bool_),arm_names=np.array(ARMS),reset_seeds=np.array(SEEDS))
    eng.update(runtime_file='runtime.json',resolved_config=runtime['resolved_config'],actor_weight_names=actor_names,
               mu_proxy={'num_samples':1,'output_shape':[1,1],'model_num_samples':512},discount=proxy.discount,
               non_actor_before_sha256=other_before,model_before_sha256=all_before,gates=gates)
    for i in range(8):
        if time.monotonic()-start>args.max_seconds: raise TimeoutError('bounded internal480s budget')
        z = model.encode(torch.from_numpy(observations[i:i+1]).to('cuda:0'),None)
        bind(False)
        fp,fp_rng = propose(z,7501+i)
        noop,noop_rng = propose(z,7501+i)
        gates['fp_noop_exact'] &= torch.equal(fp,noop) and fp_rng==noop_rng
        bind(True)
        qp,q_rng = propose(z,7501+i)
        bind(False)
        gates['rng_pairing_verified'] &= fp_rng==q_rng
        seed(7601+i)
        shared=(torch.randn(3,488,1,device='cuda:0')*2).clamp(-1,1)
        seed(7801+i)
        replacement=(torch.randn(3,24,1,device='cuda:0')*2).clamp(-1,1)
        receipt={'reset_seed':SEEDS[i],'policy_seed':7501+i,'common_seed':7601+i,'replacement_seed':7801+i,
                 'pool_score_seed':7701+i,'mu_score_seed':7901+i,'fp_proposal':fp_rng,'q_proposal':q_rng,
                 'fp_noop':noop_rng,'pool_score':[],'mu_score':[]}
        for arm,prior in enumerate((fp,qp,replacement)):
            actions=torch.cat((prior,shared),dim=1)
            values,vrng=score(z,actions,7701+i)
            idx=torch.topk(values.squeeze(1),64,dim=0).indices
            elite_value,elite_actions=values[idx],actions[:,idx]
            weight=torch.exp(.5*(elite_value-elite_value.max(0).values))
            weight=weight/weight.sum(0)
            mu=(weight.unsqueeze(0)*elite_actions).sum(dim=1)/(weight.sum(0)+1e-9)
            std=((weight.unsqueeze(0)*(elite_actions-mu.unsqueeze(1))**2).sum(dim=1)/(weight.sum(0)+1e-9)).sqrt().clamp(.05,2)
            j,jrng=score(z,mu.unsqueeze(1),7901+i,small=True)
            record=dict(actions=actions,values=values[:,0],elite_indices=idx,weights=weight[:,0],mu=mu,std=std,
                        mu_values=j[0,0],masses=weight[idx<24].sum(),counts=(idx<24).sum())
            for key,val in record.items(): raw[key][i,arm]=val.detach().cpu().numpy()
            receipt['pool_score'].append(vrng)
            receipt['mu_score'].append(jrng)
        gates['rng_pairing_verified'] &= all(x==receipt['pool_score'][0] for x in receipt['pool_score']) and all(x==receipt['mu_score'][0] for x in receipt['mu_score'])
        gates['common_scores_close'] &= bool(np.allclose(raw['values'][i,0,24:],raw['values'][i,1,24:],atol=1e-5,rtol=1e-6) and np.allclose(raw['values'][i,0,24:],raw['values'][i,2,24:],atol=1e-5,rtol=1e-6))
        raw['completed'][i]=True
        rng_receipts.append(receipt)
        eng.update(completed_states=i+1,elapsed_seconds=time.monotonic()-start,gates=gates)
        write_json(out/'engineering.json',eng)
    bind(False)
    gates['original_config_unchanged'] = model.cfg is cfg and cfg.num_samples==512 and cfg.iterations==6 and mu_cfg.num_samples==1
    other_after=helper._state_digest(model.state_dict(),other_names,torch)
    all_after=helper._state_digest(model.state_dict(),sorted(state),torch)
    gates['non_actor_unchanged'] &= other_after==other_before
    gates['weight_restore_exact'] &= all_after==all_before
    raw['metadata_json']=np.array(json.dumps({'schema':'policy-prior-support-raw-v1','reset_seeds':SEEDS,
                      'arm_names':ARMS,'manifest_sha256':MANIFEST_SHA,'protocol_sha256':PROTOCOL_SHA}))
    raw_path=out/'raw_policy_support.npz'
    require(not raw_path.exists(),'raw output already exists')
    with (out/'raw_policy_support.writing').open('wb') as f: np.savez_compressed(f,**raw)
    os.replace(out/'raw_policy_support.writing',raw_path)
    snapshot={n:{k:v.detach().cpu() for k,v in q.items()} for n,q in quant.items()}
    torch.save(snapshot,out/'quant_snapshot.pt')
    write_json(out/'rng.json',rng_receipts)
    torch.cuda.synchronize()
    eng.update(status='complete',completed_states=8,elapsed_seconds=time.monotonic()-start,
               gates=gates,non_actor_after_sha256=other_after,model_after_sha256=all_after,
               raw_sha256=sha(raw_path),quant_snapshot_sha256=sha(out/'quant_snapshot.pt'),
               rng_sha256=sha(out/'rng.json'),runtime_sha256=sha(out/'runtime.json'))
    write_json(out/'engineering.json',eng)
    write_json(out/'summary.json',{'status':'complete','job_id':allocation['job_id'],'completed_states':8,
               'gates':gates,'raw_sha256':eng['raw_sha256'],'elapsed_seconds':eng['elapsed_seconds'],
               'decision':'pending_independent_cpu_verification','scope':'FP score, first internal update only'})


def main():
    parser=argparse.ArgumentParser()
    for key in ['manifest','source-root','checkpoint','config','protocol','input-freeze','output']:
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--max-seconds',type=float,default=480)
    args=parser.parse_args()
    from allocation_guard import require_allocation
    allocation=require_allocation()
    require(args.output.name==str(allocation['job_id']) and args.output.parent.name=='artifacts','numeric job output')
    args.output.mkdir(parents=True,exist_ok=True)
    eng={'status':'running','allocation':allocation,'runner_sha256':sha(__file__),'completed_states':0}
    write_json(args.output/'engineering.json',eng)
    try:
        run(args,allocation,eng)
    except Exception as exc:
        eng.update(status='inconclusive_budget' if isinstance(exc,TimeoutError) else 'implementation_inconclusive',
                   error=f'{type(exc).__name__}: {exc}')
        write_json(args.output/'engineering.json',eng)
        raise


if __name__=='__main__': main()
