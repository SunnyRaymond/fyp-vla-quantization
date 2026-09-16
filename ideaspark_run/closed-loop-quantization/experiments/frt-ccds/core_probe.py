"""Actual V100 preliminary Stage A probe; no fitting or DEV selection."""
from allocation_guard import require_allocation
allocation = require_allocation()
import argparse
import copy
import json
import os
from pathlib import Path
import time
import torch
import smoke_runner as smoke
import frt_core as core

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
started=time.monotonic()
runtime=smoke._runtime(a.root)
identity=smoke._checkpoint_identity(runtime)
assert 'V100' in identity['gpu']
assert identity['recorded_epoch']==65
model=runtime['model']
model.requires_grad_(False)
adapter=core.SourceHistoryAdapter(model)
groups=[g for g in smoke._linear_groups(model) if g['family']=='predictor']
modules=[(m['path'],smoke._module_for_path(model,'predictor',g['index'],m['relative'])) for g in groups for m in g['linear']]
snapshot=core.snapshot_modules(modules)
obs,act,state,info=runtime['dset'][60]
nh=model.num_hist
fs=int(runtime['model_cfg'].frameskip)
length=nh+2
assert len(act)>=length*fs
actions=act[:length*fs].reshape(length,-1).unsqueeze(0).to(runtime['device'])
initial={k:v[:nh*fs:fs].unsqueeze(0).to(runtime['device']) for k,v in obs.items()}
with torch.no_grad():
    history=adapter.encode(initial,actions[:,:nh])
    fp=adapter.one_step(history,actions[:,nh:nh+1])
    _,source=model.rollout(initial,actions)
    source_error=float((fp-source[:,nh:nh+1]).abs().max())
    appended=adapter.append(history,fp)
    fp2=adapter.one_step(appended,actions[:,nh+1:nh+2])
    source2_error=float((fp2-source[:,nh+1:nh+2]).abs().max())
    x_original=history.clone()
    core.materialize_rtn(modules)
    q0=adapter.one_step(history,actions[:,nh:nh+1])
    delta=core.mask_action_delta(q0-fp,adapter.action_mask(fp))
    slot_error=float((core.insert_slot_delta(appended,delta)-adapter.append(history,q0)).abs().max())
    core.restore_modules(modules,snapshot)
    target=core.transport(adapter,appended,delta,actions[:,nh+1:nh+2])
    null=core.transport(adapter,appended,torch.zeros_like(delta),actions[:,nh+1:nh+2])
    target_again=core.transport(adapter,appended,delta,actions[:,nh+1:nh+2])
    checks={'source_one_step_max_abs':source_error,'source_two_step_max_abs':source2_error,
      'slot_equivalence_max_abs':slot_error,'zero_delta_transport_max_abs':float(null.abs().max()),
      'fp_replay_max_abs':float((target-target_again).abs().max()),
      'old_history_unchanged':bool(torch.equal(x_original,history)),
      'action_delta_exact_zero':bool(torch.all(delta[adapter.action_mask(delta)]==0)),
      'q0_delta_norm':float(delta.norm())}
handles=core.register_soft_quantizers(modules)
params=[v for h in handles for v in (h.parametrization.alpha,h.parametrization.scale_raw)]
optimizer=torch.optim.Adam(params,lr=1e-4)
step_times=[]
torch.cuda.reset_peak_memory_stats()
for i in range(8):
    torch.cuda.synchronize()
    tick=time.monotonic()
    with torch.enable_grad():
        clean=adapter.one_step(history,actions[:,nh:nh+1])
        transported=core.transport(adapter,appended,delta,actions[:,nh+1:nh+2])
        loss=core.weighted_mse(clean,fp)+core.weighted_mse(transported,target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in params)
        grad_nonzero=any(bool(torch.any(v.grad!=0)) for v in params)
        optimizer.step()
    torch.cuda.synchronize()
    step_times.append(time.monotonic()-tick)
peak_allocated=torch.cuda.max_memory_allocated()
peak_reserved=torch.cuda.max_memory_reserved()
ledger=core.materialize_hard(handles)
with torch.no_grad():
    hard=adapter.one_step(history,actions[:,nh:nh+1])
checkpoint=a.output/'probe_hard.pt'
core.atomic_torch_save(checkpoint,{'weights':core.snapshot_modules(modules),'ledger':ledger})
core.restore_modules(modules,snapshot)
loaded=torch.load(checkpoint,map_location='cpu')
core.restore_modules(modules,loaded['weights'])
with torch.no_grad():
    reloaded=adapter.one_step(history,actions[:,nh:nh+1])
checks['hard_reload_max_abs']=float((hard-reloaded).abs().max())
checks['soft_alpha_removed']=not any('parametrizations' in n for n,_ in model.named_parameters())
checks['gradient_nonzero']=grad_nonzero
checks['all_finite']=bool(torch.isfinite(hard).all() and torch.isfinite(delta).all())
numerical_pass=all(checks[k]<=1e-5 for k in ['source_one_step_max_abs','source_two_step_max_abs','slot_equivalence_max_abs','zero_delta_transport_max_abs','fp_replay_max_abs','hard_reload_max_abs']) and all(checks[k] for k in ['old_history_unchanged','action_delta_exact_zero','soft_alpha_removed','gradient_nonzero','all_finite'])
result={'stage':'A-preliminary-core-probe','status':'complete','numerical_pass':numerical_pass,
 'allocation':allocation,'runtime_identity':identity,'history_shape':list(history.shape),'checks':checks,
 'backward_step_seconds':step_times,'steady_mean_seconds':sum(step_times[2:])/len(step_times[2:]),
 'peak_allocated_bytes':peak_allocated,'peak_reserved_bytes':peak_reserved,
 'elapsed_seconds':time.monotonic()-started,
 'limitations':['batch1 unweighted probe; not final Wz-weighted fit timing','physical replay and independent verifier remain for full Stage A','no Stage B or mechanism result']}
core.atomic_json(a.output/'core_probe.json',result)
print(json.dumps(result),flush=True)
if not numerical_pass: raise SystemExit(2)
