"""Run the official Wall planner with bounded execution and auditable outputs."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', required=True, type=Path)
p.add_argument('--output', required=True, type=Path)
p.add_argument('--n-evals', type=int, default=50)
p.add_argument('--max-mpc', type=int, default=12)
a = p.parse_args()
root=a.root.resolve()
out=a.output.resolve()
out.mkdir(parents=True, exist_ok=True)
os.environ['DATASET_DIR']=str(root/'data')
os.environ['WANDB_MODE']='disabled'
sys.path.insert(0, str(root/'source'))

import numpy as np
import torch
import imageio
from omegaconf import OmegaConf
import plan
from planning.cem import CEMPlanner

torch.set_num_threads(8)
torch.set_grad_enabled(False)
if not torch.cuda.is_available():
    raise RuntimeError('GPU required; refusing accidental CPU evaluation')
torch.cuda.reset_peak_memory_stats()
start=time.monotonic()
original_cem=CEMPlanner.plan
def timed_cem(self,*args,**kwargs):
    torch.cuda.synchronize()
    started=time.monotonic()
    result=original_cem(self,*args,**kwargs)
    torch.cuda.synchronize()
    record={'stage':self.logging_prefix,'seconds':time.monotonic()-started,
            'n_evals':a.n_evals,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
    with (out/'planning_times.jsonl').open('a') as stream:
        stream.write(json.dumps(record)+'\n')
    print('CEM_TIMING',json.dumps(record),flush=True)
    return result
CEMPlanner.plan=timed_cem

original_load=plan.load_model
def load_model(*args, **kwargs):
    model=original_load(*args, **kwargs)
    modes={name:m.training for name,m in model.named_children()}
    model.eval()
    # The decoder is visualization-only in the official evaluator. Save actual
    # environment videos instead of decoding every CEM diagnostic rollout.
    model.decoder=None
    (out/'model_metadata.json').write_text(json.dumps({
        'original_child_training_modes':modes,
        'parameters_without_decoder':sum(t.numel() for t in model.parameters()),
        'gpu':torch.cuda.get_device_name(),
        'torch':torch.__version__,
        'dtype':str(next(model.parameters()).dtype),
    },indent=2))
    return model
plan.load_model=load_model

original_eval=plan.PlanEvaluator.eval_actions
def eval_actions(self, actions, action_len=None, filename='output', save_video=False):
    result=original_eval(self,actions,action_len,filename,save_video)
    logs,successes,obs,states=result
    if filename=='output_final':
        lengths=np.full(len(successes),np.inf) if action_len is None else action_len
        records=[]
        for i,success in enumerate(successes):
            frames=obs['visual'][i]
            steps=actions.shape[1]*self.frameskip if not np.isfinite(lengths[i]) else int(lengths[i])*self.frameskip
            frames=frames[:steps+1]
            goal=self.obs_g['visual'][i,0]
            with imageio.get_writer(str(out/f'case_{i:02d}.mp4'),fps=12) as writer:
                for frame in frames:
                    writer.append_data(np.clip(np.concatenate([frame,goal],axis=1),0,255).astype(np.uint8))
            final_state=states[i,steps]
            records.append({'case_id':i,'env_seed':int(self.seed[i]),
                'success':bool(success),'executed_env_steps':steps,
                'goal_distance':float(np.linalg.norm(self.state_g[i][:2]-final_state[:2])),
                'initial_state':np.asarray(self.state_0[i]).tolist(),
                'goal_state':np.asarray(self.state_g[i]).tolist(),
                'final_state':np.asarray(final_state).tolist()})
        np.savez_compressed(out/'trajectories.npz',states=states,goals=self.state_g,
            normalized_actions=actions.detach().cpu().numpy(),action_lengths=lengths)
        (out/'cases.json').write_text(json.dumps(records,indent=2))
    return result
plan.PlanEvaluator.eval_actions=eval_actions

cfg=OmegaConf.load(root/'source/conf/plan_wall.yaml')
del cfg['hydra']
del cfg['defaults']
cfg.ckpt_base_path=str(root/'checkpoints')
cfg.model_name='wall_single'
cfg.n_evals=a.n_evals
cfg.planner.max_iter=a.max_mpc
cfg.n_plot_samples=0
cfg.saved_folder=str(out)
cfg.wandb_logging=False
OmegaConf.save(cfg,out/'evaluation_config.yaml')
os.chdir(out)
logs=plan.planning_main(OmegaConf.to_container(cfg,resolve=True))
cases=json.loads((out/'cases.json').read_text())
summary={'complete':len(cases)==a.n_evals,'n_evals':len(cases),
    'successes':sum(c['success'] for c in cases),
    'success_rate':sum(c['success'] for c in cases)/len(cases),
    'elapsed_seconds':time.monotonic()-start,
    'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
    'peak_gpu_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
    'max_mpc_rounds':a.max_mpc,'max_env_steps':a.max_mpc*25,
    'source_commit':'0a9492fa12044b852ae9e001cc74604b79c8bb0c',
    'protocol_note':'Bounded official-checkpoint evaluation; 300-step cap from environment registration, not a verified paper Table 1 cap.',
    'official_metrics':logs}
(out/'summary.json').write_text(json.dumps(summary,indent=2,default=lambda x:x.item()))
(out/'SUCCESS').touch()
print(json.dumps(summary,indent=2,default=lambda x:x.item()),flush=True)
