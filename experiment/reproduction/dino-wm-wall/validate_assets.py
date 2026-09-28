import json
import os
import sys
import subprocess
from pathlib import Path

root=Path('/scratch/users/ntu/yguo017/dino-wm-wall')
sys.path.insert(0,str(root/'source'))
os.environ['DATASET_DIR']=str(root/'data')
os.environ.setdefault('TORCH_HOME',str(root/'cache/torch'))
subprocess.run([
    '/scratch/users/ntu/yguo017/openvla-oft-vla-eval-libero/venvs/bootstrap/bin/uv',
    'pip','install','--python',sys.executable,'setuptools==70.0.0'
],check=True)
import torch
import hydra
from omegaconf import OmegaConf
import plan

cfgpath=root/'checkpoints/outputs/wall_single/hydra.yaml'
original=cfgpath.with_name('hydra.original.yaml')
if not original.exists():
    original.write_bytes(cfgpath.read_bytes())
cfg=OmegaConf.load(original)
cfg.env.dataset.data_path='${oc.env:DATASET_DIR}/wall_single'
OmegaConf.save(cfg,cfgpath)
print(OmegaConf.to_yaml(cfg),flush=True)
# Download the pretrained encoder and its source on a CPU allocation.
encoder=hydra.utils.instantiate(cfg.encoder)
print('encoder ready',type(encoder).__name__,flush=True)
ckpt=root/'checkpoints/outputs/wall_single/checkpoints/model_latest.pth'
payload=torch.load(ckpt,map_location='cpu')
print('checkpoint keys',list(payload),flush=True)
print('checkpoint epoch',payload.get('epoch'),flush=True)
datasets,trajs=hydra.utils.call(cfg.env.dataset,num_hist=cfg.num_hist,num_pred=cfg.num_pred,frameskip=cfg.frameskip)
print('train/val trajectories',len(trajs['train']),len(trajs['valid']),flush=True)
obs,*_=trajs['valid'][0]
print('sample visual',obs['visual'].shape,flush=True)
summary={'checkpoint_bytes':ckpt.stat().st_size,'epoch':payload.get('epoch'),
    'keys':list(payload),'num_hist':cfg.num_hist,'frameskip':cfg.frameskip,
    'train_trajectories':len(trajs['train']),'val_trajectories':len(trajs['valid'])}
(root/'asset_validation.json').write_text(json.dumps(summary,indent=2))
(root/'VALIDATED').touch()
