"""Called only after allocation_guard.sh inside the CPU preparation allocation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

p = argparse.ArgumentParser()
p.add_argument('--base', type=Path, required=True)
args = p.parse_args()
base = args.base.resolve()
subprocess.run(['bash', str(base/'control/allocation_guard.sh')], check=True)
root = base/'modelroot'
for name, dest, expected in [('outputs.zip','checkpoints',953204628),('wall_single.zip','data',1668205895)]:
    path = root/'downloads'/name
    if path.stat().st_size != expected:
        raise ValueError('Archive size does not match recorded source: '+name)
    with zipfile.ZipFile(path) as archive:
        members = [info for info in archive.infolist() if not info.filename.startswith('__MACOSX/')]
        if name == 'outputs.zip':
            members = [info for info in members if '/wall_single/' in '/'+info.filename or info.filename.rstrip('/') in ('outputs','outputs/wall_single')]
        for info in members:
            target = (root/dest/info.filename).resolve()
            if not target.is_relative_to((root/dest).resolve()):
                raise ValueError('Unsafe archive path')
            archive.extract(info,root/dest)  # Fully reads and checks each extracted member CRC.
        print('Extracted/CRC checked', name, len(members), flush=True)
source = root/'source'
(source/'env/__init__.py').write_text('from gym.envs.registration import register\nregister(id="wall", entry_point="env.wall.wall_env_wrapper:WallEnvWrapper", max_episode_steps=300, reward_threshold=1.0)\n')
dino = source/'models/dino.py'
text = dino.read_text()
assert '"facebookresearch/dinov2", name' in text
dino.write_text(text.replace('"facebookresearch/dinov2", name','"facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8", name'))
sys.path.insert(0,str(source))
os.environ['DATASET_DIR'] = str(root/'data')
import torch
import hydra
from omegaconf import OmegaConf
cfgpath = root/'checkpoints/outputs/wall_single/hydra.yaml'
original = cfgpath.with_name('hydra.original.yaml')
original.write_bytes(cfgpath.read_bytes())
cfg = OmegaConf.load(original)
cfg.env.dataset.data_path = '${oc.env:DATASET_DIR}/wall_single'
OmegaConf.save(cfg,cfgpath)
encoder = hydra.utils.instantiate(cfg.encoder)  # Cache official encoder in CPU allocation.
checkpoint = root/'checkpoints/outputs/wall_single/checkpoints/model_latest.pth'
payload = torch.load(checkpoint,map_location='cpu')
assert payload['epoch'] == 65
_, trajectories = hydra.utils.call(cfg.env.dataset,num_hist=cfg.num_hist,num_pred=cfg.num_pred,frameskip=cfg.frameskip)
assert len(trajectories['valid']) >= 50
result = {'status':'complete','job_id':os.environ['SLURM_JOB_ID'],'host':os.uname().nodename,
          'torch':torch.__version__,'checkpoint_epoch':int(payload['epoch']),
          'checkpoint_bytes':checkpoint.stat().st_size,'valid_trajectories':len(trajectories['valid']),
          'source_commit':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
          'archive_validation':'size and extracted-member CRC; no SHA inventory',
          'gpu_work_performed':False}
(base/'prepare_summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
