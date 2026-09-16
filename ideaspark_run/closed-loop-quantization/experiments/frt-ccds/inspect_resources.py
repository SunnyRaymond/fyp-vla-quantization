"""Read-only resource integrity inventory, exclusively on a SLURM CPU node."""
from allocation_guard import require_allocation
allocation = require_allocation()
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--base', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
t0 = time.monotonic()
root = a.base / 'modelroot'
checkpoint = root / 'checkpoints/outputs/wall_single/checkpoints/model_latest.pth'
digest = hashlib.sha256()
with checkpoint.open('rb') as f:
    for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
        digest.update(chunk)
modules = {}
for name in ['torch', 'torchvision', 'hydra', 'omegaconf', 'numpy', 'gym', 'decord', 'einops']:
    module = importlib.import_module(name)
    modules[name] = str(getattr(module, '__version__', 'import_ok'))
prior = json.loads((a.base/'prepare_summary.json').read_text())
data_dir = root/'data/wall_single'
data_entries = sorted(x.name for x in data_dir.iterdir())
result = {
    'status': 'complete', 'allocation': allocation,
    'base': str(a.base), 'modules': modules,
    'checkpoint_bytes': checkpoint.stat().st_size,
    'checkpoint_sha256': digest.hexdigest(),
    'source_commit': subprocess.check_output(['git','-C',str(root/'source'),'rev-parse','HEAD'],text=True).strip(),
    'source_modified_files': subprocess.check_output(['git','-C',str(root/'source'),'diff','--name-only'],text=True).splitlines(),
    'prior_preparation': prior, 'data_top_entries': data_entries,
    'filesystem_free_bytes': shutil.disk_usage(a.base).free,
    'quota_note': 'filesystem free space is not personal quota',
    'elapsed_seconds': time.monotonic()-t0,
    'reuse': 'existing venv, checkpoint, patched source, dataset and torch cache; no downloads or changes',
}
a.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
