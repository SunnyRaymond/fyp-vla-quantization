"""Inspect the trusted checkpoint schema inside a real CPU allocation."""
import hashlib
import json
import os
from pathlib import Path


def main():
    from allocation_guard import require_allocation
    allocation = require_allocation()
    top = Path('/tc1home/UG/yguo017/v100_newangles_ccds')
    asset = top/'tdmpc2_q_coupling_ready5'
    out = top/'artifacts'/os.environ['SLURM_JOB_ID']
    path = asset/'checkpoints/cartpole-balance-1.pt'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != '4919e562d7f22f41a11118d1db1a0ebcb0e2b4681fc7fc594d772f0d5940869b':
        raise ValueError('Checkpoint identity mismatch')
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=False)
    state = payload.get('model', payload)
    rows = {key: {'shape':list(value.shape), 'dtype':str(value.dtype)} if torch.is_tensor(value) else {'type':type(value).__name__} for key,value in state.items()}
    report = {'allocation':allocation, 'checkpoint_sha256':digest, 'top_keys':list(payload), 'state':rows, 'model_constructed':False, 'inference':False}
    text = json.dumps(report, indent=2)
    if len(text.encode()) > 60000:
        raise ValueError('Schema report exceeds small artifact bound')
    (out/'checkpoint_schema.json').write_text(text)
    for name in ('layers','world_model','init'):
        source = asset/f'source/tdmpc2/common/{name}.py'
        data = source.read_bytes()
        if len(data)>60000:
            raise ValueError('Source exceeds text bound')
        (out/f'current_{name}.txt').write_bytes(data)
    print(json.dumps({'state_keys':len(rows), 'status':'complete'}))


if __name__ == '__main__':
    main()
