"""CPU allocation: independently check B arrays, checkpoint grids, and gates."""
import hashlib
import json
import sys
from pathlib import Path
from allocation_guard import require_allocation

allocation = require_allocation()
import numpy as np
import torch
import verify_b_helpers as v

artifact = Path(sys.argv[1]).resolve()
a_verification = Path(sys.argv[2]).resolve()
output = Path(sys.argv[3]).resolve()
manifest_path = artifact / 'manifest.json'
manifest = v._json(manifest_path)
a_result = v._json(a_verification)
assert a_result['engineering_pass'], 'A has not passed independent verification'
manifest_result = v._manifest_audit(manifest)
b_result = v._stage_b_audit(artifact, manifest, v._manifest_sha256(manifest_path), a_result['stage_a'])
checkpoint_errors = []
checkpoint_rows = []
meta = v._json(artifact / 'stage_b.json')
for fit in meta['fit_runs']:
    key = f"{fit['method']}:{fit['seed']}"
    path = artifact / 'checkpoints' / f"{fit['method']}_seed_{fit['seed']}.pt"
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != fit['checkpoint_sha256']:
        checkpoint_errors.append(key + ': checkpoint hash mismatch')
    payload = torch.load(path, map_location='cpu')
    state = payload['state_dict']
    ledger = payload['hard_ledger']
    if len(ledger) != 24 or any('parametrizations.' in k for k in state):
        checkpoint_errors.append(key + ': soft state or wrong target count')
    weight_hash = hashlib.sha256()
    max_grid_error = 0.0
    for row in ledger:
        weight = state[row['path'] + '.weight'].numpy()
        scale = np.asarray(row['scale'], dtype=weight.dtype).reshape((-1, 1))
        if not np.isfinite(weight).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            checkpoint_errors.append(key + ': nonfinite weights or invalid scale')
            continue
        integer = np.rint(weight / scale)
        error = float(np.max(np.abs(weight - integer * scale)))
        max_grid_error = max(max_grid_error, error)
        if np.any(integer < -7) or np.any(integer > 7) or error != 0.0:
            checkpoint_errors.append(key + ': weight is not exact declared W4 grid')
        weight_hash.update(row['path'].encode())
        weight_hash.update(weight.tobytes())
    if weight_hash.hexdigest() != fit['hard_map_sha256']:
        checkpoint_errors.append(key + ': hard map hash mismatch')
    checkpoint_rows.append({'method_seed': key, 'targets': len(ledger), 'max_grid_error': max_grid_error})
    del payload, state
result = {
    'schema': 'frt-verification-v1', 'allocation': allocation,
    'manifest': manifest_result, 'stage_a_verified_job': '64689',
    'stage_b': b_result,
    'checkpoint_audit': {'pass': not checkpoint_errors, 'errors': checkpoint_errors, 'rows': checkpoint_rows},
    'engineering_pass': bool(manifest_result['pass'] and b_result.get('engineering_pass') and not checkpoint_errors),
    'mechanism_gate_pass': bool(b_result.get('mechanism_pass') and not checkpoint_errors and manifest_result['pass']),
    'status': b_result.get('status') if not checkpoint_errors and manifest_result['pass'] else 'engineering_fail',
}
v._write_json(output, result)
print(json.dumps({key: result[key] for key in ('status', 'engineering_pass', 'mechanism_gate_pass')}), flush=True)
