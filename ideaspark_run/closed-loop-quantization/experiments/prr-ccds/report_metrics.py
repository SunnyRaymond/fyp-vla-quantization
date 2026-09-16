"""Descriptive PRR report from existing arrays; compute allocation only."""
import json
from pathlib import Path
from allocation_guard import require_allocation

allocation = require_allocation()
import numpy as np

root = Path('/tc1home/UG/yguo017/prr_ccds/artifacts/64706')
verification = json.loads((root.parent / '64707/verification.json').read_text())
assert verification['engineering_pass']
with np.load(root / 'raw_final.npz', allow_pickle=False) as z:
    meta = json.loads(str(z['metadata_json'].item()))
    eps = [r['episode_id'] for r in meta['record_rows']]
    order = list(dict.fromkeys(eps))
    wz = z['wz'].astype(np.float64)
    terminal = np.mean((z['terminal_error'].astype(np.float64) * wz)**2, axis=(-3,-2,-1))
    clean = np.mean((z['clean_error'].astype(np.float64) * wz)**2, axis=(-3,-2,-1))
    donor = np.mean((z['donor_recovery_error'].astype(np.float64) * wz)**2, axis=(-3,-2,-1))
    arms = ['q_local','q_recovery','l_local','l_recovery']
    means = {}
    for a, arm in enumerate(arms):
        sl = slice(a*3,a*3+3)
        means[arm] = {'terminal':float(terminal[sl].mean()), 'clean':float(clean[sl].mean()),
                      'donor_recovery':float(donor[sl].mean()),
                      'donors':dict(zip(meta['donors'],donor[sl].mean(axis=(0,2)).tolist()))}
    pairs = {}
    for family, li, ri in [('q',0,3),('l',6,9)]:
        rows = []
        for s in range(3):
            for ep in order:
                ix = [i for i,x in enumerate(eps) if x == ep]
                l, r = float(terminal[li+s,ix].mean()), float(terminal[ri+s,ix].mean())
                rows.append({'seed':1201+s,'episode':ep,'local':l,'recovery':r,'improvement_percent':100*(1-r/l),'direction_improved':r<l})
        pairs[family] = rows
    # Reconcile the report to the already completed independent verifier.
    for family, li, ri in [('q',0,3),('l',6,9)]:
        gate = verification['metrics']['pair_gates'][family]
        assert abs(float(terminal[ri:ri+3].mean())-gate['global_recovery_h2']) < 1e-12
        assert abs(float(terminal[li:li+3].mean())-gate['global_local_h2']) < 1e-12
    usage = {}
    for arm in arms:
        f = json.loads((root / f'methods/{arm}_seed_1201.json').read_text())
        usage[arm] = {k:f.get(k) for k in ('non_target_state_bytes','logical_w4_integer_bytes','scale_bytes')}
out = {'schema':'prr-report-metrics-v1','allocation':dict(allocation),'reconciled_to_verification':True,
       'arm_means':means,'episode_pairs':pairs,'checkpoint_tensor_storage_seed1201':usage,
       'scope':'descriptive_existing_DEV_only_no_new_training_or_gate_changes'}
target = Path(__file__).parent / 'report_metrics.json'
target.write_text(json.dumps(out,indent=2))
print(json.dumps({'status':'complete','output':str(target)}))
