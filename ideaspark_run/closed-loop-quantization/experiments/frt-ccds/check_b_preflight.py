"""Bounded compute-only sanity check before spending the formal fit budget."""
import json
import sys
from pathlib import Path
from allocation_guard import require_allocation

require_allocation()
import numpy as np

p = Path(sys.argv[1])
m = json.loads((p / 'stage_b.json').read_text())
assert m['complete'] and len(m['fit_runs']) == 9
assert all(r['reload_equal'] and r['exact_w4'] and r['binary_final'] for r in m['fit_runs'])
with np.load(p / 'stage_b.npz', allow_pickle=False) as a:
    assert a['transport_error'].shape == (11, 11, 12, 196, 404)
    assert a['clean_error'].shape == (11, 12, 196, 404)
    assert np.isfinite(a['transport_error']).all() and np.isfinite(a['clean_error']).all()
    assert np.count_nonzero(a['clean_error'][0]) == 0
    assert np.count_nonzero(a['transport_error'][0]) == 0
    assert np.count_nonzero(a['bank_deltas'][..., a['action_mask']]) == 0
    assert len(set(a['record_keys'].tolist())) == 12
    for key in ('clean', 'transport'):
        expected = np.mean((a[key + '_error'].astype(np.float64) * a['wz']) ** 2, axis=(-2, -1))
        assert np.allclose(expected, a[key + '_mse'], rtol=2e-5, atol=1e-10), key
result = {'engineering_pass': True, 'scientific_result': False, 'reason': 'two-update end-to-end preflight only'}
(p / 'preflight_check.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
