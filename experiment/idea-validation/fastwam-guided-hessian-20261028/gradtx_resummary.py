"""CPU re-summary of the gradtx diagnostic (ravqc.json gradtx) from the files its GPU job stored: ravqc_worker's
gradtx_summary / gradtx_report on results/ravqc/gradtx/<arm>/{meta.pt, <case>.pt} with the arm's fwrc2r3 training.json
and step.json, for every arm named on the command line. The recomputed parts and per-group sums must equal the GPU
job's summary (same rows, same split seed); summary.json and RESULTS.zh.md are then rewritten with the added fields
(the GPU job keeps pbs_jobid, this job goes to resummary_pbs_jobid). Nothing here runs the model: the runtime modules
a GPU stage imports are stubbed as in test_ravqc.py."""
import math
import os
from pathlib import Path
import sys
import types

import torch

HERE = Path(__file__).resolve().parent
for name, attrs in (('sampler', dict(_require_compute_allocation=lambda: None)),
                    ('lp_worker', dict(CASES=list(range(2000, 2080)), Work=type('Work', (), {}))),
                    ('rotation_bank', dict(stream_for=lambda n: 'video')), ('softbin_loss', {}),
                    ('frozen_reader', dict(_bf16_forward_context=None, _checkpoint_context_fn=None, BF16Reader=None))):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules.setdefault(name, module)
sys.path.insert(0, str(HERE))
import ravqc_worker as rc  # noqa: E402
import worker as wk  # noqa: E402


def same(a, b, path=''):
    if isinstance(a, dict):
        assert set(a) == set(b), (path, sorted(set(a) ^ set(b)))
        for k in a:
            same(a[k], b[k], f'{path}.{k}')
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f'{path}[{i}]')
    elif isinstance(a, float):
        assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-15), (path, a, b)
    else:
        assert a == b, (path, a, b)


def resummarize(results, arm, job, cases):
    """results = <remote_root>/results/ravqc; cases in the order gradtx_final read them (lp.CASES)."""
    root = results / 'gradtx' / arm
    old = wk.read(root / 'summary.json')
    meta = torch.load(root / 'meta.pt', map_location='cpu', weights_only=True)
    rows = [torch.load(root / f'{c}.pt', map_location='cpu', weights_only=True) for c in cases]
    training = wk.read(results / arm / 'training.json')
    step = wk.read(results / arm / 'step.json')
    s = rc.gradtx_summary(meta, rows, [h['cases'] for h in training['history']], step, rc.RC['gradtx'])
    for k in ('parts', 'per_group', 'batch1_check'):
        same(s[k], old[k], k)
    s.update({k: old[k] for k in ('protocol', 'pbs_jobid', 'arm', 'step_job')}, resummary_pbs_jobid=job)
    wk.save(root / 'summary.json', s)
    (root / 'RESULTS.zh.md').write_text(rc.gradtx_report(s), encoding='utf-8')
    return s


def main():
    results = Path(wk.read(HERE / 'protocol.json')['remote_root']) / 'results' / 'ravqc'
    for arm in sys.argv[1:]:
        resummarize(results, arm, os.environ.get('PBS_JOBID'), list(rc.lp.CASES))
        print(f'{arm}: parts / per_group / batch1_check reproduced; per_group_parts added', flush=True)


if __name__ == '__main__':
    main()
