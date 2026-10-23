"""Reuse the exact baseline records and fit only eight-teacher action scales."""
import argparse
import math
import os
from pathlib import Path
from anchor_selfcheck import require_allocation
from reuse_reference import read, save, load_reference, reuse_control


def main():
    require_allocation()
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    p = read(parser.parse_args().protocol)
    import torch
    torch.set_num_threads(4)
    check = read(Path(os.environ['ARTIFACTS']) / 'selfcheck.json')
    if not check['passed'] or check['pbs_jobid'] != os.environ['PBS_JOBID']:
        raise ValueError('Same-allocation loss check required')
    cache, _, rows = load_reference(torch, p)
    actions = torch.stack([cache['teacher'][str(c)]['action'][:10, :6].float()
                           for c in p['calibration']['case_ids']])
    stats = {'action': {'mean': actions.mean((0, 1)).tolist(),
                        'scale': actions.std((0, 1), correction=0).clamp_min(p['reader']['std_floor']).tolist()}}
    nmse = []
    for case in p['calibration']['case_ids']:
        z = cache['teacher'][str(case)]['future'].float()[:, :, 1:]
        e = cache['parent'][str(case)]['future'].float()[:, :, 1:] - z
        nmse.append(float(e.square().sum() / z.square().sum().clamp_min(1e-12)))
    mean_nmse = sum(nmse) / len(nmse)
    if not math.isfinite(mean_nmse) or mean_nmse <= 0:
        raise ValueError('Initial C8 NMSE must be positive')
    count = reuse_control(p, rows)
    receipt = read(Path(p['teacher_heldout_directory']) / 'receipt.json')
    if not receipt['complete'] or receipt['queries'] != 80 or receipt['case_ids'] != p['evaluation']['case_ids']:
        raise ValueError('Existing complete heldout teacher required')
    result = {'protocol': p['protocol'], 'unit': 'prepare', 'complete': True, 'passed': True,
              'pbs_jobid': os.environ['PBS_JOBID'], 'heldout_used': False,
              'calibration_case_ids': p['calibration']['case_ids'], 'stats': stats,
              'initial_mean_nmse': mean_nmse, 'baseline_records': count,
              'baseline_checkpoint_updates': 8, 'baseline_new_model_queries': 0,
              'new_teacher_queries': 0, 'model_queries': 0, 'new_closedloop_episodes': 0,
              'coefficient_policy_frozen': True, 'reference_binding': cache['binding']}
    save(Path(p['remote_root']) / 'results/prepare/summary.json', result)
    print('READER_PREPARE_COMPLETE', result, flush=True)


if __name__ == '__main__':
    main()
