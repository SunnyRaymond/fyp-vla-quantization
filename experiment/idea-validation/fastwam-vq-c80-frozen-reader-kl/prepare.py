"""Reuse existing C80 teachers and action statistics without model forwards."""
import argparse
import os
from pathlib import Path
from anchor_selfcheck import require_allocation
from reuse_reference import read, save, load_reference


def main():
    require_allocation()
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    p = read(parser.parse_args().protocol)
    import torch
    torch.set_num_threads(4)
    check = read(Path(os.environ['ARTIFACTS']) / 'selfcheck.json')
    if not check['passed'] or check['pbs_jobid'] != os.environ['PBS_JOBID']:
        raise ValueError('Same-allocation selfcheck required')
    cache, _, rows = load_reference(torch, p)
    stats = read(p['c80_stats_file'])
    if (not stats['complete'] or stats['heldout_used']
            or stats['calibration_case_ids'] != p['calibration']['case_ids']
            or stats['seed_indices'] != [0] or stats['action'] != p['c80_action_stats']
            or stats['action']['count_per_coordinate'] != 800):
        raise ValueError('Frozen existing C80-only action statistics differ')
    # Recompute the six small moments to confirm compatibility with these teachers.
    actions = torch.stack([cache['teacher'][str(case)]['action'][:10, :6].double()
                           for case in p['calibration']['case_ids']])
    mean = actions.mean((0, 1))
    scale = actions.std((0, 1), correction=0).clamp_min(p['reader']['std_floor'])
    torch.testing.assert_close(mean, torch.tensor(stats['action']['mean'], dtype=torch.float64), rtol=0, atol=1e-12)
    torch.testing.assert_close(scale, torch.tensor(stats['action']['scale'], dtype=torch.float64), rtol=0, atol=1e-12)
    receipt = read(Path(p['teacher_heldout_directory']) / 'receipt.json')
    if (not receipt['complete'] or receipt['queries'] != 80
            or receipt['case_ids'] != p['evaluation']['case_ids']):
        raise ValueError('Complete existing heldout teacher receipt required')
    result = {'protocol': p['protocol'], 'unit': 'prepare', 'complete': True, 'passed': True,
              'pbs_jobid': os.environ['PBS_JOBID'], 'heldout_used': False,
              'calibration_case_ids': p['calibration']['case_ids'],
              'stats': {'action': stats['action']}, 'statistics_source': p['c80_stats_file'],
              'reference_binding': cache['binding'], 'calibration_teachers_reused': 80,
              'heldout_teacher_receipt_queries': 80, 'model_queries': 0,
              'new_teacher_queries': 0, 'new_training_updates': 0, 'new_closedloop_episodes': 0}
    save(Path(p['remote_root']) / 'results/prepare/summary.json', result)
    print('C80_READER_PREPARE_COMPLETE', result, flush=True)


if __name__ == '__main__':
    main()
