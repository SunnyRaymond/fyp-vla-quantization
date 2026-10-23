"""Read saved stage metrics in a CPU allocation; never load a model or run a forward."""
import json
from pathlib import Path

from accounting import (CALIBRATION_CASES, REMOTE_ROOT, UPSTREAM_JOB,
                        read_json, read_pt, require_allocation,
                        require_completed_upstream)


def main():
    job, host = require_allocation()
    root = Path(REMOTE_ROOT)
    require_completed_upstream(root, UPSTREAM_JOB)
    import torch
    torch.set_num_threads(1)
    protocol = read_json(root / 'protocol.json')
    results = root / 'results'
    progress = read_json(results / 'progress.json')
    parent = Path(protocol['parent_root'])
    reference = read_json(parent / 'results/reference/reference_receipt.json')
    checkpoint = read_pt(torch, parent / 'results/training/rho1.5/checkpoint.pt')
    h_history = checkpoint['run_state']['training_history']
    assert len(h_history) == 8
    del checkpoint

    moments, endpoints = [], []
    for case in CALIBRATION_CASES:
        saved = read_pt(torch, results / 'moments' / f'{case}.pt')
        assert saved['protocol'] == protocol and saved['case_id'] == case
        moments.append({'case_id': case, **saved['metrics']})
        del saved
        endpoint = read_json(results / 'calibration/queries' / f'{case}_0.json')
        assert endpoint['case_id'] == case and endpoint['seed_index'] == 0
        endpoints.append({k: endpoint[k] for k in
                          ('case_id', 'first10motor_mse', 'future_latent_nmse', 'mixed_loss')})

    def mean(rows):
        assert [r['case_id'] for r in rows] == CALIBRATION_CASES
        return {key: sum(r[key] for r in rows) / 8 for key in
                ('first10motor_mse', 'future_latent_nmse', 'mixed_loss')}

    def relative(before, after):
        return {key: 100 * (after[key] / before[key] - 1) for key in
                ('action_mse', 'future_latent_nmse', 'mixed_loss')}

    r0 = progress['rotation_history'][0]
    r8_first = {'action_mse': moments[0]['first10motor_mse'], **moments[0]}
    before_last = progress['vq_history'][-1]
    after_last = {'action_mse': endpoints[-1]['first10motor_mse'], **endpoints[-1]}
    initial = reference['normalization']
    output = {
        'upstream_job': UPSTREAM_JOB, 'postprocessing_job': job, 'host': host,
        'new_model_forwards': 0, 'new_teacher_forwards': 0,
        'h_vq_training_history': h_history,
        'r_training_history': progress['rotation_history'],
        'r_vq_training_history': progress['vq_history'],
        'r8_w16a4_c8': {'per_case': moments, 'mean': mean(moments)},
        'r8_vq8_c8': {'per_case': endpoints, 'mean': mean(endpoints)},
        'h_vq0_c8': initial,
        'h_vs_r_no_vq_case2000': {
            'h_w16a4': r0, 'r8_w16a4': r8_first,
            'relative_change_percent': relative(r0, r8_first),
            'scope': 'one training context, same seed; H is pre-update-1, R is frozen R8'},
        'vq_update8_same_case2070': {
            'before': before_last, 'after': after_last,
            'relative_change_percent': relative(before_last, after_last)},
        'missing': ['complete H-W16A4 C8 endpoint metrics',
                    'H-W16A4 and R8-W16A4 heldout endpoints',
                    'complete R8-VQ0 C8 endpoint metrics',
                    'complete H-VQ8 C8 endpoint metrics'],
    }
    destination = results / 'diagnosis_saved.json'
    temporary = destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    temporary.replace(destination)
    print('SAVED_DIAGNOSIS_COMPLETE ' + str(destination), flush=True)


if __name__ == '__main__':
    main()
