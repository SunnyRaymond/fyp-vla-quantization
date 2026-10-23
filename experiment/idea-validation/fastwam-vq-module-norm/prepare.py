"""Prepare module-matched probes and storage costs in one CPU PBS allocation."""
import json
import os
from pathlib import Path
from common import PROTOCOL, contexts, guard, save


def main():
    guard()
    import torch
    from prepare_norms import make_updates
    from storage import accounting
    from evaluate import key
    torch.set_num_threads(4)
    root = Path(os.environ['ROOT'])
    p = json.loads((root / 'protocol.json').read_text())
    old = Path(p['previous_root'])
    previous = json.loads((old / 'protocol.json').read_text())
    for field in ('calibration_cases', 'selection_cases', 'test_cases', 'evaluation_seed_indices',
                  'selected_epoch', 'sampler', 'initial_arm', 'distill_result', 'future_reader_root'):
        if p[field] != previous[field]:
            raise RuntimeError('Frozen input/checkpoint configuration changed: ' + field)
    directions = torch.load(old / 'results/analysis_exact/directions.pt', weights_only=True, map_location='cpu')
    old_updates = torch.load(old / 'results/analysis_exact/updates.pt', weights_only=True, map_location='cpu')
    exact = json.loads((old / 'results/analysis_exact/analysis.json').read_text())
    if (directions['protocol'] != previous['protocol'] or old_updates['protocol'] != previous['protocol']
            or exact['status'] != 'complete' or not exact['full_member_metrics']):
        raise RuntimeError('Previous exact calibration artifacts are incomplete')
    probes = make_updates(directions, old_updates, exact['norms'], previous, p)
    out = root / 'results/prepare'; out.mkdir(parents=True, exist_ok=True)
    probes.update(protocol=PROTOCOL, source_protocol=previous['protocol'])
    torch.save(probes, out / 'updates.pt')
    save(out / 'norms.json', probes['norms'])
    save(out / 'reuse.json', probes['reused_labels'])
    selection = json.loads((old / 'results/analysis_capture/selection.json').read_text())
    costs = accounting(p, directions, probes, selection, out)
    save(out / 'storage.json', costs)
    all_contexts = contexts(p)
    reused = 0
    for shard in range(4):
        olddir = old / 'results' / f'intervene-{shard}'
        summary = json.loads((olddir / 'summary.json').read_text())
        if (summary['protocol'] != previous['protocol'] or summary['status'] != 'complete'
                or summary['shard'] != shard or summary['row_count'] != [96, 96, 84, 84][shard]):
            raise RuntimeError('Previous intervention shard is incomplete')
        dest = root / 'results' / f'evaluate-{shard}'; dest.mkdir(parents=True, exist_ok=True)
        for c in (v for v in all_contexts if v['shard'] == shard):
            for label, old_label in probes['reused_labels'].items():
                source = olddir / f'{key(c, old_label)}.json'
                row = json.loads(source.read_text())
                if (row['protocol'] != previous['protocol'] or row['context'] != c or row['label'] != old_label
                        or not row['teacher_reader_unquantized'] or not row['future_injection_verified']):
                    raise RuntimeError('Cannot reuse an unverified or different intervention')
                row.update(protocol=PROTOCOL, label=label, allocation_profile=label.split('_')[0], reused=True,
                    source_protocol=previous['protocol'], source_label=old_label, source_file=str(source),
                    reuse_verified_by='CPU bitwise identical support/weights plus same checkpoint, inputs, reader and sampler')
                final = dest / f'{key(c, label)}.json'
                if final.exists() and json.loads(final.read_text()) != row:
                    raise RuntimeError('Existing reused row differs')
                save(final, row); reused += 1
    if reused != 60 * len(probes['reused_labels']) or reused < 120:
        raise RuntimeError('All exact reusable conditions required, including original shared probes')
    result = dict(protocol=PROTOCOL, phase='prepare', status='complete', job_id=os.environ['PBS_JOBID'],
        profiles=p['allocation_profiles'], epsilons=p['intervention_step_norms'], arm_count=12,
        full_context_count=60, final_condition_count=720, reused_condition_count=reused,
        new_condition_count=720-reused, planned_new_queries=2*(720-reused), new_gradient_queries=0,
        selected_modules=list(directions['modules']), selected_codewords=32,
        module_norm_tolerance=p['module_norm_relative_tolerance'],
        pairwise_module_norm_tolerance=p['pairwise_module_norm_relative_tolerance'],
        module_norms_verified=True, reuse_bitwise_verified=True, norms=probes['norms'], storage=costs)
    save(out / 'summary.json', result)
    save(Path(os.environ['ARTIFACTS']) / 'summary.json', result)
    print(f'MODULE_MATCHED_PROBES_COMPLETE reused={reused} new={720-reused}', flush=True)


if __name__ == '__main__':
    main()
