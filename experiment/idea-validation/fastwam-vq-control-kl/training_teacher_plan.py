"""Identify eligible teacher-cache reuses without loading a model or its outputs."""
import os
from pathlib import Path

from aggregate import completed
from control_common import guard, read, save


def same_observations(np, left, right):
    return (set(left) == set(right)
            and all(left[key].dtype == right[key].dtype
                    and left[key].shape == right[key].shape
                    and np.array_equal(left[key], right[key]) for key in left))


def load_observation(np, path):
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key].copy() for key in stored.files}


def main():
    guard()
    import numpy as np
    toy = {'camera': np.zeros((2, 2), dtype=np.uint8)}
    assert same_observations(np, toy, {'camera': toy['camera'].copy()})
    assert not same_observations(np, toy, {'camera': np.ones((2, 2), dtype=np.uint8)})
    assert not same_observations(np, toy, {'camera': np.zeros((2, 2), dtype=np.float32)})
    assert not same_observations(np, toy, {})
    here = Path(__file__).resolve().parent
    p, config = read(here / 'protocol.json'), read(here / 'training_data_protocol.json')
    root = Path(p['remote_root'])
    prepared = completed(root, 'training_pairs')
    definition = read(root / config['dataset_subdir'] / 'definition.json')
    if (prepared.get('config') != config or definition.get('config') != config
            or definition.get('parent_config') != p
            or sorted(row['case_id'] for row in definition['cases']) != config['case_ids']):
        raise ValueError('Prepared original-C80 training inputs differ')
    diagnostic = read(root / 'data/calibration_pairs/definition.json')
    sources = {row['case_id']: row for row in diagnostic['cases']}
    receipts = {shard: completed(root, f'calibration_{shard}') for shard in (0, 1)}
    records = []
    for case in definition['cases']:
        case_id, shard = case['case_id'], case['shard']
        old_case = sources[case_id]
        new_meta = read(root / config['dataset_subdir'] / str(case_id) / 'metadata.json')
        old_meta = read(root / 'data/calibration_pairs' / str(case_id) / 'metadata.json')
        if (case['row'] != old_case['row'] or case['shard'] != old_case['shard']
                or case['clean_observation'] != old_case['clean_observation']
                or case['clean_metadata'] != old_case['clean_metadata']
                or new_meta['sampler_seed'] != old_meta['sampler_seed']
                or new_meta['description'] != old_meta['description']
                or new_meta['training_use'] is not True or old_meta['training_use'] is not False):
            raise ValueError('Same-context/prompt/seed training-to-diagnostic binding differs')
        equal = same_observations(
            np, load_observation(np, new_meta['noisy_observation']),
            load_observation(np, old_meta['noisy_observation']))
        directory = root / 'results' / f'calibration_{shard}' / 'records'
        clean_source = directory / f'{case_id}_bf16_clean.pt'
        noisy_source = directory / f'{case_id}_bf16_noisy.pt'
        if not clean_source.is_file() or not noisy_source.is_file():
            raise FileNotFoundError('Completed calibration teacher source missing')
        records.append({
            'case_id': case_id, 'shard': shard,
            'training_metadata': str(root / config['dataset_subdir'] / str(case_id) / 'metadata.json'),
            'source_calibration_pbs_jobid': receipts[shard]['pbs_jobid'],
            'clean_teacher_reuse_eligible': True,
            'clean_teacher_source': str(clean_source),
            'noisy_observations_exactly_equal': equal,
            'noisy_teacher_reuse_eligible': equal,
            'noisy_teacher_source': str(noisy_source) if equal else None,
            'new_noisy_teacher_query_required': not equal})
    eligible = sum(row['noisy_teacher_reuse_eligible'] for row in records)
    summary = {
        'protocol': p['protocol'], 'unit': 'training_teacher_plan',
        'pbs_jobid': os.environ['PBS_JOBID'], 'complete': True,
        'training_data_protocol': config['protocol'], 'contexts': len(records),
        'source_training_pairs_job': prepared['pbs_jobid'],
        'eligible_clean_reuses': len(records), 'eligible_noisy_reuses': eligible,
        'planned_new_teacher_queries': len(records) - eligible,
        'new_model_queries': 0, 'training_updates': 0,
        'heldout_inputs_used': False, 'evaluation_trajectory_inputs_used': False,
        'records': records,
        'scope': 'raw-array/prompt/seed reuse eligibility only; actual teacher tensor/input-identity binding must be verified when consumed; no model outputs loaded or queries run'}
    save(root / 'results/training_teacher_plan/summary.json', summary)


if __name__ == '__main__':
    main()
