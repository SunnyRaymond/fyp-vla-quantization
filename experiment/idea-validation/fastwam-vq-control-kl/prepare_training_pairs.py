"""Prepare seeded noisy training views for the frozen original C80 contexts."""
import argparse
import os
from pathlib import Path

from control_common import guard, read, save
from prepare_pairs import corrupt, make_plan


CHILD_PROTOCOL = 'control-kl-training-data-v1'
PARENT_PROTOCOL = 'fastwam-vq-control-kl-v1'
CASE_IDS = list(range(2000, 2080))
DIAGNOSTIC_SEED_BASE = 760000000
DIAGNOSTIC_SEED_STRIDE = 100
DIAGNOSTIC_SEED_OFFSET = 17


def _validate_config(p, original, config):
    source = p['source_relational_protocol']['calibration']
    if p.get('protocol') != PARENT_PROTOCOL:
        raise ValueError('Parent protocol differs from the frozen control-KL protocol')
    if (config.get('protocol') != CHILD_PROTOCOL
            or config.get('parent_protocol') != p['protocol']):
        raise ValueError('Training child protocol is not bound to this parent')
    if config.get('case_ids') != CASE_IDS or source.get('case_ids') != CASE_IDS:
        raise ValueError('Training case IDs must exactly match original C80 calibration IDs')
    if original.get('status') != 'complete':
        raise ValueError('Original calibration plan is incomplete')
    if original.get('calibration_cohorts', {}).get('c80') != CASE_IDS:
        raise ValueError('Original plan C80 IDs differ from the child protocol')
    rows = original.get('inputs', [])
    by_id = {row.get('case_id'): row for row in rows}
    if len(by_id) != len(rows) or not set(CASE_IDS).issubset(by_id):
        raise ValueError('Original plan has missing or duplicate C80 contexts')
    expected = {
        'corruption_seed_base': 860000000,
        'corruption_seed_stride': 100,
        'corruption_seed_offset': 29,
        'training_use': True,
        'dataset_subdir': 'data/training_pairs',
        'sampler_seed_index': 0,
        'new_demonstrations': 0,
        'new_states': 0,
        'heldout_inputs_used': False,
        'evaluation_trajectory_inputs_used': False,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Training child configuration differs from the frozen data protocol')


def make_training_plan(p, original, config):
    """Bind the same balanced C80 plan to independent training-only noise seeds."""
    _validate_config(p, original, config)
    base = make_plan(p, original)
    cases = []
    for item in base['cases']:
        case_id = item['case_id']
        seed = (config['corruption_seed_base']
                + case_id * config['corruption_seed_stride']
                + config['corruption_seed_offset'])
        diagnostic_seed = (DIAGNOSTIC_SEED_BASE
                           + case_id * DIAGNOSTIC_SEED_STRIDE
                           + DIAGNOSTIC_SEED_OFFSET)
        if seed == diagnostic_seed:
            raise ValueError(f'Training corruption seed overlaps diagnostic case {case_id}')
        cases.append({**item, 'protocol': config['protocol'],
                      'parent_protocol': p['protocol'], 'corruption_seed': seed,
                      'training_use': True})
    return {
        'protocol': config['protocol'],
        'parent_protocol': p['protocol'],
        'definition': 'control-kl-training-noise-pairs-v1',
        'parent_config': p,
        'config': config,
        'cases': cases,
        'camera_scope': base['camera_scope'],
        'assignment': base['assignment'],
        'scope': ('One separate sensor-noise training view per original C80 clean context; '
                  'seed independence does not imply distinct pixels or states. No new '
                  'demonstrations or states are added, and no heldout inputs or evaluation '
                  'trajectories are used.'),
        'training_use': True,
        'environment_steps': 0,
    }


def _same_arrays(np, saved, expected):
    return (set(saved.files) == set(expected)
            and all(np.array_equal(saved[key], expected[key]) for key in expected))


def prepare(root, p, config, original, coverage, np, Image, wrapper, job_id,
            numpy_compatibility):
    if (coverage.get('protocol') != p['protocol']
            or coverage.get('calibration_contexts') != 80
            or coverage.get('demonstration_overlap') != 0):
        raise ValueError('Completed coverage inventory does not certify the original C80')
    plan = make_training_plan(p, original, config)
    data = root / config['dataset_subdir']
    definition_path = data / 'definition.json'
    if definition_path.is_file():
        if read(definition_path) != plan:
            raise ValueError('Training pair definition is already frozen with different bindings')
    else:
        save(definition_path, plan)

    completed = []
    new_views = reused_views = new_files = reused_files = 0
    for case in plan['cases']:
        source_metadata = read(Path(case['clean_metadata']))
        row = case['row']
        binding_fields = ('case_id', 'demo_id', 'frame_index', 'sampler_seeds')
        if any(source_metadata.get(key) != row.get(key) for key in binding_fields):
            raise ValueError(f'Original calibration metadata binding differs for {case["case_id"]}')
        sampler_index = config['sampler_seed_index']
        if len(row['sampler_seeds']) <= sampler_index:
            raise ValueError(f'Original sampler seed is missing for {case["case_id"]}')
        if row['sampler_seeds'][sampler_index] != case['sampler_seed']:
            raise ValueError(f'Original sampler seed differs for {case["case_id"]}')
        original_context = {key: source_metadata[key] for key in binding_fields}
        original_context['sampler_seed_index'] = sampler_index
        original_context['sampler_seed'] = row['sampler_seeds'][sampler_index]
        original_context['description'] = source_metadata['description']

        with np.load(case['clean_observation'], allow_pickle=False) as stored:
            clean = {key: stored[key].copy() for key in stored.files}
        noisy = corrupt(np, Image, wrapper, clean, case)
        if any(not np.array_equal(clean[key], noisy[key])
               for key in clean if key != 'agentview_image'):
            raise ValueError(f'Non-agentview observation changed for {case["case_id"]}')
        if np.array_equal(clean['agentview_image'], noisy['agentview_image']):
            raise ValueError(f'Independent sensor perturbation was an identity for {case["case_id"]}')

        directory = data / str(case['case_id'])
        output = directory / 'observation_noisy.npz'
        metadata_path = directory / 'metadata.json'
        record = {**case, **original_context,
                  'description': source_metadata['description'],
                  'original_context': original_context,
                  'noisy_observation': str(output), 'training_use': True,
                  'camera_changed': True}
        output_existed = output.is_file()
        metadata_existed = metadata_path.is_file()
        if output_existed:
            with np.load(output, allow_pickle=False) as saved:
                if not _same_arrays(np, saved, noisy):
                    raise ValueError(f'Existing training view differs for {case["case_id"]}')
        if metadata_existed and read(metadata_path) != record:
            raise ValueError(f'Existing training metadata differs for {case["case_id"]}')

        directory.mkdir(parents=True, exist_ok=True)
        if not output_existed:
            temp = output.with_name(output.name + '.' + str(os.getpid()) + '.tmp')
            with temp.open('wb') as handle:
                np.savez_compressed(handle, **noisy)
            temp.replace(output)
        if not metadata_existed:
            save(metadata_path, record)
        reused_views += int(output_existed)
        new_views += int(not output_existed)
        reused_files += int(output_existed) + int(metadata_existed)
        new_files += int(not output_existed) + int(not metadata_existed)
        completed.append(record)

    summary = {
        'protocol': p['protocol'],
        'parent_protocol': p['protocol'],
        'child_protocol': config['protocol'],
        'unit': 'training_pairs',
        'config': config,
        'pbs_jobid': job_id,
        'complete': len(completed) == 80,
        'contexts': len(completed),
        'records': len(completed),
        'case_ids': [case['case_id'] for case in completed],
        'training_data_prepared': True,
        'training_started': False,
        'training_updates': 0,
        'teacher_queries': 0,
        'new_model_queries': 0,
        'environment_steps': 0,
        'heldout_inputs_used': False,
        'evaluation_trajectory_inputs_used': False,
        'original_clean_inputs_modified': False,
        'new_demonstrations': 0,
        'new_states': 0,
        'new_views': new_views,
        'reused_views': reused_views,
        'new_files': new_files,
        'reused_files': reused_files,
        'scope': plan['scope'],
        'libero_numpy_compatibility': numpy_compatibility,
    }
    save(root / 'results/training_pairs/summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    guard()
    import numpy as np
    from PIL import Image
    from libero_state_loader import ensure_numpy_compat

    numpy_compatibility = ensure_numpy_compat()
    import libero.libero.envs.env_wrapper as wrapper
    from aggregate import completed as completed_receipt

    here = Path(__file__).resolve().parent
    p = read(here / 'protocol.json')
    config = read(here / 'training_data_protocol.json')
    source = p['source_relational_protocol']['calibration']
    root = Path(p['remote_root'])
    original = read(Path(source['plan_file']))
    coverage = completed_receipt(root, 'coverage_inventory')
    summary = prepare(root, p, config, original, coverage, np, Image, wrapper,
                      os.environ['PBS_JOBID'], numpy_compatibility)
    print({'unit': 'training_pairs', 'complete': summary['complete'],
           'contexts': summary['contexts'], 'new_views': summary['new_views'],
           'reused_views': summary['reused_views']}, flush=True)


if __name__ == '__main__':
    main()
