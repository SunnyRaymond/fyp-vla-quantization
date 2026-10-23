"""Independent sensor perturbations of all original C80 calibration snapshots."""
import argparse
from collections import Counter
import os
from pathlib import Path
import random
from types import SimpleNamespace

from control_common import guard, read, save
from coverage_inventory import FAMILIES


def make_plan(p, original):
    source = p['source_relational_protocol']['calibration']
    rows = {r['case_id']: r for r in original['inputs']}
    cases = []
    for suite_index, suite in enumerate(original['suite_order']):
        selected = [rows[i] for i in source['case_ids'] if rows[i]['suite'] == suite]
        if len(selected) != 20:
            raise ValueError('Expected all20 original C80 contexts per suite')
        for ordinal, row in enumerate(selected):
            family_index = (ordinal+suite_index) % 5
            severity = (2, 4, 7, 10)[(ordinal//5+family_index+suite_index) % 4]
            cases.append({'case_id': row['case_id'], 'suite': suite,
                'row': row, 'family': FAMILIES[family_index], 'severity': severity,
                'noise_level': family_index*10+severity,
                'corruption_seed': 760000000+row['case_id']*100+17,
                'sampler_seed': row['sampler_seeds'][0], 'shard': ordinal % 2,
                'clean_observation': str(Path(source['root'])/'data'/row['observation_file']),
                'clean_metadata': str(Path(source['root'])/'data'/row['metadata_file'])})
    if len(cases) != 80 or {r['case_id'] for r in cases} != set(source['case_ids']):
        raise ValueError('Calibration pairs must retain all original C80 contexts')
    return {'protocol': p['protocol'], 'definition': 'calibration-noise-pairs-v1',
        'cases': cases, 'camera_scope': 'agentview only; wrist and all other observation arrays unchanged',
        'assignment': 'per suite: four contexts per corruption family, severities2/4/7/10 once each; independent deterministic seeds',
        'scope': 'calibration-only response diagnosis; original snapshots unchanged; no evaluation states or new development set',
        'training_use': False, 'environment_steps': 0}


def corrupt(np, Image, wrapper, obs, case):
    image = obs['agentview_image']
    if image.dtype != np.uint8:
        image = (image*255).astype(np.uint8)
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    try:
        np.random.seed(case['corruption_seed'])
        random.seed(case['corruption_seed'])
        value = getattr(wrapper, case['family'])(Image.fromarray(image), severity=case['severity'])
        if case['family'] != 'motion_blur':
            value = value.astype(np.uint8)
    finally:
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)
    value = np.asarray(value)
    if value.shape != image.shape or value.dtype != np.uint8:
        raise ValueError('Corrupted camera shape/dtype differs from actual Plus step')
    return {**obs, 'agentview_image': value}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selfcheck', action='store_true')
    args = parser.parse_args()
    if args.selfcheck:
        import numpy as np
        rows = [{'case_id': 2000+i, 'suite': str(i//20), 'sampler_seeds': [910000000+i],
                 'observation_file': f'inputs/case_{2000+i}.npz', 'metadata_file': f'inputs/case_{2000+i}.json'}
                for i in range(80)]
        original = {'inputs': rows, 'suite_order': [str(i) for i in range(4)]}
        p = {'protocol': 'toy', 'source_relational_protocol': {'calibration': {
             'case_ids': [r['case_id'] for r in rows], 'root': '/toy'}}}
        plan = make_plan(p, original)
        for suite in original['suite_order']:
            selected = [c for c in plan['cases'] if c['suite'] == suite]
            assert Counter(c['family'] for c in selected) == {family: 4 for family in FAMILIES}
            for family in FAMILIES:
                assert sorted(c['severity'] for c in selected if c['family'] == family) == [2, 4, 7, 10]
        clean = {'agentview_image': np.zeros((4, 4, 3), dtype=np.uint8),
                 'robot0_eye_in_hand_image': np.ones((4, 4, 3), dtype=np.uint8)}
        calls = []
        def fake(image, severity):
            calls.append(severity)
            return np.full_like(image, np.random.randint(1, 255))
        case = {**plan['cases'][0], 'family': 'gaussian_blur'}
        rng, python_rng = np.random.get_state(), random.getstate()
        result = corrupt(np, SimpleNamespace(fromarray=lambda a: a),
                         SimpleNamespace(gaussian_blur=fake), clean, case)
        assert calls == [case['severity']] and not np.any(clean['agentview_image'])
        assert np.array_equal(result['robot0_eye_in_hand_image'], clean['robot0_eye_in_hand_image'])
        assert np.array_equal(np.random.get_state()[1], rng[1]) and random.getstate() == python_rng
        print('Calibration pair checks passed: all C80, balanced severities, one corruption call, unchanged wrist/clean input and RNG')
        return
    guard()
    import numpy as np
    from PIL import Image
    from libero_state_loader import ensure_numpy_compat
    numpy_compatibility = ensure_numpy_compat()
    print({'libero_numpy_compatibility': numpy_compatibility}, flush=True)
    import libero.libero.envs.env_wrapper as wrapper
    p = read(Path(__file__).with_name('protocol.json'))
    root = Path(p['remote_root'])
    source = p['source_relational_protocol']['calibration']
    original = read(Path(source['plan_file']))
    if original.get('status') != 'complete':
        raise ValueError('Original calibration plan is incomplete')
    plan = make_plan(p, original)
    data = root/'data/calibration_pairs'
    definition_path = data/'definition.json'
    if definition_path.is_file() and read(definition_path) != plan:
        raise ValueError('Calibration perturbation definition is already frozen')
    save(definition_path, plan)
    completed, reused = [], 0
    for case in plan['cases']:
        metadata = read(Path(case['clean_metadata']))
        if any(metadata[k] != case['row'][k] for k in ('case_id', 'demo_id', 'frame_index', 'sampler_seeds')):
            raise ValueError('Original calibration metadata binding differs')
        with np.load(case['clean_observation'], allow_pickle=False) as stored:
            clean = {k: stored[k].copy() for k in stored.files}
        noisy = corrupt(np, Image, wrapper, clean, case)
        if any(not np.array_equal(clean[k], noisy[k]) for k in clean if k != 'agentview_image'):
            raise ValueError('Sensor perturbation changed a non-agentview observation')
        directory = data/str(case['case_id'])
        output = directory/'observation_noisy.npz'
        row = {**case, 'description': metadata['description'],
               'noisy_observation': str(output), 'training_use': False,
               'camera_changed': not np.array_equal(clean['agentview_image'], noisy['agentview_image'])}
        if not row['camera_changed']:
            raise ValueError('Calibration sensor perturbation was an identity')
        if output.is_file():
            with np.load(output, allow_pickle=False) as stored:
                if set(stored.files) != set(noisy) or any(not np.array_equal(stored[k], noisy[k]) for k in noisy):
                    raise ValueError('Existing calibration pair differs from frozen deterministic perturbation')
            reused += 1
        else:
            directory.mkdir(parents=True, exist_ok=True)
            temp = output.with_suffix('.npz.tmp')
            with temp.open('wb') as handle:
                np.savez_compressed(handle, **noisy)
            temp.replace(output)
        save(directory/'metadata.json', row)
        completed.append(row)
    summary = {'protocol': p['protocol'], 'unit': 'calibration_pairs', 'complete': True,
               'pbs_jobid': os.environ['PBS_JOBID'], 'contexts': len(completed),
               'new_model_queries': 0, 'training_updates': 0, 'environment_steps': 0,
               'original_clean_inputs_modified': False, 'heldout_inputs_used': False,
               'training_pool_modified': False, 'reused_pairs': reused,
               'new_pairs_this_segment': len(completed)-reused,
               'libero_numpy_compatibility': numpy_compatibility, 'records': completed}
    save(root/'results/calibration_pairs/summary.json', summary)
    print({'unit': 'calibration_pairs', 'complete': True, 'contexts': len(completed)}, flush=True)


if __name__ == '__main__':
    main()
