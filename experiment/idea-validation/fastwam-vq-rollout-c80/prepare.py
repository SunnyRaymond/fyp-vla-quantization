"""Collect fixed student-visited observations, then label them with BF16."""
import argparse
import gc
import os
from pathlib import Path
import random

import common


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--phase', choices=('collect', 'teacher'), required=True)
    a.add_argument('--protocol', type=Path, required=True)
    args = a.parse_args()
    common.guard()
    import numpy as np
    import torch
    import sampler
    import runner
    import calibration_source
    import clipping_reference as ref
    from checkpointing import atomic_torch_save
    p = common.validate(common.read(args.protocol))
    root = Path(p['remote_root'])
    rows = common.expected_rows(p)
    source_plan = common.read(p['calibration']['original_plan'])
    original_rows = {r['case_id']: r for r in source_plan['inputs']}
    sources = common.read(root / 'results/sources/summary.json')
    if not sources.get('complete') or sources.get('case_ids') != p['rollout']['visited_ids']:
        raise RuntimeError('All 40 extracted source states are required')
    out = root / 'results/prepare'
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261017)
    state = {'protocol': p['protocol'], 'pbs_jobid': os.environ['PBS_JOBID'],
             'complete': False, 'phase': args.phase, 'completed': 0, 'new_queries_this_process': 0}

    def progress(**values):
        state.update(values)
        common.save(out / 'progress.json', state)
        print('ROLLOUT_PREPARE_PROGRESS ' + __import__('json').dumps(state), flush=True)

    deadline = ref.Deadline()
    progress()
    runtime = None
    try:
        runtime = runner.Runtime(out)
        runtime.model.eval()
        runtime.arm('bf16')
        original_data = Path(p['calibration']['original_plan']).parent
        ref_row = original_rows[2000]
        ref_obs, ref_meta = calibration_source.load_observation(np, original_data, ref_row)

        def query(obs, meta, row, banks=None):
            deadline.check()
            result = ref.query(runtime, sampler, torch, runtime.datum(obs), meta['description'], row['sampler_seeds'][0], banks)
            state['new_queries_this_process'] += 1
            return result

        if args.phase == 'collect':
            from clip_banks import ClipBanks
            import render_reference as render
            from libero.libero import benchmark, get_libero_path
            from libero.libero.envs import OffScreenRenderEnv
            from libero.libero.utils.utils import postprocess_model_xml
            baseline = torch.load(p['baseline_checkpoint'], map_location='cpu', weights_only=True)
            banks = ClipBanks(runtime, p['calibration']['bank_directory'], baseline['module_masters'], 'clip_only', p['clipping'])
            for value in runtime.model.parameters(): value.requires_grad_(False)
            for group in (banks.books, banks.logs, banks.ratios):
                for value in group.values(): value.requires_grad_(False)
            expected = common.read(Path(p['baseline_checkpoint']).with_name('reference_action.json'))
            action, _, _ = query(ref_obs, ref_meta, ref_row, banks)
            parity = float((action.float() - torch.tensor(expected['action'])).abs().max())
            if parity > 1e-6: raise RuntimeError('Collector baseline reference parity failed')
            suites = {name: benchmark.get_benchmark_dict()[name]() for name in ref.SUITES}
            assets = Path(get_libero_path('assets'))
            completed = []
            for case in p['rollout']['visited_ids']:
                row, old = rows[case], original_rows[case]
                target = root / 'data' / row['observation_file']
                metadata_path = root / 'data' / row['metadata_file']
                if target.exists() and metadata_path.exists():
                    common.load_observation(np, p, row)
                    completed.append(case)
                    progress(completed=len(completed), last_case=case, source_parity_max_abs=parity)
                    continue
                deadline.check(120)
                source = root / 'data/source_snapshots' / str(case)
                snapshot = common.read(source / 'source.json')
                if snapshot['plan_row'] != row: raise RuntimeError('Source snapshot row changed')
                flat_state = np.load(source / 'state.npy', allow_pickle=False)
                xml = postprocess_model_xml(snapshot['model_xml'], {})
                xml, _ = render.relocate_libero_asset_paths(xml, assets)
                suite = suites[row['suite']]
                task = suite.get_task(row['task_id'])
                if task.name != row['task_name']: raise RuntimeError('Original task identity changed')
                seed = row['environment_seed']
                random.seed(seed); np.random.seed(seed % 2**32)
                env = OffScreenRenderEnv(bddl_file_name=str(suite.get_task_bddl_file_path(row['task_id'])),
                                        camera_heights=runtime.ev.LIBERO_ENV_RESOLUTION,
                                        camera_widths=runtime.ev.LIBERO_ENV_RESOLUTION)
                try:
                    env.seed(seed); env.reset(); env.reset_from_xml_string(xml); env.sim.reset()
                    restored = env.set_init_state(flat_state)
                    obs, meta = calibration_source.load_observation(np, original_data, old)
                    checks = {}
                    for key in render.REQUIRED_OBSERVATIONS:
                        difference = float(np.max(np.abs(np.asarray(restored[key], dtype=np.float64) - obs[key].astype(np.float64))))
                        checks[key] = difference
                        tolerance = 0 if key.endswith('_image') else 1e-6
                        if difference > tolerance:
                            raise RuntimeError(f'Restored source observation differs: {case}/{key}={difference}')
                    action, _, counts = query(obs, meta, row, banks)
                    commands = runtime.command(action)
                    done, executed, visited = False, 0, restored
                    for command in commands[:10]:
                        visited, _, done, _ = env.step(command.tolist())
                        executed += 1
                        if done: break
                    arrays = {str(k): np.asarray(v) for k, v in visited.items()
                              if np.asarray(v).dtype.kind in 'biuf'}
                    if not set(render.REQUIRED_OBSERVATIONS) <= arrays.keys() or any(not np.isfinite(v).all() for v in arrays.values()):
                        raise RuntimeError('Visited observation missing or nonfinite')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.with_suffix('.tmp').open('wb') as stream: np.savez_compressed(stream, **arrays)
                    target.with_suffix('.tmp').replace(target)
                    state_path = target.with_suffix('.simstate.npy')
                    np.save(state_path, env.sim.get_state().flatten(), allow_pickle=False)
                    common.save(metadata_path, {**meta, 'rollout': row['rollout'],
                        'observation_origin': 'fixed student-visited state', 'statepath': str(state_path),
                        'rollout_receipt': {'executed_actions': executed, 'terminal_reached': bool(done),
                            'source_render_max_abs': checks, 'initial_normalized_action': action.tolist(),
                            'executed_commands': commands[:executed].tolist(), 'counts': counts,
                            'pbs_jobid': os.environ['PBS_JOBID']},
                        'observation_arrays': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in arrays.items()}})
                finally:
                    env.close()
                completed.append(case)
                progress(completed=len(completed), last_case=case, source_parity_max_abs=parity)
            common.save(out / 'collection_summary.json', {**state, 'complete': True, 'collector_queries': 40,
                        'visited_contexts': 40, 'case_ids': completed, 'source_parity_max_abs': parity,
                        'render_parity_passed': True, 'collector_training_updates': 0})
        else:
            collection = common.read(out / 'collection_summary.json')
            if not collection.get('complete') or collection['case_ids'] != p['rollout']['visited_ids']:
                raise RuntimeError('Frozen rollout collection incomplete')
            old_teacher = torch.load(common.teacher_path(p, ref_row), map_location='cpu', weights_only=True)
            check_action, check_future, _ = query(ref_obs, ref_meta, ref_row)
            teacher_parity = max(float((check_action.float() - old_teacher['action'].float()).abs().max()),
                                 float((check_future.float() - old_teacher['future'].float()).abs().max()))
            if teacher_parity > 1e-6: raise RuntimeError('BF16 shared-teacher parity failed')
            for index, case in enumerate(p['rollout']['visited_ids']):
                row = rows[case]
                path = common.teacher_path(p, row)
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    saved = torch.load(path, map_location='cpu', weights_only=True)
                    if saved['plan_row'] != row or saved['sampler_seed'] != row['sampler_seeds'][0]:
                        raise RuntimeError('Existing teacher binding changed')
                else:
                    obs, meta = common.load_observation(np, p, row)
                    action, future, counts = query(obs, meta, row)
                    atomic_torch_save(torch, path, {'action': action, 'future': future, 'counts': counts,
                        'plan_row': row, 'sampler_seed': row['sampler_seeds'][0], 'pbs_jobid': os.environ['PBS_JOBID']})
                progress(completed=index + 1, last_case=case, teacher_parity_max_abs=teacher_parity)
            for case in p['rollout']['anchor_ids']:
                saved = torch.load(common.teacher_path(p, rows[case]), map_location='cpu', weights_only=True)
                if saved['plan_row'] != rows[case]: raise RuntimeError('Reused anchor teacher differs')
            plan = {'protocol': p['protocol'], 'status': 'complete', 'inputs': list(rows.values()),
                    'calibration_cohorts': {'c80': p['calibration']['case_ids']},
                    'anchor_ids': p['rollout']['anchor_ids'], 'visited_ids': p['rollout']['visited_ids'],
                    'original_plan': p['calibration']['original_plan'], 'heldout_changed': False}
            common.save(root / 'data/plan.json', plan)
            summary = {'protocol': p['protocol'], 'pbs_jobid': os.environ['PBS_JOBID'], 'complete': True,
                       'contexts': 80, 'visited_contexts': 40, 'anchor_contexts': 40, 'new_teacher_queries': 40,
                       'case_ids': p['calibration']['case_ids'], 'teacher_queries': 80, 'collector_queries': 40,
                       'teacher_seed_indices': [0], 'source_parity_max_abs': collection['source_parity_max_abs'],
                       'teacher_parity_max_abs': teacher_parity, 'render_parity_passed': True,
                       'extra_parity_queries': 2, 'training_updates': 0,
                       'collection_pbsjobid': collection['pbs_jobid']}
            common.save(root / 'results/teacher/receipt.json', summary)
            common.save(out / 'summary.json', summary)
            progress(complete=True, phase='complete', contexts=80)
        return 0
    except TimeoutError as exc:
        progress(complete=False, phase='checkpointed', reason=str(exc))
        return 75
    finally:
        if runtime is not None and runtime.env is not None: runtime.env.close()


if __name__ == '__main__':
    raise SystemExit(main())
