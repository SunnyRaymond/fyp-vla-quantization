"""CPU-only replay of every collected state against its untouched next transition."""
import argparse
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import runner as r
r._require_pbs_allocation()
import numpy as np
import torch
from experiments.libero import libero_utils as ev
from libero.libero import benchmark, get_libero_path

parser = argparse.ArgumentParser()
parser.add_argument('--task-id', type=int, required=True)
parser.add_argument('--data-dir', required=True)
args = parser.parse_args()
out = Path(os.environ['ARTIFACTS'])
assert not torch.cuda.is_available(), 'This preflight requires CPU-only allocation'

def restore_cpu_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())

# CPU replay tests physics/observable restoration only. CUDA/model RNG is still
# checked by the actual GPU score runner, without this override.
r._restore_rng = restore_cpu_rng
suite = benchmark.get_benchmark_dict()['libero_goal']()
task = suite.get_task(args.task_id)
env, description = ev.get_libero_env(task, 256, 42)
r._RUNTIME = SimpleNamespace(ev=ev)
result = {'task_id': args.task_id, 'task': description, 'status': 'running',
          'scope': 'all collected states, native physics replay; no model or CUDA RNG validation',
          'measurements': []}
try:
    initial_states = torch.load(Path(get_libero_path('init_states')) / task.problem_folder / task.init_states_file,
                                map_location='cpu', weights_only=False)
    for episode in range(4):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        env.reset()
        env.set_init_state(initial_states[episode])
        for _ in range(30):
            env.step(ev.get_libero_dummy_action())
        payload = torch.load(r._find_state_file(Path(args.data_dir), args.task_id, episode),
                             map_location='cpu', weights_only=False)
        assert len(payload['states']) == 4
        for snapshot in payload['states']:
            restored = r._restore_snapshot(env, snapshot)
            next_obs, _, done, _ = env.step(snapshot['reference_action_chunk'][0].tolist())
            physics = r._capture_physics(env)
            images = r._image_obs(next_obs, ev)
            image_error = float(np.mean([np.mean(np.abs(images[k].astype(float) - snapshot['reference_next_images'][k].astype(float))) / 255
                                         for k in ('image', 'wrist_image')]))
            record = {'episode': episode, 'replan_idx': snapshot['replan_idx'],
                      'pre_obs_max_abs': r._obs_max_abs_error(snapshot['obs'], restored),
                      'next_image_l1_01': image_error,
                      'next_qpos_max_abs': float(np.max(np.abs(physics['qpos'] - snapshot['reference_next_physics']['qpos']))),
                      'next_qvel_max_abs': float(np.max(np.abs(physics['qvel'] - snapshot['reference_next_physics']['qvel']))),
                      'done_equal': bool(done) == snapshot['terminal_after_first_action'],
                      'contacts_equal': r._filter_task_contacts(r._capture_contacts(env)) == snapshot['reference_next_contacts']}
            result['measurements'].append(record)
            print(json.dumps(record), flush=True)
            assert record['pre_obs_max_abs'] <= 1e-4 and image_error <= 1e-6, record
            assert record['next_qpos_max_abs'] <= 1e-10 and record['next_qvel_max_abs'] <= 1e-10, record
            assert record['done_equal'] and record['contacts_equal'], record
    assert len(result['measurements']) == 16
    result['status'] = 'passed'
except Exception as exc:
    result['status'] = 'failed'
    result['error'] = repr(exc)
    raise
finally:
    (out / 'collected_replay_preflight.json').write_text(json.dumps(result, indent=2) + '\n')
    env.close()
