"""CPU-only real simulator restore check before model allocations."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import runner as r

r._require_pbs_allocation()
from experiments.libero import libero_utils as ev
from libero.libero import benchmark, get_libero_path
parser = argparse.ArgumentParser()
parser.add_argument("--task-id", type=int, default=0)
args = parser.parse_args()
out = Path(os.environ['ARTIFACTS'])
suite = benchmark.get_benchmark_dict()['libero_goal']()
task = suite.get_task(args.task_id)
env, description = ev.get_libero_env(task, 256, 42)
r._RUNTIME = SimpleNamespace(ev=ev)
try:
    env.reset()
    states = torch.load(Path(get_libero_path('init_states')) / task.problem_folder / task.init_states_file, weights_only=False)
    obs = env.set_init_state(states[0])
    for _ in range(30):
        obs, _, _, _ = env.step(ev.get_libero_dummy_action())
    measurements = []
    for boundary in [30, 40]:
        if boundary == 40:
            for _ in range(10):
                obs, _, _, _ = env.step([0.1, 0, 0, 0, 0, 0, -1])
        snapshot = r._capture_snapshot(env, obs, task_id=args.task_id, episode=0, t=boundary, replan_idx=0)
        action = [0.1, 0, 0, 0, 0, 0, -1]
        native_obs, _, native_done, _ = env.step(action)
        native_physics = r._capture_physics(env)
        restored = r._restore_snapshot(env, snapshot)
        replay_obs, _, replay_done, _ = env.step(action)
        replay_physics = r._capture_physics(env)
        record = {
            'boundary': boundary,
            'pre_obs_max_abs': r._obs_max_abs_error(snapshot['obs'], restored),
            'next_obs_max_abs': r._obs_max_abs_error(native_obs, replay_obs),
            'next_qpos_max_abs': float(np.max(np.abs(native_physics['qpos'] - replay_physics['qpos']))),
            'done_equal': bool(native_done == replay_done),
        }
        measurements.append(record)
        print(json.dumps(record), flush=True)
        assert record['next_obs_max_abs'] <= 1e-4 and record['next_qpos_max_abs'] <= 1e-10 and record['done_equal']
        obs = replay_obs
    (out / 'restore_preflight.json').write_text(json.dumps({'task_id': args.task_id, 'task': description, 'measurements': measurements, 'status': 'passed'}, indent=2))
finally:
    env.close()
