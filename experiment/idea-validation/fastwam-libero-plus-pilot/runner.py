"""Paired Optional-IDM Plus evaluation inside a real PBS allocation."""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import socket
import statistics
import sys
import time
from pathlib import Path

BASE = Path('/scratch/users/ntu/yguo017/fastwam-smoke')
ROOT = Path('/scratch/users/ntu/yguo017/fastwam-libero-plus-pilot-20261005')
ARMS = ('bf16', 'w4a8', 'w4a4', 'w4a4kv4')
PROTOCOL = 'fastwam-optional-idm-plus-pilot-v1'


def save(path, data):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def guard():
    host = socket.gethostname().split('.')[0]
    nodes = Path(os.environ['PBS_NODEFILE']).read_text().split()
    if (not os.environ.get('PBS_JOBID') or 'login' in host
            or host not in {node.split('.')[0] for node in nodes}):
        raise RuntimeError('Approved PBS compute allocation required')


def load_manifest(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    rows = data['variants']
    counts = collections.Counter((row['suite'], row['dimension']) for row in rows)
    if len(rows) != 1400 or len(counts) != 28 or set(counts.values()) != {50}:
        raise RuntimeError(f'Invalid frozen cohort: {len(rows)} variants, {counts}')
    if len({str(row['variant_id']) for row in rows}) != len(rows):
        raise RuntimeError('Duplicate variant IDs')
    return data, rows


def seed_for(index, replan=0, environment=False):
    return (800_000_000 if environment else 900_000_000) + index * 1000 + replan


def load_assigned_slots(path, rows):
    doc = json.loads(Path(path).read_text(encoding='utf-8'))
    slots = doc.get('slots')
    if doc.get('protocol') != PROTOCOL or not isinstance(slots, list) or not slots:
        raise ValueError('Retry requires a nonempty slot list for this frozen protocol')
    seen, assigned = set(), []
    for slot in slots:
        index, arm = slot.get('index'), slot.get('arm')
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(rows) or arm not in ARMS:
            raise ValueError(f'Invalid assigned slot: {slot}')
        if slot.get('variant_id') != str(rows[index]['variant_id']) or (index, arm) in seen:
            raise ValueError(f'Duplicate or mismatched assigned slot: {slot}')
        seen.add((index, arm))
        assigned.append({'index': index, 'arm': arm, 'variant_id': str(rows[index]['variant_id'])})
    return sorted(assigned, key=lambda s: (s['index'], ARMS.index(s['arm'])))


class QueryMeter:
    """Two CUDA events bound the continuous denoising interval, without nested sums."""
    def __init__(self, model, torch, steps):
        self.torch, self.steps = torch, steps
        self.active = False
        self.start = torch.cuda.Event(enable_timing=True)
        self.stop = torch.cuda.Event(enable_timing=True)
        self.v_schedule = model.infer_video_scheduler.build_inference_schedule
        self.a_step = model.infer_action_scheduler.step

        def schedule(*args, **kwargs):
            if self.active:
                self.start.record()
                self.started += 1
            return self.v_schedule(*args, **kwargs)

        def step(*args, **kwargs):
            result = self.a_step(*args, **kwargs)
            if self.active:
                self.completed += 1
                if self.completed == self.steps:
                    self.stop.record()
            return result

        model.infer_video_scheduler.build_inference_schedule = schedule
        model.infer_action_scheduler.step = step

    def begin(self):
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats()
        self.baseline_bytes = self.torch.cuda.memory_allocated()
        self.started = self.completed = 0
        self.active = True
        self.wall_start = time.perf_counter()

    def end(self):
        self.torch.cuda.synchronize()
        wall = (time.perf_counter() - self.wall_start) * 1000
        self.active = False
        if self.started != 1 or self.completed != self.steps:
            raise RuntimeError(f'Incomplete denoising timing scope: {self.started}, {self.completed}')
        return {
            'denoising_latency_ms': self.start.elapsed_time(self.stop),
            'full_query_wall_ms': wall,
            'peak_allocated_gpu_GB': self.torch.cuda.max_memory_allocated() / 1e9,
            'baseline_allocated_gpu_GB': self.baseline_bytes / 1e9,
            'peak_reserved_gpu_GB': self.torch.cuda.max_memory_reserved() / 1e9,
        }


class Runtime:
    def __init__(self, out):
        from libero_state_loader import ensure_numpy_compat
        numpy_compat = ensure_numpy_compat()
        print('LIBERO_NUMPY_COMPAT ' + json.dumps(numpy_compat), flush=True)
        import torch
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
        from experiments.libero import eval_libero_single as ev
        from libero.libero import benchmark
        from quantization import Quantization

        self.torch, self.ev, self.benchmark, self.out = torch, ev, benchmark, out
        if torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one allocated CUDA GPU required')
        props = torch.cuda.get_device_properties(0)
        mask = os.environ['CUDA_VISIBLE_DEVICES']
        if str(props.uuid).removeprefix('GPU-').lower() != mask.removeprefix('GPU-').lower():
            raise RuntimeError('CUDA GPU UUID does not match allocation')
        with initialize_config_dir(version_base='1.3', config_dir=str(BASE / 'FastWAM/configs')):
            self.cfg = compose(config_name='sim_libero', overrides=[
                'task=libero_optional_idm_2cam224_1e-4', 'mixed_precision=bf16',
                f'ckpt={BASE}/checkpoints/libero_optional_idm_2cam224.clean.pt',
                f'EVALUATION.dataset_stats_path={BASE}/checkpoints/libero_optional_idm_2cam224_dataset_stats.json',
                '+EVALUATION.action_infer_mode=idm', 'EVALUATION.num_inference_steps=10',
                'EVALUATION.sigma_shift=1.0', 'EVALUATION.compile_action_infer=false',
                'EVALUATION.use_action_ensembler=false', 'EVALUATION.visualize_future_video=false',
            ])
        print('MODEL_LOADING', flush=True)
        self.model = instantiate(self.cfg.model, model_dtype=torch.bfloat16, device='cuda')
        self.model.load_checkpoint(str(self.cfg.ckpt))
        self.model.to('cuda').eval()
        self.processor = instantiate(self.cfg.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(self.cfg.EVALUATION.dataset_stats_path)))
        self.q = Quantization(self.model, group_size=128, kv4=True)
        self.meter = QueryMeter(self.model, torch, steps=10)
        self.height, self.width = map(int, self.cfg.data.train.video_size)
        self.suites = {}
        self.env = None
        self.weight_converted = False
        self.calls = 0
        self.arm_calls = collections.Counter()
        from omegaconf import OmegaConf
        (out / 'resolved_config.yaml').write_text(OmegaConf.to_yaml(self.cfg), encoding='utf-8')
        save(out / 'runtime.json', {
            'protocol': PROTOCOL, 'pbs_jobid': os.environ['PBS_JOBID'], 'host': socket.gethostname(),
            'gpu': props.name, 'gpu_uuid': str(props.uuid), 'gpu_total_GB': props.total_memory / 1e9,
            'torch': torch.__version__, 'mode': 'idm', 'inference_steps': 10, 'real_quant': True,
            'libero_numpy_compatibility': numpy_compat,
            'quantization': self.q.summary(), 'libero_module': __import__('libero').__file__,
        })
        print('MODEL_READY', flush=True)

    def arm(self, arm):
        if arm not in ARMS:
            raise ValueError(arm)
        if arm == 'bf16' and self.weight_converted:
            raise RuntimeError('BF16 cannot run after irreversible W4 conversion')
        if arm != 'bf16' and not self.weight_converted:
            self.q.disable()
            self.q.convert_weights()
            self.weight_converted = True
            self.torch.cuda.empty_cache()
        self.q.enable(arm)
        self.current_arm = arm

    def task(self, row, index):
        suite_name = row['suite']
        if suite_name not in self.suites:
            self.suites[suite_name] = self.benchmark.get_benchmark_dict()[suite_name]()
        suite = self.suites[suite_name]
        task = suite.get_task(int(row['task_id']))
        if row.get('task_name') and task.name != row['task_name']:
            raise RuntimeError(f'Manifest task identity changed: {row}, {task.name}')
        if self.env is not None:
            self.env.close()
        # Plus parses perturbation suffixes from a string path, unlike the original helper's Path.
        from libero.libero.envs import OffScreenRenderEnv
        self.description = task.language
        self.env = OffScreenRenderEnv(
            bddl_file_name=str(suite.get_task_bddl_file_path(int(row['task_id']))),
            camera_heights=self.ev.LIBERO_ENV_RESOLUTION, camera_widths=self.ev.LIBERO_ENV_RESOLUTION,
        )
        from libero_state_loader import load_task_init_states
        initial_states, self.init_states_path = load_task_init_states(suite, row['task_id'])
        if not len(initial_states):
            raise RuntimeError('Missing initial state')
        self.env.seed(seed_for(index, environment=True))
        import numpy as np
        np.random.seed(seed_for(index, environment=True))
        random.seed(seed_for(index, environment=True))
        self.env.reset()
        obs = self.env.set_init_state(initial_states[0])
        for _ in range(30):
            obs, _, done, _ = self.env.step(self.ev.get_libero_dummy_action())
            if done:
                raise RuntimeError('Task completed during settling; invalid evaluation slot')
        return obs

    def datum(self, obs):
        image, proprio, _ = self.ev._obs_to_model_input(
            obs, cfg=self.cfg, processor=self.processor, width=self.width, height=self.height,
            device='cuda', dtype=self.model.torch_dtype,
        )
        return {'image': image, 'proprio': self.torch.as_tensor(proprio, device='cuda', dtype=self.model.torch_dtype)}

    def infer(self, datum, seed, measure=True):
        if measure:
            self.meter.begin()
        with self.torch.inference_mode():
            action = self.model.infer_action(
                prompt=self.ev.DEFAULT_PROMPT.format(task=self.description),
                input_image=datum['image'], proprio=datum['proprio'], action_horizon=32,
                num_video_frames=9, num_inference_steps=10, sigma_shift=1.0,
                seed=seed, rand_device='cpu', tiled=False, compile_action_infer=False, action_infer_mode='idm',
            )['action']
        metrics = self.meter.end() if measure else {}
        if measure:
            metrics['query_index_in_arm'] = self.arm_calls[self.current_arm]
            metrics['first_query_in_arm'] = self.arm_calls[self.current_arm] == 0
        if tuple(action.shape) != (32, 7) or not bool(self.torch.isfinite(action).all()):
            raise RuntimeError('Expected finite normalized [32,7] action')
        self.calls += 1
        self.arm_calls[self.current_arm] += 1
        return action, metrics

    def command(self, action):
        import numpy as np
        commands = self.ev._denormalize_action(action, self.processor)[0]
        commands[..., -1] = commands[..., -1] * 2 - 1
        commands = self.ev.invert_gripper_action(commands)
        commands[..., -1] = np.sign(commands[..., -1])
        if commands.shape != (32, 7) or not np.isfinite(commands).all():
            raise RuntimeError('Invalid controller commands')
        return commands

    def episode(self, row, index):
        started = time.monotonic()
        evidence_before = self.q.summary()
        obs = self.task(row, index)
        cap = 700 if row['suite'] == 'libero_10' else 400
        steps, replans, success = 0, 0, False
        metrics = []
        action_sample = []
        while steps < cap and not success:
            action, metric = self.infer(self.datum(obs), seed_for(index, replans))
            metric['replan'] = replans
            metrics.append(metric)
            if replans < 3:
                action_sample.append(action.float().cpu().tolist())
            commands = self.command(action)
            for command in commands[:min(10, cap - steps)]:
                obs, _, done, _ = self.env.step(command.tolist())
                steps += 1
                if done:
                    success = True
                    break
            replans += 1
            if replans % 5 == 0:
                print(f'REPLAN variant={row["variant_id"]} arm={self.current_arm} index={replans} steps={steps}', flush=True)
        self.env.close()
        self.env = None
        evidence_after = self.q.summary()
        evidence = {key: evidence_after.get(key) for key in (
            'real_quant', 'weight_state', 'packed_weight_tensor_bytes', 'bf16_target_weight_absent',
            'activation_bits', 'activation_storage', 'native_int4_tensorcore',
        )}
        for key in ('integer_gemm_calls', 'native_int4_gemm_calls', 'kv_packed_prefills', 'kv_layer_reads'):
            evidence[key] = evidence_after.get(key, 0) - evidence_before.get(key, 0)
        if self.current_arm != 'bf16' and (evidence['integer_gemm_calls'] <= 0
                or evidence['packed_weight_tensor_bytes'] is None or evidence['packed_weight_tensor_bytes'] <= 0
                or evidence['bf16_target_weight_absent'] is not True):
            raise RuntimeError('Episode did not execute actual packed-weight integer GEMM')
        if self.current_arm == 'w4a4kv4' and (evidence['kv_packed_prefills'] <= 0 or evidence['kv_layer_reads'] <= 0):
            raise RuntimeError('Episode did not read the packed KV4 cache')
        if self.current_arm in ('w4a4', 'w4a4kv4') and evidence['native_int4_gemm_calls'] <= 0:
            raise RuntimeError('A4 episode did not execute native INT4 Tensor Core GEMM')
        return {
            'protocol': PROTOCOL, 'variant_id': str(row['variant_id']), 'index': index,
            'suite': row['suite'], 'dimension': row['dimension'], 'task_id': int(row['task_id']),
            'original_task': row.get('original_task'), 'subtype': row.get('subtype'),
            'arm': self.current_arm, 'complete': True, 'success': success,
            'termination': 'success' if success else 'control_step_timeout', 'steps': steps,
            'replans': replans, 'episode_seconds': time.monotonic() - started,
            'env_seed': seed_for(index, environment=True), 'state_id': 0,
            'initial_state_path': self.init_states_path,
            'description': self.description, 'metrics': metrics, 'first_action_chunks': action_sample,
            'real_quant_evidence': evidence,
        }


def sample_reference_rows(rows, mode):
    if mode == 'preflight':
        return [next((i, row) for i, row in enumerate(rows)
                     if row['suite'] == 'libero_spatial' and row['dimension'] == 'camera_viewpoints'),
                next((i, row) for i, row in enumerate(rows)
                     if row['suite'] == 'libero_10' and row['dimension'] == 'robot_initial_states')]
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault((row['suite'], row['dimension']), (i, row))
    return list(groups.values())


def fixed_timing(runtime, references, arm, out):
    records = []
    for datum in references:
        runtime.description = datum['description']
        seed = seed_for(datum['index'])
        for _ in range(2):
            runtime.infer(datum, seed, measure=False)
        for repeat in range(3):
            before = runtime.q.summary()
            action, metrics = runtime.infer(datum, seed)
            after = runtime.q.summary()
            evidence = {key: after.get(key) for key in (
                'real_quant', 'weight_state', 'packed_weight_tensor_bytes', 'bf16_target_weight_absent',
                'activation_bits', 'activation_storage', 'native_int4_tensorcore',
            )}
            for key in ('integer_gemm_calls', 'native_int4_gemm_calls', 'kv_packed_prefills', 'kv_layer_reads'):
                evidence[key] = after.get(key, 0) - before.get(key, 0)
            records.append(dict(metrics, arm=arm, index=datum['index'], variant_id=datum['variant_id'],
                                repeat=repeat, sampler_seed=seed, real_quant_evidence=evidence,
                                action=action.float().cpu().tolist()))
    save(out / f'timing_{arm}.json', records)
    return records


def main():
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['preflight', 'performance', 'shard'], required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'manifest.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int, default=4)
    parser.add_argument('--slots', type=Path, help='explicit missing variant/arm slots for a shard retry')
    parser.add_argument('--export-storage', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest, rows = load_manifest(args.manifest)
    if not 0 <= args.start < args.stop <= len(rows):
        raise ValueError('Invalid manifest index interval')
    if args.slots is not None and args.mode != 'shard':
        raise ValueError('--slots is only valid for shard retries')
    assigned = load_assigned_slots(args.slots, rows) if args.slots is not None else None
    reference_rows = sample_reference_rows(rows, args.mode) if args.mode != 'shard' else []
    selected = ([(i, rows[i]) for i in sorted({s['index'] for s in assigned})] if assigned is not None
                else [(i, rows[i]) for i in range(args.start, args.stop)] if args.mode == 'shard'
                else reference_rows if args.mode == 'preflight' else [])
    expected = len(assigned) if assigned is not None else len(selected) * 4
    from quantization import self_check
    self_check()
    save(args.out / 'self_check.json', {'passed': True})
    runtime = Runtime(args.out)
    try:
        references = []
        if args.mode != 'shard':
            runtime.arm('bf16')
            for index, row in reference_rows:
                obs = runtime.task(row, index)
                datum = runtime.datum(obs)
                references.append(dict(datum, description=runtime.description, index=index, variant_id=str(row['variant_id'])))
            first = references[0]
            runtime.description = first['description']
            a = runtime.infer(first, seed_for(first['index']), measure=False)[0]
            b = runtime.infer(first, seed_for(first['index']))[0]
            difference = float((a - b).abs().max())
            if difference > 1e-6:
                raise RuntimeError(f'Timing boundary changed BF16 result: {difference}')
            save(args.out / 'reference_gate.json', {'passed': True, 'max_abs': difference})
        if assigned is not None and not any(s['arm'] == 'bf16' for s in assigned):
            index, row = selected[0]
            runtime.arm('bf16')
            obs = runtime.task(row, index)
            runtime.infer(runtime.datum(obs), seed_for(index), measure=False)
            runtime.env.close()
            runtime.env = None
            save(args.out / 'retry_prime.json', {'index': index, 'variant_id': str(row['variant_id']),
                                               'sampler_seed': seed_for(index), 'episode_counted': False})
        results = []
        all_timings, storage = [], {}
        for arm in ARMS:
            arm_indices = {s['index'] for s in assigned if s['arm'] == arm} if assigned is not None else None
            if assigned is not None and not arm_indices:
                continue
            runtime.arm(arm)
            if args.export_storage and arm in ('bf16', 'w4a8'):
                storage[arm] = runtime.q.export_packed_weights(args.out / f'storage_{arm}')
                save(args.out / 'storage.json', storage)
            if references:
                all_timings.extend(fixed_timing(runtime, references, arm, args.out))
            execution = [(i, row) for i, row in selected if arm_indices is None or i in arm_indices]
            if arm != 'bf16':
                random.Random(20261005 + ARMS.index(arm)).shuffle(execution)
            for index, row in execution:
                save(args.out / 'progress.json', {'status': 'running', 'arm': arm, 'variant_id': str(row['variant_id']),
                                               'completed': len(results), 'expected': expected})
                result = runtime.episode(row, index)
                with (args.out / 'episodes.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(result, allow_nan=False) + '\n')
                results.append(result)
                print(f'EPISODE_COMPLETE arm={arm} variant={row["variant_id"]} success={result["success"]} seconds={result["episode_seconds"]:.2f}', flush=True)
        timing_summary = {}
        for arm in ARMS:
            arm_records = [r for r in all_timings if r['arm'] == arm]
            if arm_records:
                timing_summary[arm] = {'median_denoising_ms': statistics.median(r['denoising_latency_ms'] for r in arm_records),
                                      'peak_allocated_gpu_GB': max(r['peak_allocated_gpu_GB'] for r in arm_records),
                                      'paired_reference_contexts': len(references), 'queries': len(arm_records)}
        for arm in timing_summary:
            timing_summary[arm]['speedup_relative_to_FP'] = timing_summary['bf16']['median_denoising_ms'] / timing_summary[arm]['median_denoising_ms']
        summary = {'protocol': PROTOCOL, 'mode': args.mode, 'complete': True, 'episodes': len(results),
                   'indices': [i for i, _ in selected], 'timing': timing_summary, 'storage': storage,
                   'population': manifest.get('population_counts'), 'real_quant': True,
                   'quantization_runtime': runtime.q.summary()}
        if assigned is not None:
            summary['assigned_slots'] = assigned
        save(args.out / 'summary.json', summary)
        save(args.out / 'progress.json', {'status': 'complete', 'completed': len(results), 'expected': expected})
        print('EVALUATION_COMPLETE', flush=True)
    finally:
        if runtime.env is not None:
            runtime.env.close()
        runtime.q.close()


if __name__ == '__main__':
    main()
