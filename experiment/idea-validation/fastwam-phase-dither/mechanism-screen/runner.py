"""Frozen full-path predictor screen. Run only inside a real PBS allocation."""
import argparse
import json
import math
import os
import random
import re
import socket
import time
from collections import defaultdict
from pathlib import Path

BASE = Path('/scratch/users/ntu/yguo017/fastwam-smoke')
SAMPLER_SEEDS = (2026, 2027)


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def guard():
    host = socket.gethostname().split('.')[0]
    assert os.environ.get('PBS_JOBID') and 'login' not in host
    assert host == os.environ['OTC_ALLOCATION_HOST']
    assert host in {x.split('.')[0] for x in Path(os.environ['PBS_NODEFILE']).read_text().split()}


def runtime():
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    from experiments.libero import eval_libero_single as ev
    from libero.libero import benchmark

    with initialize_config_dir(version_base='1.3', config_dir=str(BASE / 'FastWAM/configs')):
        cfg = compose(config_name='sim_libero', overrides=[
            'task=libero_optional_idm_2cam224_1e-4', 'mixed_precision=bf16',
            f'ckpt={BASE}/checkpoints/libero_optional_idm_2cam224.clean.pt',
            'EVALUATION.task_suite_name=libero_goal', 'EVALUATION.task_id=0',
            'EVALUATION.num_trials=12', 'EVALUATION.sigma_shift=1.0',
            '+EVALUATION.action_infer_mode=first_frame',
            'EVALUATION.compile_action_infer=false', 'EVALUATION.device=cuda',
            'EVALUATION.use_action_ensembler=false', 'EVALUATION.visualize_future_video=false',
            f'EVALUATION.dataset_stats_path={BASE}/checkpoints/libero_optional_idm_2cam224_dataset_stats.json',
        ])
    print('MODEL_LOADING', flush=True)
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device='cuda')
    model.load_checkpoint(str(cfg.ckpt))
    model.to('cuda').eval()
    assert model.torch_dtype == torch.bfloat16
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(cfg.EVALUATION.dataset_stats_path)))
    task = benchmark.get_benchmark_dict()['libero_goal']().get_task(0)
    env, description = ev.get_libero_env(task, ev.LIBERO_ENV_RESOLUTION, 2026)
    print('MODEL_READY', flush=True)
    return model, processor, cfg, ev, env, task, description


class Screen:
    def __init__(self, out):
        import torch
        from quant import Quantizer
        self.torch, self.out = torch, out
        self.model, self.processor, self.cfg, self.ev, self.env, self.task, self.description = runtime()
        self.q = Quantizer(self.model, group_size=128)
        self.q.unit_test()
        self.expected_sites = None
        self.forward_count = defaultdict(int)
        self.horizon = int(self.cfg.EVALUATION.action_horizon or (int(self.cfg.data.train.num_frames) - 1))
        self.height, self.width = map(int, self.cfg.data.train.video_size)
        save(out / 'runtime_config.json', {
            'checkpoint': str(self.cfg.ckpt), 'dataset_stats': str(self.cfg.EVALUATION.dataset_stats_path),
            'action_mode': 'first_frame', 'sigma_shift': 1.0, 'num_inference_steps': 20,
            'compile_action_infer': False, 'torch_dtype': str(self.model.torch_dtype),
            'action_horizon': self.horizon, 'input_hw': [self.height, self.width], 'group_size': 128,
            'gpu': torch.cuda.get_device_name(0), 'pbs_jobid': os.environ['PBS_JOBID'],
            'source_identity': 'prior snapshot signatures checked within allocation',
            'packed_kernel': False,
        })
        self.max_block = {}
        for name in self.q.modules:
            m = re.search(r'blocks\.(\d+)\.', name)
            if m:
                stream = self.stream(name)
                self.max_block[stream] = max(self.max_block.get(stream, 0), int(m[1]))

    @staticmethod
    def stream(name):
        if 'video' in name:
            return 'video'
        if 'action' in name:
            return 'action'
        if 'proprio' in name:
            return 'proprio'
        raise RuntimeError(f'Unclassified denoiser module: {name}')

    def group(self, site):
        name = site.split('#')[0]
        stream = self.stream(name)
        if stream == 'proprio':
            return 'proprio'
        m = re.search(r'blocks\.(\d+)\.', name)
        if not m:
            return stream + '.condition'
        depth = int(m[1])
        if stream == 'video' and depth == self.max_block[stream]:
            if not name.endswith(('self_attn.k', 'self_attn.v')):
                return None  # Executed dead endpoint branch; still W4A8.
        segment = min(2, 3 * depth // (self.max_block[stream] + 1))
        return stream + '.' + ('early', 'middle', 'late')[segment]

    def table(self, parameters):
        return {s: parameters.get(self.group(s), 0.0) for s in self.expected_sites}

    def infer(self, datum, mode, sampler_seed, draw_seed=0, phases=None, clips=None, trace=True):
        torch = self.torch
        self.q.begin(mode=mode, seed=draw_seed, phases=phases or {}, clips=clips or {})
        if not trace:
            assert mode == 'bf16'
            self.q.disable()
        t0 = time.perf_counter()
        with torch.inference_mode():
            result = self.model.infer_action(
                prompt=self.ev.DEFAULT_PROMPT.format(task=self.description),
                input_image=datum['image'].to('cuda'), proprio=datum['proprio'].to('cuda'),
                action_horizon=self.horizon, negative_prompt=str(self.cfg.EVALUATION.get('negative_prompt', '')),
                text_cfg_scale=float(self.cfg.EVALUATION.get('text_cfg_scale', 1.0)),
                num_inference_steps=20, sigma_shift=1.0, seed=sampler_seed, rand_device='cpu',
                tiled=bool(self.cfg.EVALUATION.get('tiled', False)), compile_action_infer=False,
                action_infer_mode='first_frame',
            )['action'].float().cpu()
        stats = self.q.end()
        assert torch.isfinite(result).all() and stats['nonfinite_values'] == 0
        if mode in ('phase', 'independent'):
            assert stats['overload_codes'] == 0, stats
        sites = list(self.q.executed_sites)
        if not trace:
            assert sites == []
        elif self.expected_sites is None:
            self.expected_sites = sites
        else:
            assert sites == self.expected_sites, 'Executed Linear/site coverage changed across arms'
        self.forward_count[mode] += 1
        return result, stats, time.perf_counter() - t0

    def collect(self, episodes):
        torch = self.torch
        from libero.libero import get_libero_path
        initial = torch.load(Path(get_libero_path('init_states')) / self.task.problem_folder / self.task.init_states_file,
                             weights_only=False)
        data = []
        for episode in episodes:
            self.env.seed(2026)
            self.env.reset()
            obs = self.env.set_init_state(initial[episode])
            for _ in range(30):
                obs, _, done, _ = self.env.step(self.ev.get_libero_dummy_action())
                assert not done, 'Terminated in reference warmup'
            for position in range(2):
                image, proprio, _ = self.ev._obs_to_model_input(
                    obs, cfg=self.cfg, processor=self.processor, width=self.width, height=self.height,
                    device='cuda', dtype=self.model.torch_dtype)
                if not torch.is_tensor(proprio):
                    proprio = torch.as_tensor(proprio, dtype=self.model.torch_dtype)
                datum = {'episode': episode, 'position': position,
                         'image': image.cpu(), 'proprio': proprio.cpu(), 'refs': {}}
                for sampler_seed in SAMPLER_SEEDS:
                    datum['refs'][sampler_seed] = self.infer(datum, 'bf16', sampler_seed)[0]
                if not data:
                    repeat = self.infer(datum, 'bf16', SAMPLER_SEEDS[0])[0]
                    error = float((repeat - datum['refs'][SAMPLER_SEEDS[0]]).abs().max())
                    assert error <= 1e-6, ('BF16_REPEAT', error)
                    unhooked = self.infer(datum, 'bf16', SAMPLER_SEEDS[0], trace=False)[0]
                    identity_error = float((unhooked - datum['refs'][SAMPLER_SEEDS[0]]).abs().max())
                    assert identity_error <= 1e-6, ('HOOK_IDENTITY', identity_error)
                    save(self.out / 'reference_gate.json', {'repeat_max_abs': error, 'hook_disabled_identity_max_abs': identity_error})
                data.append(datum)
                if position == 0:
                    action = self.ev._denormalize_action(datum['refs'][SAMPLER_SEEDS[0]], self.processor)[0]
                    action[..., -1] = action[..., -1] * 2 - 1
                    action = self.ev.invert_gripper_action(action)
                    if bool(self.cfg.EVALUATION.get('binarize_gripper', False)):
                        import numpy as np
                        action[..., -1] = np.sign(action[..., -1])
                    assert len(action) >= 10
                    for a in action[:10]:
                        obs, _, done, _ = self.env.step(a.tolist())
                        assert not done, 'Reference ended before second frozen observation'
            print(f'COLLECT episode={episode} observations=2', flush=True)
        torch.save(data, self.out / 'frozen_observations.pt')
        return data

    def objective(self, data, mode, parameters, draw_seeds, label):
        total = []
        phases = self.table(parameters) if mode == 'phase' else {}
        clips = {s: parameters.get(self.group(s), 1.0) for s in self.expected_sites} if mode == 'rtn' else {}
        for datum in data:
            for i, draw_seed in enumerate(draw_seeds):
                sampler_seed = SAMPLER_SEEDS[i % 2]
                pred, _, _ = self.infer(datum, mode, sampler_seed, draw_seed, phases, clips)
                total.append(float((pred - datum['refs'][sampler_seed]).square().mean()))
        score = sum(total) / len(total)
        print(f'OBJECTIVE {label} mse={score:.9g}', flush=True)
        return score

    def calibrate(self, data, mode, groups, values):
        parameters = {g: (0.0 if mode == 'phase' else 1.0) for g in groups}
        checkpoints, history = [], []
        for sweep in range(2):
            for group in groups:
                trials = []
                for value in values:
                    candidate = dict(parameters, **{group: value})
                    score = self.objective(data, mode, candidate, [101, 102], f'{mode}/{sweep}/{group}/{value}')
                    trials.append((score, value))
                parameters[group] = min(trials)[1]
                history.append({'sweep': sweep, 'group': group, 'trials': trials, 'selected': parameters[group]})
            checkpoints.append(dict(parameters))
            save(self.out / f'{mode}_calibration.json', {'history': history, 'checkpoints': checkpoints})
        return checkpoints


def bootstrap_lower(values):
    rng = random.Random(91573)
    samples = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(2000))
    return samples[49], samples[1949]


def pilot(screen, data):
    cal = [d for d in data if d['episode'] in (4, 5)]
    dev = [d for d in data if d['episode'] in (6, 7)]
    test = [d for d in data if 8 <= d['episode'] <= 15]
    groups = sorted({screen.group(s) for s in screen.expected_sites} - {None, 'video.condition'})
    assert len(cal) == 4 and len(dev) == 4 and len(test) == 16 and len(groups) >= 6
    phase_checkpoints = screen.calibrate(cal, 'phase', groups, (0, .25, .5, .75))
    direct_checkpoints = screen.calibrate(cal, 'rtn', groups, (.94, .98, 1, 1.02))
    phase_scores = [screen.objective(dev, 'phase', c, [201, 202], f'phase-dev/{i}')
                    for i, c in enumerate(phase_checkpoints)]
    direct_scores = [screen.objective(dev, 'rtn', c, [201, 202], f'direct-dev/{i}')
                     for i, c in enumerate(direct_checkpoints)]
    learned = screen.table(phase_checkpoints[min(range(2), key=lambda i: phase_scores[i])])
    direct = direct_checkpoints[min(range(2), key=lambda i: direct_scores[i])]
    direct_table = {s: direct.get(screen.group(s), 1.0) for s in screen.expected_sites}
    effective_sites = [s for s in screen.expected_sites if screen.group(s) is not None]
    shuffled = [learned[s] for s in effective_sites]
    random.Random(8842).shuffle(shuffled)
    permuted = dict(learned)
    permuted.update(zip(effective_sites, shuffled))
    deltas = {round((permuted[s] - learned[s]) % 1, 8) for s in effective_sites}
    nongauge = len(deltas) > 1
    save(screen.out / 'locked_selection.json', {
        'groups': groups, 'learned': learned, 'permuted': permuted, 'direct': direct,
        'phase_dev_scores': phase_scores, 'direct_dev_scores': direct_scores, 'permutation_nongauge': nongauge,
    })
    modes = {'w4': 'w4', 'rtn': 'rtn', 'independent': 'independent', 'shared0': 'phase',
             'learned': 'phase', 'permuted': 'phase', 'direct': 'rtn'}
    rows, errors = [], defaultdict(list)
    for datum in test:
        for i, draw_seed in enumerate((1101, 1102, 1103, 1104)):
            sampler_seed = SAMPLER_SEEDS[i % 2]
            order = list(modes)
            random.Random(9381 + datum['episode'] * 100 + datum['position'] * 10 + i).shuffle(order)
            for arm in order:
                phases = learned if arm == 'learned' else permuted if arm == 'permuted' else {}
                clips = direct_table if arm == 'direct' else {}
                pred, stats, seconds = screen.infer(datum, modes[arm], sampler_seed, draw_seed, phases, clips)
                mse = float((pred - datum['refs'][sampler_seed]).square().mean())
                errors[(arm, datum['episode'])].append(mse)
                rows.append({'arm': arm, 'episode': datum['episode'], 'position': datum['position'],
                             'sampler_seed': sampler_seed, 'draw_seed': draw_seed, 'mse': mse,
                             'forward_seconds_fake_quant': seconds, 'overload_codes': stats['overload_codes']})
        print(f'TEST episode={datum["episode"]} position={datum["position"]}', flush=True)
        save(screen.out / 'test_progress.json', {'completed_rows': len(rows)})
    per_episode = {arm: [sum(errors[(arm, e)]) / len(errors[(arm, e)]) for e in range(8, 16)] for arm in modes}
    means = {arm: sum(x) / len(x) for arm, x in per_episode.items()}
    strongest = min(('rtn', 'independent', 'direct'), key=means.get)
    differences = [a - b for a, b in zip(per_episode[strongest], per_episode['learned'])]
    permutation_differences = [a - b for a, b in zip(per_episode['permuted'], per_episode['learned'])]
    confidence = bootstrap_lower(differences)
    improvement = 1 - means['learned'] / max(means[strongest], 1e-20)
    gates = {'relative_gain_at_least_10pct': improvement >= .1,
             'improves_6_of_8': sum(x > 0 for x in differences) >= 6,
             'bootstrap_lower_positive': confidence[0] > 0,
             'permutation_nongauge': nongauge,
             'permutation_removes_gain': means['permuted'] > means['learned'] and sum(x > 0 for x in permutation_differences) >= 6}
    result = {'decision': 'CONTINUE_COARSE_RECIPE' if all(gates.values()) else 'NO_GO_FOR_EXPANSION',
              'scope': 'Full-path W4A8 fake-quant, coarse phase recipe, task0 fixed observations only',
              'gates': gates, 'mean_mse': means, 'episode_mse': per_episode,
              'strongest_matched_baseline': strongest, 'relative_gain': improvement,
              'paired_absolute_gain_bootstrap95': confidence, 'forward_counts': dict(screen.forward_count),
              'raw_rows': rows, 'native_kernel': False, 'closed_loop': False}
    save(screen.out / 'result.json', result)
    print('MECHANISM_DECISION', result['decision'], means, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('preflight', 'pilot'), default='preflight')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    guard()
    args.out.mkdir(parents=True, exist_ok=True)
    screen = Screen(args.out)
    try:
        data = screen.collect((4,) if args.mode == 'preflight' else range(4, 16))
        save(args.out / 'coverage.json', {'executed_sites': screen.expected_sites,
             'groups': {s: screen.group(s) for s in screen.expected_sites},
             'selected_modules': list(screen.q.modules), 'full_path': True,
             'action_mode': 'first_frame', 'steps': 20, 'horizon': screen.horizon})
        screen.q.weight_quantize()
        if args.mode == 'preflight':
            datum = data[0]
            outputs = {}
            for arm, mode, phases in [('w4', 'w4', {}), ('rtn', 'rtn', {}), ('shared0', 'phase', {}),
                                     ('phase_half_action', 'phase', {s: .5 if screen.stream(s) == 'action' else 0 for s in screen.expected_sites}),
                                     ('independent', 'independent', {})]:
                pred, stats, seconds = screen.infer(datum, mode, 2026, 1101, phases)
                outputs[arm] = {'mse': float((pred - datum['refs'][2026]).square().mean()),
                                'stats': {k: stats[k] for k in ('nonfinite_values', 'overload_codes', 'zero_rows', 'quantized_calls', 'site_count')},
                                'forward_seconds': seconds, 'action': pred.tolist()}
            save(args.out / 'preflight.json', {'engineering_pass': True, 'outputs': outputs,
                 'site_count': len(screen.expected_sites), 'forward_counts': dict(screen.forward_count)})
            print('PREFLIGHT_PASS', flush=True)
        else:
            pilot(screen, data)
    finally:
        screen.env.close()


if __name__ == '__main__':
    main()
