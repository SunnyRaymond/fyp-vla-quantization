"""Frozen paired LeWM block-local pilot. Numerical work requires live PBS."""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import sys
import time

ARMS = ('balanced_base', 'flat_adapter', 'block_adapter_global16',
        'block_adapter_local16', 'block_identity_global16')
PRIMARY = 'block_adapter_global16'
METRICS = ('spearman', 'top30_recall', 'standardized_elite_regret',
           'relative_latent_mse', 'normalized_response_mse', 'response_cosine', 'response_scale')


def guard():
    host = platform.node().split('.')[0].lower()
    job = os.environ.get('PBS_JOBID', '')
    nodefile = Path(os.environ.get('PBS_NODEFILE', '/missing'))
    if not job or not nodefile.is_file() or any(x in host for x in ('login', 'head', 'submit')):
        raise RuntimeError('Real PBS compute allocation required before data/model/tensor operations')
    nodes = {x.split('.')[0].lower() for x in nodefile.read_text().split()}
    if host not in nodes:
        raise RuntimeError('Current host absent from PBS_NODEFILE')
    return host


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dependencies(freeze):
    transfer = Path(freeze['source']['balanced_rows_path']).parents[3]
    # The cache is TRANSFER/teacher-screening/artifacts/JOB/cache.pt.
    path = transfer / 'terminal-response-loss' / 'runner.py'
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location('terminal_response_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = module.load_reference_modules()
    return module, reference, transfer


def make_student(arm, seed, reference, models, state=None):
    import torch
    torch.manual_seed(seed)
    student = reference.instantiate_student('balanced_base') if arm == 'balanced_base' else models.build_model(arm)
    if state is not None:
        student.load_state_dict(state, strict=True)
    return student.to('cuda')


def select_episodes(freeze, utility, output):
    import h5py
    source, evaluation = freeze['source'], freeze['evaluation']
    used_ids, used_metadata = utility.read_used_episode_ids(source['used_episode_manifests'])
    manifest = read(source['phase2_manifest_path'])
    train = [int(x['episode_id']) for x in manifest['splits']['train']]
    heldout = [int(x['episode_id']) for x in manifest['splits']['heldout']]
    with h5py.File(source['dataset_path'], 'r') as handle:
        lengths = [int(x) for x in handle['ep_len'][:]]
    valid = [i for i, length in enumerate(lengths) if length >= 26]
    random.Random(evaluation['selection_seed']).shuffle(valid)
    if valid[:520] != heldout + train:
        raise ValueError('Fresh shuffle does not reproduce authoritative Phase2 prefix')
    excluded = set(train + heldout)
    for values in used_ids.values():
        excluded.update(values)
    selected = [ep for ep in valid[evaluation['selection_start']:] if ep not in excluded][:evaluation['episodes']]
    if len(selected) != evaluation['episodes'] or len(set(selected)) != len(selected):
        raise ValueError('Insufficient distinct fresh episodes; no shrink/duplication allowed')
    record = {'ordered_episode_ids': selected, 'valid_count': len(valid),
              'selection_start': evaluation['selection_start'], 'selection_seed': evaluation['selection_seed'],
              'used_manifests': used_metadata, 'train_disjoint': not bool(set(selected) & set(train)),
              'result_dependent_selection': False, 'prior_shuffle_prefix_skipped': evaluation['selection_start']}
    write(output / 'selection.json', record)
    return selected, record


def fresh_rows(freeze, reference, official, episode_ids):
    # Existing builder validates batches of exactly 8 episodes / 24 anchors.
    old_seeds, old_anchors = reference.ema.FRESH_SEEDS, reference.ema.FRESH_ANCHORS
    reference.ema.FRESH_SEEDS = tuple(freeze['evaluation']['action_prefix_seeds'])
    reference.ema.FRESH_ANCHORS = tuple(freeze['evaluation']['anchors'])
    rows, batches = [], []
    try:
        for start in range(0, len(episode_ids), 8):
            batch, meta = reference.ema.build_fresh_rows(official, Path(freeze['source']['dataset_path']),
                                                       episode_ids[start:start + 8], freeze)
            rows.extend(batch)
            batches.append(meta)
    finally:
        reference.ema.FRESH_SEEDS, reference.ema.FRESH_ANCHORS = old_seeds, old_anchors
    if len(rows) != len(episode_ids) * 3 or len({x['context_id'] for x in rows}) != len(rows):
        raise ValueError('Fresh episode/anchor coverage differs')
    if any(tuple(row['latent_history'].shape) != (1, 1, 192) for row in rows):
        raise ValueError('Fresh pilot must start from H1 with no previous action token')
    return rows, {'episodes': len(episode_ids), 'contexts': len(rows), 'blocks': len(rows) * 2, 'batches': batches}


def train(freeze, reference, official, rows, arm, seed, models, output, completed):
    import torch
    student = make_student(arm, seed, reference, models).train()
    counts = {'total': sum(p.numel() for p in student.parameters()),
              'trainable': sum(p.numel() for p in student.parameters() if p.requires_grad)}
    optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad),
                                 lr=3e-4, weight_decay=.01, betas=(.9, .999), eps=1e-8)
    torch.manual_seed(seed + freeze['training']['training_seed_offset'])
    schedule = torch.Generator(device='cpu').manual_seed(seed + freeze['training']['schedule_seed_offset'])
    history, started = [], time.monotonic()
    for step in range(1, freeze['training']['updates'] + 1):
        indices = torch.randperm(len(rows), generator=schedule)[:8]
        context = torch.cat([rows[int(i)]['latent_history'].expand(64, -1, -1) for i in indices])
        actions = torch.cat([rows[int(i)]['future_actions'] for i in indices])
        targets = torch.cat([rows[int(i)]['teacher_targets'] for i in indices])
        prediction = student(context, actions)
        latent_loss, per_horizon = reference.base.recurrent_loss(prediction, targets)
        student_cost = reference.score._student_costs_with_gradient(official, rows, indices, prediction)
        teacher_cost = reference.score._effective_teacher_costs(rows, indices, 'score_distill')
        score_loss = reference.score.score_distill_loss(student_cost, teacher_cost)
        loss = latent_loss + .1 * score_loss
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(per_horizon).all()):
            raise FloatingPointError(f'Nonfinite training loss seed={seed} arm={arm} step={step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step == 1 and not all(bool(torch.isfinite(p.grad).all()) for p in student.parameters() if p.grad is not None):
            raise FloatingPointError('Nonfinite gradient')
        optimizer.step()
        item = {'step': step, 'latent_loss': float(latent_loss.detach()),
                'score_loss': float(score_loss.detach()), 'loss': float(loss.detach()),
                'per_horizon_mse': per_horizon.detach().cpu().tolist()}
        history.append(item)
        if step % 250 == 0:
            write(output / 'progress.json', {'stage': 'train', 'seed': seed, 'arm': arm, 'step': step,
                                             'completed_fits': completed, 'expected_fits': 15,
                                             'elapsed_s': time.monotonic() - started})
            print(json.dumps({'seed': seed, 'arm': arm, 'step': step, 'loss': item['loss']}), flush=True)
    state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    torch.save({'schema': freeze['schema'] + '.checkpoint', 'seed': seed, 'arm': arm,
                'step': 3000, 'state_dict': state, 'training_rows': freeze['source']['balanced_rows_path']},
               output / f'{seed}_{arm}_step3000.pt')
    write(output / f'training_{seed}_{arm}.json', {'history': history})
    ratio = statistics.median(x['latent_loss'] for x in history[-10:]) / max(history[0]['latent_loss'], 1e-12)
    result = {'arm': arm, 'seed': seed, 'steps': len(history), 'parameters': counts,
              'seconds': time.monotonic() - started, 'first': history[0], 'terminal': history[-1],
              'last10_to_first_ratio': ratio, 'finite': True}
    del student, optimizer
    torch.cuda.empty_cache()
    return state, result


def summarize(blocks):
    by_episode = defaultdict(list)
    for block in blocks:
        by_episode[str(block['episode_id'])].append(block)
    if len(blocks) != 96 or len(by_episode) != 16 or any(len(v) != 6 for v in by_episode.values()):
        raise ValueError('Expected16 independent episodes, each6 nested blocks')
    medians = {ep: {metric: statistics.median(b[metric] for b in items) for metric in METRICS}
               for ep, items in by_episode.items()}
    return {'block_medians': {m: statistics.median(b[m] for b in blocks) for m in METRICS},
            'episode_medians': {m: statistics.median(v[m] for v in medians.values()) for m in METRICS},
            'episode_values': medians,
            'by_stratum': {s: {m: statistics.median(b[m] for b in blocks if b['stratum'] == s) for m in METRICS}
                           for s in ('early', 'middle', 'late')},
            'minimum_spearman': min(b['spearman'] for b in blocks),
            'minimum_top30': min(b['top30_recall'] for b in blocks),
            'response_floor_blocks': sum(b['response_floor_active'] for b in blocks),
            'worst_top30': sorted(blocks, key=lambda b: b['top30_recall'])[:3]}


def evaluate(freeze, utility, reference, official, student, rows):
    import torch
    student.eval()
    if hasattr(student, 'prepare_for_inference'):
        with torch.no_grad():
            student.prepare_for_inference()
    blocks = []
    for row in rows:
        for index, seed in enumerate(freeze['evaluation']['action_prefix_seeds']):
            item = utility.evaluate_block(reference, official, student, row, index, seed)
            # Reuse legacy response/fidelity metrics, but use the actual native
            # topk selector for the two planner-facing primary metrics.
            with torch.no_grad():
                actions = row['future_actions'][index].to('cuda')
                context = row['latent_history'].to('cuda').expand(300, -1, -1)
                prediction = student(context, actions)
                costs = reference.base._official_objective(official, row, prediction)
                teacher_costs = row['teacher_objective'][index].to('cuda')
                teacher_top = torch.topk(teacher_costs, k=30, largest=False).indices
                student_top = torch.topk(costs, k=30, largest=False).indices
                item['top30_recall'] = float(torch.isin(teacher_top, student_top).sum()) / 30
                item['standardized_elite_regret'] = float((teacher_costs[student_top].mean() -
                    teacher_costs[teacher_top].mean()) / teacher_costs.std(unbiased=False).clamp_min(1e-6))
                item['elite_selector'] = 'native torch.topk largest=False default sorted'
            blocks.append(item)
    return blocks, summarize(blocks)


def timing(freeze, reference, official, student, row, order_seed):
    import numpy as np
    import torch
    student.eval()
    if hasattr(student, 'prepare_for_inference'):
        with torch.no_grad():
            student.prepare_for_inference()
    results = {}
    for batch in freeze['evaluation']['latency']['batch_sizes']:
        actions = row['future_actions'][0, :batch].to('cuda')
        context = row['latent_history'].to('cuda').expand(batch, -1, -1)
        functions = {'student': lambda: student(context, actions),
                     'teacher': lambda: reference.base.official_teacher_targets(official, context, actions)}
        samples = {key: [] for key in functions}
        with torch.no_grad():
            for _ in range(freeze['evaluation']['latency']['warmup']):
                for fn in functions.values():
                    fn()
            for repeat in range(freeze['evaluation']['latency']['repeats']):
                order = ('student', 'teacher') if (repeat + order_seed) % 2 == 0 else ('teacher', 'student')
                for key in order:
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    functions[key]()
                    end.record()
                    end.synchronize()
                    samples[key].append(begin.elapsed_time(end))
        results[str(batch)] = {key: {'median_ms': float(np.median(value)), 'p90_ms': float(np.quantile(value, .9))}
                               for key, value in samples.items()}
        results[str(batch)]['teacher_reduction'] = 1 - results[str(batch)]['student']['median_ms'] / results[str(batch)]['teacher']['median_ms']
    return results


def paired(freeze, raw, control, treatment):
    import numpy as np
    seed_results, episode_seed_values = {}, defaultdict(lambda: defaultdict(list))
    for seed in freeze['training']['seeds']:
        left = {b['pairing_key']: b for b in raw[f'{seed}/{control}']}
        right = {b['pairing_key']: b for b in raw[f'{seed}/{treatment}']}
        if set(left) != set(right) or len(left) != 96:
            raise ValueError('Pairing failed')
        by_ep = defaultdict(lambda: defaultdict(list))
        for key in left:
            for metric in METRICS:
                by_ep[str(left[key]['episode_id'])][metric].append(right[key][metric] - left[key][metric])
        per_ep = {ep: {metric: statistics.median(v[metric]) for metric in METRICS} for ep, v in by_ep.items()}
        seed_results[str(seed)] = {'episode_deltas': per_ep,
                                 'median_deltas': {m: statistics.median(v[m] for v in per_ep.values()) for m in METRICS}}
        for ep, values in per_ep.items():
            for metric, delta in values.items():
                episode_seed_values[ep][metric].append(delta)
    aggregate = {ep: {m: statistics.median(values[m]) for m in METRICS} for ep, values in episode_seed_values.items()}
    rng = np.random.default_rng(freeze['evaluation']['bootstrap_seed'])
    identities = sorted(aggregate)
    resample = rng.integers(0, len(identities), size=(freeze['evaluation']['bootstrap_repeats'], len(identities)))
    intervals = {}
    for metric in METRICS:
        values = np.array([aggregate[ep][metric] for ep in identities])
        ci = np.quantile(np.median(values[resample], axis=1), [.025, .975])
        intervals[metric] = {'median_delta': float(np.median(values)), 'ci95': ci.tolist()}
    return {'control': control, 'treatment': treatment, 'per_seed': seed_results,
            'episode_deltas': aggregate, 'aggregate': intervals,
            'unit': 'episode; three training seeds fixed, six nested blocks'}


def decisions(freeze, summaries, comparisons, mechanics):
    comparison = comparisons['block_vs_flat']
    aggregate, thresholds = comparison['aggregate'], freeze['gates']['relative']
    conditions = {
        'regret_effect': aggregate['standardized_elite_regret']['median_delta'] <= thresholds['elite_regret_delta_max'],
        'top30_effect': aggregate['top30_recall']['median_delta'] >= thresholds['top30_delta_min'],
        'regret_ci': aggregate['standardized_elite_regret']['ci95'][1] < 0,
        'top30_ci': aggregate['top30_recall']['ci95'][0] > 0,
        'improved_episodes': sum(v['standardized_elite_regret'] < 0 for v in comparison['episode_deltas'].values()) >= 10,
        'all_seeds_nonworse': all(v['median_deltas']['standardized_elite_regret'] <= 0 and
                                  v['median_deltas']['top30_recall'] >= 0 for v in comparison['per_seed'].values()),
        'no_response_floor': all(summaries[f'{seed}/{PRIMARY}']['evaluation']['response_floor_blocks'] == 0
                                 for seed in freeze['training']['seeds']),
        'integrity': bool(mechanics['passed']),
    }
    per_seed = {}
    absolute = freeze['gates']['absolute']
    for seed in freeze['training']['seeds']:
        item = summaries[f'{seed}/{PRIMARY}']
        result = item['evaluation']
        median = result['block_medians']
        fidelity = (median['spearman'] >= absolute['median_spearman_min'] and
                    result['minimum_spearman'] >= absolute['minimum_spearman_min'] and
                    median['top30_recall'] >= absolute['median_top30_min'] and
                    result['minimum_top30'] >= absolute['minimum_top30_min'] and
                    median['relative_latent_mse'] <= absolute['median_relative_latent_mse_max'] and
                    all(v['spearman'] >= .95 and v['top30_recall'] >= .75 for v in result['by_stratum'].values()))
        cheap = item['latency']['300']['teacher_reduction'] >= .2
        per_seed[str(seed)] = {'absolute_fidelity': fidelity, 'cheap': cheap,
                              'causality': item['causality_passed'], 'converged': item['training']['last10_to_first_ratio'] <= .8}
    relative = all(conditions.values())
    ready_b = relative and all(v['cheap'] and v['causality'] and v['converged'] for v in per_seed.values())
    return {'relative_quality': 'GO' if relative else 'NO-GO', 'conditions': conditions,
            'per_seed': per_seed, 'stage_b_eligible': ready_b,
            'stage_c_predictor_eligible': ready_b and all(v['absolute_fidelity'] for v in per_seed.values()),
            'claim': 'predictor quality evidence; no task success claim without closed-loop'}


def write_readable(result, output):
    brief = {key: value for key, value in result.items() if key not in ('arms', 'comparisons', 'fresh_evaluation')}
    brief['arms'] = {}
    for key, value in result['arms'].items():
        item = {name: part for name, part in value.items() if name != 'evaluation'}
        item['evaluation'] = {name: part for name, part in value['evaluation'].items() if name != 'episode_values'}
        brief['arms'][key] = item
    brief['comparisons'] = {}
    for key, value in result['comparisons'].items():
        item = {name: part for name, part in value.items() if name != 'per_seed'}
        item['per_seed'] = {seed: {'median_deltas': stats['median_deltas']} for seed, stats in value['per_seed'].items()}
        brief['comparisons'][key] = item
    write(output / 'summary_brief.json', brief)
    lines = ['# LeWM 分块 predictor 配对结果', '',
             f"PBS `{result['pbs_job_id']}`，{result['runtime']['gpu']}；完成 {result['completed_fits']}/15 fits。",
             '', f"**主相对质量 gate：{result['decision']['relative_quality']}。**",
             '指标均为真实LeWM固定候选动作bank的predictor评价，不是任务成功率。', '',
             '192D状态保维正交换坐标、6×32D局部更新、16D消息；Flat与分块保留相同三步latent与已消费action窗口。旧balanced_base仅直接读当前action，因此与旧模型的差异不能纯归分块。', '',
             '|Seed|模型|Top30 recall median|Elite regret median|Response MSE median|B300 predictor ms|',
             '|---|---|---:|---:|---:|---:|']
    for key, item in result['arms'].items():
        seed, arm = key.split('/')
        m = item['evaluation']['block_medians']
        lines.append(f"|{seed}|{arm}|{m['top30_recall']:.4f}|{m['standardized_elite_regret']:.4f}|{m['normalized_response_mse']:.4f}|{item['latency']['300']['student']['median_ms']:.3f}|")
    lines.extend(['', '## 主比较与诊断对照', '',
                  '每个episode先对六个blocks取配对差值median，再对三个固定training seeds取median；以下bootstrap以16个episode为单位。candidate不是独立实验单位。', '',
                  '|对照（treatment-control）|Top30 delta / 95%CI|Elite regret delta / 95%CI|',
                  '|---|---|---|'])
    for name, comparison in result['comparisons'].items():
        a = comparison['aggregate']
        top, regret = a['top30_recall'], a['standardized_elite_regret']
        lines.append(f"|{name}|{top['median_delta']:+.4f} / {top['ci95']}|{regret['median_delta']:+.4f} / {regret['ci95']}|")
    lines.extend(['', '## 冻结判据与边界', ''])
    for name, passed in result['decision']['conditions'].items():
        lines.append(f'- {name}: {"PASS" if passed else "FAIL"}')
    lines.extend(['', f"Stage B: {result['stage_b']['status']}；Stage C: {result['stage_c']['status']}。",
                  '未运行的CEM/闭环不能由latent误差或候选bank排序替代。相对改善不替代绝对fidelity门槛；固定三个training seeds不代表训练随机性总体。', '',
                  '正交adapter参数包含在预算；identity诊断arm少36864个可训练参数。推理计时计入实际旋转、消息和反变换，Q作为冻结权重物化；排除encoder/CEM/environment。', '',
                  '数据：直接复用25213164 reconstructed balanced rows；新的16个episodes来自冻结shuffle1000之后，排除训练及所列旧任务manifest。',
                  result['preprocessing_boundary'], '',
                  result['source']['source_version_boundary'], '',
                  '原始summary.json、逐arm evaluation JSON、training JSON/checkpoints和fresh_rows.pt均保留于本PBS run目录；本地仅取回小型摘要/报告。', '',
                  '方法来源：Kassis et al. (2026), [Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065)，当前v2，实验设计流程来源。'])
    (output / 'RESULTS.zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(args):
    host = guard()
    freeze, output = read(args.freeze), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if freeze['schema'] != 'lewm-block-local-pilot-v1' or tuple(freeze['arms']) != ARMS:
        raise ValueError('Freeze/arm identity mismatch')
    if freeze['training']['updates'] != 3000 or freeze['training']['contexts'] != 512:
        raise ValueError('Training contract drift')
    import torch
    import models
    import model_checks
    from stage_b import mechanism_selfcheck
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    stage_b_checks = mechanism_selfcheck(freeze)
    write(output / 'stage_b_mechanism_preflight.json', stage_b_checks)
    if stage_b_checks['status'] != 'PASS':
        raise RuntimeError(f"Stage B native-update mechanism check failed: {stage_b_checks['checks']}")
    utility, reference, transfer = dependencies(freeze)
    reference_snapshot = output / 'reference_snapshot'
    reference_snapshot.mkdir(exist_ok=True)
    for module in (utility, reference, reference.base, reference.score, reference.temporal,
                   reference.ema, reference.score.dense if hasattr(reference.score, 'dense') else reference.score):
        source_file = Path(module.__file__)
        shutil.copy2(source_file, reference_snapshot / f'{source_file.parent.name}_{source_file.name}')
    probe_student = make_student('balanced_base', freeze['training']['seeds'][0], reference, models)
    if sum(p.numel() for p in probe_student.parameters()) != freeze['model']['baseline_parameters']:
        raise ValueError('Reference balanced_base parameter count drift')
    del probe_student
    checks = model_checks.run_checks(device='cuda')
    if 'passed' not in checks:
        checks['passed'] = checks.get('status') in ('PASS', 'passed')
    write(output / 'mechanism_tests.json', checks)
    if not checks['passed']:
        raise RuntimeError('Model mechanism checks failed')
    rows_cpu = torch.load(freeze['source']['balanced_rows_path'], map_location='cpu', weights_only=False)
    rows_cpu = utility.validate_training_rows(rows_cpu, Path(freeze['source']['phase2_manifest_path']))
    if any(tuple(row['latent_history'].shape) != (1, 1, 192) for row in rows_cpu):
        raise ValueError('Training pilot must start from H1 with no previous action token')
    reference.base.load_interface_contract(Path(freeze['source']['interface_probe_path']))
    rows = [{key: value.to('cuda') if torch.is_tensor(value) else value for key, value in row.items()} for row in rows_cpu]
    official = reference.base.load_official_checkpoint(Path(freeze['source']['stablewm_home']))
    official.requires_grad_(False)
    ids, selection = select_episodes(freeze, utility, output)
    evaluation_rows, evaluation_meta = fresh_rows(freeze, reference, official, ids)
    torch.save(evaluation_rows, output / 'fresh_rows.pt')
    summaries, raw, states, completed = {}, {}, {}, 0
    for index, seed in enumerate(freeze['training']['seeds']):
        order = list(ARMS[index:]) + list(ARMS[:index])
        for arm in order:
            key = f'{seed}/{arm}'
            state, training = train(freeze, reference, official, rows, arm, seed, models, output, completed)
            student = make_student(arm, seed, reference, models, state)
            blocks, evaluation = evaluate(freeze, utility, reference, official, student, evaluation_rows)
            coordinates = None
            if hasattr(student, 'rotation'):
                q = student.rotation._cached_q
                eye = torch.eye(192, device=q.device)
                coordinates = {'orthogonality_max_abs': float((q.T @ q - eye).abs().max()),
                               'distance_from_identity_rms': float((q - eye).square().mean().sqrt()),
                               'identity_frozen': arm == 'block_identity_global16'}
                if coordinates['orthogonality_max_abs'] > 2e-4:
                    raise ValueError('Trained coordinate transform lost orthogonality')
            causality = reference.base.causality_test(student, evaluation_rows[0], freeze['evaluation']['action_prefix_seeds'][0], 1e-6)
            causal_pass = all(v['passed'] for v in causality.values())
            latency = timing(freeze, reference, official, student, evaluation_rows[0], index)
            write(output / f'evaluation_{seed}_{arm}.json', {'blocks': blocks, 'summary': evaluation})
            summaries[key] = {'training': training, 'evaluation': evaluation, 'causality': causality,
                              'causality_passed': causal_pass, 'latency': latency, 'coordinates': coordinates}
            states[key], raw[key] = state, blocks
            completed += 1
            write(output / 'summary_progress.json', {'completed_fits': completed, 'expected_fits': 15,
                                                      'current_seed': seed, 'current_arm': arm})
            del student
            torch.cuda.empty_cache()
    pairs = {'block_vs_flat': ('flat_adapter', PRIMARY),
             'communication': ('block_adapter_local16', PRIMARY),
             'coordinates': ('block_identity_global16', PRIMARY),
             'old_recipe': ('balanced_base', PRIMARY)}
    comparisons = {name: paired(freeze, raw, left, right) for name, (left, right) in pairs.items()}
    decision = decisions(freeze, summaries, comparisons, checks)
    parameter_checks = {arm: summaries[f"{freeze['training']['seeds'][0]}/{arm}"]['training']['parameters'] for arm in ARMS}
    if parameter_checks['block_adapter_global16'] != parameter_checks['block_adapter_local16']:
        raise ValueError('Global/local parameter budget mismatch')
    for arm in ('flat_adapter', 'block_adapter_global16', 'block_adapter_local16'):
        if abs(parameter_checks[arm]['total'] / 775872 - 1) > .02:
            raise ValueError('Matched student exceeds frozen parameter tolerance')
    stage_b = {'status': 'NOT_RUN_GATE_FAILED'}
    if decision['stage_b_eligible']:
        from stage_b import run_stage_b
        factories = {arm: (lambda arm=arm: make_student(arm, freeze['contingent']['stage_b_first_training_seed'],
                     reference, models, states[f"{freeze['contingent']['stage_b_first_training_seed']}/{arm}"]))
                     for arm in ('flat_adapter', PRIMARY)}
        stage_b = run_stage_b(reference, official, factories, evaluation_rows, freeze, output)
    result = {'schema': freeze['schema'] + '.summary', 'pbs_job_id': os.environ['PBS_JOBID'], 'host': host,
              'runtime': {'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
                          'precision': 'float32 TF32 off'},
              'source': freeze['source'], 'selection': selection, 'fresh_evaluation': evaluation_meta,
              'parameters': parameter_checks, 'arms': summaries, 'comparisons': comparisons,
              'decision': decision, 'stage_b': stage_b,
              'stage_c': {'status': 'PENDING_TRIGGER_CHECK' if decision['stage_c_predictor_eligible'] and stage_b.get('absolute_passed')
                          else 'NOT_RUN_GATE_FAILED'},
              'completed_fits': completed, 'expected_fits': 15,
              'training_action_information': 'new flat/block have recent3 consumed actions; old recipe currentaction only',
              'preprocessing_boundary': freeze['source']['preprocessing_isolation'],
              'statistics': 'episode paired; nested anchors/candidates; fixed3 training seeds'}
    write(output / 'summary.json', result)
    write_readable(result, output)
    write(output / 'DECISION.json', {'decision': decision, 'stage_b': stage_b, 'stage_c': result['stage_c']})
    write(output / 'DONE.json', {'status': 'STAGE_A_COMPLETE', 'completed_fits': completed,
                                'stage_b': stage_b['status'], 'stage_c': result['stage_c']['status']})
    write(output / 'progress.json', {'stage': 'complete', 'completed_fits': completed, 'expected_fits': 15})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    arguments = parser.parse_args()
    try:
        result = run(arguments)
        print(json.dumps({'status': 'COMPLETE', 'relative_quality': result['decision']['relative_quality']}), flush=True)
    except Exception as exc:
        if os.environ.get('PBS_JOBID'):
            write(arguments.output_dir / 'FAILURE.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise
