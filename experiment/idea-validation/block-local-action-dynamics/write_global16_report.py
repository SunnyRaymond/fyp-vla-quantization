"""Compute-only comparison of the preauthorized additional global16 contrast."""
import argparse
import json
import random
from pathlib import Path

from allocation_guard import ensure_allocation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    allocation = ensure_allocation(require_gpu=True)
    import numpy as np
    import torch
    from controlled_system import make_datasets
    from models import build_model
    from run_experiment import _benchmark, _bootstrap_contrast, _evaluate_rollouts

    out = args.output
    config = json.loads((out/'FREEZE.json').read_text())
    parent = Path(config['primary_reference']['remote_run'])
    primary = json.loads((parent/'summary.json').read_text())
    parent_config = json.loads((parent/'FREEZE.json').read_text())
    additional = json.loads((out/'summary.json').read_text())
    assert len(additional['runs']) == 3 and all(r['arm']=='learned_global16' for r in additional['runs'])
    system, datasets, metadata = make_datasets(config, 'lowrank_coupled', 'cuda')
    references = {(r['arm'],r['training_seed']):r for r in primary['runs'] if r['condition']=='lowrank_coupled'}
    rows, contrasts, rebased = [], [], []
    for extra in additional['runs']:
        seed = extra['training_seed']
        assert abs(extra['train_delta_energy'] - references[('dense',seed)]['train_delta_energy']) < 1e-8
        extra_npz = np.load(out/extra['episode_files']['test'])
        extra_error = np.nanmean(extra_npz['terminal_error'][:,2,:],axis=1)
        extra_npz.close()
        for control in ('dense','learned_global4','learned_local4','learned_block'):
            ref = references[(control,seed)]
            with np.load(parent/ref['episode_files']['test']) as arrays:
                reference_error = np.nanmean(arrays['terminal_error'][:,2,:],axis=1)
            contrast = _bootstrap_contrast(extra_error,reference_error,seed+260926)
            contrasts.append(dict(training_seed=seed,reference_arm=control,**contrast))
        # Rebenchmark the original checkpoints and global16 in this same allocation.
        timings = {}
        arms = ['dense','learned_global4','learned_global16']
        random.Random(seed+92026).shuffle(arms)
        for arm in arms:
            ref = extra if arm=='learned_global16' else references[(arm,seed)]
            root = out if arm=='learned_global16' else parent
            cfg = config if arm=='learned_global16' else parent_config
            checkpoint = torch.load(root/ref['checkpoint'],map_location='cpu',weights_only=True)
            model = build_model(arm,cfg,seed,checkpoint['state_dict']['mean'],dense_hidden=checkpoint['dense_hidden']).to('cuda')
            model.load_state_dict(checkpoint['state_dict'])
            model.eval().prepare_inference()
            if arm!='learned_global16':
                rechecked = _evaluate_rollouts(torch,model,datasets['test'],[10],[0,10,20],40,metadata['delta_energy'],128)
                with np.load(parent/ref['episode_files']['test']) as arrays:
                    np.testing.assert_allclose(rechecked[:,0,:],arrays['terminal_error'][:,2,:],atol=1e-6,rtol=1e-5)
                rebased.append({'training_seed':seed,'arm':arm,'original_episode_errors_reproduced':True})
            timings[arm] = _benchmark(torch,model,datasets['test'],[1,300],10,20,60)
            del model
        dense_error = references[('dense',seed)]['test']['primary_episode_mean']
        quality_pass = extra['test']['primary_episode_mean'] <= dense_error + max(0.1*dense_error,0.02)
        speed_pass = {str(batch):timings['learned_global16'][f'complete_rollout_final_batch{batch}_ms']['median'] <=
                      0.8*timings['dense'][f'complete_rollout_final_batch{batch}_ms']['median'] for batch in (1,300)}
        rows.append({'training_seed':seed,'global16_error':extra['test']['primary_episode_mean'],
                     'global4_error':references[('learned_global4',seed)]['test']['primary_episode_mean'],
                     'dense_error':references[('dense',seed)]['test']['primary_episode_mean'],
                     'parameters':extra['parameter_count'],'same_allocation_latency':timings,
                     'q_alignment':extra['q_alignment'], 'train_seconds':extra['train_seconds'],
                     'peak_memory_mib':extra['peak_memory_mib'],'quality_gate_pass':quality_pass,
                     'speed_gate_pass_by_batch':speed_pass})
    comparison = {'scope':'additional_global16_lowrank_only','allocation':allocation,'primary_job':config['primary_reference']['job_id'],
                  'trigger':'primary dev only; all three seeds global4 better than local4 and learned_block',
                  'runs':rows,'paired_episode_bootstrap_contrasts':contrasts,'reference_reproduction_checks':rebased,
                  'runtime':{'torch':str(torch.__version__),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
                             'matmul_tf32':torch.backends.cuda.matmul.allow_tf32},
                  'limitation':'global16 adds parameters; no global16/local16 capacity-matched training contrast'}
    (out/'global16_comparison.json').write_text(json.dumps(comparison,separators=(',',':')),encoding='utf-8')
    lines = ['# global16 附加对照', '',
             '该对照由主实验低秩 dev 的预设门槛触发，只增加 lowrank_coupled 的三个 seeds；同 data、loss、1500 steps 和最后 checkpoint。45 个 primary runs 保留。', '',
             '| Seed | Dense h10 | Global4 h10 | Global16 h10 | Global16 params | B1: dense / g4 / g16 ms | B300: dense / g4 / g16 ms |',
             '|---:|---:|---:|---:|---:|---|---|']
    for row in rows:
        timing = row['same_allocation_latency']
        b1 = ' / '.join(f"{timing[a]['complete_rollout_final_batch1_ms']['median']:.3f}" for a in ('dense','learned_global4','learned_global16'))
        b300 = ' / '.join(f"{timing[a]['complete_rollout_final_batch300_ms']['median']:.3f}" for a in ('dense','learned_global4','learned_global16'))
        lines.append(f"| {row['training_seed']} | {row['dense_error']:.5f} | {row['global4_error']:.5f} | {row['global16_error']:.5f} | {row['parameters']} | {b1} | {b300} |")
    lines += ['', '原始 dense/global4 checkpoints 在本 allocation 重新计时，且逐 episode h10 error 与主实验记录一致。延迟包含初始坐标变换、native rollout 和最后重构。', '',
              '## 配对差分', '', '| Seed | Global16 - Reference | Mean difference | Episode bootstrap 95% CI |', '|---:|---|---:|---|']
    for row in contrasts:
        lo,hi = row['bootstrap_95pct_mean_ci']
        lines.append(f"| {row['training_seed']} | {row['reference_arm']} | {row['mean_paired_difference']:.6f} | [{lo:.6f}, {hi:.6f}] |")
    lines += ['', '## 解释边界', '',
              'global16 同时增加通信维数和参数量，没有训练 local16 等参对照，不能单独归因为通信带宽。其结果是附加探索，不替换 global4/local4 primary 对照，也不支持视觉任务或 CEM 结论。', '',
              '训练时长、Torch allocator peak memory、Q overlap、同 allocation latency 和逐 seed 配对差分见 global16_comparison.json。', '',
              '实验设计引用：Kassis et al. (2026). Scientific Agent Skills. https://doi.org/10.48550/arXiv.2609.00065', '']
    (out/'REPORT.zh.md').write_text('\n'.join(lines),encoding='utf-8')
    (out/'DECISION.json').write_text(json.dumps({'additional_runs':3,'reference_reproductions_pass':True,'no_further_extension':True},indent=2))
    print(json.dumps({'additional_comparison_complete':True,'runs':3}))


if __name__=='__main__':
    main()
