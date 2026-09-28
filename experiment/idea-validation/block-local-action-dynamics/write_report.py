"""Build the complete pilot report on compute; keep episode arrays on compute."""
import argparse
import itertools
import json
from pathlib import Path
from statistics import mean

from allocation_guard import ensure_allocation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ensure_allocation(require_gpu=True)
    out = args.output
    summary = json.loads((out / 'summary.json').read_text())
    config = json.loads((out / 'FREEZE.json').read_text())
    runs = summary['runs']
    if len(runs) != 45:
        raise ValueError('Primary report requires all 45 runs')
    groups = {(c, a): [r for r in runs if r['condition'] == c and r['arm'] == a]
              for c, a in itertools.product(config['conditions'], config['arms'])}
    table = []
    for (condition, arm), rows in groups.items():
        table.append({
            'condition': condition, 'arm': arm,
            'h10': mean(r['test']['primary_episode_mean'] for r in rows),
            'h20': mean(r['test']['horizon_episode_aggregates']['20']['mean'] for r in rows),
            'action_response': mean(r['test']['action_response_true_energy_normalized_mean'] for r in rows),
            'batch1_ms': mean(r['latency']['complete_rollout_final_batch1_ms']['median'] for r in rows),
            'batch300_ms': mean(r['latency']['complete_rollout_final_batch300_ms']['median'] for r in rows),
            'eachstep300_ms': mean(r['latency']['each_step_reconstruction_batch300_ms']['median'] for r in rows),
            'parameters': rows[0]['parameter_count'],
            'quality_pass_seeds': sum(r['test']['dense_quality_gate_pass'] for r in rows),
        })
    lookup = {(r['condition'], r['arm']): r for r in table}
    dev_trigger = all(
        next(r for r in groups[('lowrank_coupled', 'learned_global4')] if r['training_seed'] == seed)['dev']['primary_episode_mean']
        < next(r for r in groups[('lowrank_coupled', control)] if r['training_seed'] == seed)['dev']['primary_episode_mean']
        for seed in config['training']['seeds'] for control in ('learned_local4', 'learned_block')
    )
    decision = {'global16_dev_trigger': dev_trigger, 'expected_primary_runs': 45, 'actual_primary_runs': len(runs)}
    (out / 'DECISION.json').write_text(json.dumps(decision, indent=2), encoding='utf-8')
    lines = [
        '# 第一轮受控系统实验结果', '',
        '本轮完整执行 3 个 conditions × 5 个 arms × 3 个 training seeds，共 45 次训练。这里只回答受控 64D 系统中的结构与计算问题，不支持视觉任务、CEM 或 closed-loop 结论。', '',
        '模型用完整正交变换保留 64D 状态，分成四个 16D 块。每个块接收同一个已知 8D action；global4 另接收四维全局状态摘要。local4 的参数量与输入宽度相同，摘要只读取本块。', '',
        '训练固定 1500 steps；最后 checkpoint；train/dev/test episode 分离；同 seed 配对 minibatch draws。下表是三个 training seeds 各自 episode 汇总值的平均，区间请读后面的逐 seed 配对结果。', '',
        '| Condition | Arm | h10 error | h20 error | Action-response error | Params | Quality seeds | B1 ms | B300 ms | Each-step B300 ms |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in table:
        lines.append(f"| {row['condition']} | {row['arm']} | {row['h10']:.5f} | {row['h20']:.5f} | {row['action_response']:.4f} | {row['parameters']} | {row['quality_pass_seeds']}/3 | {row['batch1_ms']:.3f} | {row['batch300_ms']:.3f} | {row['eachstep300_ms']:.3f} |")
    lines.extend(['', 'h10/h20 为 train delta energy 归一化的末步 MSE；Action-response 列以真实 action 扰动造成的状态差分 energy 归一化，越小越好。它是诊断，不替换 primary h10。延迟含初始变换、native rollout、最终重构；每步重构接口另列。Q 在部署前冻结缓存，缓存准备不计入每条 rollout。', '',
                  '## 相对 dense 的门槛', '',
                  '质量门槛为 h10 <= dense + max(0.1*dense, 0.02)。速度门槛为 complete rollout 至少快 20%，B1/B300 分开判断。', '',
                  '| Condition | Arm | B1 reduction | B300 reduction |', '|---|---|---:|---:|'])
    for row in table:
        ref = lookup[(row['condition'], 'dense')]
        lines.append(f"| {row['condition']} | {row['arm']} | {1-row['batch1_ms']/ref['batch1_ms']:.1%} | {1-row['batch300_ms']/ref['batch300_ms']:.1%} |")
    lines.extend(['', '## 逐 training seed 配对对照', '',
                  '差分 = left - right，负值表示 left 更好。Bootstrap 以 episode 为单位，在单个 training seed 内作描述；三个 training seeds 不支持对训练随机性的总体推断。', '',
                  '| Condition | Seed | Left - Right | Mean difference | Episode bootstrap 95% CI |', '|---|---:|---|---:|---|'])
    for row in summary['paired_episode_bootstrap_contrasts']:
        low, high = row['bootstrap_95pct_mean_ci']
        lines.append(f"| {row['condition']} | {row['training_seed']} | {row['left_arm']} - {row['right_arm']} | {row['mean_paired_difference']:.6f} | [{low:.6f}, {high:.6f}] |")
    lines.extend(['', '## 后续门槛与解释限制', '',
                  f"低秩条件 dev 上，global4 对 local4 和 learned-block 在全部三个 seeds 一致改善：{'是，触发预设 global16 附加对照' if dev_trigger else '否，不触发 global16 附加对照'}。", '',
                  '坐标 subspace overlap 仅作描述，块的排列与块内旋转不唯一；它不能代替 heldout dynamics error。参数量为优化器可训练张量元素数，包含正交参数化的冗余元素，不等于有效自由度或 FLOPs。', '',
                  '独立与 rank-4 条件由设计保证存在局部结构。稠密条件覆盖更广的状态交互，但没有证明所有坐标系下都不可分。一次固定训练预算的失败可能来自优化，也可能来自结构容量；本轮保留 NO-GO/inconclusive，不增加训练步数或放宽门槛。', '',
                  '## 运行与复现', '',
                  f"PBS job：{summary['allocation']['job_id']}；compute host：{summary['allocation']['hostname']}。", '',
                  '必要机制测试见 mechanism_tests.json；完整配置见 FREEZE.json；逐 run 数值、延迟和 seed 对照见 summary.json。Checkpoints、per-episode arrays、training.tsv 与源代码 snapshot 保存在该 PBS run 目录。GPU 利用率和显存每 30 秒写入 job.log。', '',
                  '实验设计参考：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065', ''])
    (out / 'REPORT.zh.md').write_text('\n'.join(lines), encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
        for ax, condition in zip(axes, config['conditions']):
            for ai, arm in enumerate(config['arms']):
                rows = groups[(condition, arm)]
                values = [r['test']['primary_episode_mean'] for r in rows]
                ax.bar(ai, mean(values), color='C'+str(ai), alpha=.65)
                ax.scatter([ai]*len(values), values, color='black', s=15)
            ax.set_title(condition)
            ax.set_xticks(range(5), ['dense', 'random', 'learned', 'global4', 'local4'], rotation=35)
            ax.set_ylabel('Test h10 / train delta energy')
            ax.grid(axis='y', alpha=.2)
        fig.tight_layout()
        fig.savefig(out / 'h10_comparison.svg')
        plt.close(fig)
    except ImportError:
        (out / 'plot_status.txt').write_text('Matplotlib unavailable; all numerical evidence and report retained.\n')
    print(json.dumps(decision))


if __name__ == '__main__':
    main()
