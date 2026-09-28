# LeWM terminal action-response loss：冻结协议

本实验检验一个单一假说：在保留 `balanced_base` 的 h256 serial rollout 和原始训练目标时，增加 bank-centered terminal action-response loss，能否改善 student 对不同 action candidates 的相对 latent response，并降低 fresh candidate banks 上的 teacher-objective elite regret。

协议与所有门槛冻结在 [FREEZE.json](FREEZE.json)。对照与 treatment 从同一原始初始化开始，训练数据、candidate slates、随机种子、optimizer 和 3000 updates 完全配对。唯一变化是 treatment 增加 `lambda_response=0.01` 的 terminal response loss。

## 两个训练臂

| 臂 | 训练目标 |
|---|---|
| `balanced_base` | 原有 horizon-weighted free-running latent MSE + `0.1 ×` context-normalized teacher-score SmoothL1 |
| `terminal_response_loss` | 同一目标 + `0.01 × L_response` |

两臂均复用 job `25213164.pbs101` 从 Phase 2 rows 重建并冻结的同一组 512 balanced rows 与 teacher targets：`early/middle/late=171/171/170`，每个 context 使用相同 64 条候选动作。新作业不重建这些 rows，也不重新计算训练 teacher targets。两臂使用相同 h256 初始化、AdamW、batch size 8、training/context/candidate seeds 与 3000 updates。teacher 冻结；推理时不引入 teacher。

对每个 context 的 64 个候选，以第五步 latent 为 terminal response。令 `c_i=z_i-mean_j(z_j)`，loss 为：

```text
L_response = mean_i mean_d[(cS_i - cT_i)^2]
             / max(mean_i mean_d[cT_i^2], 1e-6)
```

分母是 teacher bank-centered terminal latent MSE，`1e-6` floor 在训练前冻结。student 侧有梯度；teacher latent 与归一化分母 detach。不得新增架构、改变旧 loss 权重或做 `lambda` sweep。

## Episode 隔离与评估

评估 episode 从与历史流程相同的 `valid` 顺序生成：枚举 HDF5 `ep_len >= 26` 的 episode IDs，以 `random.Random(20300903)` shuffle 一次，从 `valid[600:]` 开始依序排除三份 frozen selection manifests 中的全部 IDs，然后取前 8 个。三份排除清单分别是 seeded-pilot 32 episodes、real-observation 80 episodes、closed-loop 50 episodes。selection 只读 episode metadata；PBS runner 必须在任何 fresh evaluation inference 或 scoring 前写出有序 episode IDs，并确认数量、顺序和排除关系正确。训练与 terminal checkpoint 保存先完成。若身份检查失败，fail closed，不替换 episode。

这一步是 metadata-only selection；不读取旧的 validation metrics。每个被选 episode 评估 early/middle/late 三个 anchors，每个 anchor 使用 action-prefix seeds `20301007` 和 `20301008`，每个 block 300 candidates，共 48 个 blocks。control 与 treatment 使用同一个 block 的 context、goal 和 candidate actions。所有 block 和 candidates 都嵌套在 episode 内，统计独立单位是 episode。

每个 block 计算：

- 300 个候选上的 normalized terminal response MSE；
- teacher-objective Spearman 与 teacher Top-30 recall；
- student scorer 选出的 Top-30 在 teacher objective 下的 standardized elite regret；
- 全部五个 horizon 的 relative latent MSE（沿用原 absolute gate 口径）。

先按 episode、anchor、action-prefix seed 配对相同 candidate bank，在每个 block 计算 treatment minus control；每个 episode 对六个配对 block delta 取 median，再汇总八个 episode delta。两臂各自的 episode median 另行报告，仅作描述，不能代替 matched delta。primary response 指标使用同一个 frozen response-loss 公式；elite regret 按 teacher candidate-cost population standard deviation 归一化，越低越好。report per-episode deltas 和最差 blocks；不把 300 candidates 或 48 blocks 当成独立样本，也不做 candidate-level 显著性检验。

## Frozen gates

相对 mechanism gate 必须同时满足：response normalized MSE 的 episode-median delta `<= -0.05`、standardized teacher-elite regret 的 episode-median delta `<= -0.05`、regret 严格改善至少 `5/8` episodes，以及 Top-30 recall 的 episode-median delta `>= 0`。

若任一 fresh block 的 teacher terminal contrast energy 低于 `1e-6`，归一化 response 指标依赖 denominator floor；此时 mechanism gate 记为 `INCONCLUSIVE`，不能 GO。报告 floor-active block 数和原始 teacher contrast energy，不删除或替换 block。

此外，treatment 仍须通过继承的 absolute predictor gate：median/minimum Spearman `>=0.95/0.80`，median/minimum Top-30 `>=0.75/0.50`，median relative latent MSE `<=0.25`，positive Spearman/Top-30 blocks 各至少 `36/48`，相对 teacher predictor latency reduction 至少 `20%`，并通过 integrity、convergence 和 causality 检查。early/middle/late 每个 stratum 还须分别达到 median Spearman `>=0.95`、median Top-30 `>=0.75`、positive Spearman/Top-30 blocks 各 `>=12/16`。相对改善不能替代 absolute 或 stratum gate。

只有 relative mechanism、treatment absolute、stratum、integrity、convergence、causality 和 latency 全部通过，才记为整体 GO。否则保留 NO-GO 或 inconclusive；不在该评估集上调参、改变阈值、补 episode 或重跑不同 seed。

## 预先冻结与运行边界

terminal `step_3000` checkpoint 保存后才允许执行 fresh evaluation inference 和计算 gate。episode ID selection 和 overlap assertions 必须先于 fresh evaluation inference 或评分；这不影响先完成训练和保存 checkpoint。不得根据验证指标选择 snapshot、调 `lambda_response`、改 candidate sampling 或增加训练轮数。

HDF5、teacher/model inference、training 与 evaluation 仅在带有效 `PBS_JOBID` 且经过 `PBS_NODEFILE` 检查的 PBS compute allocation 内运行；禁止在 login/head/submit node 执行。GPU utilization 与 memory 每 5 秒记录到该作业的 `job.log`。不查询或使用 banked reset credits。

本协议停在 predictor-level。`official CEM`、planner viability 与 closed-loop 均为 `NOT_RUN_BY_SCOPE`；本实验不产生它们的授权或结论。
