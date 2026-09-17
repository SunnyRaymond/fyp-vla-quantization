# RankCal / DINO-WM Wall smoke 结果

日期：2026-09-09。PBS job：16176462.pbs101。1 × A100-SXM4-40GB，退出码 0，分配实际用时 160 秒，即 0.0444 GPU-hours。suite 148.39 秒；数值与闭环检查 61.68 秒。没有运行正式 screening 或重新运行 IdeaSpark pipeline。

## 已有证据

- 实际加载 Wall epoch 65 checkpoint，FP32，decoder=None。
- 两例显式 layout、seed、initial/goal 状态；非零动作的 25-step replay 一致。
- FP32/NOOP 分数完全相同，量化后恢复 FP32 权重的分数完全相同。
- 原始 CEM 与 adapter 在共同关闭 inner environment evaluator、同 RNG 的条件下，CEM5 输出动作完全一致。这不是对原始 early-stop 行为的等价证明。
- 12 个 encoder block 与 6 个 predictor block 已核实，各 family 内量化成本相同。
- 两例各完成一次 CEM5 + MPC 执行，分别 6.33 / 6.31 秒；每例执行 25 个环境 step。单轮未成功不代表完整 episode 失败。
- W4/W8 分数均有限。固定 300 candidates、30 elites 的集合与 FP32 重合率：W4 为 63.3% / 36.7%，W8 均 96.7%。这证明这两个 probe 存在量化排序扰动，不证明 RankCal 有效。

## 低显存与吞吐

同样完成 24 次 candidate-pool scoring，已 warmup，独立进程通过 barrier 开始；两种 worker 数的分数均与参考一致，elite 集合一致。

| 同 GPU worker 数 | 测量用时 | pools/s | 每 worker peak allocated | 每 worker peak reserved | scoring 窗口 GPU utilization 均值 | device 显存峰值 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 29.87 s | 0.804 | 3.87 GiB | 5.48 GiB | 98.1% | 6131 MiB |
| 2 | 33.65 s | 0.713 | 3.87 GiB | 5.48 GiB | 100% | 12257 MiB |

2-worker 吞吐比为 0.888，此工作负载建议 1 worker/GPU。这里只各测一次，是调度决策的初步依据。utilization 来自已分配 GPU UUID 的 1 秒采样，表示 GPU 忙碌时间，不能解释为达到理论 FLOPS 上限。

显存低与计算忙碌可以同时存在：无梯度、decoder=None，候选按单 episode 的 300 条计算，没有同时保留许多 episode 的计算图。闭环 peak allocated 4.15 GiB、reserved 6.17 GiB；量化使用 FP32 中模拟的权重量化，不能声称 native W4/W8 显存或加速收益。

## 初步结论与下一阶段

工程 smoke 通过，可以继续制作和验证 fast screening runner。RankCal 是否有效仍未知：site-only 排序信号能否指导 allocation、联合量化后能否保留优势，以及新 episodes 的闭环收益，都未被本次实验验证。

保留 CEM5。后续 FP32、W4 和 allocation variants 使用同样 planner；若 FP32 在 development episodes 本身很差，先诊断 planner 预算，不将其归因于 RankCal。正式 screening 的冻结 episode/allocation manifests、校准信号与 allocation 实现仍待完成。

以本次单轮约 6.32 秒作粗略外推，24 test episodes × 7 variants × 最多 12 MPC rounds 约为 3.54 GPU-hours 的闭环计算；这不是总预算，尚不包括 calibration、development、加载、失败重试和分析，也未实测 W4 variants 全程。因此此前 5–10 GPU-hours 的 screening 预算暂保留。并行用不同 GPU 上的独立 job；不能从本次结果假定单 GPU 双 worker 会提速。日历时间仍依赖排队和实现完成时间。

原始证据：`artifacts/16176462.pbs101/`，独立核验汇总：`verification.json`，PBS 记录：`qstat_final.txt`，设备采样：`device_samples.csv`。
