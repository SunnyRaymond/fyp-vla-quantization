# Selective teacher correction protocol

## 目的与边界

本实验测试一种 block-level selective correction：main student 先对完整 300-candidate bank 排序，sentinel student 只用于估计不确定性；只有不确定 block 才调用 full teacher300，并返回 teacher 的 exact top30。实验固定在 observation-level candidate ranking，不能推出 official CEM deployment、planner latency 或 closed-loop PushT 成功。

main 是 `25239551.pbs101` 的 CEM mixed treatment checkpoint；sentinel 是同一 job 的 original-bank continuation control checkpoint。两者均为 h256、同一 anchor-aligned step3000 起点和 1000-update budget 的历史产物；它们用于本轮 paired fresh evaluation，非本轮 concurrent retraining。

## Fresh slices 与共享 bank

沿用 selection seed `20300903`，按同一 valid shuffle 排除旧 prefix：calibration 使用 `valid[576:584]`，test 使用 `valid[584:592]`。每个集合包含 8 episodes、3 anchors（early/middle/late）、2 action-prefix seeds `20301101/20301102`，共 24 contexts、48 CEM trajectories、144 个 round blocks（round 10/20/30）。episode 是 replicate，先聚合 episode 内 18 个 block，再跨 8 episodes 汇总。

两集合的 bank 都由 frozen anchor-aligned step3000 student 做 30-iteration CEM 产生：每轮 300 candidates、top30、horizon5、candidate 0 为 pre-update mean，elite std 使用 unbiased std；teacher 只做 shadow labeling，不参与 CEM proposal 或更新。每个集合内部 main、sentinel、selective、random 使用同一 bank 与 teacher costs。

## 不确定性与 threshold

对同一 block 计算 main 与 sentinel 的 candidate ranking，统一使用 pinned LeWM solver 的 `torch.topk(cost, k=30, largest=False, sorted=True)` 语义。定义单一 scalar：

`u = 1 - |top30_main ∩ top30_sentinel| / 30`。

两组完整 ranking 的 disagreement 只作为 secondary diagnostic 报告，不与 `u` 组合、不参与 threshold 或 sweep。

calibration 只根据 `u` 的离散经验分布确定一个 threshold。候选集合是 calibration 中相邻 distinct `u` levels 的 midpoints，采用严格 `u > tau` 触发 teacher call；保留 `0 < calibration call rate <=0.25` 的候选，并选择 call rate 最大者，并列时选更大的 `tau`。若没有可行候选，作业输出 `INCONCLUSIVE` 并停止，不静默改变 tie 规则。midpoint 加严格不等号会让相同 `u` 的 block 一起处理，不按 block ID 或 test 集内部排名打破 ties。calibration teacher labels 仅用于质量诊断，不参与 threshold 选择。

test 不重新估计 threshold。confident block 返回 main student top30；uncertain block 对同一 300-candidate bank 运行 full teacher 并返回 teacher top30。sentinel 从不直接决定返回的 candidate。

random baseline 使用 analytic same-budget expectation，不引入 Monte Carlo seed：对每个 episode，若 selective 实际调用 `m` 个 teacher，则 random fallback 的期望 episode mean regret 为 `(1-m/18) * main_episode_mean_regret`。它只用于 same-budget paired comparison，不依赖一次随机抽样。

## Metrics 与 gates

每个 returned top30 的 standardized teacher regret 为：

`(teacher mean cost(returned top30) - teacher mean cost(teacher top30)) / max(population std(teacher costs), 1e-6)`。

primary 分别计算 main、selective、analytic-random 的 episode mean（每 episode 18 blocks），再报告 8 episode 的 median、p90，以及 block-level median/p90 和 round-level结果。风险指标把 main standardized regret `>=1.5` 定义为 catastrophic，报告 test catastrophic block 数与被 teacher call 捕获的比例；若 test catastrophic block 少于 4，risk gate 为 inconclusive/FAIL，不借助 calibration 或 test 调阈值。

冻结 gate 同时要求：test teacher-call rate `<=0.35`；selective−main 的 episode median delta `<=−0.10`；至少 5/8 episodes 严格改善；catastrophic risk capture `>=0.75` 且 test catastrophic blocks `>=4`；selective−analytic-random 的 episode median delta `<=0`；以及 selective 相对 teacher-only full300 的 native mean latency reduction `>=0.30`。所有 finite、pairing、shared bank、checkpoint provenance 检查必须通过。任一关键 gate 失败即 STOP，不进入 adaptive CEM 或 closed-loop。

## Timing

quality 与 timing 都覆盖 144 个 test trajectory blocks。timing 进行 3 warmups、10 repeats，并在固定交错顺序下测量 `teacher_only`、`main_student_only`、`two_student_uncertainty` 和 `selective`。每次测量用 `perf_counter` 包围 CUDA synchronize，包含 candidate H2D、两 student scoring、uncertainty、full teacher300 与 top30 selection；一次性的 model load、dataset read 和 shared row construction 不计入，latent/state/goal 的 GPU cache 准备在各 arm 保持一致。报告各 arm mean/p95、teacher-only mean/p95 与 selective latency reduction。

## Provenance 与运行约束

main/sentinel checkpoint、anchor driver、stable-world-model、dataset 和 fresh rows 只在 PBS compute allocation 读取；login node 只做轻量提交、状态查看和小文件控制。PBS wrapper 要求非空 `PBS_JOBID`、非 login hostname，使用实际 `nvidia-smi` PID 每 5 秒写 `job.log`，并在退出时清理。checkpoint、fresh rows、trajectory banks 留在 cluster，仅回传 summary、status、log 和 GPU telemetry 小文件。

实验设计组织参考：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. https://doi.org/10.48550/arXiv.2609.00065 。本会话已核对当前 arXiv record；正式引用使用 latest DOI，不附 version suffix。该背景引用不构成 LeWM/PushT 实测证据，也不改变本实验的冻结 gate。
