# Wall fast screening 执行记录

2026-09-09：用户授权下一轮，采用 FAST_SCREEN.zh.md 的 CEM5 协议。此文件更新该设计文档中“仅设计、无提交”的历史状态。1 worker/A100，每 job 1 GPU，最多并发 4 个本实验 jobs；不重新运行 IdeaSpark pipeline，不训练模型。

## 在 development 结果产生前冻结的资源门槛

首先完成同一组 8 个 development episodes 的 FP32、all-W4、all-W8。下列阈值仅用于快速研究资源决策，不是统计显著性或普适有效性判断。

1. 任一配置出现数值非有限、目标/RNG不一致、结果缺失：先解决实现问题，不能将其计为有效研究结论。
2. FP32 至少成功 5/8，且 all-W8 比 FP32 少成功不超过 1 例，才能自动进入昂贵 held-out 评估。否则先报告 reference/高精度 sanity 不足，不将其解释为 RankCal 无效。
3. 量化差距采用 paired 计数：FP32成功/W4失败的例数，减去W4成功/FP32失败的例数，至少为 2，才自动投入 24-episode held-out。若差距更小，只允许完成有界的离线机制测量，随后报告当前 screening 缺少可辨别收益的空间，不自动扩充 test。
4. 三种 signal allocations 只由 calibration 8 episodes 拟合；development probes 仅检查稳定性。test 固定 24 个新 episodes，不按结果追加、换 seed、修改配额或 planner。

早停是按用户的快速 go/no-go 偏好节省资源；未通过资源门槛不等于证明所有 RankCal 实现无效。若发现门槛本身与实际环境/数据的含义不符，保留原门槛与原因，不静默改写。

## 实现与证据

- 已通过的 smoke：16176462.pbs101，详见 SMOKE_RESULTS.zh.md。
- screen_runner.py：完整 episode、固定 split、逐例动作/状态/结果与 FP32 pools。
- probe_runner.py：site-only W4/W8 probes，RankCal、LocalMSE、ScoreError 及两个配额匹配 random allocations。
- screen.pbs / run_stage.py：每 job 保存实际运行代码、参数、输出、退出状态；有 walltime 与内部 timeout。
- PBS 实际分配时间累计为 GPU-hours；不把排队时间算 GPU-hours。

当前状态：全部阶段完成并通过核验，详细结论见 SCREEN_RESULTS.zh.md，实际分配时间见 jobs.json。Development FP32 / W4 / W8 分别成功 7/8、5/8、6/8，满足上述预先冻结的资源门槛。Calibration 24 pools 与 development 26 pools 的 site probes 均通过独立数值重算。五种 distinct allocations 和 24 个 held-out targets 已在提交前冻结于 TEST_FREEZE.json。Held-out 7 方法共 168 条完整轨迹、joint fidelity 与 freeze audit 均通过；最终资源决策为 no-go：证据不足以扩大当前方案。

Targets 的 dataset index 分配在产生结果前固定：smoke 保留 0–1，calibration 2–9，development 10–17，pilot_test 18–41。每 split 使用既定 env seed namespace，CEM seed = namespace + 10000 + episode index。各模型读取共享 target manifest；观测、状态与 layout 纳入一致性校验。

报告 paired wins/losses，并附 conservative 95% paired-difference CI：分别对 P(win) 与 P(loss) 建立 97.5% Clopper–Pearson intervals，经 Bonferroni 合成差值区间。另列 exact two-sided McNemar p 值，仅作 exploratory 个别比较，不声称多重比较后的显著性。

调度优化（在 calibration allocations 产生前确定）：若方法得到逐 site 完全相同的 bit mapping，在同一 checkpoint、quantizer、targets 与 seeds 下只运行一份闭环评估，保留显式 alias 映射并共享结果。这些不是独立重复实验；不将相同结果重复计入样本量。不同 mappings 仍分别运行，不按性能结果去重。
