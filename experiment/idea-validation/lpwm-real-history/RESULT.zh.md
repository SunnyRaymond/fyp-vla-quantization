# LpWM real-history mechanism result

## 判定

本次固定候选机制筛查为 **NO_GO**：16 个配对 task-context（8 tasks × 2 anchors）、8 个 non-floor tasks 均达到预先冻结的最低样本数；history 相对 cold 的 task-level median elite-regret reduction 为 **5.635%**，低于 10% 门槛，且只有 **4/8** tasks 改善，少于所需 5/8。按冻结规则停止，不运行 closed-loop。

这里的 NO_GO 仅表示当前 checkpoint、PushT cohort 与固定协议下，real-history 未通过推进到 paired closed-loop 的 mechanism gate；它不是关于所有 real-history 方法的普遍结论，也不是任务成功率或部署速度结论。

## Primary 指标

每个 task 先平均可用 anchors，再以 `(R_cold - R_history) / max(R_cold, 0.01)` 计算 relative elite-regret reduction。正值代表 history regret 较低，负值代表较高。

| 原 task index | cold regret | history regret | reduction |
|---:|---:|---:|---:|
| 0 | 2.5520 | 1.6921 | +33.70% |
| 2 | 0.6222 | 1.6887 | −171.40% |
| 3 | 2.2710 | 1.7869 | +21.32% |
| 4 | 0.0686 | 0.0548 | +20.07% |
| 5 | 2.0994 | 2.2616 | −7.73% |
| 8 | 1.2308 | 1.2546 | −1.94% |
| 9 | 0.3978 | 0.3992 | −0.35% |
| 12 | 2.3507 | 2.0775 | +11.62% |

任务间差异明显：任务 0、3、4、12 改善，任务 2 的 history regret 明显更高；其余任务略差。候选样本不是独立统计单位，也未进行显著性检验。该 primary 是从同一 300-candidate bank 中按各 predictor 选出的 top-30 对照 bank 的 true top-30 oracle regret；不是 success predicate。

## Secondary 诊断

以下均为事后描述，先在 task 内平均两个 context/anchors，再汇总 8 个 task；它们不改变 primary gate。

- 与真实未来 encoder latent 的 1–5 步 MSE：task-level median 从 cold **0.007542** 到 history **0.007917**；仅 **2/8 tasks（4/16 contexts）** 的 history MSE 更低。
- action-response 方向 cosine 的 task-level median 从 **0.7222** 到 **0.7083**；仅 **2/8 tasks** 改善。
- 每 context candidate-level magnitude-ratio median 先在 task 内平均，再对 8 tasks 取 median：cold **1.5563**，history **1.6156**。比值高于 1 表示预测 latent terminal response 的幅度通常大于该固定 bank 相对 index-0 reference 的真实未来 latent response。所有 context 的真实非零响应候选均未出现 predicted-zero response。

这些 latent response 诊断不能直接解释为物理空间的过推或根因，也不构成因果证明。整体上它们没有显示 history 带来稳定改善。

## 协议与执行证据

- 固定来源为 `YilunKuang/lpworldmodel@bdd812d9432cccda8c350086006401b436f91982`；使用已完成 sparse checkpoint（2 epochs），不重训。cohort 为原 29 个 formal failures 中升序前 8 个 task `[0,2,3,4,5,8,9,12]`，planning seed 99，H=5。
- 每个 context 使用相同的 300 个 native CEM action candidates：iteration 0 的 100 个固定 early proposals、最后一次已评估 population 的 170 个 non-elites，以及全部 30 个 native elites；保留并记录重复项。real-history arm 加入真实 `t−10,t−5,t` 视觉帧及两段对齐的已执行 action blocks，future 仍为原 5 blocks。
- 新加的 task-0 `obs_0` 重复图像 GPU-entry probe、native action shape gates 与 16-context history1 parity 均通过。该 probe 只验证 `transform_obs_visual` 和 encoder 接口，不是 future-prediction 质量证据。
- 成本门通过：native capture（包含 `eval_every=1` 的环境评估）耗时 360.95 s；anchor 1 的 300 候选 full-prefix physics replay 为 44.31 s；首 context cold/history forwards 与真实 future encoder 合计 1.76 s。冻结公式预计完整 mechanism screen 为 **2095.84 s**，低于 4800 s 内部上限。truth 从 task init 与完整已执行 prefix 重放，而非从不完整 7D anchor reset。
- GPU job `25578290.pbs101`：PBS `F / Exit_status=0 / Stageout_status=1`，wrapper/runner 均 exit 0，walltime **00:24:27**，运行节点 `x1000c0s1b0n1`。PBS 源请求 1 GPU/8 CPU/64 GB/90 min，调度器接受 1 GPU/16 CPU/110 GB/90 min；资源形状差异原因未核实。Stageout 状态保留为 1，job log 与结果文件可读取。
- 前两次 GPU attempt 均为技术失败，且没有 quality result：`25577811.pbs101` 因错误假设 action dim=15 而未进 cost gate；`25578046.pbs101` 因 generic `transform_obs` 对 visual-only 输入索取缺失的 `proprio` 而在成本校准中失败。它们不计作机制 NO_GO。第三次唯一获批修复使用 native visual-only `transform_obs_visual` 分支后完成筛查。

本实验没有训练或改动模型，没有跑完整 50-task paired closed-loop，没有测得端到端加速；按 primary NO_GO，closed-loop 阶段不启动。

## 结果文件

- 远端原始汇总：`/scratch/users/ntu/yguo017/lpwm-real-history/artifacts/25578290.pbs101/results/run_summary.json`
- 成本门：`/scratch/users/ntu/yguo017/lpwm-real-history/artifacts/25578290.pbs101/results/cost_calibration.json`
- 预检与接口门：`/scratch/users/ntu/yguo017/lpwm-real-history/artifacts/25578290.pbs101/results/runner_preflight.json`
- PBS/GPU telemetry 与终态日志：`/scratch/users/ntu/yguo017/lpwm-real-history/artifacts/25578290.pbs101/job.log`
- 冻结协议与尝试记录：[FREEZE.json](FREEZE.json)、[PROTOCOL.zh.md](PROTOCOL.zh.md)、[attempts.md](SUBMISSION/attempts.md)
