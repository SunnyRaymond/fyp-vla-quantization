# Activation clipping checkpoint 组合与 C8 端点评估

## 已完成与判断

两GPU job为 `25851758.pbs101`（35m33s）、`25851759.pbs101`（35m19s），CPU聚合为 `25852915.pbs101`（8s）；PBS/artifact均exit0，完成marker与完整summary匹配。新评估192 queries齐全，实际sampler执行193次（多出一次未复用的startup parity）；无训练、无新增teacher。初始参考action parity均为0，GPU selfcheck与job.log中的GPU utilization/显存采样已取回。600秒subagent监控已停止。

完整数值见 [RESULTS.zh.md](results/RESULTS.zh.md)，原精度结果见 [comparison.json](results/comparison.json)，范围核对见 [completion_evidence.json](completion_evidence.json)。

| 新组合 | Held-out first10 motor RMSE相对baseline | Future latent NMSE相对baseline |
|---|---:|---:|
| joint weights + ratio=1 | +2.762% | +4.144% |
| baseline weights + joint ratios | +1.452% | -3.758% |

保留joint ratios时，joint相对baseline仍是action +0.517%、latent +0.211%。因此joint weights没有在去掉clipping后变成可直接采用的改进；joint ratios可以独立改善latent，却损害action。当前结果体现weight与activation的配套适配及action/latent权衡，没有同时优于baseline的新组合。

| C8最终checkpoint相对baseline | First10 motor RMSE | Future latent NMSE | 原mixed loss |
|---|---:|---:|---:|
| weight_only | -4.096% | -28.393% | -21.490% |
| clip_only | -8.648% | -33.702% | -27.409% |
| joint | -5.307% | -33.707% | -24.895% |

四组完整C8端点明确显示：继续优化降低了原训练目标和两个主要训练集误差，但这些收益没有在原held-out均值上泛化。该现象与对小C8及固定sampler seed的过拟合一致；它降低了“训练目标完全没有下降”这一解释的可信度，但不能据单次fitting确定所有退化机制。clip_only的C8 mixed loss改善最大，held-out latent却退化最多，尤其说明不能按calibration mixed loss选择activation参数。

Channel scaling的下一步建议：保留原baseline作为参考；另立受限验证，D从identity、clipping从1开始，固定H128顺序，配套weight等价变换与VQ局部重拟合；保留VQ-refit-only对照。先以局部output重建、identity约束与独立development contexts筛选，再考虑短程action+latent联合更新。同步报告activation误差、weight-VQ误差与最终output误差；推进须同时改善独立development的action与latent，不能仅凭C8 loss。当前40-context held-out已用于分析，不继续作为反复调参的选择集。本轮未提交channel scaling训练。

## 冻结评估方案

本轮只评估现有最终 checkpoint，不重新训练，不重新生成 teacher。原四组结果保留在 `../fastwam-vq-activation-clipping/`。

| 单 GPU worker | 新 held-out 组合 | C8 最终 checkpoint |
|---|---|---|
| joint_w_ratio1 | joint weights + ratio 全部精确为1 | baseline、weight_only |
| baseline_w_joint_ratios | baseline weights + joint ratios | clip_only、joint |

每个新组合保留原40 contexts × 2 sampler seeds = 80 paired queries、原输入 plan rows 和共享 BF16 teacher。C8四组各完整8 contexts，使用原 seed_index=0 和 C8 teacher cache。共192个评估 queries。每个allocation另做一次baseline参考action parity；负责baseline C8的worker会复用该次输出，实际query执行数单独记录。

C8同时报告 first10 motor MSE、RMSE、future latent NMSE 和原始 mixed loss（rho1.5、冻结normalization_c），并保留 full32/late22 motor 与 gripper 指标。Held-out按 seeds→case→suite→四suite等权聚合；C8每suite两例、四suite等权。不会把tokens、frames或sampler seeds当作独立训练重复。

权重source：baseline为C8 rho1.5 update8（GPU `25821658.pbs101`，CPU补验证 `25822474.pbs101`）；后三组来自clipping实验固定额外update8。新组合只交换既有FP32 masters与ratios，部署仍沿用BF16 books/gains与A4。H128/Smooth0.5、VQ indices、B32 exponents、KV和所有非目标参数固定。评估时所有模型、books、row gains、ratios均冻结，代码不创建optimizer或调用backward。

两个GPU job分别申请1GPU、1h59m，6840秒软截止，逐query原子保存；只允许在PBS确认exit75且progress为checkpointed时续跑未完成部分，完整分母保持不变。每15秒将GPU utilization与显存写入本job的job.log。最终聚合在CPU-only PBS allocation完成。

监控由显式指定 `gpt-6-luna`、`xhigh`、`fork_turns=none` 的subagent执行，每600秒只读检查当前handles。该配置是spawn工具接受的指定值；subagent无法独立读取serving variant。Q/R未变保持安静，查询失败记为unknown，不因queue comment自行重交。完成需PBS终态/exit0、artifact exit0、PHASE_COMPLETE和完整匹配summary同时满足。

ASPIRE2A login node仅小文件控制、连接、提交和状态查询；模型、推理和聚合均在实际PBS compute allocation。保留SSH host-key verification；不读取输出credentials，不使用banked reset。

两组组合用于区分独立clipping效用、weight更新效用与共同适配；C8端点用于区分训练目标未改善和训练/held-out分歧。结果仅是fixed-input decoded-BF16 VQ+A4诊断，不构成统计显著性、closed-loop成功率或native INT4速度证据。Channel scaling只在结果完成后判断，不在本轮自动训练。
