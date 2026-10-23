# C80 clip_only：沿用原 held-out

从当前 C8 `rho1.5 / update8` baseline 出发，仅训练614个 activation clipping ratios；模型 weights、VQ codebooks、row gains、indices、B32 exponents、Smooth0.5/H128 transforms 全部冻结。

用户最终指定不使用新的 development set。曾提交的 CPU 数据准备 `25853579.pbs101` 已按指示取消，其数据不进入本轮训练或评估。原 development 方案只保留为取消记录。

## 已完成结果

原40 held-out contexts ×2 seeds，以下数值均与同一个 baseline、同一个 teacher 比较。负百分比表示误差下降。

| 配置 | First10 motor RMSE | Future latent NMSE |
|---|---:|---:|
| 原 baseline | 0.044296774 | 0.060877371 |
| 旧 C8 clip_only /8 updates | 0.045434657 (+2.57%) | 0.068720982 (+12.88%) |
| C80 clip_only /80 updates | **0.043848437 (−1.01%)** | **0.054848124 (−9.90%)** |

本轮通过预设的两项 primary metric screen。先对每个 context 的两 seeds 求平均，40个 contexts 中，action改善23个，latent改善30个。C80 calibration 上，action RMSE下降5.18%、latent NMSE下降13.72%、mixed loss下降12.76%；原 held-out mixed loss下降6.36%。这支持“扩大 calibration 覆盖有帮助”，但 C8与C80的更新数也不同，不能只归因于数据量。

尚有实质性剩余问题：

- 原 held-out full32 motor RMSE为0.050111666，较 baseline **上升5.50%**；late22为0.051734803，**上升6.61%**；gripper为0.059316845，**上升3.97%**。
- 长 horizon 退化主要来自 `libero_object`：full32 +28.79%、late22 +34.69%；其 latent 也上升3.33%。其余三个 suites 的 latent 均改善。
- 训练 loss 只直接约束 first10 motor 和 future latent，没有直接监督 late22或gripper。当前证据同时提示未监督输出的保护不足和剩余泛化差距；不能把所有问题都归为训练集过拟合。
- 最终 ratios：mean0.95706、median0.98167、min0.72309；531/614低于1，93/614低于0.9，没有触及0.5下界。梯度范数最大0.00116，低于 norm clip1.0，因此该设置没有限制 Adam 的实际 ratio 步幅。

下一步建议保留 C80覆盖、冻结 weights，在同一训练预算下优先做一个带弱 late22 motor 约束的 clip_only 对照；32步 teacher action 已缓存，无需新增 teacher。更低学习率或回归 ratio=1 的 prior 是独立的后续对照。先保护长 horizon 输出，再考虑带局部重建约束的 channel scaling；此时需同时观察 activation error、weight-VQ error和最终 action/latent，而不是只优化 activation。

两个 GPU 作业实际累计 **1小时48分23秒**，均低于2小时：

| 阶段 | PBS job | 实际 allocation walltime | 验收 |
|---|---|---:|---|
| 缓存与 baseline | 25853617.pbs101 | 00:33:31 | F/exit0、marker、matching summary |
| clip_only80与完整评估 | 25854128.pbs101 | 01:14:52 | F/exit0、marker、matching summary |
| CPU汇总 | 25854912.pbs101 | 00:00:15 | F/exit0、marker、320 endpoint records |

GPU selfchecks、15秒 utilization/VRAM日志、80次更新和 weights 冻结检查均通过。监控 subagent 已停止，无后续 GPU 作业。完整数值见 `results/comparison.json`；终态证据见 `completion_evidence.json`。

## 固定比较条件

- C80：原 cases2000–2079，共40 tasks、80 contexts，各用原 seed_index=0，一遍80 updates。四 suites 交错更新。
- 训练：Adam，clip learning rate0.01，gradient norm clip1.0，首次更新后清空 Adam state，沿用先前设置。
- Loss：first10 ×6 motor MSE + `1.5 × 0.028303017887797623 × future latent global NMSE`，排除 clamped frame0。
- 只使用最终 update80，不根据 held-out 选择 checkpoint 或 early stopping。
- 正式评估：原 cases3000–3039 ×2原 sampler seeds，共80 paired queries；共享原 BF16 teacher。四 suites 等权。
- 原 held-out baseline 直接复用：first10 motor RMSE **0.04429677443113178**，future latent NMSE **0.060877370962407445**；重新汇总必须完全一致。
- 补齐 C80 baseline 与最终 clip_only 的完整80-query 端点评估，以观察 calibration 与 held-out 的收益差异。

原 held-out baseline 的主指标直接取自旧 query records；附带 action MSE 由每条已保存 RMSE 平方还原，因此 mixed loss 含原 float32 RMSE 的舍入误差。C80端点的 action MSE 则直接计算。正式判断使用原 RMSE 与 latent NMSE，两项均不受该还原影响。

## 作业

1. `prepare`：复用8份 C8 teacher、补齐72份 C80 teacher；复用已有7份 C8 baseline endpoint，另用首例 parity forward 提供 endpoint，并补齐其余 C80 baseline；复用原80-query held-out baseline。
2. `train_eval`：训练 ratios80次，逐更新 checkpoint；完整评估 C80 与原 held-out。
3. `aggregate`：CPU 汇总所有320条 endpoint records，输出 `results/comparison.json` 和 `RESULTS.zh.md`。

每个 GPU allocation 申请1小时59分，软截止6840秒，预留退出时间；仅确认 PBS terminal exit75 且 checkpointed 时续跑。每15秒记录 GPU utilization 与 VRAM。`gpt-6-luna / xhigh` subagent 每600秒监控，Q/R未发生实质变化时保持安静。

`results/train_eval/deployment.pt` 保存最终 ratios 及冻结 baseline checkpoint 的引用；不重复复制 weights。`checkpoint.pt` 另含 Adam、RNG 和已完成 update history，可精确续跑。

## 解释边界

C8旧 clip_only 的 calibration mixed loss 降27.41%，原 held-out action/latent 却上升2.57%/12.88%，提示泛化不足。本轮 C80 同时增加覆盖与更新数，因此结果不能独立证明“数据量”或“训练时长”是哪一项造成变化。原 held-out 已用于多次诊断；本轮是固定 cohort 的工程比较，不作无偏模型选择或统计优越性声明。

只有原 held-out 的 action 与 latent 两项均改善，才有理由继续讨论 channel scaling；否则先检查 clipping 更新幅度、loss 对齐与局部重建约束。所有结果均为 fixed-input decoded-BF16 VQ+A4 数值评估，无 closed-loop 或 native INT4 性能结论。

## 结果出来后的训练改进候选

以下只是待验证方向，本轮不同时修改：

| 优先级 | 改进 | 本场景的理由与限制 |
|---|---|---|
| 1 | 更广的 calibration 覆盖 | C80覆盖40 tasks、80 demos，优先于同一少数 context 重复训练；本轮即检验这条路线，但更新数也增加。 |
| 2 | 更保守的 ratio 更新 | 降低学习率或增加 `(ratio−1)^2` prior，限制偏离已验证 baseline。普通 AdamW decay 会把 ratio 推向0，不等同于回归1的 prior。 |
| 3 | 跨 context 梯度平均 | 减少单例梯度噪声；固定总 forward 数时，accumulation 会减少 optimizer steps，必须明确比较预算。 |
| 4 | 对齐保护项 | 现有 loss 不直接约束 late22 motor 或 gripper；可增加小权重保护项，但这会改变目标，不能当作纯数据改进。 |
| 5 | 局部重建后再全局微调 | 加入 block output/denoising 输出约束，减少多层相互补偿；须同时检查 activation error、weight-VQ error 和最终 action/latent。 |

文献依据与迁移边界：

- [AQLM 官方实现](https://github.com/Vahe1994/AQLM/blob/main/README.md)提供 calibration 数量、validation、block fine-tuning、early stopping 和保留最佳 checkpoint 的设置。它支持“数据与训练策略也值得优化”，不证明某个 LLM sample 数适用于 Fast-WAM。本轮按用户要求固定最终 checkpoint，未使用新 development set。
- [QuIP#](https://arxiv.org/abs/2402.04396)是 weight-only VQ 工作，其 fine-tuning 不能直接当成 A4 activation 优化的验证。
- [OmniQuant](https://arxiv.org/abs/2308.13137)联合优化 learnable weight clipping 与 equivalent transformation；[FlatQuant](https://arxiv.org/abs/2410.09426)用 block-wise calibration 优化 weights/activations 的变换。两者提供 W/A 联合考虑的参考，但不是本 microscaling VQ 的直接结果。
- [PV-Tuning](https://arxiv.org/abs/2405.14852)研究离散压缩参数的优化，提示 STE/连续参数更新可能不是唯一选择；实现成本高于当前614-ratio实验，暂后置。
