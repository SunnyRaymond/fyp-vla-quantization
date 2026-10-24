Q-Clip 280-episode LIBERO-Plus 闭环结果

最终结果：Q-Clip **207/280（73.93%）**，Q0 A+Z+KL **208/280（74.29%）**。Q-Clip 相对直接父 checkpoint 净少 1 个成功（−0.36 个百分点），配对为 **11 胜、12 负、257 持平**。这轮没有体现总体闭环成功率收益；固定输入误差的改善尚未转化为本 cohort 的成功率提升。

作业 `25939858.pbs101` / `r128qcl0001` 使用单一四卡 allocation，申请 8h，实际运行 **05:02:17**；PBS 为 F / exit 0，artifact exit 0，`PIPELINE_COMPLETE` 与完成 gate 均通过。四个 worker 各 70/70、rc=0，总共 280/280 个新 episode，并复用 17 个参考 arm 的 4,760 个已完成 reference slots。

所有 arm 沿用同一份冻结 manifest：4 suites × 7 扰动维度 × 10 variants，保持相同 initial state、environment/sampler seeds、settle steps、action execution horizon 和 episode cap。胜/负指同一 slot 中 Q-Clip 成功而 comparator 失败／Q-Clip 失败而 comparator 成功；持平包含共同成功和共同失败。

总体比较（所有分母均为 280）：

| 配置 | 成功数 | 成功率 |
| --- | --- | --- |
| BF16 reference | 216/280 | 77.14% |
| Microscaling VQ A+Z | 204/280 | 72.86% |
| Microscaling VQ A+Z+KL (Q0) | 208/280 | 74.29% |
| A+Z+KL + Q-Clip | 207/280 | 73.93% |
| SpinQuant adapted W4A4 | 203/280 | 72.50% |
| C80 R128 W16A4 | 203/280 | 72.50% |
| 旧 C80 clip-only | 207/280 | 73.93% |

旧 C80 clip-only 与本轮 Q-Clip 是不同 checkpoint；两者成功总数相同不代表 episode outcome 相同。

| Q-Clip 相对 | 胜 | 负 | 平 | 净成功数 | 成功率差（百分点） |
| --- | --- | --- | --- | --- | --- |
| Microscaling VQ A+Z+KL (Q0) | 11 | 12 | 257 | -1 | -0.36 |
| Microscaling VQ A+Z | 13 | 10 | 257 | +3 | +1.07 |
| BF16 reference | 8 | 17 | 255 | -9 | -3.21 |
| SpinQuant adapted W4A4 | 15 | 11 | 254 | +4 | +1.43 |
| 旧 C80 clip-only | 11 | 11 | 258 | +0 | +0.00 |

按 suite 比较（每行 70 episodes）：

| Suite | BF16 | Q0 A+Z+KL | Q-Clip | Q-Clip − Q0 |
| --- | --- | --- | --- | --- |
| libero_10 | 54 | 51 | 49 | -2 |
| libero_goal | 51 | 47 | 48 | +1 |
| libero_object | 55 | 55 | 55 | +0 |
| libero_spatial | 56 | 55 | 55 | +0 |

Goal 净增加 1 个，libero_10 净减少 2 个；Object、Spatial 的总数相同，但 Object 内仍有 4 胜、4 负，Spatial 有 1 胜、1 负。

按扰动维度比较（每行 40 episodes）：

| 扰动 | BF16 | Q0 A+Z+KL | Q-Clip | 净变化 | 胜/负/平 |
| --- | --- | --- | --- | --- | --- |
| background_textures | 26 | 26 | 25 | -1 | 2/3/35 |
| camera_viewpoints | 21 | 19 | 19 | +0 | 2/2/36 |
| language_instructions | 38 | 39 | 38 | -1 | 0/1/39 |
| light_conditions | 35 | 35 | 36 | +1 | 3/2/35 |
| objects_layout | 35 | 34 | 34 | +0 | 1/1/38 |
| robot_initial_states | 34 | 31 | 30 | -1 | 1/2/37 |
| sensor_noise | 27 | 24 | 25 | +1 | 2/1/37 |

Light conditions、sensor noise 各净增加 1；background textures、language instructions、robot initial states 各净减少 1；camera viewpoints 和 objects layout 总数持平。

变化最大的 suite × 扰动格子：Object/light conditions 8→10（2 胜、0 负），libero_10/sensor noise 6→8（2 胜、0 负），libero_10/robot initial states 10→8（0 胜、2 负）。每格只有 10 个 variants；这些是本 cohort 的描述性变化，不能据此概括普遍的扰动鲁棒性。

28 个 suite × 扰动格子（成功数；Δ 为 Q-Clip − Q0，每格分母 10）：

| Suite / 扰动 | Q0 | Q-Clip | Δ | 胜/负 |
| --- | --- | --- | --- | --- |
| libero_10::background_textures | 3 | 4 | +1 | 1/0 |
| libero_10::camera_viewpoints | 5 | 4 | -1 | 0/1 |
| libero_10::language_instructions | 10 | 10 | +0 | 0/0 |
| libero_10::light_conditions | 8 | 7 | -1 | 0/1 |
| libero_10::objects_layout | 9 | 8 | -1 | 0/1 |
| libero_10::robot_initial_states | 10 | 8 | -2 | 0/2 |
| libero_10::sensor_noise | 6 | 8 | +2 | 2/0 |
| libero_goal::background_textures | 7 | 7 | +0 | 1/1 |
| libero_goal::camera_viewpoints | 4 | 4 | +0 | 0/0 |
| libero_goal::language_instructions | 9 | 9 | +0 | 0/0 |
| libero_goal::light_conditions | 9 | 9 | +0 | 1/1 |
| libero_goal::objects_layout | 6 | 6 | +0 | 0/0 |
| libero_goal::robot_initial_states | 7 | 8 | +1 | 1/0 |
| libero_goal::sensor_noise | 5 | 5 | +0 | 0/0 |
| libero_object::background_textures | 8 | 7 | -1 | 0/1 |
| libero_object::camera_viewpoints | 7 | 7 | +0 | 1/1 |
| libero_object::language_instructions | 10 | 9 | -1 | 0/1 |
| libero_object::light_conditions | 8 | 10 | +2 | 2/0 |
| libero_object::objects_layout | 9 | 10 | +1 | 1/0 |
| libero_object::robot_initial_states | 6 | 6 | +0 | 0/0 |
| libero_object::sensor_noise | 7 | 6 | -1 | 0/1 |
| libero_spatial::background_textures | 8 | 7 | -1 | 0/1 |
| libero_spatial::camera_viewpoints | 3 | 4 | +1 | 1/0 |
| libero_spatial::language_instructions | 10 | 10 | +0 | 0/0 |
| libero_spatial::light_conditions | 10 | 10 | +0 | 0/0 |
| libero_spatial::objects_layout | 10 | 10 | +0 | 0/0 |
| libero_spatial::robot_initial_states | 8 | 8 | +0 | 0/0 |
| libero_spatial::sensor_noise | 6 | 6 | +0 | 0/0 |

安装与启动验证：复用 source job `25937633.pbs101` 的 final80 Q-Clip ratios，直接父 checkpoint 为 `25921795.pbs101` 的 final8 reader-KL。只安装 614 个已学习的 clipping ratios，weights、VQ、R 与 ratios 全部冻结，新训练更新数为 0。四 worker 的 Q0 完整 action/future anchor 差异均为 0；Q-Clip 八项 source metrics 重放误差全部为 0；formal inference action parity 差异均为 0。每 worker 执行 4 次 startup query；本轮没有补跑 held-out RMSE，也没有新增 baseline episode。

性能记录仅描述本实现的实际运行范围：共 6,892 次正式 action queries，其中后续 6,888 次 full action query wall time 均值 **6.969 s**、中位数 **6.927 s**、P95 **7.425 s**；对应 denoising GPU interval 均值 **6.898 s**。PyTorch peak allocated **28.478 GB**，peak reserved **28.750 GB**（十进制 GB），不能与 nvidia-smi 的 MiB 直接混用。GPU utilization / 显存以 15 秒间隔记录在远端 job.log；本地仅保存结束前的 8 KiB tail，尾部模型释放后的空闲采样不代表推理显存。

本实验使用 decoded BF16 dense MXVQ4 weights、dynamic A4 activations、BF16 KV cache；上述 timing 不构成 native low-bit kernel 加速证据。本报告只读取 compute allocation 已完成的配对汇总，不在 login node 或本机重新加载模型／聚合 episode payload，也未进行新的显著性检验或多 seed 复测。

本轮按预定 final80 checkpoint 完成，无额外 sweep。Q0 A+Z+KL 仍作为当前参考配置保留，监控已结束。原始分组结果见 `summary.json`，完成证据见 `FINAL_COMPLETION_EVIDENCE.json`，复核快照见 `pulled/evaluation_25939858.pbs101.json`。
