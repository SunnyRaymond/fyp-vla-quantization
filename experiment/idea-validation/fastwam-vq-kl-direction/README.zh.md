# KL training/evaluation parity 与真实方向诊断

目标：先解释为什么完整 C80 上，旧 MSE final80 的 latent-KL/双-KL目标比相应新 checkpoint 更低，再安排下一轮 action+latent 信息论 loss 实验。当前仅诊断，不保存训练后权重。

| Unit | 已有 checkpoint | 诊断目标 |
|---|---|---|
| mse_dual | 原 per-channel MSE+NMSE final80，25862598 | 双 KL |
| dual_dual | 双 KL final80，25878625 | 双 KL |
| latent_latent | Latent KL final80，25878623 | MSE+latent KL |
| action_action | Action KL final80，25878620 | Action KL+NMSE |

全部沿用旧 C80 case2000–2079、seed0、冻结 teacher、soft-bin 坐标、两个 loss coefficients、原 λ，以及614 ratios和1,806,863 channel log2D。没有新 development set、teacher forward、gripper训练项或held-out调参。

每组在同一参数下，对完整80cases逐一比较 checkpointed/grad-enabled 与 no-grad 输出的 action/future max_abs；严格要求0。训练 forward 的每个未缩放 objective 单独 backward，之后在FP32累积其参数 surrogate gradients 的1/80，得到完整 C80 平均方向，避免先缩放 loss 再通过BF16传播产生额外变化。另在预先固定的2000/2020/2040/2060四个原C80case计算action与latent分项梯度范数/夹角；这四个case的分项结论不外推为全C80统计。

对完整C80平均梯度比较两种Adam方向：已有checkpoint的saved moments，以及相同超参的fresh moments。每个方向取1、1/2、1/4步；从原参数重新开始投影到原bounds后，各评完整C80，报告真实loss变化、各项原指标、case配对变化及surrogate的一阶预测。该方向是final80位置上的反事实下一步，不是已丢失的第79步参数的重建，也不能直接解释旧日志里第80步的全部跳变。完整C80平均梯度用于移除抽样噪声，不声称与历史单case更新相同。

无参数训练或最终模型部署文件写入；中间保存的梯度和JSON仅供诊断/断点续作。模型权重不变；每个方向结束后恢复所有ratio/D并检查最终恢复。若checkpointed/no-grad parity失败，停止方向评估，先定位差异。

资源：每个GPU job上限01:59:00、软截止6840秒，每15秒将GPU利用率/显存写入所属job.log。只在真实PBS compute allocation做模型、数值和重I/O，login仅小控制文件/qstat/qsub。退出75并保存前缀状态的作业，只有核实scheduler和artifact确实以75终止后才能显式resume；不因排队或观测超时重提。gpt-6-luna/xhigh每600秒只读监控。

诊断结束后，以C80的parity、真实方向、动量与步长证据安排匹配对照的下一轮；不根据held-out结果选择新的超参。后续仍需原80个held-out paired queries、原seeds、共享teacher与固定checkpoint比较。当前信息论loss是人为定义的soft-bin数值分布KL，不等于已识别的真实policy density；不能预先保证其闭环收益。
