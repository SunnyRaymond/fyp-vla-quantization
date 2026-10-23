# Stage2 固定 rho 完整训练长度比较

2026-10-21。两条训练轨迹均已完成。rho=2.25 在第1、8、16步完整 outer 上的 motor RMSE 均低于 rho=1.0，总体候选排序及 log rho 割线方向没有翻转；但最终两个候选均劣于固定 rho=1.5，未通过既定门槛。保留 rho=1.5，不进入阶段3。

两支均精确恢复相同的初始 master parameters、Adam 和 RNG，按冻结 C8 顺序训练两轮，共16步。每个 milestone 评估完整 outer8×seed0，评估后精确恢复训练状态。c=0.028303017887797623、loss、学习率、Adam reset 规则和 deterministic_v2 设置保持冻结。每步614个 targets 的梯度连接与有限性均通过；第1步与原 preflight 的重放差异均为0。

| Updates | rho1.0 motor RMSE | rho2.25 motor RMSE | log rho 割线 | RMSE较低的rho | 两端相对差距 |
|---|---:|---:|---:|---:|---:|
| 1 | 0.049855095 | 0.045758589 | −0.005051614 | 2.25 | 8.569% |
| 8 | 0.054402293 | 0.045896162 | −0.010489350 | 2.25 | 16.962% |
| 16 | 0.044524720 | 0.044443510 | −0.000100144 | 2.25 | 0.183% |

割线定义为 [J_H(2.25)−J_H(1.0)]/[log(2.25)−log(1.0)]；相对差距使用两个端点的平均值作分母。三次差距均超过预设0.1% deadband，但最终差距已明显缩小。这里只测量这两个候选之间的割线，不能将其当成 rho=1.5 处的局部导数。H1只经历第一个 C8 context，H8和H16分别经历完整一轮和两轮。

固定 rho=1.5 的缓存仅用于 final16 比较，其 motor RMSE 为0.041933100。没有推断或补造该对照在H1/H8的结果。

| 候选 rho | first10 motor | full32 motor | late22 motor | first10 gripper | future latent NMSE | 合格 |
|---|---:|---:|---:|---:|---:|---|
| 1.0 | +6.18% | +5.15% | +5.33% | +5.76% | +51.07% | 否 |
| 2.25 | +5.99% | +8.25% | +8.49% | +13.20% | +37.82% | 否 |

所有百分比均相对固定 rho=1.5。冻结门槛为 first10 motor 至少改善1%，且其余四项均不得退化超过5%；两个候选的主指标和各项次指标均不满足。所有 final16 模型训练预算相同，按 case 聚合后四 suite 等权。

完整 outer 没有出现方向反转，但分 batch 后有不同偏好：batch A 在H1偏好2.25，到H8/H16偏好1.0；batch B在三个 horizon 都偏好2.25。final16 的 batch A 分别为0.042218174和0.042328597，batch B分别为0.046831266和0.046558423。因此，全体平均方向稳定并不代表每个 outer 子集都给出相同方向，也不能用它消除原搜索中交替 batch 与 inner context 固定关联的问题。

| outer case | suite 与 phase | rho1.0 motor 变化 | rho2.25 motor 变化 |
|---|---|---:|---:|
| 4000 | spatial early | +16.44% | −0.59% |
| 4001 | spatial middle | +12.56% | +10.09% |
| 4002 | object middle | −4.98% | +5.85% |
| 4003 | object late | −8.84% | −2.62% |
| 4004 | goal early | −4.64% | −3.05% |
| 4005 | goal middle | −3.64% | +27.80% |
| 4006 | LIBERO-10 middle | −9.44% | −4.65% |
| 4007 | LIBERO-10 late | +57.90% | +15.39% |

rho1.0 有5/8 cases改善，但4007贡献约114.19%的净退化，其余case合计有所抵消；不能只凭case wins判定更好。rho2.25 有4/8 cases改善，4005和4007合计贡献约94.44%的净退化。它们是现有 outer 上的风险诊断，不是新的 confirmation 结果；本阶段没有评估新 confirmation。

本次结果没有提供“同一对候选从单步到16步发生总体排序翻转”的证据，因此不能据此认定增加 unroll 长度即可解决原 pilot 失败。原搜索的 warm-state 单步目标与常数 rho 从头训练的目标仍在定义上不同；阶段1也已确认保留训练路径会改变 endpoint，但本阶段没有重现原搜索每轮的 warm-state 比较。

另一个直接可见的问题是：final16 的中间参考点 rho=1.5 优于两个探针端点。即便使用完整16步割线，其负方向也不足以支持离开1.5。后续选择必须比较真实 proposal endpoint 与 incumbent，不能根据最后一次方向自动交付 rho；本轮保留既有 incumbent 并停止扩展。

| rho | PBS 实际 GPU walltime | GPU·h | GPU telemetry samples |
|---|---:|---:|---:|
| 1.0 | 00:48:40 | 0.811111 | 194 |
| 2.25 | 00:46:43 | 0.778611 | 187 |
| 合计 | 01:35:23 | 1.589722 | 381 |

每项申请walltime均为01:59:00，使用1张A100；成本包含模型加载与checkpoint I/O，排队时间不计入GPU·h。两项均通过 PBS F/Exit0、artifact Exit0、PHASE_COMPLETE及匹配的完整summary。GPU利用率与显存按15秒记录。监控子代理显式配置为gpt-6-luna/xhigh，以600秒间隔检查；05:30的一次本地连接封装错误记为 inconclusive，由root在05:33补查，随后恢复正常，监控已在两项完成后停止。

实验共新增32次训练updates、48次候选推理，复用8份outer teacher及8条固定rho1.5 endpoint，新增teacher为0。完成后的比较仅对56条缓存小JSON做离线算术汇总，没有加载模型、tensor或checkpoint，没有新增model forward，也没有提交CPU比较作业或阶段3。原协议和原实验结果未改动。

证据范围为单一冻结C8训练轨迹、outer8×seed0、decoded BF16/fake A4的fixed-input teacher fidelity；两个候选0.183%的差距不构成训练seed复现证据。本阶段不支持closed-loop、native kernel性能或全局最优rho结论。

数据与完成证据：[离线比较JSON](offline_comparison.json)、[完成核对](final_verification.json)、[提交记录核对](submission_receipt.json)、[rho1.0终态](../stage2_rho1/terminal_status.json)、[rho2.25终态](../stage2_rho225/terminal_status.json)。各horizon五项指标、逐suite、逐case、batch方向和成本凭据均保存在比较JSON及相邻结果目录中。

分析流程采用 statistical-analysis 与 experimental-design 的配对单位和控制变量规范。流程来源：Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, Aubrey M. Brueckner. 2026. [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065)。此引用说明分析流程，不作为本实验结论的证据。
