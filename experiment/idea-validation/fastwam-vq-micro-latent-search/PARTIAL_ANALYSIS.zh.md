# Microscaling + action/latent distillation：前两个候选的阶段分析

本次纳入 ρ=0.1（25805582.pbs101）与 ρ=0.5（25805512.pbs101）。两者均通过完整完成检查：PBS F/Exit0、artifact exit0、PIPELINE_COMPLETE、匹配 protocol/run/job、614 targets、2 epochs/16 updates，以及完整 epoch0/1/2 selection history。续跑前的训练记录保留在完整 history 中。

CPU 汇总 25810166.pbs101 已通过 F/Exit0、artifact/marker 和 source-job/cohort 检查。第一次 CPU 汇总 25810115.pbs101 因 Python module 缺少 GCC prerequisite 失败，补齐 module 后成功；没有改训练数值流程。

## 测量范围

selection 是预先冻结的 cases1000–1003，每 case 两个 paired sampler seeds。先平均同 case 的两个 seed，再对四个 cases 等权平均；独立比较单位按四个 cases 理解，不能把 seeds 当成八个独立样本。误差均相对同条件 BF16 teacher；parent 是尚未训练的 Microscaling VQ+A4，epoch0 两组完全一致。

future NMSE 衡量两个 clean future latent frames 的全局误差，排除 clamped frame0；temporal delta NMSE 衡量 latent 帧间变化的误差，不能直接当成物体运动准确率。

本次新增模型 forward=0，test forward=0，没有读取或下载 .pt。只汇总既有小 JSON，数值计算位于真实 PBS CPU allocation。

λ=ρ×c，c=0.03033027132999191。ρ0.1 的 λ=0.0030330271329991913，ρ0.5 的 λ=0.015165135664995956。ρ 用于归一化初始 calibration loss 的相对强度；它不代表整个训练期间的实际 loss 比例或梯度比例。

## 汇总结果

括号中是相对未训练 Micro parent 的误差变化，负值表示下降。

| 版本 | first10 motor RMSE | full32 motor RMSE | late22 motor RMSE | first10 gripper RMSE | future NMSE | temporal delta NMSE |
|---|---:|---:|---:|---:|---:|---:|
| Micro parent | 0.047929 | 0.048523 | 0.048719 | 0.020474 | 0.050511 | 0.089023 |
| ρ0.1 / epoch1 | 0.044994（−6.12%） | 0.044515 | 0.044245 | 0.016201 | 0.032543（−35.57%） | 0.052796 |
| ρ0.1 / epoch2 | 0.041802（−12.78%） | 0.044273 | 0.045242 | 0.020333 | 0.029060（−42.47%） | 0.046638 |
| ρ0.5 / epoch1 | 0.044357（−7.45%） | 0.045075 | 0.045272 | 0.016673 | 0.027324（−45.91%） | 0.043556 |
| ρ0.5 / epoch2 | 0.042850（−10.60%） | 0.045170 | 0.046025 | 0.019828 | 0.027881（−44.80%） | 0.045997 |

四个训练后 checkpoints 均通过预定 20% guards（late22、first10 gripper、first10 case P90），这些 aggregate guard 指标甚至均低于 parent。guards 相对的是 parent，不是上一个 epoch，也不意味着每个 case 都改善。

在当前两个候选中，按原定 action 主指标、guards 和 1% near-tie 规则，暂时最优为 **ρ0.1 / epoch2，λ≈0.003033**；strict minimum 与该结果一致。其余三个 R1 与 R2 尚未纳入，不能称为全局 optimal λ。

## 发现一：future 更接近 teacher，不保证 action 更接近 teacher

同在 epoch2，ρ0.5 的 future NMSE 比 ρ0.1 低 4.06%，但 first10 motor RMSE 高 2.51%；full32 高 2.03%，late22 高 1.73%，first10 P90 高 3.71%。ρ0.5 的 gripper 则低 2.49%。因此两个版本之间存在不同输出指标的取舍，没有一个在所有指标上都占优。

| case | parent action | ρ0.1 action | ρ0.5 action | parent future | ρ0.1 future | ρ0.5 future |
|---|---:|---:|---:|---:|---:|---:|
| 1000 | 0.047442 | 0.048116 | 0.050529 | 0.024892 | 0.024125 | 0.020553 |
| 1001 | 0.048735 | 0.039681 | 0.041310 | 0.058474 | 0.032859 | 0.030680 |
| 1002 | 0.042495 | 0.041251 | 0.039673 | 0.044314 | 0.029449 | 0.022457 |
| 1003 | 0.053045 | 0.038158 | 0.039889 | 0.074366 | 0.029810 | 0.037837 |

epoch2 的 ρ0.1 在 action 上赢 ρ0.5 的 3/4 cases；ρ0.5 在 future 上赢 ρ0.1 的 3/4 cases。case1000 特别有解释力：两个版本的 future 均优于 parent，但 action 均差于 parent；ρ0.5 的 future 更好，action 退化更大。case1001 也呈现两个候选之间相同方向的取舍，case1002 则 action/future 同时更支持 ρ0.5。

这支持的描述性结论是：当前 global future NMSE 不能作为 action fidelity 的充分替代指标。它没有证明具体原因。可能涉及 action-sensitive latent 方向与全局 latent 误差的错配，以及不同输出对共享量化参数的要求不同；现有 JSON 没有区域/通道误差或梯度夹角，不能断言背景占比、梯度冲突或机制因果。

## 发现二：均值改善存在 case 依赖

ρ0.1 / epoch2 相对 parent 的 first10 改善来自 3/4 cases；case1003 对净改善贡献约 60.74%，case1001 约 36.94%，case1002 约 5.07%，case1000 是负贡献 −2.75%。这些是对 case RMSE 净差总和的贡献，不是概率或显著性。

epoch1 的 ρ0.5 平均 first10 比 ρ0.1 好 1.42%，实际却只赢了 1/4 cases：主要由 case1003 的大幅改善拉低均值。epoch2 的 action 排名反转后，ρ0.1 同时赢 3/4 cases。因此当前支持较低 ρ 的证据比单纯比较两个均值更具体，但四个历史 selection cases 仍不足以证明独立泛化优势。

## 发现三：继续训练也有输出与 horizon 的取舍

ρ0.1 从 epoch1 到 epoch2：first10 再下降 7.10%，future NMSE 再下降 10.70%；但 late22 上升 2.25%，gripper 上升 25.50%。后两项仍低于 parent，故不会触发以 parent 为参考的 guards。

ρ0.5 从 epoch1 到 epoch2：first10 再下降 3.40%，但 future NMSE 上升 2.04%、temporal delta NMSE 上升 5.60%，late22 上升 1.66%，gripper 上升 18.92%。ρ0.5 的最佳 future checkpoint 在 epoch1，而最佳 first10 action checkpoint 在 epoch2。

这说明 λ 与 epoch 都需要按预定目标选择。first10 loss 直接监督前十步 motor，late22 与 gripper 没有对应的直接 action loss；当前结果与这种监督范围的局限相容，但不能仅凭此认定全部退化原因，也不能把 selection future 回升直接定性为过拟合。

## 对当前实验的判断

Microscaling 下，action+latent distillation 已在这四个 selection cases 上同时降低 action 均值和 future 均值；较强 latent 权重并没有单调提高 action fidelity。因此仍应以 action 主指标选 checkpoint，让 latent 指标用于解释取舍，保持既有 guards、cohort 和两轮搜索规则。

**ρ0 action-only 的同配置对照仍然必要。** 目前两组都加了 latent loss；相对未训练 parent 的改善，也可能部分来自 action distillation 本身。不能单独归因 latent loss，更不能通过旧 Row-scale action-only 结果推断 Microscaling 下 latent loss 的净贡献。

本次不发布 round2 choice，不冻结最终 λ，也不评测候选 test cohort。按既有计划，第一轮五个候选纳齐后，再生成第二轮的三个候选（包含 ρ0），两轮完成后冻结全局选择并评测原22 cases×2 seeds。

证据仅是 decoded-BF16/fake-A4、固定输入的 teacher fidelity；没有新增 closed-loop 或 native-kernel 证据。本次不恢复定时监控。

原始 CPU 汇总与自动表格位于 `retrieved/results/partial-25810166.pbs101/`；验证记录为 `partial_analysis_terminal.json`。
