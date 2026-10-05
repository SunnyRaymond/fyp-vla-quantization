# FastWAM W4A8 Phase-Controlled Subtractive Dither

这是 week ending 2026-10-04 周报涉及的 PTQ 实验包。实验从 `ideaspark_run/wam-lowbit-ideas_2/` 移入此目录；原作业编号、scratch 路径、协议及原始结果的来源记录保持原样。

## 阅读入口

| 目录 | 内容 |
|---|---|
| [proposal/final_candidate.json](proposal/final_candidate.json) | 初始 phase 候选定义；这是实验前 proposal，结论以结果报告为准 |
| [mechanism-screen](mechanism-screen/RESULT.zh.md) | 全 first_frame Linear 路径的 W4A8 初筛、phase/direct PTQ 校准、七 arms 对照；含数学及架构 gate |
| [new-draw-a8](new-draw-a8/README.zh.md) | 新 draws 2101–2104 的 independent / RTN / locked phase 对照；从已有汇总单独提取 A8 字段 |
| [paired-followup](paired-followup/RESULT.zh.md) | 新 draws 下的完整 action 输出误差分解；保留完整 suite 需要的 `closedloop.py` helper |
| [libero-goal-suite](libero-goal-suite/RESULTS.zh.md) | 2000 episodes 的完整 LIBERO-Goal 比较与聚合证据 |

## Idea 与当前判断

起点是 [Q-WAM v1 Appendix A.2](https://arxiv.org/html/2609.33269v1#A2) 的 final-action error formulation：局部 rounding residual 经剩余网络及 denoising steps 传播到最终 action。它对零均值、跨位点及与下游 Jacobian 不相关的假设，使 model-wide error 简化为逐层项；相邻 denoising steps 仍可能产生相关 residual。

我们据此测试 quantizer 是否能控制这些交互：共享 uniform draw，给不同位点加 phase offset，再用 subtractive dither 重建，改变 residual 的联合相关性。固定输入、无 clipping 时的边际性质不能推广为完整共享-draw 网络无偏；完整推导见 [数学 gate](mechanism-screen/MATH_GATE.zh.md) 与 [架构 gate](mechanism-screen/ARCHITECTURE_GATE.zh.md)。

初始八组 phase 在冻结 draws 上有小幅 MSE 改善；新 draws 改变了与 independent dither 的排序，完整 suite 也没有显示成功率优势。当前粗粒度 recipe 的扩展 NO-GO 保持不变。尚需通过 weight/activation 的细致 sensitivity analysis 区分机制限制与实现、分组或校准问题。

## 使用与恢复

源码按同级目录引用 `mechanism-screen/runner.py`、`paired-followup/decompose.py` 和 `paired-followup/closedloop.py`。PBS 文件是实际执行的 scratch 环境快照；本地搬目录不表示远端目录已迁移。恢复环境后，需按实际 scratch 根目录部署相应控制文件，并配置原 FastWAM/container/checkpoint/stats 与 source identity 文件。

旧 pilot 的 `locked_selection.json`、`frozen_observations.pt` 等来自远端 `wam-phase-screen-20261003/artifacts/25666715.pbs101/`；本地文本结果与 calibration history 不能代替完整二进制输入。`new-draw-a8` 本地没有 A8 raw `result.json`，其摘要明确标记为混合 aggregate 的字段提取。全量大 trace/模型留在原 scratch 或本地被忽略文件中；此包提供报告证据与运行源码，不声称无需外部输入即可完整复跑。

所有模型计算、重 I/O 和结果重聚合须在获批 PBS allocation；login node 仅做轻量控制。各 GPU launcher 在运行期间每 15 秒写入 GPU utilization 与显存记录。
