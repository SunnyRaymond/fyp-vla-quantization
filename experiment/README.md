# 当前实验与完整交接

从 **[HANDOFF.zh.md：完整技术交接](HANDOFF.zh.md)** 开始。它说明当前研究问题、Microscaling VQ 表示、A+Z+reader-KL、训练/VJP、inference engine、数据合同、主要结果、代码依赖和未完成工作。

逐项台账覆盖本目录保留的 **85 个 idea-validation + 2 个 reproduction，共 87 个目录**，包含完成、部分完成、取消、方案未提交和仅 protocol 的状态，不把旧计划写成已有结果。

| 阅读顺序 | 文档 |
|---|---|
| 1 | [总设计、实现和当前结论](HANDOFF.zh.md) |
| 2 | [基础管线与 baselines](handoff/01_FOUNDATIONS_AND_BASELINES.zh.md) |
| 3 | [VQ、latent 及机制诊断](handoff/02_VQ_AND_LATENT.zh.md) |
| 4 | [Scaling、clipping、calibration 和控制诊断](handoff/03_SCALING_CLIPPING_AND_DATA.zh.md) |
| 5 | [各种 KL 与优化落差](handoff/04_KL_AND_OPTIMIZATION.zh.md) |
| 6 | [R128、当前 matched loss 与 AQLM](handoff/05_R128_MATCHED_LOSSES_AND_AQLM.zh.md) |
| 7 | [Inference engine、标准 LIBERO 与 Cosmos](handoff/06_ENGINE_AND_COSMOS.zh.md) |
| 8 | [环境、资产与接手操作](handoff/07_OPERATIONS.zh.md) |
| 按目录查 | [HANDOFF_INDEX.json](HANDOFF_INDEX.json) |

当前 Fast-WAM 算法参考为 `25921795 / reader_kl final8`：matched LIBERO-Plus 闭环 208/280，对照 A+Z204/280、BF16216/280；推荐 runtime 为 C+decoder+buffer，full 80 action 精确 parity、独立 future anchor 精确，policy-call p50 0.887096 s。主 Linear 仍是 decoded-BF16/cuBLAS，不是 native INT4/VQ GEMM。

Cosmos 最新六臂 ours230/280、BF16233/280，已包含 reader-KL；旧 C8 A+Z226/280 另存。AQLM fixed-input 两臂已完整，ours 在 40 个 held-out contexts 的报告误差全部更差；后续闭环在 525/560 条时按用户要求停止，共同 245 条为 reconstruction179/245、ours125/245，ours5 胜 59 负 181 平，缺 35 条不算失败且不自动恢复。实时状态读各实验当前 handle 和 monitor，本文档不是自动恢复旧作业的指令。

2026-10-27 新增标准 LIBERO 结果：同一 Q0+推荐 engine 完整 **391/400（97.75%）**，Spatial/Object/Goal/Long 为 98/99/97/97（各 100）；四份完整 action/future anchor 均 0，6415 次 warm policy-call p50 为 0.899527 s。此面板不是 Plus280，未跑同 400-slot BF16 对照；见[完整结果](idea-validation/fastwam-libero-engine-20261027/RESULTS.zh.md)。

初稿日期为 2026-10-26，本次增补为 2026-10-27；新增文档没有更改实验代码、cohort 或集群作业。写作时只读取本地资料，未重跑训练/推理。原始大数据、模型和远端资产不全在 Git 里，恢复边界详见操作手册。
