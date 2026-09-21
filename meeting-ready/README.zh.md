# 组会材料入口

这里把目前三条重要、互相独立的路线整理成可继续实验的 bundle。原实验目录保留不动；从现在起，组会阅读和后续修改优先从这里进入。

## 1. Iteration cache：跨模型与 planner 验证矩阵

目录：[01-iteration-cache-cross-model-planners](01-iteration-cache-cross-model-planners/)

先看：

1. [MEETING_CARD.zh.md](01-iteration-cache-cross-model-planners/MEETING_CARD.zh.md)
2. [RESULT.zh.md](01-iteration-cache-cross-model-planners/RESULT.zh.md)
3. [README.zh.md](01-iteration-cache-cross-model-planners/README.zh.md)

核心结论：LeWM PushT CEM 和 DINO-WM Wall GD 已通过各自 system gate；DINO-WM Wall CEM 与 PushT CEM 保持 decision behavior 但未通过 latency gate。完整矩阵和四个 case 的继续入口都在该 bundle 内。

## 2. Horizon-weighted recurrent student（当前最佳 predictor student）

目录：[02-horizon-weighted-recurrent-student](02-horizon-weighted-recurrent-student/)

先看：

1. [MEETING_CARD.zh.md](02-horizon-weighted-recurrent-student/MEETING_CARD.zh.md)
2. [reports/RESULT_HORIZON_WEIGHTED.zh.md](02-horizon-weighted-recurrent-student/reports/RESULT_HORIZON_WEIGHTED.zh.md)
3. [CONTINUE_EXPERIMENTS.zh.md](02-horizon-weighted-recurrent-student/CONTINUE_EXPERIMENTS.zh.md)

核心结论：shared recurrent latent-transition student 将 predictor latency 降到 `10.719 ms`，相对 teacher reduction 为 `99.6936%`；recurrent architecture 的 ranking 改善方向一致，horizon weighting 继续提供很小但稳定的正向信号，不过仍未通过 frozen replacement gate。

## 3. LeWM latent-delta oracle（已完成）

目录：[03-lewm-latent-delta-oracle](03-lewm-latent-delta-oracle/)

先看：

1. [MEETING_CARD.zh.md](03-lewm-latent-delta-oracle/MEETING_CARD.zh.md)
2. [Step 1 结果](03-lewm-latent-delta-oracle/reports/RESULT_STEP1_COMPRESSIBILITY.zh.md)
3. [Step 2 结果](03-lewm-latent-delta-oracle/reports/RESULT_STEP2_ORACLE_RANK.zh.md)
4. [README.zh.md](03-lewm-latent-delta-oracle/README.zh.md)
5. [PROTOCOL.zh.md](03-lewm-latent-delta-oracle/PROTOCOL.zh.md)

核心结论：Step 1 的 `rank≤64` reusable model-delta structure gate **FAIL**；Step 2 的 fixed candidate ranking 从 model-PCA rank `64` 起通过，但相同 rank `64/96` 接入 official fixed-observation CEM 后均未保留 first action。该 oracle 先完整运行 LeWM，不是 speedup，也没有 closed-loop 结论。

## 前两条加速路线的关系

| 路线 | 优化对象 | 是否近似 teacher | 当前证据 |
|---|---|---|---|
| iteration cache | 同一次 CEM solve 中重复的 fixed-context encoding | 否 | exact、稳定约 `29.4%` full-planner reduction |
| recurrent student | action-conditioned predictor rollout | 是 | 极大 predictor-only speedup，ranking 正向但尚未达到 replacement gate |

它们作用在不同边界，理论上可以组合，但当前没有做二者组合后的端到端实验，因此组会上应分别汇报收益，不能直接相加。

## 使用边界

- bundle 内已经包含继续修改实验所需的代码、freeze、protocol、PBS、manifest/baseline summary 和关键小型结果。
- 大型 official source、dataset、checkpoint 和 Python runtime 没有重复复制；各 bundle 的 README/continuation 文档列出了这些外部资产。
- ASPIRE2A login node 仍只用于上传小型控制文件、提交和状态查询；模型加载、训练、推理、benchmark、重 I/O、下载、解压、编译和依赖安装必须在 PBS compute allocation 内完成。
- 原始 experiment 目录未移动或删除，便于保留 provenance；后续工作不需要再从原目录拼装材料。
