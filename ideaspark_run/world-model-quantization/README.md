# World Model × Quantization：当前研究入口

2026-09-10：用户选择探索 **CEM-Update PTQ**，替代原 RankCal 的全候选排序 calibration 目标。目前只有新方案文档，尚未实施新实验。

**新对话请先读 [CEM_UPDATE_PTQ_HANDOFF.zh.md](CEM_UPDATE_PTQ_HANDOFF.zh.md)**：修改动机、方法定义、与旧方案的区别、分阶段验证、停止条件和 ASPIRE2A 约束均在其中。

- [旧 RankCal screening 结果](experiments/dino-wm-wall/SCREEN_RESULTS.zh.md)：weak/mixed signal；按当时快速筛选策略 no-go。
- `phase0` 至 `phase4`：旧 RankCal 的 pipeline 历史，不是新方法的执行规范。
- `experiments/dino-wm-wall/`：旧实验与冻结证据，保留不覆盖。
- 新实验建议另存 `experiments/cem-update-ptq/`；本次未创建或运行该实验。
