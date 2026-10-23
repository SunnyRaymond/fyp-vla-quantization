# VQ 与 activation 两阶段 soft-bin + relational KL

已完成：两个新 endpoint 的完整 C80 与原 held-out 评估、旧端点复用和配对汇总均已结束。见 [最终实验报告](results/ANALYSIS.zh.md)、[原始比较结果](results/comparison.json) 与 [完成证据](completion_evidence.json)。当前结果支持保留 MSE-VQ8 + relational-KL activation80 作为 reference；本轮两个阶段均改 KL 后，八项总体指标在两个 split 上均变差。

仅替换原 C8/8updates 的 VQ action+latent aware 训练目标；其后保持当前 C80/80updates per-channel D + per-Linear clipping 实现。VQ 从原始 C8 bank 开始，不应用旧 MSE-aware masters；activation 从新 VQ final8、D1/r1 开始。两个阶段均用相同冻结 KL objective，无 MSE/NMSE 反传项。

数据、teacher、seeds、C80 mean/std、alpha、tau、beta、lambda 全复用；VQ 阶段更新原 C8 的8个 contexts，activation 更新原 C80 的80个 contexts。VQ final8 和 activation final80 各评完整 C80/80queries 与原 heldout40cases×2seeds/80queries，保存 action/future 以复用。固定 checkpoint，不用 heldout 选步数，无 development set，无闭环。

主要比较新两阶段 KL pipeline 与既有 MSE-VQ + KL-activation relational final80；纯 softbin 和历史 MSE activation 作参照。VQ-only endpoint 用于分离 activation 阶段的变化，不当作同预算 final80 对照。每组单条训练轨迹，配对统计按 case，不将两个 seed 当独立重复。

每个 GPU 作业 01:59:00、软截止6840秒，每15秒记录 GPU utilization/memory。gpt-6-luna/xhigh subagent 每十分钟只读监控；Q/R无实质变化安静。Login node 只轻量状态/提交/小文件控制；所有模型、张量、聚合和重I/O均在真实 PBS allocation，保留 host-key verification，不输出 credentials。
