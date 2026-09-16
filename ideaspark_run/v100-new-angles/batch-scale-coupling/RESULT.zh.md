# Batch Scale Coupling — novelty / applicability no-go

2026-09-13。结论：**novelty_no_go，同时当前 baseline 未发现可干预的跨 candidate 动态 scale 路径**。本候选不提交 GPU 实验，预留 102–107 未使用。这不是经验上的不可行证明。

想法是检验同一个 action candidate 是否因同批其他 candidates 改变而得到不同量化分数，再以独立 scale 消除这种影响。但 [Quantamination](https://arxiv.org/abs/2604.26505) 已直接研究 dynamic per-tensor quantization 的跨 batch coupling 及 per-token 隔离。独立检索还检查了 [QuantWM 官方实现](https://github.com/huawei-noah/noah-research/tree/master/QuantWM) 的 quantizer、PTQ layers、observer 和 Wall 配置：所查看路径中的 dynamic per-token/per-channel scale 不跨 candidate batch，layer-wise observer 使用固定校准。此判断只覆盖审查过的路径，不宣称所有软件实现均如此。

人为加入一个当前不存在、且已有直接文献解释的跨 batch quantizer，再证明隔离有效，不能给本轮提供足够的新切面。因此停在 prior-art 与实现适用性筛选；不修改 baseline 制造故障，不把未运行写成实验失败，也不通过调参挽救。原始 [idea brief](IDEA_BRIEF.zh.md) 保留，便于审查被放弃的假设与实验设计。

ResearchStudio-Idea 使用到 literature / falsification gate 即结束；用户已授权跳过无关阶段。GPU 消耗：0。
