# 初版候选：主代理独立检查

2026-09-08。对象为两条分支初次 `phase2_generate_output.json`；不是 pipeline 审稿结论，也不修改 guarded fields。以下问题应由 coherence / critique / implementability 各自按证据处理。

## RankCal

1. **单点与联合配置的差异。** I_g 在其余位置均为 FP16 时估计，但最终是许多位置同时量化。累加 I_g 的分配目标可能被交互项破坏。应执行至少一个两位置协同/抵消反例，明确这一分数只是启发式，或说明可识别部署影响所需条件；不可声称为联合最优。
2. **预算和负对照可实现性。** 各位置的 peak-byte cost 通常不能直接相加为整体峰值；activation 生存期、共享 tensor、weight copies、scales 和实际 kernel layout 都影响结果。任意置换 bit assignments 不一定同时保持 bit histogram 和 bytes；需要实际可执行的匹配规则。
3. **quantizer 支持边界。** `torch.ao.quantization` 不是所有 W4A4/W8A4 组合在 A100 上已有可用执行路径的证明。fake quant 和实际 kernel 支持应分开。
4. **验证规模。** 六轴 full nuisance grid 的组合数量很大，M0 又由预算决定。12 GPU-days 是未实测估计；不能把大网格称为已经可在单卡轻松完成的最小试验。此处仅报告风险，不擅自改写锁定的 compute_budget/falsification。
5. **novelty。** 需要对照 VAML / VaGraM 等旧谱系，以及普通 task-aware PTQ。teacher-relative rank agreement 不直接等于真实环境任务质量。

## Causal Consequence Calibration

1. **同样存在干预背景不一致。** C_r 在 all-other-FP16 背景估计，却在联合 low-bit 配置中用作 top-B protection；它并不天然等于量化部署背景中的边际收益。
2. **被估计量与任务后果。** action-chunk MSE + 下一帧 pixel L1 是 paired downstream discrepancy，不等于任务成功、接触风险或 marginal task gain。一步观测还不能识别后续 chunk/replan 的全部后果；需明确假设与可证伪范围。
3. **保护单位与真实代价。** branch/module/denoising-step/axis tuple 的数量预算不自动匹配真实 bytes 或 latency。动态 weight precision 的多个副本与切换成本、activation schedule 必须纳入部署边界。
4. **可复现干预。** 相同 simulator state 必须包含必要的 controller/RNG/history/cache 状态；仅重置可见位置不必然产生有效配对。最终实现须给出可执行的恢复语义。
5. **模型与预算身份。** 要锁定真正有 future/action 耦合的 Optional IDM checkpoint 与 idm 模式，而非普通 first_frame checkpoint。1.2 GPU-day 的 80GB-class equivalent 没有测量转换时不能等同于 A100 40GB 的固定时长；inference 能放下不证明全部校准能放下。

## 两条 idea 的区分

两者目前都采用单点干预分数→precision allocation。具体 estimand 和系统结构有所不同，但最终应说明为什么它们需要不同机制，而非仅将 ranking error 换成 action/pixel discrepancy。不能因为分别叫 WM 和 WAM 就视为两个充分不同的算法贡献。
