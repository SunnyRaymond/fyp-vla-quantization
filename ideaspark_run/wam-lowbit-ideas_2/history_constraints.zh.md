# WAM W4A4/W4A8 历史约束

截至 2026-10-03。只整理可防止候选重复旧失败 recipe 的结论；不产生新 idea、不运行实验。

## 已失败范围与禁止照搬项

- **OTC-PTQ / action-aware `C_r`**：历史 recipe 是 Fast-WAM Optional IDM 的 `idm` 路径、LIBERO-Goal paired replay、W4 对称 per-output-channel fake RTN；记录为 32 states、256 rows。`C_r=d_a+d_o` 中 observation 约占 99.77%，OTC 与 Local MSE 选中相同 top-2 sites，CAL-CHECK 稳定性也弱于 action-only，故该 recipe 为 NO-GO。不要只改名或重写同一 observation 主导的加和分数。此结论只覆盖该 checkpoint、`idm` 路径及校准/检查划分；不等价于所有 WAM、`first_frame`、joint 路径或原生 INT4 的失败。
- **RankCal / CEM-Update**：当前 run 摘要记载其 DINO-WM 原 recipe 为 NO-GO；历史筛查中 RankCal 未改善 CEM elite/action fidelity 或 held-out success，CEM-Update 虽工程实现通过，机制未通过。不要再以相同的全候选排序代理或 CEM 下游损失包装新名字。指定 WM 历史目录缺失，无法在本轮复核更细任务、门槛和数值。
- **V100 reference-branch coupling**：旧结论为机制 NO-GO：6 组复用 Wall pairs、H=1、balanced 3×3 分配下，0/6 达到联合 10% 门槛（A 1.29%–2.79%，B 2.30%–5.52%）。仅此配置被否决；不能把它扩展成所有误差耦合机制都无效。重复 conditioning 本身已有既有架构先例；DINO-WM 的空间广播屏幕只支持窄的一步视觉 MSE 结果，不证明 WAM W4 控制收益或部署加速。

## 仍未知

- `C_r` 的 observation dominance 是否会在不同 WAM checkpoint、`first_frame`/joint 模式、W4A4 与 W4A8 下持续；当前记录没有这些对照。
- 以上历史结果都未证明 WAM 原生 packed INT4 的 kernel、端到端延迟或闭环收益；混合精度例外、真实有效位宽及目标硬件仍需各自说明。
- 新候选若与失败 recipe 的模块、误差分数、分配规则及校准目标实质相同，应视为重复；要解除此约束，必须有不同且可测的机制假设。这里不替新候选作判断。

## 来源与范围

可读摘要：`ideaspark_run/wam-lowbit-ideas_2/CONTEXT.zh.md`（提供 OTC、RankCal、CEM-Update 的历史 NO-GO 概述）。历史 OTC protocol 细节及 V100 数值来自已有历史摘要索引，本轮未重查原始实验文件。

按要求限定的旧目录在当前工作树均缺失：`ideaspark_run/world-action-model-quantization/`、`ideaspark_run/world-model-quantization/`、`ideaspark_run/v100-new-angles/`（索引曾指向 `READ_RESULTS.zh.md`）。因此旧 WAM 候选、完整 WM 记录及 V100 原件未能在限定目录内复核；不据此推断其余候选结论。
