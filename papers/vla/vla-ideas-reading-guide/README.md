# VLA 三个研究方向：自读资料库

根据师兄的 [VLA_idea.pdf](../../../idea/VLA_idea.pdf) 整理 · 来源核验日期：**2026-09-05**。

本库有 **30 个 numbered reading entries、31 份本地 PDF**：方向一 10 项、方向二 8 项、方向三 12 项；OAT 另附一份同研究线扩展稿。每项均有中文 README、English technical terminology、分时阅读路线、未作答的 Reading Questions 与 Meeting Card。阅读状态统一从 `unread` 开始。

这里的“有建树”按对问题的实际贡献筛选：建立基础 formulation、提出可复用方法、提供直接 VLA evidence，或构成关键对照。成熟基础与 2026 preprints 明确分开；近期论文入库不等于已经形成共识或已被独立复现。

| 方向 | 独立文件夹 | 最先读 | 需要区分的贡献 |
|---|---|---|---|
| 1. 让 token 预算跟着控制后果走 | [Control-aware action tokenization](01-control-aware-action-tokenization/README.md) | FAST → OAT → ActionCodec → SA-VLA | reconstruction / modelability / semantics / state conditioning / control consequence |
| 2. 让 VLA 先试探，再动手 | [Active probing and dual control](02-active-probing-and-dual-control/README.md) | Push to know! → Predictive Visuo-Tactile → CoMe-VLA → Dual Control | view selection / parameter identification / uncertainty detection / probing policy |
| 3. 把控制知识装进 VLA | [Control distillation and world models](03-control-distillation-and-world-models/README.md) | MPC-Net → DPC → TD-MPC2 → LaWAM → VLSA | teacher distillation / direct policy optimization / online solver / explicit safety layer |

[PDF inventory](PDF_INVENTORY.md) · [原总结文献逐项核对与纠正](REFERENCE_AUDIT.md) · [搜索范围与来源限制](SEARCH_REPORT.md) · [回主 Reading List](../reading-guide/README.md)

## 第一轮应带着哪些问题读

1. **方向一的 variable budget 不能单独作 novelty。** OAT 已支持 prefix-based anytime decoding；FAST 的 BPE 输出也不能一概称为每段固定 token 数。更具体的问题是预算是否由可测的 control consequence 决定。[OAT 阅读入口](01-control-aware-action-tokenization/03-oat/README.md)
2. **方向二不能概括为 VLA 从不主动获取信息。** CoMe-VLA 已研究主动感知；Push to know! 和 Predictive Visuo-Tactile 已做物理参数的主动交互估计。你需要明确 hidden variable、sensors、task-conditioned probing 与 policy integration 的差别。[方向二](02-active-probing-and-dual-control/README.md)
3. **方向三的宽泛结构已有历史。** MPC-Guided Policy Search、MPC-Net 已把 MPC 信息转入 neural policy；TD-MPC2、LaWAM 已减少或避免 future-image decoding。DPC 的严格 safety 部分涉及 online barrier intervention；ABNet 的 closed-form layer 也仍在 inference structure 中。[方向三](03-control-distillation-and-world-models/README.md)

以上是基于已读原文的范围判断，不是对三个具体 idea 的完整 novelty verdict，也不预先替你回答 Reading Questions。

## 分时阅读路线

### 2-hour orientation

每篇约 15 minutes：OAT、SA-VLA、Push to know!、CoMe-VLA、MPC-Net、DPC，共 90 minutes。余下 30 minutes 看本页、REFERENCE_AUDIT，并写下每个方向“已有的机制 / 尚未明确的最小问题”各一句。此路线用于定位，不表示读完全部论文。

### 1-day route（约 6 hours）

- 方向一 2 hours：FAST 20 min → OAT 60 min → ActionCodec 20 min → 填 comparison card 20 min。
- 方向二 2 hours：Push to know! 60 min → CoMe-VLA 30 min → PI-VLA / PFD 对照 15 min → card 15 min。
- 方向三 2 hours：MPC-Net 40 min → DPC 40 min → LaWAM / VLSA 25 min → card 15 min。

### 1-week route（约 10–14 hours）

Day 1–2 读方向一的 primary route；Day 3–4 读方向二的 primary route；Day 5–6 读方向三的 primary route；Day 7 用三张 Meeting Cards 和一张跨方向表讨论取舍。其余方法作为按问题展开的 extensions，不要求一周读完 30 项。

## 三张比较卡（读完后填写）

| 方向 | 必填字段 |
|---|---|
| Token budget | token / bit definition；可变预算由谁决定；distortion；contact timing；matched policy；wall-clock / success |
| Active probing | hidden variable；prior / posterior；可用 sensors；probe action set；information gained；探测时间和风险 |
| Control distillation | teacher information；student objective；world-model state；inference 剩余模块；guarantee assumptions；teacher-off degradation |

共同记录 `simulation / real robot`、trials、seed、baseline、training data、hardware、P50/P99 policy latency、实际 feedback frequency、success 与 failure type。这里只是阅读时要找的 evidence；未启动训练或占用 GPU。

## 与第一个 report 的关系

按你的要求，第一个 report 只新增两篇，并留在主阅读库独立整数编号：[#23 WaterSIC](../reading-guide/papers/23-watersic/README.md)、[#24 GRACE](../reading-guide/papers/24-grace/README.md)。本库的 30 项不混入主库 core count。
