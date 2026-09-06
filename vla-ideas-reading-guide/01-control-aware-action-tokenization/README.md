# 方向一：让 token 预算跟着控制后果走

原文：[VLA_idea.pdf pp.1–2](../../VLA_idea.pdf) · [三个方向总览](../README.md)

**10 个编号条目，11 份 PDF（OAT 含扩展稿）。** 问题是压缩误差如何转成 task cost，而不只是码长或 action MSE。Action tokenization 的 vector quantization 与压缩 model weights 的 QAT/PTQ 是两个不同层级。

| # | Paper / notes + PDF | 在这个方向上的贡献 | 阅读层级 |
|---:|---|---|---|
| 1 | [FAST](01-fast/README.md) | DCT + BPE 的通用 action compression interface | 基础；RSS 2025 |
| 2 | [VQ-VLA](02-vq-vla/README.md) | synthetic/real trajectory scaling 的 learned tokenizer | 基础；ICCV 2025 |
| 3 | [OAT](03-oat/README.md) | ordered prefix、total decodability、anytime budget；附 7 月扩展稿 | **先精读**；2026 preprints |
| 4 | [ActionCodec](04-actioncodec/README.md) | 从 VLA optimization 反推 tokenizer design | **先精读**；2026 preprint |
| 5 | [FASTer](05-faster/README.md) | neural tokenizer + block-wise decoding + action expert | 方法扩展；preprint |
| 6 | [NAC](06-nac/README.md) | neural audio codec / RVQGAN 用于 action signals | 方法扩展；2026 preprint |
| 7 | [X-Tokenizer](07-x-tokenizer/README.md) | semantic interface 与 asymmetric residual quantization | 语义对照；2026 preprint |
| 8 | [SA-VLA](08-sa-vla/README.md) | state-conditioned detokenization | **直接对照**；2026 preprint |
| 9 | [FlashVLA / Think Twice, Act Once](09-flashvla/README.md) | visual pruning + action reuse 的效率对照 | systems companion；preprint |
| 10 | [BEAST](10-beast/README.md) | B-spline、uniform token length、parallel decoding、smooth trajectory | 解析对照；NeurIPS 2025 |

## Primary route（约 3–4 hours）

FAST 30 min → BEAST 30 min → OAT 60 min → ActionCodec 45 min → SA-VLA 30 min → 填表。随后按需要选 VQ-VLA / FASTer / NAC（learned codec）、X-Tokenizer（semantic alignment）或 FlashVLA（系统效率）。

## 阅读时应修正的前提

“上面所有 tokenizer 都是固定码率”不成立。OAT 可按 prefix length 解码，FAST 的 BPE 会产生随内容变化的 token length；BEAST 则明确强调给定配置下 uniform length。要分别写清 `fixed maximum capacity`、`variable output length`、`online risk-conditioned allocation`。

本方向可以研究的更细问题是：能否把 contact failure / task progress / constraint margin 引入 distortion，且在相同数据、policy、wall-clock budget 下优于 reconstruction、semantic 或 state-aware baselines。这是待验证问题，不能因为本库未发现完全相同组合便宣布首次。

## 本方向对照卡（留空）

| Paper | Distortion / loss | Token budget 是否可变 | 谁决定 budget | 是否优化 closed-loop consequence | 真实 latency 与 control rate |
|---|---|---|---|---|---|
| FAST | | | | | |
| OAT | | | | | |
| ActionCodec | | | | | |
| SA-VLA | | | | | |
| 我的构想 | | | | | |

## 进一步展开

原总结的 OmniSAT、Wall-OSS-0.5、VLA-Pruner 已保留为 [扩展入口](../REFERENCE_AUDIT.md)，暂不要求与本方向的第一轮主线一起阅读。[WaterSIC](../../reading-guide/papers/23-watersic/README.md) 可补 rate–distortion 思路，但其 layer-output MSE 不是 action consequence。
