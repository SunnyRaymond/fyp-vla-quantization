# 23. WaterSIC

**WaterSIC: Information-Theoretically (Near) Optimal Linear Layer Quantization**  
Egor Lifar, Semyon Savkin, Or Ordentlich, Yury Polyanskiy · ICML 2026  
本地版本：**arXiv:2603.04956v2，2026-06-01，32 页** · 阅读状态：`unread`

- [本地 PDF](paper-arxiv-v2.pdf)
- [arXiv 与版本记录](https://arxiv.org/abs/2603.04956v2) · [ICML 官方条目](https://icml.cc/virtual/2026/poster/62598) · [OpenReview](https://openreview.net/forum?id=fCPgAHIciE)
- [返回 Reading List](../../README.md) · [师兄的原始 report](../../../../../idea/report_icml26.pdf)

## 为什么单独读

这是 report 中写作 “watersic” 的论文，正式名称为 **WaterSIC**。它直接讨论 weight-only PTQ 的 rate–distortion limit，以及 GPTQ 的误差补偿应如何与不同 input columns 的 quantization rate 结合。与师兄的 KLT + waterfilling 构想高度相关，应先看清它已实现的部分，再讨论差异。

ICML 收录身份已由官方条目核对；本地文件是带 ICML 排版的 arXiv v2，并未冒充单独下载的 proceedings PDF。报告称其为 Spotlight；本次官方页面显示 Poster，未独立确认 Spotlight 等级。

## Background / Problem

从单个 linear layer 的 `Y = WX` 出发，量化目标是用更短的 description 表示 W，同时控制 activation distribution 下的 output MSE。首先区分三个量：weight MSE、layer-output MSE、最终 LLM perplexity。前两个之间需要 covariance，最后一个还涉及跨层传播。

## Method / Key innovation

沿 `rate–distortion lower bound → Cholesky / SIC → column-dependent lattice spacing → entropy coding` 阅读。Section 3 的 PlainWaterSIC 用不同 column spacing 平衡量化误差；Equation (12) 给出 `αᵢ = c / |ℓᵢᵢ|`。它利用 sequential error compensation 接近 waterfilling，不要求 decoder 显式持有完整 PCA rotation。

Section 4 的 practical WaterSIC 还包含 correction、rescalers、实际 rate targeting 等步骤；不能把完整系统的收益全部归给一个 spacing 公式。WaterSIC-FT 是另外加入 fine-tuning 的版本，应与未 fine-tune 结果分开。

## Results：去哪里找证据

| 要核对的结论 | 本地 PDF 定位 | 读时记录 |
|---|---|---|
| rate–distortion lower bound | Section 3.1，p.4 | weight distribution、decoder 可获得的信息 |
| 0.255 bits rate gap | Theorem 3.3，p.6；Appendix B | asymptotic high-rate / low-distortion 条件，不能直接改写为所有低位宽 LLM 的 guarantee |
| 不同 columns 分配不同 rate | Algorithms 1–2，pp.5–6；Section 4 | spacing、entropy、metadata 各是什么 |
| LLM quality–size trade-off | Section 5，Tables 1–2，pp.8–9；Figures 1–3 | base model、context length、rate 计法、是否 FT |

上述是作者的理论与实验报告，未在本地复现 benchmark。**1–4 bits/weight 的平均 entropy rate 不等于每个元素都按原生 INT1–INT4 kernel 存储或执行。** storage compression 的收益需要另行验证解码开销、packing、memory bandwidth 和端到端 latency。

## 对 FYP / KLT idea 的启发与边界

这是 LLM layer reconstruction 方法，不是 control-aware action tokenizer，也没有直接证明 VLA closed-loop success。可比较的差异包括 distortion 是否来自任务后果、activation statistics 是否随 rollout 改变、rate 是否包含 rotation/scale/entropy metadata，以及目标硬件能否执行得到的格式。仅把名称换成 KLT + waterfilling 不足以建立新的贡献。

## 阅读路线

- **20 minutes**：p.1 → pp.4–6 的 problem / Algorithm 2 / Theorem 3.3 → Tables 1–2 → Section 5 的 limitations。
- **90 minutes**：推导 Equation (12)；对照 GPTQ 的 shared spacing；拆开 PlainWaterSIC、practical WaterSIC、WaterSIC-FT；重建一张同模型同 rate 的结果表。
- **3 hours**：读 Appendix B 的证明条件与 Appendix E 的 entropy coding 实验，再列出把 layer MSE 换成 task distortion 需要改变哪些假设。

## Reading Questions（读完自己回答）

1. 为什么 activation covariance 的 diagonal variance 不足以决定最优 column spacing？
2. `αᵢ`、rate 和 conditional innovation variance 的方向关系是什么？
3. waterfilling lower bound 允许 decoder 知道什么？实际部署又知道什么？
4. 0.255 bits 是 additive rate gap，还是 distortion percentage？何时成立？
5. Entropy estimate、compressed file size 和固定 bit-width 的数字能否直接放在同一横轴？
6. 哪个 ablation 最能隔离 unequal-rate allocation 的贡献？
7. FT、rescalers 和 propagated-error correction 各带来什么额外成本？
8. 如果用于 VLA，应优先替换 distortion、calibration distribution，还是 quantization format？如何证伪选择？

## Meeting Card（留空）

- Problem：
- 核心机制与公式：
- 最强证据及条件：
- 与 KLT + waterfilling 构想的重合：
- 尚未覆盖的最小问题：
- 想向师兄确认的问题：
