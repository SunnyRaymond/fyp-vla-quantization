# A Survey of Quantization in LLM: Unlocking Potential Hardware Efficiency

> **Reading-list role**: Optional companion — newest dedicated LLM quantization survey snapshot  
> **Verification**: `verified-full-text` — JCST 2026 peer-reviewed journal article  
> **Recommended effort**: 30–45 minute map read；重点 Sections 3, 6–8

> **Local full text**: [paper.pdf](paper.pdf)

## 1. Paper identity

| Field | Value |
|---|---|
| Authors | Yi-Dong Chen, Kai-Jun Zheng, Zhen-Hua Guo, Qi-Hao Zhang, Yong-Hua Zhang, Ji-Dong Zhai |
| Year / version | January 2026; received 2025-09-24, accepted 2025-12-29 |
| Venue / status | *Journal of Computer Science and Technology* 41(1):341–358; peer-reviewed survey |
| DOI | [10.1007/s11390-026-5979-1](https://doi.org/10.1007/s11390-026-5979-1) |
| Primary source | [official article page](https://jcst.ict.ac.cn/article/doi/10.1007/s11390-026-5979-1) · [official PDF](https://jcst.ict.ac.cn/cn/article/pdf/preview/10.1007/s11390-026-5979-1.pdf) |

**Page boundary.** Local PDF page 1 是 JCST cover；article 从 PDF p. 2 / printed p. 341 开始。因此下面 locator 使用 local PDF page。

## 2. Why this is the “new” survey

截至 2026-08-25，本次检索中最新、标题与 scope 都直接聚焦 `LLM quantization` 的 peer-reviewed survey 是这篇 JCST 2026 article。它引用到 2025 methods，并把 quantization 放进完整 lifecycle：pretraining、fine-tuning/PTQ-QAT、inference、KV cache 和 kernel generation。

“最新”不等于“最完整”。这篇只有 18 个 article pages、86 references，优势是快且 systems-oriented；若要长期查 taxonomy/frameworks，应该同时使用 [Low-bit LLM survey](../020-low-bit-llm-survey/README.md)。

## 3. Survey map

| Layer | Paper coverage | Representative items | Locator |
|---|---|---|---|
| Quantization basics | symmetric/asymmetric, scale/zero-point, per-tensor/channel/group | uniform mapping、granularity | Sec. 3, PDF pp. 5–8 |
| Pretraining | AMP, FP8, FP4, low-precision communication/optimizer states | Transformer Engine, MS-AMP, FP8-LM, BitNet | Sec. 4, pp. 8–10 |
| PTQ / QAT | no-retraining calibration vs quantization-aware adaptation | GPTQ, AWQ, QLoRA, LLM-QAT | Sec. 5, pp. 10–11 |
| Inference objects | weight-only, weight-activation, mixed precision, outliers | GPTQ/AWQ, SmoothQuant/QServe, QUIK/Atom/MixQ | Sec. 6, pp. 11–13 |
| KV cache | dynamic range、outliers、long-context memory | KVQuant, KIVI, SKVQ, ZipCache, AQUA-KV | Sec. 7, pp. 13–14 |
| Kernel generation | hand-tuned vs compiler/DSL vs automated generation | Marlin, Triton, TVM, Ladder, Tilus, QFactory | Sec. 8, pp. 14–15 |

## 4. What this survey does well

1. **Lifecycle view**：把 training precision、PTQ/QAT、inference quantization 分开，避免只把 quantization 理解成下载一个 W4 checkpoint。
2. **Object view**：明确 weights、activations、outliers 和 KV cache 是不同 target，不能只报一个 nominal bit-width。
3. **Hardware view**：Section 8 直接讨论 packing、dequantization、memory hierarchy、hand-tuned kernel 与 compiler fragmentation，和 Final Year Project 的 deployment focus 高度一致。
4. **Recency**：references 覆盖到 2025 的 EfficientQAT、SliM-LLM、Marlin、Tilus、QFactory 等，明显比 2024 survey snapshot 更新。

## 5. Important caveats while reading

- **不是 systematic review**：paper 没有给 reproducible database search、inclusion/exclusion 或 quality assessment；它是 expert narrative survey。
- **PTQ 不是 fine-tuning**：Section 5 将 PTQ/QAT 都放在 “Fine-Tuning with Quantization” 下，taxonomy 方便但术语容易误导。PTQ 的定义恰恰是不更新 pretrained model parameters；QAT 才包含 training/adaptation。
- **method 与 system evidence 混合**：某些 speedup 来自 algorithm + kernel + hardware co-design，不能归因给 bit-width 本身。
- **authors' intellectual proximity**：paper 讨论并引用 authors' MixQ/QFactory-related line of work；这不使结果失效，但 comparison claims 应回到 original papers 验证。
- **reference quality 不均匀**：86 entries 混合 conference papers、arXiv、software/documentation；survey 本身不能替代 primary source。
- **2026 后续缺口**：即便 January 2026 已很新，也不会覆盖 2026 年后续 methods；快速领域中的 framework/kernel support 还需查 current documentation。

## 6. How to use it with the current reading route

最有效顺序不是从头逐字读，而是：

1. 先用 Figure 3（PDF p. 4）给当前 papers 定位。
2. 读 GPTQ 后回看 Sec. 6.1（pp. 11–12），确认它是 weight-only Hessian-aware PTQ。
3. 读 SmoothQuant/AWQ 后回看 outlier、mixed-precision 与 weight-activation discussion（pp. 12–13）。
4. 读 Sec. 7–8（pp. 13–15），把 algorithmic compression 接到 KV-cache memory 与 actual kernels。
5. 需要更细 taxonomy/implementation table 时，再切到 2025 *Neural Networks* survey。

## 7. Reading questions

1. 为什么 lifecycle taxonomy 与 quantized-object taxonomy 必须同时保留？
2. `W4A16`、`W8A8`、`W4A8KV4` 分别在 memory/compute path 上省掉什么？
3. 一个 algorithm 报 PPL near-lossless，为什么仍可能没有 end-to-end speedup？
4. hand-tuned Marlin-like kernel 与 compiler-generated kernel 的 portability/performance trade-off 是什么？
5. 对 VLA，除了 weights/activations/KV cache，还应把哪些 state/action buffers 纳入 quantization taxonomy？
6. 本 survey 哪些 2025 claims 值得回到 primary paper 做 replication-quality check？

## 8. Weekly meeting card

- **Role**：2026 dedicated LLM quantization snapshot，而不是新 quantization algorithm。
- **Best contribution**：把 pretraining → PTQ/QAT → inference objects → KV cache → kernel generation 连成一条 deployment chain。
- **Best section for this project**：Sec. 8 high-performance quantization kernels。
- **Biggest limitation**：narrative survey、search protocol 不透明；coverage 很新但不等于 complete/current implementation truth。
- **Use with**：GPTQ/SmoothQuant/AWQ primary papers + 2025 low-bit survey。

## 9. Status & evidence boundary

- **Source claim**：bibliographic status、paper taxonomy、86 references、Sections 3–9 内容来自 official JCST full text。
- **My interpretation**：它是本次检索中的 latest dedicated peer-reviewed snapshot，以及与 VLA deployment route 的对应关系。
- **Open question**：2026 methods、modern VLA/MoE/multimodal kernels 与 current framework support 需要持续更新。

