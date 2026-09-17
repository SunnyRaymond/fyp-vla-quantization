# A Survey of Low-bit Large Language Models: Basics, Systems, and Algorithms

> **Reading-list role**: Optional companion — comprehensive low-bit LLM quantization reference  
> **Verification**: `verified-full-text` — *Neural Networks* peer-reviewed journal record；local copy is arXiv v3 author manuscript  
> **Recommended effort**: Reference read；先用 figures/tables，按需读 Sections 2–5

> **Local full text**: [paper-arxiv-v3.pdf](paper-arxiv-v3.pdf)

## 1. Paper identity

| Field | Value |
|---|---|
| Authors | Ruihao Gong, Yifu Ding, Zining Wang, Chengtao Lv, Xingyu Zheng, Jinyang Du, Yang Yong, Shiqiao Gu, Haotong Qin, Jinyang Guo, Dahua Lin, Michele Magno, Xianglong Liu |
| Local version | arXiv:2409.16694 v3, revised 2025-11-12 |
| Venue / status | *Neural Networks* 192, article 107856, December 2025; peer-reviewed journal article |
| DOI | [10.1016/j.neunet.2025.107856](https://doi.org/10.1016/j.neunet.2025.107856) |
| Primary source | [arXiv](https://arxiv.org/abs/2409.16694) · [publisher record](https://www.sciencedirect.com/science/article/pii/S0893608025007361) |

**Version boundary.** Local file 是公开可下载的 arXiv v3 author manuscript，不是 Elsevier typeset Version of Record；venue/volume/article-number 由 DOI metadata 核验。Crossref record 列出 153 references，说明它的 coverage 明显比 18-page JCST survey 更广。

## 2. One-sentence takeaway

这篇 survey 以 `Basics → Frameworks/Systems → Efficient Training/PEFT → QAT/PTQ → Toolkits/Benchmarks → Future Directions` 组织 low-bit LLM field，特别适合查 number format、granularity、kernel/dataflow 与 algorithm family 的相互位置。

## 3. Structure and high-value locators

| Question | Best place to read | What to extract |
|---|---|---|
| 同 bit-width 为什么表现不同？ | Sec. 2, PDF pp. 2–5 | INT/FP/custom formats、tensor/token/channel/group granularity、static/dynamic |
| framework 真正支持什么？ | Table 2 + Sec. 3, pp. 6–12 | algorithms、bit-width、hardware、model family；注意 support ≠ every configuration deployable |
| low-bit training 与 quantized PEFT 如何区分？ | Sec. 4, pp. 13–15 | low-bit training、partial parameter fine-tuning、QLoRA-like structures |
| PTQ methods 如何分类？ | Figure 8 + Sec. 5.2, pp. 16–22 | equivalent transformation、compensation、mixed precision、combination、vector quantization |
| SmoothQuant/AWQ/GPTQ 在哪里？ | pp. 18–20 | scaling transformation vs Hessian compensation |
| toolkits 与 evaluation 怎么选？ | Table 5 + Sec. 5.3, p. 23 | LLMC、LMQuant、benchmarks/backends |
| 接下来有什么 gap？ | Sec. 6–7, pp. 23–24 | KV cache、multimodal/MoE、hardware co-design、extremely low bit |

## 4. The taxonomy you should keep

### 4.1 Three independent axes

1. **Lifecycle**：pretraining、fine-tuning/PEFT、post-training/inference。
2. **Quantized object**：weights、activations、KV cache、gradients、optimizer states。
3. **Error-control mechanism**：equivalent transformation、Hessian/compensation、mixed precision、rotation/reordering、codebook/vector quantization、QAT/distillation。

同一个 method 可以同时占多个格子。例如 QServe 是 W4A8KV4 setting、使用 scaling/rotation，并且包含 serving kernels；只把它写成 “4-bit method” 会丢掉大部分信息。

### 4.2 GPTQ, SmoothQuant, AWQ in this map

- **GPTQ**：Sec. 5.2.2 `Compensation`；用 $H^{-1}$ 在 quantizing weights 时补偿 remaining weights。
- **SmoothQuant**：Sec. 5.2.1 `Scaling Transformation`；把 activation outlier difficulty 迁移到 weights，目标是 W8A8。
- **AWQ**：也使用 scaling，但 activation 是 weight saliency signal，目标通常是 regular W4A16/W3A16 weight-only path。
- **SpinQuant/QuaRot**：`Rotation Transformation`；改变 representation geometry，让 outliers 变得更均匀。

## 5. What this survey does better than the 2026 short survey

- **Breadth**：28 pages、153 references；包含 number formats、framework table、training/PEFT、MLLM/MoE、vector quantization 与 toolkits。
- **Systems detail**：Figure 4–6 把 cache/memory traffic、weight-only、W&A 和 KV-cache quantization 的 speedup path 分开。
- **Method taxonomy**：Figure 8 按 equivalent transform / compensation / mixed precision / combination 组织 PTQ，比单纯按 bit-width 更能解释 method lineage。
- **Practical lookup value**：Table 2/5 适合做 implementation shortlist，而非每次从几十篇 primary papers 重新搜索。

## 6. Limitations and freshness boundary

- local v3 虽在 2025-11 更新，主体 evidence 仍高度集中在 2021–2024，不能覆盖 2026 的 SLQ、new FP4 recipes 或最新 runtime state。
- framework/support tables aging 很快；部署前必须查 current official documentation 和 actual kernel matrix。
- 它是 narrative survey，不是 PRISMA-style systematic review，也没有对每篇 source 做统一 risk-of-bias assessment。
- 大量 method claims 来自 original authors' reported PPL/accuracy/speedup，hardware、model、group size 与 kernel 并不 matched；survey table 不能当 leaderboard。
- LLM/VLM coverage 不能直接证明 VLA closed-loop robustness；continuous action、temporal error accumulation 和 control frequency 基本不在 scope 内。

## 7. How to read it

### 30-minute route

1. Figure 1（p. 3）：先看整张 taxonomy。
2. Figures 2–3（p. 5）：granularity + static/dynamic。
3. Figure 5（p. 9）：区分 weight-only 与 W&A speedup path。
4. Figure 8（p. 17）：把 SmoothQuant、GPTQ、AWQ、SpinQuant 放进 PTQ map。
5. Table 5 + Sec. 6（pp. 23–24）：toolkits 与 future gaps。

### 2-hour reference route

1. **0–25 min**：Sec. 2，建立 format/granularity vocabulary。
2. **25–50 min**：Sec. 3，只记录与你 hardware 有关的 framework/backend rows。
3. **50–70 min**：Sec. 4，区分 low-bit pretraining、QLoRA-style memory saving 与 deployable quantized model。
4. **70–105 min**：Sec. 5.2，按 transformation → compensation → mixed precision → combination 读。
5. **105–120 min**：Table 5、future directions，并写出你自己的 VLA extensions。

## 8. Reading questions

1. 为什么 nominal `4-bit` 不能决定真实 memory footprint 与 latency？
2. static/dynamic、per-token/per-channel/group-wise 分别把 runtime overhead 放在哪里？
3. equivalent transformation 与 Hessian compensation 是 orthogonal 吗？组合时哪一个先做会影响结果吗？
4. weight-only decode speedup 与 W&A prefill speedup 的 arithmetic intensity 条件有何差别？
5. benchmark 应怎样同时覆盖 perplexity、reasoning、long-context、throughput、P99 latency 与 energy？
6. 如何把 Figure 8 扩展成 VLA quantization taxonomy：vision encoder、language backbone、action expert/head、history/KV、proprioception 各放在哪里？

## 9. Weekly meeting card

- **Role**：comprehensive lookup/reference，不是 new algorithm。
- **Core taxonomy**：Basics → systems → training/PEFT → QAT/PTQ mechanisms → toolkits。
- **Best figure**：Figure 8 PTQ algorithm map。
- **Best warning**：framework supports an algorithm 不等于 every model/hardware/bit setting 都能 deploy。
- **Biggest freshness gap**：2026 methods 与 current runtime support 需要另做 live update。

## 10. Status & evidence boundary

- **Status**：peer-reviewed *Neural Networks* article；local full text 为 arXiv v3 author manuscript。
- **Source claim**：paper structure、figures/tables 与 method taxonomy 来自 local full text；venue metadata 与 reference count 来自 DOI/Crossref record。
- **My interpretation**：把它定位为 comprehensive reference，并与 JCST 2026 latest snapshot 配对。
- **Open question**：2026-current quantization survey 是否会出现更系统的 reproducible review；目前仍需 primary-paper + live systems documentation 补足。

