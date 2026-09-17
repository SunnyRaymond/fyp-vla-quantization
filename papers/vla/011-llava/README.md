# Visual Instruction Tuning (LLaVA)

> **Reading-list role**: BitVLA companion — multimodal initialization and training-curriculum foundation  
> **Verification**: `verified-full-text`; local file is NeurIPS 2023 venue final  
> **Recommended effort**: **Core companion read**

> **Local full text**: [paper.pdf](paper.pdf)

## 1. Paper identity

| Field | Value |
|---|---|
| Title | *Visual Instruction Tuning* |
| Model name | LLaVA — Large Language and Vision Assistant |
| Authors | Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee |
| Venue | NeurIPS 2023 Oral |
| Primary source | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html) · [arXiv:2304.08485](https://arxiv.org/abs/2304.08485) |
| Project / code | [Project page](https://llava-vl.github.io/) · [Official repository](https://github.com/haotian-liu/LLaVA) |

## 2. One-sentence takeaway

LLaVA 用 language-only GPT-4 将 image captions 与 bounding boxes 改写成 158K visual instruction samples，再用两阶段训练把 frozen CLIP ViT-L/14 经 linear projector 接入 Vicuna，证明简单 architecture 加高质量 visual instruction tuning 就能产生强 multimodal assistant。

## 3. 为什么 BitVLA 要引用 LLaVA？

BitVLA 不是把 LLaVA checkpoint 直接变成 robot policy，而是复用其 **training pattern**：

1. 先冻结 vision encoder 与 language backbone，只训练 connector，使 visual tokens 对齐 language embedding space。
2. 再冻结 vision encoder，训练 language backbone + connector 做 visual instruction following。
3. 完成稳定 multimodal initialization 后，BitVLA 才做 Quantize-then-Distill 与 robotics training。

BitVLA 的实际 components/data 已经更换：`CLIP → SigLIP-L`、`Vicuna → BitNet b1.58 2B4T`、`595K CC3M → 558K LLaVA-1.5 alignment data`、`158K LLaVA-Instruct → 10M MAmmoTH-VL subset`。所以论文中 “Following LLaVA” 指 curriculum lineage，不是 identical model。

## 4. Problem

- Text-only Instruction Tuning 已能让 LLM follow instructions，但当时缺少大规模 image-language instruction data。
- 普通 image-caption pretraining 教模型描述图像，却不一定教它以 conversation、detailed description 或 complex reasoning 的方式响应 user instruction。
- 直接人工收集高质量 multimodal conversations 成本高；paper 因而探索 GPT-assisted data generation。

## 5. Method

### 5.1 GPT-assisted visual instruction data

对 COCO image，paper 不把 raw pixels 传给 GPT-4，而是用两类 textual proxy：

- captions：描述 scene semantics；
- bounding boxes：提供 object identities 与 approximate spatial locations。

GPT-4 基于这些 context 生成三类 data：58K conversations、23K detailed descriptions、77K complex-reasoning samples，总计 158K。

### 5.2 Architecture

`image X_v → frozen CLIP ViT-L/14 g(·) → visual features Z_v → trainable linear projector W → visual tokens H_v → Vicuna decoder + text tokens → answer tokens`

核心映射为：

`H_v = W · Z_v`

visual tokens 被投影到与 word embeddings 相同的 dimensionality，然后与 instruction/history tokens 一起进入 autoregressive language decoder。

### 5.3 Two-stage training

**Stage 1 — Feature Alignment**

- Data：filtered CC3M 595K image-caption pairs。
- Frozen：CLIP vision encoder + Vicuna。
- Trainable：linear projector `W` only。
- 目标：让 image features 成为 language model 可消费的 visual tokens。

**Stage 2 — Visual Instruction Tuning**

- Data：LLaVA-Instruct-158K；另有 ScienceQA-specific setting。
- Frozen：vision encoder。
- Trainable：projector + Vicuna。
- Objective：autoregressive next-token likelihood，但 **loss 只落在 Assistant answer/stop tokens**；system message、Human instruction 与 image tokens 只提供 condition。

这与 BitVLA Quantize-then-Distill 中 `L_LM` only on answer tokens 的设计直接同源。

## 6. Key innovation

- LLaVA 的历史贡献不只是 “CLIP + LLM”，而是把 **visual instruction data generation、simple connector、two-stage alignment/instruction tuning、open evaluation** 组成一个可复用 recipe。
- 它证明 data format 与 supervision interface 可以比复杂 connector 更关键。
- 后续大量 VLM/VLA work 所说的 “LLaVA-style training” 通常就是 connector alignment → instruction tuning 这条路线。

## 7. Experiments and main results

| Claim | Evidence | Locator | Caveat |
|---|---|---|---|
| Instruction tuning 对 visual chat 至关重要 | LLaVA-Bench COCO overall `21.5 → 85.1` relative score；no instruction tuning vs full data | Table 4, PDF p. 7 | evaluator 是 text-only GPT-4；benchmark small |
| Data diversity matters | conversation-only 73.8；all three response types 85.1 | Table 4, PDF p. 7 | data quantity 与 type 同时变化 |
| ScienceQA performance strong | LLaVA 90.92%；LLaVA + GPT-4 judge 92.53% | Table 7, PDF p. 9 | 92.53% 不是 LLaVA-alone；包含 GPT-4 ensemble/judge |
| Feature-alignment pretraining helps | skip Stage 1: `90.92 → 85.81` on ScienceQA | Table 8, PDF p. 9 | task-specific setting；不等于 universal causal proof |

## 8. Limitations

- GPT-4 data generation 没看到 raw image，只看到 captions/boxes；proxy 中遗漏或错误的信息会进入 supervision。
- LLaVA-Bench 的 GPT-4 judge 不是 independent human evaluation，且 benchmark size 很小。
- Model 继承 CLIP/Vicuna hallucination 与 bias；paper 明确讨论 visual hallucination risk。
- Original LLaVA 是 visual assistant，不输出 robot actions，也没有 temporal control 或 closed-loop evaluation。
- LLaVA 与后续 LLaVA-1.5/LLaVA-NeXT 不应混为一个固定 architecture/version。

## 9. Why it matters for BitVLA

- 它解释 BitVLA 为什么先获得 stable multimodal model，再量化 vision encoder：如果 connector/alignment 还没建立，quantization error 与 modality-alignment error 会混在一起。
- Answer-token-only loss 让 image/instruction 成为 context，而不会要求模型重建 input；这对 VLM instruction tuning 更自然。
- BitVLA 的 Quantize-then-Distill 进一步说明：已有 LLaVA-style alignment 在 W1.58A8 perturbation 后可能漂移，因此需要 frozen BF16 teacher 做 intermediate feature anchoring。

## 10. How to read it

### 20-minute route

1. Sec. 3 + Table 1（PDF pp. 3–4）：GPT-4 到底看到什么。
2. Fig. 1 + Sec. 4.1（p. 4）：CLIP → linear projector → Vicuna。
3. Sec. 4.2（p. 5）：two-stage training 与 answer-token mask。
4. Tables 4/7/8（pp. 7–9）：区分 LLaVA-alone 与 LLaVA+GPT-4。

### 60–90-minute route

1. 手写一个 multimodal conversation sequence，标出哪些 positions 计算 loss。
2. 比较 linear projector、Flamingo gated cross-attention、BLIP-2 Q-Former 的 capacity/cost。
3. 分析 captions/boxes proxy 会造成哪类 hallucination。
4. 回到 BitVLA，逐模块标注 LLaVA lineage 与 BitVLA-specific change。

## 11. Reading questions

1. 为什么 Stage 1 只训练 projector，而 Stage 2 才更新 LLM？
2. Answer-token-only loss 如何避免 model 在 user prompt 上浪费 capacity？
3. GPT-4 没看到 raw image 时，生成的 “complex reasoning” 有多少真正 grounded in pixels？
4. BitVLA 若在 Quantize-then-Distill 中只用 feature MSE、不保留 answer-token loss，会发生什么？

## 12. Weekly meeting card

- **Problem**：怎样低成本构造 visual instruction data 并把 vision encoder 接入 instruction-following LLM？
- **Key idea**：GPT-4 synthetic instructions + CLIP-to-Vicuna projector + two-stage alignment/tuning。
- **Best evidence**：full instruction data 将 LLaVA-Bench COCO relative score 从 21.5 提到 85.1（Table 4）。
- **Biggest limitation**：synthetic data 与 GPT-4 evaluation 都依赖 text proxy；original model 不是 robot controller。
- **Connection to BitVLA**：BitVLA 复用 curriculum，替换 backbone/data，并在之后加入 W1.58A8 distillation 与 robotics training。

## 13. Evidence boundary

- **Source claims**：architecture、data counts、training stages、loss masking 与 results 来自 NeurIPS final full text。
- **My interpretation**：LLaVA 对 BitVLA 的主要影响是 training interface，而非 robot-policy architecture。
- **Primary links**：[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html) · [arXiv](https://arxiv.org/abs/2304.08485) · [Project](https://llava-vl.github.io/)

