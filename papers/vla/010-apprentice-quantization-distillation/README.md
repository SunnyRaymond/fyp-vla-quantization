# Apprentice: Using Knowledge Distillation Techniques to Improve Low-Precision Network Accuracy

> **Reading-list role**: BitVLA companion — direct conceptual predecessor of Quantize-then-Distill  
> **Verification**: `verified-full-text`; local file is arXiv v1  
> **Recommended effort**: **Targeted read**，重点读 Scheme-C 与 BitVLA 的差别

> **Local full text**: [paper.pdf](paper.pdf)

## 1. Paper identity

| Field | Value |
|---|---|
| Authors | Asit Mishra, Debbie Marr |
| Year / version | arXiv v1, 2017-11-15 |
| Venue / status | ICLR 2018 conference paper |
| Primary source | [arXiv:1711.05852](https://arxiv.org/abs/1711.05852) · [OpenReview](https://openreview.net/forum?id=B1ae1lZRb) |
| Local-version note | OpenReview PDF blocked unattended acquisition；local copy uses the readable arXiv v1 full text |

## 2. One-sentence takeaway

Apprentice 早在 BitVLA 之前就系统研究了 `full-precision teacher → low-precision student`，其中 Scheme-C 先给 student 一个 full-precision initialization，再降低 precision 并在 Knowledge Distillation 下 fine-tune；因此 Quantize-then-Distill 的 broad idea 不是 BitVLA 首创，但 BitVLA 的 VLA-specific module placement、loss 与 W1.58A8 recipe 是新的 instantiation。

## 3. Provenance verdict: BitVLA 发明了什么？

### 3.1 结论

- **不是 broad concept 的发明者**：Apprentice 在 2017 年已经把 low-precision quantization 与 Knowledge Distillation 结合，并明确给出 “lower precision, then fine-tune with a full-precision teacher” 的 Scheme-C。
- **是具体 named recipe 的提出者**：目前核验到的 primary sources 中，`Quantize-then-Distill` 这个名称及其 BitVLA implementation 由 BitVLA 提出；BitVLA 没有把 Apprentice 作为该 stage 的 direct citation。
- **不能把 Apprentice 叫作 BitVLA method 的 exact original paper**：两者的 architecture、target module、supervision 与 evaluation 均不同。更准确的定位是 **direct conceptual predecessor / prior art**。

### 3.2 Scheme-C 与 BitVLA 对照

| Dimension | Apprentice Scheme-C | BitVLA Quantize-then-Distill |
|---|---|---|
| Target | ImageNet ResNet classifier | SigLIP-L vision encoder inside a VLM/VLA |
| Student initialization | full-precision trained student weights | full-precision vision-encoder counterpart |
| Teacher | trained full-precision classifier | frozen BF16 vision encoder |
| Student precision | ternary weights or W4A8 variants | W1.58A8 |
| Distillation target | primarily teacher logits / class predictions | per-layer hidden-state MSE alignment |
| Task supervision | hard class labels + distillation | answer-token Language Modeling loss + representation alignment |
| Trainable modules | low-precision student | only student vision encoder；BitNet backbone and connector frozen |
| Downstream evidence | ImageNet / CIFAR classification | VQA preservation + robot policy adaptation |

## 4. Background and problem

- Full-precision DNN deployment is memory- and compute-heavy, especially at small inference batch sizes where parameter storage dominates.
- Quantization reduces arithmetic and storage cost；Knowledge Distillation lets a student learn from a stronger teacher distribution instead of only hard labels.
- The paper asks whether combining these two compression mechanisms can recover the accuracy lost by ternary or 4-bit inference.

## 5. Method

### 5.1 Distillation objective

For input `x`, teacher prediction `p^T`, student prediction `p^A`, teacher logits `z^T`, and hard label `y`, the paper writes:

`L = α H(y, p^T) + β H(y, p^A) + γ H(z^T, p^A)`

The three terms respectively train the teacher, train the student from hard labels, and make the student match teacher knowledge. The reported setup uses Cross-Entropy and `α=1, β=0.5, γ=0.5`；the exact active terms depend on the training scheme.

### 5.2 Three schemes

1. **Scheme-A — joint training**：full-precision teacher 与 low-precision student 一起从 training process 中学习。
2. **Scheme-B — fixed trained teacher**：teacher 已训练；low-precision student 从 scratch 学习，只更新 student。Paper reports convergence about 10–20% fewer epochs than Scheme-A in tested settings。
3. **Scheme-C — low-precision fine-tuning**：teacher 与 student 先有 full-precision weights；降低 student weights/activations precision 后，以较低 learning rate 在 teacher supervision 下 fine-tune。

Scheme-C 最接近 BitVLA：它承认 quantization 后的 student 需要从好的 initialization 出发，再用 teacher 约束 fine-tuning trajectory。

## 6. Main results

| Claim | Evidence | Locator | Caveat |
|---|---|---|---|
| Scheme-C 优于 Scheme-A/B | ResNet-50 ternary Top-1 error `25.3 → 24.7`；W4A8 `25.5 → 25.1` | Table 4, PDF p. 10 | improvement 小；不是 Transformer/VLA evidence |
| Low-precision + Knowledge Distillation 缩小 full-precision gap | ternary ResNet-50 reaches 24.7% error vs 23.8% full-precision baseline | Sec. 5.4, PDF p. 10 | old ImageNet recipe；hardware/runtime evidence limited |
| Fixed teacher 可加快 convergence | Scheme-B students converge around epoch 80–85 vs about 105 for Scheme-A | Fig. 5, PDF p. 9 | not a matched modern large-model training study |

## 7. Key difference from BitVLA

Apprentice 主要让 student 模仿 teacher 的 **output behavior**；BitVLA 的关键是保护 pretrained vision representation geometry：

`L_BitVLA = L_LM(answer tokens) + λ · mean_l ||h_l^BF16 - h_l^W1.58A8||²`

所以 BitVLA 更接近 `quantization-aware feature distillation`。它的 novelty 不应写成“首次把 Quantization 和 Knowledge Distillation 结合”，而应写成“把 native low-bit VLM backbone、W1.58A8 vision QAT、intermediate representation alignment 与 VLA adaptation 组织成一个 end-to-end recipe”。

## 8. Limitations

- Evidence 来自 CNN classification，不直接证明 VLM semantic alignment 或 closed-loop robot control。
- Scheme-A/B/C 同时改变 initialization 与 optimization schedule，不能把全部 gain 归因于一个单独 mechanism。
- Paper 主要报告 model accuracy/model size；真实 device latency、energy 与 kernel support 证据不足。
- Distillation 依赖可访问的 full-precision teacher 与 training data；teacher inference 也增加 training cost。

## 9. How to read it

### 20-minute route

1. Sec. 4（PDF p. 4–5）：理解 teacher/student objective。
2. Sec. 5 opening（p. 5）：画出 Scheme-A/B/C。
3. Sec. 5.4 + Table 4（pp. 9–10）：只深读 Scheme-C。
4. 回到 [BitVLA](../009-bitvla/README.md) Sec. III-B：逐项比较 target module、loss、frozen modules。

### 60-minute route

1. 复算 ternary/W4A8 model-size intuition。
2. 区分 logit distillation、soft-target distillation、intermediate-feature distillation。
3. 把 Scheme-C 改写成 modern QAT pseudocode。
4. 设计 controlled baseline：同一 SigLIP initialization，比较 QAT only、output KD、intermediate KD、两者同时使用。

## 10. Reading questions

1. Scheme-C 的 gain 来自 teacher signal，还是只来自 pretrained initialization + low learning rate？
2. 对 VLM 来说，matching logits、matching hidden states 与 matching attention maps 哪个最能保护 multimodal alignment？
3. Teacher 和 student architecture 相同但 precision 不同时，Knowledge Distillation 是 compression，还是 quantization regularization？
4. BitVLA 为什么没有提供一个 `QAT without teacher` 的 robot-policy controlled ablation？

## 11. Weekly meeting card

- **Problem**：怎样减少 low-precision network 的 accuracy loss？
- **Key idea**：让 full-precision teacher 监督 ternary/4-bit student；Scheme-C 从 full-precision initialization 降 precision 后继续 distill。
- **Best evidence**：Scheme-C 将 ternary ResNet-50 Top-1 error 从 25.3% 降至 24.7%（Table 4）。
- **Biggest limitation**：CNN classification evidence 不能自动迁移到 VLA closed-loop behavior。
- **Connection to BitVLA**：broad idea predates BitVLA；BitVLA 的 contribution 是 VLM/VLA-specific W1.58A8 feature-alignment recipe。

## 12. Evidence boundary

- **Source claim**：three schemes、objective 与 ImageNet results 来自 Apprentice full text。
- **My interpretation**：Scheme-C 是 BitVLA Quantize-then-Distill 最直接的 conceptual predecessor。
- **Not established**：未发现更早使用 exact name `Quantize-then-Distill` 的 primary source；这不等于完成 exhaustive patent/prior-art search。
- **Primary links**：[arXiv](https://arxiv.org/abs/1711.05852) · [OpenReview](https://openreview.net/forum?id=B1ae1lZRb)

