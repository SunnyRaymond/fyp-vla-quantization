# HBVLA: Pushing 1-Bit Post-Training Quantization for Vision-Language-Action Models

> **在本指南中的角色：** 李老师新增指定阅读，active core `#20`；当前核心里最直接的 `VLA × 1-bit PTQ × closed-loop evaluation` 论文。  
> **本地论文：** [paper-arxiv-v2.pdf](paper-arxiv-v2.pdf)  
> **建议先修：** [OpenVLA](../001-openvla/README.md)、[BitVLA](../009-bitvla/README.md)、[OPTQ/GPTQ](../018-gptq-optq/README.md)，以及 `binary quantization`、`Hessian-aware PTQ`、`Haar transform`。  
> **阅读状态：** `verified-full-text`；表示本地全文与来源已核对，不表示已独立复现。

## 1. Paper identity 与 version boundary

| Field | Record |
|---|---|
| Title | *HBVLA: Pushing 1-Bit Post-Training Quantization for Vision-Language-Action Models* |
| Authors | Xin Yan, Zhenglin Wan, Feiyang Ye, Xingrui Yu, Hangyu Du, Yang You, Ivor Tsang |
| Venue / status | arXiv preprint；截至 2026-09-04 未核实到正式 venue record |
| arXiv | [2602.13710](https://arxiv.org/abs/2602.13710)，v1 2026-02-14；v2 2026-08-20 |
| Code | 截至本次核对未找到可确认的 official repository |
| Local artifact | arXiv v2，9 pages，3,881,276 bytes，SHA-256 `2095e124c2146c611c474855dc826ca2fa02b331b9d736375ff706cbbd289d05` |

**名称边界。** arXiv title 与用户给出的标题写作 `HBVLA`；PDF 正文和 method label 也常写作 `HB-VLA`。本笔记用 `HBVLA` 指 paper identity，用 `HB-VLA` 指 method。

**版本边界。** 本地文件是 arXiv v2。正文多次指向 appendix、pseudocode 和额外实验，但这份 9-page v2 在 references 后结束，**没有 appendix**。因此，不能把文中“详见 appendix”当作本地可审计 evidence，也不能声称 implementation 已完整公开。

## 2. 一句话结论

`HB-VLA` 用 action-sensitive Hessian、embodiment-aware drift weighting 与 Haar-domain binary quantization，把多个 VLA backbone 压到平均约 `1.02–1.13 bits/weight`；它在模拟环境中是很强的 1-bit PTQ baseline，但 real-robot success 仍下降 `12.5–23.4 percentage points`，所以这篇更像“1-bit 已经可用但远非无损”，而不是“1-bit VLA 已被解决”。

## 3. 先把问题说清楚

通用 LLM/VLM quantization 常优化 layer reconstruction、perplexity 或静态视觉语义。VLA 的 loss surface 不同：

- action 输出进入 closed loop，小的单步误差可能在未来 observation 中累积；
- 不同 action dimensions 对 task drift 的影响不同，例如 translation、rotation、gripper error 不应等权；
- vision encoder、language model 与 action head 对 1-bit error 的敏感度明显不同；
- 只看 model size 或 open-loop error，不能证明 robot success 保持。

论文要回答的是：**不 retrain 的 1-bit weight-only PTQ，能否利用 action sensitivity 来保护对控制最重要的 weights？**

## 4. Method：四步 mental model

### 4.1 用 action loss 修正 Hessian

常规 GPTQ/OBQ 类方法用 calibration activations `X` 构造二阶近似。`HB-VLA` 进一步为 tokens/columns 引入由 action drift loss gradient 得到的重要性权重 `S`：

$$
\widetilde H = X S X^\top.
$$

直觉：同样大小的 layer reconstruction error，如果更容易改变最终 action，就应该在 quantization 中付出更高代价。

### 4.2 embodiment-aware drift weighting

论文不是把所有 action dimensions 直接平均，而是用 embodiment 的 kinematic Jacobian 构造 `ρ_j`，对更容易产生末端位姿 drift 的 dimension 加权。这里的贡献是**把 generic activation sensitivity 转成 control-oriented sensitivity**。

需要警惕：这依赖 robot embodiment 与 action representation；换 robot、换 normalization 或换 action head，权重是否仍合适需要重新验证。

### 4.3 按 saliency 分列

根据 rectified Hessian 把 columns 分成 salient 与 non-salient 两组：

- salient columns：需要更细致保护；
- non-salient columns：允许更激进的重排与 binary approximation。

这不是 mixed-bit allocation；最终仍是接近 1-bit 的 weight representation，只是不同 column group 使用不同 transform path。

### 4.4 Haar-domain group-wise binarization

- 对 non-salient group：先做 sparse orthogonal permutation，再在 Haar domain 做 group-wise 1-bit quantization；
- 对 salient residual：采用 column-wise Haar transform，尽量减少关键方向的 distortion；
- activations 仍为 `BF16`，因此它是接近 `W1A16` 的 weight-only PTQ，不是 full 1-bit compute pipeline。

```text
calibration trajectories
        │
        ├─ activation X + action-drift gradient ─> rectified Hessian
        │
        ├─ salient / non-salient columns
        │
        └─ Haar-domain binary quantization ─> ~1.02–1.13 bits/weight

inference: BF16 activations + binary/scale metadata weights
```

## 5. Experiments：主张、证据与边界

所有论文实验均报告在 `NVIDIA A800` 上完成，但 9-page PDF 没有给出足够完整的 latency protocol。

### 5.1 Simulation / benchmark results

| Model / benchmark | Full precision | HB-VLA `LM+ViT` | HB-VLA `All` | Locator | Interpretation |
|---|---:|---:|---:|---|---|
| `π0.5` / LIBERO average success | 97.1 | 92.7 | 87.9 | Table 4, p. 7 | action head 也 binarize 后额外掉点明显 |
| `OpenVLA-OFT` / LIBERO average success | 97.1 | 90.3 | 83.5 | Table 4, p. 7 | 极低 bit 对 action path 的损伤不能忽略 |
| `CogACT` / SIMPLER Visual Matching | 74.8 | 70.0 | 67.2 | Table 4, p. 7 | 保持多数性能，但不是 near-lossless |

文中 component sensitivity tables 的共同趋势是：`action head` 最敏感，`vision encoder` 相对最稳健。这支持 component-aware policy，但不自动证明同一 ranking 能迁移到所有 VLA architecture。

### 5.2 Memory

| Model | FP weight memory | HB-VLA | Reduction |
|---|---:|---:|---:|
| `π0.5` | 4.60 GB | 0.83 GB | about 82% |
| `OpenVLA-OFT` | 15.20 GB | 2.74 GB | about 82% |
| `CogACT` | 30.50 GB | 5.50 GB | about 82% |

Locator: Table 5, local PDF p. 7。这里是 **weight memory**，不能直接替代 peak runtime memory；activations、KV/cache、temporary buffers 与 robot stack 仍需另测。

### 5.3 Latency headline 不够完整

论文文字报告最高 `2.93× inference speedup`、`65.9% latency reduction`。但本地 v2 没有给出能审计这一 headline 的完整 table、per-model latency、batch/sequence/action-chunk setting、kernel details 或 timing methodology。

因此组会可说“authors report 2.93×”，不能说“表明所有 VLA 在 A800 上稳定 2.93×”，更不能外推到 `A100`、edge GPU 或 real-time control stack。

### 5.4 Mobile ALOHA real-world evidence

| Task | FP success | HB-VLA success | Absolute drop |
|---|---:|---:|---:|
| Pick-and-Place | 86.7% | 63.3% | 23.4 pp |
| Sequenced Instruction | 95.8% | 83.3% | 12.5 pp |
| Flexible Folding | 83.3% | 66.7% | 16.6 pp |

Locator: Figure 3 / real-world paragraph, local PDF pp. 6–7。Pick-and-Place 使用 30 trials，其余 task 各 24 trials。

这是论文最值得认真读的表述冲突：abstract 把 physical-world degradation 描述得很轻，但 `12.5–23.4 pp` 是实质性下降。合理结论是 **HB-VLA 是所比较 binary PTQ 中很强的方案**，不是“相对 FP 几乎不掉点”。

## 6. Ablation 中要追的两个问题

1. **Column criterion 是否真的 action-aware？** Tables 6–7 比较 column selection / Hessian variants，但 `Visual Matching`、`Variant Aggregation` 列标了 downward arrow，正文却把值解释为 quantization error 或 performance，命名与方向不够清楚。读表时不要只抄 bold number。
2. **收益来自哪一部分？** rectified Hessian、column partition、permutation 与 Haar quantization 同时变化；需要 matched ablation 才能区分 control-aware scoring 与 transform design 的贡献。

## 7. Limitations 与 reproducibility audit

### Authors 明示或可直接观察的边界

- VLA 与 tasks 数量仍有限，real-world trials 规模小；
- extreme 1-bit quantization 对 action head 尤其脆弱；
- 当前 evidence 不能覆盖不同 robots、sensors、action spaces 与 long-horizon tasks；
- local arXiv v2 缺失正文引用的 appendix。

### 我的 critique

- **没有公开可核实 code。** 无法确认 bit packing、scales、kernel fusion 与实际 latency path。
- **PTQ calibration scope 不透明。** calibration trajectories 的任务覆盖、样本量与 data leakage 风险需要更完整说明。
- **real-world confidence interval 缺失。** 24–30 trials 只能给粗略估计，且失败严重度没有分层。
- **平均 bits 不等于 hardware-native 1-bit throughput。** metadata、scales、unpack 与 BF16 activations 都会影响实际速度。
- **没有 power/energy、P50/P99 latency 或 control-frequency measurement。** weight memory 和 isolated inference headline 不足以证明 deployment benefit。
- **缺失 appendix 是当前最直接的审计阻塞。** 在新版本或 code 出现前，pseudocode/超参数不能视为已验证。

## 8. 对 FYP 的价值

这篇给你的 FYP 一个很清楚的 baseline shape：

1. `generic reconstruction-aware PTQ` 不够，需要 action-aware calibration/evaluation；
2. component sensitivity 应至少拆成 `vision encoder / language backbone / action head`；
3. 极低 bit 的主结果必须同时报告 `weight memory + actual peak memory + latency + closed-loop success`；
4. binary method 的 strongest claim 应与 matched `W2/W3/W4` baseline 比，而不只与其他 1-bit method 比；
5. 若 success 大幅下降，应该测 failure type 与 drift accumulation，而不是只给 aggregate average。

一个可执行的后续问题是：**在相同 VLA checkpoint、calibration data 和 runtime 下，action-aware Hessian 对 W2/W3/W4 是否仍有收益，还是只有进入 binary regime 才明显？**

## 9. 阅读路线

### 20-minute route

1. Abstract + Figure 1，写下 generic PTQ 为什么对 VLA 不够。
2. Section 3，只抓 `rectified Hessian → saliency partition → Haar binarization`。
3. Tables 4–5，分开 success 与 weight memory。
4. Figure 3，计算三项 real-world absolute drop。
5. 检查 PDF 末页，确认当前文件确实没有 appendix。

最低完成线：能解释“为什么它不是把 GPTQ 直接套到 VLA”，并主动说出 real-world degradation。

### 90-minute route

1. 回顾 [OPTQ/GPTQ](../018-gptq-optq/README.md) 的 `H = 2XX^T` 与 error compensation。
2. 逐式阅读 equations (3)–(7)，标出 action loss、gradient、`S` 与 `ρ_j` 的来源。
3. 重画 salient/non-salient 两条 Haar path。
4. 审计 Tables 1–3 的 component sensitivity。
5. 重算 Table 4 的 absolute/relative retention，避免只看 average。
6. 对照 Table 5：写出 weight memory 没覆盖的 runtime memory。
7. 审计 2.93× headline 所缺的 measurement details。
8. 写一张 meeting card，不预先回答 Reading Questions。

### 3-hour deep route

1. 完成 90-minute route。
2. 对照 `BitVLA`、`GPTQ`、`AWQ`、`QuaRot/SpinQuant`：training/PTQ、bit-width、activation precision、calibration objective、hardware evidence。
3. 为 `π0.5` 或 `OpenVLA-OFT` 设计 matched `FP16/W4/W2/W1` protocol。
4. 明确 success rate、action drift、latency、peak memory、energy、P99 与 failure severity 的 reporting table。
5. 查新 arXiv version / official code；若没有，保留“appendix missing / code unverified”状态。

## 10. Reading Questions（读完再答）

1. `S` 的单位和 shape 是什么？它作用在 token、feature 还是 weight column？
2. drift-weighted action loss 如何从 embodiment Jacobian 得到 `ρ_j`？
3. calibration data 是否含 evaluation tasks 或相近 trajectories？
4. 为什么 vision encoder 最稳、action head 最脆弱？这一现象是 architecture 还是 metric 导致？
5. Haar transform 比 Hadamard/rotation baseline 多保护了什么结构？
6. sparse orthogonal permutation 在 inference 中如何实现，是否产生额外 kernel overhead？
7. `1.02–1.13 bits` 包含哪些 scales、indices 与 metadata？
8. Table 4 的 `LM+ViT` 与 `All` 对实际 deployment 各代表什么？
9. 2.93× 的 denominator、warm-up、batch、sequence length 与 kernel 是什么？
10. weight memory 的 82% reduction 能否在 peak GPU memory 中复现？
11. real-world 下降中多少来自 isolated action error，多少来自 temporal compounding？
12. 与 W2/W3/W4 matched baseline 相比，W1 的 Pareto advantage 是否仍成立？
13. Tables 6–7 的 arrows、metric names 与正文解释是否一致？
14. appendix 缺失会阻止哪些关键复现步骤？
15. 哪个结论能迁移到你的 NSCC `A100`，哪个必须重新 benchmark？

## 11. Meeting card（读后填写）

- **Problem:**
- **Mechanism:**
- **Strongest evidence:**
- **Strongest limitation / unsupported leap:**
- **FYP decision or experiment:**
- **Question for 李老师:**

## 12. Evidence boundary

本地检查证明 PDF 可读、标题与 arXiv v2 身份一致；上面的数字是对论文表格/图的结构化摘录。它不等于 code reproduction、kernel profiling 或 real-robot independent replication。尤其不要把 average bit-width、weight memory reduction 与 end-to-end control speedup互相替代。

