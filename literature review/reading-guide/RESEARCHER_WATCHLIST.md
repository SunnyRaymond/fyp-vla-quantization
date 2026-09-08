# Researcher Watchlist: Optional Papers Only

> **Status**: identity-verified watchlist  
> **Counting rule**: 下面的 8 papers 全部是 **optional**，不是 Professor Li 明确点名的 core papers；不计入 core paper count。`OPTQ/GPTQ` 与 `QuaRot` 已按后续阅读需求建立 local companion README，其余 6 篇仍为 watchlist-only。  
> **Use**: 用 follow line 决定下一周补哪一篇，不要求一次读完。

## MIT Prof. Song Han

**Identity**: [Official HAN Lab profile](https://hanlab.mit.edu/songhan) verifies Song Han as Associate Professor with tenure at MIT EECS and HAN Lab PI. “MIT Prof. Song Han” 身份无实质歧义。

### Optional papers

| Priority | Paper | Venue / version | Why follow it |
|---:|---|---|---|
| 1 | [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149) | ICLR 2016; arXiv v5, 2016-02-15 | 建立 pruning → trained quantization → entropy coding 的 foundational compression pipeline。 |
| 2 | [EIE: Efficient Inference Engine on Compressed Deep Neural Network](https://research.nvidia.com/publication/2016-06_eie-efficient-inference-engine-compressed-deep-neural-network) | ISCA 2016 | 学会区分 nominal compression 与 hardware/dataflow 真正兑现的 latency/energy gain。 |
| 3 | [Once-for-All: Train One Network and Specialize it for Efficient Deployment](https://openreview.net/pdf?id=HylxE1HKwS) | ICLR 2020 | 从 compression 延伸到 hardware-aware specialization 与 heterogeneous edge deployment。 |

### Follow line

`Deep Compression [optional] → EIE [optional] → HAQ [core] → SmoothQuant [core] → AWQ [core]`

Side branch: `Once-for-All [optional]`，当项目重点转向不同 robot/edge hardware targets 时再读。

**建议追踪方向**: algorithm–hardware co-design for deployable quantization。每次看 headline compression ratio 时，都继续追 latency、memory traffic、energy、kernel support 与 target device。

## ISTA Prof. Dan Alistarh

**Identity**: [ISTA Alistarh Group](https://www.ista.ac.at/en/research/alistarh-group/) and [ISTA Research Explorer](https://research-explorer.ista.ac.at/person/4A899BFC-F248-11E8-B48F-1D18A9856A87) verify Professor **Dan-Adrian Alistarh**, ORCID `0000-0003-3650-940X`, Google Scholar ID `75q-6ZQAAAAJ`. “Dan Alistarh” 与 “Dan-Adrian Alistarh” 是同一人。

### Optional papers

| Priority | Paper | Venue | Why follow it |
|---:|---|---|---|
| 1 | [QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding](https://proceedings.neurips.cc/paper/2017/hash/6c340f25839e6acdc73414517203f5f0-Abstract.html) | NeurIPS 2017 | 建立 stochastic gradient quantization 与 communication-efficiency foundation。 |
| 2 | [Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1caf09c9f4e6b0150b06a07e77f2710c-Abstract-Conference.html) | NeurIPS 2022 | 理解 approximate-second-order one-shot compression 的共同框架。 |
| 3 | [OPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](papers/10a-gptq-optq/README.md) | ICLR 2023; local companion prepared | 把 second-order PTQ 扩展到 3/4-bit LLM weights；community 常称 **GPTQ**。 |
| 4 | [SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot](https://proceedings.mlr.press/v202/frantar23a.html) | ICML 2023 | 用 pruning 对照同一 curvature-aware lineage，帮助区分 quantization 与 sparsity。 |
| 5 | [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](papers/12a-quarot/README.md) | NeurIPS 2024; local companion prepared | rotation-based weights/activations/KV-cache 4-bit inference，与 core SpinQuant 直接相邻。 |

### Follow line

`QSGD → Optimal Brain Compression → OPTQ/GPTQ → SparseGPT → QuaRot`

Fast path for the current project: `OPTQ/GPTQ → QuaRot → SpinQuant [core]`。

**建议追踪方向**: principled compression error control。先理解 quantization noise，再读 approximate-second-order PTQ/pruning，最后比较 rotation-based outlier removal。

### Naming note

Accepted ICLR 2023 title 与 ISTA publication record 使用 **OPTQ**；official implementation repository 与大量 later literature 使用 **GPTQ**。它们不是两篇 paper。正式 note 应写 `OPTQ (commonly known as GPTQ)`。

## Optional weekly use

- 一周只选一篇 optional paper，不改变 core schedule。
- 如果组会重点是 hardware realization：选 `EIE`。
- 如果重点是 LLM weight PTQ：选 `OPTQ/GPTQ`。
- 如果重点是 SpinQuant 的 lineage：选 `QuaRot`。
- 如果重点是 distributed training：选 `QSGD`。
