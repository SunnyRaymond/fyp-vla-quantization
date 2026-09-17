# 08 — “Give Me BF16 or Give Me Death”?（部署分支）

## 基本信息

- 完整标题：*“Give Me BF16 or Give Me Death”? Accuracy-Performance Trade-Offs in LLM Quantization*
- 作者：Eldar Kurtić、Alexandre Marques、Shubhra Pandit、Mark Kurtz、Dan Alistarh
- 发表会议：ACL 2025
- 本地来源：[官方 ACL Anthology PDF](paper-acl-2025.pdf)
- 在本阅读路线中的定位：empirical deployment audit

## 一句话要点

在多个 Llama 3.1 model sizes、benchmarks、vLLM，以及 NVIDIA A6000/A100/H100 hardware 上，经过充分调优的 FP8 和 INT8 所保留的 accuracy 远高于较弱 baselines 给人的印象；W4A16-INT 通常最适合 synchronous latency/cost，W8A8 则通常最适合 high-throughput continuous batching。

## 为什么这篇论文属于 side branch

它并没有提出一种占主导地位的新 quantizer。它的贡献在于 comparison discipline：format、calibration、clipping、model size、GPU count、prefill/decode ratio、synchronous/asynchronous serving、latency、throughput 和 cost 会共同决定最佳 deployment 方案。

## 实验脉络

- W8A8-FP：symmetric per-output-channel FP8 weights，搭配 dynamic per-token activation quantization；不使用 calibration data。
- W8A8-INT：GPTQ weights，搭配 dynamic per-token INT8 activations；对于较难处理的 Llama 3.1 70B case，额外使用 SmoothQuant。
- W4A16-INT：4-bit GPTQ weights、16-bit activations、MSE-optimal clipping、group size 128，以及 OpenPlatypus calibration。
- Models：Llama 3.1 Instruct 8B、70B 和 405B，另包含 reasoning-model analysis。
- Hardware 与 serving：在 NVIDIA A6000、A100 和 H100 上使用 vLLM，测试 synchronous 和 asynchronous workloads。

## 重点检查的证据

- 在主要 academic benchmarks 上，8-bit formats 平均恢复约 99.75% 的 BF16 performance，W4A16-INT 则约为 99.36%。
- W4A16-INT GPTQ 与 AWQ 在 academic averages 上接近，但论文报告 tuned GPTQ 在 Arena-Hard、HumanEval 和 MBPP 上有更明显的优势。
- 对 synchronous deployment，论文报告 W4A16-INT 在 8B/70B 上将 cost per query 降低约 2–3×，latency 降低约 1.5–2.5×；在 405B 上则报告 5–7× cost reduction，并且所需 GPUs 更少。
- 对 asynchronous continuous batching，W8A8-FP 或 W8A8-INT 通常提供最高 throughput，但 crossover 取决于 model、hardware 和 workload。
- 较小的 quantized models 表现出更高的 output variability，并在困难的 reasoning evaluations 上出现更大的 relative drops。

## 局限与证据边界

- 该 study 聚焦 weight 和 activation quantization，而不是完整的 KV-cache、embedding 或 language-model-head audit。
- 研究集中于 Llama 3.1、部分 reasoning models、NVIDIA hardware、vLLM，以及当时可用的 kernel support。
- On-demand GPU cost estimates 使用特定 provider 的 price snapshot；价格可能发生变化。
- 当 BF16 baseline 接近 random chance 时，“accuracy recovery” 可能不稳定。
- Multilingual 和其他 specialized workloads 未得到全面评估。
- Power 与 energy 没有被直接测量，因此 cost-efficiency 并不等同于 energy-efficiency result。

## 对你的 FYP 有何意义

读完各 method papers 后，可以用这篇论文作为 reporting template。你的 deployment table 应包含 model 和 format、calibration、kernel、GPU count、peak memory、prefill length、decode length、batch/concurrency、time-to-first-token、inter-token latency、throughput、accuracy/task success，以及可获得时的 power。

## 阅读路线

### 30 分钟

阅读 Abstract、Section 3.2、Table 1、Tables 5–6、Figures 2–4 和 Limitations。

### 100 分钟

选定一个 model size，追踪为何推荐的 format 会在 synchronous low-latency case 与 asynchronous high-throughput case 之间变化，并检查 hardware count 是否保持匹配。

## 阅读问题——阅读后自行回答

1. 为什么 W4A16 能改善 decode latency，而 W8A8 在 compute-bound prefill 中更有优势？
2. 哪些 tuning choices 使该 GPTQ comparison 比 default-configuration comparison 更有说服力？
3. 为什么 accuracy recovery 可能超过 100%？这个 ratio 在什么情况下会产生误导？
4. GPU count 和 inter-GPU communication 如何影响 synchronous result？
5. 哪些结论在换用不同 serving engine 或 accelerator 后需要重新检验？
6. 在声称 energy consumption 更低之前，还需要哪些 measurements？

## 讨论速记

- 需要解释的核心论点：最佳 numerical format 取决于 workload shape 和 serving mode。
- 需要解释的图：latency-throughput crossover。
- 需要引用的结果：Tables 5–6，而非单一 model-quality number。
- 应主动说明的局限：cost、power、latency、throughput 与 accuracy 相互关联，但不能互换。

