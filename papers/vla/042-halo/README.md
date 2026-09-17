# 06 — HALO

## 基本信息

- 完整标题：*HALO: Hadamard-Assisted Lower-Precision Optimization for LLMs*
- 作者：Saleh Ashkboos、Mahdi Nikdan、Soroush Tabesh、Roberto L. Castro、Torsten Hoefler、Dan Alistarh
- 发表会议：NeurIPS 2025
- 本地来源：[官方会议论文 PDF](paper-neurips-2025.pdf)
- 在本阅读路线中的定位：low-precision fine-tuning、communication 与 activation storage

## 一句话要点

HALO 在 forward 和 backward matrix multiplications 的不同侧放置 Hadamard transforms，使 weights、activations 和 error gradients 能够使用 INT8 或 FP6 computation；同时，HQ-FSDP 降低 communication，quantized activation storage 降低 memory。

## 为什么 training 比 inference 更难

Inference 主要 quantize forward tensors；fine-tuning 还必须保持 weight gradients 和 propagated input gradients。Outliers 在 activations 与 output gradients 中的表现不同，因此只采用一种 rotation placement 无法保护每一次 multiplication。

## 方法脉络

1. 分别在 forward 和 backward multiplications 中应用 quantization，并测量其 sensitivity。
2. 使用 right-side Hadamard transforms 分散 weights 和 activations 中的 outlier features。
3. 使用 left-side Hadamard transforms，在 backward computation 中保护 output-gradient dimensions。
4. 将 HALO-1 定义为 lower-overhead 的 accuracy/performance trade-off，适合 FP6 等 formats。
5. 将 HALO-2 定义为更强的 forward-and-backward protection，适合 dynamic range 更窄的 INT8。
6. 把 quantized weight all-gather 集成进 HQ-FSDP，并存储 quantized activations 供 backward reuse。
7. 在 HALOPEFT 中让 LoRA 的小型 low-rank operations 保持 high precision，同时 quantize 大部分 base-model computation。

## 重点检查的证据

- 在论文报告的 Llama 3 8B single-epoch fine-tuning tasks 上，INT8 HALO-2 的 accuracy 与 BF16 相差约 1% 以内；未受保护的 INT8 HALO-0 平均下降约 40%。
- FP6 HALO-1 将 GSM8K result 从 prior baselines 的 60 分出头提升到 66.5，但仍低于 BF16 的约 69.3。
- HQ-FSDP 在所测的 three-block multi-GPU setup 上报告 1.37–1.43× speedup。
- Llama 3 8B end-to-end full fine-tuning 在 4 或 8 张 NVIDIA RTX 4090 GPUs 上达到 1.35–1.41× speedup，具体取决于 precision 和 batch size。
- HALOPEFT 的 INT8 和 FP8 results 均处于 LoRA baseline 所报告的 standard deviation 范围内；FP6 在 GSM8K 上约下降 2%。

## 局限与证据边界

- 主要 full fine-tuning results 集中在 Llama 3 8B 和 NVIDIA RTX 4090；尚未证明能够移植到其他 accelerators。
- Hadamard transforms、transposes 以及 quantize/dequantize steps 都会产生 overhead，尤其是在 small batch sizes 下。
- 测试 hardware 原生不支持 FP6 matrix multiplication；部分 FP6 speed estimates 使用 FP8 compute 作为 lower-bound proxy。
- From-scratch experiments 并非始终稳定：Appendix 报告 TinyLlama 1.1B pre-training 中 INT8 和 FP4 divergence；FP8 与 BF16 匹配，而 FP6 需要 HALO-2。
- Main paper 没有提供完整的 cross-hardware energy 或 power study。

## 对你的 FYP 有何意义

当 deployment target 包括 training 而不仅是 model storage 时，HALO 是最值得阅读的论文。Reproduction 应测量 training tokens/s、peak memory、communication volume、convergence/accuracy、batch size、sequence length 和 power，而不是只报告 kernel microbenchmarks。

## 阅读路线

### 30 分钟

阅读 Abstract、Figure 1、HALO level table、Tables 1–2、Conclusion 和 Appendix A.14。

### 120 分钟

推导 linear layer 在 forward 与 backward passes 中的三个 matrix multiplications；标出 HALO-0、HALO-1 和 HALO-2 各自使用的 tensor、quantizer 与 left/right rotation。

## 阅读问题——阅读后自行回答

1. 为什么 right-side transform 能帮助 activations，却不能完全解决 output-gradient outliers？
2. 哪个 matrix multiplication 产生 input gradient？为什么其 error 尤其危险？
3. HALO-1 与 HALO-2 之间有怎样的 accuracy/performance trade-off？
4. HQ-FSDP 如何避免传输 high-precision weights？
5. 哪些 speed claims 属于 end-to-end，哪些来自 layer 或 three-block microbenchmarks？
6. 哪些证据说明目前还不能声称 HALO 已经解决 low-precision pre-training？

## 讨论速记

- 需要解释的核心论点：rotation placement 必须与 forward/backward tensor geometry 对应。
- 需要绘制的示意图：包含 left/right transforms 的 forward、input-gradient 和 weight-gradient matrix multiplications。
- 需要引用的结果：Table 2 end-to-end speedup 与 accuracy table。
- 应主动说明的局限：full fine-tuning、PEFT 与 from-scratch pre-training 的 evidence strength 不同。

