# 07 — QMoE（专项分支）

## 基本信息

- 完整标题：*QMoE: Sub-1-Bit Compression of Trillion-Parameter Models*
- 作者：Elias Frantar、Dan Alistarh
- 发表会议：MLSys 2024
- 本地来源：[官方会议论文 PDF](paper-mlsys-2024.pdf)
- 在本阅读路线中的定位：Mixture-of-Experts compression-format 与 kernel co-design

## 一句话要点

QMoE 结合 data-dependent ternary expert quantization、可扩展的 CPU/GPU offloading、entropy-aware custom encoding 和 fused GPU decoding，将 SwitchTransformer-c2048 压缩到 0.807 bits per parameter，并使其能够在一台 commodity multi-GPU server 上运行。

## 为什么可以做到“sub-1-bit”

该 method 并不是为每个 weight 存储一个低于 1 bit 的 literal fixed-width value。Ternary quantization 会产生大量 zeros 和 low-entropy symbol stream；custom variable-length dictionary encoding 将常见 patterns 压缩到低于 raw ternary representation 的大小。论文报告的 full-model rate 也计入了 metadata 和 uncompressed dense layers。

## 方法脉络

1. 利用 sparse routing：只 quantize 占据绝大多数 parameters 的 expert layers。
2. 收集足够的 expert-specific calibration examples，同时避免把整个 model 放到一张 GPU 上。
3. 将 experts 分组以改善 GPU utilization，并用 GPTQ/OBQ-style data-dependent objective 压缩它们。
4. 使用 ternary grids 和 special-token masking 降低 validation loss。
5. 使用 GPU-decodable dictionary format 编码 low-entropy symbol sequences。
6. 将 on-the-fly decoding 与 matrix-vector multiplication fuse，使 compressed weights 无需先完整恢复。

## 重点检查的证据

- SwitchTransformer-c2048 从 BF16 的 3,142 GB 缩小到 full compressed format 的 158.6 GB，相当于 19.81× compression 或 0.807 bits per parameter。
- 其 C4 validation loss 从 BF16 的 1.18 变为默认 ternary QMoE setting 的 1.26；2-bit QMoE 报告 1.20。
- 使用默认 160k-sample setting 时，在一张 NVIDIA A6000 上完成 compression 需要 16.0 小时，另需数百 GB CPU RAM 和超过 3 TB 的 source-model disk storage。
- 在选定的 matrix shapes 上，单个 compressed kernels 最多比 BF16 baseline 快 35%。
- 论文报告 end-to-end compressed execution 与 idealized uncompressed timing estimate 的差距在 5% 以内。

## 局限与证据边界

- 主要证据与 SwitchTransformer models 绑定；当时公开的 trillion-parameter Mixture-of-Experts 选择极为有限。
- Uncompressed end-to-end reference 是 simulated 的，因为真实 BF16 c2048 需要超过 65 张 NVIDIA A6000 GPUs 或超过 130 张 NVIDIA RTX 3090 GPUs。它是 optimistic lower bound，不是 measured matched cluster run。
- Sub-1-bit storage 依赖 expert-layer dominance，以及 ternary quantization 后的 low entropy；不能自动迁移到 dense LLMs。
- Compression 虽称为 single-GPU，但 CPU RAM 和 disk requirements 仍然很高。
- Custom format 与 bespoke kernel 是 practical claim 不可分割的一部分。

## 对你的 FYP 有何意义

QMoE 很好地说明了为什么 nominal precision、entropy coding、metadata、model topology 和 execution kernel 必须一起评估。即使你自己的 model 是 dense 而非 Mixture-of-Experts，它仍能为 systems chapter 提供参考。

## 阅读路线

### 25 分钟

阅读 Abstract、Sections 3–4、Tables 4、6 和 7、runtime discussion，以及 Limitations。

### 100 分钟

追踪一个 expert 从 calibration-example collection、ternary quantization、dictionary encoding、GPU decoding 一直到 matrix-vector multiplication 的全过程，并核算每一个 stored component。

## 阅读问题——阅读后自行回答

1. 在论文报告的结果中，为什么超大 Mixture-of-Experts 比较小的 models 更能承受 extreme expert quantization？
2. Encoding 如何在保留三个 quantized values 的同时，达到低于 1 bit per parameter？
3. 哪些 model components 保持 uncompressed？它们如何计入 0.807-bit result？
4. 为什么 end-to-end BF16 comparison 必然是 idealized 的？
5. 如果只说“在一张 GPU 上不到一天”，会隐藏哪项 resource requirement？

## 讨论速记

- 需要解释的核心论点：对于一种特定的超大 Mixture-of-Experts structure，sub-1-bit full-model storage 是可行的。
- 需要绘制的流程：expert calibration → ternary PTQ → entropy encoding → fused decoding。
- 需要引用的结果：Table 6 checkpoint size 和 Table 4 validation loss。
- 应主动说明的局限：这不是 sub-1-bit arithmetic，也不是 dense-model result。

