# LLM quantization papers

本目录收录以 LLM 的 weight / activation / KV-cache quantization、低精度训练与部署为主要研究对象的阅读包，也集中存放 FP8、MX 等共享数值格式资料。2026-10-08 从 `papers/vla/` 迁入 20 个阅读包，**保留原文件夹编号**；编号沿用此前阅读路线，不要求在本目录连续。

每个阅读包的 PDF、README、supplemental、metadata 与原有辅助材料一起迁移。VLA / VLM 研究仍见 [VLA papers](../vla/README.md)，World Model 与 World Action Model 分别见 [WM papers](../wm/README.md) 和 [WAM papers](../wam/README.md)。分类按论文的主要研究对象判断；把方法用于 VLA/WAM 的可能性，不等于论文已经验证了这些模型上的效果。

## 1. Surveys

| 阅读包 | 主要用途 |
| --- | --- |
| [019. LLM quantization survey](019-llm-quantization-survey-2026/README.md) | 算法、硬件效率与研究范围的总览 |
| [020. Low-bit LLM survey](020-low-bit-llm-survey/README.md) | low-bit LLM 的基础、系统与算法参考 |

## 2. Weight / activation quantization

| 阅读包 | 主要机制与阅读重点 |
| --- | --- |
| [013. LLM.int8() / bitsandbytes](013-llm-int8-bitsandbytes/README.md) | mixed-precision decomposition 与 outlier 处理；区分算法与 library |
| [017. SmoothQuant](017-smoothquant/README.md) | 用等价缩放平衡 weight / activation 的量化难度 |
| [018. OPTQ / GPTQ](018-gptq-optq/README.md) | 二阶误差补偿的 weight-only PTQ |
| [021. AWQ](021-awq/README.md) | 用 activation 信息指导 weight quantization |
| [022. SpinQuant](022-spinquant/README.md) | 学习正交旋转，改善低比特量化；固定 arXiv v4 |
| [023. QuaRot](023-quarot/README.md) | 固定 randomized Hadamard rotations 与 W4A4KV4 inference |
| [029. Outlier Suppression](029-outlier-suppression/README.md) | Transformer activation outlier 的诊断与处理 |
| [040. AQLM](040-aqlm/README.md) | learned additive codebooks 与极低比特 weight compression |
| [041. HIGGS](041-higgs/README.md) | theory-guided data-free quantization 与 bit allocation |
| [043. QMoE](043-qmoe/README.md) | MoE compression format 与 kernel co-design |

## 3. KV cache 与 online vector quantization

| 阅读包 | 主要机制与阅读重点 |
| --- | --- |
| [108. KIVI](108-kivi/README.md) | K per-channel / V per-token、grouped/residual cache；固定 arXiv v2 |
| [025. TurboQuant](025-turboquant/README.md) | online vector quantization、rate-distortion 与 KV-cache 实验 |
| [026. RaBitQ / TurboQuant comparison](026-rabitq-turboquant-comparison/README.md) | 对照方法、理论与匹配实验，理解 baseline 争议 |

[027. 原始 RaBitQ 与 multi-bit extension](../vla/027-rabitq/README.md) 的主要研究对象是 ANN/vector search，仍保留在原有跨领域基础路线；它与这里两篇 KV-cache 相关材料的链接已接通。

## 4. 低精度训练与部署评估

| 阅读包 | 主要用途 |
| --- | --- |
| [016. NVFP4 pretraining](016-nvfp4-pretraining/README.md) | LLM pretraining 的 format / recipe 与精度边界 |
| [042. HALO](042-halo/README.md) | Hadamard-assisted low-precision fine-tuning、communication 与 activation storage |
| [044. BF16 accuracy–performance trade-offs](044-bf16-tradeoffs/README.md) | 对照量化的质量、性能与部署收益 |

## 5. 共享数值格式基础

以下两包适用于多个模型领域，在此集中管理；它们的归类表示阅读用途，而不是 LLM 专属算法。

| 阅读包 | 主要用途 |
| --- | --- |
| [014. FP8 formats](014-fp8-formats/README.md) | E4M3 / E5M2 与 OFP8 specification；区分论文和规范 |
| [015. Microscaling formats](015-microscaling-formats/README.md) | MX shared scaling 与 MXFP8 / MXFP4；区分 empirical evaluation 和 normative specification |

## 6. 推荐入口

- **Weight / activation PTQ**：LLM.int8() → Outlier Suppression → SmoothQuant → GPTQ / AWQ → QuaRot → SpinQuant。
- **KV-cache quantization**：先读 KIVI，理解 K/V 为什么区别处理，再读 TurboQuant 与 RaBitQ / TurboQuant comparison。
- **Codebook compression**：AQLM → HIGGS；MoE 专项再读 QMoE。
- **低精度训练**：先补 FP8 / MX 格式，再读 NVFP4 pretraining 与 HALO；部署取舍参考 BF16 trade-offs。

SpinQuant 和 KIVI 的正文讲解于 2026-10-08 更新，均可独立阅读，包含直觉、可计算例子、公式与变量解释、实验解读、物理页阅读路线、未作答的 Reading Questions 和空白 Meeting Card。论文报告结果与本地复现状态在各包中分开记录。
