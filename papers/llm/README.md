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
| [023. QuaRot](023-quarot/README.md) | 固定 randomized Hadamard rotations 与 W4A4KV4 inference；2026-10-09 更新详细中文教学讲解 |
| [109. PrismQuant](109-prismquant/README.md) | Quantizer-aware null-space alignment、闭式谱最优性与 compact-WY；固定 arXiv v1，区分质量协议与独立部署测试 |
| [029. Outlier Suppression](029-outlier-suppression/README.md) | Transformer activation outlier 的诊断与处理 |
| [040. AQLM](040-aqlm/README.md) | learned additive codebooks 与极低比特 weight compression |
| [041. HIGGS](041-higgs/README.md) | theory-guided data-free quantization 与 bit allocation |
| [043. QMoE](043-qmoe/README.md) | MoE compression format 与 kernel co-design |
| [110. GPTVQ](110-gptvq/README.md) | data-aware VQ、GPTQ 式二阶补偿、小码本与移动 CPU 推理；固定 arXiv v2 |
| [111. QTIP](111-qtip/README.md) | trellis-coded quantization、bitshift 并行解码与 256 维表示；固定 arXiv v4 |
| [112. QuIP#](112-quip-sharp/README.md) | RHT incoherence、E8P 八维格点码本、BlockLDLQ 与恢复微调；固定 arXiv v2 |
| [113. GuidedQuant](113-guidedquant/README.md) | 最终损失梯度指导的局部目标、分组 Fisher 近似与 LNQ；固定 arXiv v4 |

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

- **Weight / activation PTQ**：LLM.int8() → Outlier Suppression → SmoothQuant → GPTQ / AWQ → QuaRot → SpinQuant → PrismQuant。
- **KV-cache quantization**：先读 KIVI，理解 K/V 为什么区别处理，再读 TurboQuant 与 RaBitQ / TurboQuant comparison。
- **Codebook compression**：先用 GPTVQ 理解 learned VQ 与二阶补偿，再读 QuIP# → QTIP → GuidedQuant；AQLM 补充 additive codebooks，HIGGS 补充 theory-guided/data-free 路线。四篇联读入口见下节；MoE 专项再读 QMoE。
- **低精度训练**：先补 FP8 / MX 格式，再读 NVFP4 pretraining 与 HALO；部署取舍参考 BF16 trade-offs。

SpinQuant 和 KIVI 的正文讲解于 2026-10-08 更新，均可独立阅读，包含直觉、可计算例子、公式与变量解释、实验解读、物理页阅读路线、未作答的 Reading Questions 和空白 Meeting Card。论文报告结果与本地复现状态在各包中分开记录。

QuaRot 的正文讲解于 2026-10-09 更新，补充完整 INT4 算例与反例、RMSNorm/residual/FFN/attention 的等价变换推导、原论文图示，以及 LLaMA-3 质量下降和小 batch cache slowdown 的边界解读，同样包含阅读路线、Reading Questions 与空白 Meeting Card。

PrismQuant #109 于 2026-10-11 加入，保存 arXiv:2609.32429v1 的 38 页 PDF、两张原论文图与详细中文教学讲解。阅读重点是 grouped quantizer 的 range-null subspace、能量对齐的最优性范围，以及 random-weight GPU deployment study 与 pretrained-model 精度结果的不同证据边界。

## 7. GPTVQ / QuIP# / QTIP / GuidedQuant 联读指南

这四包于 **2026-10-28** 整理。每包均有固定版本的本地 PDF、详细中文 README、一张原论文方法图及来源说明、官方代码链接、版本 metadata、Reading Questions 和空白 Meeting Card。README 按直觉、手算例子、公式与变量、算法、结果及限制展开，页码均指本地 PDF 的物理页码。

### 7.1 从哪里开始？

| 顺序 | 本地讲解 | 固定版本 / 页数 | 用一句话描述问题 |
|---:|---|---|---|
| 1 | [GPTVQ：小码本与二阶补偿](110-gptvq/README.md) | 2402.15319v2 / 21 页 | 怎样学习 VQ 码本，并使压缩适合移动 CPU 解码？ |
| 2 | [QuIP#：旋转与格点](112-quip-sharp/README.md) | 2402.04396v2 / 27 页 | 怎样将分布整形与可高效解码的八维码本结合？ |
| 3 | [QTIP：用路径表示高维向量](111-qtip/README.md) | 2406.11235v4 / 25 页 | 怎样获得高维量化收益，又避开显式巨型码本？ |
| 4 | [GuidedQuant：最终损失指导](113-guidedquant/README.md) | 2505.07004v4 / 27 页 | 量化误差应该怎样按最终预测的重要性加权？ |

初读可给每篇 25–30 分钟，只走各 README 的快速路线；深入阅读时再分别安排 60–90 分钟。若对 Hessian/量化补偿不熟，先补 [GPTQ](018-gptq-optq/README.md)；若对 Hadamard 不熟，QuIP# README 中有二维手算例子。

GPTVQ v2 的 PDF 正文已扩展移动 CPU 部署内容，不能仅根据较旧网页摘要理解。QTIP v4 与 GuidedQuant v4 包含后续修订，本包使用所下载的固定版本，不把不同版本、不同论文引用的旧数字自动混合。

### 7.2 用四个维度把方法分开

| 方法 | 表示：权重存成什么 | 目标：什么误差重要 | 求解：怎样找到量化权重 | 系统：怎样执行 |
|---|---|---|---|---|
| GPTVQ | learned codebook + 向量索引 | 校准输入下的层输出误差 | data-aware EM + GPTQ 式补偿 | 小 LUT，按块解码到原生计算类型；新版有移动 CPU 实测 |
| QuIP# | RHT 变换后的 E8P/RVQ codes | 层输出误差；再通过微调恢复 | BlockLDLQ + 可选/完整 FT 路线 | 利用 E8 对称性，小表和位操作解码；专用 GPU kernel |
| QTIP | trellis 路径 bits + 状态解码规则/小表 | BlockLDLQ 框架中的局部误差 | Viterbi 选路径 + 块间误差反馈；HYB 可微调 | bitshift 窗口并行解码，1MAD/3INST/HYB 各有不同实现 |
| GuidedQuant | 随所配 LNQ/QTIP/SpinQuant 改变 | 最终损失梯度加权的层输出误差 | 分组二阶矩阵 + 原量化求解器；另提出 LNQ | 主要改变离线目标；在线速度由最终格式和 kernel 决定 |

读完应能区分：**改目标、改表示、改求解算法、改执行实现**。这些方向可以组合，但不是换一个名称就自动获得另一篇的全部收益。

### 7.3 几个必须对齐的结果口径

- **QuIP# 8.22 与 6.19**：Llama-2-7B、2-bit、WikiText2 context 4096，分别对应有 E8 无 FT、完整 FT 路线。旧引用可能使用不同版本。
- **QTIP 6.82 与 5.86**：相同模型/位宽/上下文的纯计算码无 FT 与 HYB 微调路线，不能互换。
- **GuidedQuant 的 6.82 → 6.11**：改进的是相应纯计算 QTIP 路线的目标，不是宣称超过 HYB+FT 的 5.86。
- **Context 2048 与 4096**：PPL 会随评估上下文和协议变化；不能只按模型名称合并。
- **压缩与速度**：索引位数、实际模型 bytes、峰值内存、解码开销和端到端吞吐分别报告。手机 CPU、RTX 4090、RTX 6000 Ada，以及不同框架/矩阵融合设置的结果不能直接排名。

上述数字与条件分别见各包标明的原论文表格。这些材料是文献阅读整理，**没有在本项目运行四篇方法的量化或推理复现**，也不构成 WAM/机器人闭环验证。

### 7.4 面向当前 WAM 研究的阅读重点

GPTVQ、QuIP#、QTIP 最适合帮助拆解“权重表示 → 搬运 → 解码 → 矩阵运算”的系统成本；GuidedQuant 最适合帮助讨论“局部重建误差是否保护了最终有用信息”。各 README 的 FYP 小节都把论文已验证的 LLM 结果与迁移启发分开，并解释为何 task loss、校准分布、实际矩阵形状和闭环任务仍需重新验证。
