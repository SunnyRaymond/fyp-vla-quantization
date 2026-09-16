# 新候选的补充近邻检索

2026-09-12，主 agent 的网页检索补充；不替代分支的 connector / full-text records。摘要证据仅用来定位威胁；是否覆盖候选要结合全文中的目标函数、优化对象和部署路径判断。

| 来源 | 已可确认的覆盖范围 | 对本轮的意义 |
|---|---|---|
| [VPTQ](https://github.com/microsoft/VPTQ)、[论文](https://arxiv.org/abs/2409.17066) | 权重向量 codebook/index 的极低比特压缩；存在公开实现 | 将 VPTQ 搬到 VLA 不足以单独成立机制创新；真实 kernel 兼容性要另证 |
| [RSAVQ](https://arxiv.org/abs/2510.01240)、[OpenReview 全文](https://openreview.net/pdf?id=8Ounc8L4F7) | Fisher/Riemannian error-direction guidance 与 channel sensitivity bit allocation | 单纯提出 sensitivity-aware、几何加权或 Fisher-weighted VQ 有直接近邻 |
| [VQ-VLA，ICCV2025 全文](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_VQ-VLA_Improving_Vision-Language-Action_Models_via_Scaling_Vector-Quantized_Action_Tokenizers_ICCV_2025_paper.pdf) | vector-quantized action tokenizer | 必须与数值权重量化区分，不能把动作离散化当成参数压缩 |
| [VQVLA / MotionVQ](https://arxiv.org/abs/2607.24148) | motion-aware 权重 VQ、centroid reuse 与定制 accelerator | 与上行 VQ-VLA 是不同论文；否定“VLA 权重 VQ 首次提出”的宽泛主张，作者相对 A100 的 accelerator speedup 不等于 A100 kernel speedup |
| [DA-PTQ](https://arxiv.org/abs/2604.11572) | representation-to-action compensation；trajectory motion error 指导 mixed precision | 不能只提“考虑动作漂移” |
| [DyQ-VLA](https://arxiv.org/abs/2603.07904) | kinematic proxies 触发动态 precision switching | 不能只提按运动阶段调整 bits |
| [Omega-QVLA](https://arxiv.org/abs/2605.28803) | composite rotation 与 denoising per-step scaling | 单纯 per-step range calibration 已有近邻 |
| [QuantVLA](https://arxiv.org/abs/2602.20309) | attention temperature matching 与 output-head balancing | 简单输出幅度修正或温度校准需直接比较 |
| [QuantWAMs](https://arxiv.org/abs/2607.28405) | shared-basis calibration、joint video-action Fisher、reachable-state schedule audit | 不能只提 joint saliency / closed-loop calibration |
| [Feedback World Model](https://arxiv.org/abs/2605.15705)、[作者页面](https://jingliangli.com/Feedback_World_Model/) | 由真实 transition residual 更新反馈状态，修正 world model，另有 action-aware guidance | 把一般 latent observer 用在量化 WM，仍需证明量化专属机制 |
| [Calibrate Where You Deploy 作者仓库](https://github.com/parastoopil/on-policy-calibration-vla) | 自报 SmolVLA 的 on-policy recalibration negative result，附 raw artifacts 与公开 audit 限制 | 可作设计风险线索；本轮未复算结果，非 peer-reviewed 普遍不可能性证据。其任务遗漏、seed clustering 与对照分布尚有公开问题，不采纳过强结论 |

检索包括 `VLA vector quantization weight compression AQLM VPTQ robot policy`、`world model quantization closed loop calibration quantized states policy 2026`、`VLA quantization temporal consistency quantization drift TQ-VLA`、`VLA quantization noise shaping temporal error correlation` 等。网页索引不是穷尽检索；最新条目可能缺失。没有检出 exact match 只能支持继续审查，不能证明新颖性。

## non-VQ 改写后的定向补检

STRC 草案的 phase-shuffle 干预存在归因问题；root 已决定放弃该具体配方并保留草案记录。新方向是固定真实量化残差的有限差分传播匹配。以下近邻直接限制其 novelty：

- [Sobolev Training，NeurIPS2017](https://arxiv.org/abs/1706.04859)：导数信息用于训练和模型蒸馏已有长期先验，不能声称首次保留 Jacobian。
- [GAD / Restoring Initial Noise Sensitivity，2026](https://arxiv.org/abs/2606.01651)、[官方代码](https://github.com/Hannah1102/GAD)：几何/噪声敏感性对齐，包含有限差分形式的方向响应匹配。因此 finite-difference alignment 本身不是新贡献。
- [QDrop，ICLR2022](https://arxiv.org/abs/2203.05740)、[全文](https://openreview.net/pdf?id=ySQH0oDyp7)：PTQ 重建中引入部分 activation quantization 与随机 drop；不能把 noisy-input calibration 当作新原则。
- [PD-Quant，CVPR2023](https://arxiv.org/abs/2212.07048)、[全文](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_PD-Quant_Post-Training_Quantization_Based_on_Prediction_Difference_Metric_CVPR_2023_paper.pdf)：全局 prediction difference 与 calibration distribution correction 已有直接先验。

新候选只能检验更窄的命题：在固定 scalar PTQ 配置与相同数据/优化预算下，用冻结量化器实际产生的 residual directions 匹配 recurrent world-model 的有限差分响应，是否比等范数通用扰动匹配、普通两步 rollout 重建和 noisy-input reconstruction 更有效。若无此增量，则停止该研究配方。主 agent 的补检为普通网页搜索与 primary sources 阅读，不伪称已由所有原始 connectors 覆盖。

## VQ 的 relational-family 定向补检

- [Relational Knowledge Distillation，CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html)：distance-wise 与 angle-wise 的样本间关系蒸馏已存在。
- [QATMA/TPSD，2026](https://arxiv.org/abs/2603.05964)：在 open-vocabulary object detection 中结合 QAT 与 text-anchored pairwise similarity transfer。因此“任务 anchor + pairwise relation + quantization”不是可单独主张的新原则。
- [Feature Affinity Assisted Quantized Distillation 官方代码](https://github.com/lzj994/FAQD)：关系/affinity 与 quantized distillation 是已有方法家族；本轮未完成其全文逐式审查。

VQ 候选必须比较相同轨迹与相同存储预算的普通 RKD-style VQ；若目标数学等价，就只能称为 world-model 场景下的受控适配假设，不能声称发明新 relational quantization objective。另有 representation/product-quantization retrieval 文献，不能仅因量化对象不同就忽视机制重叠，但也不能把 embedding PQ 自动当作 backbone weight VQ。

最后核对了 [QATMA/TPSD v1 §4.2 Eq.8–10](https://arxiv.org/html/2603.05964v1)：归一化 text/region embeddings 组成共同相似度矩阵，同时约束 anchor alignment 与 region relations。也核对了 [GAD v1 §4.1 Eq.3–4 及 finite-difference 段](https://arxiv.org/html/2606.01651v1)：通用 Gaussian direction 的响应匹配已有明确公式。两条新候选均需把这些作为方法来源与强对照，而非仅列成相关工作。

编辑校正（root，2026-09-12）：arXiv:2603.05964 正式标题为 QATMA: Quantization-Aware Training with Multimodal Alignment for Open-Vocabulary Object Detection；早期检索误标 CR-QAT。最终 VQ pilot 固定 indices、仅更新 codebook entries，正文中 codebook/index 的泛称不代表首轮更新 index。具体优化预算和 FRT action-slot/hard-W4 规格以最终 IDEA 卡为准；这些编辑未改变独立审查的 conditional 判定。
