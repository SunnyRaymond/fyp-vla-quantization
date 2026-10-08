# 28. DSAQuant 阅读指南

本目录固定为 arXiv **2609.04031v1**（2026-09-03），PDF 共 24 页。先读本地原文：[paper-arxiv-v1.pdf](paper-arxiv-v1.pdf)。

官方来源：[arXiv v1](https://arxiv.org/abs/2609.04031v1) · [论文 HTML](https://arxiv.org/html/2609.04031v1) · [robbyant-research/DSAQuant](https://github.com/robbyant-research/DSAQuant) · [公开 packed checkpoints](https://huggingface.co/Robbyant-Research/DSAQuant)

## 先抓住机制

DSAQuant 研究的是 **video-generation QAT**。它把 denoising 分成不同阶段：训练时用 Denoising-Stage Oriented Supervision，在早期保留 teacher distillation、后期逐渐提高 target loss；推理时用 Denoising-Stage Gated Guidance，在最后若干步关闭 classifier-free guidance（CFG），避免 CFG 放大量化后的 conditional/unconditional branch mismatch。

CFG-drop 是 inference-time sampling heuristic；它不是 QAT 独有的组件。作者也在 BF16 control 中测试了 CFG-drop，并指出 low-bit 下受益更明显。整套方法与实验均针对 video diffusion models，不构成 **WAM control QAT 已验证** 的证据。

## 分时阅读路线

| 用时 | 阅读位置（均为本地 PDF 页码） | 读完留下什么 |
|---|---|---|
| 20 分钟 | Abstract 与 Sec. 1，pp. 1–2；Fig. 1，p. 3；Fig. 2，p. 4；Table 1，p. 6 | 区分量化早期/晚期 denoising 的影响；分别说出两个 proposed components 和一个 VBench 结果 |
| 90 分钟 | Sec. 3.2，pp. 2–4；Secs. 4.1–4.2，pp. 4–5；Tables 1–3，pp. 6–7；Table 4，p. 8 | 推出 early/late loss 变化与 CFG 误差放大的逻辑；确认 loss、sampling heuristic 与指标各自对应哪个 component |
| 180 分钟 | Secs. 5.1–5.4，pp. 6–9；Appendices B–D，pp. 13–16；Appendices E–F，pp. 16–17；Tables 6–9，pp. 9、16 | 核对 baseline、训练数据、H20 GPU-days、推理硬件和消融；检查数据合成成本未单独列账的限制；完成下方 Meeting Card |

## 机制与阅读定位

- Fig. 1（p. 3）先做 stage-wise quantization 对照：只量化 early steps 与只量化 late steps，观察结构/动作和细节质量的差异。
- Sec. 3.2（pp. 2–4）分析阶段差异；Sec. 4.1（pp. 4–5）给出 mixed loss：early steps 倚重 distillation，后期转向 diffusion/flow target；Sec. 4.2（p. 5）推导 CFG 对两个分支量化误差差值的放大。
- Fig. 2（p. 4）是全方法图。Table 4 与 Fig. 4（p. 8）拆开检查 mixed loss 和 CFG-drop；Appendix D、Table 8（pp. 15–16）对 CFG-drop 做 BF16 与 quantized controls，避免把通用 sampling 收益全部算作量化特有效应。
- Appendix G（pp. 17–19）说明 timestep-conditioned linear layers 如何通过 LUT 处理；这些工程细节属于论文部署路径，不代表 WAM 代码或模型实现。

## 精度、结果、训练预算与开源状态

- 权重采用 static per-channel quantization，activation 采用 dynamic per-token quantization；主设置为 symmetric W4A4，并另报 asymmetric W3A3（Sec. 5.1，p. 6）。
- 论文报告 W4A4、W3A3 在 Wan 与 CogVideoX family 上的 VBench 结果。Abstract 报告 W3A3 下 VBench average 相对其 QAT baseline 最多提升 6.60；Tables 1–3（pp. 6–7）区分模型与测试设置，注意表中同时出现从 QVGen 论文摘录的数值和作者复跑结果。
- Table 6（p. 9）给的是 raw GPU-days，并明确区分硬件：QVGen 的引用值在 H100 上，DSAQuant 在 H20 上。DSAQuant 报告 CogVideoX-2B **6.07**、Wan 1.3B **8.19**、CogVideoX1.5-5B 约 **65**、Wan 14B 约 **64** H20 GPU-days；Appendix E / Table 9（p. 16）另报 Wan2.2-5B **6.9** H20 GPU-days。不能把这些 raw GPU-days 与 QVGen 的 H100 GPU-days 直接比较。
- Appendix E（p. 16）称训练用 140GB H20、通常每 GPU local batch=1；还说明训练样本为合成数据：Wan 1.3B 从 VidProM prompts 合成，Wan 14B 使用 TurboDiffusion latent，其它模型通过 VAE decode/re-encode 获得模型对应 latent。论文没有单列数据合成/转换的 GPU-days；表中的训练预算不能当作完整数据准备成本。
- 官方 GitHub README 说明仓库提供 inference implementation、checkpoint export 和 latency evaluation，**不含训练代码、training infrastructure、训练数据或权重文件**。但它链接的 Hugging Face 页面确有 Apache-2.0 的 packed W4A4 inference checkpoints：Wan 1.3B、Wan 14B、Wan2.2-5B，总页面文件约 11.5 GB。已发布 inference code/checkpoints 不等于已发布 training pipeline。
- arXiv v1 使用 perpetual non-exclusive license；官方 GitHub 与 Hugging Face model page 标注 Apache-2.0。各许可针对不同材料。

## 迁移到 WAM 时要重新验证

可借鉴的是“监督应匹配过程阶段”的假设，以及把训练改动和推理时启发式拆开测的思路。VDM 的 early/late denoising stage 由 latent noise level 与视频细节形成过程定义；WAM 的 action chunk、预测 horizon、replanning 与机器人 closed-loop time 并非同一个阶段概念。

若要迁移，需先定义 WAM 中对应的阶段与目标，再分别测 action fidelity、planner/CEM 选择（若使用 planner）和 closed-loop control。CFG-drop 只适用于实际使用 CFG、且 conditional/unconditional 分支明确的 WAM；视频模型中的关闭时刻与收益需要在动作生成中重新验证。本文没有测 action loss、action-only quantization 或 robot task success。

## Reading Questions（留空待读）

1. Fig. 1 的 stage-wise substitution 控制了哪些变量？为什么它能定位损伤，却不能单独证明完整训练法有效？
2. Sec. 4.1 的 loss mixing schedule 如何随 denoising step 改变？早期 KD 被认为降低了哪种方差？
3. Sec. 4.2 的误差分解中，CFG 放大的到底是哪一项？dynamic activation quantization 如何导致分支差异？
4. CFG-drop threshold 为什么选最后 9 步？Fig. 4 中性能如何随阈值变化？
5. FP16 的 CFG-drop control 有何变化？它限定了哪些“quantization-specific”解释？
6. Tables 1–3 中，哪些 QVGen 数字引用论文，哪些来自公开 checkpoint 的原始结果，哪些是作者复跑？训练数据不公开时如何影响比较？
7. Table 6/9 的 GPU-days 分别在哪种 GPU 上测量？数据合成与 latent 转换预算是否列出？
8. GitHub runtime 与 Hugging Face weights 各公开了什么？为什么这不能称为开源训练复现？
9. 在 WAM 中，如何定义阶段、对应监督以及可比较的 inference-time guidance？哪些结果仍需 closed-loop 测试？

## Meeting Card（空白）

- 我认为论文定位的失败模式是：
- Fig. 1 的 stage-wise 证据及读数：
- early 与 late 的监督目标：
- CFG mismatch 被放大的条件：
- W4A4/W3A3 设置与结果：
- H20 GPU-days 与未列出的数据准备成本：
- 已公开的 inference/training 资产：
- 可迁移到 WAM 的待验证假设：
- 需要导师讨论的问题：

## 本地与论文证据边界

本目录保存固定版本原 PDF 和阅读指南。本地没有训练或加载模型，没有运行量化、推理、benchmark，也没有验证论文中的质量、GPU-days 或 latency。以上数字均是论文或官方代码/模型页面报告值。
