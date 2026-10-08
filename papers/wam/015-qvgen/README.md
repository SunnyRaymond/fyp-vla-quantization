# 27. QVGen 阅读指南

本目录固定为 arXiv **2505.11497v5**（2026-01-28，ICLR 2026），PDF 共 34 页。先读本地原文：[paper-arxiv-v5.pdf](paper-arxiv-v5.pdf)。

官方来源：[arXiv v5](https://arxiv.org/abs/2505.11497v5) · [论文 HTML](https://arxiv.org/html/2505.11497v5) · [ModelTC/QVGen](https://github.com/ModelTC/QVGen) · [官方 checkpoint collection](https://huggingface.co/collections/Harahan/qvgen)

## 先抓住机制

QVGen 研究的是 **video-generation QAT**：低比特量化的 video diffusion model 难以收敛，作者将问题联系到较大的训练梯度范数；训练时加入低秩辅助模块 \(\Phi\) 缓解量化误差，再用 SVD 与 rank-based regularization 逐步缩减其 rank，直至移除辅助模块，目标是保留训练收益且不增加推理分支。

本文测试 video diffusion models，不是已验证的 **WAM control QAT**。VBench 画质、动作动态分数或压缩结果不能替代 WAM 的 action fidelity、planner/CEM 保真或 closed-loop control 证据。

## 分时阅读路线

| 用时 | 阅读位置（均为本地 PDF 页码） | 读完留下什么 |
|---|---|---|
| 20 分钟 | Abstract 与 Sec. 1，pp. 1–2；Fig. 1，p. 2；Fig. 2，p. 4；Table 1，p. 7 | 用两句话说明失败现象、\(\Phi\) 与 rank-decay；标出一项 VBench 指标及其对照方法 |
| 90 分钟 | Sec. 3.1，pp. 4–5；Fig. 3，p. 5；Sec. 3.2，pp. 5–6；Sec. 4.1，p. 6；Tables 1–2，pp. 7–8 | 写下梯度范数论证依赖的条件；画出辅助分支如何加入、何时缩减；分开看小模型和大模型结果 |
| 180 分钟 | Secs. 4.2–4.4，pp. 7–10；Algorithm 1，p. 20；Appendices B–D，pp. 20–23；Table A，p. 23；Appendices N–O，pp. 29–30 | 检查理论条件与训练细节；记录 GPU-days、显存口径和尚未覆盖的 WAM 问题；完成下方 Meeting Card |

## 结果、预算与开源状态

- 量化对象为 linear layers；正文采用 per-channel weight、per-token activation quantization，报告 W4A4 与 W3A3。Table 1–2（pp. 7–8）是 VBench 生成质量结果，不是控制成功率。
- 训练数据为 16K 条带 caption 的 OpenVidHQ-4M 视频。论文报告小模型训练 8 epochs、8×H100；CogVideoX1.5-5B 为 16 epochs、16×H100；Wan 14B 为 16 epochs、32×H100（Sec. 4.1，p. 6）。
- Table A（p. 23）报告 H100 GPU-days：CogVideoX-2B **9.44**、Wan 1.3B **11.11**、CogVideoX1.5-5B 约 **51**、Wan 14B 约 **182**。Table 7（p. 10）另报前两者单 GPU training memory 分别为 **67.93 GB/GPU** 与 **66.67 GB/GPU**。这些是论文报告值，不是本地复现；不要把训练预算写成推理延迟。
- 官方仓库在 2026-10-05 可见 `train.py`、`training/` 与训练脚本，并说明 Wan 1.3B 示例可用于 CogVideoX-2B；README 仍将 Wan 14B、CogVideoX 5B 的训练/推理代码列为待补项目。作者已发布代码和部分 W4A4 checkpoints。不能据此称所有论文规模的训练路径、数据或 checkpoint 都已开放。
- 论文页面标注 CC BY 4.0；代码仓库为 Apache-2.0。请分别理解论文与代码许可。

## 迁移到 WAM 时要重新验证

可迁移的只是研究线索：量化误差是否造成难优化的梯度信号，以及训练期辅助结构能否在部署前收缩掉。WAM 的学习目标、action token、observation/action conditioning 与控制轨迹都不同。

至少要重新回答：\(\Phi\) 是否降低 **executed action** 的误差，而非只降低视频或 latent loss？移除 \(\Phi\) 后，动作 chunk 与下一轮 replanning 是否保真？对具体 WAM backbone 的量化范围、kernel、推理延迟和显存有何影响？闭环任务结果是否变化？本文没有回答这些问题，也没有验证 action-only QAT。

## Reading Questions（留空待读）

1. Sec. 3 的梯度范数结论使用哪些优化假设？哪些是理论界，哪些是经验相关性？
2. Fig. 3 中梯度范数与训练 loss 的曲线，能否单独证明梯度范数是性能差异的原因？
3. \(\Phi\) 的低秩分解和 rank-decay 更新分别是什么？\(r=32\) 与衰减率如何影响训练成本？
4. 移除 \(\Phi\) 后性能保持的证据在哪些模型、bit-width 与指标上成立？
5. Table A 的 GPU-days 在训练了多少 epochs、哪些 H100 数量下测得？Table 7 的显存对应什么阶段和口径？
6. 论文的 3-bit/4-bit 结果涉及哪些生成质量指标？哪些差异不能从 aggregate score 推断？
7. 映射到 WAM 时，video target、teacher distillation 与 action endpoint loss 各自会变成什么？需要哪些独立对照？

## Meeting Card（空白）

- 我理解的目标问题：
- 最关键的理论条件：
- Fig./Table 与具体读数：
- \(\Phi\) 加入和 rank-decay 的步骤：
- 训练 GPU-days 与显存口径：
- 我认为可迁移到 WAM 的部分：
- 从本文无法推出的 WAM 结论：
- 需要导师讨论的问题：

## 本地与论文证据边界

本目录保存固定版本原 PDF 和阅读指南。本地没有训练或加载模型，没有运行量化、推理、benchmark，也没有验证论文中的 GPU-days、质量或效率结果。以上数字均是论文/官方仓库报告值。
