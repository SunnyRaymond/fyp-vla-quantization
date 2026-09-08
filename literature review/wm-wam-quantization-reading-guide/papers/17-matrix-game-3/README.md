# 17. Matrix-Game 3.0

**Matrix-Game 3.0: Real-Time and Streaming Interactive World Model with Long-Horizon Memory**  
本地版本：**arXiv:2604.08995v2** · 20 页 · 来源核验：2026-09-08 · 阅读状态：`unread`  
分类：专项 / interactive video WM

[本地 PDF](paper-arxiv-v2.pdf) · [arXiv 版本页](https://arxiv.org/abs/2604.08995v2) · [代码或官方项目入口](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-3) · [返回总指南](../../README.md)

## 背景与要解决的问题

用于看 long-horizon interactive world generation 的 memory 和效率路线，不属于机器人 manipulation WAM 的同一排名。

## 方法与核心机制

Video-pose-action-prompt 数据、预测 residual 与 frame re-injection、camera-aware memory，以及 DMD、quantization、VAE 优化。

## 结果：带着条件读证据

论文展示 720p real-time generation；官方 README 提供 base/distilled 模型路线，推理命令包含 use_int8 和 LightVAE 相关参数。

以上为原文/官方公开材料的报告，本次未运行训练、量化或完整 benchmark。页码按本地 PDF 的物理页编号，可能不同于印刷页码。

## 边界与对量化选题的意义

40 FPS 必须连同 GPU 数、resolution、distillation、decoder 和同步设置报告。README 中仍有 later-release model 项；不能据论文规模推断所有权重均已公开。Keyboard/camera action control 不等于 physical robot action success。

## 分时阅读路线

- **20 分钟定位**：abstract → method overview → 本页列出的结果与限制，写出模型实际输入和输出。
- **75 分钟理解**：方法 25 分钟，实验协议与结果 25 分钟，appendix/官方入口 15 分钟，填写 Meeting Card 10 分钟。
- **2 小时深入**：沿一个具体 failure mode 核对模型、data split、precision、planner/inference budget，记录哪些内容只有论文、哪些已有代码、哪些还缺证据。

## Reading Questions（读完自己回答）

1. Memory retrieval 在错误帧积累后是否仍正确？
2. INT8 用于哪些部分，是否真正调用 integer kernels？
3. 生成吞吐与输入动作到画面响应 latency 有何差别？
4. 如何验证 action fidelity 而不只评 VBench？

## Meeting Card（留空）

- 我要解决的具体问题：
- 模型的训练 / 测试输入和输出：
- 最关键机制与证据位置：
- 数据、checkpoint、benchmark 与 precision 条件：
- 一个重要局限或未复现点：
- 与我的量化 idea 重合的部分：
- 我准备验证的不同假设：
- 想向导师/师兄确认的问题：
