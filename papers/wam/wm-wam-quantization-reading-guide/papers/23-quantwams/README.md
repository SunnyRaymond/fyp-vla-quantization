# 23. QuantWAMs

**QuantWAMs: Calibrating at the Right Granularity for World Action Models**  
本地版本：**arXiv:2607.28405v1** · 13 页 · 来源核验：2026-09-08 · 阅读状态：`unread`  
分类：最高优先 / 直接 WAM PTQ prior art

[本地 PDF](paper-arxiv-v1.pdf) · [arXiv 版本页](https://arxiv.org/abs/2607.28405v1) · [代码或官方项目入口](https://quantwams.github.io/) · [返回总指南](../../README.md)

## 背景与要解决的问题

2026-07 的直接 WAM quantization 工作，测试 Fast-WAM 和 LingBot-VA。即使暂未确认代码公开，它也已构成必须对照的论文 prior art，不能忽略。

## 方法与核心机制

Shared-basis outlier calibration 只在 coordinate-compatible modules 汇集统计；co-training-objective saliency 用 joint video/action empirical-Fisher signal 分配 layer precision；fixed-intervention rollout auditing 在可达闭环状态上修正 protected denoising steps。

## 结果：带着条件读证据

Tables 1–2 / PDF p.7：RoboTwin 2.0、LIBERO；pp.8–9：实际 precision schedule、measurement scope、matched-budget controls；p.10：ablation；p.11：AgiBot G2。主路径 W4A4，但 top 20% candidate Linears 按 count 升到 W8A8，top 2% channels 保留 BF16，并有 A8 protected steps。实现报告使用 SM120 Blackwell NVFP4/FP8 kernels。

以上为原文/官方公开材料的报告，本次未运行训练、量化或完整 benchmark。页码按本地 PDF 的物理页编号，可能不同于印刷页码。

## 边界与对量化选题的意义

不是全模型纯 W4A4；4.8 weight bits 是按 Linear 数量平均，非 parameter-weighted model size。约 29% memory、1.4–1.6× speedup 针对选定 video/action blocks，排除 VAE、embedding、projection 和其他控制流程。PTQ controls 并未匹配额外 backward/closed-loop calibration information。真实机器人仅三任务各十次，不能据此称“与 FP16 等价”。本地 v1 只有 13 页，正文引用 Appendix A–E，但下载 PDF 未附这些 appendix；project page 的 Code 未提供外部 repository link，本次未确认公开实现。

## 分时阅读路线

- **20 分钟定位**：abstract → method overview → 本页列出的结果与限制，写出模型实际输入和输出。
- **75 分钟理解**：方法 25 分钟，实验协议与结果 25 分钟，appendix/官方入口 15 分钟，填写 Meeting Card 10 分钟。
- **2 小时深入**：沿一个具体 failure mode 核对模型、data split、precision、planner/inference budget，记录哪些内容只有论文、哪些已有代码、哪些还缺证据。

## Reading Questions（读完自己回答）

1. 哪些 modules 真正共享 latent coordinate basis，哪些不可合并统计？
2. Empirical Fisher 的 supervision 具体来自哪些原始 video/action targets？
3. Precision budget 按 parameter 数、layer count 还是 runtime bytes 定义？
4. 额外 profiling/validation rollouts 会不会成为不公平的信息优势？
5. Reported block speedup 加上 VAE、encoding、通信后剩多少？
6. 如果在 A100 上做 weight-only INT4，哪些结论可迁移，哪些不能？
7. 缺少的 Appendix A–E 是否有单独 supplementary，是否能取得作者实现？

## Meeting Card（留空）

- 我要解决的具体问题：
- 模型的训练 / 测试输入和输出：
- 最关键机制与证据位置：
- 数据、checkpoint、benchmark 与 precision 条件：
- 一个重要局限或未复现点：
- 与我的量化 idea 重合的部分：
- 我准备验证的不同假设：
- 想向导师/师兄确认的问题：
