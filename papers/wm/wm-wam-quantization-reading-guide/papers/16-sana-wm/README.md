# 16. SANA-WM

**SANA-WM: Efficient Minute-Scale World Modeling with Hybrid Linear Diffusion Transformer**  
本地版本：**arXiv:2605.15178v1** · 25 页 · 来源核验：2026-09-08 · 阅读状态：`unread`  
分类：专项 / 视频 WM 低精度部署

[本地 PDF](paper-arxiv-v1.pdf) · [arXiv 版本页](https://arxiv.org/abs/2605.15178v1) · [代码或官方项目入口](https://github.com/NVlabs/Sana/blob/main/docs/sana_wm.md) · [返回总指南](../../README.md)

## 背景与要解决的问题

2026-05 的 efficient minute-scale world model。它说明广义视频 WM 已有实际 low-precision deployment，不能把整个 WM 量化说成空白。

## 方法与核心机制

Hybrid linear attention、camera control、two-stage generation/refinement 与数据管线；蒸馏和低精度进一步降低生成成本。

## 结果：带着条件读证据

读效率实验、camera-following 和 one-minute benchmark。论文报告 distilled + NVFP4 在 RTX 5090 的 denoising 结果；当前 docs 提供 stage1/refiner 独立 bf16/fp8/fp4 选项。代码限制 FP8 使用 Hopper/Blackwell，NVFP4 使用 Blackwell。

以上为原文/官方公开材料的报告，本次未运行训练、量化或完整 benchmark。页码按本地 PDF 的物理页编号，可能不同于印刷页码。

## 边界与对量化选题的意义

控制输入主要是 camera motion，不等同 robot manipulation policy。34s 对 60s clip 的描述是作者 denoising 口径，不代表完整系统延迟或零等待 streaming。Docs、streaming checkpoint 与原 bidirectional checkpoint 应分开；A100 不能直接复现 Blackwell NVFP4 kernel 的性能。

## 分时阅读路线

- **20 分钟定位**：abstract → method overview → 本页列出的结果与限制，写出模型实际输入和输出。
- **75 分钟理解**：方法 25 分钟，实验协议与结果 25 分钟，appendix/官方入口 15 分钟，填写 Meeting Card 10 分钟。
- **2 小时深入**：沿一个具体 failure mode 核对模型、data split、precision、planner/inference budget，记录哪些内容只有论文、哪些已有代码、哪些还缺证据。

## Reading Questions（读完自己回答）

1. 哪些 Linear 被量化，camera conditioning 是否保持高精度？
2. Stage 1 与 refiner 的误差分别影响 pose 和视觉质量多少？
3. Camera control 的微小 residual 是否被量化抹掉？
4. 论文 speedup 中 distillation、quantization、architecture 如何拆分？

## Meeting Card（留空）

- 我要解决的具体问题：
- 模型的训练 / 测试输入和输出：
- 最关键机制与证据位置：
- 数据、checkpoint、benchmark 与 precision 条件：
- 一个重要局限或未复现点：
- 与我的量化 idea 重合的部分：
- 我准备验证的不同假设：
- 想向导师/师兄确认的问题：
