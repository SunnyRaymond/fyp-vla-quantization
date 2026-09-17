# 20. WorldArena 2.0

**WorldArena 2.0: Extending Embodied World Model Benchmarking on Modality, Functionality and Platform**  
本地版本：**arXiv:2605.17912v1** · 18 页 · 来源核验：2026-09-08 · 阅读状态：`unread`  
分类：Benchmark 扩展 / 新协议

[本地 PDF](paper-arxiv-v1.pdf) · [arXiv 版本页](https://arxiv.org/abs/2605.17912v1) · [代码或官方项目入口](https://github.com/WorldArena2/WorldArena-2.0) · [返回总指南](../README.md)

## 背景与要解决的问题

在 WorldArena 上扩展 visuotactile、interactive RL utility 和 real-robot/cross-platform evaluation。属于新兴 suite，本库不声称它已经成为所有 WM 论文的共同标准。

## 方法与核心机制

按 modality、functionality、platform 三维拓展；不仅评视频，也评作为 interactive environment 的作用和真实机器人任务。

## 结果：带着条件读证据

读三个 track 的输入输出协议、评估配置与 dataset。官方页面已更新 motion-quality scoring，不能跨版本直接拼接排行榜。

以上为原文/官方公开材料的报告，本次未运行训练、量化或完整 benchmark。页码按本地 PDF 的物理页编号，可能不同于印刷页码。

## 边界与对量化选题的意义

Track 1 video quality、Track 2 learned RL environment 和 Track 3 real robot 要分别解释。新 protocol 的可用资源、闭源裁判或实物设备要求影响复现成本。

## 分时阅读路线

- **20 分钟定位**：abstract → method overview → 本页列出的结果与限制，写出模型实际输入和输出。
- **75 分钟理解**：方法 25 分钟，实验协议与结果 25 分钟，appendix/官方入口 15 分钟，填写 Meeting Card 10 分钟。
- **2 小时深入**：沿一个具体 failure mode 核对模型、data split、precision、planner/inference budget，记录哪些内容只有论文、哪些已有代码、哪些还缺证据。

## Reading Questions（读完自己回答）

1. 哪些 track 可以只用本地 simulator 完成？
2. 动作条件和帧是否严格一一对齐？
3. 当 WM 作为 RL environment，policy 会不会 exploit model error？
4. 相同 quantization 在视频分数与 interactive utility 上是否不同方向？

## Meeting Card（留空）

- 我要解决的具体问题：
- 模型的训练 / 测试输入和输出：
- 最关键机制与证据位置：
- 数据、checkpoint、benchmark 与 precision 条件：
- 一个重要局限或未复现点：
- 与我的量化 idea 重合的部分：
- 我准备验证的不同假设：
- 想向导师/师兄确认的问题：

