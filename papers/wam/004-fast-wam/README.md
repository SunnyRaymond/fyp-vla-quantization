# 10. Fast-WAM

**Fast-WAM: Do World Action Models Need Test-time Future Imagination?**  
本地版本：**arXiv:2603.16666v2** · 13 页 · 来源核验：2026-09-08 · 阅读状态：`unread`  
分类：优先核心 / WAM efficiency baseline

[本地 PDF](paper-arxiv-v2.pdf) · [arXiv 版本页](https://arxiv.org/abs/2603.16666v2) · [代码或官方项目入口](https://github.com/yuantianyuan01/FastWAM) · [返回总指南](../README.md)

## 背景与要解决的问题

这篇直接询问：WAM 的益处是否主要来自训练时的视频建模，测试时还需要显式生成未来吗？它也是 QuantWAMs 的直接 baseline。

## 方法与核心机制

保留 training-time video co-training，但在 inference 跳过昂贵的 future generation；通过多个控制变体分开检验 co-training 与 test-time imagination。

## 结果：带着条件读证据

读方法中的 variants、PDF pp.7–8 的实验与 pp.12–13 的配置。论文报告 190ms latency；当前 README 已新增进一步优化和 Optional IDM，同一 checkpoint 可用不同模式。阅读库记录了源码 commit，原论文数字与新 README 数字单独保留。

以上为原文/官方公开材料的报告，本次未运行训练、量化或完整 benchmark。页码按本地 PDF 的物理页编号，可能不同于印刷页码。

## 边界与对量化选题的意义

“Fast-WAM 不需要生成 future RGB”不等于运行时没有 video backbone；也不证明所有 OOD 场景都不需要 future representations。不要用较新 Optional IDM 的结果替换旧 paper baseline 而不说明。

## 分时阅读路线

- **20 分钟定位**：abstract → method overview → 本页列出的结果与限制，写出模型实际输入和输出。
- **75 分钟理解**：方法 25 分钟，实验协议与结果 25 分钟，appendix/官方入口 15 分钟，填写 Meeting Card 10 分钟。
- **2 小时深入**：沿一个具体 failure mode 核对模型、data split、precision、planner/inference budget，记录哪些内容只有论文、哪些已有代码、哪些还缺证据。

## Reading Questions（读完自己回答）

1. 哪些模块在 test time 仍执行，哪些真正被省略？
2. Without video co-training 的对照是否训练预算匹配？
3. Original checkpoint 与 Optional IDM checkpoint 的 action scheduler 有何区别？
4. 在 OOD 条件下去掉 future conditioning 是否仍成立？
5. QuantWAMs 是否对齐了当前 repo 的同一模型版本？

## Meeting Card（留空）

- 我要解决的具体问题：
- 模型的训练 / 测试输入和输出：
- 最关键机制与证据位置：
- 数据、checkpoint、benchmark 与 precision 条件：
- 一个重要局限或未复现点：
- 与我的量化 idea 重合的部分：
- 我准备验证的不同假设：
- 想向导师/师兄确认的问题：

