# 05. V-JEPA 2.1

**V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning**  
本地版本：**arXiv:2603.14482v3** · 37 页 · 来源核验：2026-09-08 · 阅读状态：`unread`  
分类：扩展 / 更新的 video representation

[本地 PDF](paper-arxiv-v3.pdf) · [arXiv 版本页](https://arxiv.org/abs/2603.14482v3) · [代码或官方项目入口](https://github.com/facebookresearch/vjepa2) · [返回总指南](../../README.md)

## 背景与要解决的问题

这是 V-JEPA 2 的后续训练 recipe，Yann LeCun 为作者之一。它应与 2-AC 配套理解，不应只因版本号较新就替换全部 robot pipeline。

## 方法与核心机制

Dense predictive loss 同时监督 visible 与 masked tokens；deep self-supervision 作用于多个中间层；image/video tokenizers 和 model/data scaling 改善 dense、temporally consistent features。

## 结果：带着条件读证据

优先看 dense features 的方法、PDF pp.4–6 的主结果，再定位 robot grasping、depth、navigation 和 anticipation。原文报告相对 V-JEPA 2-AC 的 grasping 提升，但这些设置与分类 benchmark 不同。

以上为原文/官方公开材料的报告，本次未运行训练、量化或完整 benchmark。页码按本地 PDF 的物理页编号，可能不同于印刷页码。

## 边界与对量化选题的意义

开放了新 encoder family 不自动证明目标 action-conditioned checkpoint、robot controller 和完整 evaluation 配置都与原 2-AC 兼容。更新的 features 是否更抗量化，需要实验。

## 分时阅读路线

- **20 分钟定位**：abstract → method overview → 本页列出的结果与限制，写出模型实际输入和输出。
- **75 分钟理解**：方法 25 分钟，实验协议与结果 25 分钟，appendix/官方入口 15 分钟，填写 Meeting Card 10 分钟。
- **2 小时深入**：沿一个具体 failure mode 核对模型、data split、precision、planner/inference budget，记录哪些内容只有论文、哪些已有代码、哪些还缺证据。

## Reading Questions（读完自己回答）

1. Dense supervision 改变了哪些 intermediate activation distributions？
2. 模型大小与 feature quality 的影响有没有控制？
3. Robot experiment 采用哪个 encoder 和 post-training recipe？
4. 只换 encoder 会不会破坏旧 predictor 的 latent coordinate system？

## Meeting Card（留空）

- 我要解决的具体问题：
- 模型的训练 / 测试输入和输出：
- 最关键机制与证据位置：
- 数据、checkpoint、benchmark 与 precision 条件：
- 一个重要局限或未复现点：
- 与我的量化 idea 重合的部分：
- 我准备验证的不同假设：
- 想向导师/师兄确认的问题：
