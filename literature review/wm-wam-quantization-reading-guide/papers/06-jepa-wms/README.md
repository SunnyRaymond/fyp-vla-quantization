# 06. JEPA-WMs：What Drives Success in Physical Planning?

**What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?**  
本地版本：**arXiv:2512.24497v4** · 55 页 · 来源核验：2026-09-08 · 阅读状态：`unread`  
分类：优先核心 / 强 latent planning 对照

[本地 PDF](paper-arxiv-v4.pdf) · [arXiv 版本页](https://arxiv.org/abs/2512.24497v4) · [代码或官方项目入口](https://github.com/facebookresearch/jepa-wms) · [返回总指南](../../README.md)

## 背景与要解决的问题

Yann LeCun 参与的另一项工作；TMLR 05/2026，当前 arXiv v4。对想选择“更强 latent baseline”的研究，比只看早期 DINO-WM 更有帮助。

## 方法与核心机制

系统比较 encoder、predictor conditioning/architecture、rollout training、context、proprioception 和 planner；整合有效设计构成 JEPA-WMs。

## 结果：带着条件读证据

Table 1 / PDF p.5 是最值得先看的设计地图：sim navigation 与 real manipulation 的推荐 encoder、predictor depth、rollout/context 不同。官方仓库提供 JEPA-WM、DINO-WM 与 V-JEPA-2-AC(fixed) 的 baseline models，支持多个 simulation 环境及 DROID/RoboCasa 相关设置。

以上为原文/官方公开材料的报告，本次未运行训练、量化或完整 benchmark。页码按本地 PDF 的物理页编号，可能不同于印刷页码。

## 边界与对量化选题的意义

“fixed” baseline 与原始 V-JEPA 2-AC 不能混称；recipe 改进不是 quantization 改进。仓库 license 为 CC-BY-NC 4.0，属于公开研究代码/权重，不能不加区分地称为无限制开源。

## 分时阅读路线

- **20 分钟定位**：abstract → method overview → 本页列出的结果与限制，写出模型实际输入和输出。
- **75 分钟理解**：方法 25 分钟，实验协议与结果 25 分钟，appendix/官方入口 15 分钟，填写 Meeting Card 10 分钟。
- **2 小时深入**：沿一个具体 failure mode 核对模型、data split、precision、planner/inference budget，记录哪些内容只有论文、哪些已有代码、哪些还缺证据。

## Reading Questions（读完自己回答）

1. 哪个设计在 held-out 环境上仍然成立？
2. 训练 rollout 深度能否减轻量化 rollout drift，还是只适应 FP model error？
3. Table 1 的 sim 与 real 推荐为何不同？
4. 使用 corrected baseline 是否需要重新报告 FP reference？
5. Proprioception 的高精度旁路是否掩盖视觉量化损失？

## Meeting Card（留空）

- 我要解决的具体问题：
- 模型的训练 / 测试输入和输出：
- 最关键机制与证据位置：
- 数据、checkpoint、benchmark 与 precision 条件：
- 一个重要局限或未复现点：
- 与我的量化 idea 重合的部分：
- 我准备验证的不同假设：
- 想向导师/师兄确认的问题：
