# 09. LingBot-VA 2.0

**Native Video-Action Pretraining for Generalizable Robot Control**  
本地版本：**arXiv:2607.08639v2** · 30 页 · 来源核验：2026-09-08 · 阅读状态：`unread`  
分类：近期观察 / 开放程度单独核验

[本地 PDF](paper-arxiv-v2.pdf) · [arXiv 版本页](https://arxiv.org/abs/2607.08639v2) · [代码或官方项目入口](https://technology.robbyant.com/lingbot-va-v2) · [返回总指南](../../README.md)

## 背景与要解决的问题

Native Video-Action Pretraining for Generalizable Robot Control，2026-07。用于了解 WAM 从“改造 video generator”转向 native embodied pretraining 的发展。

## 方法与核心机制

Semantic visual-action tokenizer、from-scratch causal pretraining、sparse MoE backbone，以及预测 future latents 同时执行 action、再用真实观测 re-ground 的异步机制。

## 结果：带着条件读证据

PDF p.1 总结四个设计，随后读 tokenizer、causal learning、MoE 和部署评估。官方项目页有 Tech Report，但 GitHub/Hugging Face 按钮指向组织主页；本次查看的 lingbot-va README 仍以第一代模型为主。

以上为原文/官方公开材料的报告，本次未运行训练、量化或完整 benchmark。页码按本地 PDF 的物理页编号，可能不同于印刷页码。

## 边界与对量化选题的意义

本次未核实完整 2.0 版本的专用权重与可执行 benchmark 配置，故列为近期观察，不作为“已验证可跑的开源 baseline”。LingBot-VLA 2.0、LingBot-VA 2.0、LingBot-World 2.0 是不同工作。

## 分时阅读路线

- **20 分钟定位**：abstract → method overview → 本页列出的结果与限制，写出模型实际输入和输出。
- **75 分钟理解**：方法 25 分钟，实验协议与结果 25 分钟，appendix/官方入口 15 分钟，填写 Meeting Card 10 分钟。
- **2 小时深入**：沿一个具体 failure mode 核对模型、data split、precision、planner/inference budget，记录哪些内容只有论文、哪些已有代码、哪些还缺证据。

## Reading Questions（读完自己回答）

1. 新 tokenizer 的 semantic/action supervision 来自哪里？
2. Sparse MoE 的 active parameters 与总 parameters 各是多少？
3. Causal pretraining 与 inference cache 如何一致？
4. 正式采用前，是否已取得版本对应的 code、weights、stats、evaluation config？

## Meeting Card（留空）

- 我要解决的具体问题：
- 模型的训练 / 测试输入和输出：
- 最关键机制与证据位置：
- 数据、checkpoint、benchmark 与 precision 条件：
- 一个重要局限或未复现点：
- 与我的量化 idea 重合的部分：
- 我准备验证的不同假设：
- 想向导师/师兄确认的问题：
