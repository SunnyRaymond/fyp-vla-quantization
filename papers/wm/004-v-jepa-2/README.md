# 04. V-JEPA 2 / V-JEPA 2-AC

**V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning**  
本地版本：**arXiv:2506.09985v1** · 48 页 · 来源核验：2026-09-08 · 阅读状态：`unread`  
分类：优先核心 / 大规模 LeCun 候选

[本地 PDF](paper-arxiv-v1.pdf) · [arXiv 版本页](https://arxiv.org/abs/2506.09985v1) · [代码或官方项目入口](https://github.com/facebookresearch/vjepa2) · [返回总指南](../README.md)

## 背景与要解决的问题

若推荐者说的是“Meta 那个通过看视频学习、能操控机械臂的 world model”，这篇最吻合。V-JEPA 2 是 video representation pretraining；V-JEPA 2-AC 才是 action-conditioned robot planning 分支。

## 方法与核心机制

先在大规模视频上学习 masked latent prediction，再用 DROID 的 robot interaction data post-train action-conditioned predictor；使用 image goal 和 MPC 生成行为。Video QA 需要额外与 language model 对齐，不能把 backbone 单独说成通用问答模型。

## 结果：带着条件读证据

先读 abstract 与 robot planning 部分，再区分三组实验：Something-Something v2 的 motion understanding、EPIC-KITCHENS 的 anticipation、Franka 上的闭环 manipulation。原文报告用不足 62 小时 DROID interaction data 构建 2-AC，并在两个 lab 测试。官方 repository 提供 V-JEPA 2 与 2-AC checkpoint/代码入口。

以上为原文/官方公开材料的报告，本次未运行训练、量化或完整 benchmark。页码按本地 PDF 的物理页编号，可能不同于印刷页码。

## 边界与对量化选题的意义

Internet-scale pretraining 成本与轻量 post-training 成本必须分开；“无 task-specific data”不是“无 robot data”。原论文的真实机器人任务不是标准 LIBERO leaderboard。量化 encoder 后若只测视频分类，尚不能证明 planning 保留。代码许可证与各 checkpoint 条款应分别看。

## 分时阅读路线

- **20 分钟定位**：abstract → method overview → 本页列出的结果与限制，写出模型实际输入和输出。
- **75 分钟理解**：方法 25 分钟，实验协议与结果 25 分钟，appendix/官方入口 15 分钟，填写 Meeting Card 10 分钟。
- **2 小时深入**：沿一个具体 failure mode 核对模型、data split、precision、planner/inference budget，记录哪些内容只有论文、哪些已有代码、哪些还缺证据。

## Reading Questions（读完自己回答）

1. Action-free pretraining 与 action-conditioned post-training 的 predictor 是否同一个？
2. 2-AC 的输入包含哪些 robot state 和坐标约定？
3. Goal encoder 是否缓存；量化时应该复用什么 precision？
4. 机器人成功率的 task、trials、环境变化如何定义？
5. 要以它作为 FYP baseline，是复现 representation 还是复现闭环 planning？

## Meeting Card（留空）

- 我要解决的具体问题：
- 模型的训练 / 测试输入和输出：
- 最关键机制与证据位置：
- 数据、checkpoint、benchmark 与 precision 条件：
- 一个重要局限或未复现点：
- 与我的量化 idea 重合的部分：
- 我准备验证的不同假设：
- 想向导师/师兄确认的问题：

