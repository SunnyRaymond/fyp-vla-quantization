# 01. LeWorldModel（LeWM）

**LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels**  
本地版本：**arXiv:2603.19312v3** · 28 页 · 来源核验：2026-09-08 · 阅读状态：`unread`  
分类：优先核心 / 低成本 latent WM

[本地 PDF](paper-arxiv-v3.pdf) · [arXiv 版本页](https://arxiv.org/abs/2603.19312v3) · [代码或官方项目入口](https://github.com/lucas-maes/le-wm) · [返回总指南](../../README.md)

## 背景与要解决的问题

这是“Yann LeCun 参与、开源、适合 baseline”的重要候选。作者为 Lucas Maes、Quentin Le Lidec、Damien Scieur、Yann LeCun、Randall Balestriero。仅凭转述不能断定推荐者指的就是它；DINO-WM 和 V-JEPA 2-AC 也符合描述。LeWM 从环境的离线 observation-action trajectories 学习可规划的 latent dynamics，适合先做小规模量化实验。

## 方法与核心机制

Encoder 将图像压成 compact embedding；action-conditioned predictor 从历史 embeddings 和 actions 预测下一步。训练使用 next-embedding MSE + λ SIGReg，jointly 更新 encoder 与 predictor。SIGReg 对随机投影上的一维分布施加 Gaussian regularization，防止 collapse；不是把 representation 离散化。Inference 用 CEM 搜索 action sequence，以 terminal latent goal distance 为代价，再按 MPC 协议执行和重规划。

## 结果：带着条件读证据

PDF pp.4–6：模型、loss 和 planning；Figure 6 / p.7：四个环境。作者报告 PushT 96%、Reacher 86%、Two-Room 87%、OGBench-Cube 74%。Figure 3 / p.3：约 0.98s 对 47s 的完整 planning 比较及 fixed-FLOPs 比较。Appendix D / pp.17–18：frameskip=5、action block、CEM candidates/iterations。务必区分 Figure 3 的 fixed-compute 实验和 Figure 6 的配置，不能把各列最好的数字拼接。

以上为原文/官方公开材料的报告，本次未运行训练、量化或完整 benchmark。页码按本地 PDF 的物理页编号，可能不同于印刷页码。

## 边界与对量化选题的意义

作者报告约 15M parameters、单 GPU 数小时训练；这是原文条件，不是本地 ASPIRE2A 实测预算。LeWM 在 Two-Room 与 OGBench-Cube 并非所有方法中最优，不能称为全领域 SOTA。原文“单一有效超参数”指 λ 的调参，仍有 projection count、optimizer、architecture 和 planner 配置。Gaussian latent prior 不保证量化友好，也不保证 action ranking 保持。请配套读 #18：独立复现指出 TwoRoom 的协议差异，但该文只覆盖一个诊断环境和单 seed，不构成对整篇工作的否定。

## 分时阅读路线

- **20 分钟定位**：abstract → method overview → 本页列出的结果与限制，写出模型实际输入和输出。
- **75 分钟理解**：方法 25 分钟，实验协议与结果 25 分钟，appendix/官方入口 15 分钟，填写 Meeting Card 10 分钟。
- **2 小时深入**：沿一个具体 failure mode 核对模型、data split、precision、planner/inference budget，记录哪些内容只有论文、哪些已有代码、哪些还缺证据。

## Reading Questions（读完自己回答）

1. SIGReg 约束的是哪个 representation？随机投影数量有限时，理论与实际 loss 的关系是什么？
2. Encoder/projector/predictor/action embedder 哪些参与 inference，哪些适合先单独量化？
3. 同一批 candidate actions，低精度是否改变 top-k 排名，即使 latent MSE 很小？
4. Figure 3 的 token 数、FLOPs 与 wall-clock 条件分别是什么？
5. Appendix 与当前 config 对 goal offset、step budget、normalization 是否一致？
6. 只量化 predictor 和同时量化 goal encoder，会导致怎样不同的 cost geometry？

## Meeting Card（留空）

- 我要解决的具体问题：
- 模型的训练 / 测试输入和输出：
- 最关键机制与证据位置：
- 数据、checkpoint、benchmark 与 precision 条件：
- 一个重要局限或未复现点：
- 与我的量化 idea 重合的部分：
- 我准备验证的不同假设：
- 想向导师/师兄确认的问题：
