# 007. DreamerV3：Nature 2025 / block GRU 阅读路线

**Mastering diverse control tasks through world models**

[本地 Nature PDF](paper-nature-2025.pdf) · [Nature 原文](https://www.nature.com/articles/s41586-025-08744-2) · [官方代码](https://github.com/danijar/dreamerv3) · [本专题总路线](../READING-FACTORED-DYNAMICS.md) · [原 arXiv 阅读记录](README.md)

## 1. Paper identity 与版本

| 项目 | 内容 |
|---|---|
| Authors | Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap |
| Publication | Nature 640, 647–653, 2025；online 2025-04-02 |
| DOI | `10.1038/s41586-025-08744-2` |
| 本地版本 | Publisher version of record，19 个 PDF 物理页，来自 Nature PDF URL |
| 核对日期 / 阅读状态 | 2026-09-26 / `unread` |
| 本次范围 | 核对原文 identity、Methods/Networks 与官方 RSSM `_core`；未运行模型或 benchmark |

本目录原有 `paper-arxiv-v2.pdf` 是 arXiv:2301.04104v2，标题 *Mastering Diverse Domains through World Models*，40 页。它与这份 Nature 版本是不同文档，原 PDF 与 metadata 的历史字段均保留。本次有关 eight-block GRU 的讨论以 **Nature Methods / Networks** 为依据，不把旧稿自动改写为新版。

## 2. 先理解三个不同的“状态”

- `x_t`：真实 observation，可以是 image 或 vector。
- `z_t`：stochastic latent，采用多个 categorical distributions。
- `h_t`：recurrent deterministic memory，负责承载过去的信息。

Dreamer 的 model state 由 `h_t` 与 `z_t` 组合。**这次分成 blocks 的主要对象是 `h_t`，不是把图像 encoder 的向量直接切片后各自预测。**

它学习 world model，并用 imagined trajectories 训练 actor/critic。LeWM 的 test-time CEM goal planning 是另一种使用 world model 的方式，阅读时把两种计算路径分开。

## 3. Problem 与方法

问题：扩大 recurrent memory 会增加模型容量，但普通 dense recurrent update 的参数和计算会随 hidden width 二次增长。

Nature Methods 的 `Model sizes` 与 `Networks` 将 recurrent units 分成 **8 个 blocks**，以 block-diagonal GRU 更新。可以把主要机制理解为：

1. 从 stochastic latent、action 和完整 recurrent memory 构造共同输入。
2. 每个 block 使用自己的局部 recurrent state，加上共同输入，计算局部更新。
3. 把更新后的 blocks 拼回完整 recurrent state。

共同输入允许 blocks 交换信息。这里的 block-diagonal 约束减少局部更新中的 dense connectivity，**不表示不同 blocks 的 dynamics 完全独立**。

如果只比较一个 `D × D` recurrent matrix，与等宽的 `K` 个独立 square blocks，这一部分参数从 `D²` 变为 `D²/K`。这是结构上的计数；共同投影、gates、其他网络与运行开销仍然存在，不能把该比例称为 full-model speedup。

## 4. 证据位置与实际实现

| 要读的内容 | Nature PDF 位置 / 官方实现 |
|---|---|
| World model、actor、critic 的关系 | Figure 1；World model learning；Actor–critic learning |
| `h_t` 与 `z_t` 的不同角色 | World model learning 中的 RSSM equations |
| Eight blocks 与模型尺寸 | Methods → Implementation → Model sizes，PDF 物理页 8 |
| Block GRU 与跨块 mixing | Methods → Networks，PDF 物理页 9 |
| 官方计算步骤 | [`dreamerv3/rssm.py` 的 `_core`](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/rssm.py) |
| 任务与 model scaling | Figures 4、6、Extended Data Tables 4–5 |

官方 `_core` 的只读核对看到：先分别投影 `deter`、`stoch`、`action`，把共同输入提供给各组，再使用 `BlockLinear` 计算局部 hidden updates 和 GRU gates。这与论文的“局部 recurrent update + 共同 mixing”一致。这里的 GitHub 链接是可变的 `main`，记录的是核对日期，不是冻结的 code release。

## 5. Innovation、results 与边界

论文的主要贡献是跨多种任务与数据尺度工作的 DreamerV3 整体算法，包含 normalization、loss balancing、distribution parameterization 等多项设计。原文评估覆盖超过 150 个任务。

对当前 idea 最有用的证据是：block GRU 已被用于实际 world model，实现了结构化 recurrent computation。但 benchmark 的整体成绩不能单独归因于分块；本阅读包也没有证明这个模块在 LeWM + PushT 上更快。

需要分清：

- 8 个 recurrent blocks 不等于 8 个语义对象。
- categorical latent 不等于 low-bit weight quantization。
- 想象轨迹中的 model cost 不等于部署 actor 的 latency。
- recurrent 参数减少不自动给出 native wall-clock 或完整 CEM speedup。

## 6. 对 LeWM + PushT 的阅读任务

把它视为 **结构化小 predictor 的工程先例**。阅读时寻找可借用的 update 结构，同时记录哪些信息仍由共同输入提供。若将它用于既有 LeWM latent，新的 predictor 和必要的状态映射仍需训练；此论文未提供“固定切分冻结 LeWM encoder 输出”的验证。

## 7. 分时阅读路线

### 20 分钟：只抓这次相关的机制

1. Figure 1，区分 observation、stochastic latent、recurrent memory。
2. 直接跳到 PDF 物理页 8–9，读 Model sizes 与 Networks。
3. 看官方 `_core`，圈出共同投影与 `BlockLinear`。
4. 写下“分了什么、各块还读什么、怎样组合”的三句话。

### 90 分钟：理解它在完整模型中的角色

1. World model learning 25 分钟：追踪 posterior、prior、reconstruction 和 KL。
2. Actor–critic learning 20 分钟：标出 world model 被使用的阶段。
3. Methods 与 Extended Data Tables 4–5 20 分钟：查尺寸、blocks、共同输入。
4. Figures 4、6 15 分钟：区分整体算法证据与模块证据。
5. 填写 Meeting Card 10 分钟。

### 180 分钟：形成可比较的模块说明

1. 手算等总 hidden width 下 dense GRU 与 block GRU 的 recurrent 部分参数。
2. 只读 `_core`，逐行标注 tensor 的 flat / grouped shape 与混合位置。
3. 与 [RIMs](../026-recurrent-independent-mechanisms/README.md) 比较：全 blocks 更新还是选择更新、各块是否共享参数、通信成本在哪里。
4. 写一张仅针对 predictor-level 的待验证假设卡，不把阅读当成复现结果。

## 8. Reading Questions（留给你回答）

1. `h_t` 与 `z_t` 分别保存什么信息？为什么都需要？
2. Block GRU 拆的是 encoder feature、stochastic state，还是 recurrent memory？
3. 共同输入中为何还包含完整 recurrent state？
4. Eight blocks 的 local update 在哪些地方独立，在哪些地方耦合？
5. 模型尺寸增大时，block 数与 block width 怎样变化？
6. 给定总 hidden width，哪些参数可以按 `1/K` 减少，哪些不能？
7. 相同参数量与相同 hidden width 的对照，分别回答什么问题？
8. 原文的哪些 ablations 能归因到 block GRU，哪些只能评价整体算法？
9. 各块有没有物体语义？需要哪些证据才能作这种解释？
10. 与 RIMs 的 attention communication 相比，共同 linear projection 有何取舍？
11. 使用该结构替换 LeWM predictor 时，history 与 action conditioning 怎样接入？
12. 更小参数量、较低 FLOPs、较低 native latency 是否可能给出不同排序？
13. 如何判断速度来自分块本身，而不是更小 hidden width 或不同训练目标？
14. Dreamer imagined training 的优势能否直接转移到 CEM 多候选 rollout？为什么？

## 9. Meeting Card（留空）

- 我读的精确版本：
- 被拆分的状态与每块 dimension：
- Local predictor 的输入与参数共享方式：
- 跨块通信与组合位置：
- 最强证据及原文位置：
- 参数 / FLOPs / latency 中实际测了什么：
- 与冻结 LeWM latent 的关键差异：
- 我的一个待验证假设：
- 想向导师确认的问题：
