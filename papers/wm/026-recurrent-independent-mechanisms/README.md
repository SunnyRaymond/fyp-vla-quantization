# 026. Recurrent Independent Mechanisms (RIMs)

> **本地论文：** [paper-arxiv-v6-author-manuscript.pdf](paper-arxiv-v6-author-manuscript.pdf)（35 页）
>
> **Official resources：** [arXiv v6](https://arxiv.org/abs/1909.10893v6) · [v6 HTML](https://arxiv.org/html/1909.10893v6) · [ICLR 2021 publication PDF](https://openreview.net/pdf?id=mLcmdlEUxy-) · [author code](https://github.com/anirudh9119/RIMs)
>
> **包核验：** 已核对 v6 正文与相关 appendix；本地 PDF 的 arXiv 身份、首页标题和页数已核对并目视检查。OpenReview PDF 链接未能取得 PDF，因此本地文件是 arXiv v6 作者稿，不是 ICLR proceedings 文件；没有运行或审计作者代码。**用户阅读状态：** `unread`。

## 1. Paper identity

| Field | Record |
|---|---|
| Title | *Recurrent Independent Mechanisms* |
| Authors | Anirudh Goyal, Alex Lamb, Jordan Hoffmann, Shagun Sodhani, Sergey Levine, Yoshua Bengio, Bernhard Schölkopf |
| Publication | International Conference on Learning Representations (ICLR), 2021 |
| Pinned local version | arXiv `1909.10893v6`, dated 2020-11-17; author manuscript |
| Local artifact | `paper-arxiv-v6-author-manuscript.pdf`, 35 pages |
| Public code | [`anirudh9119/RIMs`](https://github.com/anirudh9119/RIMs) |

本地 PDF 首页面标注 arXiv v6 与 2020-11-17。论文后来发表于 ICLR 2021；为避免把作者稿误当正式会议排版，目录名和 metadata 都记录了实际下载版本。

## 2. 一句话抓手

RIMs 把 recurrent hidden state 划成若干各自带参数的模块：每一步由注意力竞争挑出少数模块，只有它们读取当前输入并更新；更新后的模块再通过注意力与其他模块交换信息。作者要测试的是“默认相对独立、需要时稀疏交互”的归纳偏置能否帮助模块 specialization 与分布变化下的泛化。

这里的“independent”是默认动力学和参数化相对分开，不是永不通信，也不保证每个 RIM 自动对应一个真实物体或因果机制。RIMs 也不是把已训练 LeWM latent 按坐标切开：它是一种可训练的 recurrent computation architecture。

## 3. Problem：一个 dense recurrent state 会把不同时间过程绑在一起

许多序列同时包含多个动态过程，但它们并非每一步都互相影响。例如，球在空中各自运动，碰撞时才强交互；视频里物体与背景也可能在不同时间尺度上变化。普通 dense RNN 每一步都允许所有隐藏单元彼此影响。要靠训练把大量不相关连接恰好压到零，模型既难学出模块边界，也可能在某个过程改变时干扰其他过程。

RIMs 改变计算流程本身：不是要求 dense RNN 从数据里“碰巧学会”隔离，而是先给多个小型 recurrent block 各自的状态和转移参数，再用选择性输入与少量跨模块注意力来控制信息流。它的假设是数据确有可复用的稀疏交互结构；若过程处处强耦合，这个 inductive bias 未必合适。

## 4. Method：竞争选择 → 独立更新 → 注意力通信

令第 `k` 个 RIM 的状态为 `h_{t,k}`，自身参数为 `θ_k`（在时间上共享）。每个时间步可以按三步读：

1. **Input attention 与 top-k activation（§2.1–2.2，Fig. 1）。** 每个 RIM 从自己的当前状态产生 query，去读取当前输入中与它相关的 key/value。输入还附带一个 null 项；按对 null 项的注意力竞争，选出最相关的 `k_A` 个 RIM。只有胜出的模块读入当前 observation。空间输入可以按位置执行竞争。
2. **Independent transition（§2.3）。** 活跃 RIM 使用自己的 `GRU` 或 `LSTM` 参数，根据自己的状态和所读输入更新。未活跃 RIM 保持原状态，像暂存的 memory；论文说明梯度仍可经过未激活步骤传递。
3. **Sparse communication（§2.4）。** 活跃模块以 query 从所有模块（包括未活跃模块）的状态中读取 key/value，并通过 residual attention 得到下一状态。作者也讨论对通信 attention 再做 top-k 稀疏化。

所以 RIMs 不是“模块之间完全没有依赖”：未活跃状态仍能给活跃模块提供上下文；关键约束是默认转移分块、输入选择性读取、每步只更新部分状态。论文给出 RIM-specific transition 参数；attention 负责调度和通信。

## 5. Results：证据是特定任务的泛化与模块化行为

论文覆盖 copying、Sequential MNIST resolution shift、bouncing balls、novel distractors、视频预测和 Atari PPO 等任务。值得先读的证据：

- **Copying / Sequential MNIST（§4.1，Table 1）。** copying 的空白等待段从训练长度 50 拉长到测试长度 200 时，RIMs 保持高表现，而若干 recurrent baselines 明显退化。Sequential MNIST 从 `14×14` 训练分辨率转到不同测试分辨率时，RIMs 相比 LSTM 更能保留表现；训练与测试序列长度一致时，方法间差异小得多。
- **Bouncing balls（§4.2，Fig. 3）。** 在物体数量变化、rollout 和 occlusion 设置中，RIMs 相比 LSTM 更好地预测球的轨迹。图中展示的重点是可变物体数下的泛化，不是视觉 slot 对象分割或精确的物理因果识别。
- **Novel distractors（§4.2，Fig. 4）。** BabyAI object-picking 任务在测试时加入未见 distractors，RIMs 优于 LSTM；作者把它解释为选择性读取能减少无关输入干扰。
- **Atari PPO（§4.3，Fig. 5）。** 在 PPO 设置保持相同、只替换 recurrent architecture 的比较中，作者报告 RIMs 整体优于 LSTM，但各游戏表现不同。不要把平均趋势读成每个游戏都提升。

§4.4 的 ablation 说明 input attention 对 Atari 结果重要，且移除模块通信会损害 copying 与 Sequential MNIST；这也意味着模块化并非把各块彻底隔离就够了。

## 6. Evidence boundary 与 LeWM + PushT 的关系

- **不是现成的 latent partition。** RIMs 的模块是 recurrent hidden blocks，每块有自己的 transition parameters，并由 attention 读取输入和其他模块。它没有给出把冻结 LeWM dense latent 直接拆分后仍保持 predictor 行为的办法。
- **需要改 predictor 并重新训练。** 若把想法迁移到 LeWM + PushT，需要先定义模块读什么 latent/action、各自预测什么状态量、如何处理模块交互，再训练出这个 predictor。它超出 checkpoint 后处理或单纯 latent slicing 的范围。
- **现有结果不能替代 FYP evidence。** 论文没有 LeWM、PushT、CEM candidate-ranking、elite overlap、first-action fidelity、native latency/memory 或闭环机器人实验。fixed-observation predictor 结果、planner fidelity 与 closed-loop success 仍应分别验证。
- **稀疏路由不等于硬件加速。** 只更新一部分模块可能减少理论上的 recurrent work，但 attention/top-k 与张量实现本身有开销。论文没有建立适用于当前硬件的 wall-clock speedup、显存或能耗结论。
- **independence 是建模偏置，不是保证。** 要看模型是否真的学到稳定 specialization、改变一个因素时其他模块是否稳健，以及跨模块信息是否被用得恰当；attention map 本身不构成因果证据。

对 FYP 更稳妥的定位是：RIMs 是“按需更新的模块化 recurrent computation”先例，可以启发 predictor 结构问题；它与 LeWM 表示稀疏化、缓存和权重量化是不同路线，迁移成本和证据门槛也不同。

## 7. Reading route

### 20 分钟：抓住结构与证据边界

1. Abstract 与 Fig. 1：用自己的话写出“select → update → communicate”。
2. §1 与 §2.2–2.4：确认模块为何被选中、何时更新、信息怎样跨模块流动。
3. §4.1 copying 与 Sequential MNIST、§4.2 Fig. 3：各选一个分布变化例子。
4. 回到本 README 第 6 节，写下为什么 RIMs 不能被称作 frozen LeWM latent 的直接分块。

### 90 分钟：能复述机制并检查 ablation

1. 精读 §2.1–2.4 和 Appendix A 的四个 desiderata：competitive mechanisms、top-down attention、sparse information flow、modular computation/parameterization。
2. 对照 §4.1 Table 1、§4.2 Fig. 3–4 与 §4.3 Fig. 5，区分序列长度外推、物体变化、distractor 与 RL 表现。
3. 读 §4.4：列出 input attention 与 inter-RIM communication 的 ablation 分别说明什么。
4. 记录每个结果使用的训练/测试变化和 baseline，暂不把不同任务合并成一个泛化结论。

### 180 分钟：写成可用于 FYP 讨论的机制卡

1. 阅读 Appendix C 的 implementation/hyperparameters 与 Appendix D/E 的实验细节，核实 `k_T`、`k_A`、模块尺寸、数据输入与 rollout 评估方式。
2. 只读浏览作者 code 的 RIM cell、attention selection、communication 入口；将论文方程映射到实现，不启动训练或 benchmark。
3. 画一张数据流图，标清每步哪些模块计算、哪些状态保持、哪些 attention 连接仍可全局读取。
4. 写下一个与 LeWM + PushT 兼容的候选接口，以及在进行任何实验前必须通过的 predictor、planner、runtime 三类 gate。

## 8. Reading Questions（先留空，读完再回答）

1. §1 中的“independent mechanisms”指的是随机变量独立，还是生成这些变量的机制参数化相对独立？
2. vanilla dense RNN 若要表示 `k` 个完全独立过程，为什么需要大量 recurrent connections 为零？
3. RIM 的 query 来自哪里，输入的 key/value 又来自哪里？这与普通 Transformer self-attention 有何不同？
4. 用 null input 的 attention score 竞争选择活跃模块，有哪些好处和潜在失败模式？
5. 未激活 RIM 状态保持不变，但梯度仍可经过。这个设计对长期记忆和训练分别意味着什么？
6. 活跃模块能读所有模块的状态时，“稀疏通信”具体稀疏在哪里？哪些依赖仍然存在？
7. 模块各自参数化减少或增加了哪些参数？attention 带来的参数与计算成本如何纳入比较？
8. copying 的长空白段为什么是检验选择性更新的干净测试？它能代表多少真实任务？
9. Sequential MNIST 的结果是长度/分辨率外推证据，还是一般图像泛化证据？
10. bouncing-balls 任务中的“模块 specialization”由什么指标或可视化支持？哪些因果主张仍未证明？
11. novel distractor 实验中，partial observability 与 unseen distractor 各自改变了什么？
12. Atari PPO 是 architecture-only comparison 吗？同一 PPO 超参数控制了什么，又没有控制什么？
13. §4.4 的两个 ablation 分别支持 input selection 和 inter-module communication 的什么作用？
14. 哪类强耦合动力学可能让 top-k activation 成为错误归纳偏置？
15. 若将 RIM 用作 LeWM predictor，输入、状态、输出与通信分别怎样定义，且是否需要重训 encoder？
16. 什么实验才足以证明 RIM 风格结构既保持 PushT CEM 决策质量，又带来实际 latency/memory 收益？

## 9. Meeting Card（留空）

- **我现在的机制图：**
- **我认为最强的一条证据：**
- **一个尚未解决的混杂或替代解释：**
- **它与 LeWM + PushT 的关系：**
- **若继续读 code，我要定位的函数/模块：**
- **下次讨论时我想问：**
