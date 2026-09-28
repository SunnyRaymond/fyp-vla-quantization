# 028. Contrastive Learning of Structured World Models

> **本地论文：** [paper-arxiv-v1.pdf](paper-arxiv-v1.pdf)（arXiv author version，21 页）
>
> **来源与代码：** [arXiv v1 HTML](https://arxiv.org/html/1911.12247v1) · [arXiv v1 PDF](https://arxiv.org/pdf/1911.12247v1) · [OpenReview paper ID](https://openreview.net/forum?id=H1gax6VtDB) · [作者代码](https://github.com/tkipf/c-swm)
>
> **发表状态：** ICLR 2020。OpenReview ID 对应的索引附件标题为本文，但 OpenReview 当前要求浏览器验证；本地固定为可公开取得的 arXiv `1911.12247v1`，不是声称为 OpenReview 最终版。
>
> **阅读状态：** `unread`。已按首面 title/authors 和页数核对本地 PDF，并依据 arXiv 正文核对 method、实验与限制；未运行代码或复现结果。

## 1. Paper identity

| Field | Record |
|---|---|
| Title | *Contrastive Learning of Structured World Models* |
| Authors | Thomas Kipf, Elise van der Pol, Max Welling |
| Venue / year | International Conference on Learning Representations (ICLR), 2020 |
| arXiv version | `1911.12247v1`，2019-11-27 |
| Local artifact | 21-page arXiv v1 author manuscript；首面作者与标题已匹配 |
| OpenReview | Paper ID `H1gax6VtDB`；正式 PDF 链接当前被 browser challenge 拦截 |

## 2. 一句话抓手

C-SWM 把一帧图像编码成一组 object-slot vectors，再用共享参数的 MLP 和 graph neural network（GNN）预测各 slot 的 latent update，并以 contrastive loss 拉近真实后继、推远负样本。**Slot 是 object-oriented representation 的归纳偏置，不代表各物体 dynamics 被硬性隔离：GNN 明确把其他 slot 的 message 汇入当前 slot。**

## 3. 背景与问题

传统 pixel reconstruction world model 可能把容量用在背景细节，也可能忽略影响未来的小目标。作者改以离线 experience buffer 中的 `(state, action, next state)` 为监督信号，在 latent space 区分真实 transition 与被扰动的负样本。多物体表示希望带来对新物体位置组合的 compositional generalization，同时无需逐像素 decoder loss。

## 4. Method：从图像到交互式 object slots

1. **抽取与编码。** CNN 输出 `K` 张 feature maps，作为候选 object masks；同一个 MLP encoder 分别把每张 map 编成 `zᵏ`。每个 slot 有自己的向量，object encoder 参数跨 slot 共享。
2. **关系与更新。** 对每个有向 slot pair 计算 `e(i,j)=f_edge([zᵢ,zⱼ])`；再计算 `Δzⱼ=f_node([zⱼ,aⱼ,Σᵢ≠ⱼ e(i,j)])`，并作 residual update `z⁺ⱼ=zⱼ+Δzⱼ`。Edge/node MLP 参数分别在所有 pair/node 之间共享。论文用一轮 message passing 和 fully connected scene graph，pairwise 部分随 `K²` 增长；消息汇总让 slot 间可以相互作用。
3. **Contrastive objective。** 正样本能量按 object slots 平均，衡量预测的下一时刻 latent 与真实下一时刻 latent 的距离；负样本将随机 buffer observation 编码后与真实后继比较，并用 margin hinge loss 拉开距离。动作按 slot 输入，离散动作常用 one-hot；论文指出其他动作表示也可用。

这套拆分提供 object slots、共享权重和关系建模的 bias；它不保证 mask 一定对应人类语义，也不强制不同对象可独立演化。

## 5. 实验、贡献与限制

论文在两个 grid-world block-pushing 环境、Atari Pong / Space Invaders 和三体物理模拟中评估。核心评估是 latent rollout 的 ranking metrics（H@1、MRR），覆盖 1、5、10-step prediction；主表报告 hold-out environment instances 上 4 runs 的 mean 与 standard error。定性结果展示了从像素中学到的 object-like masks 以及对象 latent transition graphs。对 block-pushing grid worlds，object factorization、GNN 与 contrastive objective 的组合支持未见位置组合上的多步预测；Atari 与 physics 结果则显示不同环境和 slot 数下的表现并不一致。

证据边界：这些是特定环境中的 representation / latent-prediction 结果，**不是 LeWM、PushT、CEM planner 或 closed-loop manipulation 成功证据**。作者列出的主要限制是：简单 feed-forward extractor 难以消歧同类物体；模型不处理 stochastic transitions；采用 Markov 单步 state/action 假设，没有 memory，因此部分可观测或需历史的任务需要另加机制。Fully connected message passing 的 pairwise cost 也是扩展 slot 数时要留意的实现因素。

## 6. 与 LeWM + PushT 的关系

它可作为 object-centric dynamics 的 prior art：把 pusher、T 形物体及其接触关系放进多个 latent slots，并通过 GNN 交换 interaction messages。PushT 的接触耦合尤其说明**“每个物体单独编码”不等于“每个物体独立预测”**。但 C-SWM 从图像端学习 slot extractor 与 transition，LeWM 使用其自身 learned latent；要尝试迁移通常涉及重新训练或显式改造表示接口。此文没有在 LeWM + PushT 上评估，也没有证明 object slots 会提升 CEM ranking、首动作或任务成功率。

## 7. Reading route

### 20 分钟：抓住表示和 interaction

1. Abstract、Introduction 与 Figure 1（PDF p.1、p.3）：确认问题设定和四块模型。
2. Section 2.3（PDF pp.3–4）：沿 edge message、node update、residual update 读一遍，标出跨 slot 通信发生在哪里。
3. Figure 3（PDF p.7）：看 masks 与 grid latent transition；避免把可视化理解成普遍语义保证。

### 90 分钟：理解 objective 与 evidence

1. Sections 2.1–2.3（PDF pp.2–4）：从离线 tuple 到 state-action-state energy，再到 object-wise loss。
2. Sections 4.1–4.5（PDF pp.5–6）：检查 benchmark、ranking metric、baseline 和训练/测试拆分。
3. Table 1（PDF p.8）连同 Figure 3–4（PDF pp.7–8）：区分 block-world 的多步 ranking 与 physics 定性结果。
4. Section 4.7（PDF p.9）：逐条记下 instance disambiguation、stochasticity 与 Markov/memory 限制。

### 180 分钟：形成可用于 FYP 的 prior-art 记录

1. 复核 Figure 5–12 与 Tables 2–4（PDF pp.14–17），追踪 object masks、pixel-loss comparison、loss variants 与 feature-map ablation。
2. 对照作者代码，只读定位 object extractor、shared encoders、message-passing update、negative sampling 和训练配置；不运行环境。
3. 若讨论 PushT 迁移，单独列出 observation history、slot assignment、contact interaction、action encoding、predictor cost、CEM score/rank 与 closed-loop task success 的验证缺口。

## 8. Reading Questions（留待阅读后回答）

1. CNN 的 `K` 张 maps 如何竞争或分配到 object slots？哪些视觉条件会让 mask 变成纹理/区域而非物体？
2. Shared object encoder 给 permutation symmetry 带来什么好处，又留下什么 slot-binding 问题？
3. Edge MLP 只看 `[zᵢ,zⱼ]`，而 action 进入 node update；这对 action-dependent contact interaction 有什么影响？
4. 汇总所有 incoming messages 再更新 node，丢掉了哪些 pairwise 身份信息？
5. Residual latent update `z+ = z + Δz` 对 transition function 是怎样的归纳偏置？
6. 负样本来自随机 buffer state 时，哪些“容易区分”的错误会主导 contrastive loss？
7. H@1 / MRR 的 latent ranking 与像素预测质量、policy quality 或 planning success 分别有什么关系？
8. 多步 ranking 随 horizon 变化，能否判断错误来自 encoder、单步 transition 还是误差累积？
9. `K` 设得多于场景中物体数时，空 slot 如何影响 message passing 和 loss？
10. 同类目标重叠或穿越时，简单 extractor 为什么无法稳定地 disambiguate instance？
11. 对 stochastic observations / transitions，怎样的 probabilistic latent 或 memory 扩展最小且可识别？
12. PushT 的 pusher 与 T-shape 若编码成 slots，哪一种接触信号会跨 slot 传递？如何验证不是 slot identity 偶然性？
13. `O(K²)` 的 fully connected graph 在稀疏物理接触环境里能否安全稀疏化？需要保留哪种长程作用？
14. 要把它与 LeWM + PushT 公平比较，哪些 observation、history、planner 和闭环指标必须固定？

## 9. Meeting Card（空白）

- **Paper / version：**
- **Core idea：**
- **最强证据：**
- **关键限制：**
- **与 LeWM + PushT 的关系：**
- **待讨论问题：**

## 10. Sources and code

- [arXiv v1 HTML（正文及 appendix）](https://arxiv.org/html/1911.12247v1)
- [arXiv v1 PDF](https://arxiv.org/pdf/1911.12247v1) · 本地固定版本：[paper-arxiv-v1.pdf](paper-arxiv-v1.pdf)
- [OpenReview record / paper ID `H1gax6VtDB`](https://openreview.net/forum?id=H1gax6VtDB)
- [作者官方代码：tkipf/c-swm](https://github.com/tkipf/c-swm)
