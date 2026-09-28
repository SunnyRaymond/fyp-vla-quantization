# 分块 latent dynamics：8 篇自读路线

准备日期：2026-09-26。阅读主题：**高维状态能否分成多个低维向量，用更简单的模块预测后再组合？**

本专题包含 7 个新编号阅读包，以及 #007 DreamerV3 的 Nature 2025 companion。每篇都配本地原文 PDF、中文阅读笔记、分时路线、未回答的 Reading Questions 和留空 Meeting Card。原文版本与 PDF 来源在各包内记录；阅读状态保持 `unread`。

本次只准备阅读材料，没有运行训练、模型推理或 planner experiments。

## 1. 从这里打开

| 编号 | 阅读入口 | 本专题角色 | 最值得先追踪的计算 |
|---|---|---|---|
| 026 | [RIMs](026-recurrent-independent-mechanisms/README.md) | 多个 recurrent modules 分别演进 | 局部 GRU/LSTM、选择更新、attention communication |
| 007 | [DreamerV3 / Nature 2025](007-dreamerv3/READING-NATURE-2025.md) | 结构化 recurrent computation | Eight-block GRU 与共同输入 |
| 027 | [FLAM](027-factored-latent-action-world-models/README.md) | 学出 factors，逐个预测后聚合 | Factorizer、shared FDM、aggregator |
| 028 | [C-SWM](028-contrastive-structured-world-models/README.md) | Object-factored world model | 共享 node predictor 与 GNN messages |
| 029 | [Koopman / EDMD](029-koopman-edmd/README.md) | 找到动力学适配的坐标后简单演进 | Observables、eigenfunctions、modal reconstruction |
| 030 | [DMD](030-dynamic-mode-decomposition/README.md) | 高维流场的模态演进 | Snapshot operator、eigenvalues、modes |
| 031 | [Product Quantization](031-product-quantization/README.md) | 分块运算的跨领域先例 | Subvector quantization 与组合 codebook |
| 032 | [Sub-JEPA](032-sub-jepa/README.md) | LeWM 相关的子空间 regularization | Projected views 与 anti-collapse loss |

相关起点：[LeWM #001](001-leworldmodel/README.md) · [LpWM #025](025-lpwm-sparse-representations/README.md)。

## 2. 建议阅读顺序

### 第一组：先看已经实现的模块化 predictor

**RIMs → DreamerV3 Nature → FLAM → C-SWM**

- RIMs 回答：每块维护什么状态，何时更新，如何通信？
- DreamerV3 回答：如何用 block structure 降低 recurrent connectivity，同时保留共同条件？
- FLAM 回答：怎样学习 factors，并将逐个预测的结果合回 visual features？
- C-SWM 回答：对象独立表示以后，碰撞等 interactions 应放在哪个计算模块？

### 第二组：再看“先换坐标，再让 dynamics 变简单”

**Koopman / EDMD → DMD**

如果 eigenfunction coordinates 真正成立，每个坐标可以按自己的 eigenvalue 演进。这里要重点读坐标怎样获得、近似误差在哪里、如何重构，以及有 action inputs 时哪些前提需要改变。EDMD 基础论文的 autonomous dynamics 不等同于 LeWM 的 controlled dynamics。

### 第三组：作为边界对照

**PQ → Sub-JEPA**

PQ 是分块量化与检索，没有 temporal update。Sub-JEPA 在低维 views 中施加 regularization，仍用完整 latent predictor。它们有助于分清“分块计算”的不同含义。

### 2 小时首轮

1. RIMs：25 分钟，读 architecture 与更新 / communication。
2. DreamerV3：20 分钟，Figure 1 与 Nature PDF 物理页 8–9。
3. FLAM：20 分钟，Section 4.2 与 Eq. 5，追踪 FDM 实际输入。
4. C-SWM：15 分钟，object encoder 与 relational transition。
5. EDMD：25 分钟，读基础定义与 reconstruction，先不推全部证明。
6. 剩余 15 分钟填下面的比较表；DMD、PQ、Sub-JEPA 留到第二轮。

各篇的 90 / 180 分钟路线在独立 README 中。数学路线较难时，先读直觉与符号表，再沿原文 equation 定位。

## 3. 带着同一个问题比较八篇

“分块”至少可能发生在四个层次：

1. **Representation**：一个场景用多个向量表达。
2. **Update**：局部 predictor 分别产生 next state。
3. **Communication**：块间交换预测所需的信息。
4. **Readout / recombination**：拼接、attention aggregation 或 modal sum 形成整体输出。

只看到多个 vectors，还不能判断第二层是否独立，更不能判断实际 latency。

| 工作 | 分的是什么 | 局部更新还有哪些外部信息 | 重组方式 / 阅读边界 |
|---|---|---|---|
| RIMs | Learned recurrent hidden modules | Selected encoder input 与其他 module 的 attention messages | 模块共同形成 state/readout；不是直接切冻结 LeWM latent |
| DreamerV3 | Deterministic recurrent memory | Stochastic latent、action、完整 memory 的共同投影 | 拼回 recurrent state；block-diagonal 不等于完全独立 |
| FLAM | Learned factors / slots，另有 per-factor latent actions | FDM 读取所有当前 slots 与某个 factor 的 latent action | Aggregator 还读取当前 visual features；共享 predictor |
| C-SWM | Object representations | Action 与对象之间的 GNN messages | Factored latent state；交互仍需计算 |
| EDMD | Observable / eigenfunction coordinates | 基础设置为 autonomous map | Koopman modes 重构；有限字典是近似 |
| DMD | Flow-field modal coordinates | Snapshot-fitted linear operator 的假设 | Modes 的线性叠加；非线性流动不保证严格解耦 |
| PQ | 固定 subvectors | 各自 quantizer / codebook | 拼接编码；不预测下一时刻 |
| Sub-JEPA | Regularization 的 projected views | 完整 predictor 仍读取原 latent 与 action | 没有分块 prediction/reconstruction 路径 |

## 4. 两个容易混淆的推理

### “每块更少信息”与“每块更容易预测”

小块可能丢失预测自己所需的条件。例如一个状态分为位置与速度，只看位置的模块无法唯一确定下一步位置。需要学习合适的分解，或者保留通信路径。

可以将要研究的结构写成：

\[
y_t=T(z_t)=[y_t^{(1)},\ldots,y_t^{(K)}],\qquad
\hat y_{t+1}^{(i)}=f_i(y_t^{(i)},a_t,c_t^{(i)}).
\]

这里 `T` 是状态映射，`c_t^(i)` 是跨块 context。完全独立的版本不使用 `c`；其他版本使用共享 context、message passing 或 attention。重组可为拼接加逆映射，也可为 learned aggregation。这个式子是比较框架，不是本文集已验证的新算法。

### 参数、计算量与真实速度

一个 `D × D` square update 改成 `K` 个等宽独立 blocks，这部分参数和乘加量可由 `D²` 降到 `D²/K`。若转换、通信、其他层仍占主导，整体节省会不同；很多小计算也可能带来额外执行开销。

对 LeWM / LpWM，还要区分 full latent dimension、activation sparsity、block independence 和 native execution。它们是不同属性，不能互相替代。

## 5. 全专题 Reading Questions（留给你回答）

1. 我的目标是少预测一些信息，还是将同样的信息分给多个更简单的 predictor？
2. 我想固定切坐标、学习一个线性变换，还是重新训练 factorized encoder？
3. 哪些论文学习 representation，哪些只限制 recurrent connectivity？
4. 哪些方法有每块独立参数，哪些共享同一个 predictor？
5. 哪些方法的 local predictor 实际仍读取完整 state？
6. 跨块 communication 是必要的 dynamics 条件，还是可去掉的开销？
7. Snapshot 上低相关的坐标，是否具有独立的 conditional dynamics？
8. LpWM 的 exact zeros 能否告诉我哪些坐标可以一起分块？缺少什么证据？
9. EDMD 的 eigenfunction approximation 对 controlled、contact-rich dynamics 有何限制？
10. 表示分块后，目标距离可分解是否意味着动态预测也可分解？
11. 若 frozen LeWM encoder 保持不变，我必须训练哪些新模块？
12. 小 latent error、较好 multi-step prediction、candidate ranking 与实际 latency，各回答哪个问题？
13. 相同 dimension、相同参数量、相同训练预算的比较，能否得出相同结论？
14. 哪篇提供真实 systems evidence，哪篇主要提供 quality / generalization evidence？
15. 我当前最想检验的是完全独立、稀疏耦合，还是共享小 context 的版本？

## 6. 跨论文 Meeting Card（留空）

- 我的具体问题：
- 最接近的两篇与精确版本：
- 它们分块的位置和 dimension：
- Local update 输入与参数共享方式：
- 必要的跨块依赖：
- 原文已经提供的证据：
- 我的假设与现有实现的区别：
- 尚缺的 prediction / ranking / systems evidence：
- 想向导师讨论的一个问题：

## 7. 收录与版本原则

- DreamerV3 复用 `007-dreamerv3`，新增 Nature 2025 PDF 与本专题路线；原 arXiv v2 PDF、旧阅读笔记和 metadata 历史字段保留。
- 新论文各自独立编号 `026–032`，没有将多篇合成一个论文条目。
- 本地 PDF 均保留原文；作者稿、publisher version、repository copy 或公开镜像来源分别注明。
- 配套笔记说明核对范围，不把静态阅读称为代码复现或 benchmark。
- SCOFF、WM3C、SlotFormer 等相关扩展不在本次 8 篇交付范围中，避免改变本次阅读量。
