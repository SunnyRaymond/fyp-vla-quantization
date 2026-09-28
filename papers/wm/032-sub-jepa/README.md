# 032. Sub-JEPA: Subspace Gaussian Regularization for Stable End-to-End World Models

[本地 PDF](paper-arxiv-v1.pdf) · [arXiv v1](https://arxiv.org/abs/2605.09241v1) · [原文 HTML](https://arxiv.org/html/2605.09241v1) · [作者代码](https://github.com/intcomp/sub-jepa) · [专题总路线](../READING-FACTORED-DYNAMICS.md)

## 1. Paper identity

| 项目 | 内容 |
|---|---|
| Authors | Kai Zhao, Dongliang Nie, Yuchen Lin, Zhehan Luo, Yixiao Gu, Deng-Ping Fan, Dan Zeng |
| 版本 / 日期 | arXiv:2605.09241v1，2026-05-10 |
| Publication status | 本包固定为 arXiv preprint；不据此宣称 conference acceptance |
| Local artifact | 10 页，未加密，来自版本固定的 arXiv PDF URL |
| 核对日期 / 阅读状态 | 2026-09-26 / `unread` |
| 核对范围 | 原文方法、实验设置、结果表与本地 PDF identity；未安装或运行代码 |

## 2. 一句话抓手

Sub-JEPA 把 LeWM 的 Gaussian regularization 移到多个低维 projected views。**它分解的是 regularization 的计算空间；未来状态仍由同一个完整 latent predictor 预测。** 因此这篇适合研究 representation geometry，与“很多低维 predictor 分别演进后拼回去”需要明确区分。

## 3. 背景与问题

LeWM 同时训练 encoder 和 action-conditioned predictor，除 next-latent prediction loss 外，还用 Gaussian regularizer 防止 collapse。作者提出：对整个 ambient latent 强行施加 isotropic Gaussian prior，可能不适合低 intrinsic dimension 的任务。

先区分三个概念：

- **Ambient dimension `D`**：向量有多少坐标；本文仍使用 `D=192`。
- **Subspace dimension `d_s`**：一次投影后有多少坐标。
- **Effective rank**：数据在这些坐标中实际利用多少主方向的一种统计量。

降低 effective rank 不等于缩短向量，也不直接减少 predictor 的输入或输出宽度。

## 4. Method：哪些模块变了，哪些没变

原来的 prediction path 保持为：

\[
z_t=f(o_t),\qquad \hat z_{t+1}=P(z_t,a_t).
\]

新增的是训练时的 regularization branch。对每个 `k`，用固定矩阵 `P_k ∈ R^{d_s×D}` 得到：

\[
z_t^{(k)}=P_k z_t.
\]

在每个 projected view 上采样随机方向，计算 Epps–Pulley Gaussian regularization，再对 views 与方向平均。矩阵通过随机初始化和 QR 得到 row-orthonormal rows，并保持 frozen。

**没有 `K` 个 next-state predictors，没有由预测子向量重构 full latent 的步骤。** Figure 1 与 Sections 3.1–3.3 把完整 prediction path 和 subspace regularization branch 分开；Section 4.1 明确 encoder、predictor 与训练设置沿用 LeWM，只替换 regularizer。

注意 `P_k P_k^T=I` 描述单个投影矩阵的 row orthonormality。不要据此自动认定不同 `P_k` 的子空间彼此不重叠，更不要把几何正交当成 temporal dynamics 的条件独立。

## 5. Innovation 与结果：带着条件读

核心变化是 regularization prior 的作用位置，而不是新建一个更小的 inference predictor。

| 原文证据 | 应读出的范围 |
|---|---|
| Section 4.1 | `D=192`；PushT 选 `K=16`，其他三环境选 `K=32`；配置通过 held-out validation ablation 选择 |
| Table 1 | 作者报告六个 seeds 的 LeWM/Sub-JEPA 比较；PLDM、DINO-WM 数字引用自 LeWM 文献，未在该实验内重新训练比较 |
| Table 2 | `K` 的效果依任务而变；在 PushT 中过多 subspaces 会明显降低 reported success |
| Figure 2 / Section 4.2.2 | 用 evaluation observations 的 centered covariance spectrum 测 effective rank，并观察与 planning gain 的对应关系 |
| Sections 4.4–4.7 | Physical-state probing、latent trajectories、path straightening 与 open-loop visualization |

Table 1 的 PushT：LeWM 为 `84.67 ± 6.53%`，Sub-JEPA 为 `89.00 ± 5.33%`。这是作者所用 protocol 下的报告，不是本项目复现结果，也不能直接替换本项目已有 frozen benchmark 数字。

## 6. Limitations 与当前 idea 的边界

- Predictor 仍是 full-width；本文不提供分块 inference 或分块 wall-clock 加速证据。
- Frozen random projections 不等于可独立演进的 learned dynamics factors。
- Effective rank 与规划提升的对应关系是所测环境中的经验现象，不能单独证明因果机制。
- 四个环境中的 gains 与 K-selection 不能保证迁移到任意 representation 或任务。
- 引用的 PLDM/DINO-WM 与本次重训 LeWM/Sub-JEPA 的 provenance 不同，需要分开读。

对 LeWM + PushT，它是直接相关的 **retraining-required representation baseline**。若今后把它与 block predictor 配合，仍需要另外证明子空间适合局部预测；本阅读包只准备材料。

## 7. 分时阅读路线

### 20 分钟：确认它究竟拆了什么

1. Abstract 与 Figure 1：圈出 prediction branch 和 regularization branch。
2. Sections 3.1–3.3：列出 `z_t`、`P_k`、`d_s`、`K` 的 shape。
3. Section 4.1：确认 inference predictor 没变。
4. Table 2：只看 PushT 列，记录过度分解的情况。

### 90 分钟：理解方法与证据

1. Sections 1–2 15 分钟：理解作者为什么质疑 full-space Gaussian prior。
2. Section 3 25 分钟：从投影写到 regularization loss。
3. Sections 4.1–4.3 25 分钟：检查 baseline provenance、seeds、K 的选择。
4. Sections 4.4–4.7 15 分钟：区分 probing、geometry 和 prediction evidence。
5. Meeting Card 10 分钟。

### 180 分钟：比较 representation 与 dynamics 分解

1. 与 [LeWM](../001-leworldmodel/README.md)、[LpWM](../025-lpwm-sparse-representations/README.md) 对照 regularizer 与输出 link。
2. 只读作者代码定位 projection construction 和 prediction call，不运行训练。
3. 检查不同投影之间的关系，并写出什么条件才允许独立 block prediction。
4. 与 [RIMs](../026-recurrent-independent-mechanisms/README.md) 的分块更新逐项比较。
5. 整理作者已证明的结果、你的推测和仍缺的 evidence，保持三者分开。

## 8. Reading Questions（留给你回答）

1. Ambient dimension、intrinsic dimension、effective rank 有何不同？
2. Figure 1 中哪条路径参与 inference，哪条只影响训练？
3. 单个 `P_k` 的 row orthonormality 能保证不同投影空间互相正交吗？
4. 各个 projected views 的坐标是否共同构成完整、可逆的状态分解？
5. `K=1` 的设置与原 LeWM 是否在数学和实现上完全相同？
6. Frozen projection 的收益来自什么，learnable projection 的失效可能是什么？
7. 为什么 `D=192` 保持不变，却可能降低 effective rank？
8. PushT 在较大 K 时变差，可能与哪些动态信息需求有关？
9. K 是怎样选择的？validation 与最终 evaluation 有什么区别？
10. 六个 seeds 对哪些方法是实际运行，对哪些只是引用文献？
11. Table 1 的误差条是否足以支持你想讨论的 gain？还需什么统计信息？
12. Rank 与 success 的相关关系能否识别 causal mechanism？
13. Linear probes、UMAP visualization 与 open-loop rollout 各验证什么？
14. 如果想加入低维 predictor，需新增哪些训练目标或 interaction 路径？
15. Gaussian-looking coordinates 是否意味着其 dynamics 可以独立预测？
16. 与 LpWM 相比，两者改变的是 latent distribution、sparsity，还是 inference structure？

## 9. Meeting Card（留空）

- 精确版本与任务：
- 本文真正分解的对象：
- Predictor 输入 / 输出与 inference 路径：
- Subspace construction 与跨空间关系：
- 最强结果及 protocol：
- 一个作者证据未覆盖的假设：
- 与我的分块 predictor idea 的重合与差异：
- 一个想向导师确认的问题：
