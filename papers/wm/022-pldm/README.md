# 022. PLDM - Learning from Reward-Free Offline Data

> **完整标题：** *Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models*
>
> **本地论文：** [paper-arxiv-v4.pdf](paper-arxiv-v4.pdf)
>
> **Official resources：** [arXiv:2502.14819v4](https://arxiv.org/abs/2502.14819v4) · [project page](https://latent-planning.github.io/) · [official code](https://github.com/vladisai/PLDM)
>
> **建议先修：** `offline RL`、`goal-conditioned RL`、`JEPA`、`VICReg`、`Model Predictive Control (MPC)`、`MPPI`。
>
> **阅读状态：** `verified-full-text`；表示本地 PDF 与阅读入口已核对，不表示本地复现了训练或 benchmark。

## 1. Paper identity 与 LeWM 引用边界

| Field | Record |
|---|---|
| Authors | Vlad Sobal, Wancong Zhang, Kyunghyun Cho, Randall Balestriero, Tim G. J. Rudner, Yann LeCun |
| arXiv | `2502.14819`；v1 2025-02-20，当前本地固定为 v4 2025-10-29 |
| Venue / status | 本地 v4 首页标注 `39th Conference on Neural Information Processing Systems (NeurIPS 2025)` |
| Local artifact | arXiv v4，30 pages，3,846,356 bytes |
| Code snapshot checked | official repository `main` HEAD `1bd7e564ecd961205bc18b23067b19e9ca24ac90`（2026-09-19 轻量核对，不是本地 reproduction） |

[`#001 LeWorldModel`](../001-leworldmodel/README.md) 实验部分的 `PLDM [22]` 指的就是本篇。LeWM bibliography 使用早期标题 *Stress-testing Offline Reward-Free Reinforcement Learning: A Case for Planning with Latent Dynamics Models*，而当前 arXiv v4 的正式标题是上面的 *Learning from Reward-Free Offline Data...*。`PLDM` 是论文中提出的方法简称 `Planning with a Latent Dynamics Model`，不是另一篇名为 “PLDM” 的论文。

本篇原始实验主要是 navigation；LeWM 后来把 PLDM 当成 end-to-end JEPA baseline，重新放进 `Two-Room / Reacher / PushT / OGBench-Cube` 的比较。阅读时要分开：**本篇作者报告的原始证据**与**LeWM 对 PLDM 的 downstream reimplementation evidence**。

## 2. 一句话结论

在 reward-free offline trajectories 上，PLDM 不直接学习一个固定 goal-conditioned policy，而是先用 reconstruction-free JEPA objective 学 `encoder + action-conditioned latent dynamics`，测试时再用 MPPI/MPC 搜索 actions；作者在 23 个 navigation datasets 上发现它对 data quantity/quality、new task 和 unseen layouts 的覆盖最均衡，但代价是更高的 inference-time planning compute，而且论文没有验证 manipulation、partial observability 或 real robot。

## 3. Problem：为什么不只训练一个 offline policy

输入是没有 reward label 的 offline state-action sequences：

```text
(s0, a0, s1, ..., aT-1, sT)
```

用户可能在 test time 才给出 goal，甚至换成与 goal-reaching 不同的 task。论文比较两类路线：

- **model-free reward-free offline RL：** `GCIQL`、`HIQL`、`HILP`、`CRL`、`GCBC`，训练后直接输出 action；
- **optimal control with a learned model：** PLDM 先学习 dynamics，test time 按当前 objective 搜索 action sequence。

核心问题不是“哪种方法在一个干净 dataset 上最高”，而是面对 short trajectories、random-policy data、small datasets、new tasks 和 unseen layouts 时，哪一种 learning paradigm 仍能工作。

## 4. Method

### 4.1 Reconstruction-free latent dynamics

Encoder 把 observation 映射到 latent：

$$
z_0 = h_\theta(s_0).
$$

一个或多个 action-conditioned predictors 自回归预测未来 latent：

$$
\hat z_t^k = f_\theta^k(\hat z_{t-1}^k, a_{t-1}).
$$

训练首先最小化 predicted latent 与 encoded future latent 的 squared distance。它不以 pixel reconstruction 为主目标，因此 `PLDM` 不是 `latent diffusion model`，缩写中的 `P` 是 `Planning`。

### 4.2 Collapse prevention

arXiv v4 Appendix D.1 给出的训练 objective 包含：

- `Lsim`：prediction-to-target latent similarity；
- `Lvar`：VICReg-style per-feature variance floor；
- `Lcov`：抑制 redundant correlated dimensions；
- `Ltime-sim`：temporal smoothness；
- `LIDM`：inverse dynamics modeling，从相邻 latents 预测 action。

最关键的 ablation（local PDF p. 20）显示，去掉 `Lvar` 或 `Lcov` 会让 Two-Rooms / Diverse Maze success 大幅下降；去掉 `LIDM` 对 Two-Rooms 几乎无影响，但 Diverse Maze 从约 `98.7%` 降到 `75.5%`。因此不能把 PLDM 简化成“只做 future-latent MSE”。

### 4.3 Planning and uncertainty penalty

给定 current observation 与 goal observation，PLDM 用 ensemble predictors rollout candidates，并优化：

$$
C(a) = C_{goal}(a) + \beta C_{uncertainty}(a).
$$

- `Cgoal` 累加 predicted latents 到 goal embedding 的距离；
- `Cuncertainty` 用 ensemble prediction variance 惩罚疑似 out-of-distribution transitions；
- optimizer 是 `MPPI`；
- 外层采用 `MPC`，默认每一步 replan。

新 task 不一定需要重训 encoder/dynamics。例如 chasing task 直接把 goal-distance cost 的符号反转，从“靠近某状态”变成“远离某状态”。这说明 task flexibility 来自 test-time objective，不代表模型自动理解任意 language instruction。

## 5. Experiments：先看哪组证据

### 5.1 Scope

- 23 datasets，围绕 `Two-Rooms`、`Diverse PointMaze`、`Ant-U-Maze`；
- 6 个主要 methods：`CRL`、`GCBC`、`GCIQL`、`HILP`、`HIQL`、`PLDM`；
- stress axes：data size、trajectory length、random-policy fraction、trajectory coverage、new task、unseen layout；
- 多数 main results 使用 3 seeds；selected settings 扩到 10 seeds，appendix 另做 Welch's t-test。

### 5.2 In-distribution 不是主要胜点

在 Two-Rooms 的 large/high-quality dataset 上，多种方法都接近满分。Table 2 中 PLDM 为 `97.8 ± 0.7%`，HILP 为 `100.0 ± 0.0%`，GCIQL 为 `98.0 ± 0.9%`。所以论文的强 claim 不是“PLDM 在所有条件都最高”，而是它在不同 data/generalization conditions 下较少完全失效。

### 5.3 Trajectory stitching 有明确反例

当训练数据完全没有 trajectory 穿过两房间的门时，PLDM 降到 `34.4 ± 2.7%`；GCIQL 与 HILP 仍接近 `100%`。这说明 learned dynamics + planning 并不自动解决 coverage gap，Table 1 也只把 PLDM 的 stitching 评为中等。

在 Ant-U-Maze 的 short-trajectory setting，正文报告 PLDM、HIQL、HILP 达到 `100%`，而其他 baselines 在更短 trajectories 下失败。应把这个结论限制在 state-based Ant navigation 和作者给定的 data/evaluation protocol。

### 5.4 最有辨识度的证据：unseen layouts 与 new task

- `Diverse PointMaze`：从 5/10/20/40 个 training layouts 学习，再测 held-out layouts。PLDM 是随 layout edit distance 增大仍最稳定的方法；single fixed layout 上所有方法都约 100%，差异主要来自 generalization。
- chasing task：冻结已训练模型，只改变 planning objective。PLDM 比 HILP 更能维持与 chaser 的距离。

这两组更直接支撑“test-time planning 能利用新 observation/layout/objective”，但还不能外推到 manipulation 或 partially observable environments。

### 5.5 Inference-time compute

Two-Rooms 每个 200-step episode 的作者测量（local PDF p. 23）：

| Method | Replan interval | Time / episode | Normalized PLDM success |
|---|---:|---:|---:|
| PLDM | 1 | `16.0 ± 0.13 s` | `1.00` |
| PLDM | 4 | `4.8 ± 0.09 s` | `0.95` |
| PLDM | 16 | `2.6 ± 0.07 s` | `0.90` |
| PLDM | 32 | `2.2 ± 0.07 s` | `0.62` |
| GCIQL | policy forward | `3.6 ± 0.10 s` | - |
| HIQL | policy forward | `4.0 ± 0.08 s` | - |

默认 PLDM 大约慢 4 倍；降低 replanning frequency 可以接近 policy latency，但不是零损失，也不是硬件无关的 universal speed claim。

## 6. 与 LeWM / DINO-WM 的最短对照

| Dimension | PLDM `#022` | LeWM `#001` | DINO-WM `#002` |
|---|---|---|---|
| Encoder | end-to-end | end-to-end | frozen DINOv2 |
| Collapse control | VICReg-style variance/covariance + temporal/IDM terms | SIGReg Gaussian regularization | frozen pretrained representation |
| Dynamics | action-conditioned latent predictor | action-conditioned latent predictor | predictor on pretrained patch features |
| Original paper domain | navigation stress tests | navigation + 2D/3D manipulation/control | visual planning benchmarks |
| Test-time solver | MPPI + MPC | CEM + MPC | CEM + MPC |
| Main trade-off | broad generalization, expensive planning | simpler objective, compact latent | strong pretrained visual features, not end-to-end |

**Version mismatch to audit.** PLDM v4 Appendix D.1 展示五类 loss components；LeWM Appendix C.2 把其 PLDM baseline 写成 `prediction + var + cov + time-sim + time-var + time-cov + IDM` 七项。可能涉及 earlier implementation、baseline adaptation 或 counting convention。阅读时应直接对照 LeWM code/config，不要把两个公式当作同一已确认 implementation。

## 7. Limitations 与 claim boundary

### Authors 明确写出的限制

- 全部 experiments 都是 navigation；没有 robot manipulation 或 partially observable settings；
- 默认 PLDM inference 约比 model-free baselines 慢 4 倍；
- long-horizon planning 会受 accumulated model error 影响；
- 论文给出的总研究成本估计为 `500-2000 GPU days`，不是复现单个核心结果的最小预算。

### 阅读时还要注意

- “reward-free”指 training trajectories 没有 reward annotation；test time 仍需要可计算的 cost/objective。
- `Cuncertainty` 是 ensemble disagreement proxy，不是 calibrated epistemic uncertainty 保证。
- Welch's t-test 的 pooled analysis 把不同 settings 合并；selected 10-seed tests 更容易解释，但仍不替代 per-environment effect sizes 与 confidence intervals。
- DreamerV3 / TD-MPC2 comparisons 是经过删改 reward/policy components 后的 adaptation；论文自己承认 DreamerV3 comparison 有设计不匹配，不能写成对原方法的全面否定。
- original PLDM evidence 不含 LeWM 的 PushT/OGBench results；那是后续论文的 adapted baseline evidence。

## 8. 对 FYP 的价值

PLDM 是 LeWM 与 DINO-WM 之间很重要的 comparison anchor：它保留 end-to-end representation learning，但 collapse control 比 LeWM 更复杂，也不像 DINO-WM 那样依赖 frozen foundation encoder。

若考虑 quantization，优先检查三条路径：

1. encoder quantization 是否改变 latent geometry 与 goal-distance ranking；
2. autoregressive predictor quantization error 是否随 planning horizon 累积；
3. ensemble members 的 correlated quantization error 是否让 `Cuncertainty` 虚假降低。

公平的 paired protocol 至少固定 checkpoint、dataset、MPPI candidate budget、horizon、replan interval、seeds 与 hardware；同时记录 candidate cost/ranking、first action、closed-loop success、latency 和 memory。只测 latent MSE 不足以判断 planning 是否保持。

## 9. 分时阅读路线

### 20-minute route

1. Abstract + Figure 1：分清 model-free RL 与 PLDM。
2. Section 3.3：只抓 `encoder -> predictor ensemble -> MPPI/MPC`。
3. Table 1 + Table 2：找“覆盖广”与“并非总是最好”的证据。
4. Section 4.8：看 unseen-layout generalization。
5. Appendix F + Limitations：记录 planning compute 与 domain boundary。

### 90-minute route

1. 完成 20-minute route。
2. 手写 Equations 1-7 的 data flow，标出 train-only 与 test-time components。
3. 对照 Appendix D.1 的 loss 与 ablation，判断每项的 evidence。
4. 逐项读 Sections 4.3-4.8，为 Table 1 每颗星找原始 experiment。
5. 审计 stitching 反例：为什么 no-door data 下 GCIQL/HILP 明显高于 PLDM？
6. 用 Appendix F 重算 latency-performance trade-off。
7. 填写 Meeting Card，但暂不看下面 Reading Questions 的任何外部答案。

### 3-hour deep route

1. 完成 90-minute route。
2. 对照 official code，定位 loss coefficients、ensemble size、MPPI budget 与 replanning interval。
3. 对照 [`#001 LeWM`](../001-leworldmodel/README.md) Appendix C.2，追踪五项/七项 objective discrepancy。
4. 对照 [`#002 DINO-WM`](../002-dino-wm/README.md)，画出三者 encoder、predictor、planner 的 matched/not-matched components。
5. 写一个最小 quantization experiment contract，明确哪些 metric 能证伪“planning-preserving”。

## 10. Reading Questions（读完自己回答）

1. `reward-free offline RL` 在本文中不使用 reward 的阶段到底有哪些？test-time cost 从哪里来？
2. 为什么作者把 PLDM 归为 optimal control，而不是 model-based RL？
3. `Lvar`、`Lcov`、`Ltime-sim`、`LIDM` 分别阻止哪类 degenerate representation？
4. 为什么去掉 `LIDM` 对 Two-Rooms 几乎无影响，却明显伤害 Diverse Maze？
5. ensemble disagreement 在什么情况下会低估真实 model error？
6. `Cgoal` 对整个 rollout 累加 goal distance，与只看 terminal latent 的 cost 有什么差别？
7. MPPI candidate budget、planning horizon 和 replan interval 如何共同决定 latency 与 success？
8. short trajectories 与 no-door-coverage 是两种怎样不同的 stitching challenge？
9. 为什么 GCIQL/HILP 在 no-door setting 优于 PLDM？这反驳了哪种过强 claim？
10. unseen-layout result 的 independent experimental unit 是 seed、layout 还是 trial？
11. map edit distance 是否充分刻画 visual/dynamics distribution shift？
12. chasing task 只改变 cost sign，哪些 dynamics assumptions 仍保持不变？
13. DreamerV3 / TD-MPC2 adaptation 与其原始方法有哪些不匹配？
14. pooled Welch's t-test 会隐藏哪些跨-setting heterogeneity？
15. PLDM v4 与 LeWM baseline 的五项/七项 objective 差异来自 version、实现还是 counting convention？
16. 如果 predictor 被 INT8/INT4 quantize，首先应该检查 average latent MSE、candidate ranking 还是 first action？为什么？

## 11. Meeting Card（读后填写）

- **Problem:**
- **Why model-free baselines are insufficient here:**
- **PLDM mechanism:**
- **Strongest evidence:**
- **Strongest counterexample / limitation:**
- **LeWM 使用它作为 baseline 的原因:**
- **与 DINO-WM 的一句区别:**
- **FYP quantization hypothesis:**
- **Question for 李老师:**

## 12. Evidence boundary

本地 PDF 已核对标题、版本、页数、加密状态，并抽查首页、方法、结果、latency 与 limitations 页面。论文数字均是 authors' reported evidence，本次没有运行 PLDM training、MPPI evaluation 或 GPU benchmark。Semantic Scholar 检索本次受到 HTTP 429，OpenReview connector 未安装；身份与版本以 LeWM 本地参考文献、arXiv 官方记录、project page 和 official repository 交叉确认。
