# 025. LpWM: A Case for Sparse Representations in World Models

> **本地论文：** [paper-arxiv-v1.pdf](paper-arxiv-v1.pdf)
>
> **Official resources：** [arXiv:2608.22764v1](https://arxiv.org/abs/2608.22764v1) · [HTML](https://arxiv.org/html/2608.22764v1) · [official code](https://github.com/YilunKuang/lpworldmodel) · [author publication page](https://yilunkuang.github.io/publications/)
>
> **建议先修：** [#011 LeWM reproduction](../011-lewm-reproduction/README.md)、[#022 PLDM](../022-pldm/README.md)、JEPA、SIGReg / distribution matching、CEM / MPC、LTI / LTV dynamics。
>
> **阅读状态：** `verified-full-text`；已核对 arXiv v1 正文、appendix、本地 PDF 与官方代码入口。未安装环境、下载 dataset/checkpoint、运行 training/planning/benchmark，也未做完整 code audit。

## 1. Paper identity

| Field | Record |
|---|---|
| Authors | Yilun Kuang, Yash Dagade, Quentin Le Lidec, Lucas Maes, Randall Balestriero, Yann LeCun |
| Affiliations | NYU, AMI Labs, Duke University, Mila, Brown University |
| arXiv | `2608.22764v1`，submitted 2026-08-24 |
| Local artifact | arXiv v1，28 pages，1,614,918 bytes，unencrypted |
| Venue / status | PDF 标注 `Preprint`；author publication page 列为 WM@Booth 2026 Oral |
| Public code | `YilunKuang/lpworldmodel`；本次记录的 `main` HEAD 为 `bdd812d9432cccda8c350086006401b436f91982`（2026-09-22） |

这篇 paper 有三条需要分开追踪的 claim：

1. **theory：** sufficiently high-dimensional one-hot encoding 可以把 Lipschitz controlled dynamics 近似成 action-conditioned linear latent dynamics；
2. **planning：** learned distributed sparse codes 在 PushT 的 intermediate-capacity predictors 上比 dense LeWM 更容易规划；
3. **interpretability：** sparse support 有时对应 discrete dynamics regime，magnitude 则编码 within-regime continuous state。

第一条不是第二条的直接证明，第三条也不是 vanilla sparsity 自动保证的性质。

## 2. 一句话抓手

LeWM 用 dense Gaussian-like latent；LpWM 把 anti-collapse target 改为 Rectified Laplace，并在 encoder / predictor 输出端使用 `(Rep)ReLU` 产生 non-negative、exactly sparse latent。论文的核心主张不是“zero coordinates 自动带来 sparse-kernel speedup”，而是：**在同一 PushT planning task 上，稀疏表示让较弱的 dynamics predictor 也能成功规划。**

## 3. Problem：为什么 representation geometry 可能决定 predictor complexity

End-to-end JEPA 同时学习 encoder 与 action-conditioned predictor：

$$
z_{t+1}=f_\theta(x_{t+1}),\qquad \hat z_{t+1}=g_\phi(z_t,a_t).
$$

Anti-collapse regularization 不只决定 latent 是否退化，也决定 latent 的 geometry。LeWM 的 SIGReg 以 isotropic Gaussian 为 target，几乎所有坐标都非零。LpWM 问的是：如果 latent 支持集合本身可以切换 dynamics mode，predictor 是否不再需要把所有 nonlinear regimes 混在一个 dense coordinate system 中拟合？

论文的 idealized motivation 来自 state-space quantization：把 compact state space 划成有限 cells，以 one-hot 表示 cell，再由 action-dependent transition matrix 移动 one-hot state。该构造在 latent 中是线性的，但 dimension 随 state dimension 呈 curse of dimensionality，因此只能作为动机，不能当作 learned LpWM 的保证。

## 4. Method

### 4.1 RDMReg：从 dense Gaussian target 改为 sparse rectified target

训练目标为 latent prediction loss 加 anti-collapse regularizer：

$$
\min_{\theta,\phi}\|\hat z_{t+1}-z_{t+1}\|_2
+\lambda_{\mathrm{RDMReg}}\mathcal R(z_{t+1}).
$$

`RDMReg` 在随机一维 projections 上，用 2-Wasserstein distance 将 empirical latent distribution 匹配到 Rectified Generalized Gaussian。LpWM 默认设置是：

$$
p=1,\quad \mu=0,\quad \sigma=\sqrt{1/2},
$$

即 Rectified Laplace target。对照的 dense LeWM 使用 identity link 与 isotropic Gaussian target (`p=2`)。为了公平，本文的 LeWM comparison 也使用 sliced 2-Wasserstein formulation，而不是原 SIGReg 的 Epps-Pulley loss；所以它不是对 original LeWM training recipe 的逐字复现。

### 4.2 RepReLU：forward exact zeros，backward 保留梯度

Encoder 是 from-scratch ViT，CLS token 经三层 MLP projector 得到 latent；predictor 是 AdaLN-zero Transformer 或 predictor ladder 中的较弱模型。LpWM 在 encoder 与 predictor projector 末端应用：

$$
\operatorname{RepReLU}(x)=\operatorname{sg}(\operatorname{ReLU}(x))
+\operatorname{GeLU}(x)-\operatorname{sg}(\operatorname{GeLU}(x)).
$$

Forward 等于 ReLU，因此产生 exact zeros；backward 使用 GeLU gradient，降低 dying-ReLU 风险。作者强调 RepReLU 是 optimization safeguard，不是 collapse prevention 的必要条件。

### 4.3 Predictor-complexity ladder

论文从强到弱比较六类 predictor：

| Predictor | Key property | `D=384` params | `D=4096` params |
|---|---|---:|---:|
| Deep-AdaLN(k) | 6-layer DiT-style Transformer | 25.8M | 822.4M |
| Shallow-AdaLN(k) | 1-layer AdaLN Transformer | 5.6M | 151.1M |
| MLP o LTV(k) | state-dependent low-rank operators + nonlinear readout | 0.81M | 84.7M |
| MLP o LTI(k) | fixed linear operators + nonlinear readout | 0.74M | 83.9M |
| LTI(k) | fixed linear recurrence over history | 0.59M | 67.1M |
| LTI(1) | single-frame linear state/action map | 0.30M | 33.6M |

PushT 使用 history `k=3`，因为 single observation 无法恢复 velocity；Wall 使用 `k=1`。一个重要 caveat 是 LTI variants 后仍接 RepReLU，因此 LpWM 的端到端 predictor 并非严格 linear map。

### 4.4 Planning protocol

Test time 用 CEM 在 latent space 搜索 action sequence，objective 是 predicted terminal latent 与 encoded goal latent 的 MSE。主要设置为：

- 300 candidate action sequences / CEM iteration；
- elite 30；30 CEM iterations；
- model horizon `H=5`，frameskip `5`，goal 为 25 raw environment steps ahead；
- open-loop：只规划一次，不 replanning；
- closed-loop：每执行 5 actions 后重新规划，最多 10 次；
- Wall / PushT 各 50 evaluation trajectories。

这与本项目既有 LeWM + PushT benchmark 的 exact planner settings 未必相同，不能直接拼接绝对 success rate。

## 5. Theory：one-hot linearization 证明了什么

对 compact state space 与 state 方向 uniformly Lipschitz 的 deterministic controlled dynamics，作者构造 finite one-hot encoder、cell-center decoder 与 action-conditioned transition matrix，使 decoded one-step error 至多为 `(L+1) epsilon`。

在 `[0,1]^d` 上用 `N=n^d` 个 grid cells 时，covering radius 为：

$$
\epsilon_N=\frac{\sqrt d}{2}N^{-1/d},
$$

固定 horizon `H` 的 rollout error 上界为：

$$
\|x_H-\hat x_H\|\leq
\frac{\sqrt d}{2}N^{-1/d}\sum_{k=0}^{H}L^k.
$$

应保留的边界：

- 这是 explicit quantization / one-hot construction，不是 learned distributed sparse representation 的 consistency theorem；
- transition matrix 可以随 continuous action 任意变化，未给出其 action dependence 的学习复杂度；
- `O(N^{-1/d})` 明确遭受 curse of dimensionality；
- bound 针对 fixed finite horizon；当 `L>1` 时，horizon factor 可快速增长。

因此 theory 支持“sparsity 可能换取更简单 latent dynamics”的动机，但不能单独证明 LpWM 的 empirical gain 来自 linearization。

## 6. Main Results

### 6.1 Wall：benchmark 太简单，稀疏性难以区分

在 Wall 上，dense LeWM 和 sparse LpWM 的最简单 LTI(1) predictor 都接近 100% closed-loop success。这里不能得出“sparsity 没用”，更合理的结论是 benchmark / horizon 已接近 saturation，无法区分 representation geometry。

### 6.2 PushT：优势集中在 intermediate predictor capacity

论文报告，在 PushT 上：

- lowest-capacity LTI(1)：两者都失败；
- highest-capacity Deep-/Shallow-AdaLN：LpWM 与 LeWM 接近，优势消失；
- intermediate capacity：LpWM 相对 LeWM 的 reported success-rate gap 为
  - MLP o LTI(k)：`24-57` points；
  - MLP o LTV(k)：`36-45` points；
  - LTI(k)：`11-23` points。

这正是 paper 最强、也最窄的 empirical claim：**sparsity 的收益依赖 predictor capacity 与 task complexity 的相对位置**，不是所有 predictor、所有 environment 都统一提升。

LpWM 的 validation active fraction 大约为 `0.28-0.63`（不同 predictor / dimension），所以确实产生了 exact sparse codes；但论文没有证明 active fraction 本身与 success gap 单调相关。

### 6.3 Dense baseline 不只一种

作者还把 LeWM 的 distribution-matching regularizer 替换为 VICReg，LpWM 在 PushT 的 Deep-AdaLN、MLP o LTV、MLP o LTI 上仍优于该 dense representation。这说明结果不只针对 Gaussian target；不过 VICReg 版本是本文构造的 baseline，且同样经过独立 hyperparameter sweep。

### 6.4 Mode-factored representation

在 synthetic Piecewise environment 中：

- binary support 的 Jaccard heatmap 与隐藏的 force-field zones 对齐，即使 background 不显示 zone；
- linear probe 用 support 几乎完美预测 discrete zone，continuous magnitudes 更适合预测 within-zone position；
- random-goal Piecewise 2x2、`H=5,R=1` 时，LpWM 为 `84.67 +/- 4.16`，LeWM 为 `65.33 +/- 4.16`；
- evaluation-set goals 已接近饱和，不足以区分方法。

在 OGBench-Cube 中，vanilla sparsity 的 support change 主要跟随快速的 end-effector motion，而不是 contact。加入 optional Temporal Jaccard (TJ) loss 后，support-instability 与 effector motion 的 correlation 从 `0.87` 降到 `0.40`，与 cube motion 从约 `0.26` 升到 `0.80`，与 contact 从 `0.05` 升到 `0.61`，planning success 基本不变。

所以“support = semantic dynamics mode”不是 vanilla LpWM 的普遍结论；它在 clean Piecewise 中自然出现，在 contact-rich setting 中需要 temporal prior 才转向更有意义的事件。

## 7. Evidence Boundary / Limitations

- **One training seed：** PushT hyperparameter sweeps 因 compute constraints 固定一个 training seed；3 seeds 只用于 CEM planning evaluation。
- **Best-cell reporting：** 每个 method / predictor / dimension 报 tuning grid 的 best-performing cell，且后续 sweep 会删掉已知差的 regions；comparison 不是 single preregistered recipe。
- **Planner evidence 仍需分层：** paper 同时给 open-loop 与 closed-loop success；不要把 fixed-observation predictor quality、open-loop plan 和 receding-horizon MPC 混成一个 claim。
- **No systems speedup evidence：** smaller predictor parameter count 不等于 wall-clock latency、peak memory、energy 或 throughput；exact zeros 也没有通过 sparse kernels 利用。
- **Retraining required：** LpWM 改 encoder geometry、regularizer 与 predictor output link，不能作为现有 LeWM checkpoint 的 post-hoc sparsification。
- **Comparison recipe changed：** dense LeWM baseline 使用 RDMReg-style 2-Wasserstein Gaussian matching，而非 original SIGReg loss；这是 fair controlled comparison，但不等于完全复现 published LeWM。
- **Short-horizon saturation：** OGBench-Cube goal 仅 25 raw steps ahead；作者明确指出 benchmark 可能过易。
- **Interpretability is conditional：** support 在无 temporal prior 时可退化为 motion detector；probe accuracy / correlation 不等于 causal factorization。
- **Theory-to-practice gap：** theorem 针对 one-hot quantization；learned codes是 28%-63% active 的 distributed sparse vectors。
- **Statistical reporting：** headline PushT study 缺少多 training-seed uncertainty；“up to 57”应读作配置特定的最大 gap，而非稳健平均提升。

## 8. Why It Matters for the FYP

这篇是当前 **LeWM + PushT** 主线的高相关 prior art，因为它直接改变 LeWM 的 learned representation，而不是另起一个 generative world-model family。

| Route | Needs retraining? | What changes | Primary evidence contract |
|---|---|---|---|
| LeWM baseline | yes | dense latent + expressive predictor | PushT prediction / planning / closed-loop |
| LpWM | yes | sparse non-negative latent + predictor ladder | lower predictor capacity at matched planning task |
| Exact cache | ideally no | reuse action-independent computation | bitwise score/rank/elite/first-action equivalence |
| Quantization | PTQ/QAT dependent | numerical precision of weights/activations | error + native memory/latency + planning fidelity |
| Sparse execution | likely retraining + kernel work | exploit zeros in actual compute | measured kernel/system speedup + same planning behavior |

最值得保留的 FYP hypothesis 是：**representation design 与 predictor compression 不应分开考虑；更可压缩的 predictor 可能来自更容易建模的 latent geometry。**

但不要把 LpWM 当成已有的 acceleration result。一个合理的最小验证顺序是：

1. 先复现 matched LeWM/LpWM predictor-level 与 fixed planning protocol；
2. 验证 prediction error、candidate ranking、elite overlap、first action 与 CEM trace；
3. 再测 native latency、memory、GPU utilization；
4. 最后才讨论 official closed-loop task success。

本阅读包不自动推进这些实验。

## 9. Reading Route

### 20 minutes - 抓住主张与边界

1. Abstract + Figure 1：写下 `dense LeWM -> Rectified Laplace target + RepReLU -> sparse LpWM`。
2. Section 3.1：只抓 Proposition 1、Corollary 1 与 curse of dimensionality。
3. Figure 1(b) + Section 3.3：确认 gain 集中在 intermediate predictor capacity。
4. Section 4.2：确认 vanilla sparsity 不自动得到 semantic contact modes。
5. Appendix H.1：标记 one training seed 与 best-cell sweep。

### 90 minutes - 能解释，也能质疑

1. Sections 2.1-2.3：追踪 RDMReg target、RepReLU 与 CEM cost。
2. Tables 1-3：把 predictor form、parameter count、active fraction 对齐。
3. Figure 3：区分 Wall saturation、PushT intermediate-capacity gain、VICReg comparison。
4. Figures 2 and 4：区分 Piecewise 的 natural factorization 与 OGBench-Cube 的 TJ-induced factorization。
5. Appendix D：记录 data horizon、CEM candidates / elites / iterations、open-loop / closed-loop boundary。
6. Appendix H：检查 sweep coverage、training seed 与 selection procedure。

### 3 hours - 形成 FYP prior-art card

1. 画出 LeWM 与 LpWM 的相同 call graph，只标出 regularizer、output link 与 predictor differences。
2. 给每个 PushT result 标注：predictor family、latent dimension、open/closed loop、best-cell selection、training/eval seeds。
3. 从 official code read-only 定位 RDMReg、RepReLU、predictor ladder、CEM config 与 reproduction scripts；不要运行 full grid。
4. 写一个最小 matched gate：dense/sparse x predictor capacity，先固定同一 dataset、CEM semantics 与 evaluation trajectories。
5. 单独写 systems hypothesis：parameter reduction、unstructured activation sparsity、structured sparsity分别需要什么 backend 才能转成真实 speedup。

## 10. Reading Questions（留给你回答）

1. Proposition 1 的 transition matrix 可以怎样依赖 continuous action？这个函数本身是否可能比原 dynamics 更难学习？
2. `O(N^{-1/d})` 的 curse of dimensionality 说明 one-hot construction 对 high-dimensional images 有什么现实限制？
3. Learned LpWM 是 28%-63% active，而不是 one-hot；理论动机与 empirical representation 中间缺哪条桥？
4. RepReLU 让所谓 LTI predictor 变成 piecewise-linear system；论文的“linearization”措辞应怎样限定？
5. PushT 的 gain 来自 sparsity、non-negativity、Laplace marginal、RepReLU optimization，还是这些因素的组合？需要什么 factorial ablation？
6. Dense LeWM 改用 2-Wasserstein Gaussian matching后，是否仍代表 original SIGReg LeWM 的最佳性能？
7. 为什么 LTI(1) 在 Wall 饱和、在 PushT 全部失败？history、partial observability 与 contact分别贡献多少？
8. “up to 57”来自哪个 exact predictor / dimension / open-or-closed-loop cell？平均与 worst-case gap 是多少？
9. 一个 training seed 加 best-cell selection 会怎样放大 headline gap？最小 multi-seed confirmatory design 是什么？
10. Active fraction 与 success 是否相关？同一 active fraction 下，不同 support structure 是否产生不同 planning behavior？
11. VICReg comparison 控制了哪些因素，又引入了哪些不同的 tuning degrees of freedom？
12. Piecewise support 在无视觉 zone cues 时仍能恢复 regime，这是 dynamics evidence；还能设计什么 action-shuffled negative control？
13. OGBench-Cube 中 TJ 把 support 从 effector motion 转向 contact，但为什么 planning success 不变？interpretability 与 utility 是否脱钩？
14. Goal distance 固定 25 raw steps 会如何限制 long-horizon conclusion？
15. CEM 使用 300 candidates x 30 iterations；更小 predictor 的参数优势是否会在 actual planner latency 中显现？
16. Exact zeros 若没有 sparse kernels，是否反而因更大 latent dimension带来额外 dense compute？
17. 对当前 LeWM + PushT 主线，LpWM 最合适的角色是 retrained baseline、representation ablation，还是 compression enabler？
18. 如果只允许一个 bounded experiment，应该优先验证 planning gain 的 multi-seed robustness，还是 native latency / memory？为什么？

## 11. Meeting Card

- **Paper：** Kuang et al., *LpWM: A Case for Sparse Representations in World Models*, arXiv:2608.22764v1, 2026。
- **Core idea：** 用 RDMReg 将 JEPA latent 匹配到 Rectified Laplace，并通过 RepReLU 形成 non-negative exact sparse codes，使较弱的 action-conditioned predictor 更容易建模 dynamics。
- **Strongest evidence：** PushT 的 intermediate-capacity predictors 上，LpWM 对 dense LeWM 报告 `11-57` points 的 planning-success gaps；high-capacity predictors 接近，LTI(1) 则两者都失败。
- **Theory：** finite one-hot quantization 可实现 action-conditioned linear latent transition，fixed-horizon error 随 `N^{-1/d}` 消失；但它不是 learned distributed sparse code 的保证。
- **Interpretability：** Piecewise 中 support 恢复 discrete zones、magnitude 保留 position；OGBench-Cube 中 vanilla support 主要跟随 motion，需 TJ prior 才转向 contact。
- **Critical caveat：** PushT sweep 只有一个 training seed、报告 tuned best cells；sparse latent / smaller predictor 尚无 wall-clock、memory 或 sparse-kernel speedup证据。
- **FYP connection：** 这是 LeWM + PushT 的 retraining-required representation baseline，提示 predictor compression 应与 latent geometry 联合设计，但不能作为现成 inference acceleration。
- **Question to bring to meeting：** 是否值得做一个最小 `dense vs sparse x predictor capacity` matched experiment，以检验 LpWM 的 gain 在本项目 frozen PushT/CEM semantics 下是否仍成立？

## 12. Citation

```bibtex
@misc{kuang2026lpwm,
  title         = {LpWM: A Case for Sparse Representations in World Models},
  author        = {Kuang, Yilun and Dagade, Yash and Le Lidec, Quentin and Maes, Lucas and Balestriero, Randall and LeCun, Yann},
  year          = {2026},
  eprint        = {2608.22764},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  version       = {v1},
  url           = {https://arxiv.org/abs/2608.22764}
}
```
