# WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching

> **在本指南中的角色：** 李老师新增指定阅读，active WM reading `#018`；以 token temporal curvature 做 heterogeneous caching 的 world-model inference paper。
> **本地论文：** [paper-arxiv-v2.pdf](paper-arxiv-v2.pdf)  
> **Official resources：** [arXiv](https://arxiv.org/abs/2603.06331) · [ICML 2026 accepted-paper list](https://icml.cc/Downloads/2026) · [code](https://github.com/FofGofx/WorldCache)  
> **建议先修：** `Diffusion Transformer`、feature/model-wise caching、finite difference、curvature、`cubic Hermite interpolation`、RGB/depth world models。  
> **阅读状态：** `verified-full-text`；不表示复现了 A800 latency 或 3D results。

## 1. Paper identity 与 name boundary

| Field | Record |
|---|---|
| Title | *WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching* |
| Authors | Weilun Feng, Guoxin Fan, Haotong Qin, Chuanguang Yang, Mingqiang Wu, Yuqi Li, Xiangqi Li, Zhulin An, Libo Huang, Dingrui Wang, Longlong Liao, Michele Magno, Yongjun Xu |
| Venue / status | ICML 2026 accepted；本地 PDF 首页标注 PMLR 306 / 43rd ICML, Seoul, 2026 |
| arXiv | 2603.06331，v1 2026-03-06；v2 2026-06-01 |
| Local artifact | arXiv v2 / proceedings-style author manuscript，26 pages，13,203,052 bytes，SHA-256 `436c67f3301ad02cced74cd40aaa2d8f28fab27b57616def71e7b64dbb9b341a` |

**同名边界。** 本篇不是 [WorldCache: Content-Aware Caching for Accelerated Video World Models](../017-worldcache-content-aware/README.md)。本篇的核心是 `CHTP + CAS`：把 RGB/depth tokens 按 temporal curvature 分组并采用不同 predictor；`#017` 是 probe/deep-block cache，使用 content change、saliency、optimal approximation 与 timestep schedule。

**“For Free”边界。** 指不 retrain base world model、controller overhead 很低；并不表示零 memory、零 integration、零 quality risk 或任何 hardware 都得到同样 speedup。

## 2. 一句话结论

本篇 `WorldCache` 观察到只有少量 high-curvature tokens 主导 cache error，于是按 stable/linear/chaotic 三类分别 reuse、linear extrapolate、damped Hermite predict，再只用 chaotic subset 的累计误差触发 full recomputation；作者在 A800 上报告 `1.68–3.65×` speedup 与低 controller overhead，但证据仍是 world generation / reconstruction metric，不是 closed-loop robot policy success。

## 3. Problem：uniform cache policy 错在哪

World model outputs 同时含：

- 大片缓慢变化的 background/RGB tokens；
- 几何和 depth boundaries；
- fast motion、occlusion、camera change 对应的少量 chaotic tokens。

现有 cache 往往对所有 tokens 使用同一 `reuse` 或同一 extrapolation。这样会出现两种浪费：

1. 为少数难 token 频繁重算全部 token；
2. 为追求高 hit rate，把同一线性 predictor 强加给 high-curvature token，造成 local drift。

论文问：**能否只用已有 full steps 的 feature history，在线识别 token dynamics，并对不同 token 用不同 cheap predictor？**

## 4. Method

### 4.1 CHTP — Curvature-guided Heterogeneous Token Prediction

从最近三个 `FULL` model outputs 构造 token-wise velocity 与 acceleration，再定义归一化 curvature score：

$$
\kappa_i = \frac{\lVert a_i\rVert}{\lVert v_i\rVert^2+\epsilon}.
$$

按 percentile thresholds 将 tokens 分为：

- **stable:** 变化很小，直接 reuse cached value；
- **linear:** trajectory 近似线性，用 linear extrapolation；
- **chaotic:** curvature 高，用最近 velocities 的 damped cubic-Hermite/smoothstep blending，避免简单 extrapolation 发散。

分组是 relative percentile，不是一个跨 model 可复用的 physical curvature threshold。开始时需要至少三个 full outputs 来 warm up history。

### 4.2 CAS — Chaotic-prioritized Adaptive Skipping

作者定义每个 token 的 dimensionless drift proxy：

$$
e_i = \kappa_i\lVert\Delta y_i\rVert.
$$

只聚合 chaotic token 的 error，并累计成 `E_acc`：

- `E_acc < η`：继续 cached prediction；
- `E_acc ≥ η`：触发一次 full model evaluation、刷新 history、重置 budget。

这让少量“最不可靠”的 tokens 决定 refresh，而不是用背景主导的全局 average。

```text
last three FULL outputs
        │
        ├─ finite-difference velocity / acceleration
        ├─ per-token curvature percentile
        │       ├─ stable  ─> reuse
        │       ├─ linear  ─> extrapolate
        │       └─ chaotic ─> damped Hermite prediction
        │
        └─ chaotic drift accumulation CAS ─> FULL refresh when E_acc >= eta
```

### 4.3 Cache granularity

这是 model-wise output caching：cached steps 避免运行大部分/整个 expensive model forward，只执行 controller/predictor。与 layer-wise cache 比较时要注意后者可能需要存多层 activations，memory footprint 完全不同。

## 5. Main experiments

### 5.1 Setup

- Models: `HunyuanVoyager-13B`、`Aether-5B`；appendix 增加 `LingBot-14B`；
- outputs: RGB/depth world generation 与 3D reconstruction；
- hardware: single `NVIDIA A800`；
- comparisons: model-wise 与 layer-wise caching baselines；
- reported metrics: `WorldScore` dimensions、PSNR/SSIM/LPIPS、depth/pose/reconstruction metrics、latency 与 peak memory。

### 5.2 Headline results

| Model | Baseline latency | WorldCache latency | Speedup | Selected quality change | Peak memory |
|---|---:|---:|---:|---|---:|
| HunyuanVoyager-13B | 1053.7 s | 288.6 s | 3.65× | WorldScore Dynamic 46.40 → 45.43 | 50.44 → 50.58 GB |
| Aether-5B | 179.7 s | 107.2 s | 1.68× | WorldScore Dynamic 45.22 → 44.72 | 46.58 → 46.59 GB |

Locator: Table 1, local PDF p. 7。Voyager dynamic score 保留约 `97.9%`，但单一 aggregate/dimension 的 retention 不是 physical correctness guarantee。

Table 1 的列标题写 `Memory Overhead`，数值却接近完整 peak footprint。Appendix Table 11 才把 method-added memory 解释为约 `0.02–0.03 GB`（49–161 frames）。读报告时应分别写：**peak memory 基本不变；controller/history 的额外 memory 很小。**

### 5.3 3D reconstruction

`Aether-5B` 3D experiment 报告 `55.42 → 21.20 s`（`2.61×`），`AbsRel 0.340 → 0.341`，pose metrics 接近 baseline（Table 2, local PDF p. 7）。这支持 cached prediction 没明显破坏该 reconstruction pipeline，但仍是离线 reconstruction，不是 robot navigation/controller rollout。

### 5.4 为什么 heterogeneous grouping 有用

Table 5（local PDF p. 8）：

| Predictor/grouping | PSNR | SSIM | LPIPS | Latency |
|---|---:|---:|---:|---:|
| CHTP | 25.76 | 0.791 | 0.227 | 86.94 s |
| Random grouping | 22.59 | 0.710 | 0.314 | 86.98 s |
| Uniform linear | 18.01 | — | — | comparable regime |

在接近 latency 下，curvature grouping 明显优于 random；这比只看 headline speedup 更能支撑 method mechanism。

Table 6（local PDF p. 9）中 CAS 为 `PSNR 27.10 / SSIM 0.881 / LPIPS 0.198`，优于 fixed scheduling 的 `26.18 / 0.830 / 0.216` 等结果，说明 chaotic subset 驱动的 refresh 比 fixed interval 更适合这组 trajectories。

### 5.5 Threshold trade-off

Table 4 中 `η=0.10` latency 约 109 s、quality 较高；放宽到 `η=0.35` latency 约 90.35 s、quality 下降。`η` 不是无成本 performance knob，而是 speed–drift budget。

## 6. Overhead 与 comparison fairness

Appendix Table 10 报告 controller overhead 仅 `0.054–0.060%`，例如 `0.189 s / 348.7 s` 与 `0.064 s / 107.2 s`。这支持 controller 本身很轻，但整体 speedup 仍取决于 skipped full forwards 的数量。

Layer-wise baselines 的 peak memory 超过 100 GB，论文使用 CPU offloading 才能运行；其 latency 因此混入 host-device transfer。最公平的 headline comparison 应优先看：

- same model；
- same GPU/resolution/steps/frames；
- model-wise baselines；
- 是否都使用 offloading；
- quality operating point 是否匹配。

不能把包含 CPU offload penalty 的 layer-wise latency 全部归因于本篇 caching algorithm 更快。

## 7. Limitations 与 critique

### Authors 可定位的限制

论文没有独立 `Limitations` section，但 appendix 明确指出：rapid camera motion、complex texture、abrupt geometry change 会降低 speedup，或在 threshold 太宽时产生 local drift。

### 我的 critique

- **world metric 不是 policy metric。** 没有 action-conditioned closed-loop success、planning regret、control frequency 或 real-robot safety。
- **“physics-grounded curvature”是 feature trajectory heuristic。** curvature 有动力学直觉，但不构成预测符合真实 physics 的证明。
- **uncertainty 不充分。** 表格主要是 author-run point estimates，没有 seeds/confidence intervals。
- **baseline fairness 有 offload confound。** layer-wise comparison 需要单独解释。
- **speedup 不稳定。** Voyager `3.65×` 与 Aether `1.68×` 已表明 model/schedule dependency；不可把最高值当普遍结果。
- **缺少 power/energy 与 tail latency。** average end-to-end latency 不能覆盖 interactive deployment jitter。
- **warm-up 与 discontinuity risk。** 只基于三个 full outputs 的 finite differences，遇到 scene cut/abrupt motion 可能过时。

## 8. 与 `#017 WorldCache` 的最短对照

| Dimension | 本篇 `#018` | `#017` |
|---|---|---|
| Unit | output token | deep-block feature/cache decision |
| Heterogeneity | stable / linear / chaotic | spatial saliency + temporal content |
| Predictor | reuse / linear / damped Hermite | ZOH + online alignment / warp |
| Refresh | chaotic accumulated error | threshold adjusted by motion and timestep |
| Main hardware | A800 | H200 |
| Max headline | 3.65× on Voyager | 2.36× main 35-step rows; 3.10× longer steps |

两者可组合的想法很诱人，但不能直接假设 additive speedup：一篇的 predicted/cached outputs 会改变另一篇的 drift/curvature statistics。

## 9. 对 FYP 的价值

这篇与 VLA quantization 有三个可研究交点：

1. low-bit noise 会改变 finite-difference velocity/curvature，可能让 token classification 不稳定；
2. action-relevant tokens 未必是 high-curvature tokens，需要 task/action-aware weighting；
3. cache 与 quantization 都会引入 temporally correlated error，closed-loop 中可能非线性叠加。

可执行实验：在同一 world/VLA backbone 上做 `precision × η` grid，按 token group 记录 curvature distribution、cache refresh、video/depth error、action change 与 closed-loop success；再看 Pareto frontier 是否因 quantization 系统性移动。

## 10. 阅读路线

### 20-minute route

1. Abstract + Figure 1：说清 token heterogeneity。
2. Equations 5–9：只抓 curvature、三类 predictor、CAS budget。
3. Table 1：对比 Voyager 与 Aether 的 speedup 差异。
4. Tables 5–6：找 mechanism evidence。
5. Appendix limitations + overhead tables：拆开 peak memory 与 added memory。

### 90-minute route

1. 手算一条三点 trajectory 的 velocity、acceleration 与 curvature。
2. 画 stable/linear/chaotic predictor，不看原图复述。
3. 推导为什么 `η` 增大同时提高 cache duration 与 drift risk。
4. 审计 Table 1：model-wise vs layer-wise、CPU offload、memory header。
5. 重算 Table 2 speedup 与 metric differences。
6. 用 Tables 5–6 判断 evidence 是否支持 CHTP/CAS，而不是只支持整套 system。
7. 读 Tables 7–11，记录跨 frames/steps/model 的 sensitivity。
8. 与 `#017` 做 matched/not-matched comparison。

### 3-hour deep route

1. 完成 90-minute route。
2. 阅读 repository，定位 percentile thresholds、warm-up、`η` defaults 与 Hermite damping。
3. 设计 abrupt camera/occlusion/geometry stress test。
4. 设计 matched model-wise baseline，禁止 CPU offload confound。
5. 加入 quantization 后测 curvature distribution shift 与 controller stability。

## 11. Reading Questions（读完再答）

1. curvature equation 为什么用 `||v||²` normalization，静止 token 会怎样？
2. stable/linear/chaotic percentiles 是固定比例还是随 step/model 调整？
3. 为什么最近三个 `FULL` outputs 足以估计 trajectory regime？
4. cached steps 之间是否更新 predictor state，还是只沿 full history extrapolate？
5. damped Hermite 的 damping/smoothstep 系数是什么？
6. CAS 为什么只平均 chaotic tokens，不考虑稳定组中突发变化？
7. `η` 如何 calibration，是否对每个 model/task 单独调？
8. warm-up full steps 占短 schedule 的比例是多少？
9. Table 1 哪些 baselines 因 CPU offloading 受到额外 latency penalty？
10. peak memory 与 added cache memory 应分别如何报告？
11. Voyager 与 Aether speedup 差异由 compute、cache hit 还是 schedule 主导？
12. random grouping ablation 是否严格匹配 group sizes 与 refresh count？
13. rapid camera motion 下是 hit rate 下降，还是同一 hit rate 下 quality 下降？
14. RGB 与 depth tokens 的 curvature distribution 是否不同？
15. action-relevant但 low-curvature 的 token 会不会被长期复用？
16. quantization 会把哪些 token 推入 chaotic group？

## 12. Meeting card（读后填写）

- **Problem:**
- **Mechanism:**
- **Strongest evidence:**
- **Strongest limitation / unsupported leap:**
- **与 `#017 WorldCache` 的一句区别:**
- **FYP experiment:**
- **Question for 李老师:**

## 13. Evidence boundary

本地 PDF 已通过结构、标题与页面抽查；ICML identity、arXiv record 与 official code 分开核对。报告数字仍是作者 evidence，不代表本机 A800 reproduction。不要把低 controller overhead 写成 zero-cost，也不要把 feature curvature 写成 physical correctness proof。

