# WorldCache: Content-Aware Caching for Accelerated Video World Models

> **在本指南中的角色：** 李老师新增指定阅读，active VLA reading `#035`；连接 `video world model`、content-aware cache 与 robot-data video prediction 的 systems bridge。
> **本地论文：** [paper-arxiv-v1.pdf](paper-arxiv-v1.pdf)  
> **Official resources：** [arXiv](https://arxiv.org/abs/2603.22286) · [ECCV 2026 accepted-paper list](https://eccv.ecva.net/Conferences/2026/AcceptedPapers) · [project](https://umair1221.github.io/World-Cache/) · [code](https://github.com/umair1221/WorldCache)  
> **建议先修：** `Diffusion Transformer (DiT)`、denoising timestep、feature caching、`Zero-Order Hold`、optical flow、`PSNR/SSIM/LPIPS`。  
> **阅读状态：** `verified-full-text`；不表示复现了 H200 latency 或 robotics results。

## 1. Paper identity 与 name boundary

| Field | Record |
|---|---|
| Title | *WorldCache: Content-Aware Caching for Accelerated Video World Models* |
| Authors | Umair Nawaz, Ahmed Heakl, Ufaq Khan, Abdelrahman Shaker, Salman Khan, Fahad Shahbaz Khan |
| Venue / status | project/repository 标注 ECCV 2026 accepted；官方 ECCV 2026 video listing 可核到该标题 |
| arXiv | 2603.22286 v1，submitted 2026-03-23 |
| Local artifact | arXiv v1 author manuscript，33 pages，18,828,805 bytes，SHA-256 `1c911e464cf80ea786330a0a4b41a82ea441c2c694668c54de94d901385ff7b0` |

**版本边界。** 本地是 arXiv v1，不称为 ECCV proceedings final。venue acceptance 与本地 binary version 必须分开说。

**同名边界。** 这篇不是 [WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching](../036-worldcache-heterogeneous-token-caching/README.md)。本篇用 `CFC + SWD + OFA + ATS` 决定 deep blocks 何时缓存、如何补偿；`#036` 用 token-level curvature 把 outputs 分成 stable/linear/chaotic。作者、arXiv、venue、model 和 cache granularity 均不同。

## 2. 一句话结论

本篇 `WorldCache` 以 `DiCache` 风格的 probe-then-cache 为骨架，再用 temporal change、spatial saliency、first-order correction 与 late-step scheduling 避免缓存动态物体；作者在 `Cosmos-Predict2.5` 和 `WAN2.1` 上报告约 `2.1–2.36×` latency speedup、自动指标大体保持，但 robotics evidence 仍是 video prediction fidelity，不是 closed-loop policy success。

## 3. Problem：为什么 fixed caching 不够

Video world model 的每个 denoising step 都跑完整 `DiT` 很贵。已有 training-free cache 通常：

1. 先运行前 `k` 个 blocks 作为 probe；
2. 用全局 feature drift 判断后面的 deep blocks 是否可跳过；
3. 若跳过，直接复用上次 cached output，即 `Zero-Order Hold`。

问题在于 video 的 token 不是同质的：背景面积大、变化慢，会压低全局 drift；小而关键的 moving object 可能被平均掉。旧 feature 直接复用还会造成 ghosting、blur 与 motion lag。固定 threshold 也没有利用 late denoising steps 更可预测的特点。

论文的核心问题是：**在不 retrain world model 的情况下，能否让 caching 同时感知 temporal content、spatial saliency、motion 与 denoising stage？**

## 4. Method：四个 module 各修一个失败模式

### 4.1 Base：probe-then-cache

每个 denoising step 先跑前 `k` 个 blocks，得到 probe feature；根据 drift 决定：

- `FULL`：运行剩余 deep blocks，并更新 cache；
- `CACHE`：跳过 deep blocks，复用/修正 cached result。

这是 inherited systems skeleton，不是本篇全部 novelty。

### 4.2 CFC — Causal Feature Caching

先从 raw latent 估计两步 velocity `v_t`，再把基础 threshold 调成随内容变化的形式：

$$
\tau_{\mathrm{CFC}}(t)=\frac{\tau_0}{1+\alpha v_t}.
$$

motion 越大，threshold 越严格，更容易触发 fresh compute。实现使用 ping-pong buffer 来保留 latent history。

### 4.3 SWD — Saliency-Weighted Drift

用 channel variance 生成 spatial saliency map，让高变化、小面积区域在 drift aggregation 中获得更大权重。它解决“静态背景把移动物体淹没”的问题。

关键边界：variance saliency 是 cheap heuristic，不是 object detector，也不等于 task relevance。高纹理区域可能显著但不重要，低纹理 gripper/contact 也可能重要。

### 4.4 OFA — Optimal Feature Approximation

对 stale cached feature 做 first-order prediction，而不是原样 hold：

- `OSI` 用 online least-squares vector projection 估计 gain `γ`，并 clamp 防止发散；
- optional motion warping 用 latent-space optical flow 对齐 feature。正文概括为 multi-scale matching，supplement 具体实现为 downsampled `Lucas–Kanade`；两者不要混成更强的 learned flow module。

作者报告 warping overhead 小于 cached-step latency 的 3%，但这是特定 implementation/hardware 下的 measurement。

### 4.5 ATS — Adaptive Threshold Scheduling

late denoising steps 通常更平滑，ATS 逐步放宽 cache threshold，从而把更多 compute 省在后半程。

正文 equation (13) 给出线性 relaxation；supplement 说明实际实现使用与 `N/35` 有关的 quadratic coefficient。最稳妥的表述是：**概念上是 late-step relaxation；精确 schedule 以 released code/config 为准。**

```text
latent history ─> CFC content threshold
probe features ─> SWD saliency-weighted drift ─> FULL or CACHE
                                               │
                           cached feature ─> OFA correction/warping
                                               │
denoising timestep ─> ATS late-step relaxation ┘
```

## 5. Main experiments

### 5.1 Setup

- Models: `Cosmos-Predict2.5 2B/14B`、`WAN2.1 1.3B/14B`；
- tasks: text-to-world (`T2W`) 与 image-to-world (`I2W`)；
- benchmark: `PAI-Bench`；
- standard generation: 35 denoising steps、93 frames、约 5.8 seconds at 16 FPS；
- latency: single `NVIDIA H200`，supplement 指定 H200 NVL 140 GB；
- matched batch/precision are claimed, but performance still depends on model implementation and cache policy。

### 5.2 Main quality / latency table

| Model / task | Baseline latency | WorldCache | Speedup | Overall score change | Locator |
|---|---:|---:|---:|---:|---|
| Cosmos 2B T2W | 54.34 s | 26.28 s | 2.10× | 0.748 → 0.745 | Table 1, p. 10 |
| Cosmos 14B T2W | 216.25 s | 98.61 s | 2.14× | 0.769 → 0.771 | Table 1, p. 10 |
| Cosmos 2B I2W | 55.04 s | 24.48 s | 2.30× | 0.803 → 0.798 | Table 2, p. 11 |
| Cosmos 14B I2W | 210.07 s | 99.25 s | 2.18× | 0.814 → 0.813 | Table 2, p. 11 |
| WAN 1.3B T2W | 120.04 s | 50.84 s | 2.36× | 0.7727 → 0.7721 | Table 3, p. 12 |
| WAN 14B I2W | 475.60 s | 206.73 s | 2.31× | 0.7384 → 0.7388 | Table 3, p. 12 |

这些结果支持的是：在作者设置中，自动 aggregate score 大体保持，latency 约减半。它们不支持跨 model 直接预测相同 speedup，也不能把小幅 metric change 等同于 semantic/physical correctness。

### 5.3 Ablation：每个 component 在做什么

Cosmos 2B I2W ablation（Table 4, local PDF p. 14）：

| Variant | Latency | Speedup | Overall |
|---|---:|---:|---:|
| Base | 55 s | 1.00× | 0.8027 |
| + CFC | 36 s | 1.52× | 0.8020 |
| + SWD | 33 s | 1.67× | 0.8003 |
| + OFA | 37 s | 1.49× | 0.8035 |
| + ATS | 25 s | 2.30× | 0.7977 |

最好把它读成一条 Pareto path：`CFC/SWD` 主要增加 cache hit；`OFA` 花一点 compute 换 quality recovery；`ATS` 再积极提高 late-step caching，带来最大 speedup 与小幅 aggregate drop。

### 5.4 Longer sampling budget

Figure 6（local PDF p. 29）报告从 35 到 140 steps，WorldCache latency 约 `25.0 → 66.0 s`，baseline 为 `57.0 → 199.1 s`，较长 schedule 最高约 `3.10×`。这说明固定 overhead 被更多 denoising steps 摊薄，但不能与 35-step headline 当作同一 setting。

## 6. Robotics-related evidence：最容易被误读的部分

Supplement Table 6（local PDF p. 26）在 `EgoDex-Eval` 上测 video prediction fidelity：

| Model | Baseline → WorldCache latency | Speedup | Selected quality change |
|---|---:|---:|---|
| WAN 14B | 391.9 → 171.6 s | 2.30× | PSNR 13.30 → 13.19 |
| Cosmos 2B | 70.01 → 43.24 s | 1.62× | PSNR 12.87 → 12.82 |
| DreamDojo 2B | 19.73 → 10.36 s | 1.90× | PSNR 23.63 → 23.69；SSIM 0.775 → 0.737；LPIPS 0.226 → 0.251 |

这组结果说明 cache 可用于 egocentric/robot-data video model，但它**没有 action-conditioned closed-loop rollout、policy success、control frequency 或 real-robot intervention**。而且不同 metric 可朝不同方向变化；DreamDojo 的 PSNR 略升不能覆盖 SSIM/LPIPS 变差。

## 7. Limitations 与 critique

### Authors' stated limitations

- abrupt viewpoint change、heavy occlusion 会降低 cache hit rate；
- simple motion alignment 无法覆盖所有 non-rigid/large motion；
- future work 包括 learned/online caching、stronger motion 与 uncertainty-aware warping。

### 我的 critique

- **不是 task success evidence。** `PAI-Bench` 与 image metrics 评的是生成结果，不是机器人决策正确性。
- **baseline world model 也可能生成错误世界。** cache 与 baseline 相似不代表两者都符合 physics 或 task goal。
- **“training-free”不等于 zero-cost。** 仍有 integration、threshold tuning、buffers、saliency、alignment 与 hardware-specific implementation cost。
- **指标不完整。** 缺 power/energy、P50/P99 latency、peak memory、cache memory、long-horizon drift 与 human evaluation。
- **schedule details 有 main/supplement 差异。** 复现必须以 exact code/config 为准。
- **未做 policy-in-the-loop evaluation。** 若把 world model 用于 planning，缓存误差可能改变 action selection，不能从 video fidelity 直接推出 safety。

## 8. 与 `#036 WorldCache` 的最短对照

| Dimension | 本篇 `#035` | `#036` |
|---|---|---|
| Cache granularity | probe 后的 deep blocks / feature cache | whole-model output 中的 token groups |
| Main signal | latent change + saliency-weighted feature drift | per-token temporal curvature |
| Prediction | ZOH + OSI / optional motion warp | stable reuse + linear + damped Hermite |
| Refresh | content/timestep-adjusted threshold | chaotic-token accumulated error budget |
| Models | Cosmos, WAN, DreamDojo | HunyuanVoyager, Aether, LingBot |
| Hardware | H200 | A800 |

不要直接比较 `2.3×` 与 `3.65×`：model、hardware、steps、resolution、benchmark 与 baseline family 都不匹配。

## 9. 对 FYP 的价值

本篇提供一个与 quantization 正交的 acceleration axis：**少算 denoising blocks/steps，而不是降低 weight precision**。可形成如下 matched experiment：

- same world/VLA model；
- `FP baseline / quantization only / caching only / quantization + caching`；
- 同时测 video fidelity、action selection change、closed-loop success、P50/P99 latency、peak memory、power；
- 按 static scene、fast motion、occlusion、camera jump 分层；
- 记录 cache hit rate 与 error accumulation，而不仅是 average speedup。

最重要的 research question 是：**quantization noise 会不会让 drift detector 更容易误触发或漏触发，从而改变 caching policy 的 Pareto frontier？**

## 10. 阅读路线

### 20-minute route

1. Abstract、Figure 1、Figure 2：找出 static-background failure。
2. Method overview：每个 module 只写一句“修什么问题”。
3. Tables 1–3：分开 model/task/hardware。
4. Table 4：解释 speed/quality trade-off。
5. Supplement Table 6：说清它不是 policy success。

### 90-minute route

1. 画 base probe-then-cache control flow。
2. 逐式读 CFC 与 SWD，检查 threshold direction。
3. 读 OSI gain 与 clamp，区分 prediction 与 motion warping。
4. 对照 main Eq. 13 与 supplement implementation schedule。
5. 重算 Tables 1–3 的 speedup/retention。
6. 读 Figure 6，解释 step budget 为什么改变 speedup。
7. 找作者 limitations，再补 policy-in-loop boundary。
8. 与 `#036` 填一张 matched comparison table。

### 3-hour deep route

1. 完成 90-minute route。
2. 阅读 code 中 model-specific hook、threshold defaults 与 warping path。
3. 为 fast motion/occlusion/camera jump 设计 stratified benchmark。
4. 设计 `quantization × caching` 2×2 ablation，明确同一 checkpoint/hardware/workload。
5. 写出从 video metric 到 planning/control metric 之间缺失的 causal link。

## 11. Reading Questions（读完再答）

1. probe blocks `k` 如何选，成本是否随 architecture 变化？
2. CFC 的 `v_t` 是标量还是 aggregate tensor statistic？
3. `α` 与 `τ0` 是否按 model/task 调参？调参数据是否与 test overlap？
4. SWD 的 channel variance 为什么能代表 task saliency？
5. small moving object 与 camera motion 如何被区分？
6. OSI 的 least-squares gain 在何种情况下会不稳定，clamp 范围是什么？
7. optional warping 是否包含在所有 headline rows？
8. main linear ATS 与 supplement quadratic implementation 应如何对应？
9. cache memory、buffer memory 与 peak GPU memory 是多少？
10. speedup 对 denoising steps、frames、resolution 的 sensitivity 是什么？
11. 为什么部分 automatic score 在 cached model 上略升？是 noise、regularization 还是 metric variance？
12. EgoDex-Eval 的 reference 与 metric protocol 能否代表 control usefulness？
13. long-horizon repeated use 会不会累积 world-state inconsistency？
14. 若输入包含 actions，什么 cache error 会改变 planner/policy decision？
15. quantization noise 会如何影响 drift threshold 与 OFA extrapolation？

## 12. Meeting card（读后填写）

- **Problem:**
- **Mechanism:**
- **Strongest evidence:**
- **Strongest limitation / unsupported leap:**
- **与 `#036 WorldCache` 的一句区别:**
- **FYP experiment:**
- **Question for 李老师:**

## 13. Evidence boundary

本地文件通过 PDF 结构、标题与页面抽查；来源身份由 arXiv、project/code 与 official conference listing 交叉核对。数字仍是作者报告，不是本机 H200 reproduction。尤其不要把 `training-free` 写成 free deployment，也不要把 video fidelity 写成 closed-loop robot performance。

