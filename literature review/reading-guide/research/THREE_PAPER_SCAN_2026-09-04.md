# Three-paper targeted scan — HBVLA and two WorldCache papers

> **Snapshot:** 2026-09-04  
> **Mode:** scope-bound three-way scan for the three papers assigned by 李老师  
> **Output boundary:** identity, method, strongest evidence, reading order, and unresolved questions；not a systematic review or replication claim

## 1. Inclusion set

| Core # | Exact work | Identity / local version | Role |
|---:|---|---|---|
| 20 | *HBVLA: Pushing 1-Bit Post-Training Quantization for Vision-Language-Action Models* | [arXiv:2602.13710 v2](https://arxiv.org/abs/2602.13710)；local arXiv v2 | direct `VLA × binary PTQ` evidence |
| 21 | *WorldCache: Content-Aware Caching for Accelerated Video World Models* | [arXiv:2603.22286 v1](https://arxiv.org/abs/2603.22286)；project/code label ECCV 2026 accepted；local arXiv v1 | content/saliency-aware deep-block caching |
| 22 | *WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching* | [arXiv:2603.06331 v2](https://arxiv.org/abs/2603.06331)；ICML 2026；local arXiv v2 | curvature-aware token prediction and scheduling |

`WorldCache` 是两篇不同论文，不是 version update：作者、subtitle、arXiv ID、venue、model family、hardware 与 cache granularity 均不同。

## 2. Search method 与 coverage boundary

本次只做支持阅读准备的 targeted scan：

1. 用 exact-title / method-name queries 搜索 2025–2026 paper indexes；
2. 用 arXiv records 闭合 title、authors、dates 与 local binary version；
3. 用 official conference listings、project pages 与 repositories 闭合 venue/code identity；
4. 从三份 local PDF 抽取 method、tables、limitations 与 appendix evidence；
5. 不扩张到新的 companion corpus，也不把同名/相邻 caching paper 纳入 core。

CLI 聚合搜索返回 41 个 unique raw records；exact-title 与 scope filtering 后只保留用户指定的 3 个 work identities，其余 38 个是泛 world-model、generic cache 或 title-noise records。本次检索存在以下 coverage limitations：

- `openreview-py` 未安装，OpenReview source 未进入聚合结果；
- Semantic Scholar 对 HB-VLA 与 heterogeneous-token query 返回 HTTP 429；
- OpenAlex 一度返回 HTTP 504 并 retry；
- HB-VLA 未通过 noisy query 可靠命中，最终用 direct arXiv identity 闭合；
- 因此这份报告证明三篇指定论文的身份与阅读 evidence，不声称相关工作检索完整。

## 3. WHY / HOW / WHAT

| Paper | WHY | HOW | WHAT |
|---|---|---|---|
| `#20 HB-VLA` | generic LLM/VLM PTQ 不感知 action drift，binary error 会在 closed loop 累积 | action-gradient rectified Hessian + embodiment drift weighting + salient/non-salient partition + Haar-domain 1-bit PTQ | average `1.02–1.13 bits/weight`；weight memory about 82% lower；simulation success 保持多数，但 Mobile ALOHA absolute drop 为 12.5–23.4 pp；2.93× latency headline 缺完整 table/protocol |
| `#21 WorldCache` | global fixed drift 被静态背景主导，stale features 造成 ghosting/motion lag | probe-then-cache + CFC temporal threshold + SWD saliency weighting + OFA correction/warp + ATS late-step relaxation | Cosmos/WAN main rows about `2.10–2.36×`；automatic scores 大体保持；EgoDex is video prediction fidelity, not policy success |
| `#22 WorldCache` | RGB/depth tokens dynamics heterogeneous，uniform reuse/prediction 为少数 chaotic tokens 浪费 full compute | `Curvature-guided Heterogeneous Token Prediction (CHTP)`：stable reuse / linear extrapolation / chaotic damped Hermite；`Chaotic-prioritized Adaptive Skipping (CAS)` 用 chaotic accumulated drift 触发 refresh | Voyager `3.65×`、Aether `1.68×`、3D path `2.61×`；controller overhead about 0.05–0.06%；rapid motion/geometry change remains difficult |

## 4. Cross-paper synthesis

### Shared WHY

三篇都反对“所有 parts 同等重要”的 approximation：

- `HB-VLA`：不同 components/columns/action dimensions 对 task success 的敏感度不同；
- `#21`：不同 spatial regions 与 timesteps 的 cache risk 不同；
- `#22`：不同 tokens 的 temporal trajectory 不同。

共同 design pattern 是：**先估计 heterogeneity，再把 scarce compute/precision 分配给高风险部分。**

### Divergent HOW

```text
HB-VLA:    weight columns  ─ sensitivity ─> protect salient parameters
WorldCache #21: deep blocks/regions ─ drift+saliency ─> decide full vs cached feature
WorldCache #22: output tokens ─ curvature ─> choose reuse/extrapolate/Hermite + refresh
```

它们处理的是不同 axis：`parameter precision`、`layer compute reuse`、`token temporal prediction`。潜在上可以组合，但 error detector 本身会受到 quantization/caching noise 影响，speedups 和 quality loss 都不能直接相加。

### Strongest evidence

- `HB-VLA` 的 strongest evidence 不是 abstract headline，而是跨 `π0.5 / OpenVLA-OFT / CogACT` 的 component-wise binary tests 与 Mobile ALOHA results；后者也最清楚地显示 remaining gap。
- `#21` 的 strongest evidence 是多 model/size/task 的 H200 table 加 component ablation；它表明每个 module 在 speed–quality path 中作用不同。
- `#22` 的 strongest mechanism evidence 是在近似 latency 下，curvature grouping 明显优于 random/uniform predictor，以及 CAS 优于 fixed scheduling；不是只看最高 3.65×。

### Shared unresolved gap

三篇都没有完成同一个关键闭环：

> 在 matched VLA/world-model checkpoint、hardware、workload 与 action interface 下，同时验证 `closed-loop success + latency distribution + peak memory + power/energy + failure severity`。

`HB-VLA` 有 real-robot success，但 latency evidence 不完整；两篇 `WorldCache` 有较完整 latency，但没有 policy-in-the-loop success。三者刚好形成 FYP 可追的 measurement gap。

## 5. Recommended reading order

1. **`#20 HB-VLA`**：与你的 VLA quantization 主线最直接；先建立 action-aware approximation 与 binary failure 的尺度。
2. **`#21 Content-Aware WorldCache`**：先理解 probe/deep-block cache 和 content/motion-aware refresh。
3. **`#22 Heterogeneous Token WorldCache`**：再进入 token curvature、three-way predictor 与 chaotic budget。
4. 最后只做一张 cross-paper table：`approximation unit / risk signal / compute saved / quality metric / hardware / closed-loop evidence`。

## 6. FYP-ready hypotheses

以下是假设，不是论文已证明的结论：

1. `action-aware saliency` 可用于替代或补充 WorldCache 的 generic visual saliency/curvature，使 action-critical tokens 更早 refresh。
2. W1/W2 quantization noise 会抬高 estimated drift/curvature，降低 cache hit rate；适配 threshold 后可能恢复 speed，但增加漏检风险。
3. quantization 与 cache prediction 的 temporally correlated error 会在 closed loop 非加性叠加，因此 isolated video/action metric 会高估 deployment retention。
4. 最优方案可能不是 universal W1 + maximum cache，而是 component-wise precision 与 scene/action-aware cache budget 的 joint Pareto allocation。

## 7. Source boundary

- HB-VLA：identity / dates 来自 arXiv；method/results 来自 local v2；未验证 official code/venue。
- Content-Aware WorldCache：identity / dates 来自 arXiv；ECCV acceptance 来自 project/repository 与 official listing；method/results 来自 local v1。
- Heterogeneous Token WorldCache：identity / dates 来自 arXiv；ICML 2026 identity 由 arXiv comments、official accepted-paper listing 与 PDF 首页闭合；method/results 来自 local v2。
- 所有数值均为 authors' reported results；本次只做 acquisition、structure、locator 与 claim-boundary audit。
