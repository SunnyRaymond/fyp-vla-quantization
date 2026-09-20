# 106 — VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching

- 作者：Siyu Xu, Yunke Wang, Chenghao Xia, Dihao Zhu, Tao Huang, Chang Xu
- 版本：arXiv:2502.02175v2，2025-10-21，19 pages；accepted to NeurIPS 2025
- 定位：training-free、cross-frame visual-token KV reuse；重点是加速 VLA language decoder，而不是压缩 weights 或 action representation。
- Source：[arXiv abstract](https://arxiv.org/abs/2502.02175v2) · [HTML](https://arxiv.org/html/2502.02175v2) · [project page](https://vla-cache.github.io/) · [official code](https://github.com/siyuhsu/vla-cache)
- Local PDF：[paper-arxiv-v2.pdf](paper-arxiv-v2.pdf)

## 一句话抓手

VLA-Cache 先用相邻 frame 的 patch similarity 找出视觉上静止的 tokens，再从中排除 decoder attention 判定为 task-relevant 的 tokens，最后按各 decoder layer 的 attention entropy 决定复用多少上一时刻的 KV；它利用的是 **cross-frame temporal redundancy**，而不是 single-frame token pruning。

## Background

OpenVLA、OpenVLA-OFT、CogACT 等 VLA 会在 closed-loop control 中反复处理相邻 observation。背景和静止物体往往变化很小，但 language backbone 仍会为全部 visual tokens 重算 K/V。通用 VLM acceleration 方法多在单帧内 prune 或 merge tokens，未显式利用机器人 observation stream 的时间连续性。

这篇论文的核心观察是：视觉上静止不等于控制上无关。gripper、target object 或 contact region 即使像素变化很小，也可能对下一步 action 很关键；因此直接缓存所有静止 token 会破坏 closed-loop task success。

## Problem

目标是在不 retrain、不改变 checkpoint 的前提下，减少相邻 control steps 之间重复的 language-decoder computation，同时保留最新的 task-relevant visual information。

作者要同时回答三个问题：

1. 哪些 visual tokens 在相邻 frames 中足够静止，可以成为 reuse candidates？
2. 哪些视觉上静止的 tokens 仍然与当前 instruction / action decision 相关，必须重算？
3. 不同 decoder layers 的 attention concentration 不同，reuse ratio 是否应逐层变化？

## Method

### 1. Static Token Selection

将相邻 frames 切成对应 raw-pixel patches，对同一位置的 patches 计算 cosine similarity。超过 static threshold 的 patches 进入候选集，再保留最稳定的 Top-`k` tokens：

`P_static = Top-k({P_t | Sim(P_t, P_{t-1}) >= tau_static})`

这是 low-cost visual-change proxy；它只说明 patch 看起来没变，并不证明该 token 对 action 无关。

### 2. Evict Task-Relevant Tokens

从 decoder 的 text-to-vision attention 中聚合 heads 和 selected layers，得到每个 vision token 的 task-relevance score。高于 threshold 的 tokens 被视为 task-relevant，并从 reuse candidates 中移除：

`P_reuse = P_static \ P_task-relevant`

因此 dynamic tokens 与 task-relevant tokens 每一步都重新计算，只有“视觉静止且当前任务不重要”的 tokens 才可能复用。

### 3. Layer-Adaptive Token Reuse

作者用相邻 decoder layers 的 attention entropy reduction 构造每层 reuse ratio。attention 越集中，累计允许复用的比例越高；每一层分别决定 `P_reuse` 中实际跳过重算的 subset。

### 4. Cross-Frame KV Update

在 timestep `t` 的 decoder forward 中，被复用 token 继承 `t-1` 的 per-layer K/V；其余 tokens 用当前 hidden representation 重算 K/V。实现还维护 `cache_position`、attention mask 与 rotary embedding，使 partial cache update 对齐原 token positions。论文称最大收益出现在每个 timestep 生成第一个 action token 时。

## Key Innovation

这篇论文真正有辨识度的组合不是“有一个 KV cache”，而是：

- cache boundary 跨越相邻 robot observations，而不是只在同一 autoregressive query 内；
- reuse eligibility 同时需要 pixel-level temporal stability 与 decoder-level task irrelevance；
- reuse ratio 随 decoder layer 的 attention entropy 改变；
- 保留 closed-loop success、CUDA latency 与 control frequency 三种不同层级的证据。

## Main Results

### LIBERO / OpenVLA

- Average success rate：`75.0% -> 74.7%`（-0.3 percentage points）。
- FLOPs：`1.864T -> 1.355T`（论文报告 -27.31%）。
- CUDA latency：`51.91 ms -> 31.83 ms`（约 `1.63x`）。
- Control frequency：`4.23 Hz -> 4.59 Hz`。

这组数字清楚显示 module latency gain 不会等比例变成 end-to-end control-frequency gain。

### LIBERO / OpenVLA-OFT

- Average success rate：`96.8% -> 97.4%`。
- CUDA latency：`79.05 ms -> 62.59 ms`。
- Control frequency：`65.10 Hz -> 78.98 Hz`。

这是 VLA-Cache 与 action chunking / high-frequency architecture 可叠加的证据，但不是跨硬件或跨模型的统一 speedup guarantee。

### SIMPLER / CogACT

- Visual Matching average：`74.8% -> 74.4%`；latency `54.29 ms -> 39.63 ms`。
- Variant Aggregation average：`61.3% -> 62.3%`；latency `53.54 ms -> 39.11 ms`。

CogACT 带 diffusion policy head；结果说明 cache 可作用于其前端 VLM language decoder，不等于缓存 diffusion denoising state。

### Real Robot / OpenVLA

- Kinova Jaco2 上四个 tasks，每个 method 共 `100` trials。
- Total success：baseline `81/100`，VLA-Cache `84/100`；PickPot 单项从 `19/20` 降为 `18/20`，其余三项上升。
- Latency：`64.16 ms -> 51.85 ms`；control frequency：`4.02 Hz -> 4.21 Hz`。

这些数据支持“未观察到明显 task degradation”与 practical applicability，但 81/100 对 84/100 不足以单独证明 cache 会提高真实机器人成功率。

## Most Informative Ablations

- 只按 static similarity reuse：LIBERO-Spatial success `84.4% -> 74.2%`。
- 排除 task-relevant tokens 后：回升到 `82.6%`。
- 加入 layer-adaptive reuse：`83.8%`，latency `32.22 ms`。
- 当 reused/pruned tokens 增至 `200/256` 时，VLA-Cache success 降到 `68.3%`；moderate reuse 不能外推到 aggressive reuse。
- 在 OpenVLA-OFT 上，attention proxy 的 `98.3% / 61.12 ms` 优于 object-mask proxy 的 `87.4% / 87.49 ms`。

## Limitations / Evidence Boundary

- 所有主要 simulation timing 都在单张 RTX 4090、BF16 上完成；没有跨 GPU、batch size、kernel stack 或 edge hardware 的稳定性证据。
- `1.7x speedup` 指论文定义下的 CUDA latency headline，不应改写成 robot end-to-end speedup；OpenVLA 的 control frequency 只从 `4.23` 提到 `4.59 Hz`。
- VLA-Cache 复用 language-decoder visual-token KV。它不直接适用于没有 VLM backbone 的 standalone diffusion policy，也没有证明可安全缓存 world-model rollout、action-head denoising state 或跨 episode state。
- 选择规则依赖固定 patch correspondence；强 camera motion、occlusion、deformable objects、contact transitions 与 embodiment changes 可能破坏 raw-pixel similarity 的含义。
- Task relevance 由模型自己的 attention 近似；attention score 不是因果重要性证明，也没有直接验证 false-negative reuse 的安全界限。
- 论文对 similarity threshold 与 task-relevance threshold 的符号都使用 `tau`；阅读 Table 9–10 时要结合 Appendix D 的默认值区分 `tau_static=0.996` 与 `tau_task=0.5`。
- Real-robot results 是有限任务和有限 trials；论文没有给出 confidence intervals、significance test、long-horizon recovery 或 safety-critical failure taxonomy。
- Official repository 当前明确给出 OpenVLA 与 OpenVLA-OFT evaluation 路线；CogACT / SIMPLER 和 real-robot 完整复现覆盖应单独核对，不能由 paper result 自动推断为全量代码已发布。

## Why It Matters for the FYP

VLA-Cache 是研究 VLA / WAM caching 时必须正面区分的 direct prior：它已经覆盖 training-free、cross-frame visual-token reuse、adaptive invalidation 和 closed-loop task metrics。若本项目提出 cache，novelty 不能只停留在“相邻 observation 很相似”或“复用静态 token”。

对 DINO-WM / Fast-WAM 方向，最重要的边界是：

- VLA-Cache 的 reuse unit 是 VLA language decoder 中的 visual-token KV；world model 的 rollout latent、action-conditioned branch 与 planner ranking 不是同一对象。
- 它使用 observation similarity + attention relevance 做 approximate reuse；exact action-independent prefix reuse 或 downstream-risk-based refresh 需要用各自的 correctness contract 证明。
- 公平比较必须报告 native wall-clock、cache overhead、peak memory、first-action / planner decision、closed-loop success，并固定 backbone、hardware 与 control protocol。

## Reading Route

### 20 minutes — 抓住机制与 claim boundary

1. Abstract + Section 1：圈出 `training-free`、`1.7x CUDA latency`、`15% control frequency` 各自的测量对象。
2. Figure 2 + Sections 3.2–3.4：画出 `static candidates -> task-relevant eviction -> layer-adaptive reuse`。
3. Table 1：理解为什么 visual similarity alone 会失败。
4. Table 2：对比 CUDA latency 与 control frequency 的 gain gap。

### 90 minutes — 读到可以解释与质疑

1. Section 3：逐式标出 `tau_static`、Top-`k`、`tau_task`、entropy ratio 与 per-layer `alpha_l`。
2. Section 4 + Appendix D：追踪 `cache_position`、mask、RoPE、K/V partial update 的执行边界。
3. Tables 2–5：分别记录 OpenVLA、OpenVLA-OFT、CogACT 和 real robot 的 task / hardware / metric。
4. Tables 4、8–10：找出 aggressive reuse、proxy choice 与 threshold sensitivity 的失败区间。
5. Appendix E.4 + Table 11：按 raw trial counts 而不是 average percentage 解读 real-robot evidence。

### 3 hours — 形成可用于 FYP 的 prior-art card

1. 从 official code 找到 OpenVLA 与 OpenVLA-OFT 的 cache insertion points，核对 paper pseudocode 与实际 forward path。
2. 建一张表：`reuse unit / validity signal / invalidation scope / saved compute / overhead / task metric / hardware`。
3. 选择至少一个 negative control：static-only reuse 或 aggressive `k`，检查失败是否与 Table 1 / 4 一致。
4. 将 VLA-Cache 与 [OpenVLA](../001-openvla/README.md)、[OpenVLA-OFT](../012-openvla-oft/README.md) 和 [Efficient VLA survey](../105-efficient-vla-survey/README.md) 放在同一 evidence boundary 下，不跨 backend 比 headline speedup。

## Reading Questions（留给你回答）

1. VLA-Cache 实际跳过的是 vision encoder、language decoder 的 K/V projection、attention/MLP，还是它们中的哪一部分？
2. raw-pixel patch cosine similarity 与 decoder token position 如何一一对应；resize、crop、multi-camera 输入会怎样影响 correspondence？
3. 为什么视觉上静止的 gripper / target tokens 仍可能需要重算？Table 1 能否排除其他 confounder？
4. `P_task-relevant` 的 attention 是从上一 timestep、当前 partial forward，还是特定 layers 取得；这部分 overhead 是否计入 latency？
5. attention entropy 越集中就允许复用更多 tokens 的假设，在哪些 layer 或 task 上可能反向？
6. 作者所称 partial KV update 的“valid attention results”是数学 exactness、shape validity，还是 empirical approximation？
7. 如果 reused token 的 K/V 保持旧值，但 query、other tokens 和 residual stream 已变化，误差会如何沿 layers 传播？
8. Table 2 的 CUDA latency boundary 包含 similarity、attention-score aggregation、sorting、cache gather/scatter 与 synchronization 吗？
9. 为什么 OpenVLA 的 decoder latency 提升约 `1.63x`，control frequency 却只提升约 `8.5%`？剩余 bottleneck 在哪里？
10. OpenVLA-OFT 的 frequency gain 为什么与 OpenVLA 不同；action chunking、asynchrony 或 observation cadence 各起什么作用？
11. Table 4 的 `200/256` aggressive reuse 为什么仍有 latency gain却明显掉成功率；能否定义 online fallback？
12. Real-robot `81/100` 与 `84/100` 的差异在什么统计假设下可区分；per-task trial allocation 会怎样影响结论？
13. dynamic background experiment 是否覆盖 camera motion、occlusion 和 task-relevant moving distractors，还是只覆盖 irrelevant motion？
14. 对 CogACT 而言，VLA-Cache 加速了哪些 VLM computations，哪些 diffusion-action computations 完全未触及？
15. Official code 是否完整实现 paper 中的 layer-adaptive entropy、task-relevant eviction 与所有 benchmark paths？
16. 如果把方法迁移到 world model / WAM，reuse unit、invalidation event 与 correctness metric 必须怎样重定义？

## Meeting Card

- **Paper**：Xu et al., *VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching*, arXiv:2502.02175v2 / NeurIPS 2025。
- **Core idea**：相邻 frames 中只复用“视觉静止且 task-irrelevant”的 visual-token KV，并按 decoder-layer attention entropy 调节 reuse ratio。
- **Strongest evidence**：OpenVLA LIBERO 的 latency `51.91 -> 31.83 ms`、average success `75.0% -> 74.7%`；另有 OpenVLA-OFT、CogACT 和 100-trial-per-method real-robot evaluation。
- **Critical caveat**：headline CUDA latency 不是 full control-loop speedup；attention relevance 是 proxy，cross-frame reuse 也不是 exact computation preservation。
- **FYP connection**：它占据 VLA cross-frame approximate KV reuse prior；新的 WM/WAM cache 必须明确不同的 computation boundary、validity signal 与 downstream correctness contract。
- **Question to bring to meeting**：我们要证明的是“representation 足够相似”，还是“planner/action decision 在明确误差预算内不变”；VLA-Cache 的证据能覆盖其中哪一层？

## Citation

```bibtex
@article{xu2025vlacache,
  title   = {VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching},
  author  = {Xu, Siyu and Wang, Yunke and Xia, Chenghao and Zhu, Dihao and Huang, Tao and Xu, Chang},
  journal = {arXiv preprint arXiv:2502.02175},
  year    = {2025},
  version = {v2},
  note    = {Accepted to NeurIPS 2025}
}
```
