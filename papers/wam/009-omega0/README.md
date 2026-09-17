# ω-0: A Latent Predictive World Action Model for Concurrent Humanoid Loco-Manipulation

> **Reading-list role**: Critical scan — very recent humanoid World Action Model candidate  
> **Verification**: `verified-full-text`; arXiv preprint  
> **Recommended effort**: **Critical scan**，先验证 evidence scope 再决定是否 deep read

> **Local full text**: [paper.pdf](paper.pdf)

## 1. Paper identity

| Field | Value |
|---|---|
| Authors | Zhe Li, Zhenzhe Zhang, Yangyang Wei, Wenjie Zhang, Xichen Yuan, Peiyuan Zhi, Gen Li, Xinying Guo, Fengjie Gao, Jianfei Yang, Shanghang Zhang |
| Year / version | 2026; arXiv 2608.06375, v1 2026-08-06, latest checked v2 2026-08-09 |
| Venue / status | arXiv preprint; no peer-reviewed venue verified as of 2026-08-22 |
| Primary source | [arXiv:2608.06375](https://arxiv.org/abs/2608.06375) · [DOI](https://doi.org/10.48550/arXiv.2608.06375) |
| Code / project | No separate official project/code release verified in the evidence packet |

## 2. One-sentence takeaway

ω-0 在 training 时联合预测 future video latents 与 SONIC-compatible whole-body action latents、inference 时只 denoise action，在自建 Unitree G1 的 11-task suite 上报告 81.8% success，但证据仍是单平台、自建数据的极新 preprint。

## 3. Background and prerequisites

- **Technical lineage**：VLA → video-based World Action Model → latent predictive representation → humanoid whole-body control。
- **读前知识**：V-JEPA/video latent、Diffusion Transformer、DDIM、FAST tokenization、SMPL/SMPL-X、motion retargeting、receding-horizon control、humanoid balance/whole-body controller。
- **关键 distinction**：`training-time future latent prediction` 不等于 `test-time video generation/planning`。

## 4. Problem

- **Target setting**：Unitree G1 在 household tasks 中 concurrent locomotion + torso + balance + bimanual/dexterous manipulation。
- **Bottleneck**：arm-centric VLA 不原生表示 whole-body control；test-time pixel video generation 有 latency 和 temporal-error amplification；只监督 action 又缺少 task progress/scene evolution signal。
- **Why previous methods are insufficient**：已有 WAM 多集中 tabletop/arm control，action representation 也不一定与 low-level humanoid controller compatible。

## 5. Method

### 5.1 System view

Training：`ego/exo image + language + robot state + future video` → frozen V-JEPA/T5/VLM/Wan targets + future-aware queries → joint predictor → video-latent auxiliary loss + action-DiT denoising loss。  
Deployment：`current image + language + proprioception + previous chunk prefix` → action DiT/DDIM → 66-D SONIC-compatible action latent → SONIC low-level controller → humanoid motion。

### 5.2 Core mechanism

**Stage 1 — Whole-body Action VLM.** 用 whole-body FAST tokenizer 将 unified SMPL trajectory 转成 discrete tokens；Qwen3-VL-2B-Instruct 根据 language、ego/exo observation 与 view token 预测 action tokens。

**Stage 2 — Human-to-humanoid pretraining.** Public human motion 经 SONIC simulation replay，得到 robot state 与 executable action latents；frozen Wan encoder 提供 future-video latent target，V-JEPA2.1 编 current image，T5 编 language；future video queries 与 action queries 联合训练。

**Stage 3 — Real data fine-tuning.** 用 ω-HOME real-robot data fine-tune query/state/fusion/action-DiT modules；V-JEPA、Wan 与 VLM 保持 frozen。`RTC` 用上一 action chunk 的 clean prefix 约束新 chunk continuity。

**Inference.** DDIM 直接采样 clean action latent，不生成 pixel video；约 0.14 s（>7 Hz）。horizon `H=25`，执行前 `K=8` actions 后 replan。（Appendix B, PDF p. 23）

**Data.** ω-HOME：40.3 hours、4,827 episodes、24 tasks、30 Hz，含 synchronized language、ego RGB、exo RGB-D、proprioception、whole-body motion/action latents。下游 11 tasks 共 2,220 trajectories、约 200 demonstrations/task；pretraining pool 排除这 11 tasks。（Sec. 4–5, PDF pp. 8–10）

### 5.3 What is actually new

真正新增的是：以 latent future prediction 作为 action-relevant training supervision，同时输出 controller-compatible whole-body latent；测试时不做 video generation。Qwen/V-JEPA/T5/Wan/DiT/FAST/SONIC 都是已有模块，贡献是接口与 staged pipeline 的组合。

## 6. Experiments and main results

| Claim | Evidence | Locator | Caveat |
|---|---|---|---|
| ω-0 大幅超过 paper 内 baselines | 11 tasks，10 trials/task/method：ω-0 Ego **79.1% SR / 35.8 of 41 score / 88.7% progress**；Omni **81.8 / 36.7 / 90.3**。ψ-0 SR **44.5%**；DiT4DiT progress **61.0%** | Table 2, PDF p. 13 | 单一 Unitree G1、自建 dataset/task/protocol；Omni 在部分任务用 exocentric view |
| Future-video query 对 full model 很重要 | no video query **64.5% SR / 30.6 / 77.9%** vs full Ego **79.1 / 35.8 / 88.7**；no state 60.9% SR、no VLM prefix 66.4%、no RTC 71.8% | Table 4, PDF p. 14 | component removal 改变 representation capacity，但尚不证明 learned dynamics 是 causal/physical |
| Video query 改善 distribution shift | cross-object SR **66.7→83.3%**，cross-scene **15.0→79.5%**，human transfer **20.0→60.0%** | Table 5, PDF p. 18 | 同论文、同研究组 ablation；不是 external replication |
| ω-HOME pretraining 有小幅增益 | Ego 79.1→80.4% SR；Omni 81.8→82.4%，对应 score/progress 也小升 | Table 3, PDF p. 13 | gain 较小；需与额外 data/compute 对照理解 |

## 7. Limitations

### Authors' stated limitations

论文没有独立 limitations section。作者在 Sec. 6.4（PDF pp. 13–14）明确说明 egocentric camera 对 global displacement、stepping pattern 与 torso adjustment 可见性有限，因此 Omni 在 5 个 locomotion-heavy tasks 改用 room-view exocentric observation。

### My critique

- **Internal validity**：video-query ablation 支持“auxiliary target 有用”，但不足以证明 representation 学到 causal world dynamics，而不只是 task-progress correlation。
- **External validity**：单一 Unitree G1 + SONIC、自建 11 tasks、10 trials/task；跨 controller、humanoid、site transfer 未验证。
- **Systems validity**：Omni 的 room-mounted camera 是更强 sensing assumption；与 onboard-only baselines 比较必须显式披露。
- **Reproducibility**：截至证据核验没有 peer review 或独立 code/project release；public human data 还经过 SONIC-executability filtering，可能有 selection bias。

## 8. Why it matters for this project

- 它提供一种 WAM 设计选择：world prediction 可以只作为 training-time representation objective，而不必在 test-time 生成 pixels。
- whole-body action latent 将 high-level VLA 与 low-level humanoid controller 接起来，适合研究 representation/controller interface。
- 对 efficiency，latent target + frozen encoders 把昂贵 world modeling 留在 training；deployment 只做 action denoising，但仍需评估 0.14 s 是否满足 disturbance recovery。
- 因为 evidence 很新且自建，当前更适合作为 **Critical scan / idea source**，不应先于 OpenVLA、π₀深读。

## 9. How to read it

### 20-minute route

1. Abstract + Fig. 2 pipeline：用一句话回答 video prediction 在 train 还是 test。
2. Sec. 3.2–3.4（PDF pp. 4–8）：只追 three-stage training 与 frozen/trainable modules。
3. Table 2（p.13）+ Table 4（p.14）：看 main result 与 no-video-query ablation。
4. Sec. 6.4（pp.13–14）：找 Omni 的 exocentric sensing assumption。

### 60-90-minute route

1. 复习 V-JEPA latent、DDIM 与 SONIC action latent。
2. 画 Stage 1/2/3，每个 encoder/target/module 标 frozen 或 trainable。
3. 解释 human motion 如何经 simulation replay 变成 robot-executable supervision，以及过滤带来的 bias。
4. 核对 Table 2/3/4/5；区分 in-domain、pretraining gain、component ablation、generalization。
5. 阅读 Appendix B：用 `0.14 s, H=25, K=8` 推理实际 replan cadence。
6. 写下一点尚未相信的 claim：future latent 是否比更强 visual/state encoder 本身更有用？

## 10. Reading questions

1. video-query ablation 的 +14.6 SR points 足以证明 causal dynamics learning 吗？
2. frozen Wan target 会带入什么 video-generation bias？
3. SONIC replay 丢弃不可追踪 motion 会造成什么 selection bias？
4. Omni view gain 与 world-model gain 如何解耦？
5. action latent 换 low-level controller 后还能 transfer 吗？
6. >7 Hz inference 对 humanoid disturbance recovery 是否足够，哪些 loops 由 SONIC 更高频处理？

## 11. Weekly meeting card

- **Problem**：怎样让 humanoid 同时 locomote 与 manipulate，并利用 future prediction 而不在 test-time 生成 video？
- **Key idea**：training 时 joint future-video/action latent，deployment 只 DDIM sample SONIC-compatible action latent。
- **Best evidence**：Omni 81.8% SR vs strongest baseline SR 44.5%（Table 2, p.13）；去 video query 降到 64.5%（Table 4, p.14）。
- **Biggest limitation**：单平台、自建 benchmark、10 trials/task；Omni 使用外部 camera。
- **Question for the group**：这篇证明的是 world dynamics，还是 future-aware representation regularization？

## 12. Evidence boundary

- **Source claim**：three-stage pipeline、ω-HOME statistics 与 exact tables 均来自 original preprint 的上述 locator。
- **My interpretation**：ω-0 更准确属于 “world modeling as representation learning”，而非 explicit rollout planner。
- **Open question**：跨 robot/controller transfer、causal dynamics claim 与 onboard-only performance 尚未确认。
- **Primary links**：[arXiv](https://arxiv.org/abs/2608.06375) · [DOI](https://doi.org/10.48550/arXiv.2608.06375)

