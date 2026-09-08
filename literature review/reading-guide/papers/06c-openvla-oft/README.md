# Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success (OpenVLA-OFT)

> **Reading-list role**: BitVLA companion — immediate action-decoding and downstream adaptation source  
> **Verification**: `verified-full-text`; local file is arXiv v2, work accepted to RSS 2025  
> **Recommended effort**: **Deep read**

> **Local full text**: [paper.pdf](paper.pdf)

## 1. Paper identity

| Field | Value |
|---|---|
| Authors | Moo Jin Kim, Chelsea Finn, Percy Liang |
| Year / version | arXiv 2502.19645 v2, 2025-04-28 |
| Venue | Robotics: Science and Systems (RSS) 2025 |
| Primary source | [RSS proceedings](https://www.roboticsproceedings.org/rss21/p017.html) · [arXiv](https://arxiv.org/abs/2502.19645) |
| Project / code | [Project page](https://openvla-oft.github.io/) · [Official repository](https://github.com/moojink/openvla-oft) |

## 2. One-sentence takeaway

OpenVLA-OFT 不改变 OpenVLA pretraining，而是在 downstream fine-tuning 时用 empty action embeddings、parallel decoding、action chunking、continuous actions 与 L1 regression，将 LIBERO average success 从 76.5% 提到 97.1%，同时把 action-generation throughput 提高约 26×。

## 3. Problem

- Original OpenVLA 以 autoregressive next-token prediction 输出 7 个 discrete action dimensions；一次 action 需要多次 decoder forward pass。
- 若直接输出 `K`-step chunk，naive autoregressive cost 进一步变成 `K×D` sequential predictions。
- Downstream robot setup 可能加入 wrist cameras、proprioceptive state、不同 action dimension 与更高 control frequency；base recipe 缺少这种 flexibility。

## 4. Method

### 4.1 Base OpenVLA

`third-person image → fused SigLIP + DINOv2 → 256 visual patch tokens → 3-layer projector → Llama-2 7B + language → 7 discrete action tokens`

动作先 normalize 到 `[-1,1]`，每个 dimension discretize 为 256 bins，再用 Cross-Entropy autoregressively decode。

### 4.2 OFT modifications

1. **Parallel decoding**：把 shifted ground-truth action tokens 换成只由 position encoding 区分的 empty action embeddings；将 causal attention 改成 bidirectional attention，一次 forward 同时预测全部 action positions。
2. **Action chunking**：增加 empty action positions，一次输出 `K×D` continuous values。
3. **Continuous action representation**：用 4-layer ReLU MLP action head 替换 vocabulary output layer。
4. **L1 regression**：minimize normalized predicted/ground-truth action 的 mean absolute error。
5. **Flexible inputs**：每个 camera view 产生 256 visual tokens；robot state 经 2-layer MLP 变成一个 embedding；所有 image/state/language tokens 沿 sequence dimension concatenate。
6. **OFT+ / FiLM**：ALOHA setting 用 language embedding 对 SigLIP 与 DINOv2 intermediate features 做 scale/shift，改善 language grounding。

Training 主要通过 LoRA adaptation；pretrained OpenVLA representation 保留，paper 的 ablation 显示去掉 VLA pretraining 会让 LIBERO average drop 5.2 points。

### 4.3 OFT 与 flow/diffusion 的关系

OFT paper 实际比较了 L1 与 conditional diffusion，但最终 recipe 选择 L1：

- `Cont-L1`：single forward，直接回归 action chunk conditional median。
- `Cont-Diffusion`：从 noisy chunk 多次 denoise；50 steps 时更慢，减少到 2/1 steps 会显著掉 performance。
- `π₀ flow matching`：也是 continuous generative action chunk，但用 learned vector field + ODE integration，通常约 10 steps；不是 OpenVLA-OFT architecture 的组成部分。

BitVLA 只继承 **OFT-style parallel continuous L1 action chunking**，没有实现 flow matching。

## 5. Experiments and main results

| Claim | Evidence | Locator | Caveat |
|---|---|---|---|
| PD+AC 是最大 single adaptation gain | OpenVLA avg 76.5 → PD+AC discrete 90.2 | Table I, PDF p. 6 | data filtering 与 fine-tuning details 必须 matched |
| Continuous L1 与 diffusion 在 LIBERO 接近 | L1 95.3 vs 50-step diffusion 95.4（same one-view group） | Table I, p. 6 | focused demonstrations may be largely unimodal |
| Extra wrist image/state further improve result | full OFT with additional inputs reaches 97.1% | Table I, p. 6 | input modality 与 training set group change together |
| Throughput improves about 26× | OpenVLA 4.2 Hz vs PD+AC+L1 109.7 Hz；latency 0.2396 vs 0.0729 s | Table II, p. 6 | Hz = generated actions/s, not query rate or control-loop rate |
| ALOHA high-frequency adaptation works | OFT+ runs K=25 chunks on bimanual ALOHA at 25 Hz | Sec. VI, pp. 7–10 | small task-specific datasets；full chunks executed open-loop |

## 6. Three-image input explained

ALOHA 有三路 synchronized camera views：

1. one top-down / third-person camera；
2. left-wrist camera；
3. right-wrist camera。

每张 224×224 image 都独立通过同一个 shared fused vision encoder，得到 256 patch embeddings；three views 因而形成 768 visual tokens，再与 14-D joint-state embedding、language tokens 与 action-query embeddings concatenate。它们是 **同一 control timestep 的 multi-view observations**，不是 video frames、history frames，也不是 flow-matching trajectory states。

Table III 的 benchmark 使用 `3 images + 14-D state + command + K=25`，因为它复制 ALOHA/OpenVLA-OFT+ input specification。BitVLA 自己的 LIBERO Sec. IV-B 通常是 `external camera + wrist camera` 两路；Fig. 6 efficiency benchmark 则沿用 ALOHA three-view shape。二者不应混为一个 setting。

## 7. Limitations

- L1 对 truly multimodal demonstrations 只能学 conditional median；paper 自己将此列为 limitation。
- Diffusion comparison 使用 authors' implementation/config；algorithm、step count 与 runtime framework 同时变化。
- LIBERO report 选择 best checkpoints，且多个 baseline 来自原论文；不是全部在单一 codepath 重跑。
- ALOHA demonstrations/task count 较小，三路 camera + FiLM 的效果不能直接外推到 arbitrary robots。
- `throughput = actions generated per second` 会被 chunk size 放大；single-query latency 与 replan frequency 必须同时报告。

## 8. Why it matters for BitVLA

- BitVLA 的 73 ms / 341.1 Hz headline 正是用 OpenVLA-OFT ALOHA setup 和 baseline numbers 对齐。
- BitVLA 在 causal mask 下保留 OFT-style action-query tokens；这与 original OFT 的 bidirectional mask 有差别，paper 表示 BitNet 使用 bidirectional mask 会损伤 real-world performance。
- 若要把 flow matching 加入 BitVLA，最公平 baseline 必须是同一 BitVLA backbone/input/chunk、只替换 action objective/head；否则会把 OFT gain、multi-view gain 与 flow gain 混在一起。

## 9. How to read it

### 20-minute route

1. Fig. 2 + Sec. IV-B（PDF pp. 3–4）：parallel vs autoregressive；discrete vs L1/diffusion。
2. Tables I–II（p. 6）：拆分 PD、AC、continuous representation、L1。
3. Sec. VI-A + Table III（pp. 7–10）：three cameras、14-D state、K=25。
4. Sec. VIII（p. 10）：L1 multimodality 与 pretraining/fine-tuning boundary。

### 90-minute route

1. 画出 `K×D` empty action embeddings 与 attention mask。
2. 手算 `throughput = K / latency`，并与 query rate/control rate 分开。
3. 对照 Table I 的 data/input groups，避免跨 group 直接做 causal claim。
4. 读 Appendix A/B：six architecture changes、4-layer head、state projector、FiLM。
5. 对照 [π₀](../02-pi0/README.md) 的 flow action expert，设计 matched action-head ablation。

## 10. Reading questions

1. PD+AC 的 14-point gain 中，多少来自 temporal smoothing，多少来自更少 compounding decoding error？
2. causal action queries 与 bidirectional action queries 在 expressivity/parallelism 上分别意味着什么？
3. 为什么 L1 在 focused demonstrations 上能接近 50-step diffusion？
4. `K=25` full-chunk execution 在 disturbance recovery 上付出什么代价？
5. three-view tokens 线性增加时，vision encoder、decoder attention 与 action head 谁成为 latency bottleneck？

## 11. Weekly meeting card

- **Problem**：OpenVLA downstream adaptation 慢、input/output inflexible。
- **Key idea**：parallel empty action queries + chunks + continuous L1 head；ALOHA 再加 FiLM。
- **Best evidence**：LIBERO 76.5 → 97.1；throughput 4.2 → 109.7 actions/s（Tables I–II）。
- **Biggest limitation**：L1 multimodality、small task-specific real-world data、throughput metric 易误读。
- **Connection to BitVLA**：BitVLA 的 final action stage 和 efficiency benchmark 都直接依赖 OFT recipe/setup。

## 12. Evidence boundary

- **Source claims**：architecture、three-camera ALOHA setup、results 与 limitations 来自 RSS/arXiv v2 full text。
- **My interpretation**：OFT 的核心贡献是 adaptation interface，不是新的 pretrained VLA backbone。
- **Primary links**：[RSS](https://www.roboticsproceedings.org/rss21/p017.html) · [arXiv](https://arxiv.org/abs/2502.19645) · [Project](https://openvla-oft.github.io/)
