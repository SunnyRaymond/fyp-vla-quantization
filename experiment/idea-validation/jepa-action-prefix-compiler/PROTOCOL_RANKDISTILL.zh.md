# Planner-Aware Rank Distillation — Frozen Protocol

## Hypothesis

Stage A 已表明 latent-MSE-only student 很快且没有 future-action leakage，但 candidate ranking 不足。新实验检验：在不改变 student inference interface 的前提下，训练时加入 frozen planner-score distillation，是否能修复 held-out ranking。

这不是给失败 recipe 增加 steps 或放宽阈值，而是更换 supervision mechanism。上一轮 held-out seeds 不再使用。

## Design

同一个 A100 job 内并发训练两个 paired arms：

- `latent_only`：原 dense native-latent MSE control；
- `rankdistill`：完全相同的 latent MSE，加 listwise planner-score KL。

两 arm 共享初始 weights、每一步的 anchor、action prefixes、teacher targets 与 batch schedule；update order 按 step 交替。每一步只使用一个 anchor，避免跨不同 goal 的 candidates 排名。实验以 `anchor × fresh held-out seed` 为四个 paired blocks；每个 block 内 300 candidates 是 nested measurements，不作为 300 个独立样本，也不做 population-level p-value claim。

## Planner-aware loss

对同一个 anchor/goal 下的一批 candidates，使用 official terminal latent-to-goal objective 得到 teacher/student costs。分别在 batch 内 z-score，再形成：

\[
p^T=\operatorname{softmax}(-\tilde c^T/\tau),\qquad
\log p^S=\operatorname{logsoftmax}(-\tilde c^S/\tau)
\]

\[
\mathcal L_{rank}=D_{KL}(p^T\|p^S),\qquad
\mathcal L=\mathcal L_{latent}+0.1\mathcal L_{rank},\quad \tau=1.
\]

Goal 只参与 loss；student inference 输入仍是 native latent 与 action prefix，不能读取 goal 或调用 `encode_obs`。

## Frozen schedule and gates

- training seed `20260920`，500 steps，batch 32；两个 anchors 逐 step 交替。
- fresh held-out seeds `20262920/20262921`；每 anchor/seed 300 candidates。
- absolute：四个 blocks 的 median rankdistill Spearman `>=0.99`、median top-30 overlap `>=0.95`；同时 minimum block 分别不得低于 `0.95/0.80`，避免一个 anchor 掩盖另一个。
- paired improvement：四个 blocks 的 median ΔSpearman `>=0.05`、median Δtop-30 overlap `>=0.10`，且两项都至少 `3/4` blocks 为正。
- native-latent non-inferiority：rankdistill/control mean relative-MSE ratio `<=1.25`。
- 两 arm latent-MSE component 的 last-10/first ratio `<=0.8`、causality/finite PASS；rankdistill predictor latency reduction versus teacher `>=20%`。

任一项失败即 no-go，不进入 CEM integration。即使通过，也只证明 two-anchor predictor-level ranking smoke，不证明 full planner、closed-loop、LeWM transfer 或 observation-cache composition。

## Method reference

Paired blocking、concurrent control 与 nested-unit边界参考：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
