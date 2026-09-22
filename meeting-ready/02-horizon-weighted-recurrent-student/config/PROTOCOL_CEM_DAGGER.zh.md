# CEM-DAgger：offline train-split predictor diagnosis protocol

## 边界

本 cell 不运行 student closed-loop，也不创建或调用 environment。`on-policy` 的含义是：在 frozen train-split observation context 上，用 horizon-weighted step-1500 student 驱动 CEM proposal distribution；teacher 只为这些 proposal 生成 detached dense native-latent labels。

训练完成后只做 predictor-level held-out gate 和 fixed-observation CEM mechanism gate。六个 failed evaluation cases 不进入训练 label pool。

## 固定 collector

- 选择 frozen train manifest 中按 episode ID 排序的前 8 个不同 train episodes，每 episode 1 个 context。
- CEM：`M=300`、`K=30`、30 iterations，candidate 0 强制为 pre-update `mu`。
- proposal cost 使用 warm-start student；teacher 不参与 proposal distribution。
- 在 iterations `1,5,10,20,30`，对 student top-120 候选（保持 candidate 0 且 cardinality 精确为 120）运行 teacher dense rollout label。
- 总 label rows：`8 × 5 × 120 = 4800`。
- 使用 explicit CPU innovations；保持 plain `argsort` 和 `torch.std` 默认 unbiased semantics。

## 两个 fine-tune arms

- `treatment`：每 update 16 个 CEM-DAgger rows + 16 个 exact old offline slate-replay rows。
- `replay_control`：每 update 32 个 exact old offline slate-replay rows。
- 两个 arm 从同一个 horizon-weighted step-1500 state dict 开始，使用同一 AdamW、seed、500 updates 和 horizon weights `[1/3,2/3,1,4/3,5/3]`。
- replay 使用既有 256-context × 4-query slate bank（`action_slate_seed=20260932`、`cem_seed=20260933`）的 immutable teacher targets，不重新采样旧 bank。
- 不使用 rank loss、teacher forcing、goal input、architecture change 或额外 snapshot；正式 checkpoint 只有 step 500。

## Stage 1 predictor gate

复用现有 held-out 的 8 contexts、2 action-prefix seeds、16 blocks、300 candidates 和 candidate order。要求 treatment：

- treatment 与 replay-control 两臂都必须通过 finite、causality 与 ranking guard；
- Spearman median ≥ 0.95、minimum ≥ 0.85；
- top-30 overlap median ≥ 0.80、minimum ≥ 0.50；
- causal future-action leakage ≤ `1e-6`；
- outputs finite，且没有 silent fallback。

同时报告 `treatment` 与 matched `replay_control`，但不使用旧 recurrent replacement 的 `0.99/0.95` gate。

## Stage 2 fixed-observation CEM gate

只有 Stage 1 两臂都通过才运行。使用既有六个 failed cases、对应 eval seeds、`M=300/K=30/30` rounds、checkpoints `1,5,10,30` 和 CPU innovations；这些 checkpoints 与 frozen historical teacher reference 完全对齐。

- 不重跑 historical teacher-full；读取历史 CEM trace summary/case JSON 作为只读 teacher/student reference。
- 新 treatment 与 replay-control student pool 在 checkpoint 处做 teacher shadow。
- treatment iteration-30 first-action RMS median ≤ 0.15；coordinate absolute max ≤ 0.25；teacher-cost support regret median 不高于 historical student-only median，且至少 4/6 cases 不 worse。
- treatment 相对 replay-control 要求 iteration-30 drift median 严格更低，且至少 4/6 cases 不 worse。

Stage 2 仍是 fixed-observation mechanism diagnosis，不产生 closed-loop success 或 deployment claim。

## PBS

`jobs/dino_pusht_cem_dagger.pbs` 是 single bounded GPU job。它只在 compute node 解压已 staged runtime，拒绝 login/head/submit node，并每 30 秒将 GPU utilization/VRAM 记录进 `job.log`。不在 login node 下载、安装、编译、加载模型或运行 benchmark。
