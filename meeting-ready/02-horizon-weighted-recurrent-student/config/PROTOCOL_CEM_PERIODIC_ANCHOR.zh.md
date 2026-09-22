# CEM periodic teacher-anchor protocol

本实验只回答一个固定 observation 的 mechanism 问题：在 cheap recurrent student 的多步 CEM rollout 中，每隔五轮用一次 teacher 对当前 300 个 candidate pool 进行 anchor，能否抑制 distribution drift。它不运行 environment，不声称 closed-loop success，也不把 candidate 或 iteration 当作独立样本。

## Frozen setup

- cases：`[0, 1, 2, 4, 5, 7]`；eval seeds：`[1, 100, 199, 397, 496, 694]`；identity 必须与 parent `24544733.pbs101` 一致。
- fixed observation、`30` iterations、`300` candidates、`top30`、`H=5`、packed action dim `10`。
- 每个 case/round 使用同一份 CPU standard-normal innovations，seed 为 `20261100 + case_index`；`candidate 0` 是 pre-update mean；排序为 plain `torch.argsort`；elite standard deviation 为 `torch.std` 的默认 unbiased estimator。
- 诊断 checkpoints 为 `[1, 5, 10, 15, 20, 25, 30]`。primary anchor rounds 是 1-based `[5, 10, 15, 20, 25]`，明确不包含 round 30。

三臂为：`student_only`（每轮 student top30）、`periodic_teacher_anchor_P5`（非 anchor 用 student top30，anchor 对当前 pool 全 300 个 candidate 用 teacher top30）、`shuffled_anchor_P5`（anchor 使用同一份 teacher costs，但用冻结 CPU permutation 打乱 cost-action 对应后选择 top30）。两臂共享每个 case/anchor round 的 permutation seed。

## Teacher positive reference

Historical job `24910110.pbs101` 只保留为 frozen provenance（case/seed/checkpoint identity）；它的 case JSON 不作为运行时 reference，因此不受历史 artifact 只保存 `1,5,10,30` checkpoints 的限制。每个 case 在当前 compute job 中使用官方 teacher、同一份 common CPU innovations，完整重算一次 30-round teacher-full trajectory，保存每轮 post-update `mu/sigma`，并作为三个 intervention arms 共享的 positive reference。该 reference 的 `30 calls × 300 candidates` 计入独立 `reference_teacher_*` budget，不计入任一 intervention arm 的 teacher budget；不重复为每个 arm 重算。

## Measurements and accounting

每轮保存相对当前 job teacher reference 的 `mu_rms`、`sigma_rms`、`first_action_rms`、`first_action_coordinate_abs_max`。每个诊断 checkpoint 保存 teacher top30 overlap 与 selected teacher-cost regret。anchor 轮的 diagnostic teacher call 与 mechanism teacher call 必须复用，不重复计费；其它 checkpoint 的 teacher call 计入 `diagnostic oracle candidates`。每个 arm 还保存完整 30 轮 first-action RMS trajectory，并以其均值作为 case-level trajectory AUC（这里是离散 trajectory mean）。

计数分开记录 `student_candidates`、`mechanism_teacher_candidates`、`diagnostic_oracle_candidates`；所有 outputs 必须 finite，禁止 silent fallback。teacher reference 自身以 zero drift 作为 validity consistency check。

## Primary gate

`periodic_teacher_anchor_P5` 必须同时满足：

1. validity 完整且 reference consistency zero drift；
2. iteration 30 first-action RMS median `<= 0.15`；coordinate absolute maximum `<= 0.25`；
3. 相对 `student_only`，iteration-30 RMS median 严格更低且至少 `4/6` cases 不 worse；
4. case-level trajectory AUC median 严格更低且至少 `4/6` 不 worse；
5. iteration-30 teacher-cost regret median 不高于 `student_only` 且至少 `4/6` 不 worse。

`shuffled_anchor_P5` 不得同时复现 iteration-30 RMS median 的下降和 trajectory-AUC median 的下降，且两项都达到至少 `4/6` cases 不 worse。任一 primary gate 失败都停止在 fixed-observation mechanism diagnosis，不运行 closed-loop。

## Artifacts

runner 输出 `cem_periodic_anchor_summary.json`、每 case 一个 `case_XX.json`，以及 PBS wrapper 产生的小型 `job.log`、status、exit、GPU telemetry、identity 和 `COMPLETE/FAILED` marker。checkpoint、dataset、runtime archive 与其它大文件不取回。
