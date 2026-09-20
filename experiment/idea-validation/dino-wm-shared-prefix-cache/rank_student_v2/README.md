# Rank Student V2 Stage A

这是一个全新的 V2 architecture/data redesign，不是对 V1 gate 的调参。V1
`23895140.pbs101` 的结果是诚实的 `no-go`，必须作为 prior failed recipe 保留；V2
不重解释 V1、不混入 V1 observation/checkpoint，也不把 V1 的失败改写成 V2 证据。

V2 Stage A 只做 full-teacher teacher data、train-only student training、calibration
fixed-M selection 和 held-out offline ranking screen。通过后才允许另行冻结 Stage B；本
阶段不运行 student-updated chained CEM，不作 latency、native system timing、memory 或
closed-loop success claim。

## Frozen split 与设置

- train：`wall_v2_case_20..83`，共 64 个 observations；
  calibration：`wall_v2_case_84..91`，共 8 个；heldout：
  `wall_v2_case_92..99`，共 8 个。三组完全 disjoint，且排除 V1 cases。
- 每个 observation 都是 10 个 full-teacher-only chained CEM rounds；`K=300`、
  `H=5`、`topk=30`、`opt_steps=10`。
- full teacher 生成 `J_full` 并独自更新 `mu/sigma`。student score 和 action-only
  score 仅作 offline ranking，不能参与 CEM update。

## V2 architecture identity

Context-aware student 的 architecture identity 冻结为
`v2_context_aware_action_conditioned_cross_attention`：candidate-independent `z0`/
`zgoal` visual/proprio context tokens 作为 context，H=5 candidate action sequence
作为 action tokens，使用 action queries 对 context tokens 做 cross-attention，并输出
lower-is-better score。它不做 autoregressive rollout。

Action-only negative control 的 identity 是 `v2_action_only_negative_control`：只使用
candidate action sequence，不使用 observation/goal context，也不使用 cross-attention。
它的 calibration/heldout metrics 必须报告，但没有 superiority 或 pass/fail gate。

Verifier 只检查 training summary 中这些冻结 architecture fields；它不加载 checkpoint，
因此 architecture metadata 必须由运行器诚实记录，不能将普通 MLP 标成 cross-attention。

## M selection 与 heldout gate

候选集合严格为 `[60, 90, 120, 150, 180]`。一个 observation-round pair 是一个 call，
因此 calibration 和 heldout 各有 80 calls。Verifier 从原始 `J_full`、context-aware
scores 和 action-only scores 重新 stable rank，不信任记录中的 top-M 或 recall：

1. 在 calibration 上选择最小的 M，使 context-aware recall 的 median 恰为 `1.0`，且
   min recall `>=29/30`。
2. `selection.json` 必须写明 `selected_by=calibration_only`、V2 calibration IDs 和
   `heldout_used_for_selection=false`；不能按 heldout 调整 M，也不能为每个 call 单独选 M。
3. heldout 只在这个 calibration-selected fixed M 上判定：context-aware recall 的
   median 必须为 `1.0`，min 必须至少 `29/30`，且 selected M `<=180`。
4. action-only control 在 calibration/heldout 的 compact metrics 必须记录，但不影响 verdict。

## Artifact schema

冻结细节见 [`RANK_STUDENT_V2_STAGE_A_FREEZE.json`](./RANK_STUDENT_V2_STAGE_A_FREEZE.json)。

- `teacher_data.jsonl`：每行一个 train observation，十个 rounds；每 round 包含
  `candidate_actions`、`J_full`、CEM chain states，以及
  `cem_update_source=full_teacher`、`student_used_for_cem_update=false`。
- `student_training_summary.json`：只写 compact provenance、split counts、training
  status 和 architecture identity；不要把 candidate/action/score 大 arrays 放进 summary。
- `calibration_results.jsonl`、`heldout_results.jsonl`：每行一个 observation，十个
  rounds；raw arrays 只放在这些 JSONL records，供 verifier 重算。
- `selection.json`：只写每个 M 的 `call_count`、context/action-only 的 median/min/mean
  等 compact metrics，以及 calibration-only provenance；不要写逐 candidate arrays。
- `verify_rank_student_v2.py`：read-only verifier，输出 compact `verifier.json`。

Verifier 检查 V2 split identity、schema/completeness、train-only provenance、full-teacher-only
chain update、cross-attention/action-conditioned architecture identity、selection provenance、
no heldout tuning、action-only recording 和 heldout gate。不使用 hash 代替数值或 provenance。

## 本地检查

本地只做 Python syntax 和 JSON parse：

```powershell
python -m py_compile verify_rank_student_v2.py
python -c "import json; json.load(open('RANK_STUDENT_V2_STAGE_A_FREEZE.json', encoding='utf-8'))"
```

teacher-data 生成、V2 student training 和 offline screen 必须在批准的 PBS compute
allocation 执行；login node 只用于提交与查询作业。本目录不包含也不修改 runner/PBS。

## Evidence boundary

Stage A PASS 只表示：V2 train-only teacher data 和 architecture/training provenance 完整，
calibration 选出的一个 fixed M 在 heldout offline ranking 上通过冻结 gate。它不等于
student 与 full teacher 的 chained decision equivalence，也不提供 latency、memory、native
deployment 或 closed-loop evidence。
