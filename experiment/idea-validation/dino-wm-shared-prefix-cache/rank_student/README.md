# Rank Student Stage A

这是新的 student evaluator 的冻结 Stage A：先用 `full_teacher` 产生
`wall_case_00..11` 的 teacher data，再训练 context-aware student，在
`wall_case_12..15` 上选择一个固定 `M`，最后只用这个 M 在
`wall_case_16..19` 做 offline held-out screen。

本阶段不运行 student-updated chained CEM，不测 latency、native system timing、memory
或 closed-loop success；通过后才允许另行冻结 Stage B。

## 冻结设置

- train：`wall_case_00..11`；calibration：`wall_case_12..15`；heldout：
  `wall_case_16..19`。三个 split 必须互不重叠。
- 每个 observation 运行完整十轮 chained CEM；`K=300`、`H=5`、`topk=30`、
  `opt_steps=10`。
- 每个 round 的 `J_full` 是 lower-is-better 的 full teacher objective。
  `student_context_scores` 使用 observation/goal context 和 candidate action；
  `student_action_only_scores` 是不含 context 的 negative control。
- full teacher 独自更新 `mu/sigma`。student 和 action-only control 只产生离线
  ranking，不能参与任何 CEM update。

## M 的选择与 gate

候选集合冻结为 `[60, 90, 120, 150, 180]`。每个 observation-round pair 是一个
call，因此 calibration 和 heldout 各有 40 calls。verifier 从原始 arrays 重新 stable
rank，不信任记录中的 top-M 或 recall：

1. 在 calibration 上选择最小的 `M`，使 context-aware recall 的 median 为 `1.0`，且
   min recall `>=29/30`。
2. 选择必须只使用 calibration；`selection.json` 必须明确记录
   `selected_by=calibration_only`、calibration IDs 和 `heldout_used_for_selection=false`。
3. heldout 只在该固定 M 上判定：median recall 必须为 `1.0`，min recall 必须至少
   `29/30`，且 selected `M<=180`。
4. action-only control 的 calibration/heldout recall 必须记录，但不参与 pass/fail。

## Artifact schema

冻结细节见 [`RANK_STUDENT_STAGE_A_FREEZE.json`](./RANK_STUDENT_STAGE_A_FREEZE.json)。
预期文件如下：

- `teacher_data.jsonl`：每行一个 train observation，十个 rounds；每 round 至少有
  `candidate_actions`、`J_full`、`input_mu/input_sigma`、`output_mu/output_sigma`、
  `first_action`，并标明 `cem_update_source=full_teacher`、
  `student_used_for_cem_update=false`。
- `student_training_summary.json`：记录 teacher-data 来源、训练 split、context-aware
  student 状态、action-only control 状态，并明确 calibration/heldout 未参与训练。
- `calibration_results.jsonl` 和 `heldout_results.jsonl`：每行一个 observation，十个
  rounds；每 round 有同一 candidate population 的 `J_full`、context-aware scores、
  action-only scores 和 full-teacher chain state。
- `selection.json`：只记录 calibration-derived fixed-M 选择及其 provenance。
- `verify_rank_student.py`：read-only verifier；输出 `verifier.json`（如调用方重定向
  输出则仍必须保留同样字段）。

禁止通过 hash 代替 arrays、split identity 或 ranking provenance。verifier 不加载
checkpoint、不运行 inference、不访问 CUDA，也不写入输入 artifact。

## 本地检查

本地只做 Python syntax 和 JSON parse：

```powershell
python -m py_compile verify_rank_student.py
python -c "import json; json.load(open('RANK_STUDENT_STAGE_A_FREEZE.json', encoding='utf-8'))"
```

集群上的 teacher-data 生成、student training 和 offline screen 必须在批准的 PBS
compute allocation 执行；login node 只用于提交和查询作业。

## Evidence boundary

Stage A PASS 仅表示：train-only teacher data 和 student-training provenance 完整，
calibration 选出的固定 M 在 heldout offline ranking 上通过冻结的 recall gate。它不等于
student CEM 与 full teacher 的 chained decision equivalence，也不提供 latency、memory
或 closed-loop 证据。
