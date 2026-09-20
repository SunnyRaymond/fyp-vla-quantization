# DINO-WM shared prefix cache：frozen Stage-A mechanism smoke

本轮只做 frozen Stage-A mechanism gate，不做 native/system latency gate，也不做 closed-loop evaluation。runner 只使用同目录已经存在的 `cache_core.py`，缺失或接口不完整时直接失败，不使用 reference fallback。

## 固定协议

`smoke.pbs` 申请单张 A100、16 CPU、110 GB、30 分钟，固定执行：

- 2 个 Wall validation observations：`wall_case_00`、`wall_case_01`；
- 每个 observation 2 个 fixed populations；
- 每个 population `K=300`、`H=5`、`topk=30`；
- candidate pool seed：`110000 + case_id * 100 + cem_iteration`；三条路径复用同一 candidate tensor；
- official objective：`alpha=1, mode=last`；stable ascending candidate-index tie-break；
- 三路径：官方 `baseline`、`iteration_cache`、`factorized_cache`；action-conditioned encoder/predictor/rollout suffix 不改。

每条 `mechanism.jsonl` 记录包含 verifier 要求的三路径字段：`cache_hit`、prefix/rollout 的 `equal_to_baseline`、`shape`、`dtype`、`device`、完整 300 个 objective、stable `full_order`、`topk30`、elite-derived `mu`/`sigma` 和 `first_action`。cached path 只有在所有比较 bitwise exact 时才报告 `canary.fallback_triggered=false`；出现 mismatch 会让 verifier 失败。

## Negative controls

每个 population 都实际执行两个 control，不是硬编码：

- `wrong_boundary`：把 baseline 的 candidate 0 action-conditioned rollout 结果广播给所有 candidates，`fallback_disabled=true`，比较 objective/order 是否与 baseline 不同；
- `stale_observation`：使用另一个 validation observation 的 transformed prefix，保持当前 goal/candidates 不变，`fallback_disabled=true`，实际重新 rollout 并比较 downstream mismatch。

旧的 randomized-mask/all-zero-mask control 不使用。

## 运行与产物

官方 reproduction 资产必须已在：

```text
/scratch/users/ntu/yguo017/dino-wm-wall/ASSETS_READY
/scratch/users/ntu/yguo017/dino-wm-wall/runtime-complete.tar
/scratch/users/ntu/yguo017/dino-wm-wall/data/
/scratch/users/ntu/yguo017/dino-wm-wall/checkpoints/outputs/wall_single/
```

另外 staging 以下已有实验文件到：

```text
/scratch/users/ntu/yguo017/dino-wm-wall/idea-validation/dino-wm-shared-prefix-cache/
```

至少包括 `run_smoke.py`、`cache_core.py`、`EXPERIMENT_FREEZE.json` 和 `verify_results.py`。PBS 会在模型加载和 runtime 解包前检查 `PBS_JOBID`、hostname、资产与这些文件；之后才在 compute node 解包既有 runtime。它不下载、不安装、不编译、不 hash，也不会自动 `qsub`。

手动提交：

```bash
qsub /scratch/users/ntu/yguo017/dino-wm-wall/idea-validation/dino-wm-shared-prefix-cache/smoke.pbs
```

结果位于 `$ROOT/artifacts/shared-prefix-smoke-$PBS_JOBID/`：

```text
mechanism.jsonl   # frozen Stage-A 输入
summary.json      # stage_a_mechanism_only；system_gate=NOT_RUN
verifier.json     # verify_results.py 输出
MECHANISM_PASS    # 仅 verifier exit 0 后创建
```

PBS 只有在 `verify_results.py --mechanism mechanism.jsonl` 返回 0 后才创建 `MECHANISM_PASS`。本轮不生成 system summary，不触发 system gate；closed-loop gate 保持 `NOT_AUTHORIZED`，必须等 mechanism/system 两个前置 gate 后另行运行。
