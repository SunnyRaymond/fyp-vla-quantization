# Wall GDPlanner shared-prefix portability

这一子实验只验证同一 DINO-WM Wall checkpoint 上，`encode_obs(obs_0)` 是否能从 GDPlanner 的 optimization loop 提到一次 planner-call 级别，同时保留 action encoder、predictor、`replace_actions_from_z`、objective 和 `backward` 的 action-gradient path。

## 冻结范围

- 官方 source commit：`0a9492fa12044b852ae9e001cc74604b79c8bb0c`。
- checkpoint：`wall_single/checkpoints/model_latest.pth`，reproduction 记录的 published epoch 为 65。
- observations：2 个，由官方 `PlanWorkspace.prepare_targets` 的 Wall `goal_source=random_state` 路径生成；两条 path 复用相同的 observation/goal。
- 官方 `source/conf/planner/mpc_gd.yaml` 设置：`horizon=5`、`action_noise=0.003`、`sample_type=randn`、`lr=1`、`opt_steps=1000`、`eval_every=10`。这里调用单个 direct `GDPlanner.plan`，不增加 MPC outer rounds，也不执行 environment evaluation。
- 每个 observation/path：5 次 warmup、10 次 technical repeat；每一对使用相同 seed，path order 按 observation/repeat 交错。

baseline 调用官方 `GDPlanner.plan`。两条 path 都在 planning-only 设置下关闭 model-parameter gradient accumulation（只优化 actions，`dL/daction` 仍 live）。`iteration_cache` path 在每个完整 `plan` 开始时分别编码 start/goal observation 一次，并通过已有 `cache_core.rollout_with_cache(..., mode="iteration_cache")` 重用 start prefix；start prefix detached，但 action-dependent suffix 仍对 actions 建图。两条 path 的完整 `plan` timing 都包含 preprocessing、prefix encoding、cache lookup、optimization 和 trace logging，并在计时前后同步 CUDA。

数值 gate 优先要求 final actions 和每一步 loss bitwise exact；若不是 exact，只允许使用预先冻结的 `max-abs <= 1e-5`，不能看结果后调阈值。Progression gate 另外要求两组 observation 的 paired median full-plan latency reduction `>=10%`，且最大 cached-to-baseline peak-memory ratio `<=1.10`；三项（decision、latency、memory）都 PASS 才授权后续 PushT，否则停在 Wall。

## 运行

先把本目录 staging 到已有 reproduction root：

```text
/scratch/users/ntu/yguo017/dino-wm-wall/idea-validation/dino-wm-shared-prefix-cache/gd_planner_portability/
```

同时确认已有官方资产/runtime：

```text
/scratch/users/ntu/yguo017/dino-wm-wall/ASSETS_READY
/scratch/users/ntu/yguo017/dino-wm-wall/runtime-complete.tar
/scratch/users/ntu/yguo017/dino-wm-wall/data/
/scratch/users/ntu/yguo017/dino-wm-wall/checkpoints/outputs/wall_single/
```

只在 compute allocation 内手动提交：

```bash
qsub /scratch/users/ntu/yguo017/dino-wm-wall/idea-validation/dino-wm-shared-prefix-cache/gd_planner_portability/gd_portability.pbs
```

PBS 在 runtime 解包和模型加载前检查 `PBS_JOBID`、hostname、资产、runner、freeze 和 verifier；不下载、不安装、不编译、不 hash，也不会自动提交作业。本配置是单张 A100、16 CPU、110 GB、4 小时 walltime（runner 内部 230 分钟 timeout）：2 observations × 2 paths × (5 warmup + 10 technical) 个完整 1000-step direct GD calls，未增加 MPC outer rounds。 本地只允许做 `py_compile`、JSON 解析和 shell syntax 检查。

## 产物

- `observation_manifest.json`：两组 frozen input 的 shape、来源和 eval seed。
- `plan_targets.pkl`：官方 `PlanWorkspace` 生成的两组实际 `obs_0`/`obs_g`（供同一作业内 paired calls 复用）。
- `system_summary.json`：60 个完整 planner-call records（2 observations × 15 repeats × 2 paths），包含 timing、peak memory、final actions 和 per-step losses。
- `verifier.json`：read-only numerical/timing integrity report。
- `GD_PORTABILITY_COMPLETE`：runner 正常完成。
- `GD_PORTABILITY_NUMERICAL_PASS`：仅 verifier 通过 exact 或 frozen threshold gate 后创建。

结果只能支持 planner-call observation-prefix reuse 在这个 Wall/GDPlanner/checkpoint 上的 portability evidence；不能直接外推到 LeWorldModel、其他 task 或 closed-loop success。
