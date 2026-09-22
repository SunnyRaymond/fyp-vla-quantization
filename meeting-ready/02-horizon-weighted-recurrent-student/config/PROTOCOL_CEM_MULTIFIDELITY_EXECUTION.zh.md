# CEM multi-fidelity diagnosis：PBS execution protocol

## 目的与边界

这份文件是 `CEM_MULTIFIDELITY_FREEZE.json` 对应的 execution layer。它只负责在 GPU compute-node allocation 中调用现有的
`src/run_dino_pusht_cem_multifidelity.py`，不修改 runner、不重新生成 parent pilot，也不改变 frozen cases、CEM 参数或 checkpoint identity。

这是 fixed-observation、teacher-verified 的 CEM mechanism diagnosis：比较 `teacher_full`、`student_only` 和 student-prefilter + teacher re-score arms。它不产生新的 closed-loop success rate，不替代 parent closed-loop pilot，也不作 deployment latency/memory claim。

本次没有提交远端作业。下文的远端路径均为需要用户替换和确认的占位符，不假定它们已经存在。

## 执行入口与 frozen parent contract

- PBS wrapper：`jobs/dino_pusht_cem_multifidelity.pbs`
- runner：`src/run_dino_pusht_cem_multifidelity.py`
- freeze：`config/CEM_MULTIFIDELITY_FREEZE.json`
- parent closed-loop job：`24544733.pbs101`
- student checkpoint job：`24510395.pbs101`

Runner 要求的 parent artifacts 是实际的 `closed_loop_summary.json` 和二进制 `plan_targets.pkl`。`target_manifest.json` 不能替代 `plan_targets.pkl`；如果 parent output 使用 job-ID 目录或其他命名，必须通过 override 指向真实文件。

默认 wrapper 路径沿用旧 `dino_pusht_cem_trace.pbs` 的 parent/checkpoint/runtime 约定：

```text
<DINO_WM_ROOT>/artifacts/dino-pusht-student-closed-loop-24544733.pbs101/closed_loop_summary.json
<DINO_WM_ROOT>/artifacts/dino-pusht-student-closed-loop-24544733.pbs101/plan_targets.pkl
<DINO_WM_ROOT>/artifacts/jepa-action-prefix-dino-pusht-horizon-weighted-24510395.pbs101/horizon_weighted_recurrent_step1500.pt
<DINO_WM_ROOT>/checkpoints/outputs/pusht/checkpoints/model_latest.pth
<DINO_WM_ROOT>/checkpoints/outputs/pusht/hydra.yaml
<DINO_WM_ROOT>/data/pusht_noise/
<DINO_WM_ROOT>/artifacts/pusht-python-overlay-pusht-v5/
<DINO_WM_ROOT>/PUSHT_DEPS_READY_V5
<DINO_WM_ROOT>/runtime-complete.tar
```

这些路径只是默认约定。wrapper 会在 compute node 上逐项检查，缺失即失败，不会下载、安装或编译，也不会把其他 parent 文件猜作替代品。已存在的 `runtime-complete.tar` 会在 compute node allocation 内解压到该 job 的 `$WORK/runtime`，随后使用其中的 `venv/bin/python`。

## 远端变量

提交时至少显式提供：

| 变量 | 含义 |
|---|---|
| `DINO_APC_ROOT` | 已 staged 的本 bundle 根目录，包含 `src/` 与 `config/` |
| `DINO_WM_ROOT` | 已 staged 的 DINO-WM/PushT 资产根目录 |
| `DINO_RUNTIME_ARCHIVE` | 可选；已 staged 的 runtime archive，默认是 `$DINO_WM_ROOT/runtime-complete.tar` |

可选 override：

`DINO_APC_SRC_ROOT`, `DINO_APC_CONFIG_ROOT`, `DINO_PUSHT_CEM_MULTIFIDELITY_RUNNER`,
`DINO_PUSHT_CEM_MULTIFIDELITY_FREEZE`, `DINO_STUDENT_TRACE_PARENT_ROOT`,
`DINO_STUDENT_TRACE_PARENT_SUMMARY`, `DINO_STUDENT_TRACE_PLAN_TARGETS`,
`DINO_STUDENT_CHECKPOINT`, `DINO_PUSHT_CHECKPOINT`, `DINO_PUSHT_CHECKPOINT_CONFIG`,
`DINO_PUSHT_DATA_ROOT`, `DINO_PUSHT_DEPS_ROOT`, `DINO_PUSHT_DEPS_MARKER`,
`DINO_PUSHT_CEM_MULTIFIDELITY_OUT_ROOT`, `DINO_PUSHT_CEM_MULTIFIDELITY_TIMEOUT_SECONDS`,
`DINO_PUSHT_CEM_MULTIFIDELITY_PREFILTER_M`。

`DINO_PUSHT_CEM_MULTIFIDELITY_PREFILTER_M` 可选，用空格分隔 teacher-prefilter sizes；默认是 frozen 的 `60 120`，必须满足 `30 < M <= 300`。`M=300` 是 teacher reference，不是本 intervention 的 sparse arm。

`PBS_JOBID` 由 scheduler 注入，不应手工设置。wrapper 还要求 `PBS_NODEFILE` 存在，并在 runner 前拒绝 hostname 中包含 `login`、`head` 或 `submit` 的节点。

## 远端提交命令

以下命令中的 `<...>` 必须先替换成已确认存在的远端绝对路径：

```bash
cd <remote-bundle>/meeting-ready/02-horizon-weighted-recurrent-student
qsub -v DINO_APC_ROOT=<remote-bundle>/meeting-ready/02-horizon-weighted-recurrent-student,DINO_WM_ROOT=<remote-dino-wm-root> jobs/dino_pusht_cem_multifidelity.pbs
```

如果 parent artifacts 不在默认目录，提交时增加 override，例如：

```bash
qsub -v DINO_APC_ROOT=<remote-bundle>/meeting-ready/02-horizon-weighted-recurrent-student,DINO_WM_ROOT=<remote-dino-wm-root>,DINO_RUNTIME_ARCHIVE=<remote-runtime-complete.tar>,DINO_STUDENT_TRACE_PARENT_ROOT=<remote-parent-root>,DINO_STUDENT_CHECKPOINT=<remote-student-checkpoint>,DINO_PUSHT_CEM_MULTIFIDELITY_OUT_ROOT=<remote-output-root> jobs/dino_pusht_cem_multifidelity.pbs
```

`qsub`、`qstat` 和轻量状态查询可以在 login node 执行；bundle、checkpoint、parent artifacts、data 和 dependency overlay 的同步/准备不得在 login node 做下载、重 I/O、解压、安装或编译。它们必须按集群管理员允许的 compute-node workflow 预先 staged。

## 输出与验收

默认独立输出目录：

```text
<DINO_PUSHT_CEM_MULTIFIDELITY_OUT_ROOT 或 DINO_WM_ROOT/artifacts>/dino-pusht-student-cem-multifidelity-<PBS_JOBID>
```

关键文件：

- `job.log`：wrapper、runner 及每 30 秒 GPU utilization/VRAM 采样
- `gpu_usage.csv`、`gpu_info.csv`：GPU telemetry
- `execution_identity.txt`：解析后的 parent、checkpoint、runner 和 prefilter 路径
- `case_00.json` 等：各 frozen failed case 的 arm-level rounds
- `cem_multifidelity_summary.json`：非空 runner summary
- `runner_exit_status.txt`、`final_exit_status.txt`、`job_status.txt`
- `DINO_PUSHT_CEM_MULTIFIDELITY_COMPLETE` 或 `DINO_PUSHT_CEM_MULTIFIDELITY_FAILED`

`COMPLETE` 只表示该 predictor/planner mechanism diagnosis 的 runner 成功完成；不表示 closed-loop 通过。timeout、CUDA 不可用、parent/checkpoint 缺失或 runner 非零退出都应保留失败状态，不解释成模型结论。

## 本地静态检查

不提交远端作业时，在 bundle 根目录执行：

```bash
bash -n meeting-ready/02-horizon-weighted-recurrent-student/jobs/dino_pusht_cem_multifidelity.pbs
for f in \
  meeting-ready/02-horizon-weighted-recurrent-student/jobs/dino_pusht_cem_multifidelity.pbs \
  meeting-ready/02-horizon-weighted-recurrent-student/config/PROTOCOL_CEM_MULTIFIDELITY_EXECUTION.zh.md; do
  test -z "$(grep -n '[[:blank:]]$' "$f")"
done
```

这些检查只验证 shell 语法和 trailing whitespace，不加载模型、不访问远端、不运行 runner。
