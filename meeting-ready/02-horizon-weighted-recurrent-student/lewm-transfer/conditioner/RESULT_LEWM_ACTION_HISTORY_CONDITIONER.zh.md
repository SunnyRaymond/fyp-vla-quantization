# LeWM PushT：Action-History AdaLN Conditioner 增量实验结果

## 结论

最终正式作业 `24884287.pbs101` 成功完成 predictor-level Stage A
（`Exit_status=0`）。conditioner 相比原 h256 recurrent baseline 使 Spearman 的
step-1500 median 从 `0.415374` 升到 `0.453163`，但 top-30 median 从 `0.266667`
降到 `0.250000`，minimum top-30 仍为 `0`；冻结 predictor feasibility gate 仍为
**NO-GO**。因此本次没有观察到足以称为明显/可靠提升的 planner-facing ranking 改善。

`24884154.pbs101` 是首次接线错误：protocol title 未包含原 runner 的冻结标题检查，
在 compute node 启动后 4 秒退出（`Exit_status=1`），没有加载模型、读取 HDF5、训练或
产生指标。`24884158.pbs101` 虽成功完成，但 post-run fairness review 发现它把
baseline 的 affine LayerNorm 拆成了 no-affine LayerNorm，标记为
`pilot_invalid_for_primary_comparison`，不用于以下结论。修复后最终正式作业为
`24884222.pbs101`（`Exit_status=0`）的 metrics 与正式结果一致，但 30 秒 telemetry
sampler 只有启动样本，因此标记为 `metrics_valid_telemetry_insufficient`；将
`24884287.pbs101` 作为最终正式结果。

## Conditioner 与官方 teacher 的对应关系

已直接检查 frozen LeWM repository 的 `module.py`：`ConditionalBlock` 以 action
embedding `c` 经 `SiLU -> Linear(192, 6*192)` 产生两组 shift/scale 与 residual
gates；最后一层 weight/bias 全 zero-init，调制形式为 `norm(x) * (1 + scale) + shift`。

本实验只迁移其中的轻量、无 attention 部分：latest-3 causal raw packed action
history `[3,10]` 经 `Linear(30,128) -> SiLU -> Linear(128,512)`，split 成
feature-wise `[shift, scale]`，调制 baseline 原有 affine LayerNorm 的输出；baseline
的 `LayerNorm(256) -> Linear(256,1024) -> GELU -> Linear(1024,256)` 模块、参数
初始化和 shared residual 路径保持不变。最后 linear zero-init，因此初始为 identity。
五个 horizon 共享同一 conditioner 与 recurrent cell，hidden size 仍为 `256`，无
attention、goal、encoder 或 teacher parameters。

## 公平性与边界

- baseline：原始 `LeWMCompactRecurrentTransitionStudent`，作业 `24564619.pbs101`；
- treatment：`action_history_adaln`，作业 `24884287.pbs101`；
- 两者均为 LeWM official PushT、同一 256/8 episode-disjoint context contract、同一
  initialization/training seeds、1500 updates、snapshots `500/1000/1500`、16 held-out
  blocks × 300 candidates、top-k 30 和同一 predictor timing protocol；
- treatment job 直接复用 baseline compute-side `prepared_rows.pt`，所以 teacher
  target/slate/candidate bank 没有重新抽样；
- latency boundary 是 cached H=1 latent + 5 normalized packed action tokens → 5-step
  predictor，batch 300、warmup 3、technical repeats 10、CUDA synchronize，不含
  encoder、goal encoding、CEM 或 environment。

## Step 1500 主结果

| 指标 | baseline | conditioner | 变化 |
|---|---:|---:|---:|
| 参数量 | 775,872 | 845,888 | +70,016（+9.02%） |
| Spearman median | 0.415374 | 0.453163 | +0.037789 |
| Spearman minimum | -0.404356 | -0.285363 | +0.118993 |
| top-30 median | 0.266667 | 0.250000 | -0.016667 |
| top-30 minimum | 0.000000 | 0.000000 | 0 |
| relative latent MSE median | 0.012116 | 0.011190 | -0.000926 |
| positive Spearman blocks | 12/16 | 12/16 | 持平 |
| positive top-30 blocks | 15/16 | 14/16 | -1 |
| predictor latency | 1.85293 ms | 2.45914 ms | +32.72% |

paired block-level delta（conditioner − baseline；相同 `pairing_key`）如下：

| 指标 | improve / worse / tie | median delta |
|---|---:|---:|
| Spearman（越高越好） | 13 / 3 / 0 | +0.104890 |
| top-30 overlap（越高越好） | 5 / 5 / 6 | 0 |
| relative latent MSE（越低越好） | 14 / 2 / 0 | -0.000508 |

paired Spearman 的正向 block 较多，但 top-30 没有整体改善，且最差 top-30 仍为 0；
因此不能把 latent MSE 或 Spearman 的局部收益解释成 predictor replacement。

## Snapshots、cosine 与 latency

| snapshot | Spearman median / min | top-30 median / min | relative MSE median |
|---|---:|---:|---:|
| step 500 | 0.119508 / -0.263785 | 0.116667 / 0.000000 | 0.017949 |
| step 1000 | 0.411727 / -0.195738 | 0.166667 / 0.000000 | 0.011731 |
| step 1500 | 0.453163 / -0.285363 | 0.250000 / 0.000000 | 0.011190 |

step-1500 held-out per-horizon cosine median 为：
`[0.998008, 0.996160, 0.994366, 0.993621, 0.990638]`。

Latency protocol 下，baseline job 记录 teacher/student 为 `23.52947/1.85293 ms`
（reduction `92.1251%`）；正式 conditioner job 记录 teacher/student 为
`18.84467/2.45914 ms`（reduction `86.9505%`）。跨作业 teacher absolute timing 有
GPU/runtime variation，因此只把每个 job 内的 reduction 作为 latency gate 证据；
conditioner 仍显著快于 teacher，但相对 baseline student 增加约 `32.72%` predictor
latency。最终作业 walltime 约 25 秒，PBS 使用 5 秒采样周期；job log 保留 5 条
A100 telemetry，其中训练期间有 3 条 `33% / 633 MiB` 样本。

训练 `last10 / first = 0.002756`，causality prefix 1–4 的 maximum absolute
difference 全为 `0`，integrity/convergence 与 latency gate PASS；ranking/fidelity
gate FAIL，综合 `predictor_feasibility = NO-GO`。

## Artifacts 与 claim boundary

- [conditioner summary](artifacts/24884287.pbs101/lewm_recurrent_student_summary.json)
- [conditioner job log](artifacts/24884287.pbs101/job.log)
- [conditioner job status](artifacts/24884287.pbs101/job_status)
- [conditioner context manifest](artifacts/24884287.pbs101/context_manifest.json)
- [metrics-valid telemetry-insufficient run](artifacts/24884222.pbs101/lewm_recurrent_student_summary.json)
- [invalid pilot summary](artifacts/24884158.pbs101/lewm_recurrent_student_summary.json)
- [baseline summary](../artifacts/24564619.pbs101/lewm_recurrent_student_summary.json)

正式 summary 记录了 `base_transition_unchanged=true` 与
`base_path_zero_init_equivalence=true`；本地 smoke test 在复制 baseline-owned
parameters 后得到 zero-init forward `max_abs=0.0`。

本轮只支持：在固定 LeWM PushT predictor-level cell 中，zero-init action-history
AdaLN-style affine conditioner 对平均 latent fidelity 有小幅帮助，并对部分 Spearman
block 有正向作用；它没有通过 frozen ranking gate，且带来约 32.72% student predictor
latency overhead。official CEM、planner viability、closed-loop PushT、encode_obs
speedup 与跨模型泛化均为 `NOT_RUN_BY_SCOPE`。
