# LeWM state-action prefix GRU confirmatory predictor result

## 结论先行

本轮是 temporal-balanced training 之后的单一、两臂、predictor-only confirmatory
experiment。`balanced_base` 复现 balanced temporal training 的原始
`LeWMCompactRecurrentTransitionStudent`；`state_action_prefix_gru` 只新增一个
64-D shared state-action prefix GRU，其他 training/evaluation 条件保持冻结。

结果为：GRU treatment 相对 balanced base 的 paired mechanism gate **GO**，但
treatment inherited absolute predictor gate、三个 temporal stratum gate 和 combined
primary gate 均 **NO-GO**。GRU 的 terminal overall Spearman/top-30 median 为
`0.841137/0.566667`，虽相对 base 的 `0.824964/0.533333` 有改善，仍明显低于
replacement thresholds。GRU predictor 相对 base overhead 为 `41.56%`，也超过冻结的
`35%` latency guard；teacher-relative reduction 仍为 `88.42%`，但不能替代 primary
gate。

因此本轮保留为一个有信息量但未通过的 predictor-level structure result：轻量
state-action prefix 对 paired ranking 有正向机制信号，但尚未达到 absolute fidelity
或 latency guard，不支持把 GRU student 升格为 teacher replacement。

## 作业与证据

唯一正式 job 为 `25158955.pbs101`：

| 项目 | 值 |
|---|---|
| `qstat -xf` `Exit_status` | `0` |
| walltime | `00:04:58` |
| `exec_host` | `x1000c1s3b0n1/0*16` |
| allocation | 1 GPU、16 CPU、110 GB、`gdev` |
| training | 两臂各 512 balanced contexts、3000 updates |
| fresh evaluation | 8 episodes × 3 anchors × 2 seeds = 48 blocks；300 candidates/block |
| terminal snapshot | `step_3000` |

没有 invalid pilot，也没有第二个正式 job。只取回 small summary、job log 和
`job_status`；没有取回 checkpoint、prepared rows 或 HDF5。

- [freeze](LEWM_STATE_ACTION_PREFIX_GRU_FREEZE.json)
- [protocol](PROTOCOL_LEWM_STATE_ACTION_PREFIX_GRU.zh.md)
- [runner](run_lewm_state_action_prefix_gru.py)
- [PBS wrapper](run_lewm_state_action_prefix_gru.pbs)
- [summary JSON](artifacts/25158955.pbs101/lewm_state_action_prefix_gru_summary.json)
- [job.log 与 5-second GPU telemetry](artifacts/25158955.pbs101/job.log)
- [job_status](artifacts/25158955.pbs101/job_status)

## Frozen design 与 preflight

两臂使用同一个 Phase 5 balanced training bank：512 contexts，anchor counts
`early/middle/late=171/171/170`，相同 `initialization_seed=20300901`、training
seed `20300902`、context schedule seed `20300904`、candidate slate seed `20300905`、
AdamW、batch 8、64 train candidates、3000 updates 和 score-distill loss。

Treatment 保留所有 h256 base modules：`history_adapter`、`latent_projection`、
`action_projection`、`shared_transition`、`output_norm`、`latent_update`。新增模块为：

- `prefix_init: Linear(192,64)`，初值为 `tanh(prefix_init(initial current latent))`；
- `prefix_state_projection: Linear(192,64)`；
- 一个 shared `GRUCell(10+64,64)`，每步读取当前 packed action 与当前 predicted state；
- `prefix_projection: Linear(64,256)`，加入原始 transition input。

所有 horizon steps 共用同一组 prefix modules；没有 attention、goal/teacher/encoder
input、per-horizon copies、hidden-size expansion、EMA 或 tail weighting。标准 PyTorch
初始化使用固定 `prefix_initialization_seed=20300906`，没有 zero-init 整个 projection。

local preflight **PASS**：

- base-owned parameters 与 control bitwise equal；
- synthetic unchanged-prefix causality max absolute difference `0`；
- selection exclusion、anchor bounds、same context schedule/slates 和 nested episode
  aggregation 均通过；
- base parameter count `775,872`，GRU treatment `844,096`，新增 `68,224` 参数。

Fresh test 固定为 `valid[536:544]`，排除旧 heldout、Phase 5 train、`valid[520:528]`
和 `valid[528:536]`，action-prefix seeds 为 `20300937/20300938`。test 未用于设计、
参数或 snapshot 选择。

## Terminal metrics

| arm | Spearman median/min | top-30 median/min | relative latent MSE median | positive S/top-30 | absolute gate |
|---|---:|---:|---:|---:|---|
| `balanced_base` | `0.824964 / -0.624814` | `0.533333 / 0.000000` | `0.100185` | `42/48`, `46/48` | **NO-GO** |
| `state_action_prefix_gru` | `0.841137 / -0.639435` | `0.566667 / 0.000000` | `0.094277` | `43/48`, `46/48` | **NO-GO** |

Treatment inherited gate 要求 overall Spearman median/minimum `0.95/0.80`、top-30
`0.75/0.50`、relative MSE `<=0.25`、positive blocks 至少 `36/48`，并通过
integrity/convergence/causality/latency。GRU 只通过 relative MSE、positive-block、
convergence、causality 和 teacher-relative latency reduction；ranking median/minimum
与 minimum top-30 未通过，因此 absolute gate 为 **NO-GO**。

### Descriptive snapshots

| arm / snapshot | Spearman median/min | top-30 median/min | relative MSE median | positive S/top-30 |
|---|---:|---:|---:|---:|
| base `500` | `0.496015 / -0.642350` | `0.300000 / 0` | `0.124006` | `38/48`, `44/48` |
| base `1000` | `0.631505 / -0.650503` | `0.366667 / 0` | `0.103246` | `38/48`, `43/48` |
| base `1500` | `0.706130 / -0.758520` | `0.450000 / 0` | `0.108635` | `42/48`, `45/48` |
| base `3000` | `0.824964 / -0.624814` | `0.533333 / 0` | `0.100185` | `42/48`, `46/48` |
| GRU `500` | `0.425417 / -0.641771` | `0.266667 / 0` | `0.121467` | `39/48`, `44/48` |
| GRU `1000` | `0.501021 / -0.606754` | `0.266667 / 0` | `0.102338` | `34/48`, `41/48` |
| GRU `1500` | `0.726759 / -0.750536` | `0.450000 / 0` | `0.112434` | `41/48`, `44/48` |
| GRU `3000` | `0.841137 / -0.639435` | `0.566667 / 0` | `0.094277` | `43/48`, `46/48` |

Snapshots 是 descriptive；没有 early stop，也没有以 test 选择 snapshot。

## Temporal stratum gate

Treatment 每个 stratum 要求 median Spearman/top-30 至少 `0.95/0.75`，positive
blocks 各至少 `12/16`。三个 strata 的 positive counts 均满足或超过 count gate，
但 fidelity medians 均不足：

| stratum | Spearman median/min | top-30 median/min | relative MSE median | positive S/top-30 | gate |
|---|---:|---:|---:|---:|---|
| early | `0.921711 / -0.639435` | `0.650000 / 0` | `0.042693` | `12/16`, `14/16` | **NO-GO** |
| middle | `0.805193 / -0.061133` | `0.500000 / 0.033333` | `0.089174` | `15/16`, `16/16` | **NO-GO** |
| late | `0.847419 / 0.334802` | `0.633333 / 0.200000` | `0.154460` | `16/16`, `16/16` | **NO-GO** |

GRU 在 late stratum 的 worst-case Spearman 为正，但仍不足以满足预注册 median
threshold；不能据此改写 primary decision。

## Secondary paired mechanism comparison

paired unit 是 episode；每个 episode 先对 6 个 nested blocks 取 median，再计算
`state_action_prefix_gru - balanced_base`：

| metric | median delta | gate |
|---|---:|---|
| Spearman | `+0.023855` | pass (`>=0`) |
| top-30 overlap | `+0.025000` | pass (`>=0`) |
| joint improvement/non-worsening | `5/8` episodes | pass (`>=5/8`) |

Secondary mechanism gate 为 **GO**，但不能替代 absolute/stratum primary gate。8 个
episode 的 paired deltas 为：

| episode | ΔSpearman | Δtop-30 | joint |
|---:|---:|---:|---|
| 2337 | `+0.058386` | `+0.066667` | yes |
| 3075 | `+0.080786` | `+0.050000` | yes |
| 6333 | `+0.047147` | `+0.100000` | yes |
| 12385 | `+0.004445` | `0.000000` | yes |
| 13790 | `-0.022949` | `-0.083333` | no |
| 17209 | `-0.020165` | `-0.066667` | no |
| 17396 | `+0.035050` | `+0.116667` | yes |
| 18413 | `+0.012660` | `-0.066667` | no |

## Latencies, causality and convergence

| arm | student median | teacher median | teacher→student reduction | last10/first | causality |
|---|---:|---:|---:|---:|---|
| balanced base | `1.565696 ms` | `19.292672 ms` | `91.8845%` | `0.008771` | PASS |
| state-action prefix GRU | `2.216448 ms` | `19.135489 ms` | `88.4171%` | `0.008518` | PASS |

两臂 convergence 均通过 `last10/first <=0.8`，terminal metrics finite。GRU 相对 base
predictor overhead 为 `2.216448/1.565696 - 1 = 41.5631%`，超过 frozen guard
`<=35%`；两臂 teacher-relative reduction 均超过 `20%`。因此 secondary latency guard
为 **NO-GO**，也不改变 primary gate。

Causality 对 unchanged prefix lengths `1/2/3/4` 的最大差异均为 `0`，threshold
为 `1e-6`。Latency boundary 只覆盖 cached H=1 latent + normalized packed action
prefix → five-step predictor，不包括 encoder、CEM、environment 或 closed-loop。

## GPU telemetry 与 scope

job log 中记录了 `60` 个约 5 秒的 `nvidia-smi` samples。PBS `qstat -xf` 记录 GPU
health 无错误、max VRAM 约 `684 MB`、SM utilization 峰值 `42%`、平均约 `28%`；无
Xid、ECC 或其他 GPU health error。telemetry 仅确认 workload 在 compute allocation
内运行，不替代 predictor metrics。

本轮明确：

- official CEM：**NOT_RUN_BY_SCOPE**；
- planner viability：**NOT_RUN_BY_SCOPE**；
- closed-loop：**NOT_RUN_BY_SCOPE**。

所以不能从 GRU 的 paired delta 或 predictor latency 声称 CEM、planner、closed-loop
success、encoder speedup 或 native deployment benefit。

## 解释与下一步边界

本轮支持一个有限的 mechanism-level 结论：在相同 balanced anchors、candidate
slates、initial base state、schedule、loss 和 compute budget 下，64-D shared
state-action prefix GRU 的 episode-paired ranking 方向为正，并达到 secondary
mechanism `5/8` gate。与此同时，absolute ranking 仍低于 replacement threshold，
所有 temporal strata 都未通过，且 predictor latency overhead 超过 `35%` guard。

因此不应继续在相同结构上做无冻结的 width、seed 或 latency sweep，也不应运行
official CEM/planner/closed-loop 来补救本轮 predictor-level NO-GO。当前 evidence
更适合记录为“轻量 state-action memory 提供小幅 paired ranking signal，但未达到
replacement fidelity/latency requirements”；后续若继续，必须另行冻结并提出明确的
root-cause hypothesis。

方法与 reporting provenance 参考 Scientific Agent Skills：Kassis, T., Agarwal,
V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A
Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065.
https://arxiv.org/abs/2609.00065（引用的是无版本后缀的官方 record；该方法 reference
不是本实验结果证据）。
