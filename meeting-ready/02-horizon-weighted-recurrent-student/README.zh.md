# Horizon-weighted Recurrent Student：组会与后续实验包

这是一个可以独立拿去组会、并作为后续 predictor-level 实验起点的最小 bundle。它把 `shared recurrent latent-transition student` 和其唯一的 loss treatment——`horizon weighting`——放在同一个可运行闭包中；原始文件仍保留在 `experiment/idea-validation/jepa-action-prefix-compiler/`，本目录是复制出来的工作副本。

## 先看什么

1. [MEETING_CARD.zh.md](MEETING_CARD.zh.md)：组会前的 1 页版本。
2. [RESULT_HORIZON_WEIGHTED.zh.md](reports/RESULT_HORIZON_WEIGHTED.zh.md)：本 idea 的正式结果。
3. [RESULT_RECURRENT_STUDENT.zh.md](reports/RESULT_RECURRENT_STUDENT.zh.md)：为什么 recurrent student 是当前基座。
4. [CONTINUE_EXPERIMENTS.zh.md](CONTINUE_EXPERIMENTS.zh.md)：以后只从本目录继续实验的路径、外部资产和 PBS 用法。

## 一句话结论

在冻结的 DINO-WM PushT action-conditioned predictor 上，shared recurrent student 将 predictor-only latency 降到约 `10.7 ms`，相对 frozen teacher reduction 约 `99.69%`，参数量为 direct student control 的约 `30.35%`。在这个 recurrent 基座上给较远 horizon 更高 loss 权重，使 Spearman 在 `16/16` 个 paired blocks 上同向提高，但 effect size 很小，因此仍不能宣称可以 full replacement。

## 结果边界

- Recurrent architecture：Spearman median `+0.037226`、top-30 median `+0.066667`，两者方向正确但未达到冻结 effect gate。
- Horizon weighting：Spearman median `+0.004083`，top-30 median `+0.033333`；`latent non-inferiority`、`causality`、收敛和 latency 通过，但 effect gate 失败。
- Horizon-weighted student 的 absolute ranking 为 Spearman median/minimum `0.970596/0.894798`，top-30 median/minimum `0.833333/0.600000`，未达到 replacement gate。
- 计时边界是 cached native observation latent + action prefix → predictor rollout；不包含 `encode_obs`、CEM、environment 或 closed-loop success。
- 本包不声称 LeWM/Fast-LeWM transfer、all-JEPA universality 或 closed-loop control improvement。

后续 spatial/token mixer 等 NO-GO 分支不属于本包，也没有复制；它们只作为后续证据边界，不改变这里的两个 idea。

## 目录结构

```text
02-horizon-weighted-recurrent-student/
├─ README.zh.md
├─ MEETING_CARD.zh.md
├─ CONTINUE_EXPERIMENTS.zh.md
├─ src/                 # runner 及其最小本地 import 闭包（含 cache_core.py）
├─ config/              # freeze、protocol 和共享实验契约
├─ jobs/                # 可从 bundle 目录提交的 PBS wrapper
├─ artifacts/           # manifest、baseline、summary、log、status、telemetry
└─ reports/             # 中文结果解释
```

## 当前 bundle 自带的证据

- `24466744.pbs101`：256-context Dhigh manifest；
- `24373175.pbs101`：旧 128-context manifest，用于 manifest pairing 检查；
- `24477396.pbs101`：context-density summary，作为 Dhigh 支持证据；
- `24494756.pbs101`：query-slate direct student control；
- `24503839.pbs101`：uniform recurrent student；
- `24510395.pbs101`：horizon-weighted recurrent treatment。

GPU telemetry 和 job status 也保留在上述对应目录中；没有复制 checkpoint、`plan_targets.pkl`、数据集或完整 source repository。
