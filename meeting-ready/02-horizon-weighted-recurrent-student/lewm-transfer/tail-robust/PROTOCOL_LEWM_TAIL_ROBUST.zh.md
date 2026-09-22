# LeWM PushT：Phase 3 tail-robust EMA 冻结协议

状态：`frozen / predictor-level only / pending one formal GPU job`

权威 Phase 2 输入固定为 `state-coverage/LEWM_STATE_COVERAGE_FREEZE.json`、formal
job `24926383.pbs101` 的 `context_manifest_512.json` 与
`prepared_512/prepared_rows.pt`；本轮不重新生成 manifest、prepared rows 或 historical
`512×3000` reference。

## 1. 目的与历史 reference

Phase 2 的 `512×3000` terminal（formal job `24926383.pbs101`）是只读 historical
reference。本轮不重训该 arm，也不使用 held-out 结果选择 arm、snapshot 或超参数。
本轮只新增两个 512-context、3000-update arm，用于检验固定 EMA 与 context-level
tail emphasis 是否能改善最差 held-out block。

| arm | online score objective | terminal model | 作用 |
|---|---|---|---|
| `ema_score` | 标准 score-distill batch mean | `EMA(step=3000)` | 区分 EMA 的影响 |
| `ema_tail_score` | `0.75×mean + 0.25×mean(top-2 context losses)` | `EMA(step=3000)` | 预注册 tail-robust treatment |

两个 new arms 使用同一 Phase 2 `512` manifest、同一 `prepared_rows.pt`、同一
held-out 8 contexts × 2 fresh action-prefix seeds × 300 candidates、同一 student
architecture、initialization state、AdamW、batch size 8、context schedule seed、64
candidate train bank 和 score-distill 基本公式。candidate 是 block 内的 nested
measurement，不是独立 replicate。

## 2. Frozen training contract

student 仍为无 attention 的 `LeWMCompactRecurrentTransitionStudent`：192-D latent、
packed action 10-D、shared recurrent h256、predicted-latent free-running feedback。
latent loss 完全不变，为 batch 中所有 8 contexts 与 64 candidates 的原始
horizon-weighted `recurrent_loss` mean。

score loss 对每个 context 先独立完成 Phase 2 的 teacher-score normalization、
SmoothL1 和 top-12 weight，然后得到一个 context scalar `l_i`。A 使用
`mean_i(l_i)`；B 使用：

```text
score_B = 0.75 * mean_i(l_i) + 0.25 * mean(top2_i(l_i))
total_B = latent_loss + 0.1 * score_B
```

`top2` 是 batch 8 中最大的两个 `l_i`（largest-loss / worst-context direction），
绝不在 64 candidates 内把 candidate 当作 replicate。两个 arm 均从同一 step-0
state 开始；每步均使用独立但同 seed 的 CPU context permutation，因此 schedule
逐步一致。

EMA 从 step 1 开始维护：先以共享 step-0 state 初始化，执行 online optimizer step
后更新 `ema = 0.999*ema + 0.001*online`。训练仍记录 online loss；snapshot 和所有
held-out 评估只使用对应 EMA state，terminal 固定为 `ema_step_3000`。

## 3. Frozen evaluation gates

统计 paired unit 是相同 held-out context × 相同 fresh action-prefix seed，共 16
blocks。评估报告 EMA snapshot 500/1000/1500/3000，primary 是 EMA step 3000。

每个 arm 必须独立报告 inherited absolute predictor gate：

- Spearman median `>=0.95`，minimum `>=0.80`；
- top-30 median `>=0.75`，minimum `>=0.50`；
- relative latent MSE median `<=0.25`；
- positive Spearman blocks `>=12`，positive top-30 blocks `>=12`；
- predictor-only latency reduction `>=20%`；
- finite/integrity、weighted-loss convergence、causality 全部 PASS。

其中 weighted-loss convergence 沿用 base freeze 的 `last10 / first <= 0.8`，causality
沿用 `1e-6`，且 EMA isolation、tail aggregation 与 same-init/schedule scope checks
必须 PASS。

`ema_tail_score` 另外必须不回退于 historical `512×3000` reference：Spearman
median `>=0.978317`、top-30 median `>=0.866667`、relative latent MSE median
`<=0.0175`。该 no-regression gate 只比较固定 reference，不按 held-out 结果挑选
arm 或 snapshot。A 用于 paired EMA-effect diagnosis，不替代 B 的 primary gate。

## 4. Scope 与 execution boundary

只运行 predictor-level training/evaluation、causality 和 predictor-only latency；不
运行 official CEM、planner viability、closed-loop PushT、environment interaction，
也不把 predictor latency 解读成 encoder、CEM 或 end-to-end control speedup。

正式作业是单个 bounded GPU PBS job。作业开始前检查非空 `PBS_JOBID` 与非 login
hostname；训练期间每约 5 秒将 GPU utilization 与显存写入 `job.log`。Phase 2 的
manifest、prepared rows 和 held-out bank 只在 compute node 读取，完成后只保留小型
summary、job log、job status 与 result report，不拉取 checkpoint 或 prepared rows。
