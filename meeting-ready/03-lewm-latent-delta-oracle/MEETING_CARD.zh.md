# 组会卡片：LeWM PushT latent-delta oracle

状态：`COMPLETE`；corrective PBS job：`24844057.pbs101`；退出码：`0`。

## 30 秒版本

这条路线不直接压缩整个 LeWM latent，而是询问：在保留当前高维状态的前提下，状态转移的 delta 是否集中在一个可复用的低维 subspace 中？Step 1 测量 observed/model delta 的 reusable PCA；Step 2 把完整 LeWM 的 delta projection 接回 fixed-observation planner，分开看 candidate ranking 和 CEM action。

## 冻结实验与结果

| 阶段 | 结果 | 组会可说的最短结论 |
|---|---|---|
| Step 1：reusable structure | **FAIL（formal passing ranks 为空）** | model-PCA rank 64 held-out worst-case retained/MSE=`0.5728/0.4059`；忽略冻结上限时 model rank `192` 仅为 reconstruction diagnostic。 |
| Step 1：observed/cross controls | **正式 FAIL / 超界弱证据** | observed rank `96` 仅在忽略 rank 上限时满足 worst-case reconstruction；rank 64 的 observed↔model cross-basis 仍失败。 |
| Step 1：per-trajectory oracle | **仅下界诊断** | `5×192` hindsight SVD 在 rank 3/4 已低误差，但不可复用、无 speed evidence。 |
| Step 2A：candidate ranking | model rank `64/96` **PASS** | ranking gate 在 16 个 held-out blocks 上通过；random rank64 **FAIL**，full-rank control **PASS**。 |
| Step 2B：fixed-observation CEM | rank `64/96` **FAIL** | CEM first-action drift 超过 planner gate；rank-192 full control 为零漂移。 |

## 关键数字

- candidate gate：model-PCA 最小通过 rank `64`；per-trajectory oracle 最小通过 rank `3`，但它不是 reusable arm；
- CEM planner gate：normalized L2 `≤0.15` 且 absolute difference `≤0.25`；rank64 最大为 `0.9573 / 4.9527`，rank96 最大为 `0.7699 / 3.3842`；
- CEM 是同 observation/seed 的 official fixed-observation diagnostic，不是 closed-loop，也不是 speed benchmark。

## 组会上不要说

- 不要把 per-trajectory oracle rank 3/4 写成 global reusable subspace；
- 不要把 candidate ranking PASS 写成 CEM action preservation；
- 不要声称 latency reduction、cheap predictor、closed-loop PushT success 或 end-to-end improvement。

## 入口

- [Step 1 结果](reports/RESULT_STEP1_COMPRESSIBILITY.zh.md)
- [Step 2 结果](reports/RESULT_STEP2_ORACLE_RANK.zh.md)
- [冻结协议](PROTOCOL.zh.md)
- [冻结配置](LEWM_DELTA_ORACLE_FREEZE.json)
- [权威 summary](artifacts/24844057.pbs101/summary.json)
