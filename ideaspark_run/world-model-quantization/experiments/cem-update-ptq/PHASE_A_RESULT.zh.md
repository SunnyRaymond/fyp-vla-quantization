# CEM-Update PTQ 阶段 A：旧 development 数据离线分析

状态：`complete`。本轮为 `CPU-only offline`；没有加载 model、使用 GPU、连接 SSH 或拟合权重。

## 数据完整性

读取 1 个 development `workload.pkl`、26/26 个 joint-fidelity score NPZ、旧 `joint_verification`、pool/episode 汇总。候选池为 26 个、8 个 episode、每池 `(300, 5, 10)`。
`reference_scores` 与 NPZ FP32 一致：26/26；stable top-30 与 NPZ 保存的 elite indices 一致：208/208。
旧 E2/elite-mean/score-NMSE pool rows 复核：208/208；episode 与 overall 最大绝对差分别为 `2.27e-13`、`2.84e-14`。

## 固定语义

`stable top30` 使用实际 adapter 的 stable argsort；`std` 使用 `ddof=1`（PyTorch `torch.std(dim=0)` 默认 correction=1）。`L_update = L_mu + w_iter × L_sigma`，CEM iteration 1/5 的固定权重分别为 1/0；`s=1` 解释为 action-normalized coordinate 的固定单位尺度。完整源码行证据记录在 `phaseA_results.json`。

## Overall（episode 等权）

| 方法 | L_mu ↓ | L_sigma ↓ | L_update ↓ | E2 ↓ | Elite overlap ↑ | Elite-mean MSE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 0.00000000 | 0.00000000 | 0.00000000 | 0.000000 | 1.000000 | 0.00000000 |
| all_W4 | 0.08124157 | 0.02483924 | 0.09215981 | 0.423338 | 0.255208 | 0.08124157 |
| all_W8 | 0.00171888 | 0.00094850 | 0.00193198 | 0.013121 | 0.970833 | 0.00171888 |
| RankCal | 0.04438538 | 0.01807750 | 0.05099219 | 0.253517 | 0.423958 | 0.04438538 |
| LocalMSE | 0.04384631 | 0.01871746 | 0.05145538 | 0.259170 | 0.437500 | 0.04384631 |
| ScoreError | 0.04996609 | 0.01863235 | 0.05788751 | 0.274638 | 0.396875 | 0.04996609 |
| Random1 | 0.04727840 | 0.02052792 | 0.05534153 | 0.278008 | 0.388542 | 0.04727840 |
| Random2 | 0.06596194 | 0.02294056 | 0.07577982 | 0.374698 | 0.313542 | 0.06596194 |

## 每 episode 的 L_update（完整 L_mu/L_sigma/L_update 在 JSON）

| episode | pools/points | FP32 | all-W4 | all-W8 | RankCal | LocalMSE | ScoreError | Random1 | Random2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| development:000 | 2/1 | 0.00000000 | 0.07918843 | 0.00181504 | 0.03688429 | 0.03206476 | 0.06060163 | 0.04166322 | 0.04639824 |
| development:001 | 4/2 | 0.00000000 | 0.10619849 | 0.00370421 | 0.06257202 | 0.06223529 | 0.05842477 | 0.06037155 | 0.08415467 |
| development:002 | 4/2 | 0.00000000 | 0.08455322 | 0.00196353 | 0.06254500 | 0.05584015 | 0.06693896 | 0.05574978 | 0.06824460 |
| development:003 | 4/2 | 0.00000000 | 0.09257553 | 0.00234846 | 0.03379040 | 0.03704147 | 0.04317580 | 0.04077831 | 0.08129626 |
| development:004 | 4/2 | 0.00000000 | 0.10235925 | 0.00098909 | 0.05899990 | 0.06239050 | 0.07210827 | 0.07861731 | 0.08385464 |
| development:005 | 2/1 | 0.00000000 | 0.07099476 | 0.00000000 | 0.04906286 | 0.04561548 | 0.05068134 | 0.04206605 | 0.07153345 |
| development:006 | 4/2 | 0.00000000 | 0.12196342 | 0.00354840 | 0.06812519 | 0.08454184 | 0.06606387 | 0.07305822 | 0.12062633 |
| development:007 | 2/1 | 0.00000000 | 0.07944541 | 0.00108711 | 0.03595787 | 0.03191351 | 0.04510542 | 0.05042778 | 0.05013034 |

## 描述性排序差异

mixed allocations 的 overall 排名如下（不含 FP32/all-W4/all-W8）：

- `L_update` ↓：RankCal < LocalMSE < Random1 < ScoreError < Random2
- `L_mu` ↓：LocalMSE < RankCal < Random1 < ScoreError < Random2
- `L_sigma` ↓：RankCal < ScoreError < LocalMSE < Random1 < Random2
- `e2_pairwise_disagreement` ↓：RankCal < LocalMSE < ScoreError < Random1 < Random2
- `elite_mean_normalized_action_mse` ↓：LocalMSE < RankCal < Random1 < ScoreError < Random2
- `stable30_elite_overlap_fraction` ↑：LocalMSE > RankCal > ScoreError > Random1 > Random2

## 阶段 A 判断

三项 update fidelity loss 均 finite 且在 mixed allocations 间有数值差异；measurement gate = `True`。因此旧数据足以完成阶段 A，且可进入有界的 Stage-B CAL/DEV 机制筛选。
这只说明 CEM update 目标在现有固定 pools 上可测并提供不同排序信息；没有闭环新结果，也没有证明新 idea work。不得根据旧 success 反调 `L_mu/L_sigma` 权重；本轮只使用预先冻结的 iter 权重。

## 输入与限制

脚本：`D:\Downloads\Final Year Project\ideaspark_run\world-model-quantization\experiments\cem-update-ptq\phaseA_offline_analysis.py`。旧输出保持不变；新结果位于本目录的 `phaseA_results.json` 与 `PHASE_A_RESULT.zh.md`。
