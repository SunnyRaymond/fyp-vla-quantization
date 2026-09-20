# DINO-WM PushT：h256 Context-Support Density Contrast 冻结协议

## 一句话目标

本轮只回答一个问题：在固定 NativeDinoPrefixStudent hidden_dim=256、WIDE+QUERY action-conditioned predictor、1500 updates、batch、optimizer、seeds、teacher targets、held-out blocks、causality 和 latency boundary 下，把同一批 32 个 train episodes 的 context support 从每 episode 4 个增加到 8 个，是否 materially improve held-out planner-candidate ranking？

这是独立的 context-support density contrast。唯一 treatment 是保留旧 128 个 context rows，并从同一 32 个 episodes 追加 128 个与旧 start 不重合的 rows，总数 128 → 256。不改变 model capacity、action-query mixture、trajectory、validation 或 held-out split。

这里的 density 是 context-support density，不是 query density。WIDE+QUERY 只表示固定的 action-query mixture；本协议不允许把结果写成 query density、query coverage 或数据多样性结果。

## 进入条件与 baseline

上一轮 h256 capacity-only cell 的权威结果来自 job 24456882.pbs101：capacity_effect = FAIL；ΔSpearman median = -0.0125701；Δtop30 median = -0.0166667；正向 block 数为 Spearman 5/16、top-30 7/16；full replacement = NO-GO。这些结果只授权进入 density 轴，不是本轮 density result。

本轮 low-density baseline 只读引用该次 h256 summary：

- remote job: 24456882.pbs101
- expected local artifact: artifacts/24456882.pbs101/capacity_contrast_summary.json
- selector: hidden_dim=256, WIDE-QUERY, step1500, 128 train contexts

必须先解析远端 job 结果，再读取预期 local artifact。不得重跑 low-density h256、重跑 h128、加载或修改 baseline checkpoint，也不得把 h128 summary 当作本轮 primary baseline。high/low density 的 ranking 必须按相同 (seed, context_index) block key 对齐。

## 唯一 treatment：context-support density

| 项目 | low-density | high-density |
|---|---:|---:|
| Student hidden dim | 256 | 256 |
| train episodes | 同一 32 个 | 同一 32 个 |
| contexts / episode | 4 | 8 |
| 总 train contexts | 128 | 256 |
| 新 trajectory / episode | 0 | 0 |
| old rows | 只读 | 原样保留为前缀 |
| new rows | 0 | 128，追加且排除旧 start |
| action-query mixture | 50/25/25 | 50/25/25 |

high-density manifest 必须保证：旧 128 rows 的 episode id、ordinal、start index、特征/target row 和顺序完全保留；同一 32 个 episodes 各追加 4 rows；新增 rows 的 selection_seed 固定为 20260928 + episode_id；新 start 排除旧 128 rows 的全部 starts；不生成新 trajectory/episode；validation rows/order 和 held-out contexts/seeds/candidate bank/block order 完全不变。新 rows 不能通过复制旧 context 伪装增加 support。

因此研究变量只有 context-support density，不能解释为 query density、query coverage、trajectory diversity 或 full-dataset coverage。

## Frozen student 与 target contract

Student 是官方 DINO-WM PushT action-conditioned predictor NativeDinoPrefixStudent，hidden_dim=256 固定不变。visual dim=384、proprio dim=10、primitive action dim=2；action 输入 [B,5,10]，每 token packing 5×2 primitive actions；输出五个 future native observation latents；Transformer depth=2、nhead=4、FFN=4×hidden=1024、dropout=0、GELU、norm_first=true。不把 goal 送入 student，不调用 student encode_obs；teacher rollout frozen、target detached；loss 只有 dense native-latent MSE，禁止 rank/listwise KL/goal/self-consistency/future-frame loss。

## Frozen WIDE+QUERY training bank

每个 batch 固定 32 行：16 logged action prefixes (50%)、8 independent Gaussian planner-init prefixes (25%)、8 one-step CEM elite-resample prefixes (25%)。one-step CEM bank 固定 M=64、K=8、per-coordinate variance floor=0.05，使用同一 context 的 frozen teacher terminal objective；step 1 前由 CPU generator 一次预计算，student 结果不能影响 bank。不得改变 mixture、M、K、variance floor、objective、call order 或 seeds。

即使 train contexts 从 128 增至 256，action-query mixture 仍是 50/25/25。这是更多 fixed context support，不是 query density 实验。

## Training

| 项目 | 冻结值 |
|---|---|
| updates | 1500，不得 early stop |
| batch | 32 |
| optimizer | AdamW |
| learning rate | 3e-4 |
| scheduler | none |
| hidden dim | 256 |
| loss | dense native-latent MSE only |
| teacher target | frozen、detached dense rollout |
| student encode_obs / goal call | false / false |
| optimizer | treatment 从 step 1 fresh initialize，不加载 low-density state_dict |

Seeds 全部继承 low-density h256 protocol：training_seed=20260925、context_schedule_seed=20260925、action_query_seed=20260926、role_schedule_seed=20260927、initialization_seed_observed=99、context selection seed base=20260928、heldout_action_prefix_seeds=[20264925,20264926]、timing_action_prefix_seed=20270925。保留原有 500-step NARROW compatibility burn-in（16000 context draws）和连续 1500-step WIDE schedule。不得根据结果换 seed、改 bank、调 lr、改 steps、改 mixture 或放宽 gates。

### 唯一的 context-density schedule

每个 batch 先生成权威 low-density 的 32-row schedule，再使用独立的 support schedule seed=20260929：从 32 个 positions 无放回随机选择 16 个 positions；将这 16 个 positions 替换为 append-support indices 128..255 中有放回抽样的 indices；剩余 16 个 positions 保持其 low-density index 不变。low-density schedule 的 500 个 historical steps 保持不变，high-density 每个 batch 仍恰好 32 rows。

因此每个 update 的总 batch/update 不变，新增 support exposure 固定为 16/32=50%；旧 support exposure 相应降为 16/32=50%，这正是 density treatment 本身，而不是额外的 role 或 query-mixture 变化。support position schedule 与 role schedule 独立，role schedule 保持原样；本规则不改变 logged/Gaussian/CEM 的 50/25/25 分配。

## Held-out ranking 与 density effect

held-out protocol 完全沿用 low-density h256：8 个 held-out contexts、2 个 fresh action-prefix seeds、每 block 300 candidates、top-30，共 16 个 paired blocks。两组使用相同 candidate action tensor、context、goal 和 frozen teacher costs。

每个 block 计算：

ΔSpearman = Spearman_high_density_h256 − Spearman_low_density_h256
Δtop30 = top30_high_density_h256 − top30_low_density_h256

primary density effect 是 16 个 paired deltas 的 median，而不是两个 arm-level medians 相减；positive-block count 使用严格 delta > 0。门槛保持：

| gate | 要求 |
|---|---:|
| median ΔSpearman | ≥ +0.05 |
| median Δtop-30 | ≥ +0.10 |
| positive Spearman blocks | ≥ 12/16 |
| positive top-30 blocks | ≥ 12/16 |

## Absolute、MSE、causality 与 latency

high-density h256 step1500 还需单独通过原 predictor-level gates：Spearman median ≥ 0.99、minimum ≥ 0.95；top-30 median ≥ 0.95、minimum ≥ 0.80；logged-action teacher-relative MSE 比 high-density h256 / low-density h256 ≤ 1.25；future-action leakage maximum absolute difference ≤ 1e-6；predictor-only latency reduction 相对 frozen teacher ≥ 20%。

latency boundary 仍是 cached native observation latent 加 normalized action prefix 的 predictor forward 对 frozen teacher rollout；包含 CUDA synchronize 和固定 warmup/repeat，不包含 observation encoder、CEM、environment 或 closed-loop control。high/low h256 latency ratio 仅作 diagnostic。

## 分层 decision

1. context_support_density_effect：四个 density thresholds 全部通过才 PASS，否则 NOT_ESTABLISHED。
2. full_replacement：manifest/integrity、absolute fidelity、MSE≤1.25、causality、latency 全部通过才 GO，否则 NO-GO。
3. density_supported_replacement：仅当前两者同时 PASS/GO 才 GO，否则分别报告。

density effect PASS 不自动意味着 replacement GO；replacement GO 也不自动意味着 density effect 成立。若 baseline artifact、manifest pairing 或任何 integrity boundary 无法验证，直接 NO-GO，不解释 ranking delta。

## 证据边界

允许的结论仅限：在冻结 DINO-WM PushT predictor-level contract 下，同一 32 个 train episodes 的 context-support 从 128 增至 256 是否改善 16 个 paired held-out blocks 的 candidate ranking，以及是否满足既有 predictor-level replacement gates。

禁止声称 query density/query coverage 改善、数据多样性/full-dataset coverage、new trajectory/new episode、closed-loop task success、multi-step CEM integration、完整 control-loop latency、LeWM transfer、population-level inference、all-JEPA universality、native low-bit deployment，或超出本 paired comparison 的 causal claim。

## 执行约束

本文件只冻结协议，不运行模型。真正运行时必须在 PBS compute-node allocation 内完成模型加载、teacher precompute、training、evaluation 和 GPU telemetry；login node 只允许作业提交、状态查询和轻量控制。每 30 秒把 GPU utilization 和 VRAM 写入 job log。不得下载 checkpoint，也不要求将 checkpoint 作为 primary result 返回。
