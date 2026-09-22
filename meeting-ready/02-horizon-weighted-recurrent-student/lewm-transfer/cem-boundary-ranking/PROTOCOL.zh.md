# CEM elite-boundary pairwise ranking protocol

## 目的与边界

本实验只问一个 fixed-observation predictor-level question：在同一 `anchor_aligned step3000` 起点、同一 `cem_distribution_train_rows.pt`、同一 1000 updates 和同一 AdamW/context schedule 下，加入 top30 边界 pairwise loss 是否降低 CEM full-300 teacher-cost regret。没有 planner deployment、official CEM 运行或 closed-loop 环境执行。

control 复用 `25239551.pbs101` 的 CEM-distillation treatment checkpoint，因此是 historical control reuse，不是与 treatment 同时重训的 control。summary、checkpoint 和 mixed rows 必须在 compute allocation 内验证到冻结的 canonical artifact directory `/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/cem-distribution-distill/artifacts/25239551.pbs101/` 及其 exact filenames，并核对 checkpoint/rows/provenance/1000 updates；不能只凭文件名认定来源。treatment 从同一个 anchor-aligned step3000 state 重新训练，保留原 recurrent latent loss 与 score-distill loss，只增加预冻结的 pairwise 项。

## 训练输入与 pairwise 项

两臂使用完全相同的 512 training contexts 和 64 candidates/context：原 anchor bank positions `0:32` 加上 CEM round10/20/30 固定 tail positions `32:64`。训练为 batch 8、1000 updates、AdamW `lr=3e-4, weight_decay=0.01, betas=(0.9,0.999), eps=1e-8)`，training seed=`20300982`，context schedule seed=`20300984`。

已有 score-distill 语义以 `score-distill/run_lewm_score_distill.py:131-151` 为依据，已有 training loop 以 `cem-distribution-distill/run_cem_distribution_distill.py:317-347` 为依据。新项按每个 context 的 teacher objective stable ascending order 取 positive ranks `[20,30)` 与 negative ranks `[30,40)`，形成 `10×10=100` 个 cross pairs。student objective lower is better，定义

`pair_loss = batchmean(context_mean_100(softplus((student_cost_positive - student_cost_negative) / teacher_population_std)))`，其中 temperature 固定为 `1.0`。每个 context 的 raw teacher population std `<=1e-6` 时立即 abort 并将 run 标为 invalid；否则使用 raw std，不把退化 context 静默 clamp。teacher cost 在排序与 std 计算中 detached。

仅 treatment 加 `0.05 * pair_loss`；latent loss、原 score-distill loss、其已有 `SCORE_LOSS_WEIGHT`、optimizer、schedule 和预算均不改变。`lambda=0.05` 在结果前冻结：零 margin 时贡献约 `0.05×0.693=0.0347`，与上一个 job 初始约 `0.0325` 的既有 `0.1×score_loss` 贡献相近；这是新增 auxiliary 的权重，不沿用 historical dense-rank 的替代式 `0.1`，也不做 lambda 或 pair interval sweep。

## Fresh paired evaluation

使用 selection seed `20300903` 的 `valid[568:576]`，排除 valid prefix `0:568`。8 episodes 各 early/middle/late 三个 anchor，两个新 action-prefix seeds `20301001/20301002`，共 48 blocks。两臂共享同一 frozen anchor-aligned step3000 student-driven CEM trajectory banks：30 iterations、300 candidates、top30，保存 round10/20/30；eval innovation seed base=`20301003`。teacher 只为共享 candidates 提供 targets/costs，不选择 CEM proposal，也不参与 sampling。

primary 是每个 full-300 bank 上 student top30 的 teacher-cost standardized regret：

`(teacher_mean(student_top30) - teacher_mean(teacher_top30)) / max(population_std(teacher_cost_300), 1e-6)`。

先在 episode 内对 18 个 CEM blocks（3 rounds×3 anchors×2 seeds）取 median，再在 8 episodes 上比较 treatment-control。报告 Spearman、top30 overlap、relative latent MSE、recall@60/120、full containment，以及 boundary inversion rate（100 pairs 中 student positive cost 不低于 negative cost 的比例）和 teacher boundary gap（negative teacher mean 减 positive teacher mean）。

另用同 fresh rows 的 standard current-anchor 300-candidate banks 做 forgetting guard；该 guard 与 CEM trajectory primary 分开报告。

## Gate 与停止条件

primary 通过条件为 episode-median delta `<=-0.05`、严格改善至少 `5/8` episodes、round10/20/30 各自 episode-median delta `<=0`；forgetting median delta `<=+0.05` 且没有 episode delta `>+0.20`；所有指标 finite、pairing valid。secondary 指标不改 gate。

无论 gate 结果如何，本轮均不推进 adaptive CEM、official planner 或 closed-loop。模型、mixed rows、fresh banks 和 checkpoints 只保留在 cluster；本地只回传小 summary、job log、status、终态和 GPU telemetry。

实验设计组织参考：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. https://doi.org/10.48550/arXiv.2609.00065 。本会话已核对当前 arXiv record；正式引用使用 latest DOI，不附 version suffix。该背景引用不构成 LeWM/PushT 实测证据，也不改变本实验的冻结 gate。
