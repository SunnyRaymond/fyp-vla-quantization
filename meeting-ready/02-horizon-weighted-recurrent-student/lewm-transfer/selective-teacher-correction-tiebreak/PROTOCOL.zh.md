# Selective teacher correction tie-break protocol

## 目的与边界

这是一个独立的 post-calibration refinement。它保留 `25269182.pbs101` 的 `INCONCLUSIVE` 结论，不修改原 summary，也不把原作业的未完成 test 当作结果。本轮唯一新的质量证据是独立的 `valid[584:592]` test；`valid[576:584]` 只用于冻结 tie-break 规则。

本轮仍是 fixed-observation candidate-ranking evidence。不会部署 official adaptive CEM，不做 closed-loop，也不把 teacher correction 写回 CEM proposal。

## 固定模型、数据与 CEM

main 是历史 `25239551.pbs101` 的 treatment `treatment_step1000.pt`，sentinel 是同一 job 的 control `control_step1000.pt`。两者共享 `25223859.pbs101` 的 anchor-aligned step3000 treatment 起点，历史训练是 non-concurrent reuse。runner 在 compute allocation 内锁定 checkpoint、start provenance、1000 updates 与 anchor provenance。

Test selection 使用原 `selection_seed=20300903` 的 `valid[584:592]`，排除 `[0:584)`；每个 episode 使用 early/middle/late 三个 anchor、action-prefix seeds `20301101/20301102`，共 24 contexts、48 条 context/seed trajectories、每条保存 CEM rounds 10/20/30，因此共 144 trajectory blocks。test banks 从 frozen step3000 student driver 产生，30 iterations、300 candidates、top30、horizon 5、candidate-zero 为 pre-update mean、unbiased elite std；teacher 仅作 shadow label。所有 paired policies 共享同一 test banks。

## Calibration 后 tie-break

`25269182.pbs101` 的 exact summary 与其 calibration bank provenance 必须在 compute node 上验证。summary 必须是 `INCONCLUSIVE`，原因为 `no calibration midpoint has 0 < call_rate <= 0.25`；其中 calibration 覆盖 24 contexts、48 context/seed trajectories、144 trajectory blocks、rounds 10/20/30。

runner 会加载该 job 保存的 exact calibration trajectory bank，并按同一 selection/row construction 重新计算全部 144 个 blocks 的 main/sentinel top30 overlap 与 `g`；每个 pairing key、`u` 和 `g` 都必须与 summary 对齐后才允许生成 cutoff。校验阶段不使用 bank 中的 teacher costs 做路由。

primary uncertainty 仍为

`u = 1 - |top30_main ∩ top30_sentinel| / 30`，

main/sentinel top30 都使用 pinned LeWM 的 `torch.topk(cost, k=30, largest=False, sorted=True).indices`。对 `u=1.0` 的 37 个 calibration blocks，再按预先记录的 teacher-free secondary metric

`g = 1 - Spearman(main_cost, sentinel_cost)`

降序排序。runner 必须从 exact summary 重新计算并断言第 36 个 `g=0.6875929732552584`、第 37 个 `g=0.6465405171168568`，以及 midpoint `tau_g=0.6670667451860576`；不得手抄数字或使用 teacher cost。最终规则固定为：

`call iff u > 1.0 OR (u == 1.0 AND g > tau_g)`。

严格不等号使 calibration calls 恰为 36/144=.25。tie-break rule 是在看到 calibration discreteness 后冻结的披露 refinement；test 不重新选 threshold、不使用 test ranking 或 block ID tie break。

## 指标与 gate

每个 block 用 300 个 teacher costs 的 population std（floor `1e-6`）标准化 teacher regret。每个 episode 对 18 blocks 取 mean，再跨 8 episodes 取 median。报告 block/episode mean、median、p90，以及每个 round 的 episode 结果。analytic random same-budget expectation 使用 `(1-m/18) * main_episode_mean`。

Gate 保持原设置：test call rate `<=.35`；selective−main episode mean delta 的 median `<=-.10` 且至少 5/8 episode 严格改善；main regret `>=1.5` 的 catastrophic blocks 至少 4 个且 teacher-call capture `>=.75`；selective 不差于 analytic random same-budget；native measured average latency reduction 相对 teacher300 `>=.30`；finite、shared pairing 与 provenance 必须通过。若 catastrophic blocks 少于 4，risk gate 为 inconclusive；不得用其他阈值替代。

Timing 覆盖 144 test blocks，3 warmups、10 repeats，固定交错顺序。用 CUDA synchronize 包围 `perf_counter`，包含 two-student scoring、uncertainty、teacher300/top30 branch 与 candidate H2D；一次性 model load、dataset read、shared row construction 不计入。报告 mean/p95 和 teacher-only baseline；`job.log` 用 direct `nvidia-smi` 每 5 秒 telemetry。

## 证据声明

本实验只能支持 fixed-observation selective candidate-ranking 的结果。无论 gate 是否通过，都不产生 adaptive CEM 或 closed-loop claim。相关 procedural context：Kassis et al. (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, DOI `https://doi.org/10.48550/arXiv.2609.00065`；该 citation 仅作方法背景，不改变本实验的 LeWM semantics 或 gate。
