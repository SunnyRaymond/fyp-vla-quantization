# LeWM PushT adaptive teacher schedule protocol

## 目的与边界

本实验只测 fixed-observation fully adaptive CEM 中，少量、预先固定的 teacher schedule 是否能减少累计搜索偏移。它不训练模型，不执行 environment closed-loop，也不把结果解释为 official planner deployment 或 PushT task success。旧实验目录和结果保持不变。

主模型精确绑定 `cem-distribution-distill/artifacts/25239551.pbs101/treatment_step1000.pt`，其 provenance 必须声明 source=`cem_distribution_distill`、arm=`treatment`、extra updates=`1000`。teacher 是 official LeWM。所有模型、HDF5、推理和 benchmark 只在带真实 `PBS_JOBID` 且 hostname 非 login/head/submit 的 compute allocation 内运行。

## Frozen CEM 与四臂

每个 fresh episode 的 early/middle/late 三个 anchor 和两个 action-prefix seeds 组成 6 个 paired trajectories；8 episodes 共 48 条 trajectory。每条轨迹运行 30 rounds，每轮 300 candidates、top30、horizon 5、packed action dimension 10。初始 `mu=0`、`sigma=1`，candidate zero（candidate 0）是 pre-update `mu`；用 `torch.topk(cost, k=30, largest=False, sorted=True)`，elite `std(unbiased=True)`，不做 clip。

每个 pair 只生成一份 30×300 innovation tensor，四臂共享该 tensor，但各臂独立更新自己的 `mu/sigma`：

- `student_only`：全部 30 轮由 main student scoring 和更新；
- `uniform_teacher7`：teacher rounds 为 `[4,8,12,16,20,24,28]`，其余轮由 student 更新；
- `late_teacher7`：teacher rounds 为 `[24,25,26,27,28,29,30]`，其余轮由 student 更新；
- `teacher_only`：30 轮全部由 teacher 更新。

teacher schedule round 的 teacher top30 必须真实用于下一轮 `mu/sigma`；不使用 sentinel、uncertainty 或 teacher 只作 shadow label。schedule call 数必须分别为 0、7、7、30。

## Fresh data 与 primary

fresh selection 固定为 `valid[592:600]`，selection seed `20300903`，排除 `valid[:592]`；action-prefix seeds 固定为 `20301105, 20301106`。episode 是 nested replicate，不能将 48 trajectories 或 30 rounds 当独立样本。

每臂保存 rounds 10/20/30 的 post-update `mu/sigma`。对每个 pair/round，用 teacher objective 直接评估该 post-update `mu`，并减去同 pair、同 round 的 `teacher_only` post-update `mu` objective，再除以该 pair 的 teacher-only round-1 300-candidate population std，floor `1e-6`。冻结 `primary_round=30`：每 episode 对 3 anchors×2 seeds 的 6 个 final observations 取 mean；round10/20 只作 diagnostic trajectory，不能混入 primary。

round30 分别计算 `uniform_teacher7 - student_only` 和 `late_teacher7 - student_only` 的 paired episode delta；同时报告 `uniform_teacher7 - late_teacher7`。若至少一个 7-call schedule 的 round30 median delta `<= -0.10` 且至少 5/8 episode 严格改善（delta `<0`），质量部分通过。另报告各臂相对 teacher-only 的 round10/20/30 gap、teacher objective probe、final first-action L2 drift（相对 teacher-only final `mu`），以及每 episode 的 round10/20/30 diagnostic trajectory。

secondary distribution probe 在 round30 post-update `mu/sigma` 上用独立 frozen epsilon；同一 pair 内四臂共享 epsilon transform，但各自以自己的 `mu/sigma` 生成 candidates。teacher top30 mean-cost 的标准化 gap 只作 secondary，不替代 primary。

## Timing 与 gate

native timing 覆盖完整 30-round adaptive trajectory，包括 candidate generation、H2D、student/teacher scoring、top30、`mu/sigma` update；每个 block 使用 CUDA synchronize 和 `perf_counter`，固定 arm 顺序，3 warmups 后至少 5 repeats，报告 mean/p95。schedule 的 mean latency 相对 teacher-only 必须减少至少 30%。

所有 48 paired trajectories、schedule calls、finite、shared innovations、exact checkpoint provenance 必须通过。整体仅在至少一个 7-call schedule 同时通过 primary quality 与 latency gate 时为 `PASS`，否则为 `FAIL`。无论结果如何，`official CEM` 与 `closed-loop` 均保持 `NOT_RUN_BY_SCOPE`。

方法和 reporting 组织参考 Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, latest DOI: https://doi.org/10.48550/arXiv.2609.00065。该 citation 仅为方法背景，不是 LeWM/PushT 实测证据。
