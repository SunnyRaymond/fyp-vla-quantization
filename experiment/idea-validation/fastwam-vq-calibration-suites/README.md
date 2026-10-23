# Original LIBERO four-suite calibration comparison

## Frozen comparison

Only two calibration cohorts are evaluated. Both use original LIBERO, with no LIBERO-Plus observations.

| Cohort | Per suite | Total | Coverage |
| --- | ---: | ---: | --- |
| C8 | 2 contexts from 2 distinct tasks | 8 contexts | Nested subset of C80 |
| C80 | 10 tasks × 2 distinct demonstrations × 1 snapshot | 80 contexts | All 40 tasks; early/middle/late snapshots |

The suites are `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`. C80 has 7 early, 7 middle, and 6 late snapshots per suite. The demonstration IDs and phase assignment are selected once with seed 20261013 and retained in `data/plan.json`. The corresponding dataset revision is recorded in `data/source_receipt.json`.

The complete frozen input plan is saved locally in `retrieved/data/plan.complete.json`; `retrieved/data/plan.json` retains the initial frozen snapshot. It was sampled before evaluating either cohort, with no sensitivity ranking:

| Suite | C8 task IDs | C8 phases |
| --- | --- | --- |
| LIBERO Spatial | 0, 9 | early, late |
| LIBERO Object | 0, 2 | middle, middle |
| LIBERO Goal | 5, 8 | middle, middle |
| LIBERO 10 | 3, 5 | middle, early |

C8 therefore has 2 early, 5 middle and 1 late contexts. C80 has 28 early, 28 middle and 24 late contexts. This phase-coverage difference is part of the comparison; do not resample C8 after seeing results.

The common evaluation cohort contains 40 contexts: one distinct held-out demonstration per task, with 2 sampler seeds each, yielding 80 action queries per cohort. Calibration and evaluation share tasks but never demonstration IDs. "Held-out" means held out from quantizer calibration; it does not mean held out from the released model's training data. Primary reporting averages seeds within a context, contexts within a suite, and then the four suites equally. This does not establish generalization to unseen tasks.

Input preparation restores each selected demonstration's `model_file` scene XML, then its saved MuJoCo state. This follows the original [LIBERO dataset builder](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/scripts/create_dataset.py). It generates snapshots without replaying actions or adding settling steps. Merely setting simulator state does not restore scene XML, as shown by the original [environment wrapper](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/envs/env_wrapper.py).

## Fixed method and evidence boundary

Reuse the completed Microscaling VQ implementation: all 614 target Linears, vector dimension 4, two additive codebooks of 256 BF16 entries, and a power-of-two B32 scale within each output row. The saved SmoothQuant alpha=0.5 / H128 transforms, fitting budget and random seed remain fixed. Recompute BF16W+A4 input moments for each new cohort, with equal context weights. The inherited transforms are not refitted using original-only data; this isolates the new moment/fitting cohort comparison. Local tuning and action-sensitivity fitting are disabled for this comparison, as specified in `protocol.json`.

Evaluation decodes weights into BF16 and applies simulated A4 activations. It measures same-input action fidelity against a shared BF16 teacher. There are no closed-loop episodes, native INT4 kernels, throughput measurements, or new method-selection arms. C8 versus C80 changes quantity, task coverage and phase coverage together; interpret the result as a comparison of these two calibration recipes.

## Execution and continuation

Remote campaign root: `/scratch/users/ntu/yguo017/fastwam-vq-calibration-suites-20261013`.

1. Verify CPU inventory and prepare 120 original demonstration inputs.
2. Generate the common BF16 teacher for 80 evaluation queries.
3. Submit C8 and C80 as separate jobs. Each computes its input moments, fits 614 layers and evaluates 80 queries.
4. Compare paired context errors and per-suite results after both cohorts are complete.

Every formal job requests `01:59:00`. A soft deadline checkpoints completed units and exits normally. Continuations preserve the exact plan and skip completed inputs, moments, layers and query records. A checkpointed job is not a completed experiment.

Control commands, run from the project root:

```powershell
python experiment/idea-validation/fastwam-vq-calibration-suites/control.py status --cohort cpu
python experiment/idea-validation/fastwam-vq-calibration-suites/control.py verify --cohort cpu
python experiment/idea-validation/fastwam-vq-calibration-suites/control.py submit --cohort cpu --attempt b
python experiment/idea-validation/fastwam-vq-calibration-suites/control.py submit --cohort teacher --attempt a
python experiment/idea-validation/fastwam-vq-calibration-suites/control.py submit --cohort c8 --attempt a
python experiment/idea-validation/fastwam-vq-calibration-suites/control.py submit --cohort c80 --attempt a
```

Use the next unused attempt label when continuing. Submission checks prerequisites and existing job handles; do not bypass them. Only submit teacher after CPU preparation is verified complete, and only submit C8/C80 after the teacher is verified complete. GPU jobs record utilization and memory every 15 seconds.

ASPIRE2A login nodes are used only for small control files, submission and bounded status/log reads. Data downloads, HDF5 access, rendering, model execution, fitting and aggregation require a genuine PBS compute allocation. Keep SSH host-key verification enabled and never print credentials. Failure of a compute-node network request does not authorize downloading on a login node.

## Current state

See `RUN_STATE.json` for exact handles and remaining work. Inventory job `25785457.pbs101` completed successfully: the 40 required demonstration HDF5 files are absent locally; required reading libraries are available. CPU preparation job `25785628.pbs101` failed while contacting the official Hugging Face revision API, before generating inputs. Probe `25785746.pbs101` established that host HTTPS succeeds while container certificate verification fails. The CPU HTTP container now receives the trusted host CA bundle through a read-only bind, retaining verification. Attempt b was cancelled before generating inputs to correct scene-XML replay. Attempt c, `25785804.pbs101`, uses both fixes. No C8/C80 accuracy results exist yet.

Attempt c froze the full 120-context plan after scanning all 40 task files, then failed on publisher-specific XML asset paths. Attempt d (`25786765.pbs101`) exposed a path-normalization guard that rejected legitimate parent-directory references. The renderer now relocates original LIBERO asset references only when the fully normalized path stays within the original asset root, including valid directory re-entry. Scene geometry, physics and camera attributes remain unchanged. The next attempts reuse the frozen plan directly and start at rendering; the metadata scan is retained. Initial failure logs and the frozen plan/source receipt are saved under `retrieved/`.


CPU attempt e (`25786876.pbs101`) completed with Exit_status=0 in 00:34:02 and verified all 120 contexts (80 calibration, 40 test). Its summary and exit code are saved under `retrieved/artifacts/25786876.pbs101/`. The complete plan is saved at `retrieved/data/plan.complete.json`. Shared BF16 teacher attempt a (`25787335.pbs101`, requested 01:59:00) failed on startup after 00:00:06 because the new environment omitted the pilot and diagnostic runtime directories from PYTHONPATH. The environment now includes both PILOT_ROOT and OLD_INPUTS, after the original LIBERO directories; shell syntax passed and the exact remote module files were confirmed present. The failure log and exit code are saved under `retrieved/artifacts/25787335.pbs101/`. No teacher query was completed by that attempt. The controller owns the next teacher submission and will release C8 and C80 separately after verified teacher completion. C8/C80 numerical results remain pending; queued jobs must not be resubmitted. Current live status and continuation ownership are recorded in `RUN_STATE.json`.


Teacher attempt b (`25787525.pbs101`) completed with F/Exit_status=0 in 00:11:57. The controller verified its completion marker, exit receipt and summary: 40 cases, 80 queries, 614 target Linears and 10 contexts per suite. The teacher summary and exit receipt are saved locally under `retrieved/artifacts/25787525.pbs101/`. C8 attempt a (`25787876.pbs101`) and C80 attempt a (`25787881.pbs101`) were then submitted separately, each requesting 01:59:00 with the verified teacher dependency. Both exact handles were independently confirmed Q. Numerical cohort results and the paired comparison remain pending; continuation monitors these handles and preserves completed units.


User requested a pause while the GPU jobs wait/run. The continuous controller subagent has stopped. A verified ACTIVE standalone automation, `original-libero-c8-c80-calibration-monitor`, uses gpt-6-luna / xhigh every 15 minutes on this local project. It reads current cohort handles and records deduplicated events in `scheduled_monitor_state.json`. It stays quiet for unchanged Q/R, and wakes original thread `01a11aa2-0dd6-7e11-899f-47111e161f6a` for failure, normal checkpoints requiring continuation, or both cohorts verified complete. The original thread retains all numerical continuation, repair and CPU-comparison work. Keep the local computer and app running for this local scheduled monitor.


Attempts a for C8 (`25787876.pbs101`, 00:10:17) and C80 (`25787881.pbs101`, 00:10:10) both failed with Exit_status=1 because progress() forwarded the stage keyword twice before the first calibration context. Model initialization and Microscaling self-check passed, but completed moments, banks and evaluation queries were all zero. The minimal fix defaults stage inside the kwargs dictionary and forwards it once; a pure Python regression exercised both explicit and default stages. The protocol and numerical settings are unchanged. All 120 prepared inputs and the 80-query BF16 teacher are reused. Attempts b are C8 `25789685.pbs101` and C80 `25789728.pbs101`, each requesting 01:59:00. The original automation has been converted in place to a heartbeat targeting the same independent monitoring conversation `01a13526-4ccf-7873-be3d-65cab73f52a2`; each run returns there instead of creating a new conversation. It continues every 15 minutes and wakes the original thread only for handling-worthy terminal events.


C8 attempt b (`25789685.pbs101`) completed in 01:41:10 with verified F/Exit0, completion marker and all 8 calibration contexts / 614 fitted Linears / 40 cases / 80 queries. A controller type comparison (JSON list versus Python tuple) was corrected without weakening coverage checks; live C8 verification now passes. C80 b (`25789728.pbs101`) exited normally with a verified checkpoint after all 80 calibration contexts and 614 fits, retaining 8 of 80 test queries. C80 c (`25791519.pbs101`) continues the remaining 72 queries. CPU-only C8 saved-query analysis (`25791581.pbs101`) completed and verified in 00:00:08, with zero new model forwards. The local report and analysis are in `retrieved/results/c8_analysis/`. Primary C8 suite-macro motor RMSE is 0.0464021562. Three gripper outliers (3027, 3029, 3031) each show one large first10-step difference in one seed; these will be paired with C80 only after all C80 queries complete. Full paired C8/C80 comparison remains pending.

### C80 resume failure and repair (2026-10-13)

Attempt c `25791519.pbs101` ended F/Exit_status=1 after 00:10:03. It failed before new calibration/fitting/query work: `Moment checkpoint lists a missing context: 2000`. The collection loop reused the previous restore-loop path, so per-context files were overwritten at the last context path. The preceding job nevertheless retained the correct in-memory contexts through aggregation and fitting, and saved `moments/final.pt`, 614 frozen banks, and 8 paired query records. The repair writes the current context path and validates/reuses the saved full aggregate in a compute allocation; numerical protocol and existing C8/C80 banks remain fixed. The next unused attempt is d.

Repair published (51,478-byte runner). Attempt d `25791907.pbs101` confirmed, walltime 01:59:00. It validates the saved full aggregate, restores all 614 existing frozen banks and 8 paired query records, and continues the remaining 72 queries in a GPU allocation. Root will confirm `MOMENTS_REUSED final contexts=80 modules=614` in the running log before treating recovery as verified. Protocol, transform, VQ budget and test coverage are unchanged; final comparison remains pending.

The full CPU comparison report now includes cases 3027/3029/3031, fixed from C8 preliminary findings: six same-case/same-seed gripper comparisons, saved first10 RMSE, maximum normalized-output difference and zero-based step, plus two-seed case means. This reads saved JSON outputs only and does not select a quantization method or add forwards. Existing primary metrics, 40-case/80-query coverage, 614-Linears gates and tie tolerance remain fixed. Syntax and synthetic control fixtures passed; actual full comparison awaits C80.

Actual attempt-d recovery verified in the GPU allocation: at 00:09:55 the log contains `MOMENTS_REUSED final contexts=80 modules=614`, and progress reports calibration_completed=80, fitted_linears=614, queries_completed=8 at case 3004/seed0, stage paired_shared_test_queries. No calibration forwards were repeated and the saved full moment aggregate was not reaggregated. Receipt: `c80_moment_reuse_d.json`.

### Fixed monitor and requested goal pause (2026-10-14)

The existing automation `original-libero-c8-c80-calibration-monitor` is ACTIVE every 15 minutes in the single reused monitor thread `01a13526-4ccf-7873-be3d-65cab73f52a2` (Original LIBERO C8 C80 calibration monitor). Its archived thread was restored; actual rollout confirms gpt-6-luna/xhigh and the heartbeat inherits that unchanged setting. Clean prompt reads current handles (C80 25791907.pbs101), keeps ordinary Q/R and query-count progress quiet, and is explicitly authorized to wake original thread 01a11aa2-0dd6-7e11-899f-47111e161f6a on failure, a verified resumable checkpoint, or both full cohort results. It creates no new thread per run. Results, runner repairs and comparison diagnostics are saved. On wake, root must finish the full 40-case/80-query-per-arm CPU paired comparison before marking the original goal complete.

### Full C8/C80 comparison saved (2026-10-14)

C80 attempt d 25791907.pbs101 finished F/Exit0 after 01:21:20, verified complete with calibration80, fitted614 and paired queries80. C8 remains verified complete with calibration8, fitted614 and queries80. CPU comparison 25793267.pbs101 completed F/Exit0 in 00:00:07 with PIPELINE_COMPLETE, artifact exit0 and matching 40-case/80-query-per-cohort/614-Linears summary. No model forwards were added for analysis. Source jobs are teacher25787525, C8 25789685, C80 25791907. Full report and 68,900-byte JSON are saved under retrieved/results/comparison/.

Four-suite macro: first10 motor RMSE C8=0.0464021562, C80=0.046636498 (C80 +0.505%); first10 gripper C8=0.0454658921, C80=0.045439792 (-0.0574%); 32-step motor C8=0.0523299512, C80=0.050788815 (-2.945%). C80 has no consistent advantage on the primary first10 metric; it improves the 32-step secondary metric. In the predefined C8 gripper diagnostics, case3031 improves substantially while case3029 acquires another large single-step discrepancy; case3027 remains large. These are fixed-input decoded-BF16/A4 results, not native-kernel, speed or closed-loop evidence, and the recipes jointly change context/task/phase coverage.

Monitor DNS repair is a separate pending acceptance item: the actual scheduled context forces workspace-write/network=false despite trusted project config. The ineffective project config was removed. Four exact status/verify commands are now allowlisted in C:/Users/Raymond/.codex/rules/calibration-pbs-monitor.rules; codex execpolicy check confirms allow for all four and no match for submit. The automation was restored after archiving removed it, using the same ID and the same gpt-6-luna/xhigh monitor thread. Actual scheduled execution must still prove the allowlist takes effect before claiming repair complete.

### Final delivery and monitor acceptance (2026-10-14)

Goal tool confirmed complete after both original-only C8/C80 cohorts and the full CPU paired report were verified. All requested 4-suite denominators are preserved; there are no pending PBS experiment or analysis jobs. The fixed gpt-6-luna/xhigh monitor turn 01a13634-c25e-7691-8382-6eacaef5742c executed status and verify for both current cohorts with command exit0, demonstrating the read-only DNS/network repair works in a real scheduled turn. Automatic wake-up is a separate unfulfilled capability: send_message_to_thread returned `MCP tool call requires approval, but approval policy is never`, so no automated wake message was delivered. Do not claim this wake loop is working in a future handoff. The monitor is PAUSED because this experiment is fully complete; its independent thread and scoped read-only command rules remain available. Goal cumulative active time reported by the tool: 22,386 seconds (about 6h13m).

### 后续 calibration 建议（2026-10-14，讨论记录）

建议开发迭代沿用已经冻结的 C8，正式四-suite 方法比较保留现成 C80。C80 的主要理由是完整 task/demo/phase 覆盖；当前 first10 motor 主指标并未显示其精度优势。暂不增加 calibration 总量，也不新增未经验证的中间预算 arm。本建议未修改 protocol、已有 banks 或实验状态，未提交新作业。

| 用途 | 每 suite | 总 contexts | 抽样与使用 |
| --- | ---: | ---: | --- |
| 快速调试 / 方法迭代 | 2 | 8 | 继续使用现有冻结 C8；先检查实现与拟合是否正常 |
| 正式跨-suite 比较 | 20 | 80 | 全部 10 tasks × 2 distinct demos/task × 1 snapshot/demo；复用现有数据和已完成结果 |

数据源保持 original LIBERO 四 suites 等权。C80 每 suite 7 early / 7 middle / 6 late，每个 context 的 moment 先按有效 token rows 归一化，再等权聚合；保证长输入不会因 token 数而获得额外权重。Snapshot phases 目前表示轨迹时间分位，不等于语义上的抓取/释放阶段。LIBERO-Plus 可以继续作为独立扰动评测；此次 C8/C80 比较不能证明 original-only 比旧混合 calibration 更好。

结果依据：C80 相对 C8，first10 motor RMSE +0.505%，first10 gripper -0.0574%，32-step motor -2.945%。C8 只覆盖 8/40 tasks，只有一个冻结的 calibration draw，整体 phase 为 2 early / 5 middle / 1 late；两个 sampler seeds 并不是两个独立 calibration draws。因此不能据此断言 C8 足够、最优或与 C80 等价。Gripper 聚合接近不变伴随逐 case 改善/恶化的抵消，不能视为已解决 outliers。

下一阶段 action-aware fitting 尚需独立验证：本轮 action_lambda=0，不能将其预算结论直接推广到 action-sensitivity moments。先以 C8 检查 probes/gradients/目标定义，再用同一 original-only C80 池做正式拟合；重新计算需要的 action-aware 统计量，现有 local moments/banks 仅作对照。若调 lambda、gripper 权重或候选 selection，须使用与 calibration/test 不同 demonstration 的独立 selection split；不要把已观察到的 test outlier cases 回填为 calibration，也不要按 test 结果重抽 C8。拟合目标应匹配实际执行的 action horizon，保留 first10 与 full32 分开报告。

当前 original-only 指新收集的 calibration inputs/moments/codebook fitting；Smooth0.5/H128 transforms 仍继承旧冻结版本。若后续研究完整 original-only 重拟合 pipeline，需作为另一个预先定义的比较，不能给现有结果补上这一标签。
