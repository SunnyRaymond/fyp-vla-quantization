# LeWM State/Context Coverage Phase 2：实验结果

完整的 frozen protocol、manifest/prepared-row scope checks、四个 2×2 cells 的
snapshot、paired context-level deltas、absolute predictor gates、GPU telemetry 与
scope boundary 见：

[lewm-transfer/state-coverage/RESULT_LEWM_STATE_COVERAGE.zh.md](../lewm-transfer/state-coverage/RESULT_LEWM_STATE_COVERAGE.zh.md)

一句话结论：`512×1500` coverage-at-fixed-updates arm 的 median ranking 没有达到
冻结 gate（Spearman/top-30 `0.966364/0.800000` vs `0.974011/0.816667`），但 positive
top-30 达到 `16/16`、minimum top-30 达到 `0.433333`；`512×3000` 同时增加 coverage
和 exposure 后 median 达到 `0.978317/0.866667`，仍不能分离 coverage effect，也仍
未通过 absolute predictor gate。所有 manifest 和 held-out bank scope checks PASS，
本轮没有运行 official CEM、planner viability 或 closed-loop。

正式作业为 `24926383.pbs101`（`Exit_status=0`，walltime `00:06:05`）；job log 含约
5 秒 GPU utilization/VRAM telemetry。
