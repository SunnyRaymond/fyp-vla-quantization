# LeWM PushT Phase 2：state/context coverage

本目录保存 frozen protocol、单个正式 PBS runner 和 predictor-level 结果。Phase 1
`256×1500 score_distill` formal result 作为 reference；本轮新增 `256×3000`、
`512×1500`、`512×3000` 三个 arm。新 512 contexts、HDF5 读取和 teacher preparation
只在 PBS compute node 执行。

- [冻结协议](PROTOCOL_LEWM_STATE_COVERAGE.zh.md)
- [freeze](LEWM_STATE_COVERAGE_FREEZE.json)
- [正式结果](RESULT_LEWM_STATE_COVERAGE.zh.md)
- [summary JSON](artifacts/24926383.pbs101/lewm_state_coverage_summary.json)
- [job log / 5-second GPU telemetry](artifacts/24926383.pbs101/job.log)
- [job status](artifacts/24926383.pbs101/job_status)

结论：manifest prefix、旧 256 train prepared-row prefix、held-out bank 均 bitwise
一致，但 primary `512×1500` treatment gate 为 `NO-GO`。本轮没有运行 official CEM、
planner viability 或 closed-loop PushT。
