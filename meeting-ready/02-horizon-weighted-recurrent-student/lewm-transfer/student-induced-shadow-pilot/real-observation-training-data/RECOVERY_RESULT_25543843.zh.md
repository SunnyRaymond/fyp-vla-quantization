# 真实目标采集数据的独立恢复核验

CPU-only PBS `25543843.pbs101` 已以 `Exit_status=0` 完成。它没有重跑任何环境轨迹，也没有修改原始 `25543623` 汇总或 NPZ。在 compute allocation 内读取原始汇总、620 KB NPZ 与冻结的 80-task selection manifest，确认原 PBS 失败仅由最后的 `json.dumps(..., flush=True)` 打印错误触发，并验证 80 个任务（64 train/16 validation）、141 条样本（111/30）、数组形状与有限性、每条样本的 episode/split/window 身份及动作对齐误差 `≤1e-5`。

新生成的 `collection_summary_recovered.json` 状态为 `COMPLETE_WITH_TERMINAL_OR_BUDGET_LIMITED_WINDOWS`，含独立 `recovery_provenance`，明确记录原作业 `Exit_status=1` 和原文件未改动。16 个 validation episode 均有 t0 完整窗口，14 个有 t25 完整窗口；64 个预选 train episode 中 58 个有至少一个完整窗口。此结果只证明数据可用，不是模型改进证据。固定真实目标 finetune `25543873.pbs101` 已消费该恢复汇总并提交。

证据：[恢复汇总](results/25543843.pbs101/collection_summary_recovered.json)、[recovery job log 尾部](results/25543843.pbs101/job.log.tail.txt)、[原失败记录](ATTEMPT_25543623.zh.md)。
