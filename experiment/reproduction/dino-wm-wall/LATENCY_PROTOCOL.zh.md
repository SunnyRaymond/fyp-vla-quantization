# DINO-WM Wall 端到端 latency profiling protocol

目标是用官方 `wall_single` checkpoint 跑两个 Wall cases，并定位完整流程与 planner 内部 latency；不修改模型、不重新训练、不把 profiling 数字当作 native low-bit 或 production deployment latency。

## 固定设置

- 官方 source commit：`0a9492fa12044b852ae9e001cc74604b79c8bb0c`
- checkpoint：`outputs/wall_single`，epoch 65
- Wall config：seed 99、`goal_H=5`、CEM 300 candidates / top-k 30 / 10 optimization steps
- 两个 cases 使用官方 deterministic eval seeds；共享一次模型加载，但 CEM 对 case 逐个处理
- `max_mpc=12`，case 成功或达到上限都进入 final replay/export；不会使用官方 `max_iter: null`
- 不额外消耗一个 task 做 warmup；首个调用保留为 cold-start observation，后续调用可用于 steady-state 对照

## 输出

- `latency_events.jsonl`：层级事件，包含 MPC round、CEM iteration、case index、evaluator kind、wall time 与 CUDA sync 标记
- `run_summary.json`：完成性、success、总时间、显存与 profiling 边界
- `latency_summary.json` / `LATENCY_REPORT.md`：按 label、MPC round 和 evaluator kind 汇总
- `cases.json`、`trajectories.npz`、两个 case 视频：final replay 的任务结果

细粒度 GPU 区间在边界执行 `torch.cuda.synchronize()`，因此能把异步 kernel 归属到具体 block，但会产生测量扰动；这些数值用于瓶颈定位，不直接代表无 instrumentation 的控制频率。
