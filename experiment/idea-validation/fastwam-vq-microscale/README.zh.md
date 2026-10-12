# Microscaling + additive VQ 最小数值验证

完整验证已完成：GPU job `25755896.pbs101` 为 F / Exit_status=0，实际 walltime 01:25:21，覆盖全部 614 Linears 和 52 paired queries。Test motor RMSE 下降 8.15%，14/22 cases 改善，通过预设工程门槛。完整分析、存储代价和边界见 [RESULTS.zh.md](RESULTS.zh.md)，原始小型结果保存在 `retrieved/`。

只新增 `micro_scale_vq_a4`，复用完整 action-aware 实验中已完成的 `row_scale_vq_a4` 和同输入、同 sampler seed 的 BF16 actions。既有实验结果已经看过，本轮定位为配对机制诊断。

全部 614 个 Linear 保持 d=4、两本 256-entry BF16 books、Smooth alpha=0.5、H128、dynamic per-row A4、BF16 KV。每个 Linear 仍共享两本 books。新方案沿输入维度以 32 个 weights 为块，保留原 BF16 row RMS gain，增加保存为 uint8 的共享 exponent：`S = r * 2**round(log2(block_RMS/r))`。Scales 在拟合中固定，books/indices 以原来的 8192 sample vectors、8 iterations、seed=20261006 重拟合；assignment metric 包含实际 `S**2`。本轮没有 local fine-tuning、action proxy、INT4 codebook 或 native kernel。

## 冻结的数据与计数

- Calibration：原 8 cases/各 seed index 0。
- Selection：原 4 cases/2 seeds，共 8 个新 arm queries。
- Test：原 22 cases/2 seeds，共 44 个新 arm queries。
- 主指标：前 10 步 6 维 motor RMSE，相对于同输入同 seed BF16；先在 case 内平均 seeds，再平均 cases。
- 工程门槛：test 平均 motor RMSE 下降至少 5%，至少 12/22 cases 改善；不是统计显著性声明。
- 原始拟合 input moments 没有落盘，按原 BF16W+A4 calibration 路径重算 8 个预备 queries，并对保存的原 actions 做 replay 检查；这些 queries 独立于 52 个新 arm queries。不会重新采集 BF16 teacher/sensitivity。
- Weight/局部 output 诊断复用已保存的 teacher captures。

## 运行与续跑

每个 PBS job 请求 `walltime=02:00:00`，1 GPU/16 CPU/110GB。数值工作只在 PBS compute allocation 内运行。使用原 Fast-WAM/Apptainer/venv/LIBERO-Plus 环境和 A100 hardware class；GPU 利用率与显存每 15 秒写入该作业 `job.log`。

Allocation 启动后 110 分钟为 soft deadline，并在模块/查询之间预留退出时间。Moment sums、每层 bank 和每次 query 原子保存。未完成时写 `status=checkpointed` 和 `PIPELINE_CHECKPOINTED`，下一次 <=2h 作业继续同一结果目录；完成覆盖 614 layers、8 selection contexts、44 test contexts 后才写 `status=complete` 和 `PIPELINE_COMPLETE`。

预计剩余时间不足时正常返回 `exit=0`，保存断点不会被判作实验失败。续跑会跳过已完成的 calibration cases、Linear banks 和 case/seed queries；只有正在处理且尚未保存的那一项需要重新执行。实际数值或数据错误仍保留非零退出码，便于修复后继续。

续跑时先对当前 handle 执行 `python control.py verify`。若结果为 `checkpointed`，执行 `python control.py submit --attempt b`，后续依次使用新的 attempt 名称。所有 attempts 共用远端 `results/full`；前一个作业仍为 Q/R 时不重复提交。`complete` 表示完整覆盖已经完成，此时无需续跑。

所有 source banks/results 只读。控制器通过具名提交 intent 和前一 handle 的 terminal 状态防止重复提交，allocation 中以 writer lock 防止并行写入。没有 kernel latency、闭环 episodes 或 INT4-compatible VQ 的结论。

CPU 自检作业 `25756051.pbs101` 已通过四项数值 assertions：e=0 decoder parity、尾部 shape、零行拟合、scale 平方进入 metric。请求 10 分钟，实际运行 1 分 12 秒，PBS 为 F / Exit_status=0；证据保存于 `cpu_terminal_evidence.json`。这些检查独立于已经完成的完整 GPU 验证。定时监控 `microscaling-vq-pbs-15` 已不存在于 app 中，当前没有该定时监控需要继续运行。
