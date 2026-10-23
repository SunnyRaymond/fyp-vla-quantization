# C80 per-channel scaling + clipping：280-episode 闭环评测

直接复用 `fastwam-vq-per-channel-scaling-clipping-20261018` 中 `25862598.pbs101` 完成的 **per_channel_clip / final update 80** deployment。本轮不重新训练，不替换成 B32 scaling 或 codebook joint checkpoint。

## 固定比较条件

- 沿用原 frozen manifest：4 suites × 7 perturbation dimensions × 10 variants，共 280 episodes。
- 保持相同 state 0、environment seed、sampler seed、10/10 denoising steps、32-step action horizon、每次执行 10 actions，以及原 episode cap。
- 一个 PBS task：4 GPU、24h；四个常驻 worker 按 `index % 4` 分配，每卡 70 episodes，每张卡加载一次模型。
- 复用原先十个配置的 2,800 个 reference episodes，输出 overall、suite、dimension 和 suite×dimension 的 paired win/loss/tie。
- 运行期间每 15 秒向 `job.log` 记录四张卡的 GPU utilization 和显存占用。

## 部署路径与启动检查

保留 C8 rho1.5 的 codebooks、indices、row gains、原 B32 exponents 和固定 Smooth0.5/H128；恢复源 checkpoint 的 614 个 clipping ratios 与 1,806,863 个 FP32 per-input-channel log2 scales。A4 前除以连续 scale，decoded BF16 weight 乘以同一 scale，正式 episode 前缓存补偿后的权重一次。

每个 worker 有三个不计入 episode timing 的 startup queries：复现保存的 source endpoint 七项指标；检查缓存前后的完整 action/future 一致性；检查正式 `infer` 路径的 action 一致性。源实验没有保存 final action vector，因此这项检查不称为 saved-action replay。

运行环境、采样与 episode 执行复用已完成的 C80 clip/joint 闭环代码。`source_protocol.json` 固定源训练配置；`protocol.json` 固定本轮闭环配置。源训练和 aggregate 的 scheduler/receipt 状态存于 `source_authority.json`。

## 结果边界

目标证据是 frozen 280 episodes 的闭环 success rate 与配对差异。执行使用 **decoded BF16 compensated weights + dynamic A4 + BF16 KV**；显存和 query timing 只对应此实现，不代表 native low-bit kernel 的性能。

`control.py` 保存 qsub 前的 durable intent，并按唯一 job name 恢复不确定的提交结果，避免重复提交。所有模型加载、数据读取、数值检查和聚合均在真实 PBS allocation 中执行。监控由 `gpt-6-luna / xhigh` subagent 每 20 分钟检查；队列或运行状态无实质变化时保持安静。

## 已提交任务

`25863760.pbs101` / `c80pceval0001`，2026-10-18 15:07 SGT 提交；4 GPU、24h，scheduler 放入 g3。按站点规则扩为 64 CPU / 440 GB。监控：`/root/channel_long_watch`，gpt-6-luna / xhigh，20min 间隔。

最终状态：25863760.pbs101 已以 PBS F / Exit_status=0 完成，实际 04:22:06；280/280 episodes、2800 references、4×70 worker receipts 与全部完成标记一致。结果 203/280（72.50%），见 RESULTS.zh.md。监控已停止。
