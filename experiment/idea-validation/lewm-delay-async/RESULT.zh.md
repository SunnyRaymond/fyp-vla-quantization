# LeWM PushT / Reacher 延迟与异步闭环实验

状态：16 组作业及 CPU 汇总均完成。`25589225.pbs101` 生成 800 行配对记录，`25589449.pbs101` 补算 deadline 与计划应用指标；两者的 PBS、runner、job 退出码均为 0。每个 benchmark 的 8 个条件均完整覆盖同一组 50 个任务。

## 协议

使用原版 LeWM checkpoint 与 CEM（300 candidates、30 elites、30 rounds；5×5=25 个 control ticks 动作缓存），每个任务 50 个固定 dataset start rows，episode budget 50。LeWM source 固定在 `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`，stable-worldmodel 固定在 `10c26dbd5677083fa31dba69eb738b973845e9a4`。先跑官方 50-env 同步 K0，再用相同任务列表跑 N=1 同步 K0，后者是固定 K=1/2/4/8/16 与 true-async 的配对基线。固定 K 指计划计算结束后、执行计划首动作前环境先推进 K 个 control ticks；true-async 则在 CEM 计算期间按环境 control period 持续推进。所有条件使用同一 checkpoint 和任务身份，不重训。

## 已验证

| 项目 | PushT | Reacher |
|---|---|---|
| 资产与任务 | 原有 pinned PushT HDF5/checkpoint；沿用 `25534994.pbs101` 的 50 个 start rows | 官方 `quentinll/lewm-reacher` pinned revision 的 checkpoint 与 dataset 已在 CPU PBS `25583860.pbs101` 准备；HDF5 为 98,905,882,624 bytes、转换后 checkpoint 为 72,345,781 bytes；50 个唯一 start rows 在 `25583965.pbs101` 冻结 |
| 官方 50-env 同步 K0 | `25583858.pbs101`：49/50，50 项 success vector 与先前 `25534994.pbs101` 完全一致；runner gate `PASS`，PBS `Exit_status=0` | `25584103.pbs101`：31/50；50 项冻结任务与结果完整，PBS 和 runner 退出码均为 0 |
| N=1 同步 K0 | `25584172.pbs101`：46/50；50 项冻结任务完整，PBS 和 runner 退出码均为 0 | `25585153.pbs101`：30/50；50 项冻结任务完整，PBS、runner、job 退出码均为 0 |
| 固定 K / true-async | K=1/2/4/8/16 为 45/44/45/45/44；true-async 为 44/50 | K=1/2/4/8/16 为 30/28/27/26/25；true-async `25585349.pbs101` 为 14/50。所有条件均完整覆盖冻结的 50 项任务，PBS、runner、job exit 均为 0 |

PushT 的一个 control tick 实测为 0.1 秒。Reacher 官方 K0 作业记录 `dm_control` control timestep 0.02 秒、wrapper action repeat 2，因此每个 control tick 为 0.04 秒。`Stageout_status=1` 出现在已成功退出的 PBS 作业，保留为调度系统 caveat，不覆盖 runner gate 与 `Exit_status=0` 的证据。

## 配对成功率

下表以各任务的 **N=1 同步 K0** 为配对基线（PushT 46/50，Reacher 30/50）。`gain/loss` 是相同任务上的失败变成功／成功变失败；p 为双侧 exact McNemar，属于六条件探索性比较。官方 batch-50 K0 仅用于接入验证，不参与此配对。

| 任务 | 条件 | 成功数 | gain/loss | 净变化（百分点） | p |
|---|---|---:|---:|---:|---:|
| PushT | 固定 K=1 | 45/50 | 2/3 | -2 | 1.0000 |
| PushT | 固定 K=2 | 44/50 | 2/4 | -4 | 0.6875 |
| PushT | 固定 K=4 | 45/50 | 2/3 | -2 | 1.0000 |
| PushT | 固定 K=8 | 45/50 | 2/3 | -2 | 1.0000 |
| PushT | 固定 K=16 | 44/50 | 2/4 | -4 | 0.6875 |
| PushT | true async | 44/50 | 2/4 | -4 | 0.6875 |
| Reacher | 固定 K=1 | 30/50 | 3/3 | 0 | 1.0000 |
| Reacher | 固定 K=2 | 28/50 | 3/5 | -4 | 0.7266 |
| Reacher | 固定 K=4 | 27/50 | 3/6 | -6 | 0.5078 |
| Reacher | 固定 K=8 | 26/50 | 3/7 | -8 | 0.3438 |
| Reacher | 固定 K=16 | 25/50 | 2/7 | -10 | 0.1797 |
| Reacher | true async | 14/50 | 0/16 | -32 | 0.0000305 |

固定 K 的成功数不能证明严格单调下降，单个固定 K 条件的配对证据也不足以确认差异。Reacher true async 在本冻结协议下明显退化；这一结果只证明该 baseline 在这组起点、预算、算力与控制周期下的表现，尚不能把失败单独归因于环境快速变化。

## 推理延迟与实时性

| true async | control tick | worker CEM solve P50 | 计划应用时 observation age P50 | RTF P50 | 50 个 episode 的 missed tick 总数 | 最大 tick 启动迟到 | 无计划被执行 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PushT | 100 ms | 902.2 ms | 10.25 tick（约 1.025 s） | 1.029 | 0 | 1.00 ms | 0/50 |
| Reacher | 40 ms | 982.0 ms | 26 tick（约 1.04 s；48 项有值） | 1.019 | 1 | 41.03 ms | 2/50 |

这里的 solve 时间是 30-round CEM worker 调用（含 CUDA 同步），**不是一次 JEPA predictor forward，也不包含 async control thread 的全部观测预处理**。observation age 则按环境真正推进的 control tick 计。Reacher 的首个计划约在 50-tick episode 预算的一半时才可应用；这与 14/50 的下降相符，但仍是机制推断，不是单独隔离的因果实验。Reacher 两个任务在终止前没有计划被执行，所以它们的 age 缺失；其中一个仍被环境成功判据记为成功。

true async 的 RTF 接近 1。RTF 按首个环境 step 开始至末次 step 结束计算，略大于 1 有末尾 tick 边界效应；不能单凭它声称 hard realtime。deadline trace 显示 PushT 没有漏掉整 tick，Reacher 有一个 episode 漏掉 1 tick；两者最大启动迟到的 P95 分别为 0.588 ms 与 0.212 ms，Reacher 尾部最大为 41.03 ms。固定 K 是不按墙钟节拍推进的 simulation-age 实验，其 RTF 不与 true async 作实时性能比较。原生 N=1 K0 路径没有同口径的 CEM 计时，故不从本表推断同步与异步的推理加速。

## 证据位置

- 协议与实现：[PROTOCOL_DRAFT.zh.md](PROTOCOL_DRAFT.zh.md)、[ASYNC_ARCHITECTURE.zh.md](ASYNC_ARCHITECTURE.zh.md)
- PushT K0：[batch50_official_k0.json](pusht/artifacts/25583858.pbs101/batch50_official_k0.json)
- PushT N=1 K0：[run_summary_paired_k0_k0_0_50.json](pusht/artifacts/25584172.pbs101/run_summary_paired_k0_k0_0_50.json)
- PushT K=1：[run_summary_fixed_steps_k1_0_50.json](pusht/artifacts/25585122.pbs101/run_summary_fixed_steps_k1_0_50.json)
- PushT K=2：[run_summary_fixed_steps_k2_0_50.json](pusht/artifacts/25585124.pbs101/run_summary_fixed_steps_k2_0_50.json)
- PushT 其余结果：[K=4](pusht/artifacts/25585126.pbs101/run_summary_fixed_steps_k4_0_50.json)、[K=8](pusht/artifacts/25585128.pbs101/run_summary_fixed_steps_k8_0_50.json)、[K=16](pusht/artifacts/25585129.pbs101/run_summary_fixed_steps_k16_0_50.json)、[true async](pusht/artifacts/25585130.pbs101/run_summary_true_async_k0_0_50.json)
- Reacher 冻结任务：[frozen_manifest.json](reacher/frozen_manifest.json)
- Reacher 官方 K0：[result.json](reacher/artifacts/25584103.pbs101/results/result.json)
- Reacher N=1 K0：[summary.json](reacher/artifacts/25585153.pbs101/results/summary.json)
- Reacher fixed K：[K=1](reacher/artifacts/25585340.pbs101/results/summary.json)、[K=2](reacher/artifacts/25585342.pbs101/results/summary.json)、[K=4](reacher/artifacts/25585344.pbs101/results/summary.json)、[K=8](reacher/artifacts/25585345.pbs101/results/summary.json)、[K=16](reacher/artifacts/25585347.pbs101/results/summary.json)
- Reacher true async：[summary.json](reacher/artifacts/25585349.pbs101/results/summary.json)
- 完整映射：[final_mapping.json](analysis/final_mapping.json)；正式 [800 行 episode 记录](analysis/episodes_all_20260928_deadlines.json) 与 [配对／时序摘要](analysis/episodes_all_20260928_deadlines.analysis.json)
- 作业状态：[RUN_STATE.json](RUN_STATE.json)

所有 GPU 作业在各自 job log 内定期记录了利用率与显存；`Stageout_status=1` 是 PBS caveat，scratch 中的结果、日志及 runner/wrapper 退出文件均可读取，PBS `Exit_status` 也为 0。上述统计由获批的 CPU PBS allocation 计算，原始 per-task success 与调度事件保留在各 job 的 scratch artifact 中。
