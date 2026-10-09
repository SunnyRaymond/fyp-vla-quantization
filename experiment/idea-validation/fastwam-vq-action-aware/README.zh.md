# Fast-WAM action-aware VQ 实验

本目录包含输入准备、GPU 数值实验和 CPU 汇总的 PBS 脚本。所有渲染、模型加载、拟合、推理和指标计算都必须在对应 PBS allocation 内执行；登录节点只用于小型控制文件上传、提交和状态查询。脚本在任何实际工作前检查 `PBS_JOBID`、hostname 和 `PBS_NODEFILE`。GPU 阶段每 15 秒把 GPU 利用率与显存记录写入该作业的 `job.log`。

## 固定设计

- Fresh observation 共 26 个：selection 是原始 LIBERO-Spatial task IDs `[1, 3, 6, 8]` 的 state 2；test 是全部 10 个原始 task 的 state 1，以及 6 个 LIBERO-Plus 维度各 2 个变体，优先请求 state 1。case IDs 为 1000–1025。Plus 变体会排除旧实验已选中的 variant IDs。
- 仅当 Plus 官方 state list 恰好只有 1 个状态且请求 state 1 时，才使用 state 0；不重新抽变体，也不对其他缺失状态做回退。metadata 和最终 plan 同时记录 requested/actual state、可用状态数及回退原因；summary 列出每个回退 case。原始任务的 selection state 2 与 test state 1 必须实际可用。
- 新输入是在熟悉任务上的新观察，不是 unseen-task 证据。Plus 变体可能共享底层任务，不作为独立变体成功率证据。两组 sampler seeds 是配对重复，不是独立场景。
- Calibration 只引用旧输入 IDs `[0, 2, 4, 6, 10, 12, 14, 16]`、seed index 0；源文件留在旧输入根目录，不复制进新根目录。
- 保持既定 `smooth05_hadamard` transform、模型、权重范围、A4 activation 与 VQ 容量。局部与 action-aware 拟合各做 2 次 index reassignment，之后各做 64 个 fixed-index tuning steps；全局步骤做跨 module 与跨 step 的 projection-coordinate 优化。完整参数以 `experiment_protocol.json` 为准；旧 `protocol.json` 保留根路径和输入准备配置。
- Primary metric 是每个 case 的两组 sampler seeds 平均值：同一 observation、同一 seed 下，与原始 BF16 teacher trajectory 对齐的前 10 步 × 6 维 normalized motor action RMSE。Gripper、32 步误差、局部输出误差、存储大小和拟合时间属于各自单独报告的 secondary metrics。
- 不执行预测 action，不测 success rate，也不作 native-kernel 或端到端 latency 声明。

## PBS 阶段

1. **prepare（CPU，4 CPUs、16 GB、1 小时）**：根据官方 task/variant metadata 和 loader，在 allocation 内生成 26 份固定观察、numeric NPZ、metadata、`plan.json` 与 `prepare_summary.json`。包含 30 步 dummy-action settling，不运行预测策略。完成证据要求 summary 明确报告 26 cases、4 selection、22 test 且旧 Plus variant 排除通过。
2. **preflight（GPU，1 GPU、16 CPUs、110 GB、2 小时）**：小规模数值/编码与运行时检查，不据此选择方法。
3. **full（GPU，1 GPU、16 CPUs、110 GB、8 小时）**：按冻结协议执行完整拟合与固定输入评估。只有 runner 的 `summary.json` 报告 `status=complete`，且 PBS 作业成功、退出码为 0、存在完成标记时，才视为完整阶段完成。
4. **analysis（CPU，4 CPUs、16 GB、15 分钟）**：仅在 full 的完成证据通过后汇总结果。

## 控制与结果状态

`control.py` 的 upload/submit/status/verify/pull 操作仅面向小型控制或 summary 文件，单文件上限 256 KiB，并保留 SSH host-key verification。每次 submit 会先保存 durable intent；若 qsub 返回未知结果，应先用 PBS 历史恢复原 job handle，不能盲目重提。不要取消、重启或覆盖已有结果目录；PBS scheduler 显示完成本身也不等于数值实验完成。

第一次输入准备作业（PBS `F`，exit 1，walltime 5 分 29 秒）因 Plus layout `task_id=1764` 只有 1 个官方初始状态却请求 state 1，未能完成 plan。此次修正只加入上述精确回退；固定 case、variant IDs 与随机种子不变。随后 attempt b 在脚本启动前因 CRLF shebang 失败（25734784.pbs101，exit 126）；已修正为 LF。attempt c 为 25734790.pbs101，已确认 F / exit 0，walltime 4分54秒，26个 inputs 全部完成；两条 layout singleton-state fallback 已记录。GPU preflight 为 25734831.pbs101，已确认提交；其启动及数值结果、full 和 analysis 仍待验证。提交记录、job log 与 summary 应分别核对后再报告实验结论。

## 拟合目标的静态核对

联合目标已完成针对性的源码核对：sampled rows 的 N/n 权重、context/probe 归一化、stored row scale 的 metric 换算，以及跨调用/跨模块 offset 的 coordinate update 与协议一致；此项不能替代数值验证。Preflight 的 projection 仅汇总三个实际拟合模块，其余 611 个 A4 路径模块不纳入该诊断，所以不能据此声称完整模型的 action-error projection 已验证。Full phase 会拟合并汇总全部 614 个目标模块。
