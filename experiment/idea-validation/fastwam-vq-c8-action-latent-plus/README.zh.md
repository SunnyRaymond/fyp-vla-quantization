# C8 action+latent-aware Microscaling VQ 闭环评测

**已完成**：`25822971.pbs101`于2026-10-15 18:45 SGT完成（RUN_PHASE complete），PBS/artifact exit0、PIPELINE_COMPLETE；实际4卡walltime 08:02:26。560/560新episodes和1680既有reference slots齐全，四worker各140/140且rc0，614-module source/action replay max_abs均为0，batched H128/A4 bitwise检查通过。rho0.1=201/280（71.79%），rho1.5=206/280（73.57%），既有Micro204、SpinQuant203、BF16216。

完整结果与calibration比较边界见[RESULTS.zh.md](results/RESULTS.zh.md)，机器可读聚合见[summary.json](results/summary.json)，完成证据见[completion_evidence.json](completion_evidence.json)。现有三十分钟Luna heartbeat已停用（PAUSED）。既有Micro不是同C8未训练bank，rho两组epoch也不同；结果用于完整配置比较。

最新执行安排见 `execution_plan.json`：rho0.1、rho1.5 各提交一个单卡 1h59m 训练任务，目标队列 gdev；两组训练完成后才提交单一四卡24h g3闭环评测。原作业 `25814264.pbs101` 因训练收尾接口错误退出，修复后的合并任务 `25821241.pbs101` 已按用户新安排在 Q 状态撤回。完成的 C8 reference、两组各8个 updates 和 checkpoint 保留；详细原因见 `FAILURE_25814264.zh.md`。

已提交单卡任务：rho0.1=`25821660.pbs101`（fwalt15r01），rho1.5=`25821658.pbs101`（fwalt15r15），实际均已路由gdev、1GPU、1h59m。2026-10-15 10:08 SGT核对rho0.1仍R，已完成12/16个updates。rho1.5训练与deployment导出完成，但后置metadata validator漏了collections导入，GPU作业因此exit1；修复后提交CPU-only补验证`25822474.pbs101`（1CPU/4GB/10min），不重训、不加载模型、不做新forward。旧GPU失败记录保留，详情见`VALIDATION_REPAIR_25821658.zh.md`。最新每臂状态见 `training_monitor_state.json`，句柄见 `training_handles.json`。正式四卡评测已提交：`25822971.pbs101`（fwale1015），2026-10-15 10:32 SGT核对为Q/g3、4GPU/24h；rho0.1已完整达到16updates/2epochs，rho1.5补验证已通过，两臂reference action parity max_abs均为0。

## 固定比较

两组各运行冻结的 280 个 LIBERO-Plus variants，总计 560 个 episodes。保持既有四 suites、七扰动维度、各 cell 十个 variants，以及配对环境和 sampler seeds；同一任务×扰动只有一条选中 variant。

| Arm | rho | 固定 epoch | Updates | Workers |
|---|---:|---:|---:|---|
| rho0.1 | 0.1 | 2 | 16 | 0、1，各 140 episodes |
| rho1.5 | 1.5 | 1 | 8 | 2、3，各 140 episodes |

Epoch 在闭环前固定，沿用历史第一轮各 rho 的最佳 epoch。两组完整配置同时改变 rho 和 epoch，因此结果不能单独归因于 rho。

Calibration 使用原始 LIBERO 四 suite 各两个 C8 contexts：`2000, 2019, 2021, 2025, 2051, 2057, 2066, 2070`，均取 seed index 0。初始 bank 复用已完成 C8 作业 `25789685.pbs101` 的 614 个模块。旧 latent search 的八个 cases 包含 Spatial original 与 Plus，既不是这里的 C8，旧训练 checkpoint、teacher cache 和归一化常数也没有被替代使用。

## 分阶段安排

1. C8 reference 已完整准备，并完成16次 actual infer 重放，max_abs 均为0；共享 `normalization_c=0.028303017887797623`，直接复用。
2. 两个单卡任务各自检查当前GPU上的 H128 前向、反向和 A4 operands 一致性；仅批量实现，不逐 block 调用 FWHT。
3. rho0.1 从 checkpoint 的8个 updates补到16；rho1.5 的8个 updates已经完成，仅做修复后的参考动作重放、parity 与 deployment 导出。loss、C8 inputs、Adam reset、固定epoch和seed都不改。
4. 每个arm须有权威PBS终态/exit0、`TRAINING_COMPLETE`及exact C8 training receipt，才允许提交四卡评测。仅后置metadata校验失败且训练产物完整的GPU作业，可以由独立CPU补验证作业建立成功记录；提交gate仍核对原GPU作业与新验证作业的对应关系。四卡脚本不训练、不生成teacher，验证两组已训练产物后启动四workers，各140个episodes，复用六组既有baseline的1680个完成slots。
5. 完整结果在四卡allocation内聚合；四worker完整覆盖、训练产物和参考动作parity成立后才能写 `PIPELINE_COMPLETE`。

本次分阶段恢复预计加载模型六次：两个单卡训练各一次、四卡评测每worker一次；不重复已完成的reference forwards和training updates。每个allocation独立从真实启动时刻计算软截止：训练任务使用6840秒，四卡评测85500秒。训练逐update原子保存checkpoint，评测逐完整episode持久化。

GPU utilization 和 memory usage 每十五秒写入该作业 `job.log`。Login node 只用于轻量提交、状态查询和小型控制文件；准备、训练、数值检查与结果聚合全部在真实 PBS allocation 内完成。

## 结果边界

本轮是 decoded BF16 VQ weights + 动态 A4 operands、KV BF16 的闭环成功率比较，不是 native INT4 kernel 或压缩编码等于实际 GPU 显存的证据。具体 source、seeds、inference 与 loss 设置见 `protocol.json`。

本地18项控制逻辑测试、metadata与sampler接口测试、脚本及PBS内嵌gate语法检查已通过。已完成的GPU检查有218个H128前向、反向与A4 bitwise一致性fixtures；闭环结果仍须等待正式评测。

## 监控状态

训练阶段按用户最新要求，在原对话使用 gpt-6-luna / xhigh subagent，以十分钟间隔查询两个单卡 handles。Q/R未变保持安静；查询失败视为unknown，不自动取消或重交。仅在两组完整结束、真实失败或合法断点需要续跑时通知原任务。

独立 Luna 监控对话 `01a13d27-2ad8-7683-b79b-e215a2eb03f0` 已建立，实际turn配置核对为gpt-6-luna/xhigh。它的首次查询与向原任务 `01a11a6e-7199-7d72-82ec-ca3b3aaef7e7` 的通知通路自检均成功；现有三十分钟heartbeat `c8-action-latent-pbs-monitor` 已更新为正式评测handle `25822971.pbs101` 并启用（ACTIVE）。训练阶段的本地Luna subagent已完成，两臂完整产物通过后才启用本定时监控。

闭环完整完成需PBS terminal/exit0、artifact exit0、PIPELINE_COMPLETE、summary完整560新slots和两arm各280、六组reference1680、C8训练来源和4worker receipts都成立。普通Q/R不通知，完成、失败、断点或四worker首次真实启动验证时只通知一次，并由原任务恢复和继续原goal。

同原对话heartbeat无法单独指定Luna的旧限制已由用户澄清的独立监控对话方式解决；不再等待之前的模型选项。正式闭环已完整结束；Luna终态通知成功唤醒原任务，主任务已完成aggregate与worker级验收，定时监控已停用。

远端根目录：`/scratch/users/ntu/yguo017/fastwam-vq-c8-action-latent-plus-20261015`。
