# FRT B 监控与完成交接

用户已授权 A 通过后做 B，并在长作业时设置监控。**A/B 均已完成，监控已暂停，不要恢复轮询或重跑。** GPU job `64694` COMPLETED（19分34秒），CPU 独立核验 `64696` COMPLETED（14秒）。工程核验通过，但机制门未全部通过；详见 STAGE_B_RESULT.zh.md。

heartbeat automation ID=`frt-b-v100`，已 PAUSED。以下保留为审计与用户明确要求恢复时的参考，不代表还有待执行工作。

## 权限与资源

仅 CCDS-TC1，不连接 ASPIRE2A。CCDS 使用 SLURM，不伪造 PBS_JOBID。所有模型/数组计算、下载、hash、复制 checkpoint、重 I/O 都必须在真实 SLURM compute allocation，经 allocation_guard 校验 SLURM_JOB_ID、实际 TC1N hostname、scontrol RUNNING/UserId/NodeList；GPU 另验分配。Head node 仅轻量提交/状态/小控制文件。每次文件传输与读取不超过 65536 bytes；大数组与模型留 compute。保留 SSH host key verification；凭据只由既有 frt_control.py 从本地读取，不显示/复制/上传。

实时已查 QoS：1 GPU、20 CPUs、64G、MaxJobsPU=2、MaxSubmitPU=2、MaxWall=6h。不并发多个 GPU job。A+B 总实际 GPU allocation 不超过 6 小时，包括失败与启动成本。64687=23s、64688=41s、64690=12s、64691=129s；64694 最多3600s。CPU jobs 不计 GPU-hour。

## 现成命令

本地工作目录 `D:/Downloads/Final Year Project`。控制器路径 `ideaspark_run/closed-loop-quantization/experiments/frt-ccds/frt_control.py`，后文记为 CONTROL；同目录记为 LOCAL。

- `python CONTROL status 64694`：GPU 状态。
- `python CONTROL status 64696`：CPU 核验状态，GPU 未完成时 Dependency 排队是正常的。
- `python CONTROL read artifacts/64694/preflight/preflight_check.json LOCAL/artifacts/64694/preflight_check.json`：2-step 端到端工程预检通过后才存在。
- `python CONTROL read artifacts/64694/progress.json LOCAL/artifacts/64694/progress.json`：正式训练进度。只向 parent 返回 status、completed 的数量、current_method/current_step，不返回整段 identity。
- `python CONTROL read artifacts/64694/summary.json LOCAL/artifacts/64694/summary.json`：正式流程结束的紧凑 summary。
- `python CONTROL read artifacts/64696/verification_b.json LOCAL/artifacts/64696/verification_b.json`：独立原始数组、Wz/metadata、episode gates、checkpoint SHA/hash/hard W4 grid 核验结果。
- 失败时日志为 `artifacts/stage_b_64694.log` 或 `artifacts/verify_b_64696.log`；控制器会拒绝大于64KiB的文件，不放宽限制，可用新获批 compute job 生成 compact error report。

先查原作业再操作，不重复提交。正式进度文件不存在可能仍在 preflight，先查状态与 preflight progress。长训练不需要频繁读取日志。

## 结果与停止规则

固定 manifest_b.json：3 methods × 3 fit seeds，1000 updates/batch2/Adam lr .01，CAL12records、DEV12records各6episodes，λ=4来自A的CALratio裁剪；禁止看 DEV 后调参。fresh 每个 source 使用自己的固定 base=append(Htheta,FP_next)，所有 evaluator 共享该 source context，完整11×11评分。

GPU summary 的 engineering_pass 不能替代独立 CPU 核验。最终读取 64696 的 verification_b.json，核对 manifest.pass、engineering_pass、checkpoint_audit、stage_b.gates 与 mechanism_gate_pass。stage_b 有5个门：clean相对Clean<=1.10，Q0与fresh_union各相对Clean/Random<=.95，episode macro +至少2/3seeds各4/6episodes。

完成后保存 `STAGE_B_RESULT.zh.md`，更新 `RUN_STATUS.zh.md`/`jobs.json`。用中文说明通过/未通过/工程不完整；列出 clean/transport 比较与实际 GPU allocation。即使 B 有信号，也仅为 fixed-action imagined-rollout 的 weight-only numerical emulation 机制筛选；不能声称闭环成功率或 native低比特收益。λ裁剪和6个DEVepisodes的限制要保留。结果欠优化或资源不完整不能判整个 idea 无效。

若 jobs 仍正常运行且无 actionable change，保持安静。若完成、失败或需要用户决定，才通知。完成后暂停本 heartbeat，避免继续轮询。工程失败先保存日志诊断；只有定位明确、可在现有总预算内修复时才做最小修复，复用可验证产物，不能盲重跑/扩预算/进入 C 或 TEST。若 GPU 失败则 dependency CPU 可能被自动取消，应一并记录。

监控子任务使用 gpt-5.6-luna，reasoning effort xhigh，并显式携带上述 cluster/credentials/预算限制。用有界查询后返回；不要在子任务里长时间等候或创建新用户 task。
