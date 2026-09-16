# CEM-Update PTQ 验证进度

更新：2026-09-10 19:05 SGT 左右。本文件是进行中的记录，不是最终实验结论。

## 当前证据

旧数据离线复算完成：26 pools / 8 episodes / 8 mappings，reference self-loss 为零，208 个 elite 检查通过。CEM update 指标可测，且与全候选排序产生不同的配置优劣关系。这只支持继续最小机制筛选，不能证明新方法有效。详见 `PHASE_A_RESULT.zh.md`。

新 FP32 reference 采集完成：newCAL/newDEV 各 4 episodes、8 pools；与旧 40 个目标 fingerprints 无重叠，新目标 indices 42–49，seeds 使用新 namespace。作业 `17025566.pbs101` Exit 0，实际分配 1 A100 × 317 秒，约 0.0881 GPU-hours。保存的 FP32 success 只是采集结果，不是新方法的 closed-loop 结果。

完整配置计时作业 `17042287.pbs101` 已提交，1 A100 / 15 分钟上限。搜索尚未启动，需先根据计时冻结包含 CAL、DEV、two-step replay 和清理的总预算。

## 当前阻塞与恢复顺序

连接已由用户恢复：此前 GlobalProtect VPN 掉线；随后成功查询到计时 job Exit0，实际 GPU 分配 136 秒，累计已确认 453 秒 = 0.1258 GPU-hours。计时与 FP32 reference 校验通过，已冻结两轮搜索含 DEV/replay 的 71 分钟 PBS 上限，搜索 job `17078256.pbs101` 已提交。以下连接阻塞仅保留为历史记录，不再是当前阻塞。

计时 job 最后确认于 18:06:31 SGT 为 Q（gdev overall ngpus limit）。随后 subagent 两次及 parent 一次连接 jump host 均 timeout，因此当前 job 状态未知；不能继续按 Q 报告，也不能推断其成功/失败。没有重复提交、取消或修改队列。已确认的 GPU 用量 0.0881 hours 不包括尚未核实的计时 job。

恢复连接后先通过受限 `remote_control.py status 17042287.pbs101` 核对原 job，若 F 则读取不超过 64 KiB 的 summary/benchmark_summary/allocation 与 qstat。若失败先诊断具体错误。禁止盲目重复提交。计时成功才冻结包含全部 CAL search、DEV replay、模型加载和余量的预算（上限 2 GPU-hours），再提交搜索。随后 CPU-only 独立复核，通过机制门槛才设计并执行 C。

搜索控制文件最终版已上传。独立 CPU verifier 已完成本地 self-test、语法检查与 synthetic compact-workload 兼容性检查；工程判定与研究判定分开，实际完整 raw 验证尚未运行。`verify_stage_b.py` 与 `verify_stage.pbs` 待上传并在搜索完成后提交 CPU-only allocation。

搜索已确认于 19:09:32 SGT 在 x1000c0s4b0n1 进入 R。随后状态查询返回 Authentication failed，已停止重试并请用户检查连接/认证；这是新的连接问题，不能假定为 VPN 掉线。当前没有可读取的新方法最终结果，也没有 closed-loop success 结果。恢复后仍先检查 17078256.pbs101，不重复提交。

## 待检验假设

同一个搜索器、相同起点和 W8 配额下，CEM-Update 比 MeanOnly、ScoreError、Rank 更好地保留 CEM 更新及后续输出 mean。主要检验 sigma 的增量价值；如果与 MeanOnly 得到相同 mapping，本配方停止，不根据结果追加起点或调 loss weight。

通过机制门槛后才进入新的小规模闭环确认。机制门槛不通过意味着本配方不值得继续投入，不能证明所有 update-aware PTQ 都无效。4 个 DEV episodes 只用于资源筛选，不能支持统计意义上的普遍有效性结论。

## 固定实现与约束

以 `PROTOCOL.zh.md` 和 `manifest.json` 为准：CEM5、300 candidates、30 elites、首 MPC point 的 iterations 1/5，末轮仅保护 mean；完整 joint W4/W8 swaps，最多两轮，每方法最多 71 配置。fake quantization 使用 FP32 operators，无 native kernel 加速声明。

独立 CPU verifier 将从 raw scores 和 chain arrays 复算指标、选择与门槛。所有模型、重 I/O 和完整数组复核在真实 PBS compute allocation；login 只提交、查询和读写不超过 64 KiB 的小型文件。旧 RankCal 与 IdeaSpark pipeline 不修改、不重跑。

实际 job 及 GPU 用量以 `jobs.json`、各作业 `qstat_final.txt` 和小型结果 JSON 为依据。排队时间另计，不计入分配 GPU-hours。
