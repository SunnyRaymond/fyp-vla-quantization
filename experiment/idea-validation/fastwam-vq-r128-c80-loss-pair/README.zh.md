# Fast-WAM R128 C80 loss-pair calibration

本目录负责单个 PBS calibration job：固定 C80、R128、MX-VQ 起点，比较 `az` 与 `reader_kl`。job 申请 `normal`、32 CPUs、220 GB、2 GPUs、12 小时；脚本只在 PBS 报告 `R`、实际路由为 `g2`、nodefile 与当前非 login hostname 相符后才开始准备运行环境或执行训练。运行期间每 15 秒把 GPU utilization 和显存记录写入 job log。

两臂必须共用相同冻结起点、C80 batches 和预算；每臂 8 次更新、80 次 exposures、614 个目标 Linears。成功 receipt 还须确认 rotation 冻结、初始与最终 forward parity 不超过 `1e-6`、有 final anchor、零新增 teacher queries，以及 `az` 的 `beta=0`、`reader_kl` 的 `beta>0`。本 controller 只有一个 calibration handle；下游 closed-loop 阶段由独立目录和 controller 管理。

## PBS 控制

在 PowerShell 中进入本目录，按顺序运行：

```powershell
& 'C:\Users\Raymond\miniconda3\python.exe' -Xutf8 -B control.py upload
& 'C:\Users\Raymond\miniconda3\python.exe' -Xutf8 -B control.py submit
```

`submit` 会在 qsub 前保存 durable intent。若 qsub 返回结果不明，重复执行只尝试按固定 job name `r128lp0001` 从 PBS history 恢复，不会发出第二个 job。intent 无法解析时应保留现场等待核查，不要手动删除状态文件后重提。

监控和完成后的核验、拉取命令：

```powershell
& 'C:\Users\Raymond\miniconda3\python.exe' -Xutf8 -B control.py status
& 'C:\Users\Raymond\miniconda3\python.exe' -Xutf8 -B control.py verify
& 'C:\Users\Raymond\miniconda3\python.exe' -Xutf8 -B control.py pull
```

`status` 是小文件状态快照；PBS 查询失败会标为 inconclusive。`verify` 只有在 PBS 终态 `F/X`、exit 0、g2 资源匹配、artifact 有 `PHASE_COMPLETE`，并且 root summary 与两份实际 arm summary 的 job ID、指标契约和内容均相符时才返回成功。`pull` 只拉取每份不超过 256 KiB 的 JSON receipts，保存到 `pulled/<PBS_JOBID>/`；训练 checkpoint 和其他张量文件不拉回本机。`logs` 仅查看 job log 最后 8 KiB：

长时间监控按 protocol 的 20 分钟间隔查询；队列或运行状态没有变化时保持安静，查询失败视为 inconclusive。

```powershell
& 'C:\Users\Raymond\miniconda3\python.exe' -Xutf8 -B control.py logs
```

Controller 只上传协议、PBS 脚本、环境脚本、worker 和 finalizer 等小型源文件；source banks、模型权重与其他大文件都由 compute allocation 中的作业读取。不要在 login node 上复制、校验或解压这些数据。

## 本地机械自检

```powershell
& 'C:\Users\Raymond\miniconda3\python.exe' -Xutf8 -B -m unittest -v test_control.py
```

这些 mock checks 只验证资源/receipt gate、有限大小读取和提交命令构造，不会连接 ASPIRE2A、提交 PBS job 或读取凭据。
