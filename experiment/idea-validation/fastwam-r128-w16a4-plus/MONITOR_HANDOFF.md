# R128 闭环监控交接

唯一作业 `25910102.pbs101` / `r128cl0001`，remote root `/scratch/users/ntu/yguo017/fastwam-r128-w16a4-plus-20261022`，owner thread `01a11a6e-7199-7d72-82ec-ca3b3aaef7e7`。不重新提交、不改变当前资源；四GPU、24h，server将32CPU/220GB展开64CPU/440GB属预期。

H_W16A4/C80_R128_W16A4各280，共560，4workers各140；13组旧reference共3640。没有新训练。原BF16 weights/dynamic A4/BF16KV，无VQ。四setup已通过，7源metric errors、完整uncached/cached action/future差及actual infer action差均为0。启动成功已通知用户，不重复报告。

使用 gpt-6-luna / xhigh subagent 每1800秒监控。当前目录 `control.py` 的 `direct_connect()` / `pull()` 提供保留host-key verification的轻量连接；`C:/Users/Raymond/miniconda3/python.exe -X utf8 -B` 可运行。import时仅先放本目录于sys.path。finally关闭transport/jump。worker keys为 `progress, summary, setup, runtime, failure, preparation, rc`。

查询仅保存compact `monitor_status.json`，含checked_utc/job_id/scheduler/phase/worker episode数、setup通过状态、completion gate和pull快照路径。避免打印完整protocol、indices或metadata。Q/R正常变化保持安静。查询失败仅表示不确定，保留同handle重查，不作为重提依据。只有明确失败、终态或需处理时通知root。连续两次完全无进展可轻量tail指定worker日志8KiB，不无依据宣告故障。

完成要求 `control.complete_gate=True`：PBS F/X/Exit0、artifact exit0/PIPELINE_COMPLETE/RUN_PHASE complete、summary560、两arm各280、refs3640、四workers140。若F0但gate未过或为合法checkpoint，也通知root审阅。由root做最终fresh verification、收取summary/report并更新goal；monitor只通知root后结束，不能自行resume/cancel/修改goal/source。

禁止login compute、Torch/model/推理/benchmark/真实episode聚合/heavy I/O/download/hash/大传输/解压/安装；只允许轻量PBS控制、小JSON和log tail。保留hostkey验证，不输出credentials，严禁banked reset。不建新chat、automation或额外agent，不给其他chat发消息。用户已明确授权本monitor向owner root报告监控事件。
