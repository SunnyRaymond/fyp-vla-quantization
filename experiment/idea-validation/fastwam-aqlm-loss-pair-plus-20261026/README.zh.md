# AQLM 两臂 280 × 2 闭环

2026-10-27 当前状态：已按用户指令停止闭环，reconstruction 完成280、ours完成245，保留525条记录及全部checkpoints。CPU只读分析已完成，新增训练及模型forward均为0；不自动恢复剩余35条。结果与诊断见 [STOPPED_ANALYSIS.zh.md](STOPPED_ANALYSIS.zh.md)。下文保留原冻结协议。

沿用已验证 loss-pair-plus 的冻结280-row manifest：4 suites × 7 perturbation dimensions × 10 variants/cell。两臂使用同一 state0、environment seed=800000000+index×1000、sampler seed=900000000+index×1000+replan。只改变训练 loss，不重新选择任务、seed 或checkpoint。

闭环提交单一 normal→g3 PBS allocation：4 GPUs、32 CPUs、220GB、16小时。worker0/1运行 reconstruction 的偶/奇140 episodes，worker2/3运行 ours 的同样两片；四个独立GPU进程并发，启动间隔20秒。每15秒把GPU利用率及显存写入作业日志。每20分钟由subagent轻量监控，普通进度主对话静默。

源阶段为相邻 fastwam-aqlm-loss-pair-20261026。其 protocol.closed_loop=false 只描述 fixed-input 阶段；本协议明确授权560个闭环episodes。闭环控制器先通过 bind-source 核实源 pair 的实际PBS终态、PHASE_COMPLETE、两臂final8/exposures80、C80及80个held-out records，再绑定真实job/artifact路径；未绑定或不完整时禁止上传提交。

源worker在既有final8 calibration case2000/seed0 query保存完整action/future anchor，不增加正常评估forward。每个闭环worker先作两次student query：完整anchor精确replay，以及formal infer action误差≤1e-6；不新增训练或BF16 teacher forward。源anchor的JSON/PT、arm、final8、计划行及seed必须一致。

运行使用614个目标Linear、AQLM g8/M4/K256、原有Smooth0.5/H128/冻结R0、dynamic A4和BF16 KV。codebooks/scales/indices、底模及rotations全部冻结。执行方式为decoded BF16 dense AQLM，所有耗时/显存结果仅描述此实现，不声称native低比特kernel收益。

每个worker只恢复与本协议及本臂部署绑定的连续episode前缀，拒绝重复、错误variant/seed和不匹配部署。AQLM两臂必须产生各自的新episodes，不复用此前VQ方法的episode结果。中断时保留已完成记录，只有精确560覆盖及四worker各140完整才能完成。

结果为 results/summary.json 和 results/report.zh.md：overall、suite、dimension、cell成功率，以及ours相对reconstruction的逐episode胜/负/平，配对分母280。另报first/subsequent query的latency和显存。外部reference episode slots固定为0。

所有模型加载、源文件staging、episode运行及聚合均须在实际获批PBS compute allocation内执行；login节点仅连接、提交、状态查询和小型控制文件操作。SSH host-key verification保持开启，不输出credentials。
