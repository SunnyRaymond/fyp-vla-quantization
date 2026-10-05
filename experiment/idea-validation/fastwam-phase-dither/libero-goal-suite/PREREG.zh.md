# 完整 LIBERO-Goal suite：冻结评分协议

范围为 Goal 的全部 task IDs 0..9，每 task 固定 initial-state IDs 0..49。四 arms 为 BF16、RTN W4A8、independent dither W4A8、原 pilot 锁定 learned phase W4A8。每 arm 500 episodes，总共 2000 episodes；逐 task 成功数/50，suite 成功数/500。不会用已完成子集代替总分。

使用现有 Optional IDM clean checkpoint、first_frame、BF16 reference、20 inference steps、sigma_shift=1、compile=false、32×7 action chunk、执行前10步后 replan、30 warmup、官方 pinned evaluator 的400 control-step上限与 gripper 转换。Success 为真实 simulator 的 done。四 arms 按 task、initial state、replan index 配对 seeds，各自闭环驱动后续 observations；每个执行作业先完成其全部 BF16 slots 再原地 W4量化，其余 arms 保持原 task 全部50个 initial states 的平衡执行顺序。

保持既有 G128 signed W4 与 dynamic-row A8 配方。Learned table 固定来自25666715.pbs101，不搜索、不训练、不改精度。它是 floating GEMM 的 fake-quant 实验，不能作为 packed-kernel 加速或实际内存节省证据。原机制筛选 NO-GO 不因本次 suite 评分而改写。

Timeout 是有效失败；基础设施/模型异常是未完成，不能算为策略失败或静默补抽。需要修复时保留原 artifact，只补未完成的明确 task/state/arm slot；已完成成功及有效 timeout 均不可重试。汇总拒绝缺失、重复、未完成或范围不符的结果。

原10-task PBS array 中 task0/1已完成，task2继续原作业；按用户2026-10-04指示取消尚未启动的task3..9，并将它们各拆为4个状态分片，共28个短作业。四组 initial-state IDs 为0..12、13..25、26..37、38..49，每组运行全部四arms；完整 protocol identity 和全部50个states的seed/order规则不变，顶层 execution_state_ids 仅标记本作业的执行范围。各分片16 CPU、110GB RAM、1 GPU、2小时上限，通过normal路由到gdev，array最多10个subjobs同时运行，实际并发遵循PBS队列资源限制。模型、数据、rendering、trace 与聚合计算均在真正 PBS allocation；login node只做轻量控制。每15秒将 GPU利用率与显存写入 job.log。

保留作业为25669967[0..2].pbs101；新28个分片及替代CPU-only聚合作业的实际标识以RUN_STATE.json为准。旧聚合作业25670022.pbs101已取消。替代聚合作业（1 CPU、2GB、10分钟）依赖原array及新分片array终止（afterany；PBS不支持单个array子作业依赖），并先逐份检查所需结果的wrapper exit=0与成功marker，接受保留的3份完整task结果和28份完整分片结果，按(task,arm,state)合并并拒绝重复、缺失及protocol不一致；只有核对完整2000个unique episode slots后才能计算逐task与每arm500分母的suite分数。分片结果中的 complete 仅表示该分片完成，不能代表task或suite完成。最终成功唤醒等到CPU聚合完成；任何保留task/新分片/聚合失败则提前唤醒原任务处理。

用户指定独立 gpt-6-luna xhigh 定时监控，每30分钟只读检查。主 goal 确認作业实际运行及监控配置后暂停；监控终态或需处理时通知原会话，原会话负责汇总和收尾。排队/运行无实质变化时安静，不重提、修改或取消作业，不使用 banked reset。
