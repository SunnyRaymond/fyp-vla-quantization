# Per-channel加入既有40 visited states诊断

已完成：GPU25868350.pbs101（00:37:34）与CPU25868746.pbs101（00:00:39）均F/Exit0及完整覆盖通过。结果与下一步loss建议见[ANALYSIS.zh.md](results/ANALYSIS.zh.md)。新增200端点、复用600、205 queries；不训练、不改原held-out80，监控已结束。Motor与局部响应总体改善，但gripper仍有漏open事件，建议下一轮先单独补first10 gripper监督。

仅增加已完成per_channel_clip final80；不训练、不采集新rollout、不更换原held-out80。沿用原40个C8-student visited states（奇数case2001–2079），四suite各10，原seed_index0。五输入仍为base、两相机raw brightness±1 uint8、raw EEF xyz沿单位对角方向±5mm；不改变rotation/gripper qpos。

原BF16/clip-only/joint的600端点逐条复用。新增per-channel 200端点，合计800；200份预处理输入元数据需与源结果一致。旧完整image tensor未持久化，因此额外以fresh BF16首个visited-state action精确replay检查冻结输入管线，不声称重新逐像素核对所有旧tensor。

预计205次新model queries：200个新端点，1次BF16 visited replay，3次per-channel源endpoint/uncached-cached/实际infer检查，1次同输入repeat。BF16 fresh query仅作parity，不生成新teacher端点。恢复原C8 rho1.5 weights、614 ratios、1,806,863个per-input-channel log2D；A4前除D，BF16 Wq乘D后一次缓存。所有模型、books、row gains、ratios、D均冻结。

完整报告actual command的前10步开合分歧、confusion、切换位置；额外保留带方向的事件序列、漏/多open/close events。只有有序事件方向相同时才计算step延迟，不把漏掉整个释放当作更小延迟。step0没有上一个chunk状态，因此不推断跨chunk事件。

Motor报告六维normalized与实际command的RMSE/MAE/signed bias/max，另对case内10步平均误差取绝对值后平均，避免正负偏置抵消。局部响应沿用原central与单侧finite difference、teacher-relative error、60D方向cosine/gain及gripper input flips；响应小不自动更好，不用接近零的teacher响应比值作为推进门槛。

一个GPU作业walltime01:59、软截止6840s，总GPU预算2h，不自动续提；每15s写GPU utilization/VRAM。gpt-6-luna/xhigh subagent每10min监控，普通Q/R静默。GPU只生成端点；随后CPU allocation复用相同冻结metrics聚合800端点、240个case×arm×modality响应项，输出全局及四suite。case分母40，steps/variants不是独立cases。

所有模型/数据读取/数值工作只在真实PBS compute allocation；login仅轻量控制，保持host-key verification、不输出credentials、禁止banked reset。标准库contract/metric测试可本地运行；Torch/传感器数值检查在allocation执行。

结果是fixed-input decoded-BF16 VQ+fake A4诊断，既不是per-channel自身完整轨迹，也不包含真实camera viewpoint扰动。两条方向不能代表完整Jacobian或闭环稳定性，不据此唯一归因此前280 episodes净少4个成功。结果出来后再决定loss，不自动训练。
