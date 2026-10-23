# 固定reader的等future NMSE对照

问题：action+latent future的action收益，是否超出总体future误差范数变化能解释的部分？Root亲自实现、提交和分析；subagent仅每约10min只读监控。沿用ponytail和experimental-design，运行前冻结以下协议。

## 对照与完整范围

A是action-only epoch2，B是action+latent epoch2，T是同输入同seed的BF16 teacher。复用旧缓存，不重训、不重新选择checkpoint，不生成新future。Reader固定为原始BF16 video prepare/prefill、ActionDiT及proprio，每个future重新生成K/V。

完整复用原60contexts：8calibration/seed0、4selection cases×2seeds、22test cases×2seeds。沿用原两分片，每片30contexts，paired seeds不拆开。比较A、B和新对照C，共180条逻辑配对记录：120条已有A/B reader输出，60条新C输出。

`C = T + alpha*(A-T)`，只构造future部分，frame0保留原观察值。Alpha按每个context的B NMSE确定，绝不查询action后挑alpha；B误差较大时允许alpha>1，不删减这些contexts。实际注入会转换为BF16，因此先将A/B转换成入口实际dtype，用CPU的float64计算NMSE并搜索alpha，再检查最终BF16 C是否匹配B。Teacher仍用原缓存值作为共同reference，同时报告native与实际注入的NMSE及dtype。

冻结容差：`abs(NMSE_C-NMSE_B) <= max(1e-8,1e-4*NMSE_B)`；BF16舍入后相对teacher误差方向的cosine至少0.999（零误差退化情形单独处理）。最多80次二分，记录实测差异、alpha和方向偏离；不为得到正结果放宽容差或剔除case。所有tensor、数值和IO在真实CPU allocation内。C是使用teacher信息的机制对照，不是可部署方法。

准备阶段的合成自检25851867在真实donor准备和reader查询前发现：BF16舍入的同步跳变可能使纯scalar插值的最近点也超出上述容差（实测相对差0.000100799，门槛0.0001）。该作业F/Exit1，没有启动实际准备。修复不改变容差或cohort：当二分最近点仍超差时，按预定context seed的随机顺序，在最多64个坐标各允许一个BF16 ULP调整，仅接受能缩小NMSE差的调整；不查询action，最终仍须满足同一NMSE与cosine条件。记录每个context的调整数量、seed、NMSE变化及方向偏离。调整后的C是scalar误差方向附近的BF16离散控制，不能描述成数学上完全共线。原代码和修复凭证保留，失败作业不计为完成结果。

## 查询与验证

两个GPU分片分别只查询30个C。每个作业开始/结束对该分片第一个context的A、B各重放一次，共4次端点检查，要求与原固定BF16 reader输出逐数值精确一致；正常无续跑时共68次新reader查询。每次C检查实际conditioning future、原始action noise、frame0 clamp、video DiT调用为0及action DiT调用为10；实际注入后的NMSE还必须通过同一容差。

逐query原子保存、可恢复，只有当前handle已终态F/Exit0/RESUME_READY才续交；查询失败和Q注释不授权重提。严格复用缓存A/B，CPU分析从保存action重算指标核对。先case内平均paired seeds，再等权平均cases；test分母22cases/44contexts，报告calibration和selection，不将60contexts作为60个独立cases。

主指标first10 motor RMSE，另报同目标MSE、full32/late22 motor RMSE、first10 gripper及future NMSE。报告B-C、C-A、B-A的逐case配对差值、均值、中位数、胜负和leave-one-case-out均值范围；后者是敏感性范围而非confidence interval。

## 正向信号与后续条件

只有完整60个BF16匹配和所有严格replay通过，且完整22test cases的B-C first10 motor RMSE满足以下全部条件，才继续后续实验：均值<0，中位数<0，至少12/22cases改善，逐一省略任意一个case后的均值仍<0。这是工程推进门槛，不是统计显著性或独立泛化声明；其余action指标的tradeoff照常报告。

若通过，下一步完整测量两个future时间切片的四组合A1A2、B1A2、A1B2、B1B2，并在BF16及固定action-only reader下保留完整60contexts，直接报告交互。Mask在看新结果前按latent时间索引固定。训练配置工作需先有正向定位证据；不预先展开lambda或容量搜索。若本轮无正向信号，保留负结果并结束本轮，不强行推进。

## 资源与证据边界

CPU准备和分析各20min；GPU每作业最多2h，15s GPU利用率/显存写job.log。Login只用于可信SSH、PBS与每文件不超过256KiB的小控制/汇总；禁止其上模型/tensor读取、计算、下载、重IO、安装或编译。保留host-key verification，不输出credentials，不动用banked reset。Root承担代码与科学分析；监控subagent不提交/取消/续交/改代码，不读tensor或做分析，Q/R没有实质变化对用户静默。

本轮是已见cohort上的fixed-input、teacher-assisted机制诊断；不提供真实未来准确度、闭环成功率、独立泛化、native低位kernel或VQ优于scalar的证据。运行状态见RUN_STATE.json，当前handles见对应*_handle.json。尚未完成的阶段不算已有结果。

## 已完成结果（2026-10-17）

完整60contexts和180条A/B/C逻辑记录已验收，两个GPU作业分别20m21s、19m58s，8次首尾replay的max_abs均为0。完整22test cases的B-C first10motor RMSE平均差为-0.000990490，但中位数差为+0.000042169，仅11/22改善，leave-one-case-out平均差可反转为+0.000594217。冻结推进门槛未通过，因此不启动条件后续实验；监控已停止。详见[中文结果解释](RESULTS.zh.md)及retrieved/results/analysis下完整JSON和CSV。
