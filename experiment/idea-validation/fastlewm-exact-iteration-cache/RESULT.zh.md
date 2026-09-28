# Fast-LeWM 迭代内精确 encoder cache：PASS

2026-09-27。冻结的8个B1 fixed contexts均通过 fidelity、latency 和 memory 三个门。每context做2次warmup、5次交替顺序配对计时，另做一对未计时的完整30轮trace。

| 原manifest task index | native时间中位数，秒 | cache时间中位数，秒 | matched降幅中位数 | bitwise trace |
|---:|---:|---:|---:|---|
| 0 | 0.4917 | 0.2107 | 57.110% | PASS |
| 1 | 0.4898 | 0.2102 | 57.123% | PASS |
| 2 | 0.4900 | 0.2100 | 57.162% | PASS |
| 3 | 0.4904 | 0.2104 | 57.141% | PASS |
| 4 | 0.4945 | 0.2122 | 57.114% | PASS |
| 5 | 0.4951 | 0.2109 | 57.317% | PASS |
| 6 | 0.4931 | 0.2121 | 56.936% | PASS |
| 7 | 0.4966 | 0.2130 | 57.147% | PASS |

整体指标为8个context各自5个matched rep耗时降幅中位数，再取context中位数：**57.132%**，约2.33×同输入planner speedup；超过冻结10%门。最大context peak allocated memory比为**1.00**，通过1.10门。所有计时solve最终输出finite；每条路径30次cost、9000 candidate scores。每context逐轮candidate action vectors、cost vectors、最终actions及Python float cost列表均逐字节相同，包括dtype、shape与signed zero，没有使用数值容差。完整配置与停止规则见 [FREEZE.json](FREEZE.json)。

## 机制与适用范围

每次native cost先编码goal，再编码current observation；同一B1 solve的输入不随30轮CEM变化。只在`model.encode`的action-free路径保存两份emb，首次2次编码后命中58次；native `get_cost`、rollout、action encoder、predictor、criterion和CEM均未改动。每solve重新构建缓存，首次构建计入时间。GRU执行路径、beta0、300/top30/30、H1/block25/history1保持官方设置。

这是有用的工程基线，不是新 research gap。历史LeWM已有exact iteration cache；本实验只验证其在当前Fast-LeWM路径上的有限收益。8项任务已暴露于baseline/main，其中包含原失败项5；fixed input上的trace通过，不意味着任务5闭环成功。主实验的实际B50尚未安装此缓存；同一B50 solve会依次处理不同environments，这个B1两槽实现不能直接跨environment复用。没有闭环评估或部署收益证明，不能将2.33×与主实验约5×直接相乘。

## 执行证据

GPU作业`25570885.pbs101`：`F / Exit_status=0 / Stageout_status=1`、script `EXIT_STATUS=0`，walltime`00:01:05`，host`x1000c0s0b0n0`，A100-SXM4-40GB。GPU利用率/显存每5秒写入job.log；log中显存峰值约627MiB。全部模型/数据读取、推理与统计在真实PBS compute allocation完成；没有下载、hash、共享环境修改或dataset/model取回。

源码/输入沿用已完成主实验`25569974.pbs101`和准备`25569830.pbs101`：Fast commit `de3e9dac539f5bbe6ff1656a2fb00938d62a3c7d`，checkpoint revision `f95379fe193c8bfc6a59c9d8437d5052bd72ff71`，`stable-worldmodel==0.0.6`。bootstrap复用主runner，在首次native solve前截获已预处理输入，随后按原manifest前8行分别配置B1。配对使用fresh solver/inputs和相同native generator/global RNG。

完整机器结果：[result.json](artifacts/25570885.pbs101/result.json)。本实验已收尾，未自动扩展B50/closed-loop或新预算网格。

实验设计流程参考：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. https://doi.org/10.48550/arXiv.2609.00065。流程引用不为本方法的有效性或 novelty 背书。
