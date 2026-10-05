# LIBERO-Goal 完整 suite 结果

2026-10-04 已完成并核实：10 tasks × 50 initial states × 4 arms，共 2000 个有效完成的 episodes，每 arm 的分母为 500。失败的有效 timeout 保留在分母中；聚合拒绝缺失、重复或基础设施异常导致的未完成记录。

| 配置 | 成功数 | Suite 成功率 | 相对 BF16 |
|---|---:|---:|---:|
| BF16 | 487/500 | 97.4% | — |
| RTN W4A8 | 486/500 | 97.2% | −0.2 个百分点 |
| Independent dither W4A8 | 483/500 | 96.6% | −0.8 个百分点 |
| Locked learned phase W4A8 | 481/500 | 96.2% | −1.2 个百分点 |

## 逐 task 成功数（每格分母 50）

| Task ID | Task | BF16 | RTN | Independent | Learned |
|---|---|---:|---:|---:|---:|
| 0 | open the middle drawer of the cabinet | 50 | 50 | 50 | 50 |
| 1 | put the bowl on the stove | 48 | 50 | 48 | 49 |
| 2 | put the wine bottle on top of the cabinet | 48 | 50 | 50 | 49 |
| 3 | open the top drawer and put the bowl inside | 44 | 48 | 45 | 44 |
| 4 | put the bowl on top of the cabinet | 50 | 49 | 50 | 49 |
| 5 | push the plate to the front of the stove | 50 | 50 | 50 | 50 |
| 6 | put the cream cheese in the bowl | 50 | 49 | 49 | 50 |
| 7 | turn on the stove | 50 | 50 | 50 | 50 |
| 8 | put the bowl on the plate | 50 | 48 | 49 | 48 |
| 9 | put the wine bottle on the rack | 47 | 42 | 42 | 42 |

## 配对比较与解读

每个比较配对同一 task 与 initial state，共 500 对。Learned 对 RTN：赢 7、负 12、平 481，净少 5 个成功 episodes（−1.0 个百分点）。Learned 对 independent：赢 7、负 9、平 484，净少 2 个（−0.4 个百分点）。本次锁定 learned phase 未显示优于这两个量化基线的闭环成功率，原配方的 NO-GO 保持不变；这些差值不构成统计显著性结论，也不能否定所有 phase 方法。

RTN 总分接近 BF16，但同一初始状态的成功集合存在变化：BF16 独有成功 10 对、RTN 独有成功 9 对，因此总分接近不能解释为逐 episode 等价。优先分析 task 9 的现有失败 traces：BF16 94%，三种量化配置均 84%，这是本次共同的量化退化点。Task 3 可用于分析 RTN 与 learned 的差别（96% 对 88%）。下一步建议先定位失败前的 action 分歧与抓取/放置阶段，再决定改进；本次收尾没有启动新实验。

## 协议与证据

使用 Optional IDM clean checkpoint、first_frame、20 inference steps、32×7 action chunk、执行前 10 步后 replan、30 warmup、400 control-step 上限，以及冻结的 task/state/replan 配对 seeds。量化采用既有 G128 signed W4 与 dynamic-row A8；learned table 固定来自 25666715.pbs101。结果是 floating GEMM fake-quant 的数值/闭环证据，不提供 native packed-kernel 加速或实际内存节省证据；也不作为论文环境/seed 的严格复现。

保留的任务 25669967[0..2].pbs101 与 28 个 gdev 分片 25675739[0..27].pbs101 已成功结束，PBS Exit_status=0。CPU 聚合 25675742.pbs101 在真实 allocation（host x1002c5s6b0n1）于 2026-10-04 13:50:38 SGT 完成，PBS Exit_status=0、wrapper exit=0、PIPELINE_COMPLETE 均成立。聚合实际读取 31 份来源，验证协议一致、coverage/reference gates、trace 索引/文件存在，以及完整 2000 个唯一完成 slots 后评分。gdev 分片均在 2 小时上限内结束。

PBS Stageout_status=1，因此标准 stageout 不能称为成功；已从 scratch 直接取得汇总结果和日志，计算与评分的成功由上述 PBS/wrapper/完整结果证据确认。

- [机器可读汇总](aggregate_result.json)
- [CPU 聚合日志](artifacts/25675742.pbs101/job.log)
- [实际执行的聚合代码快照](artifacts/25675742.pbs101/aggregate.py)
- [28 个分片的 PBS 终态](artifacts/25675742.pbs101/shard_pbs_terminal.txt)
- [冻结协议](PREREG.zh.md)

收尾核查时，原定时监控 libero-goal-suite-6luna-monitor 在 app 中已不存在（更新接口返回不存在，本地也无相应配置）；无需重建监控。
