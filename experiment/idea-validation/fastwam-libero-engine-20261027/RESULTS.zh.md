# Fast-WAM C80 reader-KL + inference engine：标准 LIBERO 400 episodes

**391/400，平均成功率97.75%。** 作业`26020749.pbs101 / fwlib400`正常完成：PBS F / Exit_status=0，实际walltime **01:22:44**，artifact exit0、四个worker exit0、PIPELINE_COMPLETE均已确认。四卡单作业申请walltime6小时，提前完成即退出。

| Suite | 成功数 | 成功率 |
|---|---:|---:|
| LIBERO-Spatial | 98/100 | 98.00% |
| LIBERO-Object | 99/100 | 99.00% |
| LIBERO-Goal | 97/100 | 97.00% |
| LIBERO-Long (`libero_10`) | 97/100 | 97.00% |
| Average | **391/400** | **97.75%** |

## 冻结的模型与评测条件

- 原始LIBERO四个suites、每suite10 tasks、每task官方initial states索引0–9，共400个唯一episode slots；环境与sampler seed均为42。每task按官方顺序执行10个trials。
- 基于released Optional-IDM checkpoint，复用`25921795.pbs101 / reader_kl / final8`的C80-R128 MXVQ A+Z+reader-KL。C80为原始LIBERO全40 tasks、每task两条不同demonstration的snapshots；本轮新校准/训练updates为0。
- `idm`模式，10 video steps + 10 action steps，sigma_shift=1，输出32×7 action chunk，每次执行前10步后replan；30 settling steps，普通suite最多400控制步、Long最多700步。沿原有normalizer、图像输入和gripper后处理。
- 使用原始`fastwam-smoke/libero`环境及其assets/initial states，未使用LIBERO-Plus扰动。

## 实际 engine 执行与负载

执行为**C+decoder+buffer**：packed MXVQ常驻 → 按层BF16 decode → cuBLAS BF16 GEMM，启用condition cache、三处CUDA Graph，以及H/U与weight scratch复用。复用原先decoder shape choices，没有重新tuning；不属于native INT4/VQ GEMM。

四个常驻worker各加载一次模型，通过加锁的动态task queue分配任务，Long tasks优先。worker完成episode数分别为**100、100、90、110**，成功数分别为98、98、87、108。模型总加载数为4，每worker加载、安装和anchor检查合计约304秒。

每张卡的完整action/future anchor相对冻结checkpoint最大绝对差异均为**0**。四份执行记录均覆盖614个目标modules、`video_step`/`video_prefill`/`action_step`三个Graph区域；本轮共6415次policy calls，video/action Graph replay数均等于各worker query数×10。GPU利用率和显存每15秒写入远端该job的`job.log`；评测期间采样显存约17161 MiB/卡。

闭环过程的warm policy-call p50为**0.899527秒**；计时不含observation preprocessing、模型加载/Graph capture和环境步进。此处没有同设置BF16效率对照，不能据此计算speedup。

## 结果边界与证据

这是每task前10个官方initial states、固定seed的一轮400-episode结果，不是三seed重复或显著性结论。此次未跑同400-episode BF16，也未跑其他量化baseline，因此不能把相对官方2000-episode未量化结果的差直接归因量化。本轮不保存rollout videos，以减少I/O。

失败涉及7个tasks，共9 episodes：Spatial task3/5各失败1，Object task5失败1，Goal task3失败2、task6失败1，Long task2失败1、task8失败2。逐task汇总保存在summary中。

- [冻结协议](protocol.json)
- [完整原始summary及40-task结果](pulled/26020749.pbs101/summary.json)
- [任务领取记录](pulled/26020749.pbs101/task_queue.json)
- [实际提交协议](pulled/26020749.pbs101/artifacts/protocol.json)
- [实际聚合代码](pulled/26020749.pbs101/artifacts/analysis.py)：拒绝缺失、重复或不同设置的episode slots后才生成summary。
- [最终current PBS及产物状态](monitor_status.json)

本作业的gpt-6-luna/xhigh subagent按30分钟周期监控；终态确认后已停止。
