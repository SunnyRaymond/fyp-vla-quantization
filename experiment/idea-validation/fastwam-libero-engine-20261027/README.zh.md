# 标准 LIBERO：冻结 C80 reader-KL + C+decoder+buffer inference engine

用户范围：40 tasks × 10 episodes，共 400；单一四卡、walltime 6 小时作业；不重新校准；gpt-6-luna/xhigh subagent 每 30 分钟监控。

2026-10-27 14:10（Asia/Singapore）已提交 `26020749.pbs101 / fwlib400`。14:11 的 current PBS 查询为 Q、queue g3；调度器将四卡请求自动调整为64 CPUs/440 GB，walltime仍为6小时。此处是提交时记录，实时状态以 controller 查询为准；不代表已完成400 episodes。

**已完成。** 2026-10-27 15:45 current PBS 查询为 F / Exit_status=0，实际 walltime 01:22:44；总进程和四个 worker exit均为0，PIPELINE_COMPLETE已生成。全部40 tasks×10 episodes覆盖完成，成功391/400（97.75%）。四份完整action/future anchor差异均为0。30分钟subagent监控已停止。见[完整结果](RESULTS.zh.md)和[原始summary](pulled/26020749.pbs101/summary.json)。

复用 Q0：25921795.pbs101 / reader_kl / final8，基于 released Optional-IDM checkpoint，10 video + 10 action denoising steps，sigma_shift=1，action chunk32，每次执行10步。C80 的80个 calibration snapshots 来自原始 LIBERO 全40 tasks，各取两条不同 demonstrations；此次 checkpoint、normalizer 和量化参数全部冻结。

评测使用 `fastwam-smoke/libero` 的原始 LIBERO、每 task 官方 initial states 索引0–9、seed42；四 suite 各100 episodes。不会沿用 LIBERO-Plus 环境。四个常驻 worker 各加载一次模型，通过原子任务队列动态领取 task；优先启动 Long tasks，使其他任务能填补不同 episode 时长造成的空闲。

Engine 复用已验证的 packed MXVQ → 在线 BF16 decode → cuBLAS BF16 GEMM、condition cache、CUDA Graph、共享 H/U/weight scratch 和 decoder shape choices。不重做 decoder tuning；每张卡启动时对原始完整 action/future anchor 作一次精确比较。该执行路径不属于 native INT4 GEMM。

GPU 利用率和显存每15秒写入该作业 job.log。逐 episode 原子保存结果；成功需覆盖全部400个唯一 slots、四个 worker 成功、四份 anchor 全通过及 PBS/产物正常结束。为减少 I/O，本轮不保存 rollout videos。

```powershell
& '.\nscc-access\.venv\Scripts\python.exe' -Xutf8 -B '.\experiment\idea-validation\fastwam-libero-engine-20261027\control.py' status
```
