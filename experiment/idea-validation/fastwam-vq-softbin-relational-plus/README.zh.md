# soft-bin / soft-bin + relational KL 配对闭环评测

复用源对话 `01a147a4-f04e-7243-914a-ed8f7801aad3` 已完成的两个 final80 checkpoint。每组评测原冻结的 280 个 LIBERO-Plus episodes：4 suites × 7 扰动维度 × 10 variants，共 560 个新 episodes。

| 配置 | checkpoint 来源 | 原训练作业 | 保存完整 action/future 的作业 |
|---|---|---|---|
| softbin | `fastwam-vq-dual-kl-20261019/results/dual_kl/deployment.pt` | `25878625.pbs101` | `25884637.pbs101` |
| relational | `fastwam-vq-relational-kl-20261020/results/relational/deployment.pt` | `25883750.pbs101` | `25883750.pbs101` |

两组都从原始 C8 rho1.5、ratio=1、D=1 初始化独立完成 80 updates；relational 并非接着 softbin 训练。两者只学习 614 个 clipping ratios 和 1,806,863 个 FP32 per-input-channel log2D；codebooks、indices、row gains 和原始 B32 exponents 保持冻结。本次新增训练 updates 为 0。

## 比较口径

主要指标为 closed-loop success rate；报告总体、suite、扰动维度和 suite × dimension，以及逐 episode 配对胜负。两组使用同一 manifest、environment/sampler seeds、initial state、30 settling steps、10 video + 10 action denoising steps、32-step action horizon 和每次执行前 10 个 actions；常规 cap 400，libero_10 cap 700。

复用 13 个已完成配置的 3,640 条 reference slots，包括 BF16、SpinQuant、microscaling VQ、C8 rho0.1/rho1.5、C80 clip-only/joint，以及同为 final80 的 per-channel MSE 配置。reference 不新增 rollout；后者用于帮助判断 loss 改动的影响。

运行方式为 decoded BF16 VQ weights + dynamic signed A4 + BF16 KV，Smooth0.5/H128 使用既有批量实现。延迟和显存只代表本实现，不用于声称 native W4A4 kernel 加速。源 fixed-input RMSE/KL 与本次 closed-loop 成功率分别报告。

## 作业安排

单一 PBS allocation：normal，4 GPU，32 CPU，220 GB，24h。worker 0/1 为 softbin，2/3 为 relational；各配置按 manifest index 奇偶分片，每个 worker 140 episodes。启动错开 20 秒；每 15 秒把 GPU 利用率和显存写入 job.log。

每个 worker 加载基础模型一次，共 4 次。启动时进行三次非正式 action query：源完整 action/future 重放、缓存 compensated weights 后的完整 query、正式 infer 路径检查，共 12 次启动 query。未通过已保存源输出或 uncached/cached parity 的 worker 不进入正式评测。两组无需重训、重新拟合量化参数或新的 teacher forward。

有效完成的 episode 原子追加保存；重启只接受验证通过的同协议连续前缀。85500 秒软截止时保存可恢复状态，尚未达到 560 条则不标记实验完成。完成条件同时要求 PBS/artifact exit 0、PIPELINE_COMPLETE、四 worker 各 140、两组各 280、3,640 reference slots 和 startup checks 通过。

所有模型加载、张量检查、评测和真实 reference 聚合均在 PBS compute allocation 中执行。控制器只通过保留 host-key verification 的连接进行小文件操作、提交和状态查询。

## 入口

`control.py upload` 上传小型源文件，`control.py submit` 幂等提交首个作业；`status`、`logs`、`pull` 用于轻量查询。仅在合法软截止 checkpoint 后可调用 `resume`。PBS job id 保存于 `RUN_STATE.json` 和远端 control/handles.json。

提交后由 gpt-6-luna / xhigh subagent 每 30 分钟检查。Q/R 没有实质变化时保持安静；完成、失败或需要处理时通知原任务。
