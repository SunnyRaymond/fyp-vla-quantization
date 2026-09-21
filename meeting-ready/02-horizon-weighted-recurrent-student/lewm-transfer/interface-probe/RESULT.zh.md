# LeWM PushT 官方 rollout interface probe 结果

状态：`PASS`；PBS job：`24554356.pbs101`；compute host：`x1000c0s4b0n0`；GPU：A100-SXM4-40GB。完整权威 JSON 在 [interface_probe.json](artifacts/24554356.pbs101/interface_probe.json)，作业日志在 [job.log](artifacts/24554356.pbs101/job.log)，退出状态为 `0`。

## 结论

1. `prepare_policy_info` 的 `pixels` 为 `[1, 1, 3, 224, 224]`、`torch.float32`，因此官方 policy reset 的 pixels history 是 `H=1`。其余完整 keys/shapes/dtypes 保存在 JSON 中；`action` 为 `[1,1,2]`。
2. `PlanConfig` 实际字段为 `horizon=5`、`receding_horizon=5`、`history_len=1`、`action_block=5`、`warm_start=true`，`plan_len=25`。冻结 CEM settings 为 `batch_size=1`、`num_samples=300`、`var_scale=1.0`、`n_steps=30`、`topk=30`、`device=cuda`。
3. interface candidate 使用 `[B,S,T,10]=[1,2,5,10]`。每个 10-D token 是 5 个连续 raw 2-D PushT actions 的 packed `[x1,y1,x2,y2,x3,y3,x4,y4,x5,y5]`；这只是 shape/interface probe，不是 30-iteration CEM benchmark。
4. 通过 official `JEPA.get_cost → JEPA.rollout` trace，得到 `H=1`、`T=5`、`n_steps=T-H=4`，`predicted_emb.shape=[1,2,6,192]`，即初始 1 个 latent 后恰好有 `5` 个 future predictions。
5. `model.predict` 的 sequence length `1/2/3` 均接受：输出分别为 `[1,1,192]`、`[1,2,192]`、`[1,3,192]`；action embedding 对应 shape 为 `[1,L,192]`。
6. rolling latest-3 latent + 每步 10-D action 可成功 free-running 生成恰好 5 个 targets：逐步追加 student prediction、无 teacher forcing，stacked shape 为 `[1,5,192]`。由于 policy `H=1`，probe 用最后一个真实 latent/action 做 latest-3 seed padding；这验证 predictor interface，不替代真实 3-latent training context。
7. 冻结 student protocol 的 training target horizon 应为 `5`（`z(t+1)…z(t+5)`）。Stage B wrapper 应接收官方 candidate `[B,S,5,10]`，保留 official criterion/goal/CEM contract，并返回完全一致的 `predicted_emb` shape `[B,S,6,192]`（一般写作 `[B,S,T+1,D]`，本次 `T=5,D=192`）。本次由于官方 `H=1,T=5` 也产生 5 个 future predictions，rolling probe 的 target count 与官方 rollout count 一致；这不等于已完成 Stage B viability 或 closed-loop success。

## GPU telemetry

作业日志按 30 秒记录了 A100 utilization/VRAM：启动时 `0% / 1 MiB`，模型加载后 `0% / 611 MiB`（21:08:59）。

## 证据边界

本 job 只做官方 interface/shape probe（两个 candidate samples、一次 `get_cost` trace），没有运行 frozen `num_samples=300 × n_steps=30` 的 CEM benchmark，也没有加载或回收任何 checkpoint artifact 以外的大文件；不能据此声称 latency、planner viability、task success 或 Stage B gate 通过。
