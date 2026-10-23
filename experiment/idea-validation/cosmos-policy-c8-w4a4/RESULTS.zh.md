# Cosmos Policy C8 W4A4：840 个闭环 episode 的结果

2026-10-16 20:30 SGT 核实完成。模型为 `nvidia/Cosmos-Policy-LIBERO-Predict2-2B`，使用官方 LIBERO direct policy。

Action+latent-aware Microscaling VQ 已在 Cosmos Policy 上完成 C8 校准及真实环境闭环。冻结的 280 个配对 variants 上，VQ 成功 226 个，比本地 SpinQuant-adapted 基线多 5 个，比 BF16 少 7 个。VQ 的目标 weights 编码预算比 SpinQuant 高 3.9155%，处于事先冻结的 5% 容忍范围内；这轮支持框架可迁移及当前 recipe 的闭环表现，尚不能将差异单独归因于 action 或 latent loss。

## 整体成功率与配对结果

| Arm | 成功 / 总数 | 成功率 | 相对 BF16 | 目标 weights effective bpw | 整个 DiT effective bpw |
|---|---:|---:|---:|---:|---:|
| BF16 | 233/280 | 83.21% | — | 16.000000 | 16.000000 |
| SpinQuant W4A4（adapted） | 221/280 | 78.93% | -4.29 pp | 4.125446 | 5.307830 |
| Action+latent-aware Microscaling VQ A4 | 226/280 | 80.71% | -2.50 pp | 4.286979 | 5.453279 |

VQ 相对 SpinQuant 为 **+1.79 percentage points**；相对 BF16 为 **−2.50 percentage points**。这里报告本次冻结 cohort 的观测结果，没有进行统计显著性或总体优越性声明。

| 配对比较：前者相对后者 | 前者独有成功 | 后者独有成功 | 同成或同败 | 配对数 |
|---|---:|---:|---:|---:|
| VQ vs SpinQuant | 14 | 9 | 257 | 280 |
| VQ vs BF16 | 6 | 13 | 261 | 280 |
| SpinQuant vs BF16 | 7 | 19 | 254 | 280 |

## 分 suite 与扰动维度

| Suite（每臂 70） | BF16 | SpinQuant | VQ |
|---|---:|---:|---:|
| libero_10 | 58/70（82.86%） | 60/70（85.71%） | 58/70（82.86%） |
| libero_goal | 54/70（77.14%） | 46/70（65.71%） | 49/70（70.00%） |
| libero_object | 62/70（88.57%） | 61/70（87.14%） | 62/70（88.57%） |
| libero_spatial | 59/70（84.29%） | 54/70（77.14%） | 57/70（81.43%） |

VQ 相对 SpinQuant 的净增来自 Goal +3、Spatial +3、Object +1；LIBERO-10 为 −2。相对 BF16，VQ 的主要缺口在 Goal（少 5 个）和 Spatial（少 2 个），另外两个 suite 成功数相同。

| 扰动维度（每臂 40） | BF16 | SpinQuant | VQ |
|---|---:|---:|---:|
| background_textures | 31/40（77.50%） | 29/40（72.50%） | 28/40（70.00%） |
| camera_viewpoints | 29/40（72.50%） | 26/40（65.00%） | 27/40（67.50%） |
| language_instructions | 38/40（95.00%） | 36/40（90.00%） | 38/40（95.00%） |
| light_conditions | 40/40（100.00%） | 40/40（100.00%） | 40/40（100.00%） |
| objects_layout | 36/40（90.00%） | 34/40（85.00%） | 35/40（87.50%） |
| robot_initial_states | 22/40（55.00%） | 19/40（47.50%） | 20/40（50.00%） |
| sensor_noise | 37/40（92.50%） | 37/40（92.50%） | 38/40（95.00%） |

Language、sensor noise 等维度的局部差异样本数很小；逐项结果用于定位后续诊断，不能据此宣称某种扰动下普遍更优。三个 arm 在 light conditions 均为 40/40。

## 校准、量化范围与公平条件

- 同一 C8：四个原始 LIBERO suites 各两个 snapshots，case IDs 为 `2000, 2019, 2021, 2025, 2051, 2057, 2066, 2070`。只共享 observations 与指令，teacher actions/future latents 由 Cosmos 自身构造；没有使用 Fast-WAM 模型输出。
- SpinQuant 与 VQ 分别提交独立单卡、≤2 小时 allocation，均不依赖最小闭环成功。SpinQuant 完成 200/200 updates；VQ 完成 16/16 updates。最终 checkpoint 预先固定，没有按闭环成功率选模。
- SpinQuant-adapted 使用 learned shared Cayley R128、W4 group-128、per-token symmetric A4，校准目标是 C8×5 timestep packets 的 full joint denoiser-output NMSE。
- VQ weights 使用 d4、两个 K256 BF16 codebooks、B32 uint8 power-of-two exponents、row gains 和 Smooth0.5/H128 transforms；学习 codebooks 与 row gains，固定 indices/exponents/transforms。目标为完整五步 sampler endpoint 的 16×6 motor-action MSE，加 future-image slots 6/7 的 latent NMSE；lambda 固定为 `0.0040433477161343065`，gripper 不参与该训练 loss。
- 两个低比特 arm 的 activation 均为 per-token scalar A4（qmax=7）；activation 上没有在线 VQ 搜索。SpinQuant 训练使用 FP32 STE，部署使用 BF16 operands，不能称为逐位相同的训练/部署运算。
- 同一 280 个目标 Linear、1,761,607,680 个 weights，覆盖 self/cross-attention q/k/v/output 与 MLP。BPW 包含 codes/indices、scales/exponents、codebooks、row gains、transforms；整个 DiT 的统计另计未量化参数，共 1,956,413,440 个 parameters。

两种方法的目标编码 payload 分别为 908,427,264 bytes（SpinQuant）和 943,996,928 bytes（VQ）。目标集合相同，但预算并非完全相等，训练目标与 update 数也不同，因此本轮不是 action loss / latent loss 的单因素消融。C8 与测试 variants 存在 base-task 重合，结果衡量校准后的扰动鲁棒性，不能称为 unseen-task 泛化。

## 实测执行成本与实现限制

所有低比特 weights 均从导出编码解码到 BF16，再执行 fake-A4/BF16 Linear；`native_int4_kernel_used=false`。以下是实际数值实现的完整 action-query wall time，warm query 已排除每个 worker/process/arm 的首次 query。

| Arm | Warm queries | 均值 ms | 中位数 ms | p95 ms | Query peak allocated 最大值 GB |
|---|---:|---:|---:|---:|---:|
| BF16 | 3687 | 593.99 | 589.96 | 623.59 | 5.713 |
| SpinQuant W4A4（adapted） | 3955 | 1056.53 | 1065.56 | 1123.83 | 5.713 |
| Action+latent-aware Microscaling VQ A4 | 3806 | 10042.13 | 10066.66 | 10670.13 | 6.671 |

这里的 GB 为 10^9 bytes，显存列是 PyTorch query peak allocated，不是编码大小或 native 低比特常驻显存。每臂另有 8 次 first query（四个 workers × 两次实际运行）；不同 policy 的 rollout 会产生不同 observation/query 数，所以这不是固定输入的配对 latency benchmark。没有单独测得 denoising-only latency。

**实际接线遗漏：** VQ 的在线 activation 路径是 `quantization._VQTrainState.install → train_banks.A4Input.forward → smooth_vq.transform_input → _right_r`，最后一层按 H128 block 逐段调用 FWHT。虽然加载并检查了 `batched_transform`，它没有接入这条 activation 路径。原始 summary 的 `smooth05_batched_h128_v2` 标签不能作为在线 batching 已生效的证据；本轮应按实际的 Smooth0.5 + block-H128 数值实现解读。

当前 warm query 均值约 BF16 0.594 秒、SpinQuant 1.057 秒、VQ 10.042 秒。上述源码提供了性能瓶颈线索，但没有 profiler 分解其耗时占比；不能把这些数值归为优化后 VQ 或 native kernel 的性能，也不能据编码预算推断实际推理显存节省。后续工程验证需真正接入 batched transform，先做固定输入数值 parity，再单独测量；本轮未因此修改已冻结的 checkpoint 或重跑、替换闭环结果。

运行时冻结源码：[train_banks.py](<D:/Downloads/Final Year Project/experiment/idea-validation/cosmos-policy-c8-w4a4/retrieved/artifacts/25838062.pbs101/helpers/train_banks.py>)；[smooth_vq.py](<D:/Downloads/Final Year Project/experiment/idea-validation/cosmos-policy-c8-w4a4/retrieved/artifacts/25838062.pbs101/helpers/smooth_vq.py>)。

## 作业、修复与结果来源

| 阶段 / 作业 | 结果 | 本轮证据 |
|---|---|---|
| BF16 smoke `25832557.pbs101` | F/Exit0 | 原始 Goal task0：1/1 成功、122 控制 steps；完整 RGB 与 action-only 路径 action 差异 0；有非空视频，未计入正式 280 分母 |
| Spin calibration `25834379.pbs101` | F/Exit0 | 200/200 updates；allocation 起计 3326.1 秒 |
| VQ calibration `25833623.pbs101` | F/Exit0 | 16/16 updates；allocation 起计 4125.6 秒 |
| 原评测 `25834649.pbs101` | F/Exit1，8 秒 | Apptainer 把 CUDA UUID 列表逗号误作 --env 分隔符；首个 Python 启动前失败，0 正式 slots |
| 评测修复一 `25836774.pbs101` | F/Exit1，01:26:32 | 保存 120 个完整 slots（每臂 40）；之后 fog/noise35 使用 NumPy2 已移除的 np.float_，异常 episode 未计分 |
| CPU checks `25837897.pbs101` | F/Exit0 | 23 tests；真实 index270 fog severity5 的 reset/state restore/step 回归和四 UUID 容器传递预检通过 |
| 评测续跑 `25838062.pbs101` | F/Exit0，05:41:58 | 单个四卡 / 24h g3 allocation，复用有效 120，新增 720，最终 840/840 |

UUID 传递通过命令局部 `APPTAINERENV_CUDA_VISIBLE_DEVICES` 修复；NumPy 兼容处理在任务局部将缺失的 `np.float_` 映射到同义 `np.float64`。保持 frozen fog 算法、severity、cohort、seeds 和两个 calibration 产物。最终 840 个结果来自 **前一 allocation 的 120 个有效结果 + 续跑 allocation 的 720 个**，没有将中断的 episode 当成失败，也没有缩小分母。

正式配置为 4 suites × 7 dimensions × 10 variants，每臂 280；state0，固定配对 env/sampler seeds，30 settling steps，400/700 步上限。Cosmos direct policy 使用 5 denoising steps、sigma=[80,4]、224 双相机+proprio、官方图像预处理、16-step action chunk 并执行 16 步，不使用 planning。28 个 bundles 动态分配给四个 persistent GPU workers；GPU 利用率与显存每 15 秒写入 job log。

完成核对：PBS F/Exit0、exit_code.txt=0、PIPELINE_COMPLETE、complete summary 一致；三臂各 280，所有配对 280，每 suite 70、每 dimension 40、28 个 cells 各 10，missing slots 均为空，四个 worker exit codes 均为 0。统计由 GPU allocation 内的 aggregator 生成，本地只做小型 JSON 核对和报告格式化，没有新增模型 forward。

完整结果：[summary.json](<D:/Downloads/Final Year Project/experiment/idea-validation/cosmos-policy-c8-w4a4/retrieved/results/evaluation/summary.json>)。终态：[PBS status](<D:/Downloads/Final Year Project/experiment/idea-validation/cosmos-policy-c8-w4a4/retrieved/artifacts/25838062.pbs101/pbs_status.json>)；[job log 收尾](<D:/Downloads/Final Year Project/experiment/idea-validation/cosmos-policy-c8-w4a4/retrieved/artifacts/25838062.pbs101/job_log_tail.json>)。

本轮结论是：该 framework 可以迁移到 Cosmos Policy 并跑通真实闭环；当前 recipe 的成功率落在 BF16 与 SpinQuant-adapted 之间，优势幅度有限且预算略高。独立验证 action/latent objective 的贡献、以及 native W4A4 的速度/显存收益，仍需要各自的对照实验。

