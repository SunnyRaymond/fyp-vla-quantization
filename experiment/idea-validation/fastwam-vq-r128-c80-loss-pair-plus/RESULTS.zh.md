# C80-R128 microscaling VQ：A+Z 与 A+Z+reader KL 闭环对比

完成作业 `25925636.pbs101`：PBS F / Exit_status=0，artifact exit 0，PIPELINE_COMPLETE，实际 walltime `09:56:48`。四个 worker 各 140 episodes、rc=0；新增 560 episodes 和复用 4200 reference slots 均已通过完成门。

两组共用同一 R0、VQ0、C80 数据和八批十个 contexts；只改变 reader KL 项。R0 固定，VQ 各训练 8 updates / 80 exposures。校准作业 `25921795.pbs101`，KL β=0.006083668576461056。所有 worker 的完整 action/future anchor 和正式 infer action 差异均为 0。

| 配置 | 成功数 | 成功率 |
|---|---:|---:|
| C80-R128 MXVQ4 A+Z | 204/280 | 72.86% |
| C80-R128 MXVQ4 A+Z+reader KL | 208/280 | 74.29% |
| C80-R128-W16A4 reference | 203/280 | 72.50% |
| H-W16A4 reference | 204/280 | 72.86% |
| BF16 reference | 216/280 | 77.14% |
| SpinQuant adapted W4A4 reference | 203/280 | 72.50% |
| C8 A+Z rho1.5 reference | 206/280 | 73.57% |
| C80 clip-only reference | 207/280 | 73.93% |

KL 相对 A+Z：14 胜、10 负、256 持平，净 +4 个成功（+1.43 个百分点）。相对未做 VQ 的 C80-R128-W16A4：21 胜、16 负、243 持平，净 +5。这里描述本轮冻结 cohort 的点估计，不构成统计显著性或跨 seeds 稳定性结论。

| Suite（每组 70） | A+Z | A+Z+KL | KL 净变化 | C80-R128-W16A4 | BF16 |
|---|---:|---:|---:|---:|---:|
| libero_10 | 52 | 51 | -1 | 48 | 54 |
| libero_goal | 44 | 47 | +3 | 46 | 51 |
| libero_object | 55 | 55 | +0 | 54 | 55 |
| libero_spatial | 53 | 55 | +2 | 55 | 56 |

| 扰动（每组 40） | A+Z | A+Z+KL | KL 净变化 | C80-R128-W16A4 | BF16 |
|---|---:|---:|---:|---:|---:|
| background_textures | 24 | 26 | +2 | 24 | 26 |
| camera_viewpoints | 14 | 19 | +5 | 19 | 21 |
| language_instructions | 38 | 39 | +1 | 35 | 38 |
| light_conditions | 36 | 35 | -1 | 35 | 35 |
| objects_layout | 35 | 34 | -1 | 36 | 35 |
| robot_initial_states | 33 | 31 | -2 | 31 | 34 |
| sensor_noise | 24 | 24 | +0 | 23 | 27 |

收益最大的是 camera_viewpoints：14→19/40，配对 5 胜、0 负；background_textures +2，language_instructions +1。robot_initial_states −2，light_conditions 与 objects_layout 各 −1，sensor_noise 净持平。因此不是所有扰动都改善。

执行为 decoded BF16 dense MXVQ4 weights + dynamic A4 + BF16 KV。未测 native low-bit kernel 性能；本轮未追加 fixed-input heldout RMSE，也没有新增 baseline forwards。完整 cell、配对统计和实现内 timing 见 summary.json。

首次闭环作业 25925241.pbs101 因 clean container 漏传 CUBLAS_WORKSPACE_CONFIG 在启动阶段退出（0 episodes）；修复入口后复用原校准 checkpoint，不重训。修复记录见 launch_repair_25925241.json。

Goal 完成；20 分钟 gpt-6-luna/xhigh 监控已停止。
