# Scalar W4A4 + Smooth α=0.5 + Hadamard：LIBERO-Plus 280 episodes

本轮成功 **200/280（71.43%）**。相同冻结 cohort 上，比 QuaRot 多成功 5 个（+1.79 个百分点），比 SpinQuant 少 3 个（−1.07 个百分点），比 BF16 少 16 个（−5.71 个百分点）。这些是本 cohort 的描述性差异，未作显著性或等效性判断。

作业 `25734678.pbs101` 已以 `Exit_status=0` 结束；两个 worker 均完成 140/140，worker 和 wrapper 退出码均为 0，`PIPELINE_COMPLETE` 已生成。实际 walltime **09:02:16**，使用 **1 PBS、2 张 A100-SXM4-40GB**。每卡持续运行一个 worker，模型各加载一次，共 2 次；复用原 packed bank，新增 training steps 和 calibration cases 均为 0。

## 总体与 suite 分项

| 配置 | 成功 / episodes | Success rate |
|---|---:|---:|
| BF16 | 216/280 | 77.14% |
| QuaRot adapted W4A4 | 195/280 | 69.64% |
| SpinQuant adapted W4A4 | 203/280 | 72.50% |
| Scalar W4A4 + Smooth α=0.5 + Hadamard | 200/280 | 71.43% |

每个 suite 的分母均为 70。

| Suite | BF16 | QuaRot | SpinQuant | Scalar + Smooth + Hadamard |
|---|---:|---:|---:|---:|
| Spatial | 56/70 | 51/70 | 55/70 | 53/70 |
| Object | 55/70 | 52/70 | 51/70 | 52/70 |
| Goal | 51/70 | 45/70 | 48/70 | 47/70 |
| Long | 54/70 | 47/70 | 49/70 | 48/70 |

## 7 个扰动维度

每个维度跨 4 个 suites，共 40 个 episodes。完整 28 个 suite × dimension 单元均有 10 个 episodes。

| 扰动维度 | BF16 | QuaRot | SpinQuant | Scalar + Smooth + Hadamard |
|---|---:|---:|---:|---:|
| Background textures | 26/40 | 24/40 | 23/40 | 26/40 |
| Camera viewpoints | 21/40 | 12/40 | 16/40 | 14/40 |
| Language instructions | 38/40 | 36/40 | 36/40 | 37/40 |
| Light conditions | 35/40 | 36/40 | 39/40 | 35/40 |
| Objects layout | 35/40 | 36/40 | 33/40 | 34/40 |
| Robot initial states | 34/40 | 30/40 | 31/40 | 30/40 |
| Sensor noise | 27/40 | 21/40 | 25/40 | 24/40 |

相对 BF16 的 16 个成功数差额中，Camera viewpoints 少 7 个，Robot initial states 少 4 个，Sensor noise 少 3 个；Language instructions 和 Objects layout 各少 1 个。相对 SpinQuant，Scalar 在 Background textures 多成功 3 个，但在 Light conditions 少成功 4 个，因此不能从总体仅差 3 个推断两者逐样本相同。

## 配对比较

胜：Scalar 成功、对照失败；负：Scalar 失败、对照成功；平：两者同成或同败。

| 对照 | 胜 | 负 | 平 | 净成功数差 | Success rate 差 |
|---|---:|---:|---:|---:|---:|
| BF16 | 9 | 25 | 246 | -16 | -5.71 pp |
| QuaRot adapted W4A4 | 18 | 13 | 249 | +5 | +1.79 pp |
| SpinQuant adapted W4A4 | 14 | 17 | 249 | -3 | -1.07 pp |

## 显存和 action query 耗时

单位为 GB（10⁹ bytes）与秒。Query 均为 IDM 模式，32-step action chunk、10 个 denoising steps。表中时间为各 worker 排除首个正式 query 后的平均值，包含 Scalar 的 7,333 个 query 样本；环境步进和模型加载不在 full action query 计时内。首个正式 query 前已执行 native setup diagnostic，首个 query 也不能解释为冷启动耗时。

| 配置 | 常驻 tensor 显存 | Query tensor 峰值 | Reserved 峰值 | Denoising 平均 | 完整 query 平均 |
|---|---:|---:|---:|---:|---:|
| BF16 | 24.84 GB | 25.11 GB | 25.38 GB | 0.950 s | 1.003 s |
| QuaRot adapted W4A4 | 16.01 GB | 16.28 GB | 25.54 GB | 3.746 s | 3.803 s |
| SpinQuant adapted W4A4 | 16.01 GB | 16.28 GB | 25.54 GB | 3.736 s | 3.792 s |
| Scalar W4A4 + Smooth α=0.5 + Hadamard | 15.96 GB | 16.23 GB | 25.38 GB | 6.065 s | 6.124 s |

Scalar 的 tensor 显存峰值相对 BF16 减少 **35.36%**，与另两组 W4A4 接近；其完整 query 平均为 **6.124 s**，比本次 SpinQuant 实现耗时长 **61.5%**。Scalar 的完整 query 中位数为 6.051 s，p95 为 6.674 s。

额外耗时主要落在 denoising 区间。当前 Smooth/Hadamard 路径包含 FP32 除 scale、分阶段 FWHT、A4 quantization/packing 和 native GEMM；FWHT 虽减少数学运算量，当前 eager 实现仍会产生多次 kernel launches 和中间 tensor 读写。它们是代码层面的候选原因，未进行逐算子 profiling。此处比较的是各自 closed-loop query 分布，也不是同输入 microbenchmark。

Tensor allocated 显存不包含 CUDA context 和外部 renderer 分配。Reserved 包含 allocator cache；本轮先加载完整 BF16 pipeline 再替换 W4 bank，因此 reserved 保留约 25.38 GB，不能把 16.23 GB tensor 峰值直接当作整张卡的总占用。

## 配置与证据边界

本轮沿用原 Scalar W4+A4 配置：Smooth α=0.5、signed block Hadamard H128、seed=20261006、W4 G128/BF16 scales、动态 per-row A4、KV BF16。614 个目标 Linear 的原 weight codes、scales、Smooth scales 和 Hadamard signs 均复用。实际执行 native S4 Tensor Core MMA，group-scaled FP32 accumulation 后输出 BF16。

原 fixed-input 诊断使用 decoded BF16 A/W GEMM。当前 native 运行保持原 bank 和变换，但累加算术不同：启动时 case 1 的 normalized 32×7 action RMSE 为 0.044958，first-10 motor RMSE 为 0.047790，均是 native 相对原 decoded-BF16 reference 的差异，不能解释为 native 相对 BF16 baseline 的全量误差。源 transform 和 repadding 后的 weight decode 均通过 bitwise identity 检查，native fixtures 和 PTX 检查通过。

Scalar 原 calibration/selection 仅覆盖 Spatial，且含 original 与 Plus 样本；QuaRot/SpinQuant 的 calibration 覆盖四个 original suites，因此这是完整配置的比较，不能单独归因于 Smooth 或 Hadamard。Calibration 与 selection 对本 cohort 的 exact variant overlap 为 0；原 diagnostic test 与本 cohort 有 2 个 Spatial variants 重合。该 bank 是在先前诊断结果已知后选用的复用对照，本轮未重新拟合。

每个 variant 使用 state 0、30 个 settling steps，共享原环境与 sampler seeds，普通 suite 最大 400 控制步，Long 最大 700 控制步。对照使用既有作业 `25728285.pbs101` 的 840 条结果，本轮仅新增 280 条 Scalar episodes。PBS allocation 内的实际 aggregate 逐条检查了唯一 slots、variants、seeds、native evidence 和完整 query prefix，确认 28 个单元各 10 个 episodes。

## 本地结果文件

- [完整统计 JSON](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-smooth-scalar-plus/results/summary.json>)
- [完成核对记录](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-smooth-scalar-plus/formal_completion_receipt.json>)
- [配置与 provenance](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-smooth-scalar-plus/protocol.json>)
- [原始 aggregate 入口快照](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-smooth-scalar-plus/artifacts/25734678.pbs101/aggregate.py>)

本次未做逐算子 profiling、显著性检验或额外 closed-loop 重复；以上结论限定于已冻结的 280 个 variants 和这次 native 实现。
