# Padded-coordinate feedback：method_no_go，停止

在预注册的8个新episode上，analytic padding replacement 没有减少 expert-W4 的physical action drift。八个episode全部通过binding gate；两种FP reference下的median gain分别为 **−1.1463% / −1.3189%**，仅 **3/8** 同时为正，未达到双median≥25%、至少6/8正向的门槛。

此为当前方法的经验no-go。没有发现导致结果失效的实现问题，不调整padding方法、bit数、noise或样本，不继续验证该切面。

| Episode | G1：相对FPnative | G2：相对FPanalytic |
|---|---:|---:|
| task0 / 33 | 4.8528% | 4.5793% |
| task0 / 58 | −10.2507% | −9.4919% |
| task1 / 11 | −0.8667% | −1.0211% |
| task1 / 19 | 0.8721% | 1.7444% |
| task2 / 35 | −1.4259% | −1.6166% |
| task2 / 44 | 5.4511% | 4.9915% |
| task3 / 45 | −6.2961% | −6.6187% |
| task3 / 48 | −3.0843% | −3.0299% |

FPanalytic 自身的physical变化小，所有episode满足S≤0.01E0；native quantization drift与首步padding扰动也都非退化。因此此次失败不是缺少作用对象、改变FP reference过多或工程gate未通过，而是解析padding更新没有达到所预测的改善。

CPU准备64771完成14s；GPU64772在TC1N05、Tesla V100-PCIE-32GB完成5m06s，实际workload302.643s；独立CPU64774重算完成。四臂、两固定noise、8个episode，共64个推理分支，另有no-op对照。完整state/velocity/physical action原始数组保留在 `/tc1home/UG/yguo017/v100_newangles_ccds/artifacts/64772/raw_analytic_padding.npz`。

原始数组SHA256：`6110c72e5210aaf47d12d028a36ef8151fef79e7f6d9faf10c204c915af5140b`。独立CPU通过native equality、analytic path、full Euler recurrence、physical endpoint、runtime source/checkpoint/processor绑定、sample hash、V100/allocation、no-op与weight restore检查。详细逐episode E0/S/E1/E2/P0和检查值见 [verification.json](artifacts/64774/verification.json)；执行信息见 [summary.json](artifacts/64772/summary.json)。

结论只覆盖此checkpoint的expert-only W4、固定8个离线condition。假量化没有验证native低比特加速；FP保真不等于环境success。zero-target解析路径本身不是新方法，当前经验no-go也不证明所有可能的unused-coordinate方法均不可做。
