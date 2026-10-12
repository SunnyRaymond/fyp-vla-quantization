# SpinQuant native INT4：同输入 action RMSE baseline

**完整跑完，test motor RMSE = 0.057373。** 相同观察与seeds下，低于当前action-sensitive VQ（0.061030），高于Scalar（0.054994），接近row-scale VQ（0.057052）。这些是完整配置的描述性差异，未作显著性或等效性判断。

作业 `25756064.pbs101`：PBS F、Exit_status=0、wrapper exit_code=0、PIPELINE_COMPLETE；1张A100，实际walltime **00:13:11**。模型加载1次，复用原200-step learned rotation及packed bank，没有重新训练或calibration。52次新增query含4个selection observations×2seeds及22个test observations×2seeds，另做1次BF16缓存重放检查，max_abs=0。

## 固定输入结果

主指标是 normalized action 前10步、6个motor维度相对同输入同sampler seed的BF16 RMSE。先在每个case内平均两个seeds，再对cases等权平均。BF16及既有VQ/Scalar action来自已完成作业25743693.pbs101，本轮不重新推理这些对照。

| 配置 | Selection motor RMSE | Test motor RMSE | Test gripper RMSE | Test 32-step motor RMSE |
|---|---:|---:|---:|---:|
| BF16 reference | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| BF16 weights + A4 | 0.039102 | 0.048559 | 0.015465 | 0.059609 |
| Scalar W4A4 + Smooth α=0.5 + Hadamard | 0.044648 | 0.054994 | 0.012511 | 0.059341 |
| Row-scale VQ + A4 | 0.047404 | 0.057052 | 0.031798 | 0.059709 |
| 局部输出拟合 VQ + A4 | 0.042646 | 0.057256 | 0.018110 | 0.066828 |
| SpinQuant adapted native W4A4 | 0.047347 | 0.057373 | 0.024286 | 0.068293 |
| Action-sensitive VQ + A4（λ=0.1） | 0.041168 | 0.061030 | 0.021261 | 0.066600 |
| 原始 VQ + A4 | 0.063518 | 0.071311 | 0.040502 | 0.074634 |

## 配对差异

| 对照 | SpinQuant平均RMSE差 | 相对差 | SpinQuant改善cases / 22 |
|---|---:|---:|---:|
| Scalar W4A4 + Smooth α=0.5 + Hadamard | +0.002379 | +4.33% | 5/22 |
| Row-scale VQ + A4 | +0.000321 | +0.56% | 9/22 |
| 局部输出拟合 VQ + A4 | +0.000118 | +0.21% | 7/22 |
| Action-sensitive VQ + A4（λ=0.1） | -0.003656 | -5.99% | 13/22 |
| 原始 VQ + A4 | -0.013937 | -19.54% | 19/22 |

差值为SpinQuant减对照，负值表示SpinQuant误差更低；百分比为RMSE相对差，不是成功率变化。

SpinQuant在10个Original test observations上为0.049749；在12个Plus test variants上为0.063727。两次seed是配对重复，不能将44个contexts看作44个独立场景。

## native执行证据与启动快照

**没有回退为BF16或decoded SpinQuant。** 正式52个query均记录614层调用及每query **6,436次native INT4 GEMM**，共334,672次。实际执行的run_rmse.py快照在每个query保存前强制检查native调用数>0和PTX验证通过。

监控读到的artifact/runtime.json是在pilot.Runtime构造时写入，早于BF16 reference replay、rotation.apply_native_bank和q.enable("w4a4")，所以其中packed_kernel=false、native_int4_gemm_calls=0等描述BF16启动阶段。该原始快照保留；正式query记录及native_execution_receipt.json给出本次native执行证据。

本地controller已补充逐query native metadata核对；本地runner为后续运行增加snapshot_phase及runtime_native.json。未为修正日志重新运行模型。

## 结果边界

SpinQuant复用原learned R128、W4 G128 codes/BF16 scales及动态per-row A4，KV保持BF16。现有VQ/Scalar对照为其原decoded-BF16执行；本轮按用户要求只增加native INT4 SpinQuant一组，没有decoded SpinQuant实验。训练输入与变换不同，因此此处比较完整配置。

没有执行预测action、没有closed-loop episodes，没有同输入native latency benchmark。上述误差不能直接替代成功率、速度或 unseen-task generalization 结论。

## 保存的证据

- [完整结果JSON](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-spinquant-rmse/retrieved/results/full/summary.json>)
- [native执行核对记录](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-spinquant-rmse/native_execution_receipt.json>)
- [PBS完成记录](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-spinquant-rmse/terminal_evidence.json>)
- [固定协议](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-spinquant-rmse/protocol.json>)
- [原始启动快照](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-spinquant-rmse/retrieved/artifacts/25756064.pbs101/runtime.json>)
- [实际运行入口快照](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-spinquant-rmse/retrieved/artifacts/25756064.pbs101/run_rmse.py>)
