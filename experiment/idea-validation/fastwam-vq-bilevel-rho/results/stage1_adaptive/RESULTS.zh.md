# Adaptive 阶段1结果与解释

阶段1已完成。保留adaptive训练终点明显缓解了最后rho≈2从头重训的motor退化，但仍未超过固定rho1.5；confirmation的latent误差也没有恢复。阶段2和3保持未提交。

## 主要比较

| 数据 | 固定rho1.5 first10 motor RMSE | 固定rho约2 | Adaptive | Adaptive vs rho1.5 | Adaptive vs rho约2 |
|---|---:|---:|---:|---:|---:|
| outer | 0.041933100 | 0.046343835 | 0.042808719 | +2.09% | -7.63% |
| confirmation | 0.043139591 | 0.054976732 | 0.045479680 | +5.42% | -17.27% |

confirmation逐case：adaptive相对固定rho1.5为1/8改善、7/8退化；相对固定rho约2为4/8改善、4/8退化。相对固定rho约2的总体motor改善，按原权重分解几乎全部来自5007（约100.22%的净改善；其他cases正负抵消后略为退化）。因此不能把17.27%的总体改善解释成普遍改善。

case5007 first10 motor RMSE：固定rho1.5为0.0639805，固定rho约2为0.1559205，adaptive为0.0797765。Adaptive比固定rho约2改善48.84%，仍比rho1.5退化24.69%。其seed0从rho1.5的0.0679704降至0.0623227，seed1从0.0599906升至0.0972303；平均退化主要保留在seed1。

case4007同task/phase的outer结果也只有部分恢复：adaptive比固定rho约2改善9.51%，仍比rho1.5退化26.60%。

confirmation future latent NMSE：固定rho1.5为0.0414010，固定rho约2为0.0568390，adaptive为0.0580138。Adaptive相对rho1.5退化40.13%，相对固定rho约2仍退化2.07%；case5007的latent虽然相对固定rho约2改善51.05%，其他cases（例如5003、5006）出现了较大latent退化。

## 对失败机制的含义

两个终点共享初始banks、C8顺序、16个committed updates、Adam规则和数值设置；差别是训练过程中使用的rho历史。Adaptive的rho轨迹为2.25→3→2→3→3→2→1.333333→2，每个rho提交两步；固定rho约2在16步全程使用相同常数。比较结果表明，抽取最后一个rho再从头训练，确实丢失了原路径对motor表现的影响；最后rho不足以概括adaptive模型。

这项结果没有证明单步unroll偏差是全部退化的原因，也没有直接比较相同rho候选在1步与16步下的外层方向。Adaptive自身仍劣于固定rho1.5且latent误差较大，因此直接采用adaptive终点也不足以解决原pilot的目标。若后续继续阶段2，应检验完整16步常数rho目标；本次不启动该阶段。

## 资源与完成证据

作业25883753.pbs101：请求walltime01:59:00，PBS实际GPU walltime00:17:09，1张A100，新增GPU成本0.285833 GPU·h。新增24F、0G；24份teacher只读复用，固定rho1.5与约2共48份endpoint结果只读复用，形成72条配对比较记录。完整逐case、逐seed五项指标已独立重汇总并与原始对照缓存核对。

PBS F/Exit0、artifact Exit0、PHASE_COMPLETE和matching complete summary全部通过；8轮16步adaptive源checkpoint及rho历史绑定通过，推理前后master parameters完全一致。GPU利用率与显存每15秒采集，共记录69条；gpt-6-luna/xhigh subagent按600秒等待间隔加查询开销监控，现已停止。

旧confirmation已被观察，本次为fixed-input teacher fidelity的回顾性路径诊断，不能据此宣称泛化、closed-loop成功率或native kernel性能改善。

来源：[comparison.json](comparison.json)、[GPU_COST.json](GPU_COST.json)、[final_verification.json](final_verification.json)、[telemetry_receipt.json](telemetry_receipt.json)。

## 完整指标与逐case表

仅评估已有8轮16步adaptive checkpoint；复用teacher及固定rho1.5/约2的已有结果。没有训练、更新rho或生成teacher。旧confirmation已被观察，本次为路径差异诊断。

| Split | Arm | first10 motor | full32 motor | late22 motor | gripper | future NMSE |
|---|---|---:|---:|---:|---:|---:|
| outer | current | 0.041933100 | 0.048189431 | 0.050100268 | 0.017021486 | 0.043017250 |
| outer | selected | 0.046343835 | 0.048399811 | 0.049165943 | 0.095403515 | 0.057514507 |
| outer | adaptive | 0.042808719 | 0.048647637 | 0.050548538 | 0.018710369 | 0.051524258 |
| confirmation | current | 0.043139591 | 0.045638952 | 0.046445387 | 0.016883708 | 0.041401048 |
| confirmation | selected | 0.054976732 | 0.056550184 | 0.056722655 | 0.017521921 | 0.056839021 |
| confirmation | adaptive | 0.045479680 | 0.048499588 | 0.049527573 | 0.016510035 | 0.058013752 |

## outer 逐case first10 motor RMSE

| Case | Suite | Phase | 固定rho1.5 | 固定rho约2 | Adaptive | vs rho1.5 | vs rho约2 |
|---|---|---|---:|---:|---:|---:|---:|
| 4000 | libero_spatial | early | 0.039503440 | 0.041068137 | 0.048301298 | +22.27% | +17.61% |
| 4001 | libero_spatial | middle | 0.033706155 | 0.049564917 | 0.035186689 | +4.39% | -29.01% |
| 4002 | libero_object | middle | 0.041849911 | 0.038856652 | 0.036156852 | -13.60% | -6.95% |
| 4003 | libero_object | late | 0.044852715 | 0.044265781 | 0.036256790 | -19.16% | -18.09% |
| 4004 | libero_goal | early | 0.051044147 | 0.041058604 | 0.045708265 | -10.45% | +11.32% |
| 4005 | libero_goal | middle | 0.045591000 | 0.051835030 | 0.051846892 | +13.72% | +0.02% |
| 4006 | libero_10 | middle | 0.038025040 | 0.046892770 | 0.037243340 | -2.06% | -20.58% |
| 4007 | libero_10 | late | 0.040892392 | 0.057208788 | 0.051769629 | +26.60% | -9.51% |

## confirmation 逐case first10 motor RMSE

| Case | Suite | Phase | 固定rho1.5 | 固定rho约2 | Adaptive | vs rho1.5 | vs rho约2 |
|---|---|---|---:|---:|---:|---:|---:|
| 5000 | libero_spatial | early | 0.042852154 | 0.040327589 | 0.043044534 | +0.45% | +6.74% |
| 5001 | libero_spatial | middle | 0.036165321 | 0.041857418 | 0.037967291 | +4.98% | -9.29% |
| 5002 | libero_object | middle | 0.038525308 | 0.044049362 | 0.040785844 | +5.87% | -7.41% |
| 5003 | libero_object | late | 0.034563468 | 0.036430178 | 0.038017312 | +9.99% | +4.36% |
| 5004 | libero_goal | early | 0.048340729 | 0.047137417 | 0.048668448 | +0.68% | +3.25% |
| 5005 | libero_goal | middle | 0.046629483 | 0.038617861 | 0.040776173 | -12.55% | +5.59% |
| 5006 | libero_10 | middle | 0.034059769 | 0.035473553 | 0.034801356 | +2.18% | -1.89% |
| 5007 | libero_10 | late | 0.063980501 | 0.155920476 | 0.079776486 | +24.69% | -48.84% |

全部五项逐case指标及两个seed的原始值见comparison.json。每case先平均seeds，再按suite平均，四suite等权。

证据边界：fixed-input teacher fidelity，decoded BF16/fake A4、单一C8训练轨迹；不代表closed-loop成功率、native kernel性能或训练seed复现。阶段2和3未提交。
