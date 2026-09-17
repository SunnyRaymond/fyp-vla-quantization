# Rounding persistence：scope_limited_preliminary_go，STOP

贡献审查补充：**generic method novelty no-go**。更直接的De-biasing Diffusion prior已提出stochastic weights减少跨denoising-step误差相关性；本案只保留matched-marginal SmolVLA诊断signal，见[novelty边界](../../../../idea/v100-new-angles/rounding-persistence/NOVELTY_BOUNDARY.zh.md)。不改变下面原始数值结论。

GPU **64825** 在TC1N04 V100完成5m47，内部workload341.92s；独立CPU **64831** 完成9s。全部工程、输入、source、snapshot、共同路径hash与原始数组检查通过；六个state全部非退化，**6/6**通过冻结joint gate（要求至少5/6）。停止本切面，不追加完整验证。

比较保持每个denoising step相同的三个SR draw multiset，只改变跨step使用同一draw还是轮换draw。每state的两个固定noise组成B2，所有arms同样批处理，不跨state组batch，不平均不同arm的action输出。主指标均为normalized first8×physical7；每state内平均noise与三个draw，再作六state判断。

| state | free endpoint error改善 | common-FP-path forcing改善 | joint gate |
|---|---:|---:|---|
|0|58.59%|77.60%|通过|
|1|53.55%|69.07%|通过|
|2|38.20%|54.51%|通过|
|3|74.21%|80.01%|通过|
|4|71.56%|80.53%|通过|
|5|70.65%|77.41%|通过|

Mean endpoint MSE：Frozen **0.004836900**，Cyclic **0.001674838**，RTN参考 **0.007177889**。Common-path逐step误差边际与diagonal energy匹配；forcing为固定FP状态路径的局部误差积分，不能直接解释成真实非线性rollout endpoint的精确分解。结果支持所测recipe中跨denoising-call rounding persistence影响误差积累这一最小机制前提。

证据：[CPU最终verification](artifacts/64831/verification.json)、[GPU摘要](artifacts/64825/summary.json)、[冻结协议](PROTOCOL.zh.md)、[B2修复gate](BATCH2_REPAIR_GATE.zh.md)。Raw SHA256 `00e0555fba8423af73efabb8cb56433de265e69d6eaaee86cf7232652f294c86`；大数组与112个Linear的snapshot保留在CCDS64825目录。

保留的失败和修正：原GPU **64815** 9m03外部timeout、11/12完成；CPU **64823** 正式归档 `inconclusive_budget`，从未分析其partial科学数组。唯一B2执行修复保持样本/seeds/阈值/540s上限，全部重做且不与64815拼接。CPU **64829** 的数值结果已通过，但把诊断数值 `actual_times_max_abs_error=0` 转成bool后误判失败；64831仅修正汇总为boolean gates，保留数值误差receipt为0。没有再跑GPU或改科学阈值。

此为FP32 fake-quantized数值screen；轮换三个weight snapshots的存储、切换和native kernel成本未测。不能据此声称单模型W4部署、显存或延迟收益、LIBERO task success、其他checkpoint泛化。原超时成本与B2成本均计入记录；科学go也到此停止。
