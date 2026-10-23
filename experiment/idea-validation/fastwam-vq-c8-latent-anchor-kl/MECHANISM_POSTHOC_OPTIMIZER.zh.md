# 对已缓存结果的 CPU 补充，2026-10-22

初始点完整 C8 的参数梯度已经完成。KL 与 NMSE 梯度 cosine 为 0.97046，与原总目标为 0.89123；当前已完成的初始化单步 probe 都显示 BF16 row gains 改变数为 0，而模型的 FP32 gains 有非零 Adam delta。

因此在原声明的 22 个参数 panels、16 个梯度 records、8 个 reader cases 全部保留的前提下，追加 CPU-only Adam 几何分析：按 books 与 log-row-gains 分解 g·delta，核对预测收益落在哪个参数组；比较不同 beta 的实际 Adam 提议方向；将 KL 梯度投影到 NMSE 梯度上，衡量它主要是加强已有方向还是提供新的正交方向，并检查该正交部分与 action 梯度的关系。另逐 case 对齐旧的输出 latent 梯度与新的参数梯度 cosine，避免混用均值和逐 case 口径。

只读取已保存梯度、最终 checkpoint/optimizer state 和原 C8 bank。初始化 log gains 在 CPU 重建，可能与原 GPU log 有末位 FP32 差别，报告会记录重算 norm 和 g·delta 相对保存值的偏差。它不是 bitwise master 重建或新的 GPU 实验。不改变运行中的 mechanism_worker、GPU protocol、cohort、权重条件、步长或任何推进口径；新增 model forward、backward、training update 均为 0。
