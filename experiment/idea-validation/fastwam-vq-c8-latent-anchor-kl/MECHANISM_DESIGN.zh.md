# C8 anchor KL 机制诊断，训练前冻结

目标是区分权重/有限步优化与 KL 所强调的 latent 方向。完整复用原 C8、teacher、未训练 bank 初始化、已完成 KL checkpoint 和原 baseline 结果。新增计算只用于诊断，不输出新的 trained baseline；不读取 heldout 来选参数。

**参数层。** 同一原始初始化与同一 KL 最终 checkpoint，分别对全部 8 个 C8 输入计算 action、加权 NMSE、加权 KL 的完整 books/log-row-gains surrogate 梯度。报告参数组及逐 case/全 C8 平均的 norm、cosine、跨 case 一致性。用完整 C8 平均梯度做可回滚的单步诊断：beta 为原值的 0、0.1、1 倍，步长为原 Adam delta 的 0.25、1 倍；两参数点都使用 fresh Adam，KL 最终点另用原 checkpoint moments。另测纯 KL 的 fresh-Adam 方向，将其 delta L2 norm 匹配 fresh base-Adam，避免纯方向比较被总步长混淆。所有条件均评估全部 C8，不只检查生成梯度的一个 case。每次均从同一个参数点出发，测完精确回滚 FP32 masters，并重新 decode。

**Latent 层。** 使用保存的 KL 最终 futures 与固定 teacher anchors。将 residual 在 raw latent Euclidean metric 下正交分为 KL 可见部分与 nullspace 部分；后者包含 pooling 消去的分量。对两部分分别缩减，目标是相同的 raw NMSE 降幅，取两部分较小能量的一半，确保两侧都可修复。实际 BF16 注入的 NMSE 相对目标误差不超过 0.1%，所有 case 保留。原始、teacher、两种修复分别运行同一个原 BF16 reader；核对 teacher future 的动作重放、frame0、noise 和 sampler counts。再测 reader action-error 对 latent 的梯度与 NMSE/KL 梯度关系。Teacher-assisted 修复是机制诊断，不可部署方法。

**判读。** 若 reduced beta 比原 beta 更稳定改善原目标/动作，支持局部权重不合适；若 surrogate 预测下降但真实量化 forward 上升，或小步改善、大步退步，支持有限步/离散量化优化问题。若纯 KL 确实下降却动作变差，或等 NMSE 下修复 nullspace 比修复 KL 可见方向更有利，支持当前分布强调方向与 reader 需求不一致。相反结果同样保留。单次或少数 case 的结果不升级为普遍机制，数个原因可以共存；全 C8 均值配合逐 case 分布报告。

预计新增 16 个带三项 backward 的完整 student queries、176 个无持久更新的参数 probe queries、32 个 BF16 reader-only queries（0 VideoDiT）。同一 forward 的三次 reverse pass 分别计数；内部 checkpoint recomputation 不冒充独立 queries。原 MSE+NMSE trained baseline 0 新 queries，teacher 0 新整模型 queries，0 闭环。全部张量操作位于真实 PBS compute allocation，GPU 使用记录每 15 秒写入 job log。

这组诊断的边界是两个参数点上的 C8 局部行为；不承诺唯一识别原 8-step 训练轨迹的因果原因，也不以这些条件产生的新 checkpoint 作为泛化结果。若证据不足，明确报告不足而非改判定口径。
