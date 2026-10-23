# VQ loss 更换诊断

状态：C8 缓存分析与代码复核完成；固定 checkpoint 的参数梯度诊断尚在排队。没有启动新训练，不据此选择新 loss 系数或训练步数。

## C8 是否拟合

CPU 作业 `25895158.pbs101` 已完成，PBS F/Exit0、artifact exit0、PHASE_COMPLETE、同作业完整 summary 均通过。只读取原 C8 的输出缓存和原共享 teacher；新增模型 forward、teacher forward、训练更新均为 0。C8 仍为原 8 cases、每例原 seed 0；四个 suite 各 2 cases。

| C8 metric | Raw VQ | MSE-VQ8 | KL-VQ8 |
|---|---:|---:|---:|
| First10 motor RMSE | 0.043527373 | 0.042133242 | 0.040801177 |
| Future latent NMSE | 0.068408117 | 0.054480982 | 0.043700444 |
| Action soft-bin KL | 0.15178104 | 待固定 checkpoint 诊断 | 0.12299066 |
| Latent soft-bin KL | 0.21312689 | 待固定 checkpoint 诊断 | 0.14140191 |
| Action temporal KL | 0.084253263 | 待固定 checkpoint 诊断 | 0.066448606 |
| Latent spatiotemporal KL | 0.0074527581 | 待固定 checkpoint 诊断 | 0.0050821758 |
| 完整 relational objective | 0.0080299220 | 待固定 checkpoint 诊断 | 0.0057318184 |

可确认 KL-VQ8 学到了 C8：相对 raw，四项 KL、完整 objective、RMSE/NMSE 都降低。相对 MSE-VQ8，KL-VQ8 的 C8 RMSE/NMSE 更低，但原 held-out 两项分别更高 10.33%/12.32%。这支持 calibration-specific fitting 与泛化错配的判断，不能直接解释为 KL 无法拟合或单纯训练步数太少。MSE-VQ8 的完整 KL 指标尚未补齐，不能提前声称 KL-VQ8 在 C8 上所有目标都优于 MSE-VQ8。

## 8 updates 的实际含义

两种 VQ 都只对原 C8 做一个 epoch：每步取一个 case、一个固定 seed，8 个 case 各更新一次。KL `worker.py:96-100,204-212` 与原 MSE `train_c8.py:317-320,425-440` 的预算匹配。不是 batch size 8，也不是 8 epochs。

日志的每个 loss 来自不同 case 的更新前 checkpoint；不能把这八个值当成同一个训练集上的收敛曲线。只有 final8 checkpoint，未保存一条完整 C8 的逐步评价曲线。因此现有轨迹不能证明 update8 已收敛，也不能证明追加更新会改善泛化。C8 已拟合而 held-out 相对变差，至少说明“直接多训”尚无充分依据。

## loss 权重与优化机制

KL 为 `alpha_a*(A_KL+beta_a*A_rel)+lambda*alpha_z*(Z_KL+beta_z*Z_rel)`，其中 `lambda=rho*c=0.04245452683`。

- alpha 来自旧 activation 问题的完整 C80 初始 MSE/KL、NMSE/KL 均值比（`fastwam-vq-dual-kl/worker.py:108-127`）。
- beta 来自完整 C80 teacher 加标准化 Gaussian noise 0.1 后，`0.25*mean(anchor KL)/mean(relational KL)`（`fastwam-vq-relational-kl/prepare.py:53-67,98-103`）。
- 这些是数值校准，不是 VQ 参数梯度归一化。不能从 beta_latent 约 27 直接推出 latent relation 主导参数更新。
- KL 和 MSE 均从原 raw C8 bank 开始；同一固定 indices/exponents、FP32 books/log row gain 参数化；Adam LR=0.0005、总梯度 clip=1。两者均在第一次 optimizer.step 后清 Adam state，update2 重新积累 moments。
- KL 八步日志中的总梯度范数都小于 1，因此这轮没有触发总梯度裁剪。614 个模块的梯度 connected/finite 检查通过，但这不代表四项梯度彼此兼容。
- A4 舍入和 BF16 VQ 的 backward 是 STE surrogate；VJP 测的是该 surrogate，不能当成离散前向的精确导数。

Adam 对整体 loss 的同一正标量缩放，在 epsilon 可忽略且未裁剪时会近似抵消；reset 本身并不破坏这一齐次关系。改变某个 component 的相对系数则会改变合成梯度方向与相消程度。需要同时查看分量范数/夹角、Adam epsilon、books 与 gains 的更新，以及 BF16 存储变化，不能只用总梯度大小选择学习率。

## 待完成的只读诊断

GPU 作业 `25895153.pbs101`，walltime 01:59:00，原 raw/MSE8/KL8 三个固定 checkpoint × 全部 C8 = 24 次 student sampler query。每次拆 4 项加权 KL 与原 MSE 的 2 项加权梯度，分别记录 output-space 与 VQ parameter-space norm/cosine；同时记录 Adam 状态和 FP32 master/BF16 存储变化。没有 optimizer 构造或 step，不改原 checkpoint，不用 held-out，不产生 teacher forward。

尚不能确认具体是哪项梯度导致泛化错配，也不能据此确定新 beta、学习率或训练步数。最终建议待该证据补齐后给出。
