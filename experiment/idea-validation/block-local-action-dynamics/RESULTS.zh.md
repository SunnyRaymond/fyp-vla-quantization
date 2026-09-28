# 第一轮结论：局部表示有信号，相对 toy dense MLP 的加速门槛未通过

本轮已完成 **45 个 primary runs + 3 个预设 global16 附加 runs**。所有机制检查通过，两个 PBS 作业正常退出。完整结果保留了独立、rank-4 耦合、稠密耦合三种条件；没有只选择表现较好的子集。

**学习坐标和跨块摘要改善了预测，但这套实现没有达到加速门槛。** global16 在低秩条件下的误差低于 dense，运行时间约为 dense 的 2.1–2.2 倍。global16 同时增加参数量，这项准确率结果不能单独归因于通信带宽。

**速度结论的比较范围：** 本轮参照是约 3.25 万参数的 64D dense MLP，没有测量六层 ViT predictor。这里的 NO-GO 仅表示当前实现未通过相对该 toy dense MLP 的预设门槛，不能据此判定它相对 LeWM / LpWM 的 Transformer predictor 不能加速，也不应以此作为进入小规模视觉实验的必要阻断条件。相对真实 baseline 的速度收益需要在相同表示接口、任务、预测质量、batch、horizon、精度和硬件下另行测量；本轮原始结果与门槛保持不变。

## 实验具体做了什么

真实状态由四个 16D 子系统组成，再用同一个固定稠密正交矩阵混合成 64D 观测。模型不知道真实分块；它学习完整正交坐标变换，再用四个局部 MLP 预测。所有状态维度保留，没有压缩截断。

每块接收同一个已知 8D action。global4 另接收 4D 全局状态摘要；local4 的投影参数量和 predictor 输入宽度相同，但摘要只看本块。global4/local4 各 32,576 个可训练参数，dense 为 32,552，差约 0.074%。random-block 与 learned-block 是表示对照，它们没有通信分支；其参数量单独报告，不能称为五个 arms 全部等参。

模型机制可以写成：

$$
s_t=(z_t-\mu)Q,\qquad m_t=P s_t,
$$
$$
s_{t+1}^{(k)}=s_t^{(k)}+f_k(s_t^{(k)},a_t,m_t).
$$

Q 保留完整状态，f_k 只看本块、公开 action 和摘要。其他块的状态只能经过 r 维摘要进入本块，r 为 4 或 16。对状态的 Jacobian 因而可以分为块内项与 rank 不超过 r 的通信修正项；这里说的是修正项的 rank，不能把去掉对角块后的矩阵也直接称为 rank-r。这是结构约束，不是已经学到真实分块或已经省下运行时间的证据。

每种条件 train/dev/test 为 512/128/128 个 episodes。三个 training seeds 各固定 1500 steps，配对 minibatch draws，使用最后 checkpoint。主要指标是 h10 末步状态 MSE，以训练集 delta energy 归一化。另报 h1/h5/h20、action-response、原坐标每步输出延迟、训练时间与 Torch allocator peak memory。

## 主要结果

下面是 lowrank_coupled 的三个 seeds 平均。完整三条件表和逐 seed 的 episode-paired bootstrap intervals 见主报告。

| Primary arm | h10 error | Batch-300 complete rollout | Params |
|---|---:|---:|---:|
| dense | 0.12627 | 1.324 ms | 32,552 |
| random-block | 0.92850 | 2.536 ms | 27,200 |
| learned-block | 0.26551 | 2.533 ms | 31,296 |
| learned-global4 | 0.19695 | 2.836 ms | 32,576 |
| learned-local4 | 0.27918 | 2.876 ms | 32,576 |

**表示对照有一致信号。** learned-block 在三个条件、三个 seeds 中均优于 random-block，逐 seed 配对区间也均低于零。固定随机切块的损失很大，允许坐标学习能收回相当一部分误差。它仍未达到 dense 的质量门槛。

**4D 摘要有一致的模型层面增益。** global4 对等参 local4 在三个条件、三个 seeds 中均更好。低秩条件 h10 从 0.27918 降至 0.19695。然而 independent 条件也从 0.25745 降至 0.18745，而该真实系统的跨块 Jacobian 为零。因此这不能直接证明真实系统需要这些跨块信息；通信可能在补偿尚未学好的坐标或局部 predictor。

**当前实现的速度门槛失败。** 所有 primary 分块 arms 在三个条件中均未通过质量门槛；batch-1 与 batch-300 都更慢。global4 在低秩条件下的 batch-300 延迟约为 dense 的 2.14 倍。计时已包含初始坐标变换、native rollout、通信与最终重构；Q 已提前冻结缓存，每步重构接口也单独测量。

## global16 附加对照

预设 dev 条件触发后，仅在 lowrank_coupled 增加三个 global16 seeds。data、loss、训练步数、最后 checkpoint 和评估协议相同。原始 dense/global4 checkpoints 在这个 allocation 内重新计时，并复现了主实验记录的逐 episode h10 error。

| Seed | Dense h10 | Global4 h10 | Global16 h10 | B300: dense / global16 |
|---:|---:|---:|---:|---|
| 1101 | 0.12647 | 0.19942 | 0.10791 | 1.283 / 2.791 ms |
| 1102 | 0.12899 | 0.19382 | 0.10567 | 1.287 / 2.771 ms |
| 1103 | 0.12336 | 0.19762 | 0.10258 | 1.525 / 3.327 ms |

global16 的 h10 平均约 **0.1054**，三个 seeds 均优于 dense，也均通过预设 quality gate；配对 bootstrap intervals 在这些固定 seeds 内均支持误差下降。其 batch-1、batch-300 speed gates 均失败。

global16 有 36,416 个参数，高于主实验的 dense/global4。没有训练 local16 等参对照，所以不能把其增益单独归为通信维数。追加机制测试中的同预算 dense 只是参数计数检查；实质比较使用主实验已有的 32,552 参数 dense checkpoint。

## 对原问题的启发

这轮提供了“换坐标值得做”和“少量全局摘要能帮助当前局部模型”的 pilot 证据，尚未得到一个便宜的替代 predictor。

低秩条件下，学习后、经过最优块排列匹配的真实子空间 overlap 约为 0.30–0.34；global16 约为 0.28–0.30，离完整恢复预设分块的 1 很远。该指标仅是描述：模型可能学到其他有用坐标。不过，增加通信后准确率改善、真实分块恢复仍弱，提示 predictor 可能在借助通信绕过分块困难。

工程上的另一个问题是，四个小 MLP 并不会自动比一个 dense MLP 快。当前 PyTorch 实现包含分块矩阵运算、显式 bias、摘要投影、拼接和变换；实测结果证明整体更慢。没有进行 kernel profile，不能精确归因到其中某一项。

后续最有价值的两个独立问题是：固定已知真实坐标作为 oracle 时，局部 predictor 能否收回 quality gap；以及局部计算能否通过合适的预算与实现得到真实速度收益。它们是未执行的后续问题，本轮不增加训练预算或放宽门槛。

结论只适用于这三个 64D 受控系统、固定训练预算和当前实现；它不证明视觉 latent 的局部结构，也不证明 CEM ranking 或 closed-loop 控制效果。三个 training seeds 的 episode-bootstrap 区间不能推广成训练随机性的总体结论。

## 阅读与复现

- [完整 primary 报告](results/25563194.pbs101/REPORT.zh.md)：全部 45 runs、三条件、全部对照、逐 seed 区间。
- [global16 附加报告](results/25563721.pbs101/REPORT.zh.md)：三个追加 seeds 与同 allocation 计时。
- [主实验原始 summary](results/25563194.pbs101/summary.json)；[global16 comparison](results/25563721.pbs101/global16_comparison.json)。
- [冻结协议](PROTOCOL.zh.md)；[primary FREEZE](results/25563194.pbs101/FREEZE.json)；[global16 FREEZE](results/25563721.pbs101/FREEZE.json)。
- [主实验机制检查](results/25563194.pbs101/mechanism_tests.json)；[global16 机制检查](results/25563721.pbs101/mechanism_tests.json)。

主作业 `25563194.pbs101`：A100 host `x1000c3s1b0n1`，12 分 48 秒，PBS Exit_status=0。追加作业 `25563721.pbs101`：A100 host `x1000c1s7b0n0`，2 分 43 秒，Exit_status=0。两个 job.log 均保留每 30 秒的 GPU 利用率和显存记录。Checkpoints、training.tsv、per-episode arrays 与实际 source snapshots 留在各自 compute run 目录。

追加 allocation 记录环境为 PyTorch 2.8.0+cu128、CUDA 12.8、NVIDIA A100-SXM4-40GB、matmul TF32=False。主实验使用同一已有环境路径，原始 checkpoints 的结果在追加 allocation 中复现。

完成核验覆盖完整 45 项 Cartesian combinations、三个预设附加 seeds、两个 completed manifests、两套各八项机制检查、data split identities、全部四个 horizons 与四种 timing 字段、六项原 checkpoint 误差复现、39 个逐 seed 配对对照，以及两份 GPU telemetry logs。训练时间和 allocator peak memory 在各 run 的原始 summary 中保留；allocator peak 不包含全部驱动/context 显存，而且两次作业缓存的条件数不同，不能把两次作业的这个数直接当作模型显存节省对照。

实验设计使用 experimental-design skill；引用：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. https://doi.org/10.48550/arXiv.2609.00065
