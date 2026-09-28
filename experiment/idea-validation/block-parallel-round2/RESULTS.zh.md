# 第二轮：正确坐标 oracle 与视觉 block-pushing pilot

两条实验均已完成并取回结果。**Oracle 支持“正确表示 + 足够的跨块信息”这一方向；视觉 pilot 按冻结门槛判定为 inconclusive。** 视觉中的小 predictor 确实比本接口的六层 Transformer 快，但分块的独有收益、同等精度替代和真实 LeWM 加速尚未成立。

## 两条实验怎样配合

Oracle 使用生成器的已知正确坐标，回答第一轮困难是否来自学习分块坐标。视觉 pilot 从 RGB 图像学习 object slots，回答局部预测与少量通信能否在一个视觉环境中成立。两条线路独立推进，视觉实验没有等待 oracle 的质量或速度门槛；数据准备和实现工作与 oracle 工作重叠。两项 GPU 训练并未同时启动，不声称同时占用两个 GPU 运行。

| 实验 | PBS job | 当前证据 |
|---|---|---|
| 视觉 CPU 数据准备 | 25564702.pbs101 | F，Exit_status=0；1000/128/256 episodes，7 项环境检查通过 |
| Toy oracle GPU | 25564754.pbs101 | F，Exit_status=0；30 次训练、39 个原 checkpoint 复现、4 项机制检查完成；walltime 8m25s |
| 视觉 GPU | 25564986.pbs101 | F，Exit_status=0；3 个 encoder fits、21 个 predictor fits 全部完成；walltime 24m24s |

GPU 训练与 benchmark 在真实 PBS compute allocation 内执行；job.log 按 30 秒采集 GPU 利用率和显存。原始 checkpoints、逐 episode 数据和 source snapshots 保留在各作业的 remote run 目录。

## Oracle 已经说明什么

以下为三个固定 training seeds 的 h10 归一化末步误差均值，越低越好。易懂解释见 [oracle 阅读报告](../block-local-action-dynamics-oracle/results/25564754.pbs101/RESULTS.zh.md)，逐 seed 配对区间、action-response、各 horizon 和延迟见 [oracle 原始报告](../block-local-action-dynamics-oracle/results/25564754.pbs101/REPORT.raw.zh.md)；原始数值以该目录的 summary.json 为准。

| 系统 | 正确坐标、无通信 block | 正确坐标、local4 | 正确坐标、global4 |
|---|---:|---:|---:|
| independent | 0.07520 | 0.07442 | 0.07498 |
| lowrank_coupled | 0.14619 | 0.14614 | 0.07708 |
| dense_coupled | 0.17918 | 0.17903 | 0.14390 |

第一，**坐标确实是重要瓶颈**。全部 30 个 oracle-vs-learned 配对 h10 差分都指向误差下降。独立系统在正确坐标下，无通信 block 已能取得良好结果；因此第一轮不能解释成“低维局部 predictor 本身不可行”。Oracle 得到了额外的生成器信息，并冻结了坐标参数，这是诊断结果，不是已经学会这种表示的可部署模型。

第二，**正确坐标不等于不需要通信**。lowrank_coupled 中，global4 明显优于等参数、同局部函数初始化的 local4；多给本块一个局部投影不能代替读取其他块的信息。global16 的 h10 约 0.077，与 global4 相近；这次结果没有显示必须使用 16D 消息。global16 同时改变通信宽度和参数容量，仍不能单独归因于带宽。

第三，**全局 4D 摘要也有局限**。dense_coupled 中它改善了误差并通过预设容差门槛，但每个 seed 仍比 dense predictor 误差高；“通过门槛”不等于“优于 dense”。这里的跨块依赖比 lowrank 系统广泛，不能仅凭正确坐标消除。

第四，**当前 toy 实现没有通过速度门槛**。Oracle arms 相对本轮同 allocation 重测的 toy dense MLP，在 B=1、B=300 均未达到 20% 加速。这仍然没有测量 LeWM 的实际 6-layer ViT，不能否定相对它的加速可能性。

global4-vs-local4 是等参数的信息范围对照；block-vs-global4 同时改变消息通路与参数量，只支持结构整体差异。三个 training seeds 的 episode bootstrap 不能推断训练随机性的总体分布。

## 视觉实验的结构

冻结视觉 encoder 后，七种 predictor 从头配对训练：full GNN、local、local4、global4、global16、6-layer Transformer、参数量接近 global4 的 flat MLP。比较 h1/5/10/20 的 H@1、MRR、latent MSE、自由移动/物体阻挡/边界阻挡、action-response，以及 B=1/300 的完整 h10 延迟。

机制可以理解为：同一张图像先变成五个 16D slots；局部 predictor 更新自己的 slot，global4 另外读取由全部 slots 压成的四个数。local4 读取的四个数仅来自自己，因此它与 global4 的区别是信息来自哪里。它们的参数量和初始化匹配，但参数匹配不意味着各架构的计算量相同。

H@1 h10 问的是：连续预测十步后，能否从固定的 256 个 held-out 未来状态中，把真实对应状态排在第一。MRR 记录真实状态的排名，latent MSE 记录数值距离。Action-response 则从同一状态比较两个动作，检查预测是否捕捉了动作造成的差异；这些评价都没有运行 CEM。

Stage A 的 encoder 与完整 GNN 联合训练，再固定给全部 Stage B predictors，因此表示可能偏向 GNN。这是统一表示接口的机制对照，不是比较所有架构分别训练自己的最佳 encoder；slot 编号也不能仅凭输出维数就当作已验证的物体语义。

预设的 full GNN adequacy 条件为：每个 seed 的 h1 H@1 >= 0.80，h10 H@1 >= 0.50，且 h10 相对 persistence 至少增加 0.05；三个 seeds 都要通过。此次未全部通过，仍完成了所有 24 fits 并报告 inconclusive，没有追加训练或改门槛。

当前接口中的六层 Transformer 是新的架构对照，不是 LeWM checkpoint。C-SWM 的对象 action 和格子阻挡也比 PushT 的连续接触动力学简单。视觉结果不直接支持 LeWM replacement、CEM ranking 或 closed-loop 成功。

## 视觉结果：失败点在哪里

| Seed | GNN h1 H@1 | GNN h10 H@1 | Persistence h10 | GNN 增益 | Reference adequate |
|---|---:|---:|---:|---:|---|
| 1101 | 1.0000 | 0.7578 | 0.7188 | +0.0391 | 否，未达到 +0.05 |
| 1102 | 1.0000 | 1.0000 | 0.6992 | +0.3008 | 是 |
| 1103 | 1.0000 | 1.0000 | 0.7188 | +0.2813 | 是 |

失败的是 **seed 1101 的 GNN reference 对 persistence 的增益**，不是 encoder 坍塌的证据。三个 encoder 的 test latent 标准差均非零，没有低于 1e-3 的 latent dimensions；train-fitted position probe 的 test RMSE 分别约 0.022、0.114、0.020 个格子。1101 的 Transformer h10/h20 H@1 都是 1.0，说明同一表示可以支持良好预测。

更直接的行为诊断是：1101 GNN 在初始自由移动样本的 counterfactual action-response 中，预测 RMS 幅度只有 0.0090，真实值为 0.1139，约为真实幅度的 8%。1102/1103 对应为 0.0691/0.0742、0.1185/0.1182。这说明第一个 GNN checkpoint 对动作反应不足；本轮没有进一步实验定位 optimizer、架构或预算的具体原因。

下面是 seed 1101 的一个输入与五个 encoder maps。各 map 在不同物体的格子上显示差异，但一张例图不能证明整个表示具有排他的物体语义，颜色深浅也不是已校准的物体概率。

![输入与五个 encoder maps](D:/Downloads/Final%20Year%20Project/experiment/idea-validation/cswm-block-local-pilot/results/25564986.pbs101/encoder_slots_seed1101.png)

## 少量通信：有局部线索，主指标未显示稳定收益

| Seed | Global4 h10 H@1 | Local4 h10 H@1 | 差值 | Episode paired 95% CI |
|---|---:|---:|---:|---|
| 1101 | 0.9922 | 0.9883 | +0.0039 | [0, 0.0117] |
| 1102 | 0.9922 | 0.9922 | 0 | [0, 0] |
| 1103 | 0.9922 | 0.9922 | 0 | [-0.0117, 0.0117] |

所有区间都包含 0，所以本轮主指标没有支持稳定的 global4-vs-local4 通信收益。不要把三次训练的平均数或某个次要指标改成新的成功门槛。

不过，单步物体阻挡事件提供了与机制一致的线索：

| Seed | Local4 阻挡 MSE | Global4 阻挡 MSE | 误差降低约 |
|---|---:|---:|---:|
| 1101 | 0.003355 | 0.002809 | 16.3% |
| 1102 | 0.001343 | 0.001173 | 12.7% |
| 1103 | 0.003409 | 0.002851 | 16.4% |

这三组均使用同一批 1478 个 held-out 阻挡转移。边界阻挡只需要自己的位置，两种局部模型的误差都接近零；其他物体阻挡需要邻居信息，global4 的误差较低。但它仍明显高于 Transformer 的阻挡误差；这些是次要指标的描述性观察，没有单独的 bootstrap CI。

Persistence 的 h1 H@1 在三个 seeds 中均为 100%，h10 已约为 70%，小模型的 h10 又接近 99%，提示当前以其他随机场景作候选的检索评价不够敏感。一个模型即使没有精确处理阻挡，仍可能从随机场景中选对对应场景。Rollout 子集也仅按 step-0 动作类别划分，不能把该子集的高 h10 H@1 解读为十步内所有阻挡都预测正确。

Global16 没有展示明显的 h10 优势，并同时增加消息宽度和参数量，因此不证明 16D 是必要或最小的通信量。Object slots 是否适合真实 LeWM 表示仍未测试。

## 更快：这次确实测到了，但不是分块的独有收益

以下范围覆盖三个固定 seeds；数值为完整 h10 predictor rollout 的 median，单位 ms。所有后端在同一 A100 allocation、float32、TF32 off 下测量，20 次 warmup、60 次 repeats；encoder 初始成本另列在原始 summary。

| Predictor | 参数量 | B=1 | B=300 |
|---|---:|---:|---:|
| Global4 | 22,096 | 2.69–3.15 | 2.82–2.91 |
| Flat MLP matched | 22,075 | 2.24–2.28 | 2.38–2.61 |
| Full GNN | 75,408 | 4.55–5.38 | 4.71–5.56 |
| 六层 Transformer | 1,195,024 | 13.79–15.33 | 14.59–17.23 |

Global4 相对本接口的六层 Transformer，延迟降低约 78–83%。因此，第一轮相对 tiny dense MLP 的失败，确实不能外推成“相对 heavy predictor 也不会更快”。

但参数匹配 flat MLP 在三个 seeds 和两种 batch 下都更快，h10 H@1 分别为 0.9883、0.9961、0.9922，与 global4 接近。当前数据没有证明分块是加速的必要条件，也不能把所有收益归因于分块。Global4 的 h20 H@1 和 h10 latent MSE 比 flat 更好，属于值得保留的次要观察；它没有改变冻结的主指标与整体判定。

Transformer 的 latent MSE、阻挡误差和动作响应通常更精确，所以近似相同的 H@1 不等于完全相同的动力学质量。DECISION 中单个 arm 的 `quality_and_speed_promising` 字段仅表示相对 GNN 的两个数值条件通过；在整个 reference adequacy 未通过时，不能将其读作已验证的替代方案。

易懂的逐项解释见 [视觉阅读报告](../cswm-block-local-pilot/RESULTS.zh.md)。完整 per-seed 表、MRR、h1/5/10/20、事件子集、action-response、median/p90 和 CI 见 [视觉原始报告](../cswm-block-local-pilot/results/25564986.pbs101/REPORT.zh.md)。

## 对第一轮猜想的更新

两条实验把问题收紧为：**先找到适合局部演进的表示，再保留足够的跨块信息通路，同时用能区分动作后果的评价判断精度。**

- “正确表示可以让局部 predictor 更容易工作”：toy oracle 给出了明确支持；视觉 encoder 也形成了可读出位置的 slots，但这是较有利的对象环境，不是 LeWM latent 的自动分解。
- “少量跨块信息就足够”：rank-4 toy 支持；视觉中出现阻挡误差改善，但 primary H@1 增益很小，尚未验证这一结论在真实视觉 dynamics 中普遍成立。
- “这样可能更快”：相对本接口的 heavy Transformer 已有计时支持；matched flat 的竞争结果表明，简单小模型本身也是解释，不能把收益独占给分块。

本轮结果足以指导下一个有边界的实验，但不是 LeWM/CEM 的通过证明。后续应提高同场景不同动作后果、阻挡事件和位置误差的评价辨识力，并与实际 LeWM ViT 及小 flat predictor 做相同表示接口下的对照；这些后续实验没有在本轮自动开展。也不需要把所有 toy 问题解决，或把 C-SWM 优化到所有指标完美，才允许进入 LeWM 的小规模 predictor 对照。

## 交付与核对范围

Oracle 的 30 条 records、60 个配对对照、39 个对应 checkpoint 的原结果复现均已核对；全部复现最大绝对差为 0，固定坐标训练后偏差为 0。视觉的 3+21 fits 均为 5000 updates、batch 128，配对初始化 seed 一致；四个 horizons 的数值有限，三个逐 episode NPZ 均包含 8 个 arms（含 persistence）×256 episodes×4 horizons，以及 7 个 predictors 的 action-response 数据。

CPU 的 7 项环境检查、oracle 的 4 项机制检查、视觉的 7 项模型检查及 allocation/config 检查均通过。两项 GPU jobs 都正常退出；视觉 job.log 有 50 条按 30 秒采样的 GPU 利用率/显存记录。全部负结果与 inconclusive 标签保留；未增加训练、改 checkpoint 选择、改门槛、运行 CEM 或进行闭环控制。

本地保存摘要、报告、视觉逐 episode arrays 与例图；模型、训练记录、完整逐 episode 结果与 source snapshots 保留在各 PBS run 目录，视觉数据保留在 prepared 目录，toy 数据可按冻结生成器与 seeds 重建。报告数值以各目录的原始 summary/DECISION 为准。

机制来源：[C-SWM 论文](https://arxiv.org/abs/1911.12247v2)与固定 commit 的官方实现；方法来源：Kassis et al. (2026), *Scientific Agent Skills*, https://doi.org/10.48550/arXiv.2609.00065。现代兼容差异已记录在 FREEZE/PROTOCOL 中，不声称逐配置复现原论文。
