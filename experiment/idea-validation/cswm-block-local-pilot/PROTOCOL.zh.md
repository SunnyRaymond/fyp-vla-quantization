# C-SWM 视觉分块动力学 pilot：冻结协议

本实验与已知正确坐标的 toy oracle 对照并行。toy 结果不会决定是否启动或解释性删减本实验。这里的目标是检验：在 C-SWM 的图像输入和 object-slot 表示上，局部 predictor 加少量跨 slot 信息，能否保留多步预测质量，并比完整 GNN 与六层 Transformer 更快。

## 代码来源与环境

官方来源固定为 [`tkipf/c-swm`](https://github.com/tkipf/c-swm)，commit `e944b24bcaa42d9ee847f30163437a50f0237aa0`。CPU prepare job 将在目标 commit 下载并保留 `envs/block_pushing.py`、`modules.py`、`utils.py`、`data_gen/env.py` 和 `LICENSE`。实现参考了官方 2D Shapes 的 5×5 网格、5 个对象、50×50 RGB 图像，动作 `object_id * 4 + direction_id`，以及 10×10 stride CNN object extractor、shared object MLP 和 fully-connected GNN。

这是一项机制型 pilot，并非论文设置的逐项复现。官方仓库依赖旧版 Gym / scikit-image 接口，并混合使用全局 NumPy 与 Gym RNG；本实验采用 standalone 环境和整数像素兼容 renderer，以固定 episode seeds。它保留对象形状、颜色次序、动作映射、边界限制和占位碰撞语义，但不声称跨历史依赖版本逐像素相同。代码差异会随 run 输出记录。

## 环境与数据

环境有 5 个有固定 object ID 的圆形、三角形和正方形对象，位于 5×5 网格，每格 10 像素。观测为 RGB `[3, 50, 50]`。每步均匀采样 20 个动作之一；尝试越界或进入其他对象所在格时，状态保持不变。每个 episode 恰好 40 步。

数据按**完整 episode**切分：train 1000、dev 128、test 256。三个 split 使用冻结的独立 seed `20260926/20260927/20260928`，模型 seeds 为 `1101/1102/1103`。数据准备先生成 uint8 observation、action、position 和 event 数组；position/event 只供指定的位置 probe 和 held-out 诊断。Stage A、Stage B 的训练代码只读取图像与 action，不把坐标、阻挡类别或网格状态提供给 encoder 或 predictor。

`event` 的编码为 `0=free`、`1=other-object-blocked`、`2=boundary-blocked`。测试诊断按存储的 held-out 状态和尝试动作分类，不按结果挑选测试 episode。单步子集指标使用全部测试 transitions；rollout 指标按该 episode 第 0 步动作的类别分层。

## Stage A：固定每个 seed 的视觉表示

每个 seed 独立训练一次 C-SWM-style encoder 与 full-pair GNN，共 3 fits。Encoder 使用官方 `small` 结构：10×10、stride-10 的 CNN feature maps，为 5 个 slot 提取特征，再经共享 MLP 得到每个 slot 16D latent；hidden size 为 128。GNN 使用全连接有向 pair messages 和 node aggregation，hidden size 128。

训练只用 train 图像与动作，采用官方形式的正转移能量加 margin contrastive negative energy；`sigma=0.5`、`hinge=1.0`。每个 fit 固定 5000 updates、batch 128、Adam、learning rate `1e-3`，最后一个 checkpoint 是唯一 checkpoint。dev/test 不用于 early stopping 或挑选 checkpoint。

每个 seed 的 Stage A encoder 在该 seed 全部 Stage B arms 中冻结并共享。由于它与 full GNN 联合训练，表示可能对 GNN 有偏置；这有利于解释机制，但会让本 pilot 只回答“已有 object-slot interface 是否有利”，不能单独证明 LeWM latent 能被同样分解。Stage B 不回传梯度，不为某个 arm 单独重训 encoder；缓存 latent 使用可参与 predictor backward 的普通 no-grad tensors。

## Stage B：比较 7 种 predictor

每个 Stage A encoder 下从头训练以下 7 个 predictor，5000 updates、batch 128、Adam、`1e-3`，使用最后 checkpoint。总计 3 Stage A fits + 21 Stage B fits = **24 fits**。

| Arm | 计算结构 | 回答的问题 |
|---|---|---|
| `full_gnn` | C-SWM-style fully connected pair-message GNN，hidden 128 | object-centric 视觉 reference 是否学得足够好 |
| `local` | 跨 slot 共享 MLP；每个 slot 只读自己的 16D latent 和自己的 4D action | 纯局部更新损失多少交互信息 |
| `local4` | local predictor 加每个 slot 独立的 16→4 本地摘要 | 不交换信息但增加等量投影参数能否获得相同收益 |
| `global4` | 所有 80D slot latent 先经共享线性层压成 4D，再提供给每个 slot | 少量全局通信能否补足局部更新；主机制对比为 global4 vs local4 |
| `global16` | 同上，但摘要为 16D | 放宽通信带宽后的误差/速度边界 |
| `transformer6` | 6 层 Transformer encoder、4 heads、d_model 128、FFN 512 | 与共同 slot 接口上的较重序列模型比较 |
| `flat_mlp_matched` | 直接读取全部 slots 与 action 的 flat MLP | 与 global4 参数量匹配后，区分结构收益和单纯的小模型收益 |

每个 slot 的 action vector 都由同一整数动作精确展开为 4D one-hot：被选中的 object slot 收到方向 one-hot，其余 slot 收到零。所有 arms 因此拥有完全相同的 action 信息。`local4` 使用 5 个独立、无 bias 的 16→4 投影；`global4` 使用一个无 bias 的 80→4 投影。二者新增投影参数都为 320。`global16` 使用无 bias 80→16 投影。`flat_mlp_matched` 的 hidden width 在训练前固定，使 total trainable parameter count 与 `global4` 相差不超过 5%，并在 summary 中记录实测参数量。

同一 seed 的 7 个 arms 使用同一个固定 model-initialization seed；对 `local4` 与 `global4`，局部动力学函数和摘要投影从匹配初值开始。训练目标是：

`one-step latent MSE + 0.5 × mean free-running latent MSE at horizons 1..5`

两项均除以该 seed **train split** 上的平均 latent-delta 能量。每个 seed 预先生成的 episode/start-index minibatch draws 在 7 个 arms 间完全配对；arm 执行次序按冻结种子打乱，避免固定次序与热身混淆。模型不读取 GT 坐标、类别标签、未来图像或未来 latent。

## 评估与预注册判据

所有 arms 使用同一个 seed 的 frozen encoder 和同一 test split。主要 rollout 从每个测试 episode 的 step 0 开始，报告 `h=1,5,10,20` 的 latent MSE 与 C-SWM-style H@1 / MRR。另在全部测试 transitions 上报告 action-conditioned one-step error，并按 free、other-object-blocked、boundary-blocked 分层。

对于每个 horizon，候选 bank 包含一次真实 target latent，以及同一 seed 其他 held-out episodes 的 target latents。episode 集合与次序对该 seed 的全部 arms 和 persistence baseline 固定一致，不使用 GT 坐标筛选候选或测试样本。排名按 squared Euclidean latent distance；等距时固定采用官方 evaluator 的稳定正例优先规则，并同时报告 latent variance，以识别坍塌表示。`persistence` 直接预测 `z_t`，不训练参数，用完全相同的 horizons 和 candidate banks 评估。

Action-response 诊断从相同 held-out image/latent 出发，配对比较存储动作与**同一 object 的相反方向动作**（方向 index 加 2 后对 4 取余），比较模型预测的 latent 差分与 frozen encoder 对应真实后继图像的 latent 差分。该诊断只用于测试集分析，模型训练不接触这些反事实样本。

位置 probe 只作 encoder 诊断：在 train latent 和 train position 上拟合线性 slot-to-position readouts，并仅用 train 数据决定 slot/object 对应；冻结后报告 test position error。位置 probe 结果不进入 predictor 训练或模型选择。

每个 seed 的 C-SWM visual reference 只有同时满足以下条件时才称为 adequate：test `full_gnn H@1 >= 0.80` at h1、`>= 0.50` at h10，且 h10 至少比 persistence 高 `0.05`。三 seed 全通过才称整体视觉 baseline adequate。任一 seed 未过，仍跑完全部 21 个 predictor fits，结论标为视觉比较 inconclusive，不改变训练预算或 gate。

Quality gate：每 seed 的 h10 H@1 相比 `full_gnn` 最多下降 3 个百分点。Speed gate：预测器完整 h10 rollout（包括 message/token 操作）需分别比 `full_gnn` 和 `transformer6` 至少快 20%；B=1 与 B=300 分别报告。计时固定 warmup 20 次、重复 60 次，在同一 GPU allocation、相同 precision 和同步策略下进行；encoder 初始成本单独报告。只有相同 seed 和 batch 条件同时通过 quality 与 speed gate，才标记为有质量与速度潜力。

报告逐 seed 给出配对差异与 1000 次 episode-level paired bootstrap 95% CI；episode 是抽样单位，不把同一 episode 的帧当作独立样本，也不据 3 个训练 seeds 推断 seed population。保留所有负结果。测试集不调参、不选 checkpoint、不触发追加训练。GPU job 每 30 秒记录利用率与显存。

## 结果能支持什么

本 pilot 可以回答 object-slot 图像 latent 上的 predictor-level 质量、action response 和实测 latency 问题。它不能证明替换既有 LeWM checkpoint，也不覆盖 LeWM 的 history/action 接口、CEM candidate ranking、闭环任务成功或 PushT transfer。`transformer6` 是适配相同 slots 的新 comparator，不是 LeWM 六层 ViT checkpoint。

实验设计采用 episode 为独立抽样单位、按 seed 配对 predictor arms、在测试前冻结质量/速度门槛。程序化实验设计原则参考：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. https://doi.org/10.48550/arXiv.2609.00065

## 参考

- Kipf, T., van der Pol, E., & Welling, M. (2019). *Contrastive Learning of Structured World Models*. [arXiv:1911.12247](https://arxiv.org/abs/1911.12247); [official source at frozen commit](https://github.com/tkipf/c-swm/tree/e944b24bcaa42d9ee847f30163437a50f0237aa0).
- Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. [doi:10.48550/arXiv.2609.00065](https://doi.org/10.48550/arXiv.2609.00065).
