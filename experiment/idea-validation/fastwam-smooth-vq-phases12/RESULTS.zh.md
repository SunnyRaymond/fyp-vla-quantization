# Fast-WAM Smooth + Hadamard → Weight VQ：两阶段结果

2026-10-08 · 协议 `fastwam-smooth-vq-phases12-v1` · Fast-WAM released Optional-IDM clean checkpoint

**本轮支持“先改善 activation，再压缩 weights”的前半段，但不支持“越激进地迁移困难，当前 VQ 就越能承受”的后半段。** 在 10 个 held-out 固定输入、每个两个 sampler seeds 上，BF16 weights/A4 的 motor RMSE 从 0.33724 降到 0.04529；同一个 Smooth α=0.5 + Hadamard 变换下，Scalar W4A4 为 0.05745，当前 additive VQ+A4 为 0.07831，VQ 在 10/10 个输入上都更差。

这不证明 VQ 这类方法无效。它说明本轮四维、两本 codebook、每个 Linear 独立拟合的简单实现，尚未比 Scalar 更好地接住迁移到 weights 的困难。尤其是 Action expert，当前 VQ 的普通 weight 重建误差普遍较高，需要先改进这一层，才能判断 action-sensitive protection 能带来多少额外收益。

## 1. 实际比较了什么

| 项目 | 本轮设置 |
|---|---|
| 模型 | released Optional-IDM clean checkpoint，IDM，Video/Action 各 10 个 denoising steps |
| 模型输出 | action chunk `[32,7]`，无 action ensembler |
| 固定观察 | 22 个：10 个 Original LIBERO-Spatial task，12 个 Plus variant，覆盖六类扰动、每类两个 |
| Calibration | case `[0,2,4,6,10,12,14,16]`，每个仅第一 sampler seed，共 8 contexts |
| Selection | case `[1,3,18,20]`，每个两个 seeds，共 8 contexts |
| Held-out test | case `[5,7,8,9,11,13,15,17,19,21]`，每个两个 seeds，共 20 contexts、10 cases |
| 目标 weights | 614 个 Linear，6,016,317,440 个参数；Video 306、Action 307、root proprio_encoder 1 |
| 保持原精度 | KV、bias、normalization、nonlinearities、text encoder、VAE、其他非目标模块 |
| 模型查询 | 444 次：calibration 56、selection 168、test 220，包含所有配置和对照 |
| 闭环评估 | 0 episodes；没有执行预测动作，查询后没有环境步进 |

这些固定观察来自 initial state 0、30 次 no-op 后的状态。两个 seeds 是同一个输入内的配对重复，不能当作两个独立 task。Test 中只有四个 Original tasks 和六个 Plus variants；本轮没有 SensorNoise，也不是完整 LIBERO-Plus 的 10,030 variants 评估。

主要指标是 normalized action 前 10 步、前 6 个 motor 坐标相对**同一 case、同一 seed 的原始 BF16 输出**的 RMSE：

\[
e_{c,s}=\sqrt{\frac{1}{60}\sum_{t=1}^{10}\sum_{j=1}^{6}
(a^{Q}_{c,s,t,j}-a^{BF16}_{c,s,t,j})^2},\qquad
\bar e=\frac{1}{|C|}\sum_{c\in C}\frac{e_{c,0}+e_{c,1}}{2}.
\]

先对每个 seed 算 RMSE，再对一个 case 的两个 RMSE 求均值，最后跨 case 求均值。Gripper 单独报告。数字越小，表示越接近 BF16；这不是任务成功率，也不代表与真实正确动作的误差。

## 2. Smooth 和 Hadamard 在这里分别做什么

设一层为 `y = x Wᵀ + b`，`x` 是 row vector，`W` 的形状是 `[out_features,in_features]`。

**Smooth 是逐通道交换幅度。** 若某个 activation 通道经常特别大，就把这个通道除以较大的 `S_i`，同时把对应 weight 列乘以 `S_i`。例如 activation 从 100 变成 10，weight 从 0.02 变成 0.2，乘积仍是 2。困难转移了，浮点函数没有改变。

本轮依据 calibration 的每通道 activation absmax `a_i` 与 weight 列 absmax `w_i`，设置：

\[
\widetilde S_i=\frac{a_i^\alpha}{w_i^{1-\alpha}},\qquad
S_i=\operatorname{clip}\left(
\frac{\widetilde S_i}{\operatorname{geomean}(\widetilde S)},\frac1{16},16\right).
\]

α 越大，缩放规则越偏向 activation 的范围，weight 范围的牵制越弱。α=1 时完全依据 activation absmax 决定相对缩放。**α 是候选强度，不是“强度越大、精度越好”的承诺**；截断和 Hadamard 混合以后，各项误差仍须实际测量。

**Hadamard 是混合坐标。** 它用加减和归一化，把少数特别大的通道分散到多个通道。举一个四维例子，`[100,0,0,0]` 可变成 `[50,50,50,50]`：总平方能量不变，最大坐标变小。对所有通道共用一个量化范围的 A4，这常常能让更多小值获得有效刻度。它不能保证所有输入都变得均匀，也不能消除 outlier 的能量。

本轮使用固定随机符号与 normalized block-Hadamard，block size 为 128；尾部按 2 的幂拆分。设其正交矩阵为 `R`：

\[
z=(xS^{-1})R,\qquad B=(WS)R,\qquad
zB^\top=xW^\top.
\]

因此可以离线改 weights，在每个目标 Linear 的输入处变换 activation，不需要改整个模型的残差坐标。浮点数学等价，BF16 的舍入仍可能产生漂移，所以本轮专门测了未量化的变换对照。

变换后 A4 采用 dynamic per-row absmax、signed symmetric Q4，`qmax=7`，round/clamp 后 decode 为 BF16。本轮没有 W4A8 实验。

## 3. 第一阶段：activation 确实被改善，但激进程度不宜单向增加

以下是 **selection 的 4 cases × 2 seeds**，所有 weights 保持 BF16，仅 activation 用 A4。S0.5H 表示 Smooth α=0.5 + Hadamard。

| 变换 | Motor RMSE ↓ | Activation relative RMSE ↓ | A4 zero fraction | 最大 weight peak ratio | BF16 变换漂移 ↓ |
|---|---:|---:|---:|---:|---:|
| Identity | 0.32662 | 0.49713 | 69.48% | 1.00× | 0 |
| Hadamard-only | 0.04519 | 0.19088 | 31.02% | 2.69× | 0.00151 |
| **S0.5H** | **0.04166** | 0.17429 | 28.94% | 4.68× | 0.00207 |
| S0.75H | 0.04623 | 0.17093 | 28.23% | 10.48× | 0.00198 |
| S1H | 0.05521 | 0.17214 | 28.34% | 16.27× | 0.00217 |
| Smooth α=1 only | 0.31092 | 0.39053 | 55.59% | 16.00× | 0.00166 |

Weight peak ratio 是每个 Linear 的 `max(abs(B))/max(abs(W))`，表中取 614 层的最大值；它不是所有层的平均恶化程度。Activation 指标按观测到的所有目标 Linear 调用、所有元素汇总，且在各自的**变换后坐标**计算。不同 `S` 改变其分母，不能把 relative RMSE 当作最终 action sensitivity。

读这张表可以得到三个结论：

1. **Hadamard-only 已提供 selection 上的大部分收益。** 加适中的 Smooth 又改善了一些；Smooth-only 的 motor 收益很小，gripper RMSE 还从 Identity 的 0.24130 变成 0.66494。
2. **更小的 activation 重建误差不一定给出更小的 action 误差。** S0.75H 的 activation relative RMSE 比 S0.5H 小，但 motor RMSE 更高。
3. **把 weight peak 放大到约 16 倍没有带来更好的 action fidelity。** 当前数据反对把“变换越剧烈越好”作为默认规则；应该用选择集上的最终 action 指标决定迁移程度。

按照冻结的规则，第一阶段选中 S0.5H。其 **held-out 10 cases × 2 seeds** 结果如下：

| BF16 weights / A4 | Motor RMSE ↓ | Gripper RMSE ↓ | Activation relative RMSE ↓ | A4 zero fraction |
|---|---:|---:|---:|---:|
| Identity | 0.33724 | 0.17309 | 0.49800 | 69.40% |
| S0.5H | **0.04529** | **0.01595** | **0.17425** | **28.93%** |

Motor RMSE 下降约 **86.6%**。S0.5H 的未量化 BF16 对照漂移为 **0.00198**，最大 case 漂移为 0.00286；A4 相对同变换 BF16 对照的 RMSE 为 0.04541，与相对原 BF16 的 0.04529 接近。改善不能用“只是改变了浮点答案”解释。

Hadamard-only 没有进入第一阶段的 held-out test，不能据此给出它的 held-out 数字，也不能确认它在 test 上能保留多少上述收益。

## 4. 第二阶段：当前 codebook 是怎样建立的

Scalar 使用每个 output row 内、input group size 128 的 signed symmetric W4。每组保存 BF16 scale 和 packed nibble codes；生成 codes 前先把 scale 舍入到实际保存的 BF16。

VQ 把每个 output row 的连续四个 input weights 当作一个向量 `v_g`，用两个四维 codeword 相加重建：

\[
\widehat v_g=c^{(0)}_{k_{g,0}}+c^{(1)}_{k_{g,1}},\qquad k_{g,0},k_{g,1}\in\{0,\ldots,255\}.
\]

每个向量保存两个 uint8 indices，即 16 bits / 4 weights = **4 bits/weight**，再加 codebook 等开销。每个 Linear 独立建立两本 `[256,4]` BF16 codebook，没有跨 Video/Action expert 共用。

拟合过程为：从该层采样最多 8,192 个四维向量，先拟合第一本，再拟合残差；两本交替更新 8 轮；保存 BF16 codebooks 后对全层分配 indices，再交替优化 indices 两轮。本轮没有 per-output-row scale normalization，幅度不同的 rows 需要使用同一层的两本 codebook。

距离并非完全普通的 Euclidean distance。先在 calibration 上收集该候选变换实际 **decoded Q4 输入**的 second moment `h_i = E[z_Q,i²]`，拟合近似目标：

\[
L_{local}=\sum_{o,i} h_i(B_{o,i}-\widehat B_{o,i})^2.
\]

直觉是：“经常被激活的输入通道，weight 错一点更容易造成 Linear 输出误差，因此给它更高代价。”这只是丢弃通道相关性后的 **local Linear-output proxy**；它没有测量某方向经过后续 attention、denoising 和 decoder 后对最终 action 的影响。

Selection 仍然使用最终 action RMSE，结果如下：

| 变换 | Scalar W4/A4 ↓ | Additive VQ/A4 ↓ |
|---|---:|---:|
| Identity | 0.33452 | 0.32750 |
| S0.5H | **0.05049** | 0.08072 |
| S0.75H | 0.05220 | 0.07663 |
| S1H | 0.07233 | **0.07404** |

Scalar 选中 S0.5H，VQ 选中 S1H。三项 winners 都在 test 前冻结，`test_queries_before_freeze=0`。Test 按原协议评估 Identity 与两项 weight-quantizer winners 的变换并集，从而获得同变换比较。

## 5. Held-out：VQ 目前没有胜过 Scalar

以下每一行都是同一组 **10 cases × 2 seeds**。Median/Max 先在每个 case 内平均两个 seeds，再跨 case 统计。

| 变换 / weight precision，activation 均为 A4 | Mean motor RMSE ↓ | Median case ↓ | Max case ↓ | Mean gripper RMSE ↓ |
|---|---:|---:|---:|---:|
| Identity / BF16 | 0.33724 | 0.33779 | 0.37412 | 0.17309 |
| S0.5H / BF16，第一阶段 winner | **0.04529** | 0.04211 | 0.07115 | 0.01595 |
| Identity / Scalar W4 | 0.34224 | 0.34711 | 0.38514 | 0.13843 |
| Identity / VQ | 0.34356 | 0.34903 | 0.37645 | 0.19521 |
| S0.5H / Scalar W4，Scalar winner | **0.05745** | 0.04923 | 0.10381 | 0.01621 |
| S0.5H / VQ，同变换诊断 | 0.07831 | 0.06533 | 0.15937 | 0.04101 |
| S1H / Scalar W4，同变换诊断 | 0.06820 | 0.05732 | 0.16063 | 0.01750 |
| S1H / VQ，**冻结的 VQ winner** | 0.10409 | 0.06590 | 0.31392 | 0.04372 |

在相同变换下，S0.5H 的 VQ 平均 motor RMSE 比 Scalar 高 **36.3%**，且在 **10/10 cases** 上更高；S1H 的 VQ 平均值高 **52.6%**，在 **9/10 cases** 上更高。Identity 下两者接近，符合 activation A4 失真很大的情形，但仅凭总 RMSE 不能做严格的权重/激活因果分解。

VQ 在 selection 上偏好 α=1，在 test 上出现较大尾部误差，说明这 4 个 selection cases 对当前 VQ 配方的泛化把握有限。**不能看过 test 后把 α=0.5 改称 VQ 的预先选中配置。** S0.5H/VQ 只是按既定并集规则得到的同变换诊断，正式冻结的 VQ pipeline 结果是 S1H 的 0.10409。

逐 case 的 motor RMSE 如下，均是两个 seeds 的均值：

| Case | 数据类别 | BF16/A4 Identity | BF16/A4 S0.5H | Scalar S0.5H | VQ S0.5H | Scalar S1H | VQ S1H |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5 | Original | 0.34114 | 0.03864 | 0.04429 | 0.05324 | 0.04473 | 0.05773 |
| 7 | Original | 0.35636 | 0.03899 | 0.04625 | 0.06407 | 0.05059 | 0.05808 |
| 8 | Original | 0.32362 | 0.04313 | 0.04288 | 0.06905 | 0.06512 | 0.07084 |
| 9 | Original | 0.37412 | 0.05069 | 0.05309 | 0.06660 | 0.05747 | 0.07232 |
| 11 | Plus objects_layout，2106 | 0.33444 | 0.04218 | 0.06815 | 0.07644 | 0.06242 | 0.11974 |
| 13 | Plus camera_viewpoints，0620 | 0.26900 | 0.07115 | 0.08019 | 0.12890 | 0.08811 | **0.31392** |
| 15 | Plus robot_initial_states，0600 | 0.36808 | 0.04116 | 0.05222 | 0.05554 | 0.05717 | 0.05681 |
| 17 | Plus language_instructions，1150 | 0.33122 | 0.04204 | 0.04225 | 0.05143 | 0.04919 | 0.05618 |
| 19 | Plus light_conditions，2132 | 0.34851 | 0.03649 | 0.04134 | 0.05848 | 0.04658 | 0.06095 |
| 21 | Plus background_textures，0245 | 0.32592 | 0.04843 | 0.10381 | 0.15937 | 0.16063 | **0.17434** |

Case 13 的 camera variant 是 S1H/VQ 的主要尾部问题，但同变换 Scalar 为 0.08811；这个差异提示当前 VQ 配方值得检查。它没有单独定位到某层或敏感方向，也不代表所有 camera perturbations 都会这样。

## 6. 当前瓶颈先落在 codebook 的局部质量上

下表比较相同变换下，每个 Linear 的普通 weight 重建 RMSE。`VQ/Scalar` 是每层误差比，再取层间中位数；小于 1 表示 VQ 更好。

| 变换 | 全部 614 层中 VQ 更差 | 全部层比值中位数 | Video 306 层中 VQ 更差 / 中位比值 | Action 307 层中 VQ 更差 / 中位比值 |
|---|---:|---:|---:|---:|
| Identity | 600/614 | 1.4741 | 293/306 · 1.2457 | 306/307 · 1.8213 |
| S0.5H | 442/614 | 1.0680 | 136/306 · 0.9929 | **305/307 · 1.1796** |
| S0.75H | 471/614 | 1.1011 | 165/306 · 1.0020 | 305/307 · 1.2158 |
| S1H | 508/614 | 1.1453 | 203/306 · 1.0227 | 305/307 · 1.2656 |

所有层汇总包括那一层 root proprio_encoder；Video/Action 分组只包括各自 expert。

**S0.5H 下，Video 侧普通 weight RMSE 大致接近，Action 侧 VQ 则普遍更差。** 因此现在不能把“VQ 没有赢”全归因于“某些非常敏感的方向没有被保护”。基础 codebook 的幅度适配、分组和拟合质量，也还没有达到足够强的对照水平。

这张表是各层等权的、未按 activation 加权的 weight RMSE，既不是按参数数量加权的全模型误差，也不是 action 误差。VQ 拟合采用上一节的 weighted objective，Scalar 在同一 weighted objective 上的误差尚未汇总；表中结果不能证明 VQ 的训练没有收敛，更不能单独建立最终 action 误差的因果来源。

## 7. 存储是真的，但运行仍是 BF16 reference

各变换的保存格式成本相同，614 层目标 weights 的统计如下：

| 项目 | Scalar W4 | Additive VQ |
|---|---:|---:|
| Encoded weight payload | 3,102,173,296 bytes | 3,010,674,176 bytes |
| Transform tensor payload | 9,034,315 bytes | 9,034,315 bytes |
| **含 transform 的有效 bits/weight** | **4.1370** | **4.0154** |
| 实际 serialized module files 总和 | 3,112,661,146 bytes | 3,021,279,898 bytes |

有效 bits/weight 由 `(encoded weight payload + transform tensor payload) × 8 / target weight count` 计算，不含文件容器成本。两本 codebook 每层共 4,096 bytes。VQ 的 encoded weight payload 比 Scalar 小约 2.95%，本轮也确实生成了可保存的 indices/codebooks。

推理时这些表示 decode 成 BF16 weights，在 GPU 上执行 BF16 Linear。**这不是整模型 4 bits/weight，也不是低位部署显存或 native kernel 加速结果。** KV、text encoder、VAE 等不在上述分母中。

正式作业 walltime 为 4:18:12，包含所有 444 次查询、拟合和诊断；不能当作一次 action query 的 latency。GPU log 有 1,023 次采样，峰值为 27,589/40,960 MiB；平均利用率约 40.2%，包含初始化、拟合和 instrumentation，同样不用于性能结论。

## 8. 对后续 codebook 与敏感度实验的建议

**先固定中等变换，建立更强的 VQ 对照。** 最小的下一项改动是：以 S0.5H 为起点，为每个 output row 增加一个 BF16 scale，先消除 row 间幅度差，再拟合 additive codebooks。当前 Scalar 已有 G128 scale，而 VQ 没有 row scale；这种不对称值得优先检查。每 row 一个 BF16 scale 的成本是 `16/in_features` bits/weight，例如 input width=1024 时为 0.015625 bits/weight，仍须把实际总成本报出。

先只改这一项，比较：相同 activation-weighted objective、Action/Video 各层重建、最终 action RMSE，以及实际 bits/weight。如果它还不足，再评估按 head/block 或分布聚类建立多个 codebooks。当前已经每层独立，单纯“让 Video/Action 分开建码本”并不是新改动；进一步分割需计入额外 codebook 和分区 metadata。

**随后才测最终 action sensitivity。** 当前 `h_i=E[z_Q,i²]` 不具备用户提出的“第一方向对 action 比第二方向重要 100 倍”的含义。一个可检验的定义是，在固定 case/seed/denoising context 上，设局部扰动 `δu` 到最终 action 的 Jacobian 为 `J`，以 action 各坐标归一化权重 `D` 定义：

\[
M=\mathbb E[J^\top D J],\qquad
\Delta L_{action}\approx\delta u^\top M\delta u.
\]

方向 `v` 的敏感度可定义为 `vᵀMv / ||v||²`。若两方向在**相同扰动能量**下的该值相差 100 倍，线性近似的 action squared error 就相差 100 倍，RMSE 则约相差 10 倍。必须先统一扰动范数和 action 坐标单位，才有可比较的“100 倍”。可以再用小扰动的配对有限差分检验 Jacobian 近似，而不是直接把通道 absmax 当敏感度。

保护策略才相应选择高敏感子空间的额外 residual、独立 codebook 或更高精度，并在同一实际 bit budget 下比较。Hadamard 会旋转方向，Smooth 会改变尺度，敏感度矩阵也必须跟着变换；不能直接套用原坐标下的排行。

本轮没有实现这一 action-Jacobian metric、低秩保护或 phase3，也没有完成该具体方法的 novelty audit。以上是下一阶段的定义与实验建议，不是本轮已证实的结果。当前 test 已经被分析；修改方法后，需要新的独立 held-out 输入，不能把 case 13/21 用于调参后继续称它们为 blind test。

## 9. 如何迁移到另外两个 WAM baseline

以下是基于固定版本论文的**架构推断**，尚无其他 baseline 的本地数值实验。

### Cosmos Policy：共享 DiT 要兼顾不同语义的 latent

Cosmos Policy 把 actions、proprio、future state 和 value 编码为 latent frames，沿用 video diffusion backbone。这使同一个 backbone 同时承担视觉、动作和规划相关输出。[Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning，arXiv v1](https://arxiv.org/abs/2601.16163v1)，[官方实现](https://github.com/NVlabs/cosmos-policy)。

上述每个 Linear 输入处的 `S/R` 变换与离线 weight reparameterization 仍适用。迁移时需让 calibration 覆盖不同 latent 类型和 diffusion steps，避免只按大量 video tokens 的统计决定 scale/codebook。共享 weights 保存一份；评估要包括最终 decoded action，若用于 planning，还要单独检查 value/ranking，而不能只用 video reconstruction 选择变换。

### LingBot-VA：双流架构适合分别控制迁移与误差

LingBot-VA 使用 vision/action 的共享因果建模框架和 Mixture-of-Transformers，保留 video/action streams 的参数与跨模态交互。[Causal World Modeling for Robot Control，arXiv v2](https://arxiv.org/abs/2601.21998v2)，[官方实现](https://github.com/Robbyant/lingbot-va)。

迁移时可分别校准 video、action 和跨模态映射的 Linears，分别建立 codebooks，再用共同的最终 action 指标比较。Fast-WAM 本轮结果提示不要假设两个 expert 都适合相同 α 或同样的 VQ 配方；这个提示尚需在 LingBot-VA 上验证。Causal context 与具体生成步应纳入 calibration，KV 量化则是另一个独立变量。

两种架构都支持这一局部代数变换，并不意味着它们会复现 Fast-WAM 的 α=0.5、86.6% RMSE 降幅或 Scalar/VQ 排序。量化收益仍需各自的 checkpoint、输入协议与最终输出指标验证。

## 10. 完成证据与可复查文件

| 作业 | 用途 | 终态 | Walltime |
|---|---|---|---|
| 25721014.pbs101 | Preflight，21 queries | F，Exit_status=0 | 00:33:23 |
| 25722477.pbs101 | Full 两阶段，444 queries | F，Exit_status=0 | 04:18:12 |
| 25725256.pbs101 | Compute allocation 内的 CPU 汇总 | F，Exit_status=0 | 00:00:03 |

已确认 scheduler 终态、Exit_status=0、pipeline completion marker、exit_code.txt=0 与非空 summary。CPU 汇总从原始 444 条 query 重现所有 primary means，确认每个声明的 case/seed/config context 完整且只出现一次，并核对 4,912 条 module-bank 记录与存储汇总。模型计算、拟合、数值自检和汇总均在获批 PBS compute allocation 内完成。

- [冻结协议](protocol.json) 与 [22-case 输入计划](../fastwam-a4-phases12/plan.json)。
- [正式结果 summary](results/full/summary.json) 与 [test 前冻结的 winners](results/full/frozen_winners.json)。
- [CPU 分析 summary](results/analysis/summary.json) 与 [正式作业完成日志尾部](results/analysis/full_log_tail.txt)。
- [Full scheduler evidence](full_terminal_evidence.json) 与 [Analysis scheduler evidence](analysis_terminal_evidence.json)。
- Remote full outputs：`/scratch/users/ntu/yguo017/fastwam-smooth-vq-phases12-20261006/results/full/`，原始 `queries.jsonl`、`module_bank_stats.jsonl` 与完整 job log 保留在集群。

本轮完成的是 **Fast-WAM 的 PTQ、固定输入、A4 两阶段数值验证**。它不包含 W4A8、QAT、KV 量化、最终 action sensitivity protection、其他 baseline 数值实验、native low-bit kernel 或闭环任务成功率。
