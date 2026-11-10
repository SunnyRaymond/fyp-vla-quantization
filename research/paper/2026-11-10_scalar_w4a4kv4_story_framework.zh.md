# 标量量化版的故事框架与论文框架（W4A4KV4，2026-11-10 第一版）

依据：Q-WAM 原文与阅读包、Q-WAM 与 SteerQuant 的 ICLR 2027 审稿意见（各三份）、QuantWAMs / SteerQuant / DSAQuant / QVGen 阅读包、本项目 10-28 的 W4A4 设计稿与 D1/D3 实测、LAVQ 阶段的 C1 测量、文献检索（见文末来源）。

状态标记：✅ 已有实测（注明作业）；⏳ 需要在 W4A4KV4 口径下重测或新测；❓ 需要你拍板。

---

## 0. 一页结论

**设定。** 主设定 W4A4KV4：所有目标 Linear 的权重 4 bit、输入 activation 4 bit、缓存的 K/V 4 bit。W4A4（缓存不量化）作为副列，用来和 Q-WAM、SteerQuant、QuantWAMs 的原始设定对齐。三篇 WAM 量化论文都没有量化缓存（SteerQuant 明写 KV 保持 BF16），所以 W4A4KV4 本身就是一个没人做过的设定。

**一句话。** WAM 的动作是从它自己写出来的 future 上读出来的。4 bit 下，现有方法守的是动作；future 从被写出来、存进缓存、到被动作读走，这一路没有人守。

**三条观察（动机，与现在的 C1 同源）。**

| | 观察 | 现有证据 |
|---|---|---|
| (a) | 把动作通路保护好之后，剩下的动作误差几乎全部经由 future 进来 | ✅ D3：全 A4 的 motor 0.0426；action 循环升 A8 后 0.0176，future 误差一点没变（0.0384）；全 A8 是 0.0028。保护动作通路之后，残余动作 MSE 的 97% 来自失真的 future |
| (b) | future 是在"写"的地方坏掉的，伤它最重的是"写一次、读很多次"的误差 | ✅ frame 0 的行在 10 个去噪步里逐位相同，它的 4-bit 舍入误差每一步都一样（D1：残差相关 1.000）；权重引起的 velocity 误差跨 10 步相干，和的能量是各步能量之和的 7.1 倍；逐步独立的一阶估计给 future 通路 6.9% 的动作敏感度，实测是 35–60% |
| (c) | 动作只读 future 很窄的一部分；读的"位置"稳定，读的"方向"不稳定 | ✅ reader 度量有效秩 41.5 / 37,632，98% 在第 1 个预测帧；对角（位置）对半相关 0.86，前 32 个方向对半重合只有 0.19（26073871） |

**方法：LAQ（Latent-Aware Quantization，暂名）。** 一个目标、一个预算旋钮、无训练：

1. **读出度量**：在 future 这个接口上，用"动作怎样读 future"来量误差（沿用 ① 的 reader 度量 M）。
2. **读次数**：每个误差源按"同一个误差实现被读了多少次"加权。动态 activation 每步是新的误差，读一次；权重、缓存、观测前缀是冻结的误差，被每一步重复读，损伤相干叠加。
3. **联合闭式分配**：W、A、KV 三类误差写进同一个目标，用一个拉格朗日乘子求解。权重一侧的解就是标准 GPTQ；A 和 KV 一侧的解是"哪些 token 值得多一遍 4-bit"。
4. **Gaze 引导的精修**：被选中的 token 在去噪循环里和缓存里各多做一遍 4-bit 残差量化（误差功率降到 1/196 以下）。选谁由 policy 自己的注意力给出，上一个控制步的注意力带到下一个控制步，换任务不用重新校准。全程只有 INT4 运算，权重严格 4 bit，没有高精度分支。

**和 Q-WAM 不撞车的地方。** Q-WAM 问"哪些 channel 方向伤动作"，在每层的 channel 空间里做特征分解，把前 32 个方向放进 16-bit 低秩分支。我们问"哪些 token 上的误差会被动作读到、被读多少次"，对象在 token 轴和缓存上，载体是选中 token 行上的第二遍 INT4，不存任何高精度权重。两者的数学对象、保护的轴、载体、额外存储都不同（§4.6 有逐项对照）。

**先跑什么。** 四个开环验证（§5），都能复用 D1/D3 的 eager fake-quant 管线，每臂约 10 分钟。其中 G3（同预算下 gaze 选 token 是否明显好于随机和"只按步保护"）决定载体是否成立。

---

## 1. 设定与定位

### 1.1 W4A4KV4 在 WAM 里指什么

WAM 的缓存不只是"过去的上下文"。以 Fast-WAM（IDM 模式）为例，一次 query 里有一条缓存链：

| 缓存 | 何时写 | 被谁读、读几次 | 依据 |
|---|---|---|---|
| 文本 cross-attention K/V | 每条指令一次 | 每层每步 | 引擎组件 B 已缓存 |
| 观测前缀：frame 0 的 98 个 token | 每次 query 一次 | 196 个 future token，在 10 个 video 去噪步和 1 次 prefill 里各读一次 | ✅ D1：frame 0 的行跨步 innovation 比为 0.000，说明它不依赖 future token 也不依赖时间步，是纯前缀 |
| future 缓存：prefill 写出的 video K/V | 每次 query 一次 | 32 个 action token，在 10 个 action 去噪步里各读一次 | 引擎里的 video_prefill（约 52 ms） |

所以在 WAM 里，KV4 量化的正是"现在"和"预测的 future"交给下游的那两个接口。这是把 KV4 放进故事的理由：缓存就是 future 被存放、被读取的地方。

两点要心里有数：

- Fast-WAM 的缓存本身很小（按 30 层、宽度 3072 估算约 0.1 GB），KV4 在这个模型上省不了多少显存。它的意义在于完整的 4-bit 管线，以及对缓存跨 chunk 复用的 WAM（LingBot-VA、DreamZero）的适用性。论文里不拿 Fast-WAM 的缓存显存当卖点。
- frame 0 是纯前缀这件事还带来一个无损的工程收益：观测前缀每次 query 只需要算一次，video 循环的行数从 294 降到 196。需要先确认当前引擎是否已经这样做（G0）。如果没有，这是一个与量化无关、输出逐位不变的加速，可以放进效率节。

联合去噪的 WAM（Cosmos-Policy）没有缓存，KV4 在那里指每一步对注意力的 K/V 操作数做 4-bit 量化；"写一次读多次"的对象只剩权重和干净的条件 latent。框架的定义不依赖缓存是否存在。

### 1.2 三篇 WAM 量化工作各自占了哪条轴

| | QuantWAMs（2607.28405） | Q-WAM（2609.33269） | SteerQuant（2609.39056） | 本框架 |
|---|---|---|---|---|
| 要保住的量 | 联合训练损失 / 闭环 | 最终动作 | 最终动作 | **被动作读取的预测 future**（连带动作） |
| 敏感度信号 | channel 能量；video+action 联合梯度；固定干预回放 | 动作对每层输入的 Jacobian Gram 矩阵（AOG） | 参考残差经 Jacobian 对动作的影响图（layer × step × stream） | future 接口上的读出度量 × 读次数 |
| 对跨步误差的假设 | 逐步 | 不同 token–step 的误差不相关，丢掉交叉项 | 逐 layer–step–stream | 冻结误差保留交叉项，动态误差按独立处理 |
| 保护的轴 | channel、Linear、去噪步 | channel 空间里的方向 | channel 缩放 + stream 增益 | future 的 token（位置）与缓存 |
| 载体 | BF16 channel 旁路、W8A8 层、A8 步 | 16-bit rank-32 低秩分支 | 共享缩放 D、增益 γ、截断 c | 选中 token 行上的第二遍 INT4 |
| 额外权重存储 | 名义 4.8 bit | +0.08–0.26 bpw | 0 | 0 |
| KV 缓存 | 未说明 | 未量化 | BF16 | KV4 |
| 运行时自适应 | 无 | 无 | 无 | 有（policy 的注意力） |

补充一点背景：Q-WAM 的形式化里，video 一侧是"编码一次得到的缓存条件"，求导范围是 action 采样器；它在 Fast-WAM 上量到 action expert 占 89.3% 的 action mass，于是只保护 action expert。这和我们的观察 (a) 正好接上：在测试时生成 future 的模式下，保护完 action expert，剩下的误差就是 future。

### 1.3 载体家族的先例（论文里不能声称是我们首创的东西）

- token 级混合精度：Squat（按最近一次注意力图，重要 token 8 bit、其余 4 bit）、STaMP（ICLR 2026，沿序列维做变换后少数 token 给 8 bit）。
- 对显著 token 做残差再量化：SemanticDialect（2603.02883，视频 DiT，注意力引导的 activation 分解）。
- KV 缓存按 token 显著性混合精度：ZipCache 一类；"Quantize What Counts"指出 key 比 value 更敏感。
- 相邻去噪步差分 activation 的低比特计算：Cambricon-D、Ditto、DSTAR（体系结构方向）。

这些工作选 token 的依据都是**模型自己的注意力或能量**。WAM 的不同在于：生成 future 的模块和消费 future 的模块不是同一个，值不值得保护由下游的 policy 决定。我们的新意钉在三处：保护目标（被读取的 future）、读次数、由读者定义并在运行时跟踪的选择；载体本身明确写成"沿用 token 级精修这一家族"。Q-WAM 也是这个结构：低秩 16-bit 分支来自 SVDQuant，它的贡献是选子空间的准则。

### 1.4 并行工作

2026 年 10 月之后出现的工作（如 LatentQuant、AGQ）按并行工作处理，不为它们调整设计，相关工作里至多一句。PreDE（2609.19441，离线预测量化造成的任务退化）在 9 月，相关工作里引用，不是量化器，不进主表。

---

## 2. 两份 ICLR 审稿意见给我们的硬约束

评分：Q-WAM 为 3 / 2 / 3（弱接收、弱拒、弱接收），SteerQuant 为 2 / 2 / 2（三个弱拒）。下面每一行是审稿人明确写出的问题，以及我们在框架里对应的做法。

| # | 审稿人的问题 | 出处 | 我们的硬约束 |
|---|---|---|---|
| 1 | 仿真基准饱和：LIBERO 上所有方法接近 100%，基线之间差不到 1%，没有置信区间；W4A8 反而比 W4A4 差，暴露方差 | SteerQuant dRvo | 主表用 LIBERO-Plus（BF16 77.6%），840 个配对 episode；Avg 列带 95% 置信区间。标准 LIBERO 不当主证据 |
| 2 | 真机没有基线，只有 BF16 对本方法 | SteerQuant dRvo | 真机三臂：BF16 / 最强基线 / LAQ，同一批初始条件，每任务每方法 ≥ 20 次 |
| 3 | 核心选择准则的贡献没有在闭环上单独隔离：需要同管线、同预算、只把准则换成随机或局部准则，报任务成功率 | Q-WAM 三位审稿人 | Table 3 专做"准则替换"：管线、预算、精修方式全部固定，只换选 token 的准则，报闭环成功率 |
| 4 | 加模块的同时加了高精度预算，无法归因 | Q-WAM TXNp、ctjZ | 每一行都报预算（W bpw、平均 A bit、平均 KV bit）；凡是加预算的行，都配一个同预算的随机对照 |
| 5 | 贯穿整个去噪过程的求导是否必要；校准要 13 万次 backward，时间和显存没报 | Q-WAM TXNp、D7vG | 主方法不展开 video 采样器求导：只在 reader 上打探针，读次数用前向干预测。报校准 GPU·h 和峰值显存。展开求导的版本只在附录里当验证 |
| 6 | 保护对象能否跨任务复用；没校准过的任务表现如何；重新校准多贵 | Q-WAM TXNp | 选哪些 token 在运行时由 policy 的注意力给出，不需要按任务重新校准。实验：校准任务与评测任务不重合，再加 LIBERO-Plus 的扰动 |
| 7 | 效率只报了被量化 block 的显存和 block 级延迟；Fast-WAM 上只有 1.06×；要端到端延迟、整机峰值显存 | Q-WAM 全体；SteerQuant skbH、1WVx | 效率表报端到端延迟（观测 → 动作块）、整个 policy 的峰值显存、校准时间；BF16、plain W4A4KV4、基线、LAQ 在同一引擎下测；拆出带宽、算术和精修的额外开销。速度不当主卖点（A100 上 GEMM 只占去噪时间的 46%，上限约 1.35×），主卖点是显存和保真 |
| 8 | 缺一个简单的启发式基线：只量化 video 流，action / proprio 流保持全精度 | SteerQuant skbH | 这个基线就是观察 (a) 里"保护动作通路"的那一臂，直接进主表。我们的动机图说明它为什么不够 |
| 9 | 没有和专门的 WAM 量化方法 QuantWAMs 比 | SteerQuant skbH | 主表包含 QuantWAMs、Q-WAM、SteerQuant 在同一设定下的复现 |
| 10 | 方法复杂：三套权重要设计、要优化，优化受数据采集影响大 | SteerQuant 1WVx | 无训练、闭式求解；只有一个预算旋钮；部署时只多一次选中行上的 INT4 GEMM |
| 11 | 消融只给 RMSE，和成功率对不上；"只用第一步够不够" | SteerQuant dRvo | 关键消融行报闭环成功率，不只报开环误差 |
| 12 | 有限探针下的估计可靠性没有论证 | Q-WAM ctjZ | 附录给探针数、场景数的收敛曲线和对半稳定性（位置对半相关 0.86 已有） |
| 13 | "有些做法在别的工作里已经出现过"，现有实验不足以支撑论点 | Q-WAM TXNp | 相关工作里先承认载体家族的先例（§1.3），把贡献限定在三处，并用 Table 3 证明"由谁定义重要性"确实改变结果 |
| 14 | 对只预测 future、不输出动作的 WAM 是否适用 | SteerQuant skbH | 读出度量取 M = I 时，目标退化为均匀的 future 保真，框架照常成立。正文一句话说明 |
| 15 | 可读性：SteerQuant 有两位审稿人给 Presentation 2（"要费不少力气"），Q-WAM 三位都是 3 | 两篇对比 | §4 每小节只立一个命名对象，开头一句话说清它做什么，配一个加框公式（❓ 见 §8） |

审稿人肯定的东西也值得照做：Q-WAM 的 Fig. 3a（AOG 分数与动作损伤相关 0.95，局部误差只有 0.04）被当作"直接的动机证据"；跨平台真机被认为有实用价值；"从观察到准则到机制，逻辑连贯"。我们要有一张同样分量的动机图（§6 的 Fig. 3）。

---

## 3. 故事线

### 3.1 引言的论证链

1. WAM 预测场景怎样演化，并以预测的 future 为条件生成动作。
2. 机载部署受显存限制，需要把权重、activation、缓存都压到 4 bit。
3. 现有 WAM 量化方法都以最终动作为目标：逐步、局部地估计每个计算对动作的敏感度，把额外精度给排名靠前的。动作通路因此被保护起来。
4. 我们发现这样留下了一个没人守的对象：预测的 future。三条观察 (a)(b)(c)。
5. LAQ 把 future 的保真度作为显式的量化目标，沿着它被写、被存、被读的全过程保护它。
6. 跨多个 WAM、仿真加真机的结果。

全文统一用 *predicted future*；"写 / 存 / 读"对应 *written / stored / read*。Fast-WAM 的具体结构只出现在实验设置里。

### 3.2 三条观察的完整表述

**(a) Protecting the action is not enough: what remains enters through the future.**

- 做法：把动作通路保持高精度（即审稿人要的简单基线），其余全部 4 bit，测动作误差；再把量化模型预测的 future 换成 BF16 的 future，看动作误差还剩多少（future 占比，定义沿用现有大纲 §3.1）。
- 已有数字（✅ D3，26039756；BF16 权重、per-token A4）：

  | 配置 | first10 motor | future NMSE |
  |---|---:|---:|
  | 全 A4 | 0.0426 | 0.0384 |
  | action 循环 A8，video 循环 A4 | 0.0176 | 0.0384 |
  | 全 A8 | 0.0028 | 0.0006 |

  action 循环升到 A8，future 误差在 40 个场景上逐一不变。此时的 motor 误差 0.0176 是全 A8 的 6.3 倍，按 MSE 算其中 97% 来自失真的 future。
- 另一半也要讲清楚：在不做任何保护的全 A4 模型里，action 循环自己的误差占动作 MSE 的 83%。这正是以动作为目标的方法会把敏感度量在 action expert 上的原因（Q-WAM 量到 89.3%）。它们保护了这一块之后，剩下的问题换了地方。
- ⏳ 在 W4（GPTQ）+ A4（g32）+ KV4 的正式口径下重测；同时用 Q-WAM 复现臂（ASP 加在 action expert 上）测一次 future 占比。
- 末句：*The quantization target must therefore include the predicted future itself.*

**(b) The future is damaged where it is written, and most by errors that are read many times.**

- 写的地方：只读 future 的模块修不了 future。action 循环升 A8 后 future 误差不变（上表）；LAVQ 阶段也量到 action expert 对 future 保真度的影响恰好为 0。
- 读很多次：
  - 观测前缀的行在所有去噪步里逐位相同，它的舍入误差每一步都是同一个（✅ D1：frame 0 残差相关 1.000）。缓存的 K/V 写一次、读 10 次，误差同样是冻结的。权重更是每个 token、每一步都在读。
  - 冻结的误差相干叠加。权重引起的 velocity 误差跨 10 个去噪步的和，能量是各步能量之和的 7.1 倍（✅ tfdiag，26065573，W2 权重下测得）。
  - 逐步独立的一阶敏感度看不到这一点：它给 future 通路 6.9% 的动作敏感度，把 BF16 的 future 换回去实测是 35–60%（✅ 26054103 / 26056114）。
  - 反过来，动态 activation 的舍入误差本来就接近每步独立：future token 的相邻步残差相关只有 0.25，打散相关也没有收益（✅ D3 dither 对照）。所以读次数这件事只对冻结的误差源起作用，这正好划出了两类。
- ⏳ 直接验证：往观测缓存、future 缓存里注入等能量的误差，一种冻结、一种每步重采样，比较 future 的读出失真（G2）。这张图是 Fig. 3(b)。
- 末句：*An error should be weighted by how many times it is read, not only by where it sits.*

**(c) The policy reads a narrow part of the future, stable in location but not in direction.**

- ✅ 26073871（在 BF16 点测得，与量化器无关，可直接沿用）：reader 度量有效秩 41.5，future 维度 37,632；第 1 个预测帧占 98%；前 10% 的对角元占 62%。
- 对角（"读哪里"）在两半样本之间相关 0.86；前 32 个特征方向（"沿哪个方向读"）的重合只有 0.19。读的方向随场景变，固定的方向子空间跟不上；读的位置是稳定的。
- ⏳ 把对角画成 14×28 的热力图叠在预测帧上（policy gaze）；比较 policy 的 gaze 与 video 模型自注意力显著图的重合度（G3 的一部分）。如果两者明显不同，就直接证明"由读者定义重要性"和现有 token 显著性方法不是一回事。
- 末句：*Protection should follow the tokens the policy reads, not a fixed set of directions.*

**收束句。** *Together, these observations call for a quantizer that preserves the predicted future along its whole life, where it is written, stored and read, in the metric of its only reader.*

---

## 4. 方法框架：LAQ

### 4.1 总览

LAQ 把 W、A、KV 三类量化误差写进同一个目标：被动作读到的 future 失真，加上动作通路自身的直接误差。三类误差用同一个度量（读出度量）、同一个权重（读次数）折算成同一种损伤，再用一个预算约束联合求解。解是闭式的：权重一侧就是标准 GPTQ，activation 和缓存一侧是一条阈值规则，决定哪些 token 多做一遍 4-bit。

部署时模型长这样：

1. 权重：全部 4 bit（GPTQ，与所有基线相同）。
2. 观测前缀：每次 query 算一次，写进缓存时对被读得多的 token 做精修。
3. video 去噪循环：全部 A4；policy 会读的 token 在靠后的若干步多做一遍 4-bit。
4. future 缓存：KV4；policy 会读的 token 的 K/V 多存一份 4-bit 残差码。
5. action 循环：A4；32 个 action token 全部精修（行数很少，代价可以忽略）。

超参数只有一个：预算（平均多花多少 bit）。平滑系数、组大小等沿用基线设定。

### 4.2 读出度量：在 future 上量误差（对应观察 (c)）

沿用 LAVQ 的 ①，定义和代码都已有。

\[
M_R=\mathbb E_c\big[J_R^\top\operatorname{diag}(h_c/2)\,J_R\big],\qquad
M=(1-\rho)\,\frac{D}{\operatorname{tr}M_R}M_R+\rho I,\qquad
\boxed{Z_M(\delta z)=\frac{\delta z^\top M\,\delta z}{\|z_T\|^2}}
\]

- \(J_R\)：reader（从 future 到动作的那段计算）对 future 的 Jacobian；\(h_c\)：soft-bin 动作分布的曲率。\(M_R\) 是 reader KL 在 BF16 future 处的 Gauss–Newton 矩阵，也就是 K 的二阶形式。
- \(\rho I\)（ρ = 0.1）保证 future 的任何部分都不会完全不受约束，对应 Z。
- 于是原来的 A + λZ + βK 没有丢：A 是动作通路的直接项，Z 和 K 合成 \(Z_M\)。它们现在定义的是敏感度，而不是训练损失。
- 估计：每个场景 4 个 Rademacher 探针，只在 reader 上反传，不展开 video 采样器。
- token 边缘 \(\pi_j=\operatorname{tr}(M_{jj})\) 就是离线的 policy gaze。

对所有来自生成一侧的扰动，动作误差的二阶形式都分解为 \(J_z^\top M J_z\)：同一个 M 服务于 video 一侧的每一层、每一步、每个 token，不需要像逐层 Gram 矩阵那样各估一份。

### 4.3 读次数：一个误差被读了多少次（对应观察 (b)）

对误差源 u（某个 token 在某层的 activation、某个缓存 token 的 K/V、某层的权重），记它在第 s 步进入、传到 future 的 Jacobian 为 \(J_{u,s}\)。定义读出核

\[
\mathcal K_u(s,s')=\mathbb E\,\operatorname{tr}\!\big(J_{u,s}^\top M\,J_{u,s'}\big).
\]

- **新鲜的误差**（动态 activation：每一步是一次新的舍入）：损伤 \(=\sigma_u^2\sum_s\mathcal K_u(s,s)\)。
- **冻结的误差**（权重、缓存、观测前缀：同一个实现被每一步读）：损伤 \(=\sigma_u^2\sum_{s,s'}\mathcal K_u(s,s')\)。

\[
\boxed{n_u=\frac{\sum_{s,s'}\mathcal K_u(s,s')}{\sum_s\mathcal K_u(s,s)}\in[0,T]}
\]

\(n_u\) 是有效读次数：冻结的误差相当于被读了 \(n_u\) 次。现有的逐步敏感度（AOG、Action-Impact Map）只有对角项，等于默认 \(n_u=1\)。

- 估计用前向干预，不用求导：对每一类误差源注入等能量的误差，一次冻结、一次每步重采样，读出失真之比就是 \(n_u\)。误差源只分几类（观测缓存、future 缓存、观测行、future 行的前段步 / 后段步、action 行），几十次前向即可，而且是有限误差下的精确测量。
- 探针版的完整读出核只放附录，用来验证干预测到的 \(n_u\)。

### 4.4 联合分配：W、A、KV 一个目标（对应观察 (a)）

\[
\min_{\hat W,\;b}\quad
\underbrace{A_{\text{direct}}}_{\text{动作通路}}
+\lambda\Big[\underbrace{\sum_\ell n_\ell\operatorname{tr}\!\big(\Delta W_\ell H_\ell\Delta W_\ell^\top\big)}_{\text{权重}}
+\underbrace{\sum_{s,j}\omega_{s,j}\,\sigma_{s,j}^2(b_{s,j})}_{\text{动态 activation}}
+\underbrace{\sum_{j}n_j\,\omega_j\,\sigma_j^2(b_j)}_{\text{缓存与观测前缀}}\Big]
\quad\text{s.t.}\ \ \text{cost}(b)\le B
\]

其中 \(\omega\) 是读出核的对角和（按读出度量算的敏感度），\(b\in\{0,1\}\) 表示该 token 是否精修。三项可分，各自有闭式解：

- **权重。** 冻结读法下的层目标与标准的输入二阶矩成比例（✅ traj，26066187：相干版与标准版 Hessian 的比例在所有量化臂上稳定在 0.85–0.89，逐 token 的跨步相干度中位 9.7 / 10）。所以权重一侧的最优解就是标准 GPTQ，读次数只改变各层的整体尺度。论文里可以直接写：我们检验过，标准做法在这里已经是对的。
- **activation 与缓存。** 精修把误差功率降到原来的 1/196 以下（§4.5），所以规则是

  \[
  \boxed{\text{精修单元 }u\iff\frac{n_u\,\omega_u\,\sigma_u^2}{\text{cost}_u}\ \ge\ \mu}
  \]

  \(\mu\) 由预算 B 唯一确定。A 和 KV 共用同一个 \(\mu\)，这就是"联合"的含义：多花的每一 bit 放在 A 还是 KV，由同一个阈值决定。
- **规则给出的次序**（结构上可以预期，数字待 G1–G3 填）：缓存与观测前缀排最前，因为它们被读多次（\(n_u\) 大），而只写一次（cost 小）；action token 始终精修（行数少）；其次是 gaze 高的 future token 在靠后的去噪步；其余保持 4 bit。
- **写路径一致**：一个 token 被选中，它在被写出的地方（靠后的 video 步和 prefill 的输入）和被存放的地方（缓存码）同时精修。同一个 token 身上的三次舍入（输入 A4、权重 W4、缓存 KV4）只有对齐地保护才有意义，这是 A 与 KV 必须联合设计的直接原因。

### 4.5 Gaze 引导的精修：全 INT4 的载体

**精修。** 对选中的 token：

\[
q_1=\operatorname{round}(x/s_1),\ s_1=\tfrac{\max|x|}{7};\qquad
r=x-s_1q_1;\qquad
q_2=\operatorname{round}(r/s_2),\ s_2=\tfrac{\max|r|}{7}\le\tfrac{s_1}{14};\qquad
\boxed{\hat x=s_1q_1+s_2q_2}
\]

误差上界从 \(s_1/2\) 降到 \(s_1/28\)，功率至少降 196 倍，约等于多出 3.8 bit。Linear 的输出是 \(s_1W_q^\top q_1+s_2W_q^\top q_2\)：两次 INT4×INT4 的 GEMM，第二次只在选中的行上。缓存里，选中 token 的 K/V 多存一份 4-bit 残差码。权重完全不动，没有 16-bit 通路，没有额外的权重存储。平均位宽如实报告：A 为 4(1 + f_A)，KV 为 4(1 + f_KV)，f 是被精修的比例。

举例说明量级（不是结果）：video 循环里 25% 的 future token 在最后 4 步精修，占 future token-step 的 10%；观测前缀只算一次，精修它相当于再加 5%；action 循环的 32 行可以忽略。合计平均 A 约 4.6 bit。

**选谁：gaze。**

- 离线先验：读出度量的 token 边缘 \(\pi_j\)。用于每个 episode 的第一个控制步。
- 在线跟踪：action token 对 future 缓存各 token 的注意力质量 \(g_j=\sum_{\ell,h,i,s}\alpha_{i\to j}\)，是 action 循环的副产品，不花额外计算。上一个控制步的 \(g\) 用来选下一个控制步里要精修的 token（闭环控制中相邻两次 query 的场景变化很小）。
- 观测前缀的 token 按两跳读出排序：\(g^{\text{obs}}_i=\sum_j g_j\,\alpha^{\text{video}}_{j\to i}\)，即"被 policy 读的 future token 又读了哪些观测 token"。
- 联合去噪的模型（Cosmos-Policy）里，action token 每一步都在读 future token，上一步的注意力直接用于下一步。

在线跟踪回答了 Q-WAM 审稿人关于"换任务要不要重新校准"的问题：离线量（读次数、步的轮廓、预算）与任务无关，随场景变化的那部分由运行时的注意力给出。LIBERO-Plus 里量化模型掉得最多的是相机视角扰动，静态先验在那里会错位，在线 gaze 正好检验这一点（G4）。

### 4.6 与 Q-WAM 的逐项对照（避免结构雷同）

| | Q-WAM | LAQ |
|---|---|---|
| 目标 | 最终动作 | 被动作读取的预测 future |
| 数学对象 | 每层一个 d×d 的 channel Gram 矩阵 | future 接口上的一个读出度量 + 每类误差源的读次数 |
| 保留 / 丢弃的结构 | 保留 channel 之间的协同，丢掉 token–step 之间的交叉项 | 保留跨步的交叉项（读次数）和 token 位置，不解析 channel 方向 |
| 保护的轴 | 方向（混合很多 channel） | 位置（token）与缓存 |
| 载体 | 16-bit rank-32 分支 + 权重 deflation | 选中行上的第二遍 INT4 |
| 额外权重存储 | 有 | 无 |
| "放在哪" | 按 trace 汇总选 expert | 写 / 存 / 读三处由一个阈值联合决定 |
| 校准 | 展开整个采样器、每个输入 12 次 backward | reader 上的探针 + 前向干预 |
| 运行时 | 固定子空间 | 注意力跟踪 |
| 缓存 | 不量化 | KV4 |

行文上避开 observability、Gramian、subspace protection、action mass 这些词；我们的词是 readout、read count、gaze、refinement、written / stored / read。方法图按"future 的一生"从左到右画（§6 Fig. 2），不采用"上排度量、下排分支"的两排布局。

### 4.7 附录里的理论条目

1. 读出度量是 reader KL 在 BF16 点的 Hessian（Fisher），不要求模型处在训练损失的极小点。
2. 经由 future 的分解：生成一侧任意扰动的动作二阶损伤等于 \(J_z^\top MJ_z\)，秩不超过动作维度。
3. 新鲜 / 冻结两种读法下的损伤公式与 \(n_u\in[0,T]\) 的界。
4. 可分目标下阈值规则的最优性。
5. 精修的误差界（1/196）。
6. 权重一侧：冻结读法的层目标与标准 Hessian 成比例，因此 GPTQ 的解不变。

---

## 5. 证据状态与先跑的验证

全部是开环测量，复用 `fastwam-temporal-a4-d1d3-20261028` 的 eager fake-quant 管线（每臂约 10 分钟）和 reader / 分阶段 VJP。

| 编号 | 内容 | 通过标准 | 不通过时怎么办 |
|---|---|---|---|
| G0 | 确认观测前缀在代码里确实不依赖 future token 和时间步；实现"前缀只算一次"，核对输出逐位不变，测无损加速 | 逐位一致 | 不影响故事，只少一个工程收益 |
| G1 误差账本 | W4（GPTQ，g32）+ A4（g32）下，分别打开 / 关闭观测缓存 KV4、future 缓存 KV4；再加"action token 全精修"臂。测 motor、future NMSE、\(Z_M\)、future 占比 | 动作通路保护后 future 占比 ≥ 70% | 观察 (a) 换成 VQ 阶段的 73% / 85% 作旁证，并如实缩小措辞 |
| G2 读次数 | 观测缓存、future 缓存、观测行、future 行：等能量误差，冻结对重采样，测读出失真之比 \(n_u\) | 缓存与观测行的 \(n_u\ge3\) | 观察 (b) 只保留"写的地方"和权重的 7.1×；读次数降为附录里的分析，方法保留读出度量 + gaze 精修 |
| G3 同预算选 token | 预算 5% / 10% / 20%；选择准则：policy gaze、随机、video 自注意力显著性、activation 能量、只按步（最后若干步全部 token）；另加"观测前缀精修一次" | gaze 的 \(Z_M\) 比随机和只按步低 30% 以上 | 载体换成备选（见下），故事不变 |
| G4 在线 gaze | 注意力质量与精确的 \(\pi\) 的秩相关；相邻控制步的 gaze 重合度；相机视角扰动下静态先验对在线跟踪 | 相关 ≥ 0.6，重合 ≥ 0.7 | 主方法只用离线先验，在线跟踪不写 |

之后上闭环：先 LIBERO-Plus 面板 A（280），再补到 840 配对。

**备选载体（G3 不通过时）。** 跨步残差编码：只量化本步相对上一步的增量。✅ D3 里它与 g32 叠加后 motor 0.0133、future 0.0075，是已测到的最强开环组合。它在体系结构方向有先例（Cambricon-D、Ditto、DSTAR），所以只能作为载体出现，贡献仍然落在读出度量和读次数上：第一步和读得多的位置给完整精度，其余编码增量。它需要每层保存跨步状态（约 1.2 GB），会吃掉一部分显存收益，这是它排在备选的原因。

**已有的 W4A4 闭环数字（面板 A，280，per-token A4，无 KV4）：** BF16 216；SpinQuant 203；ViDiT-Q 203；QuaRot 195；SVDQuant 108。基线到 BF16 差 13–21 个 episode，比 VQ W2 阶段的头部空间大。这些都要在 g32 + KV4 的口径下重跑到 840。

---

## 6. 论文框架

章节结构沿用你已经确认过的安排：§3 Preliminaries and Motivation（定义 + 三条观察 + 三联图），§4 Methodology，§5 实验为 设置 / 主结果 / 真机 / 消融 / 效率，§6 Limitations and Future Work，§7 Conclusion，之后是 LLM Usage Statement。

### 题目

1. **Don't Lose the Future: Latent-Aware 4-Bit Quantization of World Action Models**（延续现在的题目，改动最小）
2. Write Once, Read Many: Preserving the Predicted Future in 4-Bit World Action Models（突出读次数）
3. Future-Faithful W4A4KV4 Quantization of World Action Models

### 摘要（草稿）

> World action models (WAMs) predict how a scene will evolve and condition their actions on this predicted future. Running them on a robot's onboard computer calls for 4-bit weights, activations and key–value caches. Existing 4-bit quantizers for WAMs protect the generated action: they rank computations by a local, per-step sensitivity of the final action and keep the top-ranked ones at higher precision. We show that this leaves the predicted future unprotected. Once the action pathway is protected, XX% of the remaining action error enters through the degraded future; the errors that damage the future most are written once and read many times, in the weights, the caches and the observation prefix, and per-step sensitivity under-counts them; and the policy reads only a narrow part of the future, stable in location but not in direction. We propose LAQ, a latent-aware quantization framework that preserves the predicted future where it is written, stored and read. LAQ measures error in the metric through which the policy reads the future, weights each error source by how many times it is read, and solves the resulting allocation over weights, activations and caches in closed form: the tokens the policy reads receive a second 4-bit pass in the denoising loop and in the cache. Weights stay at exactly 4 bits, with no high-precision branch and no per-task recalibration. We evaluate LAQ across XX WAMs, in simulation and in real-world deployment. On LIBERO-Plus, LAQ at W4A4KV4 reaches XX% success on Fast-WAM, within XX points of BF16 (77.6%) and XX points above the strongest baseline, while the whole policy fits in XX GB. On an AgileX PiPER arm, it succeeds in XX/XX trials, compared with XX/XX for the strongest baseline and XX/XX for BF16.

### §1 Introduction（5 段 + 贡献 + Fig. 1）

- ¶1 WAM 依赖预测的 future，但放不进机器人。6B 的 Fast-WAM 仅 BF16 权重就要 12 GB。设定：W4A4KV4。
- ¶2 已有工作做到了哪里。LLM 与扩散模型的 4-bit 量化（平滑、旋转、GPTQ、KV4）；WAM 上的 QuantWAMs、Q-WAM、SteerQuant 都以最终动作为目标，逐步、局部地排序敏感度，把额外精度给动作通路。收束句：*None of them treats the predicted future as something to preserve, and none quantizes the caches that carry it.*
- ¶3 问题。设问式开头：*What does a 4-bit WAM still lose once its action pathway is protected?* 三条观察各一句，末尾给原则句。
- ¶4 方法。*Building on this diagnosis, we propose LAQ …*，读出度量对应 (c)，读次数对应 (b)，联合分配与精修对应 (a)。
- ¶5 结果。先说评测范围，再给仿真数字，以 AgileX 真机收尾。
- 贡献三条：
  - **C1** *We identify future-latent fidelity as a key quantization target for WAMs.*（措辞保持你定的版本）后接三条观察的一句话概括。
  - **C2** *We propose LAQ*：读出度量、读次数、W / A / KV 的联合闭式分配、gaze 引导的全 INT4 精修；权重严格 4 bit，无训练，无需按任务重新校准。
  - **C3** 跨 XX 个 WAM 的 W4A4KV4 结果与 AgileX PiPER 真机结果。
- **Fig. 1（teaser）**：左栏画 future 的一生（观测 → 写 → 存 → 读 → 动作），三处各标一个误差源和它被读的次数；右栏是成功率对平均位宽的散点，BF16 为水平虚线，基线与 LAQ 各一个点。读者看完要记住：现有方法保住了动作，丢了 future；沿着 future 的一生保护它，成功率就回来了。

### §2 Related Work（约 0.6 页）

- 2.1 World action models：两三句。
- 2.2 Post-training quantization：平滑与旋转（SmoothQuant、QuaRot、SpinQuant）、GPTQ、W4A4KV4 这条线；扩散模型（PTQD、ViDiT-Q、SVDQuant）。
- 2.3 Quantizing robot policies and WAMs：QVLA、ActQuant、QuantVLA、DyQ-VLA；QuantWAMs、Q-WAM、SteerQuant；PreDE。差异句：它们的目标是最终动作，敏感度逐步估计，缓存不量化。
- 2.4 Token-level precision：Squat、STaMP、SemanticDialect、KV 缓存的显著性量化。差异句：这些方法由模型自己的注意力或能量决定哪些 token 重要；在 WAM 里，future 的消费者是另一个模块，重要性由它决定。

### §3 Preliminaries and Motivation（约 1.5 页，含 Fig. 3）

- **3.1 Quantizing a WAM.** 只放定义：
  - WAM 的通用形式：观测与指令 → 预测的 future latent z → 动作 a；reader R 指"给定 z 计算 a"的那段计算。
  - W4A4KV4 的三个量化器（权重按组、activation 按 token 动态、缓存按 token）。
  - 缓存与"写一次读多次"：哪些量每次 query 只写一次、却被每个去噪步读取。
  - 误差度量 A、Z、K 与 future 占比（沿用现有定义）。
  - 新鲜的误差与冻结的误差。
  - 结尾一句：*Existing objectives measure the final action step by step; they never measure the predicted future, nor how often an error in it is read.*
- **3.2 Observations.** 协议段三四句，三个结论段 (a)(b)(c)，共用 Fig. 3。细节放附录 A。
- **Fig. 3（三联图）**：
  - (a) 三根柱：全 4-bit、保护动作通路、全 8-bit 的动作误差；中间那根按"经由 future / 直接"两色拆分。
  - (b) 等能量误差的两种读法：冻结对每步重采样，在观测缓存、future 缓存、动态 future token 上的 future 读出失真。这张图的分量对应 Q-WAM 的 Fig. 3a。
  - (c) policy gaze 热力图叠在预测帧上；旁边两根小柱：位置的对半稳定性 0.86，方向的 0.19。

### §4 Methodology（约 1.5 页，含 Fig. 2 与 Algorithm 1）

- 开头一段串起全流程。
- 4.1 *Measuring the future as the policy reads it*（读出度量）
- 4.2 *Counting how often an error is read*（读次数）
- 4.3 *Allocating precision jointly over weights, activations and caches*（联合分配）
- 4.4 *Refining what the policy reads*（gaze 引导的精修）
- **Fig. 2（方法图，第 3 页）**：一条从左到右的流程，画 future 的一生：观测前缀（写一次）→ video 去噪循环（写 future）→ future 缓存（存）→ action 循环（读）。上方标各处的读次数；被选中的 token 用同一种颜色贯穿三处，表示同一个 token 集合在写、存、读三处被一起保护；一条从 action 循环指回 video 循环的虚线表示 gaze 带到下一个控制步。
- **Algorithm 1**：离线（读出度量、读次数、阈值）与在线（按 gaze 选 token、精修）两段。

### §5 Experiments（约 2.5 页）

- **5.1 Setup**（三段：模型与基准 / 基线与量化设定 / 评测指标）
  - 模型：Fast-WAM（IDM，主平台）+ 第二个 WAM（❓ 见 §8）。
  - 基准：LIBERO-Plus；真机 AgileX PiPER。
  - 基线：QuaRot、SpinQuant、ViDiT-Q、SVDQuant；QuantWAMs、Q-WAM、SteerQuant 的同设定复现；"动作流全精度、其余 4 bit"的简单基线。所有方法共用同一个权重量化器（GPTQ，同样的组大小）、同一个 KV4 量化器、同一批校准数据、同一个运行时。接口层 BF16 与时间条件表对所有方法一致。
- **5.2 Main results（Table 1）**：按 WAM 分块。列：W bpw | 平均 A bit | 平均 KV bit | Spatial | Object | Goal | Long | Avg（带 95% 置信区间）。W4A4（不量化缓存）的结果放附录。
- **5.3 Real-world results（Table 2）**：BF16 / 最强基线 / LAQ，三个任务。
- **5.4 Ablation**
  - **Table 3 准则替换**（回应 Q-WAM 三位审稿人）：管线、预算、精修方式固定，只换选 token 的准则：不精修 / 随机 / activation 能量 / video 自注意力 / 逐步一阶动作敏感度 / 只按步 / 离线 gaze / 在线 gaze。报闭环成功率和读出失真。
  - **Table 4 部件阶梯**：plain W4A4KV4 → 加观测前缀保护 → 加缓存精修 → 加循环内 gaze 精修；每行带预算列，并配同预算随机对照。再加一行"读次数取 1"（即逐步独立的记账）说明读次数的作用。
  - 泛化：校准任务与评测任务不重合时的结果，一句正文加附录表。
- **5.5 Efficiency（Table 5 + Fig. 5）**：端到端延迟（观测 → 动作块）、整个 policy 的峰值显存、权重存储、校准时间与校准峰值显存；BF16、plain W4A4KV4、Q-WAM 复现、LAQ 在同一引擎下测。Fig. 5 是延迟分解：带宽、算术、精修的额外开销、观测前缀只算一次省下的部分。

### §6、§7 与声明

§6 三条笼统方向（例如更长的预测视野、缓存跨 chunk 复用的模型、与步数蒸馏的结合）；§7 结论；LLM Usage Statement。

### 附录

A 观察的完整协议与结果；B 推导（§4.7 的六条）；C 估计量的可靠性（探针数、场景数的收敛，对半稳定性）；D 权重一侧为什么标准 GPTQ 已经足够；E kernel 与引擎、延迟分解；F 各方法的预算记账；G 校准的泛化（未见任务、扰动）；H 真机设置；I W4A4（缓存不量化）的结果；J 按扰动维度拆分的结果。

---

## 7. 从 LAVQ 稿继承什么、放下什么

| 继承 | 说明 |
|---|---|
| C1 的措辞与"保住 future"的主线 | 引言 ¶1、¶3 的骨架和三条观察的写法可以直接改 |
| reader 与 reader 度量（①） | 代码、探针、26073871 的全部结果直接沿用，是观察 (c) 和读出度量 |
| future 占比的定义与换 future 的实验管线 | 观察 (a) 的测量方法不变 |
| §3 的结构（定义 + 三条观察 + 三联图）与 §5 的结构 | 不变 |
| LIBERO-Plus 840 配对协议、面板、BF16 652 / 840 | 不变 |
| 公平口径（接口层 BF16、时间条件表） | 不变 |
| AgileX PiPER 真机（BF16 已训好，开环已通过） | 不变，三臂改为 BF16 / 最强标量基线 / LAQ |
| A + λZ + βK | 从训练损失变成定义敏感度的目标（§4.2） |
| 10-28 的 W4A4 资产 | GPTQ / Hessian 管线、R128 与 Hadamard 变换、S4×S4 kernel、引擎、D1/D3 的 fake-quant 管线与结果 |

| 放下 | 说明 |
|---|---|
| VQ 基线（GPTVQ、QuIP#、QTIP）与"极低比特 → VQ"的论证 | 换成 4-bit 标量基线与 W4A4KV4 的论证 |
| ② 专家之间的比特转移、③ 格栅码初始化、④ 解码表校准 | 标量框架里没有对应物；"按对 future 的影响分配"的思想保留在联合分配里 |

---

## 8. 已定事项（2026-11-10，你的答复）

1. **方法名**：LAQ（Latent-Aware Quantization）。
2. **主表带置信区间**：只在 Avg 列加 95% 置信区间，胜负和 p 值仍然不放。
3. **§4 的写法**：保留连贯段落，每小节只立一个命名对象、开头一句说清做什么、配一个加框公式。
4. **第二个 WAM**：先用 Cosmos-Policy。它在 W4A4 下几乎无损（面板 A：BF16 233，ViDiT-Q 232），KV4 之后有没有头部空间要在跑的时候先看。
5. **验证**：G0–G4 暂时不跑，先改论文。`CVPR2027/` 已在同一天按本框架重写（缺的数据为 `\XX`，待办为红色 `\TODO{}`，摘要里的预估数字为红色 `\est{}`）。

---

## 来源

本地：`papers/wam/013-q-wam/`、`008-quantwams/`、`014-steerquant/`、`016-dsaquant/`、`015-qvgen/`；两份 OpenReview 审稿页（Q-WAM 投稿号 36786，SteerQuant 投稿号 16670）；`research/paper/2026-10-28_w4a4_latent_aware_design.zh.md`；`experiment/idea-validation/fastwam-temporal-a4-d1d3-20261028/pulled/26039756.pbs101/RESULTS.zh.md`；`research/paper/2026-11-01_paper_outline.zh.md`。

网络检索：
- QuantWAMs：https://arxiv.org/pdf/2607.28405
- PreDE：https://arxiv.org/pdf/2609.19441
- Squat：https://arxiv.org/pdf/2402.10787
- STaMP：https://arxiv.org/html/2510.26771v1
- SemanticDialect：https://arxiv.org/abs/2603.02883
- Quantize What Counts（key 比 value 敏感）：https://arxiv.org/html/2502.15075v3
- NVFP4 KV cache：https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/
- LatentQuant（10 月，并行）：https://arxiv.org/abs/2610.03959
