# 23. QuantWAMs

**QuantWAMs: Calibrating at the Right Granularity for World Action Models**  
本地版本：**arXiv:2607.28405v1** · 13 页 · 讲解更新：2026-10-05  
分类：最高优先 / 直接 WAM PTQ prior art

[本地 PDF](paper-arxiv-v1.pdf) · [arXiv 版本页](https://arxiv.org/abs/2607.28405v1) · [官方项目入口](https://quantwams.github.io/) · [返回总指南](../README.md)

本文依据本地固定版本讲解，页码按 PDF 的物理页编号。实验数字是论文报告，本地未运行量化或 benchmark。教学数字会明确标出，与论文实测结果分开。

**QuantWAMs 的核心，是用有限的校准数据，决定哪些 channel、哪些 Linear、哪些 denoising step 值得保留更高精度。** 论文认为，WAM 的量化效果不好，往往不只是因为“4-bit 误差太大”，也因为这些保护决策是在不合适的数据、统计范围或目标函数上做出来的。

下面按“直觉 → 方法 → 数学 → 实验证据”的顺序讲。

---

**先想象一个机器人抓杯子的过程。**

机器人看到图像和当前状态，WAM 根据这些条件生成动作，机器人执行一段动作，再接收新的观察：

\[
\text{观察}_j
\rightarrow \text{WAM}
\rightarrow \text{动作}_j
\rightarrow \text{环境变化}
\rightarrow \text{观察}_{j+1}.
\]

WAM 中的 world/video 建模和 action 建模存在耦合。论文研究两种结构：

| 模型 | 可以怎样理解 |
|---|---|
| **Fast-WAM** | video 和 action 有各自的 Transformer expert，通过 shared attention 交互 |
| **LingBot-V-A** | video 和 action tokens 使用同一个 DiT backbone 的权重 |

这里不要把“模型有 video 路径”理解成“每次执行动作都必须生成完整未来视频”。当前关心的是：**模型中的 video–action 路径如何耦合，以及量化会怎样影响它们。**

WAM 的量化困难来自两个特点。

第一，**路径之间相互影响**。某处量化误差对一个路径看起来很小，经过另一条路径后，可能影响动作。

第二，**动作会改变下一次输入**。第一次抓取位置偏了一点，下一张图像就不同了。之后模型面对的输入，也会偏离原来的轨迹。

因此，固定图像上的输出误差小，并不自动意味着机器人闭环执行得好。

还有两个时间尺度需要分清：

| 时间尺度 | 含义 |
|---|---|
| **outer control call** | 机器人接收观察、生成并执行动作、再观察 |
| **inner denoising step** | 一次模型调用内部，把带噪声的 action/video 逐步变成输出 |

论文第三个方法保护的是 **inner denoising step**，不是“机器人执行到第几秒”。

---

**再把量化与校准的基本概念补齐。**

一个 Linear 可以写成：

\[
y=Wx.
\]

其中：

- \(W\)：weight，是模型学到的参数；
- \(x\)：activation，是运行时进入这个 Linear 的特征；
- \(x\) 的某一维：input channel；
- 一个 Linear：一个具体的线性算子，不一定等于整个 Transformer block。

所以，论文中的 channel 通常是隐藏特征维度，不能直接理解成相机的 RGB channel 或机器人的某个关节。

**W4A4** 表示 weight 和 activation 的主要计算路径都使用 4-bit。位数降低后，数值只能落在更少的表示点上，会产生误差。

用一个简单的对称 INT4 教学例子说明：

- 如果 activation 大部分位于 \([-1,1]\)，量化步长约为 \(1/7\approx0.14\)；
- 如果其中一个值达到 20，步长可能被拉到 \(20/7\approx2.86\)。

这样，许多原本有区别的小数就会被压到同一个量化值。

这就是 outlier 为什么麻烦：**少数极大值会牺牲大量普通值的分辨率。**

这个例子只用于解释直觉。论文实际的 Blackwell 后端使用 **NVFP4、FP8 和 BF16**，并非直接使用上面的简单 INT4 网格。

PTQ 的 calibration，就是在不进行 QAT 式训练的情况下，用一些代表性数据决定量化参数，例如：

- scale 应该多大；
- 哪些 channel 保留高精度；
- 哪些 Linear 升到 8-bit；
- 哪些 denoising step 使用更高精度。

这些都是**根据有限样本作出的估计**。QuantWAMs 的出发点是：

> 不仅量化数值会出错，决定如何量化的统计估计本身也会出错。

---

**整篇论文有三种保护，它们解决三个不同问题。**

| 保护对象 | 要作出的决定 | 决策依据 | 方法 |
|---|---|---|---|
| **activation channel** | 哪些 channel 走高精度旁路？哪些模块共享这个选择？ | 变换后的 activation energy、坐标兼容性、统计稳定性 | Shared-Basis Outlier Calibration |
| **Linear** | 哪些 Linear 从主要的 4-bit 路径升级到 W8A8？ | 联合 video–action 目标下的量化损伤估计 | Co-Training-Objective Saliency |
| **denoising step** | 固定数量的 A8 保护放在哪些 step？ | 真实 rollout 快照上的固定干预误差 | Fixed-Intervention Real-Rollout Auditing |

读方法时，始终追问三个问题：

1. 这个决策在保护什么？
2. 用什么数据、什么统计量来决定？
3. 这个统计量与最终闭环成功率之间，还隔着哪些环节？

下面逐个讲。

---

**第一个方法：多个模块共享 channel 统计，有时会比各算各的更准确。**

先看一个看似合理的做法：

> 每个 Linear 都独立统计自己的 activation，再保留能量最大的少量 channel。

它的问题是校准数据很少。论文只使用 **32 条 calibration trajectories**。如果每个位置都单独估计，得到的 Top-K 排名可能受样本噪声影响。

比如，真实情况是 channel 1 比 channel 2 更重要，但某个模块刚好看到的 32 条轨迹让 channel 2 显得更大，于是保护错了。

一个自然想法是：

> 把若干模块的统计汇总起来，让估计更稳定。

这叫 **pooling**。

但共享也可能有害。假如模块 A 真正需要保护 channel 1，模块 B 真正需要保护 channel 2，把二者平均后，各自的特点就被抹掉了。

因此，论文不是主张“尽量共享”，而是主张：

**先确认能共享，再判断共享是否值得。**

**能不能共享，首先是坐标问题。**

假设两个向量长度相同：

\[
A=(u,v),\qquad B=(v,u).
\]

A 的第 1 个 channel 表示 \(u\)，B 的第 1 个 channel 表示 \(v\)。如果直接把“第 1 个 channel”的能量平均，就是把不同东西混在了一起。

因此：

> tensor 的 shape 相同，并不意味着相同 channel index 具有相同含义。

论文要求共享 mask 的模块具有兼容的 quantizer-input coordinates，也就是 **shared basis**。

模型会先对 activation 做变换：

\[
\widetilde A_i=(A_iR)S_i^{-1}.
\]

这里：

- \(R\)：Hadamard rotation，用于重新分布特征中的大值；
- \(S_i\)：diagonal smoothing scale；
- 统计是在变换后的 \(\widetilde A_i\) 上收集的。

组内使用相同的 rotation，可以保持共同的有序坐标系。不同模块可以使用不同的 diagonal scale，因为它改变幅度，不混合 channel 身份。

但如果不同模块各自使用不同的 dense rotation，就不能再直接共享相同 index 的 mask，除非额外建立坐标映射。

对应到模型：

- **Fast-WAM** 中，shared value pathway 为部分 paired output projections 提供了对齐基础；
- expert-private residual streams 不允许随意跨 expert 共享 Q/K/V 的 channel mask；
- **LingBot-V-A** 的不同 mode 使用同一个物理 Linear 的输入列，可以在变换保持坐标兼容时跨 mode 共享；
- 跨 depth 共享还需要额外判断，不能仅凭“都是同类层”就成立。

**坐标兼容，只代表比较有意义；它不代表统计分布相同。**

**具体统计什么？统计每个 channel 的平均平方值。**

对某条 trajectory，某个 channel 的 energy 是：

\[
z_i^{(n)}(c)
=
\frac{1}{T_{i,n}}
\sum_t \widetilde A_{i,t,c}^{\,2}.
\]

可以理解成：

> 在这条轨迹中，这个 channel 平均有多强。

之后再对 trajectories 平均，得到每个模块的估计。对于可共享的一组模块，进一步汇总：

\[
\widehat e_g(c)=\sum_{i\in g}\pi_i\widehat e_i(c).
\]

其中 \(\pi_i\) 表示预期部署时各成员的 exposure 权重，**不是直接按 token 数量加权**。

最后选择：

\[
\Omega_g=\operatorname{TopK}_K(\widehat e_g).
\]

例如，假设输入宽度是 1024，保护比例为 2%，那么：

\[
K=\lfloor0.02\times1024\rfloor=20.
\]

这 20 个 channel 对应的计算走高精度 exception path。

注意两个边界：

- Top-K 是对变换后的 channel energy 排序；
- energy 最大，是这个 surrogate 下的选择，不等于“这些 channel 对闭环成功率最重要”。

**什么时候 pooling 有利？关键是噪声和真实差异谁更大。**

论文把不确定性拆成两部分：

- \(\sigma^2\)：同一个模块因 trajectory 抽样产生的变化；
- \(\tau^2\)：不同模块之间真实存在的差异。

在均衡、独立等简化工作假设下，如果组内有 \(m\) 个成员，每个成员有 \(N\) 个 trajectory 样本：

\[
R_{\mathrm{ind}}=\frac{\sigma^2}{N},
\]

\[
R_{\mathrm{pool}}
=
\frac{m-1}{m}\tau^2
+
\frac{\sigma^2}{mN}.
\]

这里的 risk，是**能量估计对各成员真实能量的均方误差**，不是机器人失败概率。

两个公式的含义很直观：

- 独立估计保留模块差异，但样本噪声大；
- pooling 降低样本噪声，但引入“把不同模块拉向共同平均”的偏差。

下面是教学数值：

\[
m=4,\quad \sigma^2=1,\quad \tau^2=0.05.
\]

| 每个成员的 trajectory 数量 \(N\) | 独立估计 risk | pooling risk | 哪个更好 |
|---:|---:|---:|---|
| 5 | 0.2000 | 0.0875 | pooling |
| 20 | 0.0500 | 0.0500 | 相同 |
| 100 | 0.0100 | 0.0400 | 独立估计 |

少量数据时，共享能帮忙；数据充分后，独立估计已经准确，共享带来的偏差反而更突出。

论文得到的 crossover 是：

\[
\boxed{N<N^\star=\frac{\sigma^2}{\tau^2}}
\]

即：**抽样变化相对真实模块差异越大，pooling 越有价值。**

但作者没有只凭这个阈值就共享。还会检查 **mask stability**。

原因是，能量估计变准，不保证 Top-K 选择稳定。比如第 20 名和第 21 名分别是 10 和 9.99，极小的误差就会交换排名。

最终做法是：

1. 用架构确定坐标兼容的 candidate groups；
2. 用 calibration trajectories 估计噪声与异质性；
3. 判断是否处于 pooling 有利的区间；
4. 检查 trajectory-level bootstrap 下的 mask 稳定性；
5. 通过筛选后才共享。

Bootstrap 使用的是**同一组 32 条轨迹的重采样**，并保留不同 context 之间的配对关系。它用于检查稳定性，没有凭空增加新的独立数据。

所以，第一个方法可以理解为：

**用适度的统计共享降低小样本噪声，同时防止错误坐标和真实模块差异被平均掉。**

---

**第二个方法：哪些 Linear 值得升到 8-bit，要看联合目标下能减少多少损伤。**

只保留高精度 channel 还不够。有些 Linear 对量化特别敏感，论文给这些 Linear 分配 W8A8。

这里有两个问题：

- 怎样衡量 Linear 的重要性？
- 应该按单个 weight、column，还是整个 Linear 分配精度？

**先理解为什么要使用联合 video–action 梯度。**

预训练时的联合目标写成：

\[
\ell_{\mathrm{co}}
=
\lambda_v\ell_v+\lambda_a\ell_a.
\]

对于某个 Linear 的输出 \(y_L\)，分别求：

\[
g_v=\nabla_{y_L}\ell_v,\qquad
g_a=\nabla_{y_L}\ell_a.
\]

这些梯度是在问：

> 如果这个 Linear 的输出稍微变化，video loss 和 action loss 会怎样变化？

联合梯度是：

\[
g_{\mathrm{joint}}
=
\lambda_vg_v+\lambda_ag_a.
\]

容易忽略的细节是：

\[
(\lambda_vg_v+\lambda_ag_a)^2
\]

与

\[
\lambda_v^2g_v^2+\lambda_a^2g_a^2
\]

并不相同。前者包含：

\[
2\lambda_v\lambda_ag_vg_a.
\]

这就是 cross-objective term。

用一个单坐标教学例子，设两个 loss 权重都为 1：

| video 梯度 | action 梯度 | 先相加再平方 | 分别平方再相加 |
|---:|---:|---:|---:|
| 1 | 1 | 4 | 2 |
| 1 | -1 | 0 | 2 |

第一行中，两种目标变化方向一致，联合敏感性被加强。

第二行中，两种目标变化方向相反，在这个位置的联合梯度发生抵消。分别平方后再合并，就看不到这种关系。

论文把分别构造两个敏感性因子、然后合并的方法称为 **post-hoc fusion**。QuantWAMs 先组合梯度，再构造 empirical-Fisher 因子，从而保留交叉项。

这里要准确理解：

**它衡量的是原始联合训练目标下的局部敏感性。** 梯度抵消不代表某个位置对所有任务都不重要，也不保证量化它不会影响机器人执行。

实际 loss 权重和 normalizer 保持训练时的设置；上面的等权数字只是教学例子。

**分配精度，还要考虑真正产生的量化误差。**

一个 Linear 的量化残差是：

\[
\epsilon_L^{(b)}
=
Q_b(W_L)-W_L.
\]

论文把三种信息结合起来：

1. **输入是否经常激活这个位置**；
2. **输出变化对联合目标是否敏感**；
3. **使用这个 bit-width 实际会产生多大 weight error**。

在用于评分的 diagonal 近似下，可以把损伤写成：

\[
D_L(b)
\approx
\frac12
\sum_{i,j}
\underbrace{G_{L,ii}}_{\text{输出的目标敏感性}}
\underbrace{\Sigma_{L,jj}}_{\text{输入能量}}
\underbrace{(\epsilon_{L,ij}^{(b)})^2}_{\text{量化误差}}.
\]

其中：

\[
\Sigma_L=\mathbb E[x_Lx_L^\top],
\]

\[
G_L=\mathbb E[g_{\mathrm{joint}}g_{\mathrm{joint}}^\top].
\]

因此，高分并不只是“gradient 大”，而是：

> 这个 weight 的量化误差，经常被输入触发，而且会影响联合目标。

然后计算把 Linear 从 4-bit 升到 8-bit 的收益：

\[
B_L=D_L(4)-D_L(8).
\]

**升级能够减少越多估计损伤的 Linear，越优先获得高精度预算。**

这是一种 empirical-Fisher 近似，不能当成精确的闭环性能预测。

**为什么最终按整个 Linear 分配，而不按每个 weight？**

按单个 weight 分配，看起来更精细。但只有 32 条 calibration trajectories，极细粒度的排名容易被噪声控制。

可以想象：

- 为每个 weight 判断“该不该升级”，需要作出大量细小决策；
- 把很多 weight 的贡献汇总成整个 Linear 的分数，决策更粗，但通常更稳定。

论文最终：

- 用 **Linear 总分**决定精度；
- column 分数只作为 GPTQ 内部排序的 heuristic；
- 不用 element-level 分数直接进行最终精度分配；
- GPTQ 的误差补偿仍由输入统计 \(\Sigma_L\) 控制。

实际升级的是：

**candidate Linears 中排名前 20% 的 Linear，按数量计算。**

例如，论文 Figure 5 展示的 Fast-WAM candidate set 有：

\[
2\text{ 个 expert}
\times10\text{ 类算子}
\times30\text{ 个 block}
=600\text{ 个 Linear}.
\]

其中 120 个升级，480 个保留主要的 4-bit 路径。

这里的“20%”不等于“20% 参数”。大 Linear 和小 Linear 在 count budget 中都算一个。

**为什么有 backward pass，仍然属于 PTQ？**

因为 backward pass 用来测量敏感性，没有进行 QAT 式的模型训练。

因此更准确的称呼是：

**gradient-assisted PTQ。**

它需要原始 video–action co-training targets，也需要 backward pass。不能把它理解成“只收集 activation，完全不需要标签”的 PTQ。

---

**第三个方法：选择保护哪些 denoising step，必须避免“保护已经把敏感性藏起来了”。**

假设一次 action 生成有 10 个 denoising steps，而预算只允许其中 1 个使用 A8。

最直接的想法是：

> 测一下每个 step 的量化误差，保护误差最大的那个。

论文认为，直接测量有两类陷阱。

**第一类陷阱：测量状态不真实。**

Synthetic 或 open-loop 输入可能没有机器人实际执行产生的 observation history。

但闭环中，动作改变环境，环境再改变输入。某个 step 在 synthetic 数据上特别敏感，在实际到达的状态上未必如此；反过来也一样。

所以作者使用真实 **FP16 closed-loop rollouts** 来记录状态。

**第二类陷阱：当前保护会掩盖真实敏感性。**

看一个教学例子：

| step | 统一取消保护后的误差 | 当前配置 | 当前配置下观察到的误差 |
|---|---:|---|---:|
| 1 | 10 | 已使用 A8 | 1 |
| 2 | 4 | 低精度 | 4 |
| 3 | 3 | 低精度 | 3 |

如果只看最后一列，会觉得 step 2 最敏感，于是把保护从 step 1 移走。

但 step 1 看起来误差小，恰恰是因为它已经受到保护。

论文称这种问题为 **self-masking**。

**解决办法，是让所有候选 step 接受相同的测试条件。**

对每份记录下来的 rollout snapshot：

1. 恢复完整、不可变的参考状态；
2. 使用统一的、无 step 保护的低精度方案 \(q_0\)；
3. 与 FP16 在同一份 snapshot 上的输出比较；
4. 对每个 denoising step 汇总误差。

其 profile 是：

\[
S_{\mathrm{replay}}(t)
=
\mathbb E_{x_t\sim\mathcal D_{\mathrm{FP16}}}
\left[
\|f_{q_0}(x_t)-f_{\mathrm{FP16}}(x_t)\|_2^2
\right].
\]

这个公式有两个关键点：

- **状态来自 FP16 的真实闭环 rollout**；
- **干预固定为同一个无保护方案 \(q_0\)**。

因此它叫 **Fixed-Intervention Replay**。

snapshot 不能只包含一张图像。它还要保留 observation history、action chunk position、persistent cache 等状态。

如果先运行一次量化分支，修改了 cache，再用被修改过的 cache 跑 FP16，那么两者就不是在同样条件下比较。论文要求各分支恢复完整 snapshot、深拷贝持久状态，并匹配随机种子。

**profile 只用于提出一个 schedule 候选，不能直接给出闭环收益。**

作者从 profile 中选出误差最大的 K 个 steps：

\[
\mathcal T_{\mathrm{replay}}
=
\operatorname{TopK}_t S_{\mathrm{replay}}(t).
\]

然后：

- 保持保护数量 K 不变；
- 保持精度等级不变；
- 只改变受保护的 step index；
- 在独立的完整 closed-loop validation rollouts 上检查；
- 接受或拒绝这个候选；
- 最后冻结 schedule，再测试。

为什么还需要完整 rollout？

因为局部误差的长期影响取决于环境动力学。论文用一阶形式表达：

\[
\delta s_{j+1}
=
A_j\delta s_j+B_j\epsilon_j.
\]

其中：

- \(\epsilon_j\)：当前模型的局部误差；
- \(B_j\)：这个误差如何影响环境状态；
- \(A_j\)：已有状态偏差如何继续传播。

例如，同样 1 mm 的动作偏差：

- 在自由空间里移动，可能几乎没有影响；
- 在刚接触杯口或积木边缘时，可能改变后续接触过程。

所以：

**step 的局部输出 MSE 大，不等于保护它一定能带来最大的成功率提升。**

还有一个重要边界：profile 使用的是 **FP16 policy 到达的状态分布**，并非量化 policy 的实际部署分布。论文明确保留了这一区别。

---

**把三个方法接起来，整个 calibration 流程就容易理解了。**

论文使用四种互相分开的数据角色：

| 数据角色 | 用来做什么 | 数量或边界 |
|---|---|---|
| \(\mathcal D_{\mathrm{cal}}\) | activation 统计、group 筛选、mask、smoothing、joint saliency、GPTQ | 32 条训练 trajectories |
| \(\mathcal D_{\mathrm{prof}}\) | 构建 fixed-intervention replay profile | 另 32 条 FP16 closed-loop rollouts |
| \(\mathcal D_{\mathrm{val}}\) | 对提出的 schedule 做预先规定的接受／拒绝比较 | 独立完整 rollouts |
| \(\mathcal D_{\mathrm{test}}\) | 报告最终闭环性能 | 所有决策冻结后使用 |

四种角色的 trajectories 和 initial-state seeds 分开。

但是 **task identity 可以重合**。因此，这属于 benchmark-specific、in-distribution calibration，没有建立 unseen-task transfer 的结论。

一个完整执行顺序可以读成：

1. 固定 pretrained checkpoint 和 calibration trajectories。
2. 确定坐标兼容的 pooling candidate groups。
3. 筛选可靠的 groups，拟合高精度 channel mask。
4. 计算联合目标下的 Linear 升级收益，分配 W8A8。
5. 用独立 FP16 rollouts 构建统一干预的 step profile。
6. 提出固定预算的 schedule 候选，进行 closed-loop validation。
7. 冻结配置，进行最终测试。

这也说明，它的成本不只是“用 32 条轨迹跑一遍”。

除了 PTQ fitting，它还需要 labels、backward pass、FP16 rollout profiling 和 schedule validation。

---

**论文表格写 W4A4，但实际配置包含多种高精度保护。**

这一点会直接影响对结果的理解。

| 项目 | 实际配置 |
|---|---|
| 默认计算路径 | W4A4 |
| outlier input channels | top 2% 使用 BF16 exception path |
| candidate Linears | top 20% 按数量升级到 W8A8 |
| Fast-WAM action steps | 10 个中保护 1 个，使用 A8 |
| LingBot-V-A video steps | 20 个中保护 2 个，使用 A8 |
| LingBot-V-A action steps | 50 个中保护 6 个，使用 A8 |

这些保护在 benchmark 上固定，不是为每条测试 trajectory 动态挑选。

论文提到 nominal average weight bits 为：

\[
0.8\times4+0.2\times8=4.8.
\]

但这是 **按 candidate Linear 数量的平均**。它不是 parameter-weighted average，也没有包含所有 metadata 和模型其他组件。

因此，理解结果时应称为：

**W4A4-dominant mixed precision。**

实际后端是：

- NVFP4 W4A4 kernels；
- FP8 W8A8 kernels；
- BF16 channel bypass；
- NVIDIA RTX PRO 5000 Blackwell。

所以，论文的性能结果与具体硬件实现密切相关。

---

**实验最有说服力的部分，是最终评价了 closed-loop task success。**

Tables 1–2 的主要结果如下。这里列出平均成功率；原文还报告了三个 paired protocol seeds 的 sample standard deviation。

| 模型 | Benchmark | FP16 | QuantWAMs | 差距 |
|---|---|---:|---:|---:|
| Fast-WAM | RoboTwin 2.0 average | 91.9% | 91.7% | −0.2 percentage points |
| Fast-WAM | LIBERO 四个 suites average | 97.6% | 97.4% | −0.2 points |
| LingBot-V-A | RoboTwin 2.0 average | 92.3% | 91.6% | −0.7 points |
| LingBot-V-A | LIBERO-Long | 98.5% | 98.0% | −0.5 points |

LingBot-V-A 官方 released checkpoint 只支持 LIBERO-Long，因此不能把它的 98.0% 理解成四个 LIBERO suites 的平均。

这些结果支持：

> 在论文的 benchmark、calibration 流程和 mixed-precision budget 下，模拟环境的成功率点估计接近 FP16。

它们没有进行正式的 equivalence 或 non-inferiority 检验。

**资源结果也需要看测量范围。**

目标 video/action blocks 的 measured peak weight-plus-activation memory：

| 模型 | FP16 | QuantWAMs |
|---|---:|---:|
| Fast-WAM | 14.4 GB | 4.2 GB |
| LingBot-V-A | 13.5 GB | 3.9 GB |

Fast-WAM 的 4.2 GB 约为 FP16 的 29%，即降低约 71%。

但该统计排除了 embedding、projection、VAE 和控制 pipeline 的其余部分。

报告的 **1.4–1.6× speedup**，测量的是一次 model call 中目标 blocks 的 latency ratio。

因此，它不能直接推出：

- 整个模型显存降低 71%；
- 完整 robot cycle 加速 1.6×；
- 控制频率提高 1.6×。

最终动作周期还可能包含感知、数据传输、其他网络组件和机器人执行时间。

---

**Ablation 能帮助判断三个方法分别提供了什么证据。**

先看 Table 3 的 cumulative ladder，全部是 **LIBERO-Long**：

| 配置 | Fast-WAM | LingBot-V-A |
|---|---:|---:|
| Base | 80.8% | 80.2% |
| 加 Shared-Basis | 89.1% | 89.6% |
| 再加 Joint saliency，仍使用 synthetic schedule | 90.9% | 91.8% |
| 再加 Fixed-Intervention Replay | 95.0% | 98.0% |

Base 使用 per-context masks、post-hoc fusion 和 Synthetic Top-K。

这张表显示：按作者指定的顺序加入三个组件，性能逐步提高。

但每一步的提升都依赖前面已经启用的组件。不能把它理解成三个彼此独立、可以任意相加的收益。

另外，Fast-WAM 在这里最终是 95.0%，因为测试的是 LIBERO-Long；前面 97.4% 是四个 suites 的平均。

**关于 pooling，证据支持“筛选后共享”。**

Table 4 比较了：

- 每个 context 单独统计；
- 只共享同 depth 的 paired attention output；
- 不筛选就全局共享；
- 作者的 screened grouping。

一个有意思的结果是：paired-attention pooling 在 Fast-WAM 上有提升，但在 LingBot-V-A 上没有同样的提升。

这与方法的逻辑一致：**架构上可共享，还要看统计上是否值得共享。**

Figure 3 进一步检查小样本 pooling 的风险与 energy recovery，但这些曲线来自 calibration set 的 trajectory resampling，不能当成新的独立泛化测试。

**关于 weight allocation，结果支持这个数据规模下较粗的决策粒度。**

Table 4 的部分结果：

| 分配粒度／最终配置 | Fast-WAM | LingBot-V-A |
|---|---:|---:|
| Element allocation | 72.1% | 69.8% |
| Column allocation | 82.1% | 81.9% |
| Screened grouping + Layer allocation | 95.0% | 98.0% |

这支持论文的解释：在有限 calibration 数据下，细粒度排名可能不可靠。

它没有证明所有模型、所有数据规模下 element allocation 都会更差。

**关于联合梯度，Joint 比 Fusion 更好。**

在相同目标、loss weights、calibration 数据和 precision budget 下：

| Saliency | Fast-WAM | LingBot-V-A |
|---|---:|---:|
| Post-hoc fusion | 91.4% | 92.8% |
| Joint | 95.0% | 98.0% |

分别提高 3.6 和 5.2 percentage points。

这与“保留 cross-objective term 有价值”的解释一致。论文也明确说，闭环结果本身不能单独建立该交叉项的因果归因。

**关于 step schedule，统一干预比直接观察当前配置更有效。**

| Schedule selection | Fast-WAM | LingBot-V-A |
|---|---:|---:|
| Synthetic Top-K | 90.9% | 91.8% |
| Observational Top-K | 93.5% | 93.7% |
| Fixed-Intervention | 95.0% | 98.0% |

保护数量和精度等级相同，只改变保护位置。

这使比较更容易理解：收益来自 schedule placement 的变化，而不是额外增加更多 A8 steps。

---

**真实机器人结果说明可以执行任务，但数据规模较小。**

作者在 AgiBot G2 上测试抓苹果、叠积木和折毛巾，每个任务 10 次：

| 方法 | 抓苹果 | 叠积木 | 折毛巾 | 总成功数 |
|---|---:|---:|---:|---:|
| FP16 | 8/10 | 6/10 | 5/10 | 19/30 |
| Atom* | 5/10 | 4/10 | 3/10 | 12/30 |
| QuantWAMs | 8/10 | 5/10 | 4/10 | 17/30 |

QuantWAMs 比 Atom* 更好，三个任务都可以执行；相对 FP16，总共少成功两次。

由于每个任务只有 10 次，适合把它看成 feasibility evidence，不能据此认定与 FP16 等效。

这里的 1.4× 仍然是目标 blocks 的加速。

---

**读这篇论文时，最需要保留的判断是：它优化了三个局部 surrogate，再用闭环测试检查结果。**

三个 surrogate 分别是：

| 方法 | 优化或测量的对象 | 与最终闭环成功率之间的距离 |
|---|---|---|
| Shared-Basis | channel energy 与 mask 稳定性 | energy 高不一定等于控制上重要 |
| Joint saliency | 联合训练目标下的局部量化损伤近似 | training-loss 敏感性不等于 task-success 敏感性 |
| Fixed-Intervention | 同一 snapshot 上的一次局部输出差异 | 局部 MSE 不包含完整环境误差传播 |

它们的价值，在于改善校准决策的可靠性；最终是否有效，仍然需要看 closed-loop evaluation。

还有两个阅读边界：

- Atom*、SVDQuant* 控制了 nominal precision budget，但没有获得 QuantWAMs 使用的全部额外 gradients、profiling rollouts 和 validation 信息。因此，同预算比较不等于同校准信息或同校准成本比较。
- 本地 v1 正文引用了 Appendices A–E，但这份 13 页 PDF 到 references 结束，未包含这些 appendices。正文足够理解方法主线，完整 grouping estimator、replay 和 measurement 实现细节仍有缺口。

---

**回到 PDF，建议按下面的顺序读。**

1. **第 3 页 Figure 2**：先只找 channel、Linear、step 三条决策线，不急着理解所有符号。
2. **第 4 页 Eq. 6–7**：抓住“pooling 减少 variance，同时引入 heterogeneity bias”。
3. **第 5 页 Eq. 12–15**：重点看“先组合梯度再平方”和“升级减少的实际量化损伤”。
4. **第 6 页 Eq. 19**：找出三种 profile 的两个区别——状态分布与干预精度。
5. **第 8 页 Quantization configuration 和 measurement scope**：确认真正使用了哪些高精度保护，以及速度测量到哪里。
6. **第 9–10 页 Tables 3–5**：检查每个对照到底固定了什么、改变了什么。

读完后，尝试用自己的话回答：

- 两个模块 shape 相同，为什么还不能随便共享 channel mask？
- calibration 数据增多后，为什么 pooling 可能反而变差？
- 为什么 \((g_v+g_a)^2\) 与 \(g_v^2+g_a^2\) 代表不同敏感性？
- 已经受到保护的 step，为什么可能被误判为不重要？
- 为什么 local replay MSE 还需要完整 closed-loop validation？
- 表中的 W4A4 和 1.6×，各自包含哪些具体范围？

---

**补充：module、channel、Linear、step 分别是什么？**

第一个方法里的 module，主要指被校准的具体算子，例如某个 block 的 `self_attn.o` 或 `cross_attn.o` Linear；共享的是这些算子输入的 activation 统计和 channel mask。不是把整个 video DiT 与整个 action DiT 各当成一个 module，然后所有 channel 一起平均。

论文进一步使用 context 描述“哪个 module、什么 execution mode、哪个 depth”。LingBot-V-A 的同一个物理 Linear，可以在不同 mode 下产生不同 calibration contexts；不同 context 不一定是不同参数模块。

| 概念 | 具体含义 | 属于哪条轴 |
|---|---|---|
| Input channel | Linear 输入特征向量的一维，对应 weight matrix 的一列 | 结构轴 |
| Linear | 一个完整的线性算子，包含许多 input/output channels | 结构轴 |
| Transformer block | 包含 attention、FFN 及其中多个 Linears | 结构轴 |
| DiT | 由多个 blocks 等组件组成 | 结构轴 |
| Denoising step | sampler 中的一次去噪迭代，通常会调用包含多个 blocks 和 Linears 的网络 | 时间轴 |

channel、Linear、block、DiT 可以理解成结构上逐步变大的范围；**step 属于时间轴，因此 channel → Linear → step 不是严格的包含层级。** 同一个 Linear 的权重通常会在多个 denoising steps 重复使用，而每个 step 的 activation 会随状态改变。

channel 保护与 Linear precision allocation 都面向论文的目标 video/action DiT 路径。对 Fast-WAM，它们覆盖两个 experts；对 LingBot-V-A，它们作用于两个 token 路径复用的 shared backbone。这里的 video/world DiT 也不等于模型中所有视觉组件，例如独立 image encoder 或 VAE。

“两条路径都量化”不意味着“两条路径的所有 channel mask 都共享”。只有坐标兼容、通过统计筛选的 contexts 才共享；其余 contexts 保持各自的校准决策。

一个教学例子：假设某个 Linear 有 100 个 input channels，输入 channel 7 和 42 走 BF16 旁路；这个 Linear 又因为 layer score 高而获得 W8A8；同一个网络在若干 denoising steps 重复执行，某个受保护 step 使用 A8。这三种决策沿不同轴共同定义混合精度配置，BF16 channel exception 仍是高精度例外。

---

**Meeting Card（留空，阅读后填写）**

- 我要解决的具体问题：
- 模型的训练 / 测试输入和输出：
- 最关键机制与证据位置：
- 数据、checkpoint、benchmark 与 precision 条件：
- 一个重要局限或未复现点：
- 与我的量化 idea 重合的部分：
- 我准备验证的不同假设：
- 想向导师/师兄确认的问题：
