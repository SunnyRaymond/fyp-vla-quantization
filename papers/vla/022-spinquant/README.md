# 022. SpinQuant：通过学习旋转，让 LLM 更容易量化

**SpinQuant: LLM Quantization with Learned Rotations**  
阅读版本：**arXiv:2405.16406v4，2025-02-20，24 页**；ICLR 2025。讲解整理：2026-10-08。

[固定版本 PDF](paper-arxiv-v4.pdf) · [固定版本 arXiv](https://arxiv.org/abs/2405.16406v4) · [官方代码](https://github.com/facebookresearch/SpinQuant) · [返回目录](../README.md)

作者：Zechun Liu、Changsheng Zhao、Igor Fedorov、Bilge Soran、Dhruv Choudhary、Raghuraman Krishnamoorthi、Vikas Chandra、Yuandong Tian、Tijmen Blankevoort。

本 README 可以独立阅读，按“问题直觉 → 可计算的例子 → 网络中的实现 → 优化公式 → 实验如何解读”的顺序展开。页码均为上述 PDF 的**物理页码，从第 1 页开始数**。实验数字来自论文，本地没有训练 rotation、运行量化或复现 benchmark。标为“教学例子”“教学推导”的内容用于帮助理解，不是作者的实验。

原目录的 [paper.pdf](paper.pdf) 是此前保存的 ICLR proceedings 版本，继续保留；本文的页码和数字统一依据新保存的 arXiv v4。

## 1. 先抓住整篇论文的问题

假设模型内部有一组特征，大多数数值都不大，但少数 channel 经常出现很大的值。低位数量化要用有限的表示点覆盖这些值，范围被少数大值拉宽后，普通值之间的区别就容易消失。

SpinQuant 的想法是：**模型原本会做什么可以保持不变，但内部特征用什么坐标表示，可以改变。找到更适合量化的坐标系，就能减少量化后的损伤。**

这里的“旋转”是在 hidden feature space 中做矩阵变换，不是旋转输入图像，也不是改变机器人的运动方向。

更具体地说，论文分两步回答问题：

1. 哪些 rotation 可以放进 Transformer，并在 full precision 下保持原函数？
2. 在这些合法 rotation 中，哪些能让量化后的模型最好？

第一步提供可选择的空间，第二步通过 calibration loss 选择。论文发现，随便选一个 random rotation 并不稳定，因此提出**学习 rotation**。

阅读时先记住一个区别：

> “旋转前后 full-precision 模型等价”，与“旋转前后 quantized 模型等价”，是两件事。SpinQuant 正是利用第二件事不成立来改善量化。

## 2. 阅读所需的基本概念

### 2.1 Weight、activation 和 KV cache 各是什么？

本文使用 row-vector 约定，一个 Linear 写为：

\[
y=xW.
\]

其中 \(x\in\mathbb R^{1\times d_{in}}\) 是输入 activation，\(W\in\mathbb R^{d_{in}\times d_{out}}\) 是 weight。PyTorch 的参数存储通常使用转置形状；阅读公式时要先确认约定，不要直接照着参数 shape 判断左右乘顺序。

| 对象 | 来源与用途 | 是否随输入改变 |
|---|---|---|
| Weight \(W\) | 预训练学到的参数，如 Q/K/V projection 的矩阵 | 推理时通常固定 |
| Activation \(x\) | 当前 token 在某层的中间特征 | 会改变 |
| KV cache | 把历史 token 的 key/value activation 保存下来，供后续 decode 重用 | 随序列增长 |

因此，量化 \(W_K\) 不等于量化 K cache。前者压缩投影参数，后者压缩投影之后保存的中间数据。

**W4A4KV4** 表示论文所指的 weight、activation 和 KV cache 量化路径使用 4-bit。它不是在说 RMSNorm、RoPE、Softmax、所有 residual 运算、输出与累加器都使用 4-bit；也不能仅凭这个缩写断言某个 GPU 上实际执行了 native INT4 matrix multiplication。

### 2.2 Token、channel、head、block 不要混淆

假设 activation 为 \(X\in\mathbb R^{T\times d}\)：

- 一行对应一个 token；token 是序列位置。
- 一列对应一个 hidden channel；channel 是特征维度。
- 一个 attention head 在自己的子空间内计算 attention，head dimension 记为 \(d_h\)。
- 一个 Transformer block 包含 attention、FFN、normalization、residual 等部分，一个 block 中有多个 Linear。

论文的 \(R_1\) 混合 residual stream 的 hidden channels；\(R_2\) 混合 attention head 内的 feature channels。它们不是打乱 token 的顺序。

### 2.3 Prefill 和 decode

Prefill 一次处理输入 prompt，建立各层的 KV cache。Decode 逐个产生新 token，并读取和追加 cache。

这是两个不同的性能阶段。后面实验中的 ms/token 描述 decode；TTFT 描述到第一个 token 的时间。它们不能替换成同一个“模型延迟”。

### 2.4 PTQ、QAT 与 rotation learning

PTQ 在模型预训练完成后准备量化；QAT 通常指把量化影响纳入模型训练/微调，使模型参数适应量化。

SpinQuant 在论文中归为 PTQ，但**它需要 calibration、backpropagation 和 rotation optimization**，不能称为 tuning-free 或 training-free。原始 pretrained weights 冻结，学习的是受约束的 rotation；最终旋转可以并入权重，full-precision 函数仍等价。

“有梯度优化”与“是不是普通的 weight QAT”不是同一个判断。最准确的描述是：**frozen-weight、optimization-based PTQ，通过学习等价重参数化来适应量化。**

## 3. 为什么 outlier 会伤害低位数量化？

先用对称 INT4 教学量化器说明。取整数网格 \(-7,\ldots,7\)，让一组数共用 scale：

\[
s=\frac{\max_i|x_i|}{7},\qquad
q_i=\operatorname{clip}\left(\operatorname{round}(x_i/s),-7,7\right),\qquad
\widehat x_i=sq_i.
\]

这是一种便于手算的约定；实际 INT4 可用的整数端点和具体 clipping 规则由实现决定。

如果一组值都落在 \([-1,1]\)，步长约为 \(0.143\)。若同组中出现一个 20，步长约为 \(2.857\)。此时 0.2、0.5、0.8 很可能被映射到同一表示点。

困难不只是“大数本身有误差”，而是它让**共享 scale 的普通值失去分辨率**。

论文 Eq. (1)，第 3 页，也给出 asymmetric min-max 形式：

\[
s=\frac{x_{max}-x_{min}}{2^b-1},\qquad
\widehat x=s\operatorname{round}\left(\frac{x-x_{min}}{s}\right)+x_{min}.
\]

这里 \(b\) 是 bit-width，\(s\) 是相邻表示点之间的间距。不要把此处的 scale 与 RMSNorm 的 gain、Cayley 的 learning rate 混为同一个参数。

旋转的作用，是把集中在少数坐标上的能量分散到多个坐标，改善量化范围的使用。它不消除信号能量，也不直接删掉 outlier 所携带的信息。

## 4. 为什么旋转不改变 Linear 的 full-precision 输出？

### 4.1 正交矩阵的关键性质

对于方阵 \(R\)，如果：

\[
R^TR=RR^T=I,
\]

就有 \(R^{-1}=R^T\)。它保持向量的长度和 inner product：

\[
\|xR\|_2=\|x\|_2,\qquad
(xR)(zR)^T=xz^T.
\]

论文使用的 orthogonal transforms 包括通常意义的旋转，也可以包含反射；理解方法时抓住正交性质即可，不必要求每个矩阵都对应三维空间的转动。

### 4.2 一对变换如何抵消？

原来是 \(y=xW\)。把输入改成 \(x'=xR\)，权重改成 \(W'=R^TW\)：

\[
x'W'=xRR^TW=xW=y.
\]

这就是 function-preserving reparameterization。坐标变了，函数没有变。

但量化以后：

\[
\widehat y_R=Q(xR)\,Q(R^TW)
\]

一般不等于 \(Q(x)Q(W)\)，因为量化含有 rounding、clipping、由数据决定的 scale，通常：

\[
Q(xR)\ne Q(x)R.
\]

因此原本等价的一族 FP 模型，变成了质量不同的一族 quantized 模型。

把误差写得更具体，令：

\[
Q(xR)=xR+E_x,
\qquad Q(R^TW)=R^TW+E_W,
\]

则有教学分解：

\[
\widehat y_R-y
=E_xR^TW+xRE_W+E_xE_W.
\]

这里的误差 \(E_x,E_W\) 都依赖 \(R\)。哪怕原始 FP 输出固定，量化误差仍然会随坐标系变化；而且 activation 与 weight 的误差会共同作用。

### 4.3 一个可计算的二维例子

**以下是教学数字，不是论文实验。** 取：

\[
x=(8,1.5),\qquad
R=\frac1{\sqrt2}\begin{bmatrix}1&1\\-1&1\end{bmatrix}.
\]

不旋转时，前面的对称 INT4 量化器有 \(s=8/7\)，得到：

\[
Q(x)=(8,1.1429),\qquad
\|Q(x)-x\|_2\approx0.3571.
\]

旋转后：

\[
xR=\left(\frac{6.5}{\sqrt2},\frac{9.5}{\sqrt2}\right)
\approx(4.5962,6.7175).
\]

此时 \(s'=6.7175/7\approx0.9596\)，整数码为 \((5,7)\)。先量化，再旋转回去：

\[
Q(xR)R^T\approx(8.1429,1.3571),
\]

\[
\|Q(xR)R^T-x\|_2\approx0.2020.
\]

这个例子说明，旋转可以降低同一 bit-width 下的误差。

但它不是保证：若把输入改为 \((8,1)\)，沿用同一个 \(R\) 和量化规则，不旋转误差约为 \(0.1429\)，旋转后约为 \(0.4041\)，反而变大。

**“能量更均匀”是有用直觉；最终 rounding error 与网络任务 loss 才决定某个 rotation 是否合适。** 这也解释了为什么 SpinQuant 要学习，而不满足于“任意正交矩阵”。

## 5. Transformer 中的旋转要怎样放，才真的等价？

任意找一个 tensor 乘 \(R\) 并不够。网络含有 residual addition、normalization、RoPE 和非线性，必须在能抵消的位置放成对变换。

论文第 2 页 Figure 1 是整篇最重要的结构图。先按下面四行认位置，再读第 4–6 页的推导。

| Rotation | 作用位置 | 主要改善什么 | 是否学习 | 推理时如何处理 |
|---|---|---|---|---|
| \(R_1\) | 全网 residual stream 的 hidden basis | 读取 residual 的 Q/K/V、up/gate 等 Linears 的量化 | 是 | 并入 embedding、输入/输出 projections 和最终 head |
| \(R_2\) | 每层 attention head 内的 value 与 output projection 配对 | V cache、attention output projection 输入 | 是 | 并入 \(W_V\)、\(W_O\) |
| \(R_3\) | RoPE 之后的 Q、K 配对变换 | K cache 的 outliers | 否，保留 Hadamard | Q/K 在运行时做变换 |
| \(R_4\) | FFN 非线性及逐元素乘积之后、down projection 之前 | down projection 的输入 activation | 否，保留 Hadamard | activation 在线变换，逆变换并入 down weight |

### 5.1 \(R_1\)：换掉 residual stream 的共同坐标系

如果 residual 中一直保存 \(xR_1\)，读取它的 Linear 可以使用 \(R_1^TW\)，还原原本的投影结果；分支输出则需要转换回共同的 rotated residual basis。

Residual addition 必须两条分支使用同一 basis：

\[
(x+f(x))R_1=xR_1+f(x)R_1.
\]

如果一条分支旋转、另一条不旋转，直接相加就不再等价。\(R_1\) 是跨 residual 路径一致的变换，不是每个模块随意挑自己的坐标系。

最终的输出 head 再消去这个 basis change，所以 logits 不变。

### 5.2 RMSNorm 为什么需要先处理 gain？

先看不带 gain 的 RMS normalization：

\[
N(x)=\frac{x}{\sqrt{\|x\|_2^2/d+\epsilon}}.
\]

因为正交变换保持范数：

\[
N(xR)=N(x)R.
\]

但实际 RMSNorm 还有逐 channel gain \(D_\gamma\)。一般情况下：

\[
D_\gamma R\ne RD_\gamma.
\]

所以不能只说“RMSNorm 对旋转不敏感”就忽略 gain。论文第 5 页脚注说明，先把 gain 并入后续 weight；row-vector 约定下 \(N(x)D_\gamma W=N(x)\widetilde W\)，其中 \(\widetilde W=D_\gamma W\)。之后再做旋转等价变换。

这个步骤依赖模型结构。对不同 normalization、bias、额外分支或 tied parameters，不能直接套用一句“所有 Transformer 都旋转等价”。

### 5.3 \(R_2\)：V 与 output projection 如何配对？

在单个 head 中，令 attention weights 为 \(A\)，原来的 value 为 \(V\)，输出为：

\[
O=AV.
\]

把 value channels 旋转为 \(V'=VR_2\)：

\[
O'=AVR_2=OR_2.
\]

随后把 output projection 改为 \(R_2^TW_O\)，就得到：

\[
OR_2R_2^TW_O=OW_O.
\]

为什么 attention 中的 Softmax 不破坏这个等价性？因为这条变换作用于 V，Softmax 产生的 \(A\) 由 Q/K 决定；\(A\) 在 token 轴上混合，\(R_2\) 在 feature 轴上混合，二者可以按结合律重排。

\(R_2\) 的尺寸是 \(d_h\times d_h\)，不同 layers 分别学习。第 5 页文字说它 head-wise 使用，不应凭此进一步假定每个 head 都有一套独立学习的矩阵。

### 5.4 \(R_3\)：为什么放在 RoPE 之后？

令 \(\overline Q,\overline K\) 是已经过 RoPE 的 Q/K。使用相同正交变换：

\[
Q'=\overline Q R_3,\qquad K'=\overline K R_3,
\]

则：

\[
Q'K'^T=\overline Q R_3R_3^T\overline K^T
=\overline Q\overline K^T.
\]

attention logits 不变，Softmax 也不变，同时旋转后的 K 可以更容易量化。这里不能只旋转 K，否则 query–key dot product 会改变。

RoPE 是随 token position 改变的变换。一般 dense rotation 不能自由穿过它，所以 Figure 1 的位置关系有意义：**先 RoPE，再对 Q/K 做配对 Hadamard。**

这条路径的 Q/K 变换需要在线执行，无法简单全部并回 RoPE 之前的静态 projection weight。

### 5.5 \(R_4\)：为什么不能把它直接移到 FFN 前面？

用 row-vector 写 gated FFN 的中间 activation：

\[
u=\operatorname{SiLU}(xW_{gate})\odot(xW_{up}),\qquad y=uW_{down}.
\]

可以把 \(u\) 改成 \(uR_4\)，同时把 down weight 改成 \(R_4^TW_{down}\)：

\[
uR_4R_4^TW_{down}=uW_{down}.
\]

但通常：

\[
\operatorname{SiLU}(zR)\ne\operatorname{SiLU}(z)R,
\]

逐元素乘法也不能随意与 dense rotation 交换。因此 \(R_4\) 必须放在中间非线性计算之后，运行时先旋转 activation，再进入量化/down projection。

### 5.6 Hadamard 为什么适合在线计算？

归一化 Hadamard 的元素为 \(\pm1/\sqrt d\)，是一种结构化正交变换。对支持的维度，fast Hadamard transform 可以用类似 butterfly 的加减过程完成，复杂度为 \(O(d\log d)\)，而一般 dense rotation 是 \(O(d^2)\)。

因此 \(R_1,R_2\) 可以学习成一般矩阵并离线融合，\(R_3,R_4\) 保持可高效在线执行的 Hadamard。论文不是把四个 rotation 都学习成任意 dense matrix 再在每次推理时执行。

## 6. no_had 与 had 两个版本

| 版本 | 组成 | 使用上的主要考虑 |
|---|---|---|
| SpinQuant_no_had | 学习并融合 \(R_1,R_2\) | 较少额外运算；A8 场景已经很好 |
| SpinQuant_had | 学习并融合 \(R_1,R_2\)，另加在线 \(R_3,R_4\) | A4/KV4 更困难时，进一步保护 block 内部量化 |

“no_had 没有 online rotation”不意味着量化 kernel、packing、scale 管理也没有成本；它只是无需额外执行这些 rotation。

“had 更准确”也不是对每个模型、每个精度都严格成立。后面的实验会看到，在 A8 条件下收益较小，而 A4 更明显。

## 7. 学习 rotation 的 objective 是什么？

论文第 5 页 Eq. (2) 写成：

\[
\min_{R_1,R_2\in\mathcal M}
\mathcal L_Q(R_1,R_2\mid W,X).
\]

逐个解释：

| 符号 | 含义 |
|---|---|
| \(W\) | 固定的 pretrained weights |
| \(X\) | calibration 输入 |
| \(R_1,R_2\) | 要优化的 rotation |
| \(\mathcal M\) | 正交约束集合，论文用 Stiefel manifold 描述 |
| \(Q\) | 网络中的量化过程 |
| \(\mathcal L_Q\) | 量化网络的任务 loss，例如语言模型 cross-entropy |

对这里的 square matrices，约束就是 \(R^TR=I\)；一般 Stiefel manifold 还可以包含矩形的列正交矩阵。

它优化的是量化网络的最终 task loss，不是单纯把所有 activation 的 max value 压到最小，也不是给每一层独立最小化 MSE。

这点非常关键：某层的局部误差是否重要，取决于它怎样影响后面的网络。第 19 页 Figure 7 甚至显示，学习后少数层的 SNR 提升较大，多数层变化不大，还有个别层变差；局部指标不需要每层都同步改善。

### 7.1 Quantization 不是可微的，怎样反向传播？

rounding 的真实导数几乎处处为零。用于优化的 fake quantization 在 forward 中模拟 rounding 后再 dequantize，backward 则用 straight-through estimator（STE）传递近似梯度。

例如 forward 使用 \(\widehat x=s\operatorname{round}(x/s)\)，backward 在适用范围内近似把它当作恒等变换来传梯度。它不意味着 rounding 真的光滑，只是提供一个可优化的 surrogate。

官方 [quant_utils.py](https://github.com/facebookresearch/SpinQuant/blob/main/utils/quant_utils.py) 中有 STEQuantize/AsymSTEQuantize；这是代码侧补充，2026-10-08 查阅，未固定代码 commit，也未在本地执行。该文件的 activation quantizer 支持 per-token 及 token 内 group-wise 设置；实际 granularity 仍应随运行配置记录。

Fake quantization 可以评价数值和学习 rotation，不能单独证明显存已压缩成 INT4，或模型已经获得部署加速。

### 7.2 为什么不去优化 full-precision loss？

若完全没有量化、等价变换正确，那么不同合法 \(R\) 产生同一个 FP 函数，loss 对 rotation 没有可利用的差异。加入量化以后等价性被 rounding 打破，才出现可以选择的方向。

第 18–19 页 Appendix B.1 用单层 gradient 分析说明这一点：有量化时 gradient 一般非零，移除量化时归零。这里的“非零”针对优化所用的量化梯度处理，不能把它读成真实 rounding 函数到处有普通导数。

## 8. Cayley optimization：怎么更新才不丢掉正交性？

普通 SGD：\(R'=R-\eta G\)，通常不满足 \(R'^TR'=I\)。如果正交性丢掉，前面依赖 \(R^{-1}=R^T\) 的 function-preserving 配对就会受影响。

SpinQuant 使用论文第 5–6 页 Eqs. (3)–(4)：

\[
R'=C(Y)R,
\qquad
C(Y)=\left(I-\frac\alpha2Y\right)^{-1}
\left(I+\frac\alpha2Y\right),
\]

\[
G=\nabla_R\mathcal L_Q,
\qquad
\widehat G=GR^T-\frac12RR^TGR^T,
\qquad
Y=\widehat G-\widehat G^T.
\]

这里 \(\alpha\) 是更新 step size；它与论文 Eq. (1) 量化公式中使用的同名字母不是同一个量。

理解时分三步：

1. \(G\) 是 loss 告诉我们“rotation 应怎样改变”的原始梯度。
2. 构造 \(Y\) 后，自动有 \(Y^T=-Y\)，称为 skew-symmetric。
3. 把 \(Y\) 送进 Cayley transform，得到正交的更新矩阵 \(C\)，再左乘当前 \(R\)。

**为什么 \(C\) 正交？** 令 \(A=I-\alpha Y/2\)、\(B=I+\alpha Y/2\)。由 \(Y^T=-Y\)，有 \(A^T=B\)、\(B^T=A\)。同时 \(A,B\) 都是 \(Y\) 的多项式，因此彼此可交换。于是：

\[
C^T=(A^{-1}B)^T=AB^{-1}=B^{-1}A=C^{-1}.
\]

所以：

\[
R'^TR'=R^TC^TCR=R^TR=I.
\]

这就是“沿着合法的正交矩阵集合更新”的含义，不需要先学会完整的 differential geometry 才能读懂主线。

公式出现 inverse，并不表示实现一定每步直接算一个昂贵的显式矩阵逆。第 6 页说明采用 fixed-point iteration 计算更新；每迭代的计算量约为 naive SGD 的两倍是作者的算法说明，不是整个 calibration 或部署速度的倍数。

## 9. 主实验的完整 pipeline：一个很容易读错的细节

第 6 页 Section 4.1 与第 8 页 Table 3 应一起读。

主结果的流程是：

1. 把 RMSNorm gain 等预处理好，建立合法的旋转参数化。
2. 用 random Hadamard 初始化可学习的 \(R_1,R_2\)。
3. **在 weights 仍为 16-bit、activation 按目标设置量化的网络中，优化 rotation。** KV 设置也按目标场景处理。
4. 冻结 rotation，生成融合后的 weights。
5. **对融合后的 weights 用 GPTQ 做最终低位数量化。**
6. 根据 no_had/had 版本执行在线 Hadamard，评价最终模型。

所以最终 W4A4KV4 的 row，不代表优化 rotation 时已经使用最终 GPTQ 的 W4 weights。

为什么这样做？GPTQ 专门补偿 weight quantization error，不能消除 activation quantization error。作者把 rotation learning 的主要任务留给 activation，把随后 weight 的处理交给 GPTQ。它是一种方法组合，不应把最终表现全部归因于其中一个组件。

如果还不熟悉 GPTQ，可以把它理解为：利用 calibration activations 估计某个 Linear 的 weight 误差怎样影响输出，按顺序量化 weights，并调整尚未量化的 weights 来补偿误差。它关心的是投影结果，而不只是每个 weight 自身的 rounding 距离。这里不需要先推完整 GPTQ，先理解它与 rotation learning 的分工即可。

| 项目 | 论文主设置 | 位置 |
|---|---|---|
| Rotation calibration | 800 个 WikiText-2 samples | 第 6 页 |
| Rotation iterations | 100 iterations | 第 6 页 |
| Learning rate | 从 1.5 线性下降到 0 | 第 6 页 |
| GPTQ calibration | 128 条 WikiText-2 sequences，每条 length 2048 | 第 6 页 |
| 可学习 rotation 参数量 | 论文报告约为 weights 的 0.26% | 第 6 页 |
| Activation/KV range | asymmetric min-max，主实验选择不 clipping | 第 14 页 A.4；第 18 页 Table 12 |

800 samples 与 128 sequences 属于两次不同准备步骤；不要把它们写成一个数据集的大小，也不要把 100 iterations 改写成 100 epochs。

## 10. 实验先看比较对象，再看数字

### 10.1 Avg. 与 PPL 分别意味着什么？

主要质量评价为：

- **0-shot8 Avg.，越高越好**：BoolQ、PIQA、SIQA、HellaSwag、WinoGrande、ARC-easy、ARC-challenge、OpenBookQA 八个任务的 accuracy 均值。
- **WikiText-2 PPL，越低越好**：语言建模的 perplexity，衡量给真实文本分配概率的情况。

PPL 可以理解为模型预测下一个 token 时的平均“困惑程度”。对真实 tokens 的概率 \(p(x_t\mid x_{<t})\)，常见定义为 \(\exp[-\frac1T\sum_t\log p(x_t\mid x_{<t})]\)：给真实文本更高的概率，PPL 就更低。它不是“错误 token 的百分比”。

八任务均值不是一个单独数据集的成功率，也不是八类机器人任务的成功率。PPL 不能用相同方式计算 percentage-point 差异；66.9→64.0 则可以说下降 2.9 percentage points。

Table 1 的小模型列明确写为 **LLaMA-3.2 1B/3B**；正文有更宽泛的 LLaMA-3 命名。引用具体表行时应保留表中的版本，不把所有模型统称为同一个 checkpoint。

### 10.2 Table 1：最重要的结果是什么？

第 7 页，LLaMA-2 7B：

| 方法 | W-A-KV | 八任务 Avg. ↑ | WikiText-2 PPL ↓ |
|---|---|---:|---:|
| Floating point | 16-16-16 | 66.9 | 5.5 |
| SpinQuant_no_had | 4-8-16 | 65.7 | 5.8 |
| SpinQuant_had | 4-8-16 | 65.7 | 5.7 |
| SmoothQuant | 4-4-4 | 39.0 | 表中约 \(7\times10^2\) |
| LLM-QAT | 4-4-4 | 44.9 | 14.9 |
| SpinQuant_no_had | 4-4-4 | 56.0 | 9.2 |
| SpinQuant_had | 4-4-4 | 64.0 | 5.9 |

这里可以得到三个有边界的结论：

1. 在 W4A8KV16 下，no_had 的均值距 FP 仅 1.2 pp，had 没有增加这个均值。
2. 在 W4A4KV4 下，had 比 no_had 高 8.0 pp，说明 block 内部的在线变换在更激进的设置下很重要。
3. had 的 W4A4KV4 均值距 FP 2.9 pp，比该表 LLM-QAT 高 19.1 pp、比 SmoothQuant 高 25.0 pp。仍然存在退化，不能称为完全无损。

这些结论只针对相应模型、calibration 和评测条件。它不证明所有 QAT 方法都比 PTQ 差，也不证明任何模型的 W4A4 都能达到同样差距。

另外，表中的 AWQ/OmniQuant/QuIP# 有 weight-only 标记。它们不能直接当作与 W4A4KV4 完全相同 precision budget 的对照。作者自行运行的 baseline 与从其他论文引用的结果也有来源差别，Table 1 caption 已说明。

### 10.3 Figure 4 和 Table 2：为什么要学习，而不是随机选择？

第 4 页 Figure 4 比较 LLaMA-2 7B W4A4 的不同 rotation。作者报告 100 次随机试验，random floating-point rotation 的最好与最差约差 13 points，random Hadamard 也可差约 6 points。学习后的分布更集中、表现更好。

这说明 random seed 是实际影响结果的变量；不是一个可以不记录的小细节。图中“最高最低差距”不是标准差，也不是说每个模型都随机波动 13 pp。

第 8 页 Table 2，在四种 rotation 都存在的条件下：

| 模型 / precision | Random Hadamard \(R_{1,2,3,4}\) | SpinQuant_had | 差值 |
|---|---:|---:|---:|
| LLaMA-3 8B，W4A4KV4 | 63.9 | 65.5 | +1.6 pp |
| Mistral-7B，W4A4KV4 | 52.4 | 68.6 | +16.2 pp |

同一个研究问题在不同模型上收益差别很大。Mistral 的 +16.2 不能写成 SpinQuant 在所有模型上的固定收益。Table 2 是最直接检验“learned 是否优于 random”的对照，比只与完全不旋转的模型比较更有针对性。

### 10.4 Table 3：为什么学习时先保持 W16？

第 8 页，最终都用 GPTQ 生成 W4，LLaMA-2 7B：

| 最终 precision | 学习 rotation 时 W4 | 学习 rotation 时 W16 |
|---|---:|---:|
| W4A4KV16，Avg. | \(61.0\pm1.0\) | \(64.1\pm0.4\) |
| W4A4KV4，Avg. | \(60.9\pm0.6\) | \(64.0\pm0.3\) |

对应 PPL 分别为 6.7→5.9、6.8→5.9。比较改变的是 **rotation optimization 所看到的 weight precision**，不是最终模型少量化了一部分。

这支持作者的分工策略，但不表示其他量化器、模型或 loss 下总应选择同样的做法。表中的 \(\pm\) 按原表保留；这里没有重新计算 confidence interval，也不据此宣称统计显著。

### 10.5 Table 5：与 QuaRot 的比较怎样读？

只取第 9 页中 GPTQ 和 bit 设置对齐的两组：

| 模型，W4A4KV4 | FP Avg. / PPL | QuaRot+GPTQ | SpinQuant_had+GPTQ |
|---|---|---|---|
| LLaMA-3 8B | 69.6 / 6.1 | 63.3 / 8.0 | 65.5 / 7.3 |
| LLaMA-3 70B | 74.5 / 2.8 | 65.1 / 20.2 | 69.3 / 5.5 |

Avg. 分别提高 2.2 pp 和 4.2 pp。尤其 70B 的 PPL 改善很明显。

这是**完整方法配置的比较**，两者 rotation placement 与 online operations 也有差异，不是仅把一个矩阵从 random 换成 learned 的纯单变量实验。需要纯 learned/random 对照时，优先看 Table 2。

Abstract 还给出相对 gap reduction 的宣传数字；读论文时优先记录可直接从表核算的 absolute scores 和 pp 差异，避免把 relative gap reduction 当作 accuracy 增加的 percentage points。

### 10.6 Table 16 和 Figure 7：收益全来自 GPTQ 吗？

第 18 页 A.8 与第 20 页 Table 16 提供 RTN/GPTQ 对照，说明 learned rotation 搭配简单 RTN 也能改善表现，GPTQ 在此基础上继续提升。因此“SpinQuant 的收益就是 GPTQ”不符合消融结果；反过来，主表既用了 GPTQ，也不能全部归给 rotation。

第 19 页 Figure 7 / 第 20 页 Table 19 进一步看 SNR。作者报告 random rotation 带来 3.8 dB 改善，学习再增加 5.9 dB。SNR 是 signal-to-quantization-noise 指标；它支持误差改善机制，但不是八任务 accuracy，也不是闭环成功率。

## 11. 性能与准备成本：三种数字分别读

### 11.1 MacBook：这是 W4A8 decode 的 end-to-end 结果

第 10 页 Table 6，MacBook **M1 Pro CPU**，LLaMA-3 8B：

| 方法 | W-A | Decode latency |
|---|---|---:|
| Floating point | 16-16 | 177.15 ms/token |
| no_had | 4-8 | 58.88 ms/token |
| had | 4-8 | 63.90 ms/token |

按表中数字计算，no_had 约 3.01×，had 约 2.77×；had 比 no_had 增加约 8.5% latency，与正文“约 8%”相符。

**这不是 W4A4KV4 的速度实验。** Table 6 只列 W-A，不应自行补成 KV4。官方 repository 的 ExecuTorch 导出路径也以 W4 和 dynamic A8 为支持设置，不能把它的实际部署速度移到主表 A4/KV4 的质量行上。

### 11.2 H100：这里测的是 FP8，不是 INT4

第 14 页 A.6 明确说明使用 **W-FP8-A-FP8**，FP8 GEMM 来自 FBGEMM，在线 Hadamard 使用 Tensor Core-based kernel。

第 19 页 Table 14，LLaMA-3 70B，H100，sequence length 4096，取 batch=1：

| 配置 | TTFT | TTIT |
|---|---:|---:|
| Without Hadamard | 153.58 ms | 9.85 ms |
| With Hadamard | 158.25 ms | 10.15 ms |

增加约 3.0% 的 TTFT 和 TTIT。此表还有 batch=8/32，但没有给出对应的 FP16 baseline，因此它主要说明该 FP8 实现下 online Hadamard 的额外成本，不能由此算出相对 FP16 的总体加速，更不能称为 H100 W4A4KV4 latency。

### 11.3 Rotation optimization 是部署前成本

第 6 页与第 19 页 Table 15 报告：LLaMA-2 7B 约 25 分钟，LLaMA-3 8B 约 30 分钟，Mistral-7B 约 16 分钟；正文还报告 LLaMA-2 70B 约 3.5 小时。

这是准备 quantized model 的一次性成本，不是每次生成都重新训练，也不是换一个硬件就能保证相同用时。称“只需 100 iterations”时，仍应意识到每次包含网络 forward/backward，不能把小参数量等同于零计算成本。

## 12. 哪些结论成立，哪些还没有建立？

**论文证据支持：** 合法的等价 rotation 能改变量化误差；random rotation 的质量有波动；在所测文本 LLM 和 bit 设置上，学习 rotation 通常进一步改善质量；可融合与在线变换之间存在 accuracy/latency trade-off。

**需要保留的限制：**

- 找到的是 calibration objective 下的优化结果，不是证明得到全局最优 rotation。
- FP 等价性依赖正确放置、normalization 预处理和配对变换；浮点运算本身也可能产生数值舍入差异。
- 使用语言 calibration 和语言 benchmark，没有直接验证 vision encoder、multimodal projector、连续 action head 或 WAM denoising chain。
- 数据集均值掩盖不同任务、不同层的差异；普通 token 的误差变小不等于 rare action 的风险变小。
- 4-bit 数值模拟、packed 存储、实际 low-bit kernel、完整 latency 必须分别提供证据。
- 可融合意味着不额外执行 dense rotation，不意味着所有部署系统都无需适配。

## 13. 对 FYP 的意义：一个可借用的思路

以下是阅读后的研究推断，不是论文已经证明的 VLA/WAM 结果。

这篇最值得借用的机制是：**在不改变 FP 函数的一族坐标系中，用最终相关的 loss 选择量化较友好的坐标。** 如果目标是 action fidelity，语言 cross-entropy 未必是足够好的 proxy；需要先确认哪些变换在目标结构中仍然等价，再讨论选择 rotation 的目标。

对 WAM，还应分清误差发生在哪个 expert、stream、denoising step，以及最终如何到达 action。一个普通 Transformer 中合法的 residual rotation，不能仅凭 shape 相同就跨 video/action experts 随意共享。

本论文可以作为 rotation-based quantization 的基础阅读；它提供数学机制和文本模型证据，机器人闭环效果仍需独立验证。

## 14. 回到 PDF 的阅读路线

### 20 分钟：抓住主线

1. **第 1 页 Abstract**：确认对象是 W/A/KV 量化，目标是学 rotation。
2. **第 2 页 Figure 1**：标出 \(R_1,R_2\) 可融合，\(R_3,R_4\) 在线。
3. **第 3–4 页 Figures 2–4**：先读 outlier，再读 random variation；图中的 axes/分布各在表达什么？
4. **第 5–6 页 Eqs. (2)–(4)**：找 objective、constraint、frozen 参数，不必第一遍推全部梯度。
5. **第 7 页 Table 1**：只读 LLaMA-2 7B 的 FP/W4A4KV4 几行。
6. **第 10 页 Table 6**：在笔记写下这里测 W4A8，避免与质量行混合。

### 90 分钟：能够解释方法和实验

| 时间 | 阅读位置 | 应完成的理解 |
|---|---|---|
| 0–15 min | README 2–4；PDF 第 3 页 | 手算 scale 与二维例子，解释 \(Q(xR)\ne Q(x)R\) |
| 15–35 min | PDF 第 2、4–5 页 Figures 1、5 | 逐个解释 R1–R4 的配对、RMSNorm gain、RoPE/FFN 位置 |
| 35–50 min | PDF 第 5–6 页 | 用 \(C^T=C^{-1}\) 说明 Cayley 保持正交 |
| 50–65 min | PDF 第 6、8 页 | 还原 W16 optimization→GPTQ W4 的 pipeline；读 Tables 2–3 |
| 65–80 min | PDF 第 7、9–10、14、19 页 | 核对 Avg./PPL、Mac W4A8、H100 FP8、准备成本 |
| 80–90 min | README 12–13 | 写出成立的结论与 FYP 迁移尚缺的证据；填写自己的 Meeting Card |

### 深入阅读：按疑问进入 Appendix

- **第 15–17 页 Tables 7–9**：均值之外，具体八个任务怎样变化？
- **第 17 页 Tables 10–11**：W3A8、sample/iteration 数量的影响。
- **第 14、18 页 A.4–A.5 / Tables 12–13**：asymmetric、clipping、calibration dataset。
- **第 18–20 页 Appendix B / Figure 7 / Table 19**：FP 等价却能学到 quantization gradient，以及 layer SNR 的差异。
- **第 20 页 Tables 16–18**：RTN/GPTQ、weight-only、instruction-finetuned 设置分别回答什么问题。
- **第 21–24 页 Figures 8–11**：看不同 depth 的 activation/weight 分布；注意作者指出，某些 token outlier 旋转后仍可能表现为 token 范围增大。

## 15. Reading Questions（留待自己回答）

1. 正交变换保留哪些量？为什么它不保留量化后的结果？
2. 用二维例子说明：outlier 更均匀，为什么仍不保证 rounding error 总会减少？
3. 在 residual addition 中，一条分支旋转而另一条不旋转会发生什么？
4. RMSNorm 的 scalar normalization 与 channel-wise gain 有什么区别？
5. 为什么 \(R_2\) 能通过 \(AV\) 与 output projection 抵消，而 \(R_4\) 不能随意穿过 SiLU？
6. 为什么 Q、K 的 \(R_3\) 必须配对？Figure 1 为什么画在 RoPE 之后？
7. 哪些 rotation 是 learned，哪些固定为 Hadamard？这个划分与部署成本怎样相关？
8. 主结果学习 rotation 时是什么 weight precision？最终是什么 precision？Table 3 控制了什么？
9. STE 的 backward 是什么近似？fake quantization 能支持哪些结论？
10. Cayley transform 为什么保持正交？普通 SGD 为什么不自动满足这个条件？
11. Table 2 和 Table 5 分别更适合回答哪个问题？
12. 2.9 pp、16.2 pp、约 3×、3.5 小时分别是哪种指标、什么模型和条件？
13. H100 Table 14 的 precision 是什么？为什么不能用于声称 W4A4KV4 已加速？
14. 为什么部分 layer SNR 变差仍可能使最终模型变好？
15. 将这个思路移到 WAM 前，首先需要证明哪种 function-preserving 性质？语言 calibration 还缺什么？

## 16. Meeting Card（留空，阅读后填写）

- 我理解的具体问题：
- 输入、输出、模型与精度条件：
- 最关键机制，用自己的话解释：
- 一条最有说服力的证据及物理页码：
- 这条证据真正比较了什么：
- 一条不能由论文推出的结论：
- 与我的 FYP idea 重合的部分：
- 我尚未理解的公式或实现步骤：
- 想向导师 / 同学讨论的问题：

## 17. 来源与阅读状态

- **主依据**：本目录固定 arXiv v4 PDF，24 页；方法第 4–6 页，主实验第 6–10 页，Appendix 第 14–24 页。
- **代码补充**：官方 repository 的 README 与 quant_utils.py；用于说明 STE/配置/导出支持，不代表本地运行，也没有把 live main branch 当成固定实验版本。
- **本地交付**：PDF、教学 README 与 metadata。没有安装模型环境，没有训练、导出、benchmark 或机器人实验。
- **阅读状态**：留待用户阅读、填写问题与 Meeting Card；本文没有代填阅读结论。
