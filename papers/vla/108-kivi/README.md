# 108. KIVI：为什么 K cache 和 V cache 要用不同的量化方向？

**KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache**  
阅读版本：**arXiv:2402.02750v2，2024-07-25，15 页**；ICML 2024。讲解整理：2026-10-08。

[固定版本 PDF](paper-arxiv-v2.pdf) · [固定版本 arXiv](https://arxiv.org/abs/2402.02750v2) · [官方代码](https://github.com/jy-yuan/KIVI) · [返回目录](../README.md)

作者：Zirui Liu、Jiayi Yuan、Hongye Jin、Shaochen (Henry) Zhong、Zhaozhuo Xu、Vladimir Braverman、Beidi Chen、Xia Hu。

用户提供的这篇论文正式名称为 **KIVI**。本 README 按“KV cache 是什么 → 为什么量化困难 → K/V 为什么区别对待 → 流式实现 → 实验如何解读”的顺序展开，可以独立阅读。所有页码按本地 PDF 的**物理页码，从第 1 页开始数**。实验数字来自论文，本地没有复现模型或运行 benchmark；教学例子与推导会单独标明。

## 1. 先抓住核心想法

LLM 逐个产生新 token 时，要反复读取历史 token 的 K、V。为了避免每次都重新计算，把它们保存成 KV cache。但 batch 越大、上下文越长，这份 cache 越占显存，也越需要搬运数据。

KIVI 研究的是：**把已经算出来、要保存和重用的 K/V activation 压缩，怎样在极低 bit-width 下保留生成质量？**

它的核心不是发明复杂的新 codebook，而是三个相互配合的设计：

1. **K per-channel**：同一个 channel 在若干 token 上共用量化参数，避免 outlier channel 污染普通 channels。
2. **V per-token**：同一个 token 内的一组 channels 共用量化参数，避免其他 token 的数值范围损伤被 attention 选中的 token。
3. **高精度 residual cache**：最新的一小段 K/V 暂时保持 full precision，既支持流式到达，也保护近期上下文。

标题的 **tuning-free** 表示不需要额外微调模型或学习量化变换；不是“没有配置参数”，也不是“不计算 scale”。KIVI 仍有 group size、residual length、bit-width 和实际 kernel 的选择。

## 2. KV cache 到底缓存了什么？

### 2.1 从一次 attention 开始

先忽略 batch、layer、head，令输入 token 特征矩阵为：

\[
X\in\mathbb R^{T\times d_{model}}.
\]

经过三个投影：

\[
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
\]

在一个 attention head 中，\(Q,K,V\) 的 feature dimension 为 \(d_h\)。标准 attention 可写为：

\[
S=\frac{QK^T}{\sqrt{d_h}},\qquad
A=\operatorname{Softmax}(S+M),\qquad
O=AV.
\]

| 符号 | 可以怎样理解 |
|---|---|
| \(Q\) | 当前 token 用来寻找相关信息的 query |
| \(K\) | 各个 token 用来被匹配的 key |
| \(V\) | 被关注后实际汇入输出的信息 |
| \(S\) | Softmax 之前的 logits / raw scores |
| \(M\) | causal mask 等限制，不允许读取未来 token |
| \(A\) | Softmax 后的 attention weights |
| \(O\) | 对 V 加权求和后的输出 |

K 影响“关注谁”；V 影响“从被关注的 token 中读到什么”。这不是完全独立的两条误差通路，但能帮助理解它们为什么需要不同量化方式。

论文第 2–3 页 Eq. (1) 为了简化省略 head、\(1/\sqrt{d_h}\)、mask 和 output projection。本文补上标准形式帮助理解，不意味着论文提出了另外一种 attention。

### 2.2 Prefill：先处理整段 prompt

Prefill 对 prompt 计算各层的 K、V，并保存它们。之后生成新 token 时，不必重新投影所有旧 token。

### 2.3 Decode：不断追加一个新 token

对新 token 的特征 \(x_t\)，计算：

\[
q_t=x_tW_Q,\quad k_t=x_tW_K,\quad v_t=x_tW_V.
\]

把 \(k_t,v_t\) 追加到历史 cache，再用 \(q_t\) 与所有历史 keys 匹配，读取相应 values。

**KV cache 里存的是 K/V activation，不是 \(W_K,W_V\) 这两个 weight matrices。** Weight 是模型参数，通常不随 prompt 改变；cache 是当前请求的中间结果，会随输入和生成长度改变。

因此 KIVI 的 KV2 不等于 W2，也不等于整个模型 A2。论文的方法可以与 weight-only quantization 组合，但它自己的核心压缩对象是 cache。

### 2.4 为什么普通的一次 forward 不够评价 KV quantization？

压缩 cache 的影响体现在后续 decode 的读取和误差传播中。如果评测只看一次 prefill 后的 logits，或者只做一个没有 cache 重用的 forward，就可能没有充分使用被压缩的历史 K/V。

论文第 6 页脚注因此强调，多步 generation tasks 更适合研究这件事，单步读 logits 的 closed-end tasks 不足以体现完整影响。这个理由不等于说所有选择题都不能评估语言能力，而是它们未必覆盖 KV cache 的实际使用方式。

## 3. Cache 为什么可能成为显存和带宽瓶颈？

对统一结构的 decoder，忽略 padding、allocator 和额外 metadata，KV payload 的教学估计为：

\[
M_{KV}=2\,L\,B\,T\,H_{KV}\,d_h\,p.
\]

其中：

- \(2\)：K 与 V 各一份。
- \(L\)：layers 数量。
- \(B\)：batch size，即同时处理的 requests 数量。
- \(T\)：当前 cache 的 token 数，通常为 prompt length 加已生成长度。
- \(H_{KV}\)：KV heads 数量。
- \(d_h\)：每个 head 的 feature dimension。
- \(p\)：每个元素的 bytes，FP16 为 2。

注意是 **KV heads**，不是不加区分地使用 query heads：

| Attention 类型 | Head 关系 | 对 cache 的影响 |
|---|---|---|
| MHA | 每个 query head 有对应 KV head | KV head 数通常较多 |
| GQA | 一组 query heads 共享 KV head | 比同 query-head 数的 MHA 省 cache |
| MQA | 所有 query heads 共享一个 KV head | cache 本身已较紧凑 |

**教学例子**：\(L=32,B=1,T=4096,H_{KV}=32,d_h=128\)，FP16 的 KV payload 为 2 GiB。若只看理想的 2-bit payload，变为 0.25 GiB。

但是模型 weights 没有因此一起减少；量化还需要 scales、offsets、高精度 tail 等，所以实际总显存不会直接下降 8×。

Decode 还需要反复从 GPU memory 读取旧 cache。即使矩阵运算量没有同比增加，搬运 K/V 的带宽成本也会增加。压缩 cache 的系统收益既可能来自减少读写，也可能来自容纳更大 batch；不能只看“元素数量相同”就判断没有速度价值。

## 4. 先把 per-token 和 per-channel 的方向读对

这是读 KIVI 最容易混淆的地方。固定一个 batch、layer、KV head，把 cache 写成：

\[
K,V\in\mathbb R^{T\times d_h}.
\]

**行是 token，列是 channel。**

| 名称 | 固定什么 | 一组数从哪里取 | 哪些元素共享 scale/offset |
|---|---|---|---|
| Per-token | 一个 token，即一行 | 在这一行的 feature channels 中取一组 | 同 token 的若干 channels |
| Per-channel | 一个 channel，即一列 | 在这一列的历史 tokens 中取一组 | 同 channel 的若干 tokens |

这里“per-channel”表示每个 channel 分别有量化参数，不是把多个 channels 混到一起；“per-token”表示每个 token 分别有量化参数，不是把多个 tokens 混到一起。

论文还采用 group-wise：不是给完整无限长的一列只用一个 scale，也不是每行必须只有一个 scale。

以 \(G=32\) 为例：

- **K**：固定 channel，在 token 轴上每 32 个值形成一组。
- **V**：固定 token，在 channel 轴上每 32 个值形成一组。

回看第 2 页 Figure 1 时，关注矩阵的 token/channel 轴及共享参数的范围。不同实现可能转置 tensor 或改变 memory layout，不能仅凭某个程序的 `dim=-1` 就判断它在原始语义上是哪一种量化。

## 5. 2-bit affine quantization：只有四个表示点

论文第 3 页的量化与还原为：

\[
z=\min(X),\qquad
s=\frac{\max(X)-\min(X)}{2^b-1},
\]

\[
q=\operatorname{round}\left(\frac{x-z}{s}\right),
\qquad
\widehat x=sq+z.
\]

对 \(b=2\)，\(q\in\{0,1,2,3\}\)，每组只能用四个均匀间隔的值表示。

\(z\) 在论文中称为 zero-point，但它在这个公式中是**实数 min offset**。另一些量化资料把 zero-point 定义为整数码，此时写作 \(\widehat x=s(q-z_{int})\)。理解时按公式确认含义，不要把两种定义直接替换。

若整组是常数，\(s=0\) 需要实现上的特殊处理。下面例子中常数行可直接精确保留为该常数；这是教学约定，不是说任何除以零的公式都可直接执行。

### 5.1 一个手算例子

**教学例子**：一组数 \([0,1,2,3]\)，2-bit 可以精确表示，\(z=0,s=1\)，整数码为 \([0,1,2,3]\)。

如果改为 \([0,1,2,30]\)，则 \(s=10\)，整数码为 \([0,0,0,3]\)，还原为 \([0,0,0,30]\)。三个普通值的区别被最后一个大值吞掉。

因此，**同样都是 2-bit，谁与谁共享 scale 很重要**。KIVI 的核心选择正是在决定这件事。

### 5.2 标题中的 asymmetric 怎样理解？

至少需要分清两层：

1. **K/V 策略不对称**：K per-channel，V per-token；这是一篇论文的方法主线。
2. **标量量化的 min-max 网格不以 0 为中心**：即 affine/asymmetric quantization 形式。

它们不是同一个概念。只使用 min-max offset，却仍把 K/V 全部 per-token，不等于实现了 KIVI 的关键策略。

这也不是 Vector Quantization：每个 scalar 仍由少量整数码与该组 scale/offset 还原，没有学一个多维 codebook 并用 codeword index 替换向量。

## 6. 为什么 K 应该 per-channel？

### 6.1 论文先观察到固定的 outlier channels

第 4 页 Figure 2 画出 LLaMA-2-13B 与 Falcon-7B 不同 layers 的 K/V absolute values。K 的大值往往集中在少数固定 channels 上，跨多个 tokens 持续出现。

读图时看“哪条 feature 方向一直高”，而不是只看最高点。Per-token 会把该行的大 channel 和普通 channels 放到同一组，scale 被大值决定；per-channel 则让大 channel 只影响自己的量化范围。

### 6.2 一个四行四列的教学矩阵

**教学例子，使用 \(G=4\)，不是论文的默认 \(G=32\)：**

\[
K=\begin{bmatrix}
0.0&0.1&0.2&10\\
0.1&0.2&0.3&11\\
0.2&0.3&0.4&12\\
0.3&0.4&0.5&13
\end{bmatrix}.
\]

第 4 列持续大，前三列较小。

对第 2 行 per-token 2-bit，\(z=0.1\)、\(s=(11-0.1)/3\approx3.633\)。0.1、0.2、0.3 都被还原为 0.1。

如果 per-channel：

- 第 1 列是 \([0,0.1,0.2,0.3]\)，四个表示点刚好覆盖。
- 第 2 列是 \([0.1,0.2,0.3,0.4]\)，同样能精确表示。
- 第 3 列也如此。
- 第 4 列是 \([10,11,12,13]\)，用自己的大范围，不污染前三列。

在这个特意构造的例子中，per-channel 全部精确还原。真实 cache 不会这么整齐，所以结论不是“per-channel 必然零误差”，而是**把范围异常的 channel 隔开，可以保护其余 channels 的分辨率**。

### 6.3 K 的误差会进入 Softmax

若 \(\widehat K=K+E_K\)，当前 query 为 \(q\)，logits 误差是：

\[
\delta S=\frac{qE_K^T}{\sqrt{d_h}}.
\]

之后 \(\widehat A=\operatorname{Softmax}(S+\delta S)\)。一个 key 的误差可以改变当前 query 对很多 tokens 的相对分配；它不仅影响自己保存的值，还可能改变“哪个 token 重要”。

作为教学局部分析，Softmax 的 Jacobian 是：

\[
J_A=\operatorname{diag}(A)-AA^T,
\qquad \delta A\approx J_A\delta S.
\]

这说明同样的 \(E_K\) 在不同 query、attention 分布下产生的影响不同；不能用 K reconstruction MSE 直接等同于最终生成错误率。

### 6.4 Table 2 的证据怎样读？

第 4 页，统计对 layers 与 heads 平均，LLaMA-2-13B：

| 指标，按原表列值 | K per-token | K per-channel |
|---|---:|---:|
| K relative reconstruction statistic | 13.67 | 4.55 |
| Attention weights relative error statistic | 47.00 | 9.60 |

后者约相差 4.90×。这里关注两段证据：K 重建本身改善，经过 attention 后也改善。

原文用分式放在 Frobenius norm 内表示 relative error，例如 \(\|(K-\widehat K)/K\|_F\)。本文按原表保留统计数值，不把它改成 absolute MSE、不补成任务 accuracy，也不自行为这些数值添加百分号或当成误差上界。

另外，论文的 \(A\) 在这张表中是 **Softmax 后的 attention weights**，不是前面的 raw logits。读论文时注意它在不同式子中对 scores/logits 的简化用法。

## 7. 为什么 V 应该 per-token？这是更重要的半篇论文

### 7.1 单看 V 的分布并不能给出答案

Figure 2 中 V 没有与 K 相同的固定 outlier-channel 结构。直觉上可能觉得：既然没有明显的异常方向，两种 grouping 应该差不多。

但 V 的用途不是“重建整个 V tensor 然后结束”，而是进入：

\[
O=AV.
\]

对一个 query：

\[
o=\sum_j a_jv_j.
\]

\(v_j\) 是第 \(j\) 个 token 的 value vector，\(a_j\) 是关注它的权重。若 attention 主要集中在少数 tokens，那么这些 tokens 的误差比其他位置更直接影响输出。

Per-token 可以让每个 token 的量化范围独立。别的 token 数值再大，也不会把重要 token 的 scale 一起拉大。Per-channel 则让同一列的多个 tokens 共享范围，重要 token 可能被不重要 token 的大值伤害。

### 7.2 一个具体的 V 矩阵例子

**教学例子**：

\[
V=\begin{bmatrix}1&2&3\\0&0&0\\100&100&100\end{bmatrix},
\qquad A=(1,0,0).
\]

当前 query 只读取第 1 个 token，正确输出为 \((1,2,3)\)。

Per-token 2-bit：第 1 行自己的范围是 \([1,3]\)，四个表示点为 \(1,1.667,2.333,3\)。中间元素 2 的最大 rounding error 为 \(1/3\)，其余两个精确。

Per-channel 2-bit：每列都包含 0 与 100，四个表示点为 \(0,33.333,66.667,100\)。第 1 行的 1、2、3 全部映射为 0。虽然第 3 个 token 当前根本没有被读取，它仍通过共享 scale 破坏了第 1 个 token。

这就是“quantization of other tokens 不要伤害 important tokens”的具体含义。Per-token 没有预测哪个 token 重要，而是**提前把 token 之间的量化范围干扰隔开**，让不同 queries 的选择更安全。

这个例子为了清楚而用了较大值和极端 one-hot attention，不代表 Figure 2 的 V 实际有这样的固定 outlier。

### 7.3 为什么更小的 V 重建误差反而可以更差？

固定 \(A\)，令 \(\widehat V=V+E_V\)，则：

\[
\widehat O-O=AE_V=\sum_j a_j e_j.
\]

普通 tensor reconstruction 指标会把各 token 的误差都算进去，输出误差却按 \(a_j\) 加权。两者关注的位置不同。

**另一个教学例子，下面只是两种可能的误差分布，不是模拟某个真实 quantizer：**

\[
A=(0.99,0.01),
\quad E^{(a)}=\begin{bmatrix}0.01&0.01\\1&1\end{bmatrix},
\quad E^{(b)}=\begin{bmatrix}0.2&0.2\\0&0\end{bmatrix}.
\]

第一种的整体误差 norm 约 1.414，但输出误差为 \((0.0199,0.0199)\)，norm 约 0.0281。

第二种的整体误差 norm 仅 0.2828，更小；但输出误差为 \((0.198,0.198)\)，norm 约 0.2800，更大。

所以“整个 V 重建得更好”不自动等于“模型读出的信息更好”。关键还在于误差落在哪些 tokens。

### 7.4 Table 2 的反直觉结果

第 4 页：

| 指标，按原表列值 | V per-token | V per-channel |
|---|---:|---:|
| V relative reconstruction statistic | 4.57 | **3.73** |
| Attention output relative error \(\Delta\) | **3.55** | 49.89 |

Per-channel 的 V reconstruction statistic 更小，但 attention output error 大约是 per-token 的 **14.05×**。正文概括为近 15×。

论文还报告 attention sparsity 84.3%，作为权重集中解释的证据；不要把这写成“84.3% 的 KV tokens 被删除”。KIVI 保留全部历史 tokens，只是改变它们的表示精度。这个统计也不是保证所有 heads、所有 queries 都具有相同稀疏程度。

### 7.5 同时量化 K/V 时，还会有交互项

前面的 V 分析固定 \(A\)，便于隔离 value 的影响。若 K 量化也改变 attention，令 \(\widehat A=A+E_A\)，则有教学分解：

\[
\widehat A\widehat V-AV
=E_AV+AE_V+E_AE_V.
\]

三项分别是 attention 改变、value 改变，以及它们的交互。这说明 K/V 分开诊断很有帮助，但两者同时量化的最终效果仍需完整生成任务验证。

## 8. 知道了量化方向，为什么还不能直接实现？

### 8.1 V 比较容易流式追加

V per-token 只需要当前 token 的 channels。新 \(v_t\) 到达后，其内部每组 min/max 已经可计算，不必等待未来 token。

因此它可以独立量化、packing 后追加到已有 V cache。

### 8.2 K 的 scale 需要跨 token 决定

K per-channel 的一组包含同一 channel 上的 \(G\) 个 tokens。刚到第一个 token 时，不知道后面 \(G-1\) 个值的范围。

如果给整段不断增长的历史共用 scale，每次出现新的 max/min 都可能要改 scale、重新量化旧历史；这会增加成本，也产生额外重编码误差。

KIVI 因此采用**有界的时间分组**：已经完整的历史组保持低精度不再改写，尚未处理的最新部分留在高精度 residual cache。

## 9. Group size G 与 residual length R 是两个不同参数

| 参数 | 决定什么 | 对 K | 对 V |
|---|---|---|---|
| \(G\) | 每个 scale/offset 覆盖多少 scalar values | 每个固定 channel 跨多少 tokens | 每个固定 token 跨多少 channels |
| \(R\) | 高精度 tail 的容量/处理阈值，单位为 tokens | 积满后批量 flush | 保留最近 R 个，逐渐移出最旧 token |

论文默认 \(G=32,R=128\)，并要求 K 的 \(R\) 可以被 \(G\) 整除。一次 128-token 的 K flush，内部仍然是每 32 tokens 一组，共四个 groups；不是 128 tokens 共用一个 scale。

这里的 **residual cache** 是“尚未量化的高精度近期部分”，不是 residual connection，也不是“量化误差 \(x-\widehat x\) 的补偿向量”。不要把它读成 residual quantization / additive VQ。

## 10. KIVI 的四份 cache 怎样更新？

论文第 5 页 Figure 3 与第 12 页 Algorithm 1 要一起读。Figure 3 为简洁省略了 V 和输出计算，不能从图上推断 K/V 更新规则相同。

### 10.1 用四份数据表示完整历史

\[
K=[K_g;K_r],\qquad V=[V_g;V_r].
\]

- 下标 \(g\)：已量化的历史部分，实际保存 packed codes 与 scales/offsets。
- 下标 \(r\)：仍为 full precision 的近期部分。

K 与 V 的分界位置可以不同；它们仍对应同一组完整的 token positions。

### 10.2 Prefill 怎样处理？

先正常计算 prompt 的 K/V。论文强调，prefill attention 使用当时的原始高精度 K/V；随后保留压缩后的历史 cache 与高精度 tail，用于 decode。

Algorithm 1 的 K 处理函数取：

\[
r_K=T\bmod R.
\]

前 \(T-r_K\) 个 tokens 量化，剩下 \(r_K\) 个保持高精度。因为 \(R\) 能被 \(G\) 整除，前面的部分可以按 G 分组。

V 的近期部分则保留最后 R 个 tokens；如果整个序列短于 R，概念上就是全部留在高精度部分。论文伪代码为了简洁使用固定切片，真正实现还需要处理短序列等边界；这里讲解算法，不把伪代码当成可直接运行程序。

### 10.3 Decode 时 K 是批量 flush

每个新 \(k_t\) 先追加到 \(K_r\)：

\[
K_r\leftarrow\operatorname{Concat}(K_r,k_t).
\]

当 \(K_r\) 长度到达 R，按 per-channel、group size G 量化这一整批，追加到 \(K_g\)，再清空 \(K_r\)。

因此 K 的高精度 tail 长度呈现“从小增长到 R，再回到 0”的变化。它**不是始终保存最近 R 个 K tokens**。论文第 6 页说其平均高精度窗口约为 \(R/2\)，这是对批量 flush 周期的概括。

### 10.4 Decode 时 V 是持续的最近窗口

每个新 \(v_t\) 追加到 \(V_r\)。如果长度超过 R，把最旧的那部分移出，per-token 量化后追加到 \(V_g\)，\(V_r\) 继续保存最近 R 个。

V 不需要等待多个 tokens 凑齐才知道 scale；G 是该 token 内的 channel group。因此它可以逐 token 移出。

### 10.5 用 G=2、R=4 的教学时序走一遍

**教学例子：** 假设已有 6 个 tokens，prefill 后：

| 时刻 | K 低精度 / 高精度 token 数 | V 低精度 / 高精度 token 数 |
|---|---|---|
| T=6 | 4 / 2 | 2 / 4 |
| 追加到 T=7 | 4 / 3 | 3 / 4 |
| 追加到 T=8 | 8 / 0，K tail 到 4 后 flush | 4 / 4，只移出最旧 token |
| 追加到 T=9 | 8 / 1 | 5 / 4 |

两份 cache 都覆盖 tokens 1–T，但低/高精度分界不同。这个区别解释了为什么不能把 K 的 grouped/residual 边界直接拿来切 V 的 attention weights。

## 11. 低精度 prefix 与高精度 tail 怎样共同算 attention？

先计算 K 两部分对应的 raw logits：

\[
S_g=\frac{q\widehat K_g^T}{\sqrt{d_h}},\qquad
S_r=\frac{qK_r^T}{\sqrt{d_h}}.
\]

按 token 顺序拼接，并在完整序列上做一次 Softmax：

\[
A=\operatorname{Softmax}([S_g,S_r]+M).
\]

然后按照 **V 的 token 分界** 把 \(A\) 切为 \(A_g^{(V)},A_r^{(V)}\)：

\[
O=A_g^{(V)}\widehat V_g+A_r^{(V)}V_r.
\]

两个重点：

- 不能各自对 prefix、tail 做 Softmax 再相加；那会丢失全序列的统一归一化。
- K/V 的高精度长度不同并不矛盾，只要拼接保持同一 token 顺序、计算 V 输出时使用 V 的分界即可。

原文 Eq. (3) 与 Algorithm 1 用 \(Q(K_g)\) 等符号简写 mixed-precision multiplication，不能理解为没有 scale 就直接把 0/1/2/3 的整数码当成原始 K 来点乘。

## 12. 真正省显存、提高吞吐还需要什么系统实现？

### 12.1 Fake quantization 与 packed cache

Fake quantization 先映射到低位数网格，再还原成浮点 tensor 计算，用来诊断数值影响。若仍存 FP16，它没有真正实现 2-bit cache 的存储压缩。

真实实现必须 packing：例如把多个 2-bit codes 放进机器字，并保存每组 scale/offset。计算时再依据 codes 和量化参数恢复需要的数据。

KIVI 第 6 页说明，group-wise quantization kernel 使用 Triton；dequantization 与 matrix multiplication 以 tile 为单位融合，使用 CUDA。

### 12.2 为什么融合 dequantization 与 matmul？

如果先把完整历史 cache 全部还原成大 FP16 tensor，再做 attention，会重新产生大中间 tensor 和额外 memory traffic，削弱压缩收益。

融合实现读取一个 tile 的 packed data，在需要计算的局部还原，直接用于乘法与累加，减少完整 FP cache 的重新物化。

这不等于“所有乘法和 accumulator 都是 native INT2”。论文给出的重点是 mixed precision 与融合 dequantization；只凭 KV2 缩写不足以确定所有 operand、accumulator、输出 dtype。

### 12.3 2-bit 的实际成本不只有两位

一个包含 G 个 scalar values 的 group，除了 G 个 codes，还需要 scale、offset。

**教学估计**：若两项 metadata 各 16-bit，则低精度历史每元素平均成本为：

\[
b_{eff}=2+\frac{16+16}{G}.
\]

G=32 时为 3 bits，不是严格的 2 bits；仅这部分相对 FP16 为约 5.33×。这是说明 overhead 的假设算例，实际 metadata dtype、packing 对齐和实现以部署配置为准，本文没有声称这就是论文 kernel 的完整内存账单。

再考虑 high-precision tail，设单份 K/V 当前有 \(r\) 个 full-precision tokens，忽略 metadata 的 payload ratio 为：

\[
\rho=\frac{b(T-r)+16r}{16T}.
\]

**教学例子**：T=4096、r=128、b=2，\(\rho\approx0.1523\)，相当于约 6.56× 的 payload compression，而不是理想的 8×。实际 K 的 r 随 flush 改变，V 通常保持 R，完整估计应分别计算二者。

整个推理的 peak memory 则还包括：

\[
M_{total}=M_{weights}+M_{KV,codes}+M_{scales/offsets}
+M_{FP\ tail}+M_{workspace}+M_{other}.
\]

所以论文的 total peak memory reduction、理想 cache payload ratio，以及 group-wise effective bits 是三个不同数字。

## 13. 先读实验设置，再读“几乎无损”

主实验默认 \(G=32,R=128\)，比较 KIVI-2 与 KIVI-4。改变参数的消融、R32 的附录实验另行标注。

### 13.1 普通生成任务的指标不是同一种分数

第 6 页说明：

| Task | 论文采用的指标 | 主要能力 |
|---|---|---|
| CoQA | Exact match accuracy | 对话问答 |
| TruthfulQA | BLEU score | 生成回答与参考的相似性；不要误写成 TruthfulQA MC accuracy |
| GSM8K | Exact match accuracy | 多步数学推理后的答案 |

这些分数均以原表形式保留。CoQA/GSM8K 的准确率差可以说 pp；TruthfulQA 在这里应说 BLEU 分数差，不能同样解释为成功率下降。

### 13.2 LongBench 主表与补充表的范围不同

第 9 页 Table 4 选了八项：Qasper、QMSum、MultiNews、TREC、TriviaQA、SAMSum、LCC、RepoBench-P，分别使用 F1、ROUGE、classification 或 similarity 等指标。

主实验 maximum sequence length：Mistral 为 8192，其他模型为 4096。这个 eight-task average 是论文选定指标的均值，不能称为完整 LongBench 所有任务的统一 accuracy。

第 14–15 页 Tables 8–10 是另外一组 instruction/long-context models 与更多任务。它们的均值分母和模型版本不同，不应与 Table 4 直接拼成同一个排行榜。

## 14. Table 1：量化方向有多重要？

第 3 页，在 **LLaMA-2-13B、group size 32、全部 cache tokens 都量化**的 fake quantization 诊断中：

| 配置 | CoQA | TruthfulQA BLEU |
|---|---:|---:|
| 16-bit | 66.37 | 29.53 |
| 4-bit，K-T / V-T | 66.48 | 29.51 |
| 2-bit，K-T / V-T | 52.93 | 24.98 |
| 2-bit，K-C / V-C | 2.88 | 0.74 |
| 2-bit，K-T / V-C | 2.80 | 0.26 |
| 2-bit，K-C / V-T | 63.53 | 28.60 |

T=per-token，C=per-channel。

怎么读这张表：

1. K/V 都 per-token 的 4-bit 可以保持分数，但相同方向直接降到 2-bit 明显变差。
2. 对这份模型与测试，V per-channel 的两种 2-bit 配置都非常差。
3. 相同 2-bit 下 K-C/V-T 最好，说明不能只按 bit-width 判断质量。

这里尚未加入 KIVI 的高精度 residual window，不能把 63.53 直接当成最终 KIVI-2 的 CoQA 分数。原文为 per-channel 诊断使用 padding，使全部 tokens 可被量化；它是分析 grouping 的 controlled diagnostic，不是最终流式实现。

## 15. Table 3：方向正确以后，为什么 residual window 仍然关键？

第 8 页，先看 LLaMA-2-13B：

| 配置 | CoQA | TruthfulQA BLEU | GSM8K |
|---|---:|---:|---:|
| 16-bit | 66.37 | 29.53 | 22.67 |
| Fake 2-bit，K-C/V-T，全部 tokens 量化 | 63.53 | 28.60 | 12.21 |
| KIVI-2，G32/R128 | 66.23 | 29.84 | 20.77 |
| KIVI-4，G32/R128 | 66.38 | 29.49 | 23.65 |

GSM8K 从全量 fake quantization 的 12.21 提高到 KIVI-2 的 20.77，差 8.56 pp；KIVI-2 仍比 FP 低 1.90 pp。

作者把保留 local relevant tokens 的高精度窗口解释为关键因素。这个比较说明**仅选对 K/V 方向还不够，近期 full-precision 部分是方法的一部分**。

但它也不应被表述成“纯粹是 fake 与真实 kernel 的差别”：两种配置对 tokens 的 precision allocation 不同，KIVI 额外保留了高精度 tail。性能变化不只是是否 packing。

再看两个模型的 GSM8K：

| 模型 | FP16 | KIVI-2 G32/R128 | 差值 |
|---|---:|---:|---:|
| LLaMA-2-7B | 13.50 | 12.74 | -0.76 pp |
| Mistral-7B | 38.36 | 36.01 | -2.35 pp |

正文“约 2% accuracy drop”是一种概括，按具体表行应保留 0.76、1.90、2.35 pp 等绝对差异，不能写成所有任务严格小于 2%，也不能把 relative percent 与 pp 混用。

### Falcon 是需要保留的例外

Falcon-7B 的 CoQA：16-bit 59.83，KIVI-2 57.48，KIVI-4 59.67；GSM8K 为 4.55、3.41、4.47。

作者讨论 Falcon 的 MQA 只有一个 KV head，本身已压缩，采用 4-bit 更能保留质量。它是“所有模型都适合 KV2”这个说法的反例；不能仅从该比较进一步证明 KV head 数是唯一原因。

## 16. Table 4 与附录：均值接近，不等于每项都接近

第 9 页的 eight-task LongBench average：

| 模型 | 16-bit | KIVI-2 G32/R128 | 平均分差 |
|---|---:|---:|---:|
| LLaMA2-7B | 44.52 | 44.27 | -0.25 |
| LLaMA2-13B | 44.85 | 44.69 | -0.16 |
| LLaMA2-7B-Chat | 45.95 | 45.67 | -0.28 |
| LLaMA2-13B-Chat | 45.96 | 45.52 | -0.44 |
| Mistral-7B | 46.58 | 45.85 | -0.73 |
| Falcon-7B | 8.71 | 7.95 | -0.76 |

这些是不同指标混合后的 average score 差，不统一称为 task-success pp。它们支持所测长上下文任务均值基本保持，但仍要读 individual tasks。

例如 LLaMA2-7B 的 MultiNews 为 3.51→1.14，均值很接近仍可能隐藏某项退化。Falcon 基线本身很低，均值绝对差不大也不等于能力已经充分保留。

### LLaMA-3-8B-Instruct：2-bit 的局部下降不能被平均数遮住

第 14 页 Table 8 明确使用 **LLaMA-3-8B-Instruct**，8K context，GQA 的 8 个 KV heads 对应 32 个 query heads，G32/R128。

| 指标 | Baseline | KIVI-2 | KIVI-4 |
|---|---:|---:|---:|
| Table 8 的 15-task average | 45.21 | 44.37 | 45.31 |
| LCC | 57.00 | 50.84 | 57.36 |
| RepoBench-P | 51.22 | 46.65 | 52.03 |

2-bit average 只少 0.84，但两项代码任务分别少 6.16 与 4.57 分。4-bit 更接近 baseline。正确表述是“这个模型的多数指标保持较好，但某些任务在 2-bit 下退化，4-bit 是更稳妥的质量选择”；不是“LLaMA-3 一定不能 2-bit”，也不是“2-bit 完全无损”。

Table 9 的 Mistral-7B-Instruct-v0.2 是 32K context、8 KV heads/32 query heads；Table 10 是 LongChat-7B-v1.5-32K。不要把它们与主表的 Mistral-7B、LLaMA-2 base model 混成同一实验。

## 17. Figure 4：Needle-in-a-Haystack 验证什么？

第 7 页 Figure 4 与第 13 页 Appendix B：在大量背景文本中插入一个 **7-digit passkey**，最后要求找回它。背景为 Paul Graham essays。

读图要分开三件事：

- 横轴是 **word count**，不是 token count。作者为不同 tokenizers 的比较采用 word count，图上另注实际 token 长度。
- 纵轴 depth 是 passkey 插入文档的相对位置，不是 Transformer depth。
- 每个小图对应一个 model/precision；应将同一模型的 baseline、KIVI-2、KIVI-4 对齐看。

本地 v2 Figure 4 标注 20K words 对 LLaMA-3 约 27K tokens、对 Mistral 约 30K tokens。这组 NIAH 设置与 Table 8 的 8K LongBench 设置不是同一协议；不要据此自行补出上下文扩展的实现 recipe。

图中 LLaMA-3 的 retrieval 基本保持，Mistral baseline 与量化版本有相似的失败位置。这支持这份 passkey retrieval 测试下压缩 cache 仍能保留检索能力。

但找回一个 passkey不等于长文推理、代码生成或所有多步任务都无损；也不能把一个绿色格子当成几百次独立试验的统计等效性证明。图没有提供可用于此类推断的重复实验 denominator。

## 18. Table 5：G 与 R 的影响要分别读

第 9 页，LLaMA2-13B，KIVI 2-bit 的 GSM8K：

**固定 R=128，改变 G：**

| Group size G | Score |
|---|---:|
| 32 | 20.77 |
| 64 | 21.00 |
| 128 | 17.29 |

G 变大，每个 scalar 分摊的 scale/offset 成本变小，但共享范围包含更多数值，更难适应局部差异。这里 G32/G64 接近，G128 退化；不能进一步推出每个模型的最佳 G 都是 64。

**固定 G=32，改变 R：**

| Residual length R | Score |
|---|---:|
| 32 | 20.62 |
| 64 | 19.86 |
| 96 | 20.55 |
| 128 | 20.77 |

更长的高精度 tail 要花更多显存，但这里 accuracy 不随 R 单调增加。R128 并没有对所有 R 都大幅胜出。

另外，此表没有 R=0 row，不能仅从它单独量化“有窗口比没有窗口”的纯单变量收益。窗口的重要性主要应结合前面的 all-token fake quantization 与 KIVI 比较理解。

第 13–14 页 Tables 6–7 补充 R32：多数任务仍接近 R128，但并非完全相同。例如 Mistral-7B GSM8K，R128 为 36.01，R32 为 34.34；Falcon 的对应 KIVI-2 分数为 3.41 与 2.20。选择更短 R 的系统收益要与具体任务质量一起判断。

## 19. Figure 5：显存与吞吐到底怎么测？

第 7–8 页 Section 4.2.4，设置为：

- **模型**：LLaMA-2-7B。
- **硬件**：一张 NVIDIA A100，80GB。
- **Workload**：基于 ShareGPT 的输入/输出长度合成服务请求；平均 prompt length 161、generation length 338。
- **方法**：FP16 baseline，对比 KIVI-2 的 R128 与 R32，逐步增加 batch 到显存不能容纳。
- **指标**：peak memory 与 aggregate throughput，后者单位为 tokens/s。

论文报告 peak memory 包含 model weights 的约 **2.6× reduction**，支持至多 **4× larger batch size**，以及 **2.35×–3.47× throughput**。

读第 8 页 Figure 5：上半图横轴 batch、纵轴 memory，比较同 batch 下的内存，及同预算能装多少 requests；下半图横轴 batch、纵轴 tokens/s，比较服务吞吐随 batch 的变化。

**高吞吐部分来自允许更大 batch。** 它不是 batch=1 的每 token latency 降低 3.47×，也不是每个请求 completion time 都同比变短。图中不同方法的最大可运行 batch 不同，报告最大服务吞吐时不能假装三者使用了同一个 batch。

三个数字的含义应分别写清楚：

| 数字 | 对象 | 不能替换成什么 |
|---|---|---|
| 理想 8× | FP16→2-bit 的纯 cache codes payload | 整个模型显存 8× |
| 论文约 2.6× | 特定 workload 下、包含 weights 的 peak memory | 所有 context/batch 的固定比例 |
| 2.35×–3.47× | 该 A100 服务设置的 aggregate throughput | 单请求 latency 或机器人控制周期加速 |

作者还把进一步融合 quantization 与前序 operations 留作 future work。因此本论文证明已有实现有实际系统收益，但没有宣称所有量化开销已经消除。

## 20. 成立的结论与实际限制

**论文证据支持：** K/V 的统计结构与下游用途不同；K per-channel、V per-token 比所测的其他 2-bit grouping 更好；保留近期高精度 cache 有助于生成质量；配合 packed cache 和融合 kernel 可以降低显存并提高服务吞吐。

**需要保留的边界：**

- 2-bit 是历史 cache 的主要表示，不是每个 cache token 全部 INT2；近期 tail 保持 full precision。
- Tuning-free 不等于 parameter-free；G/R/bit-width 仍有 memory、quality、runtime trade-off。
- 不同模型/任务存在退化，Falcon 与代码任务是应关注的例子；均值接近不是无损证明。
- Table 1/2 的 grouping/error 诊断、Table 3/4 的生成质量、Figure 5 的系统表现回答不同问题。
- NIAH 是受控检索，不代表所有长上下文 reasoning。
- GQA/MQA 已经减少 cache，KV quantization 的总显存收益需要结合实际 KV-head 数与 context 判断。
- 没有直接的 VLA/WAM、continuous action 或机器人 closed-loop evidence。

## 21. 对 FYP 的意义：先确认有没有同一种 cache 使用方式

以下是研究推断，不是论文已经验证的机器人结论。

KIVI 提供的第一条启发是：**量化对象的用途决定如何看误差。** V 的普通 reconstruction 指标改善，仍可能伤害被 attention 读取的信息；如果最终关心 action，就应该追问误差怎样传到 action，而不是只比较 tensor MSE。

第二条启发是：要先确认目标模型是否有可重用、会增长、会在后续 decode 反复读取的 KV cache。Autoregressive text/action decoder 与一次性重算 attention 的 diffusion/DiT 路径，不能因为都叫 K/V 就默认拥有同样的缓存生命周期。

若 context 较短、KV heads 很少或主要瓶颈是 weights/FFN，cache quantization 的系统收益可能有限；若存在长期跨调用 cache，又要重新判断近期高精度窗口是否保护真正重要的历史信息。

阅读这篇的重点是学会把“量化方向、误差传播、流式数据结构和系统指标”连起来，不是直接认定 WAM 可以采用 KV2 而保持闭环成功率。

## 22. 回到 PDF 的阅读路线

### 20 分钟：建立清晰的主线

1. **第 1 页 Abstract**：确认压缩对象是 KV cache，数字是显存/吞吐而不是 model weights。
2. **第 2 页 Figure 1**：把矩阵画成 token 行、channel 列，分别指出 K/V 一组数的位置。
3. **第 3 页 Table 1**：固定 bit-width 比较四种 K/V grouping。
4. **第 4 页 Figure 2 / Table 2**：看 K outlier channels，以及 V reconstruction 与 output error 的反转。
5. **第 5 页 Figure 3 / Section 3.3**：认识 grouped/residual，记住图省略 V。
6. **第 8 页 Tables 3 / Figure 5**：看 residual 对生成质量的重要性，再看不同 batch 的系统收益。

### 90 分钟：能够自己走完 cache 更新

| 时间 | 阅读位置 | 应完成的理解 |
|---|---|---|
| 0–15 min | README 2–5；PDF 第 2–3 页 | 写出 cache memory formula，画 K/V 分组方向，手算四个表示点 |
| 15–35 min | PDF 第 3–5 页 | 解释 K outlier 隔离与 V attention-weighted error，核对 Table 2 |
| 35–55 min | PDF 第 5–6、12 页 | 用 G2/R4 教学时序走过 K flush 与 V window；确认全序列 Softmax |
| 55–70 min | PDF 第 6、8–9 页 | 对齐 fake/KIVI、metrics、model/precision/G/R；读 Tables 3–5 |
| 70–82 min | PDF 第 7–8、13–15 页 | 分清 NIAH、LongBench、吞吐；读 Falcon/代码任务/R32 的例外 |
| 82–90 min | README 20–21 | 用自己的话写结论边界，填写 Meeting Card |

### 深入阅读：按问题定位

- **量化方向为什么重要？** 第 2–4 页 Figures 1–2 / Tables 1–2。
- **缓存更新有没有等价的 token 对齐？** 第 5 页 Eq. (3)，第 12 页 Algorithm 1；重点看 K 与 V 不同 tail 长度，以及统一 Softmax。
- **残留高精度值是否改变比较？** 第 8 页 Table 3 caption；第 13 页 Table 6。
- **所有长文本任务都保持吗？** 第 9 页 Table 4；第 14–15 页 Tables 7–10，先分模型版本与任务分母。
- **G/R 是否越大越好？** 第 7、9 页的消融说明与 Table 5。
- **吞吐提升是不是单请求加速？** 第 7–8 页 Section 4.2.4 / Figure 5。

## 23. Reading Questions（留待自己回答）

1. K/V cache 与 \(W_K/W_V\) 分别是什么？KV2 为什么不等于 W2A2？
2. Prefill 与 decode 各做什么？为什么 cache quantization 需要多步生成评价？
3. 给定 T×d 矩阵，K per-channel 的一组元素在哪里？V per-token 又在哪里？
4. 标题的 K/V asymmetric strategy 与 affine asymmetric grid 有什么区别？
5. 用一个包含 outlier channel 的小矩阵说明 per-token K 为什么会伤害普通 channels。
6. V 没有明显固定 outlier channels，为什么也不能随意选择量化方向？
7. Table 2 中 V reconstruction statistic 与 output error 的排序为什么相反？
8. 固定 A 分析 V，与同时量化 K/V 的完整误差有什么区别？
9. G32/R128 分别表示什么？一次 K flush 内有多少 token groups？
10. K 与 V 的高精度窗口是否始终都是 R？K 为什么会出现清空？
11. T=6、G2/R4，追加到 T=8 后 K/V 各有多少低/高精度 tokens？
12. K/V 分界不同，attention 为什么仍能正确对应 tokens？为什么不能分开做 Softmax？
13. residual cache 与 quantization residual 是什么关系？
14. 真正的 packed 2-bit 存储还要保存哪些东西？为何 dequantization 要与 matmul 融合？
15. Table 1 与 Table 3 的 K-C/V-T row 为什么不是同一精度分配？
16. TruthfulQA 此处是什么 metric？Table 4 的 Avg. 与 Table 8 的 Avg. 分母有什么不同？
17. Figure 4 的 depth 与 token/channel/layer depth 分别是什么？NIAH 能证明到哪里？
18. 理想 8×、2.6×、3.47× 各是什么条件、什么指标？
19. G 或 R 变大为什么不一定提高任务分数？更小 R 的收益有什么代价？
20. 对自己的 WAM/VLA 模型，首先要确认哪一种 cache 生命周期，才能判断 KIVI 是否相关？

## 24. Meeting Card（留空，阅读后填写）

- 我理解的具体问题：
- Cache 的产生、保存与重用方式：
- 核心机制，用自己的话解释：
- 模型、任务、KV bit-width、G/R 条件：
- 一条关键实验及物理页码：
- 该实验真正固定与改变了什么：
- 一条不能由论文推出的结论：
- 与我的 FYP idea 重合的部分：
- 我尚未理解的公式或实现步骤：
- 想向导师 / 同学讨论的问题：

## 25. 来源与阅读状态

- **主依据**：本目录 arXiv v2 PDF，15 页；第 2–6 页 background/method，第 6–9 页实验，第 12–15 页算法与补充结果。
- **官方实现入口**：jy-yuan/KIVI；阅读包提供链接，未 clone、编译或运行，未声称 current code 完整复现固定论文协议。
- **本地交付**：PDF、教学 README 与 metadata。没有部署、数值 benchmark 或闭环评测。
- **阅读状态**：留待用户阅读、填写问题与 Meeting Card；未代填学习结论。
