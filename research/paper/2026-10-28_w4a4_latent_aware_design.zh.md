# W4A4 标量量化下的 latent-aware 框架：时间维噪声整形（v3 设计稿，第二版）

2026-10-28。第一版（latent-aware Gramian + 低秩分支）被判定为 Q-WAM 的延伸，已废弃，只在 §10 留作 fallback。本版换轴：**Q-WAM 问"误差落在哪个方向"，我们问"误差在去噪步之间怎样累积、能不能让后面的步把它抵消"**。度量从 d×d 的方向 Gramian 换成每层 T×T 的时间敏感核；载体从高精度分支换成全 INT4 的 innovation 编码 + 误差反馈量化器，零额外比特。A+Z+K 成为两个去噪循环上三个误差去向的度量。

依据：1a/1b/标量基线结果（见 §0）、Q-WAM/SteerQuant/QuantWAMs 阅读包、DA-PTQ/DyQ-VLA/QVLA/ActQuant/Mix-QVLA 阅读包（方向敏感度 + 混合精度这条线已很拥挤）、WorldCache 两篇（去噪步间 token 时间冗余）、rotation-baselines 的 native S4×S4 kernel。本地库中没有任何工作在去噪循环内做误差反馈或噪声整形。

> **2026-10-28 晚，D1/D3 结果（§9.5）改写了本稿的主线。** 误差反馈被证伪：h 从 0 到 −1 单调变差，跨步相干累积（M2）在 Fast-WAM-IDM 上不是主导项。innovation 编码成立，是 motor 侧最强的单项。g32 与 innovation 叠加后 motor 0.0133、future 0.0075，两项都胜 g32。另一方面，相邻步差分 activation 的低比特量化在体系结构领域已有先例（Cambricon-D、Ditto、EXION、DSTAR，§7），innovation 编码本身不能当创新点。修订后的方向见 §9.6；§3–§4 的反馈部分保留作推导记录。

---

## 0. 事实基础

| 事实 | 数字 | 含义 |
|---|---|---|
| 4 bpw 标量 GPTQ 胜 VQ | QuaRot+GPTQ W4A8 motor 0.0098 / future 0.0089；1b VQ 0.0183 / 0.0304 | 权重侧用 GPTQ，不再做 VQ |
| A4 是 W4A4 的主要损失 | BF16 216；BF16 权重 + per-token A4 204；SpinQuant W4A4 203；W16A8 future 0.0011 | 要解决的是 activation 误差 |
| 同一层权重在循环里重复使用 | video 10 步 × 294 token；action 10 步 × 32 token；prefill 1 次 | 误差有时间结构 |
| 相邻去噪步输入相似 | D1：步 1–9 innovation 比中位 0.147；absmax 比 video 0.11、action 0.17；相邻步 RTN 残差相关 video 0.52、action 0.14 | innovation 编码折合 2.6–3.2 位增益；残差相关只在 video 循环中等 |
| 竞品 A4 用 per-token-per-group-32 scale | Q-WAM B.1 | 我们的 W4A4 基线是 per-token 整行 scale，偏弱，必须补 g32 基线 |

---

## 1. 一句话与定位

> **在 imagine-then-execute WAM 里，activation 量化误差先沿 video 去噪循环累积成 imagined future 的误差，再被 action 循环读取并二次累积。我们把每层的误差写成沿去噪步的时间过程，用 action、future、reader 三个读出定义它的时间敏感核，据此设计一个只量化"本步新信息"、并用后续步抵消本步舍入误差的 INT4 activation 量化器；权重 GPTQ，主路径全 INT4。**

| | Q-WAM | SteerQuant | QuantWAMs | 本框架 |
|---|---|---|---|---|
| 问题 | 哪个方向危险 | 哪个区域危险 | 哪些通道/层/步危险 | **误差跨步怎样累积、哪些步能互相抵消** |
| 度量 | d×d Gramian | 区域标量 | Fisher / 干预 | **每层 T×T 时间敏感核，三个 head（A/Z/K）** |
| 载体 | 16-bit 低秩分支 | 共享 D + gain | BF16 通道、W8A8 层、A8 步 | **innovation 编码 + 最优误差反馈，全 INT4；核解出的 A8 步** |
| 额外比特 | +0.08–0.26 bpw，16-bit 分支 | 0 | 高精度例外 | **0**（状态缓冲在 activation 侧） |
| Future | 不生成 | 误差通道 | loss 一部分 | 独立目标 + 跨环传递函数 |

---

## 2. 误差模型：三种去向

记某循环（video 或 action）有 T=10 步，层 ℓ 在第 s 步的输入 \(x_s\)，量化后 \(\hat x_s=x_s+e_s\)。最终输出 \(o\)（a 或 z）的一阶变化：

\[
\Delta o\approx\sum_{s=0}^{T-1}J_s e_s,\qquad J_s=\frac{\partial o}{\partial x_s},
\]

\(J_s\) 包含本步剩余网络与之后所有去噪步（与 Q-WAM Eq.4 同一线性化，但不对 s 求和合并）。平方取期望：

\[
\mathbb E\|\Delta o\|^2\approx\sum_{s,s'}\operatorname{tr}\!\big(J_s^\top J_{s'}\,\mathbb E[e_{s'}e_s^\top]\big).
\]

三种去向：

- **M1 方向**：同一步内，\(e_s\) 沿哪些方向。Q-WAM 的全部内容。Hadamard 后舍入误差在方向上近似各向同性，\(\mathbb E[e_se_{s'}^\top]\approx\sigma^2\rho_{ss'}I\)，于是
  \[
  \boxed{\mathbb E\|\Delta o\|^2\approx\sigma^2\sum_{s,s'}\rho_{ss'}\,\mathcal K(s,s'),\qquad \mathcal K(s,s')=\mathbb E\langle J_s,J_{s'}\rangle_F .}
  \]
  方向被积掉之后，剩下的全是时间结构：\(\mathcal K\) 是模型的**时间敏感核**（T×T，PSD），\(\rho\) 是量化误差的**时间相关**。
- **M2 跨步累积**：\(\rho\) 由量化器决定。确定性 RTN 对相似输入给出相似误差，\(\rho_{ss'}\to1\)，损伤趋向 \(\sigma^2\mathbf 1^\top\mathcal K\mathbf 1\)，比白噪声（\(\sigma^2\operatorname{tr}\mathcal K\)）最多大 T 倍。这是现有 W4A4 方法里从未被处理的一项。
- **M3 跨环传递**：video 循环的累积误差进入 z，经 prefill K/V 成为 action 循环的静态条件误差，在 action 循环里再以 \(\rho=1\) 累积一次。

Q-WAM 处理 M1 要花高精度比特；M2、M3 可以在量化器里零成本处理，这是本框架的立足点。

**D3 实测（§9.5）：M2 不成立。** dither 打散跨步相关后，损伤增加的倍数约等于每步误差能量增加的倍数（约 2 倍）。一阶 Σ-Δ 同样只增加损伤。说明 \(\mathcal K\) 的非对角远小于对角，RTN 误差的跨步相干在本模型里不放大损伤。M3 仍然存在：video 循环 A4 单独贡献 motor 0.0176（A4_actA8 臂）。

---

## 3. 度量：时间敏感核与三个 head

### 3.1 定义

对层 ℓ、读出 \(h\in\{A,Z,K\}\)：

\[
\mathcal K^h_\ell(s,s')=\mathbb E\big\langle J^h_{\ell,s},\,J^h_{\ell,s'}\big\rangle_F ,
\]

- \(J^A=\Lambda_a^{1/2}\,\partial a/\partial x\)，\(a\) 为 first10 motor，\(\Lambda_a=I/60\)；
- \(J^Z=\Lambda_z^{1/2}\,\partial z/\partial x\)，\(z\) 为 frame1/2 的 clean future，\(\Lambda_z=I/\|z_T\|^2\)；
- \(J^K=F^{1/2}J_R\,\partial z/\partial x\)，\(J_R=\partial R_T/\partial z\) 是冻结 BF16 reader 的 Jacobian，\(F\) 是 soft-bin 分布的 Fisher（对角 60×60，闭式）。

合成核沿用校准损失的权重：

\[
\mathcal K_\ell=\mathcal K^A_\ell+\lambda\,\mathcal K^Z_\ell+\beta\,\mathcal K^K_\ell,\qquad\lambda=0.04245,\ \beta=0.00608 .
\]

**命题 1。** \(\sigma^2\operatorname{tr}(R\,\mathcal K_\ell)\) 是 latent-aware 目标 \(A+\lambda Z+\beta K\) 对层 ℓ 的时间相关为 R 的各向同性 activation 误差的二阶精确响应（A、Z 为零点平方误差，K 的 Hessian 为 Fisher；Gauss–Newton 残差项在全精度点消失）。

### 3.2 三个 head 在两个循环上各管什么

| 层所在 | \(\mathcal K^A\) | \(\mathcal K^Z\) | \(\mathcal K^K\) |
|---|---|---|---|
| action 循环（307 层） | 非零：action 步之间的累积 | 0 | 0 |
| video 循环（306 层）+ proprio | 非零，但经 z 间接 | 非零：video 步之间累积进 future | 非零：video 步累积进 z 后被 reader 读出的部分 |
| prefill、cross-attn K/V 投影（1 次） | 静态条件误差，\(\rho\equiv1\) | 同 | 同 |

- A 是 action 循环的核，与 Q-WAM 的对象相同，但形式是时间核而不是方向 Gramian。
- Z 是 video 循环对 imagined future 的核：它关心 future 的全部内容，不只是当前 reader 读到的部分。校准集之外（LIBERO-Plus 扰动）reader 会读 future 的不同部分，Z 给的是不依赖 reader 的地板。
- K 是**跨环传递函数**：video 第 s 步的误差经 z 进入 action 循环后伤多少动作。它走冻结 BF16 reader，不被 action 循环自身的量化污染；\(F\) 对 action 各维做标准化与饱和。这就是 "latent impact on action" 在时间域的形式，只有 IDM 有这一项。

三个核可能偏好不同的时间滤波：Z 对各 video 步近似均匀，K 和 A 通常偏重后段步（SteerQuant Fig.1b：最后一步比第 0 步伤动作 100 倍）。合成核把这个取舍交给同一组 λ、β。

### 3.3 两条估计路径

**路径 1：probe（一次 backward 给所有层所有步）。** 取独立高斯 \(u\in\mathbb R^{60}\)、\(v\in\mathbb R^{\dim z}\)、\(w\in\mathbb R^{60}\)，合成 cotangent \(\Lambda_a^{1/2}u\)（对 a）与 \(\sqrt\lambda\Lambda_z^{1/2}v+\sqrt\beta J_R^\top F^{1/2}w\)（对 z），后者由 reader session 反传得到——即现有 loss-pair worker 的分阶段 VJP，只换 cotangent。一次 backward 读出每层每步输入的梯度 \(g_{\ell,s}\)（对 token 求和），累加 \(g_{\ell,s}^\top g_{\ell,s'}\)。每层每 head 只存 10×10，没有 d×d 累加器；P=4–8 probe，N=40–80 context，4–18 GPU-h。把 \(u,v,w\) 分开打就得到三份核，消融免费。per-consumer attribution 同 Q-WAM A.4。可按 token 组（frame0 / future frame / action 0–9 / action 10–31）分别累加，得到组级核。

**路径 2：干预审计（无梯度，精确）。** 在 BF16 模型里，对选定（循环、层组、步集合）注入等每步能量、不同时间相关的误差：白噪声、真实 RTN 残差（相干）、整形后残差；直接测 ΔA、ΔZ、ΔK。forward-only，用 engine 每 query 0.5–0.9 s，66 block × 3 类 × 10 context × 若干步集合，数小时。它既是 Fig.1 级的机制图（"同样每步误差能量，相干与整形之间差 X 倍"），也是命题 1 一阶近似的校验。

---

## 4. 载体：innovation 编码 + 最优误差反馈的 INT4 量化器

### 4.1 量化器

每层维护跨步状态：重建 \(\hat x_{s-1}\)、最近 m 步的舍入误差 \(w_{s-1},\dots,w_{s-m}\)、上一步输出 \(y_{s-1}\)。第 s 步：

\[
\begin{aligned}
\Delta_s&=x_s-\hat x_{s-1} &&\text{innovation：只有本步新信息}\\
v_s&=\Delta_s+\textstyle\sum_{k=1}^{m}h_{s,s-k}\,w_{s-k} &&\text{误差反馈}\\
q_s&=Q_4(v_s),\quad w_s=q_s-v_s &&\text{per-token absmax INT4，}w_s\text{ 为本步舍入误差}\\
\hat x_s&=\hat x_{s-1}+q_s,\qquad y_s=y_{s-1}+W_q\,q_s &&\text{输出在 GEMM epilogue 累加}
\end{aligned}
\]

恒等式：\(y_s=W_q\hat x_s\)，而

\[
\hat x_s-x_s=w_s+\sum_k h_{s,s-k}w_{s-k}\;=\;(Hw)_s ,
\]

H 为单位下三角。所以**下游看到的误差是白噪声 w 经滤波器 H 整形后的结果**，σ² 由 innovation 的范围决定而不是 \(x\) 的范围。第 0 步 \(\hat x_{-1}=0\)，量化完整 \(x_0\)；它的误差 \(w_0\) 由后续步抵消。

### 4.2 两个命题

**命题 2（最优反馈滤波器）。** 固定 \(\mathcal K_\ell\)，在所有单位下三角 H 上，
\[
\min_H\ \sigma^2\operatorname{tr}(H^\top\mathcal K_\ell H)=\sigma^2\sum_{k}D_k,\qquad
D_k=\mathcal K_{kk}-\mathcal K_{k,>k}\,\mathcal K_{>k,>k}^{-1}\,\mathcal K_{>k,k},
\]
最优系数 \(h_{>k,k}=-\mathcal K_{>k,>k}^{-1}\mathcal K_{>k,k}\)，即"用之后各步的敏感性对第 k 步敏感性做最小二乘预测"。证明：\(\sum_sJ_se_s=\sum_kw_k\big(J_k+\sum_{s>k}h_{s,k}J_s\big)\)，各 k 独立最小化。\(D_k\) 是逆时间序的 Schur 补，\(\mathcal K=L D L^\top\) 的对角。

- 一阶 Σ-Δ（\(h_{k+1,k}=-1\)）是 \(J_{k+1}\approx J_k\) 的特例。
- 相邻步不相干（\(\mathcal K\) 对角）时 \(h=0\)，自动退化为普通量化，不会变差。
- 最后一步 \(D_{T-1}=\mathcal K_{T-1,T-1}\)：没有未来可抵消。这决定了 §4.3 的比特分配。

**命题 3（innovation）。** per-token absmax 下 \(\sigma_s^2\propto\max_t\|\Delta_{s,t}\|_\infty^2\)。相邻步相似度 \(r_s=\|\Delta_s\|/\|x_s\|\) 直接换算成有效位数增益 \(\log_2(1/r_s)\)；反馈项只加 \(O(\Delta_{\text{grid}})\) 的幅度，不改变 scale 量级。

**D3 实测。** 固定 h 的三档结果单调：h=0 最好，h=−0.5 次之，h=−1 最差（future 0.0209 / 0.0250 / 0.0349）。按命题 2，这对应 \(\mathcal K_{k+1,k}\ll\mathcal K_{k+1,k+1}\)：sampler 会把早期步的误差收缩掉，后期步更敏感，Σ-Δ 把每步误差推到更敏感的下一步，同时让每步误差能量翻倍。最优滤波器在本模型上退化为 h≈0，命题 2 只剩"非对角小时不反馈"这一条推论有用。

**算例。** 两步、等敏感、相干 ρ：\(\mathcal K=\begin{pmatrix}1&\rho\\\rho&1\end{pmatrix}\)。白噪声损伤 \(2\sigma^2\)；RTN 相干误差（相关 c）\((2+2c)\sigma^2\)；Σ-Δ \((3-2\rho)\sigma^2\)；最优 \(h_{1,0}=-\rho\) 得 \((2-\rho^2)\sigma^2\)。ρ=c=0.9：3.8 / 2 / 1.2 / 1.19。十步、完全相干（\(\mathcal K=\mathbf 1\mathbf 1^\top\)）：相干 100σ²、白 10σ²、整形 σ²。实际核在两者之间，由 D4 测出。

### 4.3 比特分配：核告诉你哪些步必须给 A8

反馈能抵消的是 \(\mathcal K_{kk}-D_k\)；抵消不了的 \(D_k\) 只能靠降低 \(\sigma_k^2\)。给第 k 步 A8 把 \(\sigma_k^2\) 降 1/16。在计算预算 \(B\)（A8 步的 GEMM 时间）下：

\[
\min_{b_k\in\{4,8\}}\ \sum_{\ell,k}\sigma_{\ell,k}^2(b_k)\,D_{\ell,k}\quad\text{s.t.}\ \sum\text{cost}\le B ,
\]

按 \(D_{\ell,k}\) 的边际收益贪心。预期结果：各循环最后 1–2 步、低相干层拿 A8；action 循环 GEMM 是 latency-bound，INT8 与 INT4 用时相同，所以 action 侧的 A8 步几乎免费；video 侧每个 A8 步约 +10% 该循环 GEMM 时间。这是 Q-WAM "action mass → 保护哪个 expert"的对应物，也是 QuantWAMs "固定一个 A8 步"的推广：**在哪一步花比特由核的 Schur 补决定**，而非人工选。

### 4.4 非循环位置

prefill（129 token，1 次）、proprio encoder、cross-attn K/V 投影只执行一次，误差对 action 循环是静态条件误差（\(\rho\equiv1\)），且没有后续步可抵消：全部 A8。计算量占比小于 5%。

### 4.5 权重

Smooth α=0.5 + signed H128，GPTQ INT4 g128 BF16 scale（4.125 bpw，Hessian 管线已有）。权重误差是静态的，对循环而言 \(\rho\equiv1\)，这是 W4 在扩散模型里比在 LLM 里更伤的原因，GPTQ 的误差反馈把它压到 RTN 的 1/5。g32（4.5 bpw）作消融。

---

## 5. 为什么这是 latent-aware 框架而不是通用扩散量化技巧

1. **度量由 A/Z/K 定义**。每层的反馈滤波器和 A8 步分配都从 \(\mathcal K^A+\lambda\mathcal K^Z+\beta\mathcal K^K\) 解出。换成只有 A 的核，video 循环的滤波器只看 reader 当前读到的方向；换成只有 Z，action 循环没有核。
2. **两环结构给每个 head 一个不可替代的时间角色**（§3.2）。K 是 video 步到 action 的传递函数，是只有 imagine-then-execute 模型才有的项。
3. **跨环传递本身是方法的一部分**（M3）：video 循环整形后的残余误差进入 z，z 的误差在 action 循环里以 \(\rho=1\) 累积，这正是 K 核所度量的量，也解释了为什么 Z/K 核要同时进入 video 层的滤波器设计。
4. **审计实验直接测三个量**。ΔA、ΔZ、ΔK 是模型无关的测量，论文 Fig.1 用它展示：同样每步误差能量，相干、白、整形三种时间结构在 future 与 action 上的损伤差距。
5. **原校准损失仍可用于最后一公里**：若需要，用 A+Z+K 端到端微调每层每步的反馈系数与 clip（参数量 614×10×m，极小，连续），复用 loss-pair worker。这是可选消融，不是主方法。

---

## 6. 跨模型适配：方法接口与三个目标模型

方法只依赖四个抽象对象，不依赖 MoT 还是共享 backbone、串行还是联合去噪：

| 接口对象 | 定义 | 怎样自动得到 |
|---|---|---|
| 循环 \(\mathcal L\) | 一个迭代采样器：T 步、固定 token 网格、每步执行同一组 Linear | 按 query 内调用次数 hook：调用 T 次且 token 形状不变的 Linear 属于该循环（condition cache 已有同类统计） |
| 读出 \(a,z\) | action chunk；该循环产出的 future latent | 模型自己的输出 |
| reader \(R_T\) | 冻结 BF16 后缀：给定 student 的 z（或切点处的 noisy future），跑原始 action 通路到 a | Fast-WAM、Cosmos 已实现；LingBot-VA 新写 |
| 静态位置 | 每 query 只执行一次、为循环提供条件的 Linear | 调用次数为 1 的 Linear |

量化器（§4.1）、核（§3.1）、滤波器（命题 2）、A8 步分配（§4.3）全部定义在这四个对象上。模型差异只改变 T、token 组和 reader 切点。

### 6.1 逐模型映射

| | Fast-WAM-IDM（主平台） | Cosmos-Policy | LingBot-VA |
|---|---|---|---|
| 结构 | MoT：video expert + action expert | 单一共享 DiT（Cosmos-Predict2-2B），9 个 latent slot | 共享 backbone，自回归 diffusion，KV cache 跨 chunk |
| 循环 | video T=10（294 token）→ prefill（1 次）→ action T=10（32 token） | **一个联合循环 T=5**，video/action/value slot 同步去噪 | video 相 T=20 → action 相 T=50 |
| z | frame1/2 clean future | future slot 6/7（终点或切点 NFE 3 的 noisy 状态） | video 相终点的预测 latent |
| a | 32×7，监督 first10 motor | slot 4，16×7，监督全 16 步 | action chunk |
| reader | BF16 prefill + action 循环 | 冻结 joint-DiT 后缀，从 NFE 3 起替换 future slot（已实现，cut_evaluation=3） | BF16 action 相（含已缓存 KV） |
| 核的 head | video 层：A（经 z）、Z、K；action 层：A | 按 **token 组**分：future 行 → Z；future 行在 s<3 的误差 → K（经后缀）；全部行 → A | video 相层：A、Z、K；action 相层：A |
| 静态位置 | prefill、proprio encoder、cross-attn K/V 投影 | 文本 cross-attn K/V、条件编码 | 观测 re-grounding 编码、上一 chunk 的 KV |
| 对本方法的意义 | 两环串行，M3 跨环传递最清楚 | **最难情形**：T=5，时间冗余最少；检验方法在 T 小时是否平滑退化（h→0，退为步级 A8 分配） | **最好情形**：action 相 50 步，相邻步最相似；也是唯一能与 QuantWAMs、Q-WAM 报告值同模型对照的平台 |
| 现状 | reader、engine、全部基线、280 面板齐全 | reader、engine 适配、六臂闭环齐全（BF16 233） | 全新：runtime、reader 切点、engine、评测都要写 |

Cosmos 的联合循环说明为什么 K 必须按 token 组定义：同一步里 future 行与 action 行共用一层权重，A 核把两者的影响混在一起，K 核单独给出"future 行的误差经后缀被读成 action"这条通道，Z 核给出 future slot 自身的保真（Cosmos 的 planning 模式直接消费它）。没有 Z/K，联合循环上就无法区分"误差伤了 future"与"误差伤了 action 行本身"。

LingBot-VA 的 KV cache 跨 chunk 复用，是跨 query 的静态条件误差；按 §4.4 规则处理（本范围内 KV 本就 BF16）。异步执行不改变每步量化。

### 6.2 第四个候选

Faster-WAM（hustvl/FasterWAM）：Fast-WAM 家族，SparseMoT + Interval KV-Fusion，官方仓库自带 LIBERO-Plus 评测脚本和 checkpoint。若时间允许，它是成本最低的第四个模型，而且其论文协议就是 LIBERO-Plus。DreamZero（14B、多卡）不做。

### 6.3 逐模型 baseline 矩阵

| baseline | 类型 | Fast-WAM-IDM | Cosmos-Policy | LingBot-VA |
|---|---|---|---|---|
| BF16 | 上界 | 216 | 233 | 待跑 |
| QuaRot W4A4（per-token A4） | 旋转 | 195 | 227 | 待跑 |
| SpinQuant W4A4 | 学旋转 | 203 | 221 | 待跑 |
| ViDiT-Q W4A4（**按 timestep 的静态 scale**） | 扩散专用 | 203 | 232 | 待跑 |
| SVDQuant 适配 | 低秩分支 | 108 | 78 | 待跑 |
| GPTQ + g32 A4（Q-WAM 量化设定，无 ASP） | 组 scale | 新 | 新 | 新 |
| Q-WAM-adapted（AOG + ASP r=32） | 方向保护 | 新 | 新 | 新 |
| SteerQuant-adapted（区域图 + D + gain） | 区域 scale | 可选 | 可选 | 可选 |
| dither A4 / 一阶 Σ-Δ A4 | 时间相关控制 | 新 | 新 | 新 |
| action 循环 A8 | 廉价混合精度 | 新 | 不适用（联合循环）→ action 行 A8 | 新 |
| Ours | — | 新 | 新 | 新 |

ViDiT-Q 值得单独说明：它按 timestep 换 **scale**，我们按 timestep 整形 **误差**；两者正交，ViDiT-Q 是本方法在时间维上最近的现有对照。Cosmos 上 ViDiT-Q 已只差 BF16 一条，闭环没有区分度，Cosmos 的比较以 held-out action/future 误差和 W4A4 kernel 路径为主。

### 6.4 跨模型实验的顺序

1. Fast-WAM-IDM 全套（§9）。
2. Cosmos-Policy：D1/D3 诊断直接复用现有 runtime；若 T=5 下 innovation 与整形收益小，如实报告并展示 A8 步分配承担主要作用——这是"方法随 T 平滑退化"的证据，不是失败。
3. LingBot-VA：先写 runtime 与 reader 切点（video 相终点），跑 BF16 与两三个主要基线，再上本方法；选 LIBERO-Long 公开 checkpoint 与 QuantWAMs 的 98.5→98.0 同口径。
4. 跨模型主图：三个模型 × {最强 W4A4 基线, GPTQ+g32 A4, Q-WAM-adapted, Ours} 的 held-out 误差与闭环成功率。

## 7. 与已有工作的关系（写 related work 用）

| 工作 | 关系 |
|---|---|
| Σ-Δ networks（O'Connor & Welling 2016）、Delta networks（Neil et al. 2017） | 在视频流/RNN 的时间轴上量化 activation 的差分并做误差反馈；我们把时间轴换成去噪步，并给出按敏感核解最优滤波器 |
| 噪声整形理论（Gerzon–Craven、Σ-Δ ADC） | 命题 2 是有限时域、时变、以下游敏感核为"带内"定义的噪声整形 |
| GPFQ / Σ-Δ 权重量化（Lybrand & Saab）、GPTQ | 权重侧的误差反馈；我们是 activation 侧、跨步 |
| Error-feedback SGD | 同一个 telescoping 技巧 |
| PTQD、TAC-Diffusion | 统计校正扩散采样器里累积的量化噪声（均值/方差）；不改量化器、不做反馈 |
| DeepCache / WorldCache | 利用去噪步间 token 时间冗余来**跳过**计算；我们用同一冗余来**省比特**：只量化 innovation |
| **Cambricon-D**（Kong et al., ISCA 2024） | UNet 加速器，相邻步差分计算，差分按通道 INT4、离群 FP16，平均 7.2 位。**innovation 编码的直接先例** |
| **Ditto**（Kim et al., arXiv 2501.11211, 2025） | DiT 加速器，第一步全精度，之后算相邻步差分，差分 INT4/INT8，平均 5.4 位 |
| EXION（Heo et al., 2025） | 把 FFN 的小差分直接置零，DiT 上平均 11.2 位 |
| **DSTAR**（Zhang et al., MICRO 2026, arXiv 2607.15846） | DiT 加速器，差分 activation 按子通道混合 INT2/4/6/8，平均 4.3 位，外加注意力复用 |
| 与这四篇的区别 | 它们都在定制加速器上、按差分**幅度**选位宽、第一步全精度、只评生成质量。我们在通用 GPU tensor core 上做统一 W4A4，权重同为 INT4，位宽由下游 A/Z/K **敏感度**决定，并处理 WAM 的两环与跨环传递 |
| QuantWAMs 的 A8 步保护、SteerQuant 的步图、DyQ-VLA 的运行时切换 | 都在"哪一步给精度"上做选择；我们的选择由 Schur 补给出，并与反馈互补 |
| Q-WAM | 方向维；与本框架正交，可叠加（§10 fallback） |

---

## 8. Kernel 与 runtime

- **Activation kernel**（融合）：Smooth 除法 + signs + FWHT → 减 \(\hat x_{s-1}\) → 加反馈 \(\sum h w\) → per-token absmax → INT4 打包 + FP32 scale → 更新 \(\hat x_s\)、写 \(w_s\)。
- **GEMM**：INT4×INT4→INT32（现有 S4×S4 `mma.sync` 路径需重写成 tile 化 kernel）→ epilogue：× a_scale × w_scale + \(y_{s-1}\)（bias 在第 0 步折入）→ 写 \(y_s\) 与输出。per-token scale，K 循环内不需要分组重缩放，是 A100/Ada 上最简单的 INT4 路径；Blackwell 上可叠加 NVFP4 组 scale。
- **状态缓冲**：video 循环 \(\hat x\)（BF16，约 0.45 GB）、\(y\)（BF16，约 0.5 GB）、\(w\) 历史 m=2（存为 Δ 的 INT8 分数，约 0.25 GB），合计约 1.2 GB；action 循环十分之一以下。每次 query 开始清零；CUDA Graph 固定地址；condition cache 不受影响。
- **显存账**：目标 blocks 11.2 GiB → 约 3.1 GiB（4.125 bpw）+ 1.3 GB 状态。相比 Q-WAM 的 16-bit 分支是把开销从权重侧移到 activation 侧，总量相近，但不增加 bpw、不加 GEMM。
- **速度上限**：GEMM 占 BF16+cache+Graph 去噪时间的 46%，A100 上 W4A4 端到端上限约 1.35×。速度不作主贡献；主贡献是精度、度量与"零比特"的载体。

---

## 9. 实验计划与 go/no-go

### 9.1 诊断（先做，都便宜）

| 编号 | 内容 | 成本 | 判据 |
|---|---|---|---|
| **D1** | 10 个 context，记录每层每步 activation：innovation 比 \(r_{\ell,s}\)；RTN 残差跨相邻步的相关 c；按 token 组分开 | 1 GPU-h | \(r\) 中位数 < 0.5 或 c > 0.5 任一成立即有戏 |
| **D2** | 干预审计：{video 早/晚层组, action 早/晚层组} × {白, 相干 RTN, 一阶整形} × 10 context × 步集合，测 ΔA/ΔZ/ΔK | 2–4 GPU-h | 相干 ≫ 白 > 整形，比例与命题 1 一致 |
| **D3** | eager fake-quant held-out 40×2，BF16 权重：per-token A4；+dither；+一阶 Σ-Δ；+innovation；+两者；g32-per-token A4；action 循环 A8；全 A8 | 每臂约 10 min | **innovation+整形 A4 的 motor/future 接近 W16A8（0.0053/0.0011）的一半以内，且优于 g32 A4** |
| **D4** | probe 估计 \(\mathcal K^{A,Z,K}_\ell\)（N=40，P=4 起）；解最优 h；按 Schur 补出 A8 步分配 | 4–18 GPU-h | \(\sum D_k/\operatorname{tr}\mathcal K\) 中位数 < 0.5 |

D3 是整个方向的 go/no-go，一天内出结果。

### 9.2 主表（Fast-WAM-IDM，LIBERO-Plus 280 + 标准 LIBERO）

| 组 | 行 |
|---|---|
| 上界 | BF16 216；W16A8；GPTQ W4A8 |
| W4A4 已有 | QuaRot 195、SpinQuant 203、Smooth/H 200、ViDiT-Q 203、SVDQuant 108 |
| W4A4 新基线 | GPTQ + per-token A4；GPTQ + g32 A4（Q-WAM 设定）；Q-WAM-adapted（AOG+ASP r=32）；GPTQ + action 循环 A8 |
| Ours | GPTQ + innovation/整形 A4（统一 h）；+ 核解 h；+ 核解 A8 步 |

### 9.3 消融

1. 时间相关控制：RTN / dither / Σ-Δ / 最优 h —— 隔离"相干累积"这一机制。
2. innovation 有无。
3. 核的 head：A / A+Z / A+Z+K 解出的 h 与 A8 步分配，同预算。
4. A8 步分配：核解 vs 固定最后一步 vs 固定第 0 步（QuantWAMs 式）。
5. 按 token 组的 h vs 按层的 h。
6. 权重：RTN / GPTQ；g32 / g128。
7. Cosmos-Policy 迁移：单 DiT 5 NFE，同一量化器，z 为 slot 6/7。

### 9.4 目标

held-out motor ≤ 0.012、future ≤ 0.012；280 闭环 ≥ 210，对 BF16 配对损失 ≤ 10，明显高于 g32 A4 与 Q-WAM-adapted 两行。

### 9.5 D1/D3 结果（作业 26039756.pbs101，2026-10-28）

设置：BF16 权重，H_W16A4 坐标，只换 activation operand；held-out 40 context × 2 seed。A4 臂与历史 H_W16A4 的 80 个 endpoint 逐位一致。完整表在 `experiment/idea-validation/fastwam-temporal-a4-d1d3-20261028/pulled/26039756.pbs101/RESULTS.zh.md`。

| 臂 | first10 motor | future NMSE | 补回 A4→A8 motor | 补回 A4→A8 future |
|---|---:|---:|---:|---:|
| A8 | 0.0028 | 0.0006 | 100% | 100% |
| A4 | 0.0426 | 0.0384 | 0 | 0 |
| A4_g32 | 0.0377 | 0.0125 | 12% | 69% |
| A4_dither | 0.0629 | 0.0698 | −51% | −83% |
| A4_sd（h=−1，无 innovation） | 0.0597 | 0.0501 | −43% | −31% |
| A4_inno（h=0） | 0.0153 | 0.0209 | 69% | 46% |
| A4_inno_sd05（h=−0.5） | 0.0156 | 0.0250 | 68% | 36% |
| A4_inno_sd（h=−1） | 0.0190 | 0.0349 | 59% | 9% |
| A4_actA8 | 0.0176 | 0.0384 | 63% | 0 |
| A4_preA8 | 0.0414 | 0.0384 | 3% | 0 |
| A4g32_inno_sd | **0.0133** | **0.0075** | **74%** | **82%** |
| A4_inno_sd_actA8 | 0.0144 | 0.0349 | 71% | 9% |

判据：

- **D1 go。** 步 1–9 的 innovation 比中位数为 0.147，低于 0.5。相邻步残差相关中位数为 0.39，低于 0.5；它只在 video 循环达到 0.52。
- **D3 预注册 no-go。** 最佳跨步臂 A4_inno 补回 motor 69%、future 46%，future 未达 50%，也高于 g32 的 0.0125。
- **dither 机制对照不成立。** A4_dither 的 future 比 A4 高 0.0314。

结论：

1. **误差反馈被证伪。** h 越负越差；不带 innovation 的 Σ-Δ 比 A4 还差 40%。原因见 §4.2 的实测段。
2. **innovation 编码成立。** A4_inno 的 motor 低于"action 循环整体 A8"，全 INT4 就做到了。
3. **g32 与 innovation 管不同的轴。** g32 压 token 内的通道离群，主要修 future；innovation 压跨步的幅度，主要修 motor。两者叠加后 motor 在 40 个 context 上全胜 g32，future 36/4 胜。叠加臂带的是 h=−1；按结论 1，h=0 应更好，尚未测。
4. **prefill 与静态位置不是瓶颈。** prefill 改 A8 只降 motor 2.7%。
5. **剩余误差的来源未拆开。** 每个循环第一步没有参考，按完整范围量化 A4；action 第 9 步 innovation 比升到 0.37。这两处是 A8 的首要候选，D3b 测。

### 9.6 修订方向（推荐）

**载体：** g32 + 闭环 innovation（h=0），GEMM 全 INT4，epilogue 累加 \(y_s=y_{s-1}+W_q q_s\)；少量位置给 A8。零额外 bpw，不需要定制硬件。

**度量：** h=0 时各步误差独立，时间核只剩对角。层 ℓ 第 s 步的损伤近似为

\[
\mathrm{dmg}_{\ell,s}=\sigma^2_{\ell,s}\big(\mathcal K^A_\ell+\lambda\mathcal K^Z_\ell+\beta\mathcal K^K_\ell\big)(s,s),
\]

其中 \(\sigma^2_{\ell,s}\) 是 innovation 量化误差能量，校准集上直接测；对角核由 probe 估计，比全 T×T 核便宜。按 \(\mathrm{dmg}\) 降幅与 A8 代价之比贪心分配 A8。

**latent-aware 的落点：** 同样的 innovation 幅度，video 层按 Z 与 K 计损伤，action 层按 A 计。K head 给出 video 步经 future 伤动作的份额，这是 WAM 独有的一项。与 DSTAR、Ditto 的区别是位宽由下游敏感度决定，而不是由差分幅度决定。

**创新性评估：** 比原稿弱。innovation 编码要引用 Cambricon-D、Ditto、DSTAR；"敏感度驱动的步级精度分配"与 QuantWAMs、SteerQuant 的步图相邻。能守住的是三点：A/Z/K 三个 head 在两环上的分工；通用 GPU 上的统一 W4A4；跨模型（Fast-WAM-IDM、Cosmos-Policy、LingBot-VA）的同一接口。是否接受这个定位，要和老师确认。

**D3b + D3c（一个 g2 作业，两卡并行）：**

| 卡 | 臂 | 问的问题 |
|---|---|---|
| 0 | A4g32_inno（h=0） | 全 INT4 能到哪 |
| 0 | A4_inno_I8、A4g32_inno_I8 | 每个循环第一次调用给 A8（I 帧）补回多少 |
| 0 | A4g32_inno_actA8 | 剩余误差里 action 循环占多少 |
| 0 | A4_inno_lin05 | 二阶预测 \(\hat x_{s-1}+0.5(\hat x_{s-1}-\hat x_{s-2})\) 能否再压 innovation |
| 0 | A4g32_inno_sd | 锚点，须与 D3 逐位一致 |
| 1 | H 坐标下收 Hessian，GPTQ INT4 g128；W4 × {A8, A4, A4_g32, A4g32_inno, A4g32_inno_I8} | 真实 W4A4 的 held-out 数字，对照 QuaRot+GPTQ W4A8 的 0.0098 / 0.0089 |

---

## 10. 备选载体与度量（供选择；本稿主线为 §3–4，跨模型见 §6）

| 候选 | 类型 | 评价 |
|---|---|---|
| innovation + 误差反馈 INT4（原主线） | 载体 | D3 后：反馈被证伪；innovation 有效但有体系结构先例（§7）。修订见 §9.6 |
| 核解的步级 A8（主线组成） | 载体 | action 侧免费；由命题 2 的 \(D_k\) 给出位置 |
| 非循环位置 A8（主线组成） | 载体 | 静态条件误差无法抵消 |
| action 循环整体 A8 | 载体/基线 | 近乎免费的强基线，必须列入 |
| 按 token 行的混合精度（frame0 vs future、action 0–9 vs 10–31） | 载体 | GEMM 按行切分即可；若组级核差异大则加入 |
| 采样器级校正（PTQD/TAC 式，每步每通道 bias，用 A+Z+K 学） | 载体 | 补均值偏差，与整形补方差互补；可选 |
| 低秩子空间分支（Q-WAM/ResQ/SVDQuant） | 载体 | fallback；与本框架正交可叠加，但会回到"延伸 Q-WAM" |
| LoRA 式误差补偿（A+Z+K 训练） | 载体 | 已知模式，只作消融 |
| 干预审计 | 度量 | 无梯度、精确；路径 2 |
| 时间敏感核（probe） | 度量 | 路径 1；T×T 而非 d×d |
| 方向 Gramian | 度量 | 已废弃为主线 |
| HSIC / evidence map（ActQuant、Mix-QVLA） | 度量 | 间接，不采用 |

---

## 11. 风险

| 风险 | 对策 |
|---|---|
| 10 步 sigma-shift 调度下相邻步不够相似，innovation 收益小 | 整形只需 Jacobian 相干（核的非对角），不需要 activation 相似；D1 分开测两者 |
| 一阶时间模型在非线性下游失效 | D2 审计直接测；滤波器系数可用 A+Z+K 端到端微调 |
| 最后一步无法抵消且最敏感 | §4.3 的 A8 步分配；action 侧免费 |
| 状态缓冲约 1.3 GB | w 存 INT8 分数；m=1–2；按层关闭 \(D_k/\mathcal K_{kk}>0.9\) 的层 |
| 竞品 g32 A4 本身已接近无损，留给我们的空间小 | D3 已排除：g32 A4 motor 0.0377，离 A8 的 0.0028 很远；g32 主要修 future |
| innovation 编码被视为 Ditto/DSTAR 的软件移植 | 贡献点放在 A+Z+K 敏感度驱动的精度分配与 WAM 两环（§9.6），innovation 只作载体组成并引用先例 |
| kernel 工程量 | 精度结论全部在 eager fake-quant 下先出；kernel 只影响速度表 |

---

## 12. 实施顺序

| 阶段 | 工作 | 复用 |
|---|---|---|
| 1 | ~~D1 + D3~~ 已完成（§9.5） | — |
| 1b | D3b + D3c（§9.6）：h=0、I 帧 A8、二阶预测；H 坐标 GPTQ W4 | temporal-a4-d1d3 的 worker 与量化器、w4a8-scalar-baselines 的 GPTQ |
| 2 | ~~D2 审计~~ 降级：M2 已由 D3 否定，只在论文需要机制图时做 | 同上 + engine |
| 3 | D4 改为只估对角核 \(\mathcal K^{A,Z,K}_\ell(s,s)\)：loss-pair worker 换 cotangent | worker.py、frozen_reader.py |
| 4 | 按 \(\mathrm{dmg}_{\ell,s}\) 分配 A8、bundle 导出；eager 主臂 held-out | w4a8-scalar-baselines 管线 |
| 5 | 新基线：g32 A4、Q-WAM-adapted、action A8 | 同上 |
| 6 | kernel：融合 activation kernel（含状态更新）+ tile 化 INT4 GEMM + 累加 epilogue；接 engine | abc_runtime、graph_runtime、现有 PTX 片段 |
| 7 | 闭环 280；Cosmos-Policy 全套（D1/D3 复用现有 runtime） | plus-engine-eval、cosmos reader/engine |
| 8 | LingBot-VA：runtime、reader 切点（video 相终点）、engine、BF16 + 主要基线 + Ours | 需新写；参照 Cosmos 适配结构 |

---

## 13. 与 v2 规划的关系

v2 路线 3（VQ + INT8）终止；第一版 v3（Gramian + 分支）废弃。保留：LIBERO-Plus 主评测、C80、engine、标量基线管线、reader 与分阶段 VJP、A+Z+K 的 λ/β。
