# Phase-Controlled Subtractive Dither：数学最小门槛

## 固定输入、固定 action 投影

对 site r 的一行固定输入，令 t_ri=x_ri/Δ_r、d_r={U+φ_r}，并令 f(z)=round(z)-z。未饱和时

$$e_ri=x̂_ri-x_ri=Δ_r f(t_ri+U+φ_r), \quad E_U e_ri=0.$$

此处 Δ_r 可由该固定行的 x_r 自适应计算；只要它在抽当前 U 前确定且不触发 clipping，条件结论仍成立。f 的复 Fourier 系数为 c_k=i(-1)^(k+1)/(2πk)（k≠0），故其相关函数为给定的周期延拓 R(τ)=1/12-τ(1-τ)/2，0≤τ≤1。

冻结下游 action 投影 J_r，设 Ŵ_r 为实际 W4 权重，v_ri=J_r Ŵ_r[:,i]。把同一 site 的坐标（也可包含 token rows）合并，定义

$$h_rk=c_k \sum_i Δ_ri v_ri e^{i2πk t_ri}.$$

一阶 action activation residual g=Σ_r J_r Ŵ_r e_r 的精确固定输入均方目标是

$$J(Φ)=E_U\|g\|^2=\sum_{k\ne0}\left\|\sum_r h_rk e^{i2πkφ_r}\right\|^2.$$

等价地，任意坐标对的协方差是 Δ_ri Δ_sj R({t_sj-t_ri+φ_s-φ_r})，再乘 action 投影内积 v_ri^T v_sj 求和。展开 J 后，所有单 site 自项与 φ 无关；只有跨 site 交叉项能受控。共同平移所有 φ_r 不改变目标，故最多有 R-1 个相对相位自由度。

## 相位何时严格有用

若比较独立 uniform dither，其目标为 J_ind=Σ_{r,k}||h_rk||²。令 C_rsk=⟨h_rk,h_sk⟩；相位目标非恒定，当且仅当至少一个 r≠s、k≠0 的 C_rsk≠0。因此可选相位严格优于独立 dither，当且仅当目标非恒定：对独立均匀抽取的相位表，交叉项期望为零，所以 J_ind 正是 J(Φ) 的相位平均值，非恒定连续目标必有低于平均值的点。对指定基线 Φ_0，严格改善的充要条件则是 J(Φ*)<J(Φ_0)；非恒定本身不保证给定的 Φ_0 不是最优点。

这不推出胜过 RTN/direct PTQ：单个标量 t∈Z 时 RTN 误差为零，而任意相位的 subtractive dither 方差仍是 Δ²/12。相位只重排激活噪声的跨 site 相关，不能消除其单 site 方差。

## 跨 observation 平均可抹去相位自由度

若校准/评估分布为 o∼D，固定相位表的目标交叉系数变为 C̄_rsk=E_o⟨h_rk(o),h_sk(o)⟩；目标非恒定当且仅当某个 C̄_rsk≠0。逐 observation 有用不代表有一张通用相位表有用：取两种等权 observation，使一对 site 的 cross-spectrum 分别为 C_k 与 -C_k，则每种 observation 都有非恒定目标、最优相位相差半周期，但平均目标中全部交叉项抵消，任何共享校准相位都无收益。此例可由投影坐标 f(s)-f(s+1/2) 构成（仅含奇次谐波），再让第二种 observation 将一 site 的归一化输入整体平移半周期，使其响应变号。

## W4 bias、A8 激活误差与强度边界

单个 Linear 对照浮点 Wx 时，令 E_W=Ŵ-W、e_A=x̂-x，则

$$Ŵx̂-Wx=\underbrace{E_Wx}_{权重误差/固定偏置}+\underbrace{We_A}_{激活误差}+\underbrace{E_We_A}_{交互项}=E_Wx+Ŵe_A.$$

融合的列和校正 -Δd Ŵ1 恰好实现 ŴΔ(q-d)；若列和、scale 或融合实现不一致，算法已不是上述 subtractive 重建。固定输入下 E e_A=0，故激活项与固定权重偏置的均方交叉为零；相位不能抵消该偏置，只能改变激活残差的投影交叉项。多层真实路径的非线性与扰动会让这个分解成为局部一阶近似。

候选 A8 用 Δ=m/126。若实际码域接受 [-128,127]，|t|≤126 且 d∈[0,1) 时 rounding 码在 [-126,127]，没有饱和；若 kernel 只接受对称码域 [-126,126]，正峰值可被 clip，零均值和 Fourier 结论即失效。未 clip 时每坐标 |e_A|≤m/252，固定输入方差为 Δ²/12；同一 d 广播会让向量坐标误差相关，不能把坐标方差当作独立项相加。W4 的 [-7,7] groupwise weight error 不受相位控制；若 group step 为 s_g 且无 clipping，单权重量化误差至多 s_g/2，clip 可更大。A8 的细误差不因位宽差自动变得有意义：总 MSE 中的收益至多来自可控的 activation 部分；若固定 W4 偏置占主导，整体相对收益上限很小。位宽或 toy 几何本身不能确定真实 action Jacobian 下的收益。

## 对候选完整方案的判定与最小真实模型 gate

本推导不构成“相位必无效”的结构性反证；固定输入投影下，跨 site 非零 cross-spectrum 足以产生优于 independent dither 的相位表。但本候选中全路径共享同一个 U：下游 x_r(U,Φ_<r) 与动态 Δ_r(U,Φ_<r) 已依赖该 draw，条件独立前提失效；故不能用上式保证全路径无偏、边际不变或实际 endpoint 改善。自适应行 scale 单独并不破坏性质，依赖当前/共享 draw 才破坏。真实非线性 Jacobian、A8 实际码域、W4 clipping 也必须按实现纳入实测。

唯一有判别力的最小 pilot：在冻结官方 WAM 的完整 W4A8 denoiser 路径（所有指定 Linear 都启用相位，不缩成单层），用真实 observation/goal logs 按 trajectory 留出；训练轨迹校准 Φ，held-out 上配对 sampler randomness 并按各 arm 的正确相关结构抽 dither。主指标为完整 action-chunk 对 BF16 reference 的 MSE；与相同 W4/A8 配置的 RTN、independent dither，以及校准 forward/search 预算相同的 direct endpoint PTQ 比较，并加 learned-Φ 的 site-label permutation 检查。只有 held-out 配对改善的置信区间排除零、并且 site permutation 消去该改善，才支持该机制；若未胜过最强匹配基线，或仅改善局部/STE oracle 指标，recipe-specific NO-GO。该 pilot 不证明 latency 或 closed-loop 收益。

## 实际首轮 pilot 的解释边界

### 这份模型源码给出的具体候选入口

FastWAM 的 action scheduler 由固定的 `linspace` 与 shift 构造 timesteps，不以 observation 或 action latent 为输入（`D:\Downloads\Final Year Project\experiment\reproduction\fastwam-smoke\FastWAM\src\fastwam\models\wan22\schedulers\scheduler_continuous.py:63-79`）。ActionDiT 把这些 timesteps 的 sinusoidal embedding 送入 time_embedding 的第一个 Linear（`D:\Downloads\Final Year Project\experiment\reproduction\fastwam-smoke\FastWAM\src\fastwam\models\wan22\action_dit.py:288-289`）；video first-frame 的 token timesteps 被置0后同样进入 time_embedding（`D:\Downloads\Final Year Project\experiment\reproduction\fastwam-smoke\FastWAM\src\fastwam\models\wan22\wan_video_dit.py:679-684`）。

因此这些首个 time Linear 的输入及动态 row scale 在不同 observations 上是固定的，而且不依赖 dither draw。这里局部 covariance 中的归一化输入相位因子不会因 observation 改变而自动均匀化。这是模型结构提供的一个具体、可检验入口，比“神经网络误差可能相关”更窄；仍要确认平均 action 投影内积是否非零且允许抵消。后续 time Linear、blocks 与 scheduler 的真实 action 投影依然依赖量化路径，不能据上述结构推出全路径收益或 novelty。

冻结协议 `PREREG.zh.md` 保留全部实际 Linear 的 W4A8 干预，但令 φ_r=θ_g(r)，把相位绑定到 video/action condition 与层段、proprio 等 groups。固定输入的一阶目标因此变为

$$J(θ)=\sum_{k\ne0}\left\|\sum_g H_{gk}e^{i2πkθ_g}\right\|^2,\qquad H_{gk}=\sum_{r:g(r)=g}h_{rk}.$$

同 group 内的相对相位已固定，只有 group 间的平均 cross-spectrum 可调；逐 site 非零交叉系数不保证 group 聚合后仍非零。故首轮失败只支持停止该粗粒度、该预算 recipe，不能否定不受限逐 site phase 搜索。首轮成功也只支持继续独立复验，并不证明 full-path 固定边际或无偏定理。

实际 TEST 使用冻结的4个新 dither seeds及2个 action sampler seeds，并在 observations 间复用它们。trajectory bootstrap 的区间是给定这组 draws 的条件性不确定性；它不包含重新抽取整组 dither/sampler seeds 的方差。不能把8条轨迹的区间当作已充分覆盖 E_{o,u,ζ} 的总体保证，也不能把每条轨迹内的 actions/draws 当额外独立样本。

site permutation 能检查所学 phase 与 site 对应关系是否重要，却不能单独分辨 full-path cross-term 调节、draw-dependent bias 和 BF16 重建舍入等来源。当前 proxy 先在 FP32 重建 activation，再 cast BF16 做浮点 GEMM；没有验证实际 packed INT4×INT8 GEMM 及其 FP32 fused column-sum epilogue。以上边界不改变冻结门槛，也不把工程 proxy 解释成 native deployment 结果。
