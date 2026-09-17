# Frozen-Residual Transport PTQ（non-VQ 候选）

**2026-09-12 实验后更新：本页保留原 FRT 方案；其 CCDS Stage B 已判 mechanism_no_go。后续候选改为 [Paired-Rollout Recovery](<D:/Downloads/Final Year Project/idea/closed-loop-quantization/revisions/paired-rollout-recovery/IDEA.zh.md>)，不继承旧结果支持。

**状态：conditional_candidate / 未独立认证 novelty。** 这是对 STRC 初稿的修订候选。STRC 已归档为 `rejected_attempt`，原因是 signed accumulation 在特定定义下会退化为 terminal-state drift，且原 phase-shuffle action injection 不能保持同一 policy。当前卡只提出一个更窄、可执行的 finite-difference transport 问题：固定一个已经部署的 W4 quantizer 造成的 residual direction，量化后的 world-model one-step map 是否能在这个真实方向上保持 FP 的传播响应。

## 假设与目标

首个 anchor 是已有 DINO-WM Wall epoch-65 artifact；这里只把它作为 numerical PTQ screen，WAM/VLA transfer 不是本轮证据。令 `X_t` 是 DINO-WM 的**完整 history state**（所有 context slots、cache/history fields 和当前预测槽），`a_t` 是固定 action/chunk。要求 runner 提供可重放的纯函数

\[
 F_\theta(X_t,a_t)\;:\; (\text{complete history},\text{action})\mapsto z_{t+1},
\]

并能在独立的 environment rollout 端点读取 Wall 的 physical `xy/goal-relative` 坐标。FRT loss 在固定 latent units 中计算；physical `xy` 只作为 held-out environment endpoint，不假设存在 latent 到 `xy` 的线性 readout。若只能得到不稳定的 latent 或不能恢复完整 history，本候选停止；不能把 latent diagnostic 写成 physical 结果。

核心假设是：在相同 W4 logical bytes、相同 calibration/evaluation budget 下，匹配 quantizer 自己产生的 residual 方向，比普通 clean output MSE、随机同范数扰动、QDrop/input-noise 或 direct two-step matching 更能预测 held-out closed-loop physical `xy` drift。这里不假设 quantization noise 很小，也不把 finite difference 宣称为严格 linearization、Jacobian 或 causal transfer。

## 精确定义与数据流

先固定 reference `FP32` 和 baseline `Q0`：`Q0` 是 symmetric per-output-channel weight-only W4 RTN，预测器目标 blocks 用同一注册清单；encoder、policy/readout 和 environment 在首个 pilot 保持 FP32。对每个 CAL 的同一 history/action/RNG 记录，计算

\[
 z_i^{FP}=F_{FP}(X_i,a_i),\qquad
 z_i^{Q0}=F_{Q0}(X_i,a_i),\qquad
 \delta_i=\operatorname{stopgrad}(z_i^{Q0}-z_i^{FP}).
\]

`delta_i` 不是重新优化出来的 noise。它被写入**预测新 latent 槽**，旧 history 不改变：

\[
 X_i^+=\operatorname{concat}(X_i[2:L],z_i^{FP}),\qquad
 X_i^+ + \delta_i^{slot}=\operatorname{concat}(X_i[2:L],z_i^{Q0}).
\]

这里 `shift/concat` 必须在接口中实际实现，而不是把向量 broadcast 到所有 history slots。源码 `experiment/reproduction/dino-wm-wall/source/models/visual_world_model.py` 的 `rollout` 每次调用 `predict(z[:, -num_hist:])`，取最后一个预测槽，再用 `replace_actions_from_z` 写入已知下一 action，最后 append。实验 wrapper 必须复用这个顺序：FP/Q0 两个新槽写入完全相同的已知 action embedding 后才取 residual，`delta` 只保留 observation/proprio 坐标，action 坐标严格为零；两条分支都保留旧 history。该路径未显示独立 KV cache，不凭空添加 cache 状态；以实际 source/config 的 shape 和 normalization 为准。对下一步固定 action `a_i'`，对任意待校准量化器 `Q_\theta` 定义

\[
 T_\theta(X_i^+,\delta_i,a_i')
 =F_\theta(X_i^++\delta_i^{slot},a_i')-F_\theta(X_i^+,a_i'),
\]

\[
 T_{FP}(X_i^+,\delta_i,a_i')
 =F_{FP}(X_i^++\delta_i^{slot},a_i')-F_{FP}(X_i^+,a_i').
\]

这是固定 `x` 与固定 `delta` 上的 finite-difference transport contrast。它测量 quantized map 如何搬运一个由 `Q0` 实际产生的方向；它没有跨 lag 累加，也没有 `Σ\|K_\ell\|^2` 或 cancellation 主张。loss 在固定 latent scale `W_z` 中计算，physical `xy` 只在独立 environment closed-loop endpoint 报告：

\[
 \mathcal L_{FRT}(\theta)=\frac1{|C|}\sum_{i\in C}
 \left[\|W_z(F_\theta(X_i,a_i)-F_{FP}(X_i,a_i))\|_2^2
 +\lambda_T\|W_z(T_\theta-T_{FP})\|_2^2\right].
\]

`W_z` 由 CAL 的 FP latent variance 加冻结 floor 得到；`λ_T`、`W_z` 和 target blocks 在打开 TEST 前锁定。不得把 `W_z` 解释成物理坐标变换，也不得用它推出 native deployment 指标。

量化参数是全体 target blocks 的联合向量 `θ={(s_g,α_g):g∈G}`，不是一串独立 site score。采用标准 AdaRound 形式；`h(α_g)∈[0,1]` 只在 CAL 搜索，最终硬化为 `{0,1}` 并丢弃 `α_g`：

\[
 q_{int,g}=\operatorname{clip}\left(\lfloor w_g/s_g\rfloor+h(\alpha_g),q_{min},q_{max}\right),
 \qquad \widehat w_g=s_g q_{int,g},\quad s_g>0,
 \quad h(\alpha_g)\in[0,1]\xrightarrow{\mathrm{freeze}}\{0,1\}.
\]

首版固定 signed symmetric `q_min=-7, q_max=7`，每个 output channel 一个正 scale；Q0 用该 channel 的 `max(abs(w))/7` 初始化，全零 channel 以 scale=1、全零整数表示。Q0 RTN 固定为 ties-to-even。拟合时用同一整数网格，正 scale 用有下界参数化；hardening 固定 `h>=0.5` 取 1，先 floor 加 hard h、再 clamp、最后转整数。hard rounding 不要求与 RTN 的 tie rule 相同，但必须固定并记录。真实序列化账本包含 scales、padding、metadata 和未量化参数。

只在 CAL 用 AdaRound-style soft rounding/STE 搜索，随后 materialize 为 exact hard W4 rounding，再从硬化权重重新运行 DEV/TEST。所有 block 仍是 W4，logical byte budget 不变；不做 W4/W8 mixed-precision allocation、不加 observer/error-feedback accumulator，也不带 FP teacher 到 deployment。拟合目标对所有 target blocks 联合定义；不把每个 site 的独立分数相加。

## 最小验证路线

1. **Interface gate：** 只做轻量源码/配置核查，确认 full `X` 序列化、真实 shift/concat、同 history/action/RNG replay、`F_FP/F_Q0` 和独立 environment 的 physical `xy` endpoint。缺 full history 或 one-step map 就停；latent→xy 的线性 readout 不是要求。
2. **CAL/DEV/TEST：** 以 episode/initial-state 为独立单位，最小 pilot 为 `6/6/12` episodes，3 个 quantization fit seeds；frames、candidate perturbations 和 block pairs 都不算独立 `n`。在单一 Wall environment 中只要求 split 间 episode/initial-state disjoint，不宣称 task-disjoint；每个 FP/Q pair 只在 pair 内使用 common random numbers。
3. **冻结 residual：** CAL 先收集 `Q0` 的真实 paired residual。DEV 同时另收 fresh `Q0` residual，并在拟合后让一次 `Q_θ` rollout 生成 fresh `Q_θ` residual；两者都 stop-gradient、只用于 transport/迁移检查，防止优化后的模型用 CAL residual 自证。
4. **匹配预算对照：** `FP32`、`Q0-RTN`、Local-MSE/AdaRound-style clean reconstruction、PD-Quant-style clean prediction-difference + regularization、direct two-step unrolled matching、QDrop/input-noise reconstruction、random same-norm `delta`，以及 FRT。random same-norm 是 GAD/Sobolev-like sensitivity control 的关键对照；若它与 Q0 direction 等效，FRT 失败。
5. **评估：** DEV 先看 clean loss、冻结 delta transport、fresh-Q0 与 fresh-Qθ transport、physical xy endpoint drift/success；选择 `λ_T`/搜索早停后只运行一次 TEST。TEST 报 paired success/progress、xy drift、action error；peak VRAM 与同步 latency 只有在真实 backend 可测时报告。

## 可证伪预测与 stop rules

主要预测是：FRT 的 `T_θ≈T_FP` 在 fresh-DEV Q0 residual 和一次 fresh-Qθ residual 检查上仍成立，并且比 local MSE、two-step、QDrop/input-noise 和 random same-norm 更能预测 TEST closed-loop physical xy drift。若 random same-norm 与 Q0 direction 的 transport 和 downstream 结果相同；FRT 只改善 CAL 而不改善 fresh DEV；或 fit seeds 方向不稳定，则立即判 no-go。`5%` 相对变化若用于首轮停机，只是资源决策阈值，不是统计显著性或文献结论。

本轮规划总上限为 **≤12 A100 GPU-hours**，包含 baselines、启动与失败成本，concurrency `1–4` 张 A100-40GB；这只是保守 smoke 规划，不承诺模型一定放得下，A100-80GB 不能作为超时后的自动 fallback。任何模型加载、重 I/O、环境准备或实验都必须在真实 PBS allocation 里核验 `PBS_JOBID`、非-login hostname 和 GPU allocation；本轮没有运行模型、实验、SSH 或 PBS。40GB/80GB 若以后都测，必须分开报告；本卡只规划 40GB smoke。logical/fake quant 数值结果不推出 native packed bytes、VRAM 或 latency。

## 近邻边界（有限检索，不是“未发现即不存在”）

- **AdaRound（arXiv:2004.10568）** 用 data/task loss 学 rounding，并化为 layer-local reconstruction；FRT 可复用 hard-rounding materialization，但 load-bearing term 是 frozen `Q0` residual 上的 `F(X+δ)-F(X)`。
- **QDrop（arXiv:2203.05740）** 在 PTQ reconstruction 中随机 drop activation quantization 以改善 flatness；它是同预算 input-noise baseline，不提供真实部署 residual direction。
- **PD-Quant（arXiv:2212.07048）** 用 global FP/Q prediction difference、regularization 和 distribution correction；正文也显示小 CAL 的 PD-only 会 overfit。FRT 不只看 final prediction difference，并用 fresh Q0 DEV residual 验证。
- **Sobolev Training（arXiv:1706.04859）** 与 **GAD（arXiv:2606.01651）** 已说明 derivative/JVP 或局部 sensitivity matching 的总体原则；FRT 不能声称首次 sensitivity alignment，只能检验固定真实 Q0 direction 是否比 isotropic/random direction 有额外信息。GAD 的公开训练代码也是重要 collateral baseline。
- **DA-PTQ（arXiv:2604.11572）** 已覆盖 virtual Jacobian、motion-driven trajectory surrogate 和 mixed precision；**QuantWAMs（arXiv:2607.28405v1）** 已覆盖 shared-basis/Fisher/reachable-state audit 与 WAM schedule repair；**Feedback World Model（arXiv:2605.15705v1）** 已覆盖 online transition-residual observer。FRT 不做这些 allocation/schedule/observer 操作，且 DINO-WM 没有 denoising schedule，因此不把 QuantWAMs schedule 当作 DINO 对照。
- `RPIQ`（OpenAlex `W7119233972`）和 `When Can Depth Replace Precision?`（OpenAlex `W7171748342`）提示 residual-projected closed-loop compensation、Gauss-Seidel quantization、increment error feedback 已有直接碰撞。FRT 不实现 residual compensation/error-feedback arithmetic；只作 frozen residual transport measurement。
- **SQIL、QVLA、BitVLA、QuantWM、QuaDreamer、已有 closed-loop QAT，以及 `Calibrate Where You Deploy`** 分别覆盖训练式 saliency/QAT、action/model quantization 或 kernel、低比特 WM/planning sensitivity、token/representation/generative quantization、训练/在线校准或作者自报 on-policy recalibration。当前 bounded search 没有验证与 FRT 完全相同的接口和 objective，但这不证明不存在；`Calibrate Where You Deploy` 的 negative result 也有公开 task/seed/control audit 限制，不能被写成普遍 impossibility。

旧 **RankCal** 与 **OTC-PTQ** 的单点/一步配方已 no-go；旧 CEM-Update 是另一个 handoff，FRT 不改名复用。FRT 的研究贡献范围严格收窄为：在一个可重放 WM one-step map 上，真实 self-induced residual direction 的传播对齐是否优于一般噪声增强和 two-step MSE；在完成独立审查和最小 pilot 前，不称为已证实的新颖方法。
