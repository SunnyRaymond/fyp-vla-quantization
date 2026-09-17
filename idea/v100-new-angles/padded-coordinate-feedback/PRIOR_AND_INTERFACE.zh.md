# PRIOR AND INTERFACE：SmolVLA padded action coordinates 的 PTQ feedback

审查日期：2026-09-13。source pin 固定为 LeRobot v0.4.4；本轮只做 source
与 primary prior 审查，不下载、不运行模型、不提交任务，也不修改既有
Flow/gradient 文件。

## Structural audit：候选路径确实存在

官方 [v0.4.4 configuration](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/smolvla/configuration_smolvla.py)
将 max_action_dim 设为 32。当前资产的 physical action dim=7 仍应由 checkpoint
metadata 现场确认；v0.4.4 代码不是把 7 写死，而是读取
config.action_feature.shape[0]。

在官方 [v0.4.4 modeling_smolvla.py](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/smolvla/modeling_smolvla.py)
中，pad_vector 将较短 action 右侧补零到 32 维。训练 forward 先调用
prepare_action，再以 32 维 actions 与 noise 构造 x_t；F.mse_loss 对 v_t 与
u_t 的完整 32 维张量计算 reduction=none。actions_id_pad 只通过
unsqueeze(-1) mask episode time positions，没有 physical-coordinate mask；
随后保留的范围也是 [:, :, :max_action_dim]，仍是 32 维。因此基于 source，
padding 坐标并未从 flow-matching loss 中排除。

推理时 sample_actions 生成 shape 为 chunk_size × max_action_dim 的 noise；
action_in_proj 的输入维是 32，action_out_proj 的输出维也是 32。Euler loop
每一步更新完整 x_t=x_t+dt·v_t。只有 _get_action_chunk 在 sampler 完成后才按
action_feature.shape[0] 截取物理输出；若当前 checkpoint 的 physical dim 确为
7，则最后 25 维不发送给 robot，但在此前的 action input、action expert、
action output 和递归状态中都存在。由于 pad 坐标的 x_t 由 full-dimensional
noise 初始化，它们也不是推理时始终为零的静态占位。

这确认了 structural path：未执行的 25 维可能通过 shared hidden/action
expert 改变后续 physical velocity。它还没有证明 effect 的数值大小，也没有
证明当前量化 recipe 实际量化了哪些 projection；这两项必须由后续 preflight
记录，不能从维度推断。

## Primary prior 与重叠边界

[VITA](https://arxiv.org/abs/2507.13231) 是最接近的 action-padding prior：
它指出在 flow-matching target 中简单 zero-padding 会产生 sparse、unstructured
action representation，并以 learned action autoencoder 替代。该结果针对训练
表示与 latent target，不是 SmolVLA 这种 full-vector Euler inference 中 padding
velocity 对 physical endpoint 的 PTQ feedback。

[QuantWAMs](https://arxiv.org/abs/2607.28405) 已做 step/state-conditioned
PTQ：在 FP16 reachable rollout snapshots 上固定 intervention，按 denoising
step 比较 low-bit 与 FP field，并选择 protected step indices。它覆盖“量化 field
在可达状态和 step 上的差异”，但没有对未执行 action coordinates 做同一 current
state 的 FP/Q slice intervention。QuantVLA 的
[paper](https://arxiv.org/abs/2602.20309) 与 [official code](https://github.com/AIoT-MLSys-Lab/QuantVLA)
也关注 action-head PTQ 和 denoising steps；Ω-QVLA 的
[paper](https://arxiv.org/abs/2605.28803) 使用 per-step activation scales。
本次审阅未见 primary source 把“physical output 只取 7 维、但 32 维 flow
state 递归”作为 PTQ feedback 机制来做同-state hybrid field test。

所以 novelty 边界是：padding 对 flow training 的风险已有 prior，step-aware
PTQ 也已有 prior；“unused padded velocity 通过递归 state 反馈到 executed
coordinates”是一个可检验的窄机制诊断，不是已认证的新 quantizer 或参数效率
方法。不能用 25 versus 7 推出 parameter efficiency。

## 可证伪的 oracle 诊断

令 v_F(x,t) 和 v_Q(x,t) 分别为同一 observation/prefix、同一 current full
state x 与同一 t 下的 FP32 和 W4 32 维 vector field。physical slice 为
坐标 0:7，padding slice 为 7:32。定义四个 oracle field：

    H_FF(x,t) = [v_F(x,t)[0:7],  v_F(x,t)[7:32]]
    H_QQ(x,t) = [v_Q(x,t)[0:7],  v_Q(x,t)[7:32]]
    H_FQ(x,t) = [v_F(x,t)[0:7],  v_Q(x,t)[7:32]]
    H_QF(x,t) = [v_Q(x,t)[0:7],  v_F(x,t)[7:32]]

每个 branch 从同一个 initial noise 开始；在每一个 branch 的同一个 current
state 上分别调用 v_F 和 v_Q，再拼接对应坐标并做下一次 Euler update。禁止把
不同 trajectory 的 velocity 直接拼接。H_QF 与 H_QQ 的 physical endpoint
差异是主要 intervention：两者都选 Q physical velocity，只把 25 维 padded
velocity 换成 FP；若之后 7 维 endpoint 发生稳定变化，说明 padded update
通过递归 full state 影响了 executed coordinates。H_FQ 与 H_FF 是对称的
FP-physical 对照。

这是 output-coordinate intervention oracle，不是可部署模型，也不能说明
effect 来自 action_in_proj、action_out_proj 还是 shared expert 的哪一层。
若要作 layer attribution，需要另行固定参数 slice；本 screen 不应追加该
扩展。

## Identifiability 与最小边界

最小 screen 可在已有 SmolVLA GPU allocation 中使用准备好的 12 个 raw samples、
固定 v0.4.4 默认 K=10、四个 H field 和同一个 initial noise，不做 rollout。每
个 sample 记录 physical endpoint delta、每步 physical/padded velocity slices、
first-step equality check，以及 ||v_Q[7:32]-v_F[7:32]||。H_QF 与 H_QQ 的
比较必须在第一步确认 physical slice 完全相同；后续 physical divergence 才
可归因于 padded-state feedback。所有 four fields 必须在各自 branch 的同一
current x_t 上同时求值。

一个方向性 falsification gate 是：若 pad velocity 已有非零 FP/Q discrepancy，
但 H_QF 与 H_QQ 的 7 维 endpoint difference 在全部配对 sample 上都不超过
预先声明的 numerical tolerance，则本候选机制在该 checkpoint/quantizer 下
停止。若出现稳定的 endpoint effect，只能报告“unused-coordinate feedback
diagnostic”；不能报告 task success、部署收益、参数节省，或把 oracle
intervention 当成 PTQ recipe。若无法证明 checkpoint physical dim=7、full
32 state recursion、同-state field calls 或 first-step slice equality，则
立即 structural/identifiability stop。

## 决定

**Structural status：go。** v0.4.4 source 明确存在 32 维 padding 的训练、输入、
输出与 Euler recursion 路径，故不是 structural no-go。

**Independent mechanism status：conditional narrow-go。** VITA 与 QuantWAMs
分别覆盖 padding representation 和 step/state-aware PTQ，但未覆盖上述
same-current-state coordinate intervention；该差异足以支持一个小诊断，尚不足以
支持新的 quantizer 方法。

**本轮 GPU status：不单独启动；已有 allocation 中通过 preflight 后可 piggyback。**
只做 12-sample、K=10、四 field 的静态 oracle screen；不扩大到新模型、更多
baseline、完整 rollout 或参数效率结论。

## METHOD_DERIVATION_AND_PRIOR：analytic padded path

### 推导与适用条件

该推导使用的是 SmolVLA v0.4.4 的实际时间方向，而不是直接套用论文中常见
的 time convention。对一个已经过 checkpoint processor、并在送入 flow model
前完成 padding 的 action，令 physical 维度为 d=7、padding 维度为 25，且

    a_pad = 0,   z_pad = initial_noise[7:32].

v0.4.4 的 source 定义 full vector bridge 为

    x_t = t z + (1-t) a,       u_t = z - a.

因此 padding slice 的 per-sample conditional path 是

    x_pad(t) = t z_pad,         u_pad = z_pad.

其导数 dx_pad/dt=z_pad 在整个 bridge 上为常数。采样从 t=1 的 noise 以
dt=-1/K 反向 Euler 到 t=0；若每一步使用同一个 z_pad，更新

    x_pad <- x_pad + dt z_pad

会把这条 sample-specific analytic path 精确地走过 K 个 Euler grid points。
x_pad/t 只在 t>0 时等价，实际实现/诊断应直接保存 initial z_pad，避免 t=0
除法。

这里的 exact 只针对该 training bridge 的 conditional target。有限模型学到的
marginal conditional field 在任意 current state 上不必等于这一个 sampled
z_pad，所以 analytic replacement 是 output-coordinate oracle intervention，
不是 padding 的 deployed ground-truth vector field，也不能据此声称 physical
action 更正确。正式 preflight 仍须逐样本断言：prepared/normalized action 的
7:32 slice 全为有限的 exact zero，checkpoint 的 original action dim 确为 7，
并且这 25 维在训练 loss、model input/output 和 Euler recursion 中均未被 mask
掉；任一断言失败即停止推导适用性。

### 四个同轨迹字段

固定每个 sample 的完整 initial noise、condition 和 K（默认 K=10）。在每一个
branch 的同一个 current full x_t 上，分别求 FP 与 Q 的完整 32 维 velocity，
再按 output slice 组成：

    FPnative   = [v_F,phys, v_F,pad]
    Qnative    = [v_Q,phys, v_Q,pad]
    FPanalytic = [v_F,phys, z_pad]
    Qanalytic  = [v_Q,phys, z_pad].

analytic branch 仍更新完整 x_t；它只替换 padded velocity，physical slice 仍
来自该 branch 当前状态下的对应模型。不能把不同 branch 或不同 current state
的 velocity 拼在一起。第一步 Qanalytic 与 Qnative 的 physical slice 必须逐
元素相同；其后 physical endpoint 的差异才可能支持“padded-coordinate feedback”
而不是一个初始 physical output 差异。

建议预先冻结两个 binding quantities（不在结果后调阈值）：

    B_F = mean ||A_FPanalytic,phys - A_FPnative,phys||^2,
    D_0 = mean ||A_Qnative,phys - A_FPnative,phys||^2,
    D_1 = mean ||A_Qanalytic,phys - A_FPanalytic,phys||^2.

只有在 B_F 不超过预先声明的 small numerical tolerance，且 D_1 相对 D_0
达到预先声明的 improvement margin，才可报告窄的 feedback diagnostic。若
FPanalytic 自身造成明显 physical shift，Q 的变化无法单独解释为“修正 quantized
padding feedback”；若 D_1 没有改善，则该机制在此 checkpoint/quantizer 下
停止。两种 drift 都应同时报告相对 FPnative 的数值，防止小的 reference shift
掩盖问题。该 gate 仍不构成 task success、部署收益或 physical ground truth
结论。

### Bounded prior 状态

[Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) 的
conditional flow matching 目标直接给出 conditional path 的 velocity regression；
上面的常数 z_pad 是 v0.4.4 线性 bridge 与 a_pad=0 的代数结果，不是新的
quantizer idea。[VITA](https://arxiv.org/abs/2507.13231) 已明确讨论将 action
zero-pad 到更高维会造成稀疏、无结构的 flow target，因而覆盖 padding 作为
representation/training risk 的先验。[QuantWAMs](https://arxiv.org/abs/2607.28405)
已用 reachable state snapshot、固定 intervention 和 denoising-step sensitivity
来选择保护步骤，覆盖了 state/step-specific PTQ auditing 的方法学邻域。

在本次 bounded primary review 中，未见把 SmolVLA 的 executed/padded output
slices 在同一 current state 下替换为 conditional analytic path、并观察其对
physical endpoint 反馈的直接诊断。故差异足以保留一个很窄的 identifiability
screen，但 analytic path 本身不具备 novelty，也不是可部署 recipe。结论维持为
**conditional narrow-go diagnostic、standalone GPU no-go**：仅在既有 12-sample
allocation 中通过上述 preflight 后 piggyback；若 binding gate 或同-state
证据不成立，立即停止，不扩展为训练、layer attribution 或完整 rollout。
