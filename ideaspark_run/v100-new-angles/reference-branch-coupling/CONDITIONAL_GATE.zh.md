# Reference-branch coupling：独立 conditional gate

范围：只读冻结 `PROTOCOL.zh.md`、teacher-bias 输入合同和 pinned DINO-WM 小源码；未加载模型、读取数组、连接 cluster 或提交作业。

## 判定

**`conditional_go / pre-data`，无科学识别阻断；尚不可执行，GPU=0。** 该 protocol 的 estimand 是“同一冻结 stochastic encoder weight map 用于 current/goal 两次调用时，downstream residual 是否比不同 map 更接近 FP residual”。它与旧 goal-coordinate 四臂 interaction 的 estimand 不同，也不是为旧经验 no-go 补跑。结果只能是固定 checkpoint、六个复用 locus 的 `scope_limited` 诊断。

## 指标是否可识别

对同一 state 定义 `r_de=y_d−g_e`、`r_F=y_F−g_F`。`A_de=mean((r_de−r_F)^2)` 是同坐标下的 propagated residual deviation；`A_S` 为三条 diagonal 的均值，`A_D` 为六条 off-diagonal 的均值。完整 3×3 pairing 使两种 arm 的 current 与 goal map 边际各自完全相同：每个 map 在 Shared 出现一次、在 Different 出现两次。因此单侧 `mean(Δy_d²)` 与 `mean(Δg_e²)` 的加权贡献被平衡，`A_D−A_S` 可作为 cross-map error-correlation 的 bounded contrast，不需要额外要求三张 map 的质量相同。若再加“每张 map 输出误差必须相近”的 gate，反而会引入 protocol 未定义的 nuisance 条件，应保持为 reporting receipt。

`B_de=(mean(r_de²)−mean(r_F²))²` 正确表示 scalar visual objective 相对 FP 的 squared deviation。它不是 `A` 的代数同义量，3×3 边际平衡也不能让平方后的 scalar pairing 完全相消；因此把 B 保持为 secondary，并要求 A、B 同时达到 10% 才记 joint positive，是合理的防止“vector residual 变好但 scalar objective 变坏”的 bounded guard。B 不能证明 FP 是 truth、planner ranking 或 task benefit；应保存每个 cell 的未平方 scalar objective，供 CPU verifier 复算，而不是只保存 B 汇总。

## 3×3、数据与量化合同

`q0,q1,q2` 的三张 frozen W4 map 逐权重使用同一 SR marginal、各自只抽一次随机数；Shared 使用 `(0,0),(1,1),(2,2)`，Different 使用其余六个 ordered pairs。这样比较的是 map coupling，且没有 activation/cache 共享。Off-diagonal 只是三 draw 的有限参考，不能改称完整独立 SR 分布。每个 cell 必须绑定同一 current/goal pair、action、FP predictor、preprocessing 和 pristine restore，并保存 map id、两侧 encoder delta、`A_de` 与未平方 scalar objective。

六条 `124..129` 是 teacher-bias 已登记的复用输入，不是 fresh independent confirmation；这对最小 mechanism screen 可接受，但结果不得推广到 Wall 数据集、长时 rollout 或闭环控制。H1 语义必须在实现中写死为 raw frame `0→5` 的一个 model transition：从实际 `model_actions_h5` 取首行，helper暴露为 `sample['action_blocks'][:1]`，形状 `[1,10]`（五个 primitive actions concat），再加batch维送入。没有名为 `model_actions_h1` 的现成input key。调用一次 official `encode(current, action)`/`predict`；不能误送全部 `[5,10]` 而变成H5。goal单独调用 `encode_obs`，不可复用current activation/cache。

## 48 个 encoder Linear 的执行边界

DINOv2 12 个 Transformer blocks 通常对应每 block 的 `qkv`、attention `proj`、MLP `fc1/fc2` 四个 Linear，48 这个数量与结构相容，但 count 本身不是身份凭证。producer/verifier 仍须记录完整相对 module names、每个 weight shape 和 allowlist；确认 patch embedding、bias、LayerNorm、proprio/action encoder、predictor 均未变。每张 map 和 RTN/FP arm 从同一 pristine snapshot 写入，fresh state readback 后 restore；不能仅用 state-dict key 猜测实际 live module。

## 最小执行前修订

没有需要改变科学假设的修订。实现合同应补足三点：

1. 明确 predictor input 的实际 shape/axes（`VWorldModel.predict`接口为 `[1,1,196,404]`，内部predictor pre-hook实际为 `[1,196,404]`，visual output slice `[196,384]`），并用 pre-hook/readback 证明 action/proprio channels 保持 FP；
2. 保存每个 state 的 `A_de[3,3]`、scalar objective `S_de[3,3]`、`A_S/A_D/B_S/B_D`，而非仅保存最终 gain；
3. 从输入 manifest 读取并逐项比较 source、checkpoint、approved adapter、helper 与 protocol 的 literal identity，不能只记录路径或固定标签。

这些是可执行性收口项，未发现会改变结论的 blocking bug。按冻结 gate，`A_D` 或 `B_D` 不足阈值的 state 为 nonbinding；至少 4/6 binding 后仍须至少 4/6 joint positive 才能 `scope_limited_preliminary_go`，否则分别记 `inconclusive_binding` 或 `mechanism_no_go`。无论结果如何 STOP，不扩 sample、Horizon、module 或 task validation。
