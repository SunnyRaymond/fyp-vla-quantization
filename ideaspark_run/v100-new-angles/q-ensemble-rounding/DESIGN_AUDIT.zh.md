# Q-ensemble stratified SR：最终数学与设计审查

日期：2026-09-13。审查对象为 `SUPPLEMENT_STRATIFIED.zh.md` 与 `INDEPENDENT_PRIOR_AUDIT.zh.md`，未运行 model、env 或数值，也未连接集群。此前 `resource_blocked` asset 状态不因本审查改变。

## 判定

结论：**conditional_go；可冻结为一次窄的 mechanism screen**。不需要额外采集 pair schedule。主问题是固定五个已训练 Q members、相同 FP H3 terminal latent 与 `pi` action 输入时，stratified stochastic rounding 是否改变官方 random-two `avg` 的期望 scoring；它不是五-member 全均值、variance theorem 或 uncertainty claim。

## Primary 的精确定义

每个 state、64 candidate actions、3 个 SR seeds 都复用同一份 FP 产生的 terminal latent 与 `pi` action；所有 arm 只改变 Q weights/rounding law。`Qall` 必须先对每个 member 的 two-hot logits 调用官方 `two_hot_inv`，得到 scalar 向量 `q=(q_1,…,q_5)`；不能平均 raw logits 后再解码。

对 unordered pair `p=(i,j)` 定义

`a_p^arm=(q_i^arm+q_j^arm)/2`，`e_p^arm=a_p^arm-a_p^FP`。

primary 是同一 candidate/seed 上全部 10 个 pair 的

`A_arm = mean_{i<j}[(e_p^arm)^2]`，

再在每个 state 内对 64 candidates 与 3 seeds 等权平均。官方 `return_type='avg'` 均匀选择 `torch.randperm(num_q)[:2]`；因此全部 10 pair 的平均是该 random-subset score 的精确条件期望，不是一次 realized pair trajectory。只需一次保存 RNG state 的 matched no-op：调用官方 `Q(avg)`，捕获其两个 indices，并与同一次 FP `Qall -> two_hot_inv -> pair mean` 比较。此 gate 通过后，primary 不需要另采 schedule，也不会把 pair sampling noise 混入 arm 比较。

matched no-op 必须比较“解码后的两个 scalar 的平均”；若 wrapper 先平均 two-hot logits，因 `two_hot_inv` 非线性会改变目标，直接阻断。若无法从官方调用得到 indices，至少用相同 device/RNG state 重放一次 `randperm` 并保存 indices；不能只比较一个不知对应 pair 的 scalar。

## M/C/A 分解是否有意义

对固定 candidate/seed/condition 写 `e_j=q_j^arm-q_j^FP`，定义

`M=mean_j(e_j^2)`，

`C=mean_{i<j}(e_i e_j)`，

则全部 10 pair 的代数恒等式为

`A=mean_{i<j}[((e_i+e_j)/2)^2]=(M+C)/2`。

因此 `A` 必须由 pair error squares 直接计算为 primary；`M` 只作 per-member marginal fairness guard，绝不能把五个 Q 整体均值的 MSE 当作 `A`。`C` 是 cross-member error product 的 attribution，不是另一个独立验证。若使用

`ΔA=A_ind-A_strat`、`ΔC=(C_ind-C_strat)/2`、`ΔM=(M_ind-M_strat)/2`，应报告 `ΔA=ΔM+ΔC` 的残差（只允许 FP64 数值误差）。当 `ΔA>0` 时，`ΔC/ΔA≥0.75` 才能说该 bounded gain 主要落在 joint error term；这是机制解释 guard，不是由恒等式本身产生的科学证据。

## Gate 的可反驳性

推荐先在每个 state 对 3×64 个 observations 聚合，再计算 ratio，不能先把八个 reset state 池化。`M_ind`、`A_ind` 与 `A_RTN` 任一不大于 `1e-8` 时，该 state 标 `degenerate/inconclusive`；不能改用 floor，也不能让它进入 median 或正负计数。

在非退化 state 上，`M_strat/M_ind∈[0.9,1.1]` 是 same-marginal fairness 的有限样本 guard。它不是 SR marginal law 的数学证明；若失败，应标 `marginal-mismatch-inconclusive`，不能直接 `mechanism_no_go`。随后定义 `g_s=1-A_strat/A_ind`。正向 joint state 必须同时满足 `g_s≥0.25`、`M` guard 和 `ΔC/ΔA≥0.75`；至少 6/8 个 state binding，并且 8 个有效 state 的 median gain `≥0.25`。此外要求 median `A_strat≤A_RTN`，且至少 6/8 state `A_strat≤A_RTN`；均值或 median 只能选一个并在运行前冻结。

这些条件实质上阻止三种伪 positive：只改善某一 member 的 `M`、只在少数 state 幸运、或把 RTN/分母退化误写成 gain。0.25、[0.9,1.1]、0.75 和 6/8 都是 bounded-pilot heuristic，不是显著性、功效或通用阈值，接受前提是报告每个 state 的 `M/C/A`、分母与 attribution。

## 正负与 inconclusive

- **`stratified_joint_preliminary_go`**：8 个 state 均非退化，至少 6 个满足上述 joint binding，median gain≥25%，RTN 的 global/6-of-8 guard 通过，FP no-op、同一 Q input、weight restore 和 SR marginal audit 全通过。
- **`mechanism_no_go`**：工程与 no-op/input controls 均通过、分母非退化且 marginal fairness 可解释，但 median gain<25%、joint binding<6/8，或 gain 主要不能由 `ΔC` 解释。此时可以记录 member-specific benefit，但不能把它包装成 stratified joint-law benefit。
- **`inconclusive`**：任一 state 的 `A_ind≤1e-8` 或 `A_RTN≤1e-8`、`M` fairness 大面积失败、FP terminal/Q input 不一致、无法证明 official pair/decode 对齐、no-op/RNG gate 失败，或 asset/source/restore 不完整。尤其 RTN 为零不能推出 stratified 胜出或失败。

若仅有一个或少数 state 的 sensitivity/degeneracy 问题，不应把该问题计入科学 no-go；但在“8 个有效 state 的 median”合同下不能悄悄丢弃它们。没有必要扩大 state、seed、candidate 或 full rollout 来挽救结果。

## 最小冻结建议

该设计不是 tautology：`A=(M+C)/2` 是 bookkeeping identity，但 arms 共享 Q inputs、保持 per-member SR marginal，并比较真实 official two-Q `avg` 的 all-pair expectation；若 M 基本不变而 C 系统改变并解释 gain，才有 bounded joint-law mechanism evidence。反之，恒等式残差通过而 `A` 无收益只能支持 mechanism no-go。

冻结时只需补清三件事：`M/A/RTN` 的 state-level denominator 与 median/mean 规则；`MSE` 对 candidate、seed、scalar member 的确切等权顺序；以及 matched RNG no-op 如何保存官方 pair indices。完成后可以不额外采 pair schedule，也无需 full validation；结论仍限于该 five-member TD-MPC2 checkpoint、8 个 reset states、64 个 H3 actions 与官方 live-Q `avg` interface。

## PROTOCOL 草案复核补充

已复核 root 新写的 `PROTOCOL.zh.md`。草案已把 arms 明确为 FP32、RTN、independent-SR、stratified-SR 四臂，并固定一次 FP H3 terminal latent/`pi` action、64 条 candidate actions、三次 rounding seed；这正是隔离 Q-rounding joint law 所需的最小输入控制。官方 `Q(return_type='all')` 返回 logits，而 `avg` 先对随机 two-member logits 做 `two_hot_inv` 再取两个 scalar 的平均；因此 `Qall→two_hot_inv→scalar5` 后穷举 10 个 unordered pairs，确实给出 uniform random-two score 的 exact finite expectation。一次 matched RNG no-op 足以验证 API reduction；primary 不必另采 pair schedule。

但提交前仍应完成下列 gate 澄清，否则 verifier 可能把统计缺陷误写成 no-go：

1. 官方 `randperm` 使用 `out.device` 的 RNG（GPU live-Q 时是 CUDA generator），matched no-op 必须保存/恢复同一 device 的 RNG state，并记录实际 pair indices；仅设 CPU seed 不能保证对应官方 pair。该 no-op 失败是 `implementation_inconclusive`，不是 mechanism no-go。
2. 当前草案把 gain median 写成 `binding set`，而本审查原先要求的是完整 8-state median。为防止挑掉困难 state，建议要求 8 个 state 的 `A_ind`、`M_ind` 与 `A_RTN` 都大于 `1e-8`，再计算固定的 median-8 gain；任何一个退化都整体 `inconclusive`，不从 median 隐藏。若 root 有意只在 binding subset 取 median，必须明确 subset、分母和不再声称 median-8。
3. cross-term attribution 必须与 gain/fairness 使用**同一 joint state set**。当前“至少 6 个改善 state”可能与至少 6 个 binding state 不是同六个；应要求至少 6 个 state 同时满足 `Mratio∈[0.9,1.1]`、gain≥25%、signed `ΔC/ΔA≥0.75`，并另报 `ΔA=ΔM+ΔC` 残差。恒等式本身只是 bookkeeping，不是科学证据。
4. `RTNzero` 必须显式定义为 `A_RTN≤1e-8`（或另一个运行前冻结的 floor）并使 state/overall 为 `inconclusive`；只有 RTN reference 非退化且 primary joint gate 已通过时，RTN 比较失败才能命名 `mechanism-positive/practical_no_go`。同理，binding 不足、输入/restore/no-op 失败均不能进入 scientific no-go。

上述是最小语义修正，不要求扩大 state、seed 或 pair sampling。修正后该 protocol 的正向结论仍只能是：在这一 checkpoint、8 个 reset states、64 条 H3 candidate actions 上，official live-Q `avg` 的期望误差可能因 stratified joint rounding 降低；不能写成通用 variance-reduction 或 uncertainty 定理。
