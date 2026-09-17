# Conditional action distribution：最小统计设计审查

日期：2026-09-13。审查对象是本目录的 `PRIOR_GATE.zh.md` 与 campaign 最新 `INDEX.zh.md`。`INDEX` 尚未登记本候选的结果或数据区间；以下是执行前的 bounded screen 合同，不是实验结果。未连接集群、未运行本地数值计算。

## 判定

结论：**conditional narrow-go / resource-and-statistics gated**，不是 `prior_no_go`。待检验的具体假设是：在固定 observation 下，PTQ 可能显著改变 noise-to-action mapping，但其条件边际分布 `P(action | observation)` 仍落在 FP 的有限样本参考带内；paired same-noise MSE 与 independent-noise distribution distance 因而可能给出不同结论。这个问题比换一个通用 metric 更具体，但不能声称发明 SWD、MMD 或 distribution-preserving quantization。

## 样本预算与随机化

当前候选只有 **FP 与 Q 两个 model arm**；A/B 是 noise block，不是 model arm。推荐把“每臂每条件 64 个 noise”定义为 64 个互不重复的 Gaussian draws，预先分成两个不重叠 block A/B，各 32 个。这样四个 model arm 的旧表述应删除；64 draws × 2 arms × 8 conditions = **1024 个 action samples**。这样一次采样同时支持：

- paired mapping 对照：`FP_A↔Q_A`、`FP_B↔Q_B`，只报告 same-noise endpoint MSE；
- FP finite-sample null：`FP_A↔FP_B`；
- Q spread/collapse：`Q_A↔Q_B`；
- candidate marginal distance：交叉的 `Q_A↔FP_B` 与 `Q_B↔FP_A`，取两者的预注册平均。

每次比较样本数相等，A/B seed namespace 不重叠，noise shape、dtype 和 observation hash 固定。交叉比较不能复用同一 noise，否则会把 coupling 误当成边际距离。若“32/64”是每臂总数，则 32 只能是方向性 pilot（A/B 各 16）；它可以揭示很大的 shift，但不能支持 marginal-preserving/dissociation 的稳定正向结论。不应因 15 分钟预算自动把 64 降为 32；若实际不足预注册数，记为 `resource_blocked` 或 `statistical_inconclusive`。

八个 condition 是分布比较的统计单位；noise 只是每个 condition 内的 Monte Carlo replicate。必须先逐 condition 计算距离，再按预注册计数或 median 汇总，不能把八个 condition 的样本池化为一个分布。若一个 episode 含多帧，应固定一帧或保留 episode clustering，不能把帧数冒充 condition 数。

## 固定低维指标

主 gate 的唯一输入应冻结为 runner 定义的 **normalized-model physical first-action 7-D slice**：所有 arm 经过同一份 checkpoint preprocessing 路径后，取 action tensor 中 physical first chunk 的 7 个坐标，随后不再追加 `postprocess`、physical-unit unnormalize 或第二次 canonical normalization。协议必须记录该 tensor 所在的明确阶段、shape、dtype 与 hash；不能正文一处写 postprocessor 输出、另一处又把它当 canonical normalized vector。gripper 1-D 可省略；若保留，只作预注册诊断，不进入主 gate。first-8-chunk 也只作描述性检查。

主指标是 fixed-projection sliced Wasserstein-1：7-D 切片使用预先生成并 hash 的 16 个方向（7 个 coordinate axes 加 9 个固定 seed 的 normalized directions），同一方向矩阵用于所有 arm、condition 和 null。额外计算 7-D energy distance 作为敏感性检查，不能看完 Q 输出后调 projection、kernel 或 bandwidth。主判断只使用 SWD；若 energy 与 SWD 方向相反，标记 `metric_discordant`，不强行选一个结论。gripper 只报告 1-D Wasserstein，不替代 7-D 主判断。

对每个 condition 定义 `W0=W_FP-FP` 与 `E0=E_FP-FP`（分别为 SWD 与 energy 的 FP-FP finite-sample reference；距离都在上面冻结的 normalized-model 7-D 坐标中计算）。只有 `W0>1e-4`、`E0>1e-8` 才计算 ratio；否则标 `reference_degenerate`，不以 floor clamping 制造稳定性。FP-FP 只是有限样本参考带，不是显著性检验。

- `marginal-preserving-screen`：`R_W ≤ 1.25` 且 `R_E ≤ 1.25`；同时 collapse control 通过。这只是“落入预设有限样本带”，绝不等价于分布相同。
- `marginal-shift-screen`：`R_W ≥ 1.50` 且 `R_E ≥ 1.25`；若同时出现 spread collapse，标签写成 `mode-collapse/shift`，不要称普通 shift。
- 其余区间、FP-FP reference 退化或 SWD/energy 不一致，均为 `inconclusive`。`1.25–1.50` 的灰区是有意保留的，不用“未显著不同”替代它。

## 对新增 mapping gate 与 positive control 的独立反驳

新增 `Dpair=mean ||Q(z_i)-FP(z_i)||²`（64 个 paired draws）以及 FP-only 的 `Dnoise=mean ||FP_A-FP_B||²`，实质上改善了可辨识性：正向 dissociation 不能只靠“Q-F distance 接近 null”，还必须先有相对于本 condition intrinsic noise 的明显 map drift。`Dnoise>1e-6` 与 `Dpair/Dnoise≥0.25` 可以作为 **bounded pilot 的 large-map-drift gate**；它是固定的 effect-size heuristic，不是 universal threshold、power 或显著性结论。两者应逐 condition 计算并报告绝对值。

这里有一个必须修正的实现细节：A/B 各 32 个 noise 时，`FP_Ai-FP_Bi` 只有 32 个 index pairs，不能同时写成“Dnoise over 64 independent pairs”。可以明确使用 32 对并记录 `n=32`，或使用预注册的两个 cross-pairing；更稳妥的是 FP-only 的全部 32×32 cross-pair mean。无论选择哪一种，Dnoise 只是尺度估计，不得伪装成独立检验。

Root 拟议的 translation control 有科学价值：将 FP-B 沿固定 normalized physical axis 1 平移，使每维 MSE 与目标 map-drift 同量级，然后用 FP-A 检查 SWD/energy 是否能发现已知 shift；`SWD ratio≥1.5` 且 `energy ratio≥1.25` 未通过时，Q 的“接近 null”应降级为 `sensitivity_inconclusive`。但 `sqrt(7*Dpair)` 依赖 Q 输出，属于 post-hoc matched-effect calibration，不是独立 positive control，存在循环性。若要让该 control 真正具有 veto 作用，应改为只由 FP-only 量确定的预注册幅度，例如 `delta_ref=sqrt(7*0.25*Dnoise)·e1`，或使用运行前冻结的固定幅度；若坚持用 `Dpair`，只能作为事后敏感性说明，不能批准 preserve/dissociation。

translation 必须在同一未裁剪 normalized coordinate 中生成，并记录是否触及 action bounds；发生 clipping 时 control 失效而不是把幅度当作相同。该 control 只证明当前 projection/metric 对一个指定 axis shift 有 sensitivity，不证明能发现所有方向或真实 action-distribution changes。

`same-noise MSE` 只作为 mapping 对照；新增 `Dpair/Dnoise≥0.25` 后，只有它满足该预注册 large-map-drift gate，才可使用“map drift with bounded marginal screen”标签。它仍不能单独触发 marginal-preserving 或 shift gate。

## Collapse 与 sampling fluctuation

`Q_A↔Q_B` 距离小本身不等于 mode collapse；同一窄分布的两个有限样本 block 也会产生小距离。Q-Q 只作 warning，不能单独标 `collapse`。主 collapse control 应使用固定 projections 的 IQR：至少 80% directions 满足 `IQR_Q/max(IQR_FP,1e-4)<0.5`，并且 `W_QQ<0.5W0`，才标记 `mode-collapse`。只有 Q-Q 小而 IQR 比未达到该条件时，标记 `sampling-null fluctuation/shape warning`，不阻断为 collapse；FP IQR 普遍接近零则 condition 为 null-degenerate。对应的 IQR 大于 2 且 Q-Q 大于 2W0 可标记 expansion warning。

## 跨 condition 验收

- **正向 bounded dissociation**：至少 6/8 condition 同时满足 `Dnoise>1e-6`、`Dpair/Dnoise≥0.25`、translation sensitivity control 通过、`marginal-preserving-screen` 且无 collapse。措辞限于“bounded preliminary screen”；不作 equivalence、significance、task success 或普适性声称。
- **负向 preservation screen**：至少 6/8 condition 为 `marginal-shift-screen`；若主要是 collapse，结论写成“shift/collapse failure mode”，不能把 collapse 当作普通边际偏移。若 sensitivity control 失败，任何靠近 null 的正向判定都降级为 `sensitivity_inconclusive`。
- **统计 inconclusive**：condition 标签混合超过 2 个、任一必要 seed/schema/condition binding 失败、null 退化、metric discordant，或只有 32 总 noise 却试图作 preserve/dissociation 正向结论。32 总 noise 仅能在效应明显且全部控制通过时记录方向性 shift，不能证明等价或稳定保真。

## V100 与停止条件

64 draws × 2 arms × 8 conditions 共 1024 个 action samples；可在真实 SLURM V100 allocation 中用固定 micro-batch 分块，但 micro-batch 只能改变执行方式，不能改变预注册样本集合。静态材料不足以保证 `≤15 min`；runner 必须记录 actual count、elapsed time、GPU identity 和 peak memory，并在 deadline/OOM 时停止为 `resource_blocked`，不得静默裁剪。任何 seed overlap、隐式 `noise=None` 缓存、condition 混用、缺 FP-FP/Q-Q 或 FP-only translation control、projection 调参，均是执行阻断。

因此该设计值得一次最小 screen；若只能提供 32 总 noise 或无法保存上述 null/collapse 证据，应直接停止，而不是扩大模型、episode 或统计方法。

## FINAL_PROTOCOL_REVIEW（追加冻结审查）

审查对象为冻结草案 `PROTOCOL.zh.md`（逐行核对，未运行实验）。结论：统计结构已接近可 freeze；**有两项必须在提交 GPU 前消除的合同歧义，另有一项需确认 runner 标签逻辑**。

### 已确认无阻断的部分

1. 草案已正确限定为 FP/Q 两个 model arm；64 draws × 2 × 8 = 1024 samples。A/B 各 32，paired 与 cross-block marginal 比较的随机化关系正确。microbatch4 对逐条 FP 的 `max_abs≤1e-5` 是合理的工程 gate；失败应标 `implementation_inconclusive`，不能转成科学 no-go。
2. `Dnoise` 使用 FP-only 的全部 32×32 cross-pair，且明确不把 1024 个距离当作独立样本；`Dpair/Dnoise≥0.25` 因而可以作为 frozen bounded-pilot 的 large-mapping binding。`sqrt(7×0.25×Dnoise)e1` 只由 FP 量确定，已消除此前以 `Dpair` 构造 translation control 的循环性。
3. `valid≥13`、至少 13 个 projection ratio 落在 `[0.5,2]`，以及“至少 8 个 valid projection ratio<0.5 且 `W_QQ/W_FF<0.5` 才标 collapse”的组合是可执行的。Q-Q 小而没有 IQR/spread 证据只标 sampling-null warning，避免把有限样本波动误报 collapse。

### 必须修正或明确

1. **Fresh-condition exclusion 仍可能越过 campaign lock。** 最新 `INDEX.zh.md` 把 0–83 标为已用、84–95 标为 PRR `test_locked`；`PROTOCOL` 的“排除 Flow 与 padded 两份 manifest 后取最早未用者”没有明文排除整个历史/locked 集合。除非那两份 manifest 已经完整列出 0–95 并在资产中 hash 绑定，否则这是 data-binding blocker：必须加入冻结的全局 exclusion manifest，不能让“earliest unused”自行挑到 84–95 或已查看 condition。
2. **指标坐标必须唯一。** 主 gate 第 15 行定义的是 `normalized model physical first-action7`，但 `Dpair` 第 19 行仍写 `physical first-action MSE`。为保证 mapping ratio 与 translation 的 `sqrt(7·...)` 推导成立，`Dpair`、`Dnoise`、SWD、energy 和 translation 必须全部使用同一 normalized-model 7-D tensor；并明确 MSE 是先对 7 个坐标取 mean、再对 draws/cross-pairs 取 mean。若 `physical` 意指另行 unnormalize，当前门槛与 translation amplitude 都不可解释，应在 freeze 前改成精确 tensor 名称。
3. **确认 projection count。** 当前 `PROTOCOL` 明确是 7 axes + 9 seeded directions，即 `K=16`。若 root 所说的 `K10` 是 intended projection count 而不是 `num_steps=10`，二者尚未一致；runner/verifier 不能自行选择 K。必须以 protocol 中一个数字为唯一权威，并相应固定 valid-projection 分母。

translation control 若因 action bound clipping、坐标阶段不一致或 metric implementation 失败，结果只能是该 condition 的 `sensitivity_inconclusive`；它不构成 `mechanism_no_go`，也不能将“Q 接近 FP null”升级为 preserve。若 control 在 ≥6 个 joint conditions 通过，才允许使用正向 bounded-dissociation gate。`mechanism_no_go` 同样必须限定为至少 6/8 个同时满足 mapping binding、reference/spread/sensitivity 可识别和 shift 的 condition；sensitivity 失败只能导致 `statistical_inconclusive`。

最终正向 gate 的 `≥6/8 joint set` 是合理的 bounded screen，但剩余两 condition 仍可能是 clean shift；结果必须逐 condition 列出并写成“至少 6/8 bounded”，不能概括为八个条件的 distribution equivalence。`0.25、1.25、1.50、IQR 0.5/2` 都是冻结 heuristic，接受的前提是报告绝对距离、valid 数量和 control 结果，不能包装成显著性或一般 power。

在完成上述 exclusion、坐标/MSE 定义和 K 值确认后，我认为该两臂 1024-draw 协议可执行，无需增加 full statistics 或额外 pilot；在确认前不应提交 GPU。
