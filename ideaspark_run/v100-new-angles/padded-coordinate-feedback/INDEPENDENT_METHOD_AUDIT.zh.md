# Padded-coordinate-feedback 独立方法审查

日期：2026-09-13。审查了 PRIOR_AND_INTERFACE.zh.md 及其中 METHOD_DERIVATION_AND_PRIOR 段（当前目录未见另一个同名文件），并核对 Flow 64761 asset identity；未改文档、未实现、未连接集群、未运行计算。

## 判定

结论：conditional narrow-go，可保留为一个最小的四臂 causal diagnostic；analytic replacement 的解释需要收窄。它不是新的 quantizer，也不是 ground-truth field。将 analytic 替换笼统称为 oracle/non-deployable 过强：它不需要 FP model oracle forward 来产生 padding 速度，使用已知的 initial noise 即可执行一个自定义 sampler ablation；但与 FP/Q physical slice 拼成的 hybrid field 仍是 counterfactual intervention，不能当作未经修改的标准 policy deployment。

建议术语改为 analytic padded-coordinate intervention 或 analytic sampler ablation。oracle 只可用来表示“反事实坐标替换”，不能暗示有真实最优 velocity 或需要一个额外 FP oracle。若要声称部署可用，还需另行测 latency、接口和 task outcome；本 screen 不做这些事。

## source 与维度合同

Flow 64761 的 asset_identity_brief 记录 checkpoint action shape=[7]、max_action_dim=32、chunk_size=50、num_inference_steps=10，模型输入/输出为完整 32 维；同一 metadata 还记录官方 camera/state mapping 与 checkpoint revision。官方 LeRobot v0.4.4 configuration 将 max_action_dim/max_state_dim 设为 32，physical action feature 由 checkpoint metadata 提供，不能从常量臆测。官方 source 参考：[configuration_smolvla.py v0.4.4](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/smolvla/configuration_smolvla.py) 与 [modeling_smolvla.py v0.4.4](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/smolvla/modeling_smolvla.py)。

文档所述接口在结构上闭合：pad action 为 physical 7 维后的 25 个零，padding 坐标仍进入 action expert 的输入/输出和完整 Euler recursion；physical 7 维只在最终 action chunk 截取。正式执行仍必须现场断言 normalized/padded action 的 7:32 为 exact zero、velocity 为 [10,50,32]、输出 slice 为 0:7，并确认没有 physical-coordinate mask。metadata 的 7/32 只能证明接口，不单独证明当前 runtime source 没有改写。

## analytic 推导核对

在 source 使用的 bridge x_t=t z+(1-t)a、target velocity v=z-a 且 a_pad=0 的条件下，padding slice 满足 x_pad(t)=t z_pad、v_pad=z_pad。反向 Euler 从 t=1 的同一 initial noise 开始，以 dt=-1/K 更新 K 步，直接注入固定 z_pad 会在每个网格点得到 x_pad=t z_pad；x_pad/t 只在 t>0 可用，t=0 必须直接使用 z_pad，当前段落对此处理正确。

关键边界是：z_pad 是 paired conditional bridge target，不是有限模型学到的 marginal conditional field 的 ground truth。FP/Q 模型在任意 current state 上的 pad velocity 可以偏离 z_pad；因此 analytic branch 是有明确代数依据的坐标 intervention，不是“恢复真实 padding velocity”。它不需要另一个 FP forward 来计算 padding，但 FPanalytic 仍需用 FP model forward 计算 physical slice；Qanalytic 同理。

四臂必须各自从同一完整 initial noise、condition 和 K 开始，并在自己的 current full x_t 上求 field：

    FPnative   = [v_F,phys(x_F,t), v_F,pad(x_F,t)]
    Qnative    = [v_Q,phys(x_Q,t), v_Q,pad(x_Q,t)]
    FPanalytic = [v_F,phys(x_FA,t), z_pad]
    Qanalytic  = [v_Q,phys(x_QA,t), z_pad]

每个 branch 都更新完整 x_t；禁止把一个 branch 的 physical velocity 与另一个 branch 的 padding velocity 拼接。第一步四臂的 current state 相同，所以应分别断言 FPanalytic/FPnative 和 Qanalytic/Qnative 的 physical slice 逐元素相同；后续 physical divergence 才有 feedback 解释。

## 最小四臂 gate

保留文档的共享 analytic FP reference，但把 gate 写成同时约束 reference shift、native Q drift 和 Q padding error。令 A 为最终 physical action，建议在相同 sample/noise 上保存：

    B_F = mean ||A_FPanalytic - A_FPnative||²
    D_0 = mean ||A_Qnative - A_FPnative||²
    D_1 = mean ||A_Qanalytic - A_FPanalytic||²
    I_Q = mean ||A_Qanalytic - A_Qnative||²
    P_F = mean ||v_F,pad - z_pad||²
    P_Q = mean ||v_Q,pad - z_pad||²

其中 P_F/P_Q 应按所有 K steps、padding coordinates 和固定样本聚合，不能只看最终 physical action。最小可解释 gate 为：

1. 四臂输入 noise/condition/K 完全相同，首步 physical equality、shape、finite 和 branch reset 全部通过；失败即 implementation/inconclusive。

2. B_F 不超过预先冻结的 reference tolerance，并最好同时满足 B_F 相对 D_0 的上限；否则 FPanalytic 自身已改变 physical endpoint，Q 的“改善”无法归因于量化 padding feedback。

3. D_0 与 I_Q 均非退化，且 P_Q > 预先冻结的 nonzero threshold。若要把结果归因于 PTQ 而不是一般 pad-field 偏差，应再要求 P_Q−P_F 超过预先冻结 margin；否则只能报告 generic analytic pad intervention。

4. 只有在 1–3 通过且 D_1 相对 D_0 达到预先冻结 improvement margin（例如 D_1 ≤ (1−η)D_0）时，才报告“analytic replacement reduces Q-vs-FP physical drift”。同时保留 B_F、D_0、D_1、I_Q、P_F、P_Q；不能只报一个改善比例。

D_1 必须以 FPanalytic 为共享 reference，D_0 以 FPnative 为 native reference；这正是排除 FP reference shift 伪收益的必要条件。D_0 接近零、P_Q 接近零、B_F 过大或首步不相等时，标记 no_binding/inconclusive，不调阈值补救。该 gate 不证明 physical action 正确、环境 success、安全性或部署收益。

## primary prior 边界

[RePaint](https://arxiv.org/abs/2201.09865) 在 reverse diffusion 每步重新注入已知图像区域，说明 known-coordinate reverse-process intervention 是已有通用模式；它不是 flow-matching action padding，也没有 PTQ feedback 解释，因此构成方法学邻域而非直接覆盖。

[Diffusion Policy](https://arxiv.org/abs/2303.04137) 使用 action-distribution score 的迭代 denoising，覆盖 action diffusion field 的推理语境，但没有 unused padded coordinates、32→7 executed slice 或 Q-only coordinate replacement。检索到的 [Constrained Flow Matching via Lagrangian Dual Flows](https://arxiv.org/abs/2607.04513) 是一般约束 flow 的 dual intervention，也没有 SmolVLA 的 padding/PTQ causal screen。

因此 analytic path 本身不能作为 novelty claim；窄差异仍是同一 current state、同一 initial noise 下，把未执行的 7:32 velocity 替换为 conditional analytic path，并观察 physical endpoint 的 Q drift 是否变化。该差异足以支持 bounded diagnostic，但不能写成新 constraint algorithm、oracle accuracy 或 deployable improvement。

## 最小执行建议

执行前只需完成：source/checkpoint identity、7/32 split 与 exact-zero pad assertion、同一 noise/condition/K、每 branch policy/cache reset、首步 physical equality、P_Q nonzero binding、B_F reference gate、D_0→D_1 improvement gate。按 sample 保存逐步 physical/pad velocities 与五个聚合量；不追加 layer attribution、更多 baseline、training、完整 environment rollout 或 task success。

最终状态建议分开写：structural_go 表示接口和推导成立；analytic_feedback_signal 表示四臂 gate 通过；no_binding/inconclusive 表示无法识别。即使 gate 通过，也只可称 unused-coordinate feedback diagnostic；若缺少 Q-specific P_Q binding，则降级为 generic intervention，不能声称 PTQ-specific mechanism。

## PROTOCOL_GATE_REVIEW（最终冻结版）

日期：2026-09-13。只读审查最终 PROTOCOL.zh.md；没有运行实验、连接集群或修改 protocol/runner。

### 判定：no blocking

最终 8-episode、expert-W4、四臂设计的 gate 具有独立机制含义，没有发现会改变结论的 reference 偷换或数学 tautology。E0 是 Qnative 相对 FPnative 的 native drift，S 是 FPanalytic 相对 FPnative 的 reference shift；S ≤ 0.01E0 先限制 analytic FP reference 的移动幅度。G1 使用 FPnative reference，G2 使用共享的 FPanalytic reference，且两者都必须达到 median ≥ 0.25、至少 6 个 binding episodes 同时为正。因此不能仅通过移动 FP reference 制造收益，也不能只凭一个 reference 通过。

E0 > 1e−6 与首步 padded MSE P0 > 1e−10 是 non-degenerate binding floor；它们不是显著性检验。E1/E2 都是最终前 8 个 chunk、7 个 physical coordinates 的 normalized action MSE，G1=1−E1/E0、G2=1−E2/E0 的分母固定为同一 native drift，数学上可解释。若实测值仅贴近 P0 下限，应如实报告 gate margin 很小，但这不会使冻结规则变成 tautology。

### 实现时必须保持的最小语义

1. positive-count 应明确实现为 binding episode 集合上的交集：count(binding_i 且 G1_i>0 且 G2_i>0) ≥ 6；两个 median 也只能在 binding episodes 上计算。不得把非-binding episode 的正 gain 混入计数，也不得只保存通过子集。

2. P0 必须取第一步、同一输入 full x_t、替换前的 raw predicted padded velocity，覆盖完整 50×25 slice，并在两固定 noise seeds 内按 protocol 聚合。四臂每个 branch 的第一步输入 state 应逐元素确认相同；只确认 noise tensor 相同而不确认实际 x_t，会留下缓存或 reset 污染的解释漏洞。

3. FPnative/Qnative/FPanalytic/Qanalytic 都要从独立 clone 的同一初始 full noise、condition 和 K=10 开始；每个 branch 在自己的 current state 上求 model output，再只替换 analytic branch 的 7:32。physical velocity 不能从另一 branch 借用。helper 保存的 states、raw predicted velocities、actual used velocities、actions 与 actual recorder times 应带有 branch/episode/seed 维度，不能从 step index 事后猜时间。

4. “排除 Flow64763 已用 episodes”必须使用 64763 manifest 的实际 task/episode IDs，而不是跳过每个 task 的前 3 个位置。选择新 8 episodes 时先完成 metadata/source-ID 绑定，再产生任何模型输出；保存 selection rule、实际 source IDs、frame、manifest/revision/hash。这样才能证明 readonly source indices fresh。

5. 每 arm 前后需要 policy/cache reset、同一权重 snapshot/restore 和 finite/shape 检查；Q 只允许 expert transformer Linear W4，action input/output projections 与 language/vision 等保持 FP32。analytic path 的 padding used velocity 应逐步等于同一初始 noise 的 7:32，并用 actual time 检查 t·z_pad 误差 ≤ 1e−5。

### 解释边界

该 gate 只能给出当前 checkpoint、expert-W4、8 个新 episodes 上的 offline padded-coordinate feedback diagnostic。通过时可报告 analytic replacement 与 Q-vs-FP physical drift 的关联；失败时分别使用 no_binding、method_no_go 或 inconclusive_binding。不能将 G1/G2 称为统计显著性、physical success、deployment gain 或新的 constrained-flow 方法。已有 RePaint/known-coordinate intervention 只构成邻域 prior，analytic path 本身仍不是 novelty claim。
