# Camera redundancy × PTQ interaction：prior gate

日期：2026-09-13　范围：targeted prior/source audit；未连接 cluster、未加载模型、未下载数据、未做本机数值计算。

## 判定

当前候选记为 **`identifiability_no_go`，GPU=0**。它可以定义一个窄的“view-mask condition × fixed-weight PTQ drift”诊断，但现有 SmolVLA 资产不能把这个交互解释成 camera redundancy 的作用。现有两路输入是 scene camera 与 wrist camera，不是同一状态下的两路冗余视角；mask wrist 会移除可能唯一的接触/夹爪信息，并改变视觉 token 的 attention padding。于是观测到的差异可由信息缺失、view-specific cue、mask/attention 路径和 quantization interaction 任一项造成。

若未来补齐同状态、action-equivalent 的冗余视角对，只应把本候选降格为 conditional-go 的诊断：固定 quantizer 后，询问 PTQ drift 是否依赖 view availability。那一结论仍不等于“冗余使 quantization 更稳”，也不涉及 task success、failure prediction 或 native low-bit speedup。

## 候选的可检验对象

固定同一 checkpoint、同一 expert W4 RTN recipe、同一 raw observation、instruction、proprio 与 diffusion noise。设视图条件 `v∈{full, wrist-masked}`，权重条件 `q∈{FP,W4}`，`y_q(v)` 是既定 screen 输出（例如已有 flow/action-chunk 输出，具体 frozen metric 需在 protocol 中绑定）。不改变 scoring function，定义量化漂移为同条件内的

`Δ_Q(v) = d(y_W4(v), y_FP(v))`，

并观察 `I_Q = Δ_Q(wrist-masked) − Δ_Q(full)`。`y_FP(wrist-masked)` 只作该条件的 reference，不能当作 task ground truth；同时必须报告 `d(y_FP(wrist-masked), y_FP(full))`，以显示 mask 本身造成的输出变化。若这个 FP view shift 已很大，`I_Q` 至多说明“量化扰动对输入可见性条件的依赖”，不能说明冗余被量化破坏。

这个 factorial 与已有切面有明确边界：instruction contrast 改变 language condition，Semantic Input Scales 改变 scale partition，Prefix KV vs suffix 改变 quantization locus/shape/Jacobian；Flow Geometry under PTQ 虽使用两路真实 camera，但没有 camera-mask factor。这里保持 weight locus、bit、scale 与 metric 不变，只改变同一输入的 view-mask condition。

## 资产与官方接口核对

当前 flow 资产记录了真实 raw keys `observation.images.image` 与 `observation.images.image2`，saved preprocessor 后分别成为 `observation.images.camera1` 与 `observation.images.camera2`；checkpoint 还声明 camera3，但 dataset 没有它，且 `empty_cameras=0`。见本地 [STATE_INTERFACE_AUDIT](../flow-geometry-drift/STATE_INTERFACE_AUDIT.zh.md) 与 [flow geometry result](../flow-geometry-drift/RESULT.zh.md)。因此现有 camera2 是 wrist view，不能被称为与 camera1 的 duplicate/redundant view。

LeRobot v0.4.4 的官方 [`prepare_images`](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/smolvla/modeling_smolvla.py#L365-L402) 先收集仍在 batch 中的 configured image keys；对每个 present key，若存在 `<image_key>_padding_mask` 就使用该 boolean mask，否则默认全 True。缺失 image 只有在 `empty_cameras>0` 时才会补 placeholder 和 false mask。`embed_prefix` 随后把 false image mask 作为 padding 传入视觉 prefix attention，见同文件 [`embed_prefix`](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/smolvla/modeling_smolvla.py#L566-L652)。

所以当前 checkpoint 下的“正确 mask”必须保留 wrist image tensor，并在 **post-preprocessor** batch 使用 `observation.images.camera2_padding_mask=False`（具体是否由 saved preprocessor 一并 rename companion key，必须由实际 batch keys 证明；不能猜 raw suffix 的行为）。删除 camera2 key 不是 mask：在 `empty_cameras=0` 下它会改变 present image 数量和 token layout。mask 也不会跳过 `embed_image`，因而不能提出省算力或省存储的 claim。SmolVLA 的 `empty_cameras` 默认和配置语义见官方 [`configuration_smolvla.py`](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/smolvla/configuration_smolvla.py)。

## 为什么现有屏幕不能识别 redundancy

1. **缺少冗余构造。** camera1/camera2 的物理位置与内容不同，没有同一 state、instruction、proprio、action 下的 duplicate scene view 或 rerendered camera pair。只做 wrist mask 是 view availability ablation；它不能估计“保留另一视角足以恢复被遮挡信息”。
2. **mask 改变了任务信息。** wrist view 可能含有 gripper/contact 的唯一证据。`Δ_Q(full)` 与 `Δ_Q(masked)` 的差异同时混入 missing cue 与 attention padding 的效应；FP masked/full 的差异不能被扣除成 redundancy effect，除非有 action-equivalent matched view control。
3. **输入改变不是 quantizer 改变。** 即使 W4 codes、scale、weight restore 完全固定，`I_Q` 仍是一个受输入内容和 mask 影响的 conditional sensitivity。它可作为诊断，但不是一个新的 quantization mechanism；将其命名为 redundancy masking 会把未识别因素写进机制结论。
4. **已有 camera prior 已覆盖关键动机。** Cross-view 工作把 wrist stream mask 掉来避免 unperturbed wrist shortcut，并在同 state/action-equivalent view pair 上施加 action-level consistency；那种设计正好说明 mask 需要 state-matched view 语义。当前 screen 没有它的 paired-view construction，也没有训练 consistency loss。

## Closest primary prior

- [Cross-View Action Consistency, arXiv:2608.06965v1](https://arxiv.org/abs/2608.06965)：在 action-equivalent、同 state/instruction/proprio/action 的视图对上做 flow velocity consistency，并在训练/评估 mask wrist stream 以避免 shortcut。它研究 cross-view invariance/training，不研究 PTQ 或 fixed W4 drift；它也说明单独拿掉 wrist view 不能自动代表 redundancy。
- [BFA++, arXiv:2602.20566](https://arxiv.org/abs/2602.20566)：把 multi-view token 的 intra-view/inter-view redundancy 与 critical-view selection 作为视觉效率问题。它是最直接的 redundancy 邻域，但不是 quantization × view-mask interaction；因此“camera redundancy 影响模型”本身没有新颖性保证。
- [QVLA, arXiv:2602.03782v1](https://arxiv.org/abs/2602.03782)：以 action-space deviation 定义 VLA PTQ sensitivity，并讨论高维视觉输入的 apparent redundancy 与 action head sensitivity。它不做多 camera mask factorial；本候选若能成立，新增的只是固定 PTQ drift 对 view condition 的交互。
- [arXiv:2605.28803](https://arxiv.org/abs/2605.28803)：官方 arXiv 页面当前标题为 **HoloQ-VLA: Uniform W4A4 Quantization of Vision-Language-Action Models**，而本 campaign 将其称作 OmegaQVLA；该 title/ID discrepancy 需保留，不能把两者未经核对当作同一直接先例。官方摘要聚焦 uniform W4A4、rotation、activation dispersion 与 per-step scaling，没有 camera-mask factor。相关 [Omega-QVLA repository](https://github.com/UCMP13753/Omega-QVLA) 也不提供本候选的 matched-view PTQ 证据。
- 本地 [Flow Geometry under PTQ result](../flow-geometry-drift/RESULT.zh.md)：已有两个真实 camera、FP/backbone-W4/expert-W4 的 drift screen，但没有 mask factor；因此它是接口资产与 metric 邻域，不能当作 redundancy 证据。

这是 targeted coverage，不是 exhaustive novelty search；上述交集“fixed expert PTQ × wrist mask”是否已有其他先例仍未认证。当前 no-go 来自 identifiability，而不是声称该交集在所有文献中都已被做过。

## 若要保留为窄诊断，必须满足的 conditional protocol

这部分只定义复核条件，不构成当前 GPU go。

**干预与配对：** 每个 sample 固定 raw bytes、state、instruction、task/episode/frame、noise 和 output metric；先经过同一 saved preprocessor，再保留 camera1/camera2，只为 camera2 附加 false padding mask。FP/W4 两臂各跑 full 与 wrist-masked；W4 只使用冻结 expert module 与既定 RTN scale/codes，完整 restore 后换条件。至少需要由外部数据提供同 state/action-equivalent 的第二 scene view，或明确把结论改名为 view-availability sensitivity。

**必要 controls：**

- FP full/masked paired shift 单独报告；它若超过预先冻结的 drift 量级，停止 redundancy interpretation。
- 同一 full image 附加全 True `camera2_padding_mask` 的 mask no-op，应与无 mask 输出在 frozen tolerance 内一致；否则停止，说明 mask plumbing 改变了路径。
- 检查 camera order、post-preprocessor mask key/dtype/shape、present keys 正好为 camera1/camera2；任何 camera3 placeholder、camera2 omission 或 silent fallback 都是 engineering failure。
- 每个 condition 的 FP reference 独立配对；不得用 FP full 作为 masked condition 的 truth，也不得把 task score 或成功率引入这个短 screen。

**最小规模与停止规则：** 若资产条件后来补齐，可用现有 8 个已冻结样本、每样本 2 个固定 noise、4 个 factorial cells，single V100 目标 ≤10 min；先完成 1 个 sample 的 schema/no-op gate，再做其余 cells。停止条件包括 identity/hash、preprocessor、mask schema、finite、restore 或 no-op 任一失败；以及没有 matched redundant view、FP mask shift 主导、或无法证明每 cell 使用同一 raw sample/noise。科学上只有在 redundancy pairing 先通过、`I_Q` 的方向/置信区间与预注册阈值都满足后，才能写“quantization–view condition interaction”；即使通过，也暂不写 redundancy causality 或 deployment claim。

## 最终记录

**Decision：`identifiability_no_go`; recommendation：不跑 GPU。** 可保留上述 narrow diagnostic 作为未来补齐 matched-view 资产后的 conditional-go 模板。当前两-camera asset 只能支持“mask wrist 后输入条件变化下的 PTQ output drift”，不能支持“视图冗余隐藏/放大量化敏感性”的机制结论。
