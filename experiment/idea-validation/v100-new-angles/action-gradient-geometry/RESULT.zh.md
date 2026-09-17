# Action-gradient geometry：mechanism_no_go

2026-09-13。**冻结的联合假设未通过，停止，不改阈值或量化强度重试。**

GPU64767（CCDS TC1N03 V100，22s）完整采集，CPU64769 独立 FP64 raw replay 通过。6 个新 episodes 中，4 个同时满足 global objective fidelity 和 local 可识别性；符合“全局保真但局部梯度明显破坏”的为 **0/6**。这达到预定的可识别数量，故结论是 mechanism_no_go，而非仅因 binding 不足而 inconclusive。

| Validation index | score Spearman | centered-score NRMSE | 4-anchor median gradient cosine | global binding |
|---|---:|---:|---:|---|
| 118 | 0.970696 | 0.232805 | 0.648655 | pass |
| 119 | 0.971383 | 0.361968 | 0.982799 | fail |
| 120 | 0.965934 | 0.236803 | 0.953055 | pass |
| 121 | 0.984661 | 0.247761 | 0.888515 | pass |
| 122 | 0.981456 | 0.232062 | 0.531163 | pass |
| 123 | 0.955907 | 0.278473 | 0.933549 | fail |

global 门槛为 Spearman>=0.90、NRMSE<=0.25；坏局部几何要求 median cosine<=0.50，并且 Q-direction 的一步 FP objective 改善不超过 FP-direction 的一半。所有 episode 的 median cosine 都高于 0.50，因此联合坏几何门槛均未过。6 个 episode 的 norm/FP-step local binding 及 FP/Q directional finite-difference checks 全部通过。

这是对 **Wall epoch65、predictor-only W4 RTN、H2、64 固定 candidates、4 固定 anchors** 的小诊断。不能据此断言所有 world models 的 action gradient 都耐量化；也不支持真实环境梯度、closed-loop success 或 native W4 性能结论。这里保真的对象是官方 terminal visual+proprio objective，不是单独 latent output。

## 身份与原始证据

checkpoint SHA256 为 `8441971becdae934fe08de5b163398390a32f6fe1fb0a8df113290e2468e142b`；DINO-WM source commit `0a9492fa12044b852ae9e001cc74604b79c8bb0c`、DINOv2 commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`。运行记录保存实际 source/helper hashes，encoder unchanged、exact weight restore、action-only autograd 均通过；无 STE 或训练。

新 indices118..123 对应底层 WallDataset IDs `546,1809,1689,1078,1069,1`，metadata 映射与排除的0..117不重叠。新 init/goal 由固定 env seeds970000..970005在相应layout初始化；candidate seeds980000..980005。没有为补证读取84..95的reserved样本状态，也没有执行 environment rollout。

原始数组 SHA256：`7caa6c7eda3ee119773ad720a248568ebfb00e25aadab8f34756cde529959700`，保留于 compute storage `artifacts/64767/raw_gradient.npz`。本地小记录：[GPU summary](artifacts/64767/summary.json)、[工程记录](artifacts/64767/engineering.json)、[独立 CPU verification](artifacts/64769/verification.json)。全部逐 episode/anchor scores、gradients、FD 原值和 step gains 保留，未只留下汇总。

首个job64766因 NumPy namespace 接线错误在产生 scores 前退出；已按原协议修复，记录见 [实施失败审计](IMPLEMENTATION_FAILURES.zh.md)。该错误已被实际完整运行和 FD checks 排除，没有证据把本次科学 no-go 归因于实现。因此不进入改进/再验证支路。

本切面来自 global objective 与 local action-gradient fidelity 的分离；[QuantWM](https://arxiv.org/abs/2602.02110) 和 [Do Transformer World Models Give Better Policy Gradients?](https://arxiv.org/abs/2402.05290) 构成近邻先例。未将通用 gradient-aware PTQ 或这个负结果包装成 novelty 证明。
