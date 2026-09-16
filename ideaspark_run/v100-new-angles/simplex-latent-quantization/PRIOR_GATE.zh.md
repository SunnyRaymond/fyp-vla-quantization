# SimNorm latent simplex-preserving A4：PRIOR GATE

日期：2026-09-13。本文只做 source/prior 与 identifiability 审查；没有下载、模型加载、数值计算或 cluster 操作。

## 判定

当前宽泛主张“固定整数预算能恢复 latent transition composition”判为 **novelty_no_go / mechanism-claim structural_no_go**。可保留一个很窄的 **conditional diagnostic**，但不建议为它单独申请 GPU，也不能把它包装成新的 quantizer 或 deployment 方法。

官方 TD-MPC2 的 `SimNorm` 只是把最后一维 reshape 成大小为 `simnorm_dim` 的组并逐组执行 `softmax`；在固定 source 中它被用于 encoder 和 latent dynamics 的输出，默认 `simnorm_dim=8`、`latent_dim=512` 时是 64 个 8-simplex。TD-MPC2 论文把该 latent 说成用于 normalization/stable continuous regression，并没有赋予各组物理或语义的“composition”含义。[官方 `layers.py`](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/common/layers.py)、[官方 `world_model.py`](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/common/world_model.py)、[TD-MPC2 paper](https://arxiv.org/abs/2310.16828)

## 假设与最大反对

候选假设是：对每个 group 的 `p∈Δ^7`，把 `15p` 变为整数 `q` 且 `sum(q)=15`，再用 `q/15`，比逐坐标 `round(15p)` 后 renorm 更能保留 action-conditioned latent dynamics 的 groupwise competition，并降低 3-step imagined endpoint 或 model score drift。

最大反对有三点。第一，两臂若都做 renorm，`sum(p)=1` 已经恒成立；以 simplex residual 为 primary 是 tautology。第二，largest-remainder 是固定和整数 simplex 上的标准 nearest-point rounding；若它降低 one-step L2，只能先解释为几何投影误差下降，不能直接解释为 transition composition 被恢复。必须检查 multi-step downstream drift 是否超出 one-step distortion 可解释范围。第三，`q/15` 只是 fake-quant activation，除非另有 packed kernel、存储格式与 latency 证据，不能声称真实 A4 deployment 或 bytes 节省。

## 先例与独立性

[Simplicial Embeddings](https://arxiv.org/html/2204.00616v2) 已把表示分成多个 simplex，用 softmax 施加 groupwise sum constraint，并明确讨论对 simplex 做 hard discretization；这直接覆盖“simplex representation + discretization”的大框架。[Softmax Bias Correction for Quantized Generative Models](https://arxiv.org/abs/2309.01729) 已实证低比特 softmax 输出的 sum drift/bias，并用 normalization-dimension correction 修复；[APQ-ViT](https://arxiv.org/abs/2303.14341) 也已把 preservation of Softmax structure 作为 PTQ 目标。此次 bounded pass 没有找到 TD-MPC2 SimNorm dynamics 上 `sum(q)=15` 的 exact recipe，但“对 normalized activation 保持 simplex”仍是明显 generic overlap，不能据此认证 novelty。

真正尚未决定的问题只能写成：在固定 TD-MPC2 checkpoint、固定输入和同一 4-bit grid 下，**constrained lattice projection 相对 RTN+renorm 是否减少三步 dynamics 的 drift，且收益不只是 one-step activation MSE 优势**。若没有这个 residual-propagation 对照，实验只能是 quantizer sanity check。

## 若坚持做的最小 8-state diagnostic

仅在已有合法 TD-MPC2 asset/source gate 解除后，使用 8 个 fresh `cartpole-balance` reset observations 与固定的短 H3 action pool；不调用 environment `step`，不做长时 rollout 或闭环 success。三臂为 FP、A4 `RTN+renorm`、A4 `largest-remainder(sum=15)`。第一版只量化 **dynamics 的 SimNorm 输出**，encoder、reward、policy、Q 和 planner 保持 FP，以免把 observation encoder 误差混入 transition claim。两种 A4 必须共享 `q∈[0,15]`、scale/grid、dtype、candidate order；记录每组 integer sum、与 FP 的 one-step L1/L2、每个 H3 step 的 latent endpoint drift 及固定 model score/action-order secondary。

事前 gate 只允许有限结论：所有 group sum 正确只是 engineering gate；若两种 A4 在记录 cell 上几乎没有差异，标 `no_binding/inconclusive`。若 fixed-budget 仅降低 one-step L2，或 3-step gain 可由该 L2 降幅解释，标 `projection-only mechanism_no_go`。只有在至少 6/8 states 出现预注册的 endpoint/score residual-propagation 改善，并在按 one-step distortion 归一化后仍保留，才可称该 checkpoint 上的 `conditional_preliminary_go`；仍不能称 composition theorem、通用 simplex quantizer、deployment 或长时控制收益。不要增加 bits、group size、温度、task 或完整闭环来挽救 null。

## 实现待定与停止条件

必须先冻结 `SimNorm` hook 的位置（只在 dynamics 输出）、`round` 的 tie rule、`sum(q)=15` 的 tie-break、FP reference 与每步 latent 的 shape；不得把 encoder 与 dynamics 同时量化后再声称 transition-specific。若只能得到 group-sum、latent MSE，而不能保存逐步 dynamics/score 对照，则直接停止为 `identifiability_no_go`。本审查不解除现有 campaign 的资源、source、checkpoint 或 cluster gate。
