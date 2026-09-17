# FINAL_BOUNDED_CUTS

日期：2026-09-13  
范围：只审查现有 corpus、已冻结实验和 primary source；不连接 cluster、不加载模型、不读取数值 raw。`INDEX.zh.md` 中已有 11 个实测 STOP、既有 prior/no-go，以及另行处理的 EulerJacobian 均排除。以下是本轮最后两个最接近的结构候选；结论是均不值得解冻一次 GPU screen。

**修订记录（2026-09-13）。** 原文 DINO-WM source URL 写错为 `world-model-2026/dino-wm`；现改为已核验的官方仓库 [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm)，并固定 source commit `0a9492fa12044b852ae9e001cc74604b79c8bb0c`。本地 source 与该 commit 的 `repeat`/`cat` 路径一致。

## 结论

本轮没有新的 conditional-go。两个候选都能形成简单的 FP/W4 对比，但结果只能说明“某一模块较敏感”，不能识别所声称的语义机制。按 C02/C04 边界，两者均为 `identifiability_no_go`，`GPU=0`；这是未测试的研究判断，不是已做实验后的数值 no-go。

## 1. DINO-WM tiled state/action broadcast quantization

**状态：** `identifiability_no_go; novelty_no_go; GPU=0`。

**研究假设。** 在 `concat_dim=1` 的 Wall checkpoint 中，proprio 和 action embedding 会沿 visual patch 轴复制，再与 visual features 拼接；当前 epoch65 的已核验结构为 `num_hist=1`、`384+10+10=404`。因此，量化 state/action encoder 可能产生跨 patch 完全相干的 latent error，经 predictor rollout 放大；这与逐 patch 变化的 predictor/visual error 可能不同。

**已有证据。** 官方 DINO-WM source 的 `encode` 明确对 proprio/action 做 patch 维 `repeat`，`predict` 随后把 patch 轴展平；`rollout` 逐步重新注入 action embedding。现有 Wall protocol 将 action/proprio encoder 保持 FP32，已做 screen 主要量化 predictor；因此该具体 locus 尚未被本 campaign 直接测试。结构事实来自已固定 commit 的 [DINO-WM official source](https://raw.githubusercontent.com/gaoyuezhou/dino_wm/0a9492fa12044b852ae9e001cc74604b79c8bb0c/models/visual_world_model.py)，checkpoint 结构来自本地 `reproduction/dino-wm-wall/source/models/visual_world_model.py` 和已冻结 manifest。

**Closest prior。** [DINO-WM](https://arxiv.org/html/2411.04983) 已把 image/goal latent 与 action-conditioned prediction 作为同一 planning graph；[QuantWM](https://arxiv.org/html/2602.02110) 已系统覆盖 DINO-WM 的 encoder/predictor、weight-only/activation PTQ 及 granularity。故“encoder W4 比 predictor W4 更敏感”会落入已有 module/locus sensitivity，而不是 broadcast-coherence 机制的新证据。

**最小方案（仅作若重开时的草案）。** 固定同一 6-state/动作/seed，比较 FP、仅 state/action encoder W4、仅 predictor W4，保存每 patch latent error 与 rollout error；单 V100 可在 10 min 内完成。所有 predictor、输入、planner 和量化 recipe 固定，并保留 first-step FP no-op。

**必要反事实与负对照。** 真正识别 broadcast 需要在不改变其他计算的情况下，把同一 encoder error 变成独立 patch error，或只注入一个 patch。两者都不是 native PTQ：前者改变随机变量结构，后者改变 predictor 输入语义。若仅用自然的 visual-encoder W4 或 patch permutation，差异同时混合了模块、输入内容和误差预算；若结果只显示 aggregate latent/action drift，即直接触发 `identifiability_no_go`。不得把这种人工 decorrelation 当作模型部署方案。

**停止规则。** 找不到保持 checkpoint graph、参数预算和 native PTQ 语义的 broadcast-only counterfactual；或结果只能由 encoder-vs-predictor locus 解释；或需要改写 patch 输入，均停止且不提交 GPU。已有 QuantWM 覆盖也足以将新颖性降为 `novelty_no_go`。

## 2. SmolVLA cross-attention versus causal self-attention PTQ

**状态：** `identifiability_no_go; novelty_no_go; GPU=0`。

**研究假设。** SmolVLA 的 VLM 与 action expert 通过 interleaved cross-attention（CA）和 causal self-attention（SA）生成 action chunk。量化 CA 或 SA 的 Linear weights 可能造成不同的 denoising-time / action-position drift：CA 扰动条件信息注入，causal SA 扰动 action-token 内部传播。

**已有证据。** 官方 [SmolVLA paper](https://arxiv.org/html/2506.01844) 描述 interleaved CA+SA，并讨论 CA/SA ablation 与 causal SA 的未来 action dependency；官方 [v0.4.4 modeling source](https://raw.githubusercontent.com/huggingface/lerobot/v0.4.4/src/lerobot/policies/smolvla/modeling_smolvla.py) 显示 `predict_action_chunk` 使用显式 noise，`denoise_step` 经 `vlm_with_expert.forward` 处理 prefix/suffix mask。现有 Flow Geometry screen 只做完整 backbone W4 与完整 expert W4，没有 CA/SA 拆分；但这留下的是 allowlist 工程空缺，不等于机制可识别。

**Closest prior。** SmolVLA paper 自身已将 CA/SA 作为结构分支做消融；已有 Flow Geometry 结果已覆盖 backbone/expert 的 PTQ drift。进一步比较 CA-W4、SA-W4 仍是同一 quantization recipe 下的 module/locus sensitivity；[QuantWM](https://arxiv.org/html/2602.02110) 的 encoder/predictor sensitivity 也说明仅换量化位置不足以建立新的 numerical mechanism。未作 exhaustive novelty search，不能宣称文献中绝无该命名。

**最小方案（仅作若重开时的草案）。** 固定 6 个现有 SmolVLA conditions、两 noise、10-step sampler，比较 FP、CA-only W4、SA-only W4；记录每一步、每个 action position 的 velocity，并做同 shape FP/RTN 与 no-op。若 exact v0.4.4 module allowlist 已被 runtime 证明，计算量可压在单 V100 10 min 内。

**必要反事实与负对照。** 要把 position drift 归因于 causal propagation，至少要有保持 CA/SA 参数量、输入和 attention mask 不变的 branch-only counterfactual，例如同一层内等预算的 token-local perturbation。简单地对 CA 与 SA 各量化一次会混合层数、参数量、prefix/suffix token 数和输入路径；位置曲线也可能只是 action-token 内容或 shared output projection 的敏感性。若必须引入 token permutation、改 causal mask，或调用 Euler/Jacobian 才能分离，均属于已排除的语义/微分干预，不应作为新 screen。

**停止规则。** runtime 无法给出严格互斥且同预算的 CA/SA weight allowlist；只能报告 branch drift 而无 native counterfactual；或差异在 FP/RTN/no-op 控制下无法排除 output-head、token-count、输入路径混杂，均为 `identifiability_no_go`。即使观察到 CA 与 SA 数值不同，也只可写 conditional diagnostic，不能升级为机制 go。

## 已筛除的相邻想法与边界

- TD-MPC2 reward/two-hot decoder、policy `tanh`/clamp、SimNorm、termination：分别撞上 decoder-tail、policy-prior、Simplex、Bellman/结构 no-go；当前 single-task seed3 也没有可用 variance path。不会用新的 scalar metric 重新命名。
- SmolVLA prefix `state_proj` cache versus suffix action path：确定性 weight-only PTQ 下，cache 与重算是同一函数；比较两处权重只是输入/locus 混杂，撞上 Prefix-KV 与 null-input 的既有否决。
- DINO-WM goal/current shared quantization、camera redundancy、action reinjection、temporal persistence、mixed-bit/calibration、generic noise：均已在 INDEX 或 shortlist 中有明确停止记录，本轮不重演。

## 证据边界

本轮只做了 targeted primary-source/source-code review，未作 exhaustive literature search；“novelty_no_go”表示与现有 campaign/prior 的可区分性不足，不是全领域不存在先例。后续如出现 native、matched、same-budget 的结构反事实和新增 asset 不需要下载，候选可另行重审；在当前资源和 10 min single-V100 约束下，不应为获得一个数字而放宽 identifiability gate。
