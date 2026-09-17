# WM/WAM/VLA numerical quantization：两个新切面最小筛选

**日期**：2026-09-13（Asia/Singapore）  
**状态**：pre-study screen；两个候选都**未测试，novelty 未认证**。本文只冻结可在一张 V100 上做的最小筛选，不把条件性信号写成方法结论。

## 共同边界

- 锚点是现有 DINO-WM Wall epoch65 predictor。**已核验的 checkpoint/runtime `num_hist=1`**（PRR manifest/records 的 `DEFAULT_NUM_HIST=1`，且 adapter 会拒绝不一致的 loaded model）；source 默认配置里的 `num_hist=3` 不能覆盖 checkpoint 事实。`concat_dim=1`、`action_emb_dim=10`、`proprio_emb_dim=10` 仍给出每个视觉 patch 的 `384 + 10 + 10 = 404` 维输入。只做 fake numerical quantization screen；不声称 native kernel、实际 latency、显存收益或跨模型迁移。
- planner、CEM objective、horizon、candidate sampling 和 action selection 全部固定；不改评分函数，不做 `RankCal` all-candidate ranking，不做 quantization ensemble/selection-bias，也不重演 `TR-PVQ`、FRT、PRR、OTC 或 CEM-Update 混合精度搜图。
- 方案组织采用 `idea-spark` 的 `C04 heterogeneous_decomposition`（按可检验的异质块拆分）与 `C02 controlled_diagnostic_design`（固定 reference、naive baseline、一个匹配负对照）；`C17 targeted_self_supervised_objective` 只作 pattern overview 对照，没有引入 latent relation 或额外训练目标，以免重演 `TR-PVQ`。
- 所有模型加载、latent rollout、评估和重 I/O 必须在真实 CCDS `SLURM` compute allocation 内执行，并在脚本最前面检查 `SLURM_JOB_ID`、实际 hostname、partition 和 GPU；不能在 head/login node 推理或准备大型数据。此次没有提交远程作业。

## 候选 1：Recency-Split Latent-History Cache PTQ（RSLH）——现有资源/模型结构 no-go

### 干预对象与机制

干预 `VWorldModel.rollout` 中反复送入 predictor 的 latent history buffer `z[:, -num_hist:]`，而不是 predictor weights、planner score 或 action head。原始假设是对每次滑窗中的**较旧两个 history slots**使用固定的 per-channel symmetric W4 fake quantizer，只保留最新追加的 slot 为 FP16；predictor/encoder 和原有 planner 保持当前 reference 格式。该干预不使用 residual/error feedback、observer online update、codebook、bit allocator 或 ensemble。

假设是：`num_hist=3` 的旧 latent 在后续 autoregressive rollout 中被重复读取，误差会沿 history 窗口传播；最新 slot 离下一次 transition 最近，保留它可把“缓存精度”与“新旧位置”分开。**但现有 Wall epoch65 实际只有一个 history slot，因此该 age locus 在本资源/模型结构上不存在。** 不能把 source 的 `num_hist=3` 改写进 checkpoint，也不能把改过的模型当作同一 anchor；这里不能直接推出 attention KV cache 或 WAM/VLA 的结论。

### Closest prior 与差异

- [DINO-WM](https://arxiv.org/html/2411.04983) 提供 action-conditioned visual world model 及 rollout 结构；它不是 latent-history cache PTQ 的研究。
- [QuantWM](https://arxiv.org/html/2602.02110v1) 在 DINO-WM 上系统比较 weight/activation PTQ、量化粒度与 rollout/task degradation；其干预重点是模型参数/activation quantizer，而不是显式按 history age 量化 persistent `z` buffer。
- [KIVI](https://arxiv.org/html/2402.02750) 与 [KVQuant](https://arxiv.org/html/2401.18079) 证明 LLM KV cache 可因 key/value 分布差异采用不同轴/粒度；[DeltaKV](https://arxiv.org/abs/2602.08005) 进一步利用 KV residual similarity。它们是 attention KV、token/cache 压缩近邻，不是 DINO-WM 的 action-conditioned latent history；本候选不做 residual coding、token eviction 或 learned cache。
- [QuantWAMs](https://arxiv.org/html/2607.28405v1) 已涉及 WAM activation calibration 与 rollout auditing，因此“历史 activation 有风险”本身不能算新颖；可能的窄差异仅是固定 age mask 作用于 DINO-WM `z` buffer。novelty 必须由后续完整检索和实验证据重新判断。

### Naive baseline（仅解释 no-go，不执行）

**All-history-W4**：若有三-slot checkpoint，三个 history slots 均用同一套 per-channel W4 fake quantizer；权重、encoder、planner 和 objective 与 RSLH 完全相同。现有 epoch65 只有一个 slot，因而不存在可匹配的 all-history baseline；不得通过修改 `num_hist` 伪造该比较。

### 唯一负对照（不可实例化）

**Reverse-age control**：三-slot 设计下同样只保留一个 FP16 slot、同样的 bytes 和 quantizer，但保留最旧 slot FP16、量化最新两个 slots。现有单-slot checkpoint 没有“最旧/最新”可交换位置，所以该负对照也没有合法实现。

### ≤1h V100 最小实验：不执行

因为 loaded Wall epoch65 的 `num_hist=1`，RSLH 在该资源/模型结构上于接口门前停止，不申请 V100、不采集数据、不规划新的 checkpoint。任何强行改 `num_hist=3` 的实现都回答了另一模型的问题，不能作为该 screen 的实验。

### 停止规则

1. 已满足 `resource/model-structure_no_go`：checkpoint/runtime `num_hist=1`，不足以构成“旧两个 vs 最新一个”的 RSLH intervention。
2. 若未来另有真正的三-slot checkpoint，必须先独立核验 model structure、history materialization 和 FP no-op；不能复用本 Wall epoch65 的结论或 registry。
3. 不得以修改 `num_hist`、重训/重存 checkpoint 或删减 history 来绕过该 no-go；本候选从现有 Wall screen 中移除。

## 候选 2：Semantic-Block Activation PTQ（SBAQ）

### 干预对象与机制

干预 predictor 的实际输入 activation `z`（在 positional addition 之前），按已知语义边界使用三套固定 affine observers/scales：visual `384` 维、proprio `10` 维、action `10` 维；每个 block 使用同一个固定 activation bit（建议先 A8，若现有 fake-quant wrapper 只支持 A4 则用 A4），predictor weights 的 W4 设置、planner 和 objective 均固定。这里是输入 activation 的 quantizer layout，不是 SmoothQuant 的 function-preserving rescaling；因为随后会加 positional embedding 并经过 LayerNorm，不能声称等价变换。

机制假设是：visual block 的幅度/维度可能主导一个 404-d full-tensor observer，使低维 action/proprio conditioning 得到较粗的有效分辨率；semantic block observers 可以在相同 activation bits 下隔离这种量化误差。它不搜索每块 bit，不做 branch gating/rotation，不校准 score，也不改变 action selection。

### Closest prior 与差异

- [QuantWM](https://arxiv.org/html/2602.02110v1) 已比较 per-tensor、per-token 等 activation granularity；SBAQ 只提出在 DINO-WM 输入接口按 visual/proprio/action 语义边界固定分组。
- [QuantWAMs](https://arxiv.org/html/2607.28405v1) 的 shared-basis activation calibration、outlier protection 与 rollout audit 是强近邻；若其 coordinate-compatible calibration 已覆盖该 partition，本候选应视为 prior collision，而非新方法。
- [SmoothQuant](https://arxiv.org/html/2211.10438) 将 activation difficulty 迁移到 weights 的等价变换；本 screen 不做该迁移。[QuantVLA](https://arxiv.org/abs/2602.20309) 涉及 VLA selective quantization/temperature/output-head balancing，也使“action-aware activation”这一宽泛表述缺乏 novelty。

因此 SBAQ 的可辩护差异只有“固定、无搜索、输入端 semantic partition + matched anti-semantic control”；这是待验证的小切面，不能宣称新颖。

### Naive baseline

**Full-input observer**：对 404-d `z` 使用一套 standard affine observer（相同 activation bits），W4 weights、数据、seed、planner 和 objective 与 SBAQ 完全相同。

### 唯一负对照

**Permuted-block control**：固定预注册 channel permutation 后，仍使用相同的 `384/10/10` block sizes、相同 observer 数量和相同 calibration data；只打乱 semantic labels。若它与 SBAQ 相当，semantic boundary 不是 load-bearing 机制。

### ≤1h V100 最小实验

在真实 CCDS `SLURM` GPU allocation 内，取 4–6 个新的 Wall states/episodes，固定 action pool、`H=5`、64 candidates 和原始 terminal DINO objective；比较 FP reference、full-input observer、SBAQ、permuted-block control。先用单次 calibration 统计各 block 的 scale/range，再做短 fixed-pool rollout；若接口已支持，追加 2–4 个短闭环 episode。记录 input-to-prediction latent NMSE、first-action agreement 及 endpoint distance/success。禁止把 scale 选择、planner ranking 或 action ensemble 引入 screen；结果只回答 semantic partition 是否值得继续。

### 停止规则

1. 若 quantizer 不能挂在实际 `z` 输入 op 之前，或 FP no-op 不能逐元素复现 reference，标记 `implementation_failure`。
2. 若 SBAQ 与 full-input observer 的实际 scales/outputs 相同，标记 `no_mechanism`，停止。
3. 若 SBAQ 在 paired endpoint 上不优于 full-input observer 且不优于 permuted-block control，标记 `mechanism_no_go`，停止。
4. 若只有 activation NMSE 改善、first-action/endpoint 不改善，降级为 diagnostic-only；鉴于 QuantWM/QuantWAMs 的强近邻，不做额外 bit/partition 搜索，不升级为完整项目。

## 证据边界与来源

上述候选是基于现有代码形状、既有 screen 约束和以下原始/作者来源构造的**待筛选假设**：

- DINO-WM：<https://arxiv.org/html/2411.04983>
- QuantWM：<https://arxiv.org/html/2602.02110v1>
- KIVI：<https://arxiv.org/html/2402.02750>
- KVQuant：<https://arxiv.org/html/2401.18079>
- SmoothQuant：<https://arxiv.org/html/2211.10438>
- QuantWAMs：<https://arxiv.org/html/2607.28405v1>
- DeltaKV：<https://arxiv.org/abs/2602.08005>
- QuantVLA：<https://arxiv.org/abs/2602.20309>

文献近邻检查截至 2026-09-13；没有进行 exhaustive literature review，也没有运行本地/远程实验。所有“机制”“closest prior”和“novelty”均应在最小实验及后续正式审计后重写。
