# Candidate-Batch Dependence under Dynamic A8 Activation Quantization

**日期**：2026-09-13（Asia/Singapore）  
**状态**：conditional diagnostic screen；未测试，novelty 未认证，且有一个很近的 security prior。本文不建立新模型、不采集数据、不提交作业。

## 先回答源码核查问题

结论是：**QuantWM 官方 `quant_utils` 没有一个明确实现的“跨 candidate batch 共享动态 per-tensor scale”路径。** 这点必须和候选假设分开。

1. 官方 [`omni_quantize/quantizer.py`](https://github.com/huawei-noah/noah-research/blob/master/QuantWM/quant_utils/omni_quantize/quantizer.py#L555-L805) 的 `UniformAffineQuantizer` 保存 `dynamic` 和 `dynamic_method`，但 `forward` 实际按 `dynamic_method` 分支；`per_token` 与 `per_channel` 都调用同一个 `per_token_dynamic_calibration`。该函数在无 `group_size` 时使用 `reduce_shape=[-1]` 的 `amin/amax`，只归约最后一维，保留所有 leading dimensions。对 `x[B,R,C]`，得到的是 `[B,R,1]` 的 scale，而不是一个跨 `B` 的 scalar。
2. `group_size` 分支先把输入 reshape 成 `[-1, group_size]`，随后仍按每一行/每组的最后维归约；它不会自动把 candidate 维合并成共享 scale。源码 docstring 写了 `per_token and per_cluster`，但 `forward` 只接受 `per_token` 或 `per_channel`，不能据 docstring 补出一个未实现的 `per_cluster`。
3. 官方 [`models/ptq/layers.py`](https://github.com/huawei-noah/noah-research/blob/master/QuantWM/models/ptq/layers.py#L968-L985) 的 `quantize_activation_per_token_absmax` 同样只按最后一维求 `xmin/xmax`；[`QAct.forward`](https://github.com/huawei-noah/noah-research/blob/master/QuantWM/models/ptq/layers.py#L1195-L1274) 的 `token_wise` 是逐 token 动态，`layer_wise` 则使用 observer 预先得到的固定参数。
4. [`observer/base.py`](https://github.com/huawei-noah/noah-research/blob/master/QuantWM/models/ptq/observer/base.py#L318-L353) 对 activation 把 leading dimensions 展平为样本；[`observer/minmax.py`](https://github.com/huawei-noah/noah-research/blob/master/QuantWM/models/ptq/observer/minmax.py#L392-L419) 在 `layer_wise` calibration 时再把 channel extrema 压成 scalar。这个 scalar 是跨 calibration inputs 的静态参数，不是每个 forward co-batch 重新计算的动态 per-tensor scale。官方 Wall planning config 也把 `calib_mode_a` 写成 `layer_wise`（见 [`plan_wall.yaml`](https://github.com/huawei-noah/noah-research/blob/master/QuantWM/conf/plan_wall.yaml#L386-L404)）。
5. 官方 [`awq/quantizer.py`](https://github.com/huawei-noah/noah-research/blob/master/QuantWM/quant_utils/awq/quantizer.py#L334-L408) 的 `pseudo_quantize_tensor` 输入名是 `w`，按 weight row/group 求 extrema；不要把这个 weight-only path 当作 candidate activation batching 证据。

因此本候选若要测试 `PT-A8`，必须**显式新增一个只用于 screen 的 quantizer arm**：对当前 predictor activation `x[B,R,C]` 用

\[
s_B=\max_{b,r,c}|x_{b r c}|/127,
\qquad Q_B(x)=\operatorname{clip}(\operatorname{round}(x/s_B),-128,127)s_B,
\]

即每个 co-batch 只有一个动态 symmetric A8 scale。不能声称这是 QuantWM 官方当前默认行为，也不能把 screen arm 伪装成官方 reproduction。

## 候选机制

锚点为已核验的 DINO-WM Wall epoch65 predictor；当前 checkpoint/runtime 使用 `num_hist=1`，所以本候选不依赖 history-age，也不修改 `num_hist`。在同一实际 predictor activation op 上，将 `B` 个 CEM action candidates 一起送入 quantizer：

| arm | scale 作用域 | 角色 |
|---|---|---|
| `FP` | 无 quantization | reference |
| `PT-A8` | 每个 forward co-batch 一个 `s_B`，归约 candidate、rollout/token 和 channel | 待测 candidate-batch coupling |
| `PC-A8` | 每个 candidate 一个 `s_b`，归约该 candidate 内的 `R,C` | candidate-isolated repair |
| `PTok-A8` | 每个 candidate/token-row 一个 `s_{b,r}`，只归约 `C` | naive baseline；对应官方 token-wise 语义 |
| `Fixed-A8` | 预注册的单一 static `s*`，不读取当前 co-batch | 与 FP 配对的固定-scale negative control 部分 |

`PT-A8` 的可检验机制是：某个 action rollout 产生 outlier 时，`s_B` 变大，量化网格变粗，**同一 target candidate** 的 activation、terminal latent、固定 objective score 甚至 top-k/elite membership 会随 co-batch 成员改变。`PC-A8` 和 `PTok-A8` 是精度作用域对照，不改 bit budget、score function、CEM update 或 action selection rule。

### Naive baseline

`PTok-A8`：沿用每个 token/patch row 的动态 A8 scale。这是合理的 naive baseline，因为它保持输入 candidate 的独立性，但 scale 数量和 online reduction 更多；不把它称作新方法。

### 唯一负对照

`NC-FP/Fixed-A8` 是一个 matched same-shape control family：对同一个 `x[B,R,C]`、同一 target candidate、同一 co-batch swap，分别记录 FP bypass 与预注册 static `Fixed-A8`（`s*` 不由当前 batch 重算）。两者都不让 co-batch 成员改变量化参数；若 target 在该 control 下仍随 co-batch 变化，说明是 shape、GPU batching 或 runner nondeterminism，不能归因于 dynamic scale coupling。该 control 不增加 candidate ranking 或额外模型。

## Closest prior 与 novelty 风险

- [QuantWM](https://arxiv.org/abs/2602.02110) 是直接的 WM prior：在 DINO-WM 上研究 weight/activation PTQ、粒度和 planning horizon，并报告 activation granularity 的收益不一致；它没有在该 paper/官方上述代码中给出 candidate-batch shared dynamic scale 的 planning diagnostic。它仍使“activation quantization 影响 planning”这个宽主张没有 novelty。
- [Quantamination](https://arxiv.org/abs/2604.26505) 是更近的机制 prior：其核心正是 per-tensor dynamic activation quantization 在 batched inputs 上计算 combined min/max，使一个 input 改变另一个 input 的 quantization，并指出 per-token dynamic quantization 可消除该 cross-sample side channel。它不是 WM/CEM paper，但已覆盖“跨 batch shared scale → co-sample contamination”这一机制；因此本候选若只报告该效应，standalone novelty 应标 `novelty_no_go`。
- 可保留的窄切面是 action-conditioned WM planner 中的**同一 candidate 在固定 co-batch swap 下的 task-score/elite instability**，以及 `PC-A8` 是否在相同 A8 bits 下消除它；这只是 conditional screen，不能预先称为方法贡献或 deployment fix。

## C02/C04 pattern provenance

- `C02 controlled_diagnostic_design`：同一 initial state、同一 target candidate、同一 candidate position、同一 seed，只替换 co-batch 成员；FP/static control 负责区分 dynamic-scale contamination 与 batching artifact。paired target metrics 是诊断单位，candidate rows 不当成独立样本。
- `C04 heterogeneous_decomposition`：把 quantizer 的作用域拆成 batch-wide、candidate-wide 和 token-wide 三个轴；只改变 scale reduction axis，保持 A8、activation op、planner、objective 和 action pool 不变。该拆分是实验结构，不是额外的 learned allocation。

## ≤1h V100 最小实验（只规划，不采集）

### 数据与执行边界

- 全局数据 `0..95` 已用/保留；`96..101` 归 antithetic rounding。本候选仅规划 `102..107`（6 个新的 source episode/initial-state IDs），当前不采集、不写 raw arrays、不占用 registry。
- 将来若批准执行，只能在真实 CCDS `SLURM` GPU allocation 内加载 Wall epoch65、运行 predictor 和写出小结果。脚本必须先检查非 login hostname、非空 `SLURM_JOB_ID`、`scontrol` allocation、partition 和 `CUDA_VISIBLE_DEVICES`；head/login 不做模型加载、推理、重 I/O 或 benchmark。

### 固定 co-batch screen

在每个 planned ID 上冻结一个 action pool `K=64`、`H=5`，用 `B=16` 分批。选择一个 target candidate `a_i`，在两个 matched co-batches 中保持 `a_i` 的 tensor、batch position、initial state、RNG 和所有目标参数不变；只把一个预注册的 high-range/outlier action candidate 与普通 candidate 交换。四个 dynamic/reference arm（`FP`, `PT-A8`, `PC-A8`, `PTok-A8`）和 `Fixed-A8` 都使用同一个 predictor activation hook 与原始 terminal DINO objective。

每个 target 只报告以下 paired quantities：

1. `PT-A8` 的 `log(s_B^{swap}/s_B^{base})`，以及 `PC-A8`/`PTok-A8` 的 target scale 是否保持不变；
2. target 的 activation/terminal latent difference 与 unchanged planner score difference；
3. 在固定 `K=64` pool 的 top-k/elite membership flip。该指标只作为 co-batch perturbation 诊断，不改评分函数，也不做 all-candidate ranking、selection ensemble 或 CEM update；
4. 若 runner 已有短闭环入口，再对同一 6-ID screen 做最多 2–4 个短 episode；若没有则停在 fixed-pool 诊断，不把 score signal 写成 task success。

这是 `6 IDs × 2 co-batches × 5 arms` 的一次 inference-only gate，预算上限一张 V100 **≤1 GPU-hour**；不追加 seed、horizon、bit 搜索或完整 300-candidate/10-step CEM suite。

## 停止规则

1. 若不能在真实 predictor activation op 上插入 quantizer，或 FP bypass 在同 shape co-batch swap 下不能逐元素复现 reference，标记 `implementation_failure`，不改成另一个 hook。
2. 若 `PT-A8` 的 target `s_B` 在 6 个 planned IDs 中少于 4 个出现预注册阈值（建议 `|Δ log s| > log(1.01)`），标记 `no_batch_scale_coupling`，停止。
3. 若 `s_B` 有变化，但 target latent/score/elite 与 `NC-FP/Fixed-A8` 没有额外变化，标记 `diagnostic_only_no_planning_consequence`，停止。
4. 若 `PC-A8`/`PTok-A8` 不能消除 target 的 co-batch dependence，或在至少 4/6 IDs 上没有比 `PT-A8` 更稳定，标记 `repair_no_go`，停止；不调 observer、scale clipping 或 batch size 直到结果变好。
5. 若只有 pooled score/elite 汇总变化、paired target effect 不稳定，标记 `inconclusive`，不能称 planner improvement。
6. 文献层面已存在 [Quantamination](https://arxiv.org/abs/2604.26505) 的直接 cross-batch dynamic per-tensor 机制；若 screen 没有显示 WM/CEM-specific downstream consequence，标记 `novelty_no_go`，不扩展为论文方法。
7. 最多使用规划区间 `102..107`；不因 preliminary signal 自动打开 `108+`，不触碰 `0..95` 或 `96..101`。

## 证据边界

源码检查和上述方案只支持以下判断：官方 QuantWM activation paths 主要是 layer-wise static calibration 或 last-dimension token-wise dynamic quantization；candidate-batch shared dynamic per-tensor A8 是一个需要显式实现的 screen arm。没有运行此 arm，也没有证据证明 Wall checkpoint 中存在该 coupling、能改变实际 task outcome，或 `PC-A8` 是有效 deployment repair；所有机制、可行性和 novelty 均待屏幕实验及后续完整审计。

**源码与论文来源（访问/核查：2026-09-13）**：

- QuantWM official repository tree：<https://github.com/huawei-noah/noah-research/tree/master/QuantWM/quant_utils>
- QuantWM OmniQuant activation quantizer：<https://github.com/huawei-noah/noah-research/blob/master/QuantWM/quant_utils/omni_quantize/quantizer.py>
- QuantWM PTQ activation layer：<https://github.com/huawei-noah/noah-research/blob/master/QuantWM/models/ptq/layers.py>
- QuantWM observer reshape/calibration：<https://github.com/huawei-noah/noah-research/blob/master/QuantWM/models/ptq/observer/base.py>
- QuantWM minmax observer：<https://github.com/huawei-noah/noah-research/blob/master/QuantWM/models/ptq/observer/minmax.py>
- QuantWM Wall planning config：<https://github.com/huawei-noah/noah-research/blob/master/QuantWM/conf/plan_wall.yaml>
- [An Empirical Study of World Model Quantization](https://arxiv.org/abs/2602.02110)
- [Quantamination: Dynamic Quantization Leaks Your Data Across the Batch](https://arxiv.org/abs/2604.26505)
