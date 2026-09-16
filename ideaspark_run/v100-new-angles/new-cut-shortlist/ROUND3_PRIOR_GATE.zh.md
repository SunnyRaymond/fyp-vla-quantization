# ROUND3 PRIOR GATE：WM/WAM/VLA numerical quantization 新切面

审查日期：2026-09-13。本文只做 C02/C04 机制筛选，不运行模型、数值或
cluster，不下载新模型，不改既有 protocol/result。输入是本地
[`INDEX.zh.md`](../INDEX.zh.md)、近期 RESULT/PRIOR/no-go 和已核验的
SmolVLA、TD-MPC2 seed3 assets。每项都把“研究假设”“已有证据”和“待定实现”
分开；novelty 只写为检索边界内的风险判断，未认证。

## 共同边界

现有可复用资产是 LeRobot `v0.4.4` 的 SmolVLA，以及严格兼容的 TD-MPC2
official seed3 checkpoint。SmolVLA 官方实现先为 image/language/state 建立
prefix，再建立一次 `past_key_values`，在固定 `num_steps` 的循环中把同一 cache
传给每次 `denoise_step`，最后执行 `x_t = x_t + dt * v_t`；这正是本轮可以
单独观察“跨 denoising call 的量化误差记忆”的结构证据（见官方
[`modeling_smolvla.py`](https://raw.githubusercontent.com/huggingface/lerobot/v0.4.4/src/lerobot/policies/smolvla/modeling_smolvla.py)，
约 725--819 行）。SmolVLA 的模型/任务背景见其
[`primary paper`](https://arxiv.org/abs/2506.01844)。

TD-MPC2 source 的 `WorldModel.Q` 在 `return_type='avg'` 下随机取两个已有
critics，先调用 `two_hot_inv` 再求平均；`two_hot_inv` 对 101-bin logits 做
softmax、support 加权和 `symexp`。官方 pinned
[`world_model.py`](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/common/world_model.py)
和 [`math.py`](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/common/math.py)
是判断 decoder-tail 项的依据。所有 proposed screen 仍使用原 objective/API；
不替换 score function，不做 full rollout、training、QAT 或 native low-bit
性能宣称。历史 0--123 的输入/预留区不重新使用；若 root 冻结某项，须另登记
fresh namespace。

全局排除关系如下：本轮不做 RankCal 的 candidate global ranking、OTC 的
observation-dominated `C_r`、CEM-Update 的 update search、FRT residual transport、
PRR paired recovery 或 TR-PVQ trajectory relation；也不重演已经停止的
gradient、antithetic member coupling、semantic interface scale、flow-geometry、
padding feedback、conditional action marginal、TD five-member coupling 和
value-head centering。两条 SmolVLA 项的读出是 paired per-step field/cache
evidence，不能借 endpoint 相关性冒充 task success。

## 1. Denoising-call rounding persistence（SmolVLA）

**状态：`conditional-go`（仅窄 diagnostic；novelty 未认证，固定低比特部署的
实用收益不作结论）。**

### 研究假设

把同一 W4 weight 的 stochastic rounding (SR) draw 在一个 action chunk 的
10 次 denoising call 中如何复用作为唯一自变量：

* `SR-frozen`：每个 sample/weight 只抽一次、随后 10 步复用；
* `SR-redraw`：每一步按预注册 `(sample, step, seed)` 抽新 draw。

两者使用同一 W4 scale、bit budget、随机分布和 marginal expected weight；
只改变 rounding perturbation 的跨-step correlation。假设是 `SR-frozen` 的
误差可能沿 Euler path 保持相同方向，造成 endpoint drift；`SR-redraw` 可能
降低这种 coherent accumulation。这个方向测试的是“weight perturbation 的
时间相关性”，不是改变 denoising step 数或选择较幸运的 action。

### 已有证据与 closest prior

[PTQD](https://arxiv.org/abs/2305.10657) 已明确讨论 denoising-step 中的
correlated/uncorrelated quantization noise、mean/variance deviation 与误差
积累；[AccuQuant](https://arxiv.org/abs/2510.20348) 则在 PTQ 中显式模拟多步
denoising 以处理 accumulated error。它们使“多步会累积量化误差”本身不新。
本项的窄差异是：不训练 quantizer、不做 bias/variance correction，而在已有
flow-matching action head 上只改变同一 W4 weight noise 的 frozen-versus-
per-step redraw schedule。该差异不足以认证方法 novelty；若要求 deterministic
fixed-weight kernel，SR-redraw 也应提前标为 practical no-go。

### Naive baseline、负对照与 identifiability

Naive baseline 是现有 `W4-RTN-static`：同一量化 weight 在所有步骤固定，另保留
FP32 arm。主比较只在 `SR-frozen` 与 `SR-redraw` 之间进行，避免把 SR 的平均
偏差与时间相关性混在一起。最低工程负对照是在 `W4-RTN-static` 上打开/关闭
“per-step redraw”开关；两次输出必须逐元素相同（FP32 tolerance
`1e-6`），否则是 wrapper/PRNG 污染，不是研究效应。

固定 raw image、state、instruction、initial noise、10-step time grid、W4
scale/code 和 model state。除正常 endpoint action 外，保存每一步 `x_t,v_t`
和在同一 FP path 上的 field error `v_Q(x_t^FP)-v_FP(x_t^FP)`，以区分：

1. 每步 field MSE 已变好/变坏；
2. 每步 marginal 近似相同但 cross-step error autocorrelation 改变；
3. 只是由不同 PRNG 或输入 path 造成的 endpoint 差异。

若无法把每步 quantizer seed、weight code 和 common FP path 固定下来，结果只能
是 `implementation_inconclusive`，不得解释为 accumulation mechanism。

### <=10 min V100 最小 screen 与 gate

在已准备的 SmolVLA raw samples 中新登记 6 个 paired samples、两个固定
noise seeds；每 sample 跑 FP32、W4-RTN-static、SR-frozen、SR-redraw，固定
10 steps，保存 `v_t/x_t` 与 endpoint action。没有 full task rollout。单张
V100 的预期工作量约为 `6 × 4 × 10` denoise calls，目标 wall time <=10 min；
任何重操作仍须真实 allocation guard 先核验 hostname/owner/partition/node。

先过工程 gate：FP/no-op、RTN 开关负对照、所有 input/noise/time/scale
fingerprint 一致且 finite。机制 gate 预注册为：至少 5/6 samples 的
`SR-redraw` endpoint drift 相对 `SR-frozen` 严格下降 >=10%，同时 common-FP
path 的 per-step field MSE 比值在 `[0.95, 1.05]` 内；否则标
`mechanism_no_go` 或 `inconclusive`，不能以 RTN 比较绕过。若 field MSE 本身
变化超过 5%，停止并把它归为 marginal quantizer difference。即使通过，也只
称 SmolVLA checkpoint-specific temporal-correlation diagnostic，不推出真实
机器人成功率、native kernel 加速或可部署 SR policy。

### 与已有切面边界

它不涉及 antithetic 的两个 model/member 或 candidate-pool joint law，不做
gradient/action direction、flow-geometry ranking、step-count refinement、
padding coordinate feedback 或 conditional marginal；也不改 semantic input
scale、batch scale、FRT/PRR 的 residual/update/LoRA。与已有 flow-step prior
相同的地方是观察 iterative path，新增的唯一干预是 **同一个 weight draw 的
跨-call persistence**；PTQD/AccuQuant 的 overlap 仍使 novelty 保守。

## 2. Prefix-KV reuse as an activation-quantization memory（SmolVLA）

**状态：`conditional-go`（需 runtime cache-shape preflight；novelty 未认证，
不作通用 KV-cache recipe 宣称）。**

### 研究假设与独立变量

SmolVLA 的 prefix `past_key_values` 由 image/language/state prefill 一次生成，
随后 10 次 suffix denoising 都复用；suffix token 则每步由当前 `x_t` 和
timestep 重新生成。候选切面把 activation quantization 的**持久化对象/位置**
作为自变量，而不是调 bit allocation：

* `prefix-KV-A8`：对 prefill 产生的每层 K、V 做 symmetric int8
  per-tensor quantize-dequantize 一次，再复用这份 cache；
* `suffix-A8`：对每步 `suffix_out`（进入 `action_out_proj` 前）做同 recipe 的
  int8 quantize-dequantize；prefix K/V 保持 FP。

每个 activation family 的 scale 从独立、预先冻结的 calibration split 得到，
不按 DEV 结果选择；两臂 bit width、zero-point、calibration size 和 total
activation budget 相同。假设是 prefix-KV 的同一 perturbation 跨 10 步被重复
读取，因而其 field-error autocorrelation 与 endpoint drift 会不同于每步更新
的 suffix perturbation。该假设能被“两个 locus 在共同 FP path 上没有差异”推翻。

### 已有证据与 closest prior

官方 SmolVLA source 的 `past_key_values` reuse 是直接结构证据；它不是推测的
history slot。通用 LLM 文献已经研究 KV activation quantization，例如
[KVQuant](https://arxiv.org/abs/2401.18079) 的 per-channel/pre-RoPE/non-uniform
KV 方案，故“KV cache 可以量化”不是新 claim。本项只留下一个较窄的 VLA
flow-matching diagnostic：同一个 prefix cache perturbation 是否因 action
chunk 的 repeated denoising reuse 而呈现不同的 temporal error memory。它没有
提出 KVQuant 的 channel/bit allocation，也没有把此差异包装成新 deployment
method。现有 Semantic Input Scales 已测试输入 interface scale partition；
这里的对象是内部 prefill cache 与 suffix output，仍有实现和 prior overlap
风险。

### Naive baseline、负对照与 identifiability

Naive baseline 是 `suffix-A8`（静态 per-family A8、每步重新量化 suffix）；
FP32 保留作 common path anchor。负对照是 `FP-cache-clone`：对 FP prefix
`past_key_values` 做 detached deep clone 后重复使用，必须与原 FP cache
逐元素相等（tolerance `1e-6`）。若 cache clone 或 K/V nested shape 改变输出，
先停在 engineering failure。另须确认 quantize-dequantize 后的 K/V 没有 alias
回写 FP cache，并记录每层 K/V shape、scale fingerprint 和实际读取次数。

固定 raw input、instruction、initial noise、time grid、checkpoint 和
`use_cache`；每一步在同一 FP `x_t` 上重放一次，保存 prefix/suffix 两种 arm
的 `v_t`、endpoint、per-step field error 与 lag-1 autocorrelation，不保存大
activation bank。主 gate 不是 task success：只有当至少 5/6 samples 的
`|corr(prefix-KV)| > |corr(suffix)|` 且 endpoint drift 方向预注册、并有
>=10% effect size 时，才给窄 mechanism support；否则 `mechanism_no_go` 或
`inconclusive`。若 cache nested structure、scale source 或 common-path replay
无法锁定，则直接停止，不用 fallback 猜 API。

### <=10 min V100 最小 screen 与边界

使用 6 个新 paired SmolVLA samples、固定两个 noise seeds；四臂为 FP32、
FP-cache-clone、prefix-KV-A8、suffix-A8，10 denoise steps，保存小型 raw
velocity/action/工程 manifest。prefill cache 已生成后只需约
`6 × 4 × 10` suffix calls，目标单 V100 <=10 min；不执行 environment rollout。
若需要重新 prefill 才能确保 cache ownership，必须先在 preflight 证明时间预算，
否则 resource/engineering stop。该切面不是 semantic scale 的重命名：干预
对象是 **内部、跨-step reused K/V activation**，不是 visual/proprio/action
输入接口；也不改变 solver steps、action distribution、gradient、padding、
FRT/PRR 或 five-Q coupling。

## 3. Support-aware Q decoder-tail rounding（TD-MPC2 seed3）

**状态：`identifiability_no_go`（保留为负结论，不申请 GPU）。**

### 研究假设与拟议干预

官方 TD-MPC2 的 101-bin Q head 不是直接 scalar regression：`two_hot_inv` 对
logits 做 softmax、support 加权，再做 `symexp`；planner 的真实路径用
`return_type='avg'`。一个看似不同的 recipe 是在固定 W4 row scale/bit budget
下，让最后 Q head 的 integer rounding 优先降低 **decoded scalar** 的局部误差，
而不是只降低 raw-logit MSE；naive baseline 是同一 head 的 per-row RTN，FP32
是 anchor。它研究 bin-tail/decoder curvature 对 numerical error 的影响，
不改 planner `avg`、不增加 critic、也不改 score function。

### Closest prior 与否决理由

[AdaRound](https://arxiv.org/abs/2004.10568) 已把 data/task-aware weight
rounding 作为 PTQ 方法，[GPTQ](https://arxiv.org/abs/2210.17323) 已使用
layerwise second-order information 做 one-shot low-bit rounding；因此“用输出
误差指导 rounding”没有足够 novelty。它还直接靠近本地已冻结的
`value-head-gauge`：后者虽然只做 softmax-invariant common-mode centering，
本项改成 decoder-sensitive bin-wise rounding，但二者都只触碰同一 101-bin
head。

更根本的问题是 identifiability：FP decoded Q 只是 learned critic reference，
不是环境 return、Bellman target 或 planner success oracle。即使 decoded-Q
MSE 降低，也不能知道真实 value error 或 action choice 是否改善；这正是本地
Bellman-consistency 与 value-head-gauge 的边界。用更多 cached states、更多
rounding seeds 或另一个 self-referential decoder loss 不能解除该缺口。

### 最小 falsification（只说明为什么不跑）

若强行做 diagnostic，8 个 fresh reset state × 64 个 fixed FP-imagined H3
actions × `FP32 / W4-RTN / decoder-aware-W4`，用官方 `two_hot_inv` 和固定
`avg` 记录 raw logits、decoded Q、pair selection；一次 V100 应在 10 min 内。
但在没有外部 value/return anchor、且不做 rollout/training 的前提下，任何
positive gate 都只能证明“自我 reference 更接近自己”，不能支持 quantization
mechanism 或 policy claim。因此正式 gate 在执行前已经是 `identifiability_no_go`；
不得因 raw-logit 或 decoded-Q 数字漂亮而启动。工程负对照仍应是 FP decoder
round-trip/no-op exact，失败只说明实现问题。

## 决策摘要

| 候选 | 独立干预对象 | 状态 | 建议 |
|---|---|---|---|
| Denoising-call rounding persistence | 同一 W4 weight SR draw 的跨-step reuse vs redraw | `conditional-go`；novelty 未认证 | 只有 pre-registered common-path gate 通过且有已有 SmolVLA allocation 才考虑 <=10min diagnostic |
| Prefix-KV reuse memory | 内部 prefix K/V cache A8 vs suffix output A8 | `conditional-go`；novelty 未认证 | 先做 cache-shape/alias preflight；不能用 fallback 猜 nested API |
| Support-aware Q decoder-tail | TD-MPC2 101-bin Q head 的 decoder-aware rounding | `identifiability_no_go` | 不申请 GPU；保留作为排除证据 |

本文件没有认证任何新颖方法，也没有消费数据 namespace、模型加载或 cluster
allocation。后续若 root 选择第一或第二项，只需另写冻结 protocol、重新登记
fresh samples，并把工程证据、common-path raw 和 negative control 放在结果前；
不得把本轮 prior gate 当成实验结果。

