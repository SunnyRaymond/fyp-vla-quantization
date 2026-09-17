# FRT CCDS-TC1 A/B pilot protocol v1

本文件是 **Frozen-Residual Transport PTQ (FRT)** 的 conditional pilot protocol。它只检验一个窄的、可证伪的机制：在固定的 DINO-WM Wall one-step map 和固定 W4 logical map 下，由一个冻结的 Q0 产生的 residual direction 是否比 clean-only 和同范数 random direction 更能保留 latent transport。Physical `xy` 只在 Stage A 记录可重放 interface capability；本轮不执行 DEV/TEST closed-loop success。

这不是 novelty 认证、native low-bit deployment benchmark 或 WAM/VLA transfer。FP32 operators 加 weight-only numerical emulation 不能推出 packed bytes、Peak VRAM、latency、control frequency 或 energy。没有 TEST，也没有 Stage C；所有结果只属于本次 Wall epoch-65 conditional pilot。

## 1. 研究问题、范围与固定对象

研究对象是 DINO-WM Wall epoch-65 artifact。`F_FP(X,a)` 和 `F_Q(X,a)` 必须是同一个 runner 提供的可重放 one-step predictor map：输入完整 history state `X` 和固定 action/chunk `a`，输出下一预测 latent slot。`X` 包含所有 context slots、proprio/action fields、normalization 和实际存在的 cache/history fields；不能把一个 latent vector 当作完整 state，也不能把 delta broadcast 到全部历史槽。

首轮只量化注册的 predictor target blocks：

~~~
predictor.transformer.layers.0 ... predictor.transformer.layers.5
~~~

encoder、policy/readout、environment 和其他参数保持 FP32。所有三个拟合方法使用完全相同的 target block set、W4 parameter freedom、initial state、optimizer、update count、hardening schedule、fit seeds 和 materialization path。每个 fit seed 产生一份 deterministic CAL minibatch schedule；该 schedule hash 在同 seed 的三个方法间必须相同，三个 seeds 可以有不同 schedule 以提供真实的 fit perturbation。只允许 `transport_source` 改变。

固定 source identity：

~~~
checkpoint: official Wall epoch65
source_commit: 0a9492fa12044b852ae9e001cc74604b79c8bb0c
dinov2_commit: 7764ea0f912e53c92e82eb78a2a1631e92725fc8
torch: 2.2.0+cu121
reference: FP32 weights and FP32 activations
execution: weight-only numerical emulation; no native low-bit claim
~~~

FRT 的 paired record 首先在同一 `X,a,RNG` 上计算并冻结：

~~~
z_fp  = F_FP(X, a)
z_q0  = F_Q0(X, a)
delta_q0 = stopgrad(z_q0 - z_fp)
~~~

`Q0` 是 symmetric per-output-channel W4 RTN。两条分支都先使用同一个已知 action embedding 完成 action replacement，再形成 residual；action coordinates 必须逐元素为零。将 `z_fp` 按源码的真实 shift/concat 顺序追加为新槽，得到 `x`；只把 `delta_q0` 加到新槽得到 `x_delta`。旧 history 不变。

对下一固定 action `a'`，transport target 为：

~~~
T_FP(x, delta, a') = F_FP(x_delta, a') - F_FP(x, a')
T_Q (x, delta, a') = F_Q (x_delta, a') - F_Q (x, a')
~~~

这是固定状态上的 finite-difference transport contrast，不是 causal kernel、严格 Jacobian 或 linearization，也不实现 residual compensation、error-feedback accumulator、online observer 或 deployment-time FP teacher。

## 2. A/B stage 与执行边界

Stage A 是 raw collection/interface audit。它在拟合前完成 CAL paired FP/Q0 records、`W_z`、CAL loss component scale、random bank 和 full-history legality 的可审计输出。当前 runner 的实际产物是 `stage_a_summary.json` 与 `bank_stage_a.npz`；bank 保存 `history`, `x_history`, `action`, `next_action`, `fp_current`, `q0_current`, `delta`, `fp_transport`, `random_delta`, `random_transport`, `wz` 及 `metadata_json`。Verifier 必须从这些 raw arrays 重算 shape、finite、Q0 delta、history shift 和 random weighted-norm 关系，不能只相信 runner 写出的 `pass` 字段。Stage A 不打开 DEV gate，也不选择 hyperparameter。

Stage B 是同一批冻结规则下的 three-method fit 与 DEV audit。当前 runner 的基础产物为 `stage_b_summary.json`, `bank_cal.npz`, `bank_dev.npz` 与 `methods/<method>_seed_<seed>.json`；机制结论还要求 runner 另存完整的 `common_bank.npz`（所有 evaluator × 所有 bank 的 raw transport matrix）。先判断 engineering gate，再判断 mechanism gate。engineering 失败、结果缺失、硬化未完成、common-bank 矩阵不完整或资源/时间不足，都不能写成 scientific no-go。只有所有 B raw outputs 完整、硬 W4/reload/state contract 通过，而事前机制门失败时，才记为 `mechanism_no_go`。

所有模型加载、重 I/O、数组读取/校验和计算都必须发生在真实 CCDS-TC1 SLURM compute allocation 中。执行前必须由既有 `allocation_guard.py` 核验非空 `SLURM_JOB_ID`、实际 `TC1N##` hostname、`scontrol show job` 为 `RUNNING`、UserId 和 allocated NodeList 一致；不得伪造环境变量。login/head node 只做提交、状态和小型控制文件操作。原始 NPZ 留在 compute-side artifact directory，verifier 只回传小型 JSON summary。

## 3. 冻结 split、targets、seeds 与预算

已检查本项目的 historical manifests 和 CEM-Update artifacts。旧目标/episodes 使用 `0–49`；`0–59` 整段在本 protocol 中保留，以防遗漏未列入 manifest 的历史记录或 target。FRT 只允许以下新 target range；runner 必须生成新的 fingerprint，并在 A raw output 中回写它们。

| split | local episodes | dataset indices | environment namespace | CEM namespace | independent unit |
|---|---:|---:|---:|---:|---|
| CAL | `frt_cal:000..005` | `60..65` | `800000 + local_index` | `810000 + local_index` | episode / initial state |
| DEV | `frt_dev:000..005` | `66..71` | `900000 + local_index` | `910000 + local_index` | episode / initial state |

`0–59` 不可作为 FRT CAL/DEV；CAL 与 DEV target fingerprints、initial states、environment seeds 和 record keys 必须互不相交。每一 episode 至少需要一个 one-step record；如果 runner 产生多个 frame/probe，它们仍嵌套在同一个 episode，不能当作额外独立 `n`。不运行 TEST，不打开 Stage C。

每个拟合方法固定三个 fit seeds：`1201, 1202, 1203`。seed 是拟合随机性，不改变 targets、record order 或 DEV random bank。方法 ID 及唯一拟合组合为：

~~~
clean:            1201, 1202, 1203
random_same_norm: 1201, 1202, 1203
frt:              1201, 1202, 1203
~~~

另外记录 `FP32` 和固定 `Q0_RTN` audit baselines；它们没有 fit seed，也不能从 baseline 的结果选择 FRT 配置。

三个 fit seeds 通过 seeded minibatch sampling 产生不同的 CAL record exposure；batch size、shuffle/permutation 规则、gradient accumulation 和每个 seed 的 schedule hash 在 A timing/stability check 后冻结。相同 seed 的 Clean、Random、FRT 必须逐 batch 使用同一 index schedule，且记录 exposure count；不能用 dropout、method-specific shuffle 或不同 batch budget 制造 seed 差异。

`fit_updates` 与 A 前未冻结的 timing choice 不得伪造为已冻结数字。当前 runner 实际使用 Adam 单一 `lr`、fraction/logit soft-rounding、temperature `2.0→0.1` 和 regularization `0.01→0.1` 的线性 schedule；这些实际 ledger 字段必须由 A timing/stability check 记录，并在 B 启动前把 update count、batch size、lr 写入新的 manifest revision/hash。B runner 只能读取该 revision，不能通过命令行静默覆盖。这个选择不使用 DEV endpoint 或机制结果。

## 4. W4、loss、W_z 与 random control

Q0 和全部拟合后的 map 都使用同一 signed symmetric W4 数学：`q_min=-7`、`q_max=7`，每个 output channel 一个正 scale。Q0 scale 为 `max(abs(w))/7`；全零 channel 使用 `scale=1`、全零 integer。Q0 RTN 固定 ties-to-even。

拟合时可以使用 AdaRound-style soft variable，但只允许在 A 之后冻结的、完整的 CAL update schedule 中使用。固定参数化和 hardening contract 为：

~~~
q_int = clip(floor(w / s) + h(alpha), -7, 7)
h(alpha) in [0, 1] during CAL
final hard h = 1[h(alpha) >= 0.5]
materialize integer q_int, discard alpha, then reload and re-evaluate
~~~

`s>0` 采用固定的正 scale parameterization；scale、rounding variable、initialization 和 optimizer 不能因方法而变。必须完成 A 冻结版本中的全部 `fit_updates` 和最后一个 hardening step；不能因为 CAL loss 暂时不再下降而 early cut soft phase。最终每个 target group 必须是实际硬 W4，`alpha` 不得出现在部署 checkpoint 或 serialized map 中。Verifier 只接受有 checkpoint hash、hard-map hash、`binary_final=true`、`alpha_discarded=true` 和 `reload_equal=true` 的 fit run。Core runner 的实际 fraction/logit initialization、soft schedule 和 scale/alpha learning rates 必须原样写入 fit ledger，并在三种方法间一致；本协议不假定 `alpha=0` 或某个未审计温度公式。

主方法的唯一拟合 loss 为：

~~~
L_clean = mean_i || W_z (F_theta(X_i, a_i) - F_FP(X_i, a_i)) ||_2^2
L_transport(source) = mean_i || W_z (T_theta(x_i, delta_source_i, a'_i)
                                      - T_FP(x_i, delta_source_i, a'_i)) ||_2^2
L_theta = L_clean + lambda_T * L_transport(source)
~~~

`clean` 是 clean-only control，即 `L_theta=L_clean`；它仍使用同一个 optimizer/update/materialization budget 和同一 W4 freedom，不能通过额外 steps 或更多参数获得优势。`random_same_norm` 与 `frt` 使用相同的固定 `lambda_T`，唯一差异是 `delta_source`。不在 DEV 调 `lambda_T`，不根据 fit seed 选 lambda。

Stage A 在任何拟合前按 CAL 的 Q0 raw output 计算并报告两个 component 的量级；manifest 中的 `lambda_T=1.0` 只是已执行 A 的 provisional placeholder，不能作为 B 的 frozen value：

~~~
lambda_raw = max(mean(L_clean_Q0), 1e-12) /
             max(mean(L_transport_Q0), 1e-12)
lambda_T = clip(lambda_raw, 0.25, 4.0)
~~~

Stage A 必须保存 `mean(L_clean_Q0)`、`mean(L_transport_Q0)`、`lambda_raw`、clipped `lambda_T` 及是否触碰 clamp；若 transport component 很小，记录 `scope_warning` 和实际 component ratio。这属于解释限制或可能的 underoptimization，不能事后调参，也不能单独写成 FRT scientific no-go。B 启动前必须把 A 的 clipped `lambda_T` 写入新的 manifest revision；DEV 不能覆盖，也不能在 DEV 调整。

`W_z` 是 latent-unit 的 diagonal weighting，不是 latent-to-physical `xy` readout。仅从 CAL 的 FP next-slot outputs 计算 population variance；对 active observation/proprio coordinates：

~~~
std_j = sqrt(max(population_var_j, 0))
W_z,j = 1 / max(std_j, 1e-3)
~~~

`std_floor=1e-3`、population estimator、active mask、`W_z` 全部写入 A raw output 并冻结。action coordinates 不参与 loss，`W_z[action]=0`；active/action mask 必须互斥且覆盖完整 slot。不得用 DEV variance 重新估计或把 `W_z` 解释成 physical coordinate transform。

`random_same_norm` 在相同的 active `W_z`-whitened space 生成。当前 runner 用 CPU `torch.Generator` 为每个 split 生成一次 standard-normal bank（CAL seed `940000`、DEV seed `940001`），再逐 record 令 `||W_z r_i||_2 = ||W_z delta_q0_i||_2`；action coordinates 置零。随机 direction bank 只生成一次并由所有方法共享。q0 weighted norm 为零的 record 不能被静默改成其他 norm；A 必须把它标为 invalid/inconclusive 并停止方向机制结论。

## 5. Common-bank DEV 设计

DEV 在拟合前只收集一次 fresh Q0 bank，并从同一 bank 生成一次 random bank。它们的 record keys、`X`、`a`、`a'`、RNG、`W_z` 和 target fingerprint 对所有 evaluator 完全相同。

拟合完成后，每个 `method × fit_seed` 只在同一 DEV records 上运行一次，收集一个 fresh Qtheta residual bank：

~~~
fresh:clean:1201 ... fresh:clean:1203
fresh:random_same_norm:1201 ... fresh:random_same_norm:1203
fresh:frt:1201 ... fresh:frt:1203
~~~

最终 bank 顺序固定为 `q0`, `random_same_norm`，再按上述字典序排列的九个 `fresh:*` bank。每一个 evaluator（`FP32`, `Q0_RTN` 和九个拟合后的 method×seed）必须对 **所有** bank 计算 transport error；禁止只在自己的 fresh bank 上评分。Raw `transport_error[evaluator, bank, record, ...]` 必须是完整矩阵，Verifier 会检查 square bank/evaluator shape、finite values 和每个 bank 的 action mask。

主要比较在两个 shared views 上报告：

1. `q0` bank：FRT、Clean、Random 三种拟合 map 全部评分同一 frozen Q0 direction。
2. `fresh_union`：九个 fresh Qtheta banks 合并成一个预先声明的 union，所有 evaluator 全部评分这组 bank；不得按方法挑自己的 bank。

`random` bank 和各个 fresh bank 的 norm、cosine/alignment、record coverage 和 transport ranking 作为诊断报告。fresh Qtheta direction 明显偏离 Q0 时，应解释为 Q0-proximal fit 或 direction mismatch；若同时没有 fresh Q0/FRT transfer，按 no-go 处理。direction diagnostic 本身不被包装成 causal claim。

本轮不执行 DEV physical environment rollout、success 或 action-error comparison，因此不存在 Stage C 或 TEST endpoint。Stage A 可以记录 source/runner 是否能在独立 replay path 暴露 physical `xy`/goal-relative interface，但这只是 capability audit；没有实际 endpoint 输出时，不得补写 closed-loop consequence。

## 6. 预声明 mechanism gates 与 aggregation

对每个 `method × seed × episode`，先对该 episode 的所有 records 取 macro mean；再在六个 episode 上取 macro mean。三个 seeds 保留分布，不把 records、frames、bank pairs 或 candidate pairs 当作 independent `n`。设 lower-is-better。

Clean tolerance（相对 Clean control）预声明为：

~~~
FRT clean <= 1.10 * Clean clean + 1e-12
~~~

需同时满足：macro mean 通过；每个 seed 至少 `4/6` episodes 通过；至少 `2/3` fit seeds 通过。该规则是经济性 screen，不是 non-inferiority inference 或显著性检验。

Transport improvement 对 `q0` bank 和 `fresh_union` 各自计算，并分别与 `Clean`、`Random` 比较：

~~~
FRT transport <= 0.95 * comparator transport + 1e-12
~~~

每一个 comparator/view 都需同时满足 macro mean、每 seed 至少 `4/6` episodes、至少 `2/3` fit seeds；只有全部预声明 view/comparator 都通过，才记 `mechanism_gate_pass=true`。门槛附近（相对差异不超过 `1e-6`）记 `ambiguous`，不擅自升级为通过或 no-go。

这组 gates 的作用是验证 “Q0 direction 具有额外信息”，而不是单纯验证 FRT 能降低 CAL loss。若 FRT 只改善 CAL、不改善 fresh DEV；只在自己的 bank 上好；与 Random/Clean 在 shared views 等效；或 fresh Qtheta direction 与 frozen Q0 严重失配且没有 transfer，则在 engineering 完整后记 `mechanism_no_go`。若 fit runs、raw bank、hardening、endpoint、allocation 或 walltime 不完整，记 `resource_incomplete`/`underoptimized`，不能记 scientific no-go。

## 7. 资源计划与 stop rules

先做短 timing/stability refinement，再启动完整 B；实际只有一张 V100、QoS 为 `normal`、`cpu=20,gpu=1,mem=64G,maxjobs=2,maxwall=6h` 的条件可用。当前规划总上限为 **A+B ≤6 V100 GPU-hours**，包括模型加载、A collection、三方法×三 seeds fit、DEV fresh banks、I/O、失败和重试成本；不是运行时保证。建议预算记账：A ≤1.5 GPU-hours，B ≤4.5 GPU-hours。A 只能从 manifest 的候选 update/LR schedule 中按 timing、loss-component magnitude、scale stability 和完整 hardening 可行性选择；若九个 fit runs 加 common-bank DEV 无法在总 cap 内完成，应在 B 前停止或将新的 update count 作为显式 manifest revision 冻结，并把结果标为 underoptimization/resource decision。不能偷偷少跑 seed、少跑 method、缩短 soft schedule、减少 bank 或切换更大 GPU 来制造结果。

以下任一项成立，立即停止当前 pilot：

- A 无法证明完整 history、真实 shift/concat、新预测槽、同 history/action/RNG replay、action mask 或合法 `x_delta`。
- Q0/FRT map 不是 exact hard signed W4，clamp/scale axis/tie rule/hardening/reload 不能审计。
- CAL/DEV target 或 initial state 与 `0–59`/彼此重叠，或 fresh fingerprint 缺失。
- common DEV bank 不完整，或 evaluator 只评分自己的 fresh bank。
- 资源不足、deadline/SLURM allocation 失败、fit 未完成最终 hardening，或结果只剩 summary 而无 raw audit arrays。

前四项属于 engineering/interface failure；最后两项属于 resource/incomplete。只有完整 raw B 通过 engineering gate 后，Clean tolerance 或 shared Q0/fresh-union transport gates 失败，才写成当前 FRT 配方的 `mechanism_no_go`。此 no-go 只约束本次 fixed Q0/W4/Wall 配方，不推广到全部 decision-aware quantization。

## 8. Raw artifact contract（供 runner 与 verifier 对接）

Experiment directory 必须保留本文件和 `manifest.json`。当前 runner 的 run artifact directory 使用以下 compact JSON、method ledgers 和 compute-side NPZ；路径由 `manifest.artifact_contract` 冻结：

~~~
stage_a_summary.json / bank_stage_a.npz
stage_b_summary.json / bank_cal.npz / bank_dev.npz
methods/<method>_seed_<seed>.json
common_bank.npz
verification_a.json 或 verification.json
~~~

`stage_a_summary.json` 的 schema 为 `frt-stage-a-v1`，至少包含 runner 的 `runtime_identity`, `gates`, `backward_timing` 和 `safe_fit_steps`。`bank_stage_a.npz` 的 `metadata_json` 保存 episode/dataset mapping，且至少包含：

~~~
history[N,T,P,D], x_history[N,T,P,D]
action[N,1,A], next_action[N,1,A]
fp_current[N,1,P,D], q0_current[N,1,P,D], delta[N,1,P,D]
fp_transport[N,1,P,D], random_delta[N,1,P,D]
random_transport[N,1,P,D], wz[D], metadata_json[scalar]
~~~

Verifier 从 `q0_current-fp_current` 重算 `delta`，检查 `x_history` 是历史窗口移位后追加 `fp_current`，并逐 record 检查 random 与 Q0 的 `W_z`-weighted norm。当前 bank 只保存 `[D]` Wz；若 runner 同时保存 explicit action mask，Verifier 还会审计 action coordinates 为零。仅有 aggregate scalar 不能替代这些 raw arrays。

`stage_b_summary.json` 的 schema 为 `frt-stage-b-v1`；每个 `methods/<method>_seed_<seed>.json` 必须保存完整 `steps`, `batch_size`, `lr`, `lambda_transport`, `rounding`, `fit_trace`、hard ledger 和 checkpoint 路径。`bank_dev.npz` 保存同样的 shared DEV Q0/random bank。机制 gate 还要求 `common_bank.npz`：它必须包含固定 `bank_ids`、`evaluator_ids`、`bank_deltas`、`active_mask`、`action_mask`、`wz`，以及所有 evaluator × all-bank 的 raw `clean_error`, `transport_error`, `clean_mse`, `transport_mse`。Verifier 会按 runner 的完整 `P*D` denominator 重算 scalar metrics。本轮不要求或接受 DEV closed-loop endpoint 字段。

~~~
bank_ids[B], evaluator_ids[E]
clean_error[E,N,1,P,D]
transport_error[E,B,N,1,P,D]
clean_mse[E,N], transport_mse[E,B,N]
~~~

Verifier 从 error vectors 重算 `clean_mse`/`transport_mse`，并检查它们与保存 scalar 的一致性。`B=11`，`E=11`，顺序严格按本节 Common-bank 定义；任何缺 row、重复 bank、method-only bank 或非 finite row 都是 engineering failure。NPZ 允许很大，但 verification summary 必须小于 64 KiB；summary 不得包含 raw arrays、credentials、private key、token 或远端登录信息。

CLI 固定为：

~~~
python verify_frt.py --stage a --artifact-dir ARTIFACT --manifest manifest.json --output verification_a.json
python verify_frt.py --stage b --artifact-dir ARTIFACT --manifest manifest.json --output verification_b.json
python verify_frt.py --stage all --artifact-dir ARTIFACT --manifest manifest.json --output verification.json
~~~

`--stage a` 只做 A raw audit，便于先取得小型 CPU allocation summary；`--stage b` 读取 B raw outputs，并在 artifact 中存在 `verification_a.json` 时核对 A 的 manifest/W_z identity；`--stage all` 先 A 后 B。除 `--self-test` 外，verifier 在打开 manifest、JSON 或 NPZ 前必须调用 `allocation_guard.require_allocation()`。

## 9. 解释边界

通过本 protocol 只能得到以下层次之一：`conditional_signal`（预声明机制 screen 在本 pilot 通过）、`mechanism_no_go`（完整 B 但机制门失败）、`engineering_fail`（接口/硬 W4/common-bank 错误）或 `resource_incomplete`/`underoptimized`（未完成，不能作科学 no-go）。即使 `conditional_signal`，也只能说本次 Wall one-step fixed-byte screen 支持进一步审查；不能说 FRT 是首次、普适或已部署的 quantization method，更不能把 fake/logical W4 结果写成 native compression/latency。
