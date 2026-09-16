# 独立 idea 审查（2026-09-12）

本审查按 `ponytail` 的最小完整原则执行，读取了 `EXISTING_IDEAS_REVIEW_2026-09-12.zh.md`、`PRIOR_SCOUT_2026-09-12.zh.md`、`FEASIBILITY_AUDIT_2026-09-12.zh.md`，以及 `closed-loop-quantization` 的 Phase 1、`TENTATIVE_MECHANISM.zh.md`、`STATUS.zh.md` 和 Phase 0 近邻全文。没有运行模型、实验或集群任务。`vq-action-geometry` 目前只有 Phase 0 和 `STATUS.zh.md`，因此对它只审当前草案，不把它当成已完成的 canonical candidate。

## 判定

| 分支 | 当前判定 | 是否认证 novelty | 关键原因 |
|---|---|---|---|
| `closed-loop-quantization`：STRC/TRFC（当前 Phase 2 candidate） | **abandon** | 否 | 当前 `A(q)` 用自然 q→FP state drift，`A_pi(q)` 却用 FP path 注入 shuffled action residual；两者不是同一 intervention，`Δ_coh` 无法归因于 temporal order。此前 `K_r`/sign、phase alignment 与 ordinary terminal-drift 风险仍在。后续 Frozen-Residual Transport PTQ 是新机制，需另审。 |
| `vq-action-geometry`：TR-PVQ | **revise** | 否 | action-trajectory Gram 对 VPTQ/AQLM/RSAVQ/MotionVQ 有潜在实际差异，但只有一句机制摘要；Gram 的旋转不辨识、goal anchoring、relational/task-aware VQ 碰撞和 native kernel 均未关闭。 |

## non-VQ：Signed Temporal Residual Calibration

### 机制自洽性与数学审查

候选最有价值的部分是把量化模型的真实 paired FP/Q closed-loop residual 序列作为 calibration object，并用完整 map 下的 `D_H(q)` 选择 joint block，而不是把静态 local error 或单步 consequence 直接换名。若 `D_H(q)` 真的是对同一 physical state coordinate residual 序列先做带符号的折扣和，再取平方范数，则展开式中的 cross-lag inner products 保留了正负抵消；这一点不同于 `Σ ||e_t||²`。

但当前 `STATUS.zh.md` 仍首先定义 `K_r(ell)=E[e_{t+ell}r_tᵀ]`。这是配对轨迹上的 cross-covariance/association，不是 causal transfer；同一初始状态、timestep、任务阶段、动作幅度、随机数和策略反馈可以同时改变 `r_t` 与后续 `e`。若后续 allocator 使用 `Σ_ell ||K_r(ell)||²`，每个 lag 的平方还会直接丢失符号，不能支持 persistent-versus-canceling 解释。必须二选一：移除 `K_r`，把带符号的 `D_H(q)` 作为唯一 load-bearing score；或明确只声称预测性 association，并另设真正的 residual intervention 来测试方向。

`D_H` 本身只有在以下条件同时满足时才保留该 claim 的符号抵消：`e` 是固定 physical state/transition coordinate 中的向量；`W`、state units、`H`、`gamma` 在 map 选择前冻结；每条 FP/Q 轨迹的 action noise、environment seed 和 start state 配对；phase control 保持每一步 residual 的幅度、timestep 分布、任务/状态 strata 和 action magnitude，只改变时间顺序。现在 Phase 2 candidate 的 `A(q)` 来自自然 q 分支的 `d_l(q)=φ(s^q)-φ(s^FP)`，而 `A_pi(q)` 来自 FP path 上注入 shuffled `r°` 的 cloned replay；这不是同一 treatment path 的 original-order versus shuffled-order 对照，`Δ_coh` 因而混入自然量化 rollout 与注入 rollout 的模型/反馈差异。这个 load-bearing mismatch 需要重定义机制，故当前具体 candidate 判为 **abandon**，不能仅靠补一句实现说明放行。

“same-state simulator replay”也不够具体。需要写出 `r_t` 是同一状态同一随机源下的 Q action 减 FP action，`e_{t+ell}` 是哪一个可观测 physical transition/state difference，未来 quantized calls 是否继续执行，以及如何将单个早期 residual 与后续反馈效应分开。若 DINO-WM Wall 只能提供 pixels/latent 而没有稳定 physical state coordinate，不能把 latent coordinate 的正负称为 physical cancellation；应改用固定、可解释的 end-effector/object pose 等状态，或放弃该符号机制在此 anchor 上的 claim。

### 是否只是旧 single-site scoring 换量

当前“对一个 joint block 做 W4→W8 conditional swap，重新测完整 map 的 `D_H`”有机会避免旧 single-site scoring，但仅写出这句话还不够。必须固定 graph-derived group（例如一个 transformer/dynamics block 或预先声明的 coupled video/action block），记录每次在已有 swaps 下的 conditional reduction，并报告至少一个预注册 two-group interaction。若所有 swap 都从 all-W4 独立测量、或 interaction 近似为零而仍宣称 temporal joint calibration，实质仍是旧的单点 ranking 换了 score；若 score 被独立相加，直接判 no-go。

### 与实际近邻的差异边界

| 近邻 | 已核对的机制 | STRC 只有在什么条件下才真正不同 |
|---|---|---|
| DA-PTQ（`arxiv:2604.11572`） | virtual planar serial-chain Jacobian、trajectory-level motion surrogate、gradient-based layer ranking，以及 cross-space affine/low-rank compensation。 | STRC 必须使用实际 paired FP/Q residual sequence、signed cross-lag cancellation 和 phase intervention；“trajectory drift-aware”或另一个 Jacobian/motion magnitude 不能算差异。 |
| QuantWAMs（`arxiv:2607.28405v1`） | shared-basis calibration、joint video-action empirical Fisher、FP/Q reachable-state replay 和 fixed-budget denoising-step schedule repair。 | STRC 必须把 output object 定义成 residual temporal mode，并做 conditional weight/block map selection；复用 reachable-state replay 或只修 denoising schedule 会被其覆盖。 |
| Feedback World Model（`arxiv:2605.15705v1`） | 推理时读取真实 transition，更新 latent feedback state，并做 action-aware guidance。 | STRC 必须是离线、固定 quantizer-derived calibration map，不在部署时维护 observer；任何额外 delta-action accumulator 都会落入通用 observer/error-feedback 家族。 |
| MARR（`arxiv:2605.17997`） | module-specific residual reconstruction coefficient 与 PID feedback，平衡 cross-layer compensation 和 Hessian-approximation bias。 | STRC 必须测 physical closed-loop transition residual，而不是把 module reconstruction residual 加上 PID。 |
| RSAVQ（`arxiv:2510.01240`） | FIM/Riemannian error-direction guidance 与 channel sensitivity bit allocation。 | 与 non-VQ STRC 不同，但不能以“geometry/sensitivity”宽词宣称新颖；VQ 分支需逐项对照。 |
| VPTQ（`arxiv:2409.17066`） | second-order/channel-independent VQ、codebook/index，以及 residual/outlier quantization。 | 与 non-VQ STRC 不同；它是 VQ 基线，STRC 不得把 fixed low-bit fake quant 的 residual score 写成 codebook novelty。 |
| MotionVQ/VQVLA（`arxiv:2607.24148`） | execution-state motion-aware dynamic precision、merged-centroid vectorized GEMM 和 custom accelerator。 | STRC 不应声称 motion-aware VQ/accelerator；若转 VQ，必须证明 trajectory relation objective 与其 dynamic motion schedule 和 hardware path 分开。 |

因此 non-VQ 目前是“可修补的研究假设”，不是已认证 novelty。其贡献面应收窄为：在固定部署 map 下，能否用受控的 signed temporal residual object 预测 held-out closed-loop outcome，并在 matched resource 下击败同样的 magnitude/geometry baseline；不要声称发现 causal transfer kernel。

### 当前 candidate 的硬阻断与 no-go 条件

1. **定义对象：** 删除或降级 `K_r`；明确 `r_t`、`e_{t+ell}`、state coordinate、`W` 估计集、`H`、`gamma`、trajectory split、共同随机源和未来 Q feedback。
2. **同一路径对照：** 若保留 temporal idea，必须用 original-order residual injection 与 shuffled injection 共享同一 FP cloned path、snapshot、controller、state/action strata；自然 q drift 可另报 endpoint，不能与 `A_pi` 直接相减。这个修复已经改变当前 candidate 的 load-bearing estimand，不能把旧稿标为 revise 后 advance。
3. **定义 allocator：** joint groups 从模型 graph 预注册；每次重测完整 current map 的 conditional reduction；加入 unsigned-energy、sign-randomized、single-site/additive、random-map 和 same-bytes latency controls。
4. **定义 endpoint：** 主要判据是 held-out physical accumulated drift 与 task success/return，`D_H` 只能作为 calibration signal；FP/Q calibration、validation、test 按 episode/initial state 分离。
5. **定义 deployment：** 若只有 DINO-WM 既有 fake W4/W8 emulation，就只作 numerical PTQ screen；native packed bytes、peak VRAM、kernel latency 不能从 nominal bits 推断。

以下任一条件成立即 no-go：`Σ||K_r||²` 被当作 signed cancellation score；`A(q)` 与 `A_pi(q)` 来自不同 treatment path；shuffle 与 original 只改变 state/action alignment；`e` 无稳定 physical coordinate；若 `e` 是 transition increments 则 `gamma=1` 只等价于 terminal displacement；conditional swap 退化为独立 single-site/additive score；unsigned/sign-randomized control 与 STRC 相同；或 phase-shuffle 在 downstream outcome 上仍等效。

### ≤4×A100 可实施性

`DINO-WM Wall` 是最现实的首个 anchor：既有闭环 CEM 约 4.06 GiB、probe sampled device memory 12,820 MB，单卡可做 inference-only paired replay；但 checkpoint/dataset 不在本地，不能称离线可重跑。`Fast-WAM` 的 OSMesa + `first_frame` 单卡路线已验证，完整 `idm` 与联合 activation/backward 未验证；`OpenVLA-OFT Goal` 的 BF16 inference 有两次 2×A100 参考，但 calibration 显存风险明显更高。推荐先用 DINO 单卡建立 physical residual/phase-control gate，再决定是否 transfer 到 WAM/VLA。

4×A100 只代表最多并发卡数，不代表可以把全模型复制、backward、长 rollout 和所有 map 穷举都视为已可行。所有未来 heavy I/O、模型加载和计算都必须在真实 PBS allocation 中核验 `PBS_JOBID`、非-login hostname 和 GPU allocation；login node 仅做轻量控制。native claim 必须同时报告同 workload 的 packed checkpoint bytes、Peak VRAM、同步 throughput 和 FP/BF16 对照。

## VQ：Trajectory-Relational Product VQ

当前 `vq-action-geometry/STATUS.zh.md` 只有一句草案：在冻结 DINO-WM 上，用量化前后 imagined-trajectory 的 pairwise action-geometry Gram 作为 codebook/index fitting 目标，并沿用 VPTQ/AQLM 的 packed lookup。草案尚未给出 `phase2` candidate、数据流、codebook 更新、group/block 目标、fixed-byte map 或 falsification；因此不能 `advance`，也不能认证 novelty。

其潜在差异是把 weight VQ 的 fitting target 从局部 weight/output MSE 改成由 quantized imagined trajectories 产生的 candidate-action relation，而不是 VQ-VLA 的 action tokenizer。这一差异只有在关系目标随 quantized model 改变、并且 codebook/index 更新实际经过该目标时成立；对固定输入 action 计算 Gram 再拟合 codebook 只是离线 action scoring，与权重量化无关。

Gram 还存在不可辨识性：全局 orthogonal rotation 保持 Gram 不变，平移在 centered Gram 中也被消掉，但 planner 的 action coordinates、gripper sign、goal direction 和 dynamics 并不具有这些不变性。必须加入预先冻结的 goal/coordinate anchor（或明确证明下游 planner 对该变换不敏感），并将 relational-only、relational+anchor、local-MSE 和 VPTQ-like fitting 做同 bytes 对照。还应直接比较 relational knowledge distillation / task-aware VQ 这类近邻家族；本轮没有完成该碰撞检索，故不可认证“首次”。

建议删除泛化的 rate-allocation 叙事，预注册一个 fixed-byte budget 和固定 group map；否则它会同时变成 RSAVQ 的 sensitivity allocation、MotionVQ 的 dynamic precision schedule 和旧 mixed-precision ranking 的混合物。VPTQ/AQLM 的 packed lookup 格式也只是存储表示，不能证明 DINO-WM 在 A100 上有 native kernel。必须单独测 packed checkpoint size、实际 device memory、kernel throughput 和 action/return；否则只能称 fake/logical compression。

4×A100 上 DINO-WM 规模可作为 1×A100 pilot，但 imagined trajectories × codebook candidates × block coordinate updates 的数量要有 bounded screen 和 stop rule；40GB 与 80GB 应分开报告。VQ 分支在补齐 objective、goal anchor、collision retrieval 和 native/fake boundary 前，保持 **revise**。

## 新版 non-VQ：Frozen-Residual Transport PTQ

新版 artifact `closed-loop-quantization/phase2_generate/frozen_residual_transport_ptq.json` 和 `IDEA.zh.md` 已出现。当前判定为 **revise**，`novelty_certified=false`；它是可以实际检验的窄假设，但尚不足以 `advance`。相较 STRC，它删除了有问题的 `K_r`、跨 lag signed accumulation、phase injection、observer 和 mixed-bit allocation，改成固定 `Q0` 在 paired history/action 上产生真实量化 residual `δ`，再在下一预测槽上比较
`T_θ=F_θ(x+δ,a')-F_θ(x,a')` 与 `T_FP`。这是一个定义清楚的 finite-difference transport contrast；它没有声称 causal transfer 或严格 Jacobian，这一点是实质改进。

novelty 仍有一个集中且关键的风险：`δ` 只是一个由 Q0 产生的方向，而 loss 是对该方向的局部响应匹配。QDrop/input-noise、Sobolev/JVP/GAD sensitivity matching、PD-Quant 的 global prediction difference，以及普通 two-step unroll 已经覆盖了相邻原则。当前 random same-norm control 是必要但不充分：必须加入显式 Jacobian/JVP（或等价 finite-difference 多方向）对照，且固定相同 W4 参数自由度、CAL/DEV 预算和 endpoint。只有在 Q0 direction 在 fresh DEV 上持续优于 random 和 JVP 控制，才能称为 quantization-specific residual transport；若相同，则应 **abandon**，不能把方向选择包装成 WM novelty。

还需硬化四项定义。第一，`P` 必须是固定、可验证的 physical `xy/goal-relative` readout；若 `F` 只有 DINO latent，就把结果标为 latent diagnostic，不能使用 physical drift 结论。第二，说明 `x+δ` 在完整 history 的新槽、normalization、cache 和 shape 上确实是合法输入；不能只在向量层面注入。第三，拟合后的实际 `δ_θ=F_θ-F_FP` 可能偏离冻结 `δ_Q0`；DEV 需报告两者的 norm、cosine/alignment 和 transport ranking，明确该方法是 Q0-proximal 还是可迁移方向。第四，joint scalar scale/rounding 只有在预注册 two-block interaction `I_gh` 非零且跨 seed 稳定时才不是若干 local PTQ loss 的重命名。

候选已有 direct two-step、QDrop/noisy-input、random same-norm、clean reconstruction 和 PD-Quant-style controls，方向基本正确；应补 JVP/Sobolev/GAD control，并要求所有 baseline 使用同一个 physical readout、同一 `δ`/random δ、同 bytes。FRT 不携带 FP teacher、observer 或 online correction，这使它与 Feedback World Model 的 online residual observer、RPIQ 的 residual compensation/error-feedback、DA-PTQ 的 virtual Jacobian/motion surrogate、QuantWAMs 的 Fisher/reachable-state schedule repair 保持差异。AdaRound 只提供 rounding/materialization，不能算 FRT novelty。该差异仍是 conditional：本轮有限检索未认证“首次”，relational/sensitivity 近邻若出现 exact protocol 即停止扩张 claim。

实现上，`CAL=6/DEV=6/TEST=12`、3 seeds 和 ≤12 A100 GPU-hours 只能视为计划上限，不能当作已验证 runtime 或统计保证。DINO-WM Wall 的既有约 4.06 GiB CEM/约 12,820 MB probe 支持单卡 inference-only screen，但 checkpoint、dataset、FRT interface 和 native W4 backend 尚未由本审查验证。应先做轻量 interface gate，再在真实 PBS allocation 内运行；只有 `PBS_JOBID`、非-login hostname 和 GPU allocation 同时满足时才可模型加载、重 I/O 或计算。40GB 与 80GB 分开报告，logical/fake W4 不推出 packed bytes、Peak VRAM 或 latency。

FRT 的 no-go 条件是：physical readout 或完整 history 槽不成立；random same-norm 或显式 JVP 在 fresh DEV 与 downstream endpoint 等效；只改善 CAL 而不迁移到 fresh Q0；最终 `δ_θ` 与冻结方向严重失配却仍作泛化声明；two-block interaction≈0；或性能只来自 clean MSE/parameter freedom。上述任一项成立就 **abandon FRT**。

## 给 root 的执行结论

当前 STRC Phase 2 candidate 标 **abandon**；它不能通过局部改名继承到 FRT。新版 FRT 标 **revise**，但不认证 novelty，必须先通过 Q0-direction 对 random/JVP 的 fresh-DEV 与 downstream 区分门。VQ/TR-PVQ 标 **revise**；若 Gram anchor、relational-VQ collision 或 fixed-byte/native boundary 无法关闭则 **abandon**。不因交付两个 idea 而放行任一分支。
