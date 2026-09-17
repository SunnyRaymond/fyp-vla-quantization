# VQ 草案修订回应（2026-09-12）

本文件逐项回应 INDEPENDENT_IDEA_REVIEW_2026-09-12.zh.md 对 TR-PVQ 的 revise 判定。候选已收窄为固定 actual-byte ceiling、固定 group map 的 weight VQ 目标；它仍是待检验的 conditional research hypothesis，未认证 novelty，也不声称 canonical pipeline DONE。

## 1. 变化对象是 quantized model 的 predicted trajectories

固定的 action probe 只是输入，不是随量化改变的 Gram。令 e 为 episode initial state/goal，k 为固定 action probe，h 为 imagined horizon，P∈{F,Q} 分别为 FP32 DINO-WM 和量化模型。相同 initial observation、goal、action probe、planner random source 与 rollout/environment seed 在 FP/Q 间配对：

z^P(e,k,h) = rollout_P(z_e, a(e,k,0:h))。

随 quantization 变化的是 Q model forward 产生的 predicted latent/observation output z^Q；固定 action 序列自身不产生量化 signal。辅助 relation representation r^P(e,k) 从同一次 rollout 的 predicted visual/proprio outputs 取得，pooling、flattening、channel normalization 在 CAL 前冻结。它可以是 terminal output 或固定的 all-step concatenation，但必须和 implementation manifest 一致。

DINO-WM 的 planner score 不使用自定义 temporal-average latent distance。必须直接调用 reproduction/dino-wm-wall/source/planning/objectives.py 中 create_objective_fn(alpha, base, mode)，并让 CEMPlanner 调用完全相同的 objective_fn：

J^P(e,k) = objective_fn(rollout_P(o_e,a(e,k)), encode_obs(o_g))。

source 的 mode="last" 是默认路径，mode="all" 是另一个明确配置；alpha、base、shape broadcasting、goal encoding、CEM 的 sort/elite update 都冻结。当前 SCREEN anchor 是 H=5、topk=30、num_samples=300、var_scale=1、opt_steps=5；这是既有 screen 配置的 implementation anchor，不是本候选的实验结果，也不把 source 默认 opt_steps=10 写进结果。

## 2. goal anchor、Gram 表示与对角审查

g^r(e) 是同一 frozen source output coordinate 中从 goal observation 得到的 auxiliary target。定义

u^P(e,k) = r^P(e,k) - g^r(e)，

B^P(e,i,j) = u^P(e,i)^T u^P(e,j)，

G^P(e,i,j) = B^P(e,i,j) / (||u^P(e,i)|| ||u^P(e,j)|| + epsilon)，i != j。

B 是以共同 goal anchor 为中心的 unnormalized Gram，G 是 cosine-normalized diagnostic，主损失只使用 off-diagonal。G 的 diagonal 近似 1，不能说它是 goal score；B 的 diagonal 才是 ||u||²。absolute representation anchor 在共同 source output coordinate 中固定 rotation/translation，实际 planner score J 另作为直接 score anchor。

量化 fitting 目标为

L_abs = mean ||r^Q-r^F||² + beta_J mean (J^Q-J^F)²，

L_off = mean_{i != j} (B^Q(i,j)-B^F(i,j))²，

L_TR = L_abs + lambda_rel L_off。

这里 L_off 删除 diagonal：B diagonal 已由 r absolute anchor 间接约束，实际 CEM score J 已单列，G diagonal 又接近常数。若只改善 J 或 diagonal 而 held-out off-diagonal、first action 和 return 不改善，关系项不被算作独立贡献。B/G 是 auxiliary relation representation；绝不把它们写成原 CEM score 的定义。

## 3. VQ 更新和预算收窄

对预注册 weight block b，使用固定 additive/product lookup：

hat w(b,m) = sum_r C(b,r)[q(b,m,r)]。

d、R、codebook size、vector grouping、block map、index bits、codebook dtype 和 scales 在 calibration 前锁定，沿用 VPTQ/AQLM 可表示的 packed layout。先定义共同 actual-byte ceiling C_bytes；每个 variant 报告实际 bytes、codebook/index/metadata breakdown 和 slack C_bytes - bytes。不同 overhead 不得被称为天然 same bytes；超 ceiling 的 variant 淘汰或降低 storage 后重新登记。删除 rate allocation、dynamic precision、execution-state schedule、centroid reuse 和 custom accelerator。

首 pilot 固定 q indices，只优化 codebook entries，避免对每个 weight vector/codeword candidate 重跑完整 rollout。初始化为同 layout 的 VPTQ-like/local-MSE codebook。每个 codebook update 的流程是：

1. 当前 Q map 在 CAL paired probes 上做一次 batched forward，得到 predicted outputs、J^Q、r^Q 和 B^Q。
2. 固定所有 q，经 lookup/gather 与 Q rollout 直接对 L_TR 反向传播，用 Adam 更新 codebook entries；无需 STE，也不声称非线性 trajectory loss 有 block least-squares 闭式解。
3. 每次拟合最多 200 optimizer steps，K≤8 paired probes/batch、H=5、至少两个 fit seeds。先单卡 forward/backward smoke，再冻结全部必要对照都能落在总 ≤16 allocated GPU-hours 内的共同更新和数据预算。
4. 首轮 indices 始终固定；shared entry 的更新同时影响引用它的多个 vectors。没有逐 vector 完整 rollout 搜索，没有离散 assignment 优化主张。

若预算不足以完成公平对照或有效优化，记录 resource/optimization-inconclusive；不能用两次未收敛更新宣判方法无效。CAL 拟合，DEV 选择是否保留预注册 lambda_rel，TEST 确认冻结 map。

## 4. 近邻碰撞与 matched controls

RSAVQ（arXiv:2510.01240）使用 language-loss Fisher/Riemannian natural-gradient error direction 与 channel-curvature bit allocation；本候选固定 byte map，不使用 FIM projection 或 channel allocator。VPTQ/AQLM/QuIP# 提供 packed/index、additive 或 lattice/residual VQ 基础；本候选不重新声称这些 storage representation 新颖。QuantWAMs/QuantWM 分别提供 joint video-action Fisher/reachable-state schedule 与 world-model/planner sensitivity audit，本候选测试 codeword fitting 对 imagined candidate relation 的影响。VQ-VLA（arXiv:2507.01016）是 residual VQ-VAE action tokenizer，不压缩 backbone weights；VQVLA/MotionVQ（arXiv:2607.24148）是 motion-aware precision、centroid reuse、custom accelerator，本候选不复用这些路线。

relational knowledge distillation/task-aware VQ 是直接 collision family，不能以“没有 FIM”作为 novelty 充分条件。主比较为 FP32 reference、VPTQ-like local-MSE VQ、L_abs only、L_TR、same-target scalar、independent-codebook VQ。它们共享 C_bytes ceiling 和 inference graph，但实际 bytes/slack 单独报告；FP32 不是压缩 variant，也不伪称 same-byte。

数据按 CAL/DEV/TEST episode、initial state 分离，单一 Wall environment 不宣称跨任务泛化：CAL 只拟合 C，q 固定；DEV 只选择固定的 lambda_rel 或停用关系项；TEST 最后一次确认。episode/task/initial state 与 quantizer-fit seed 是单位，frames、horizon、probes、pairs 不充独立 n。common random numbers、同 task initial state 配对，至少不同 fit seeds 检查方向；小 pilot 不作显著性/功效承诺。

load-bearing variable 是 lambda_rel，包括 0。必须有 goal-score/diagonal-only、stratified pair-label shuffle、scalar、independent-codebook 与 same-byte random/index perturbation controls。若 L_TR 不超过 L_abs、local-MSE/VPTQ-like，或 shuffle 等效，或 scalar/independent-codebook 同样有效，或关系项只改善 CAL/intermediate geometry 而不改善 held-out action neighborhood、first action、return/success，则 no-go 或降级。

## 5. FP32、fake/native 与资源边界

DINO-WM Wall local reference 是 FP32；FP32 是 numerical baseline。codebook/index 可用 FP16 storage，但 round-trip 与 conversion error 单列；另行 BF16/FP16 native conversion 是新 baseline。fake/logical quant 只能支持数值/planner 行为，不能推断 native bytes、VRAM 或 latency。native claim 需真实 packed loader/kernel、完整 checkpoint bytes、peak VRAM、同步 throughput/control frequency，并与 FP32 及另行定义的 conversion reference 对照；无 custom accelerator 证据不报告 MotionVQ 类 speedup。

最多 4×A100 是并发上限。codebooks-only pilot planning cap 约 8–16 allocated A100-hours（仅资源规划推测，未测量；40GB 保守 batch，80GB 独立可选配置），若达到 cap 或 rollout/batch 超限就停止扩展。未来重 I/O、模型加载与计算必须在真实 PBS allocation 检查非空 PBS_JOBID、非-login hostname 和 GPU allocation；本轮未执行 SSH/PBS、模型、实验或 native kernel。新增 paid connector/API budget 假设为 $0。

## 6. pipeline 偏离和当前结论

phase0 使用 actual bundled connector 做了小 pool；arXiv/OpenReview 因依赖缺失跳过，Semantic Scholar 返回 429，OpenAlex 提供主体结果；host refs 解析出 4 篇，VQVLA 与 Where Bits Matter unresolved。bulk weak-relevance tagging 为人工精简表，fulltext pool 在旧 ResearchStudio environment 中无输出后中断，未伪称完成。RSAVQ official HTML、DINO-WM local source、QuantWAMs local fulltext、VQ-VLA/VQVLA local notes 是关键近邻证据。

phase1/phase2-select 是依照已有产物手写的最小记录；phase3 collision retrieval、canonical navigator、完整 fulltext sentinel 未完成。audit_input.json 会携带缺项、直接碰撞词和 stop conditions 给 fresh-context 独立审查。当前只能称“待审查的条件式 TR-PVQ 假设”；仅当 collision review、matched controls 与 no-go 条件通过后才可进入小 pilot，不声称现已 novel、已有效或已 native deployable。

最终补充：ordinary RKD-style VQ 与 QATMA/TPSD（arXiv:2603.05964，旧检索误标 CR-QAT）是直接碰撞。对照使用相同 trajectory/goal anchor、bytes ceiling 与更新预算；若等价，仅保留领域适配假设。独立审查见 ../FINAL_IDEA_REVIEW_2026-09-12.zh.md；上述更新预算修正由 root 在审查后补齐，尚无实验。
