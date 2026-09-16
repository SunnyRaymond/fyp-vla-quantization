# Goal/reference latent coordinate × PTQ：prior/source gate

日期：2026-09-13。范围：只读核对本地 pinned DINO-WM source 与两条 QuantWM/DINO-WM primary prior；未加载模型、未读取数据、未做本机数值、未连接或提交 cluster。该文件是 prior/source gate，不是实验 protocol。

## 判定

主机制判定：**`identifiability_no_go`，GPU=0**。四臂可以测出“同一 fixed encoder PTQ 对 planning objective 的双侧条件交互”，却不能在当前模型与当前 arm 约束下识别“goal/reference 与 current 共享量化误差的 common-mode cancellation”。

更窄的诊断命名可以暂时保留为 `goal-current view of encoder-PTQ interaction`，但不能称为新的 cancellation method、planner 修复或 deployment recipe。novelty 只做 targeted coverage，记为 **`insufficiently_differentiated / unresolved`**，不宣称完成 exhaustive scooping。

## 实际 goal/reference 路径

本地 `plan.py` 的 `PlanWorkspace` 先建立 dataset/environment/file goal。`goal_source=dset` 时，`sample_traj_segment_from_dset` 取验证轨迹片段并由环境 replay 得到 `obs_0` 与 `obs_g`；`random_state` 由环境分别 `prepare` 初始/目标 state；`file` 直接读取预先保存的两组 observation。它们之后共享同一个 `Preprocessor`（`plan.py:144–152`）。

真正进入 planning objective 的路径在两个 planner 中是：

- `GDPlanner.plan` 对 `obs_0`、`obs_g` 各做一次 `transform_obs`，先调用 `wm.encode_obs(trans_obs_g)` 得到并 detach goal latent，再将 transformed current observation 送入 `wm.rollout`；见 `planning/gd.py:78–98`。
- `CEMPlanner.plan` 同样先单独调用 `wm.encode_obs(trans_obs_g)`，然后对每条 candidate 用相同 transformed current observation 做 `wm.rollout`；见 `planning/cem.py:71–113`。
- `VWorldModel.rollout` 以 current `obs_0` 和第一组 action 编码为 `z`，再反复调用 FP predictor；见 `models/visual_world_model.py:284–309`。`encode_obs` 同时经过 visual encoder 与 proprio encoder，`encode` 再把 visual/proprio/action 拼接进 predictor input；见 `visual_world_model.py:91–134`。
- `objectives.py` 的默认 `last` objective 计算 predicted final visual/proprio latent 与 goal visual/proprio latent 的 MSE（`alpha` 加权 proprio）；这不是另一个 hidden score，而是当前 planner 已使用的 objective。

因此这里确实存在 current/goal 的两个 `encode_obs` 调用，但它们不是同一 tensor 的两次测量：输入 observation 不同，goal latent 还被缓存后与每个 candidate 的 FP predictor rollout 比较。source 没有独立的 goal encoder、reference normalization 或可学习 alignment head。

## 四臂能测什么，以及不能测什么

候选 arms 设为 `q_c,q_g∈{F,Q}`，其中 Q 只对冻结的 visual DINO encoder 做同一份 fixed W4 recipe，predictor、proprio encoder、action encoder 与其它模块保持 FP32。令 `E_F/E_Q` 是 visual encoder，`P_F` 是现有 FP predictor，则 planning loss 近似为

`L(q_c,q_g;a) = || P_F(E_{q_c}(o_c), a) − E_{q_g}(o_g) ||²`，

并可把 `L(Q,Q)−L(Q,F)−L(F,Q)+L(F,F)` 作为 2×2 interaction contrast。它能回答：在固定 current/goal pair 和既定 objective 下，同时改变两侧 encoder 的 loss 变化是否偏离单侧变化之和。

但“shared common-mode cancellation”需要更强的表示假设。写 `e_c=E_Q(o_c)−E_F(o_c)`、`e_g=E_Q(o_g)−E_F(o_g)`，则 current 侧的 `e_c` 先进入 **未量化的** `P_F`，goal 侧的 `e_g` 直接进入 target。两者只有在 `P_F` 对该表示扰动近似等变、且 `e_c` 与 `e_g` 在同一坐标变换下具有可比较的共同成分时，才可把 `L(Q,Q)` 的变化称为 cancellation。一般 DINO feature 的输入依赖误差并不满足这个条件；`L(Q,Q)` 变小也可能只是当前/目标两个不同输入的误差偶合、FP predictor 的 out-of-distribution response，或 MSE 对两侧共同坐标变换的普通不变性。

因此 current-only 与 goal-only 两个 arm 是有用的 mismatched-coordinate stress tests，不是 deployment baselines；both-shared-Q 是真正的 encoder-only quantized condition，但它单独不能证明 common-mode mechanism。若把 predictor 也量化或对 predictor 做等变 coordinate adapter，问题会变成另一条 intervention，且违反本候选的“predictor 始终 FP”约束。

## Local source facts 与工程边界

1. **同一 encoder 已是原方法结构。** DINO-WM 的 source/paper 将 current observation 与 goal observation 都送入 `enc`，并用 predicted latent 与 `z_g` 的 MSE 做 visual goal reaching；本地 `GD/CEM` 实现与该路径一致。因而“两侧应在同一 representation”是 baseline consistency requirement，不是本候选新提出的 mechanism。
2. **量化 locus 必须冻结。** 当前 `VWorldModel.encode_obs` 包含 visual 与 proprio 两个分支；若“encoder-only Q”连 `proprio_encoder` 也一起量化，就会在 objective 的 visual/proprio 两项同时改变，无法解释 visual goal coordinate effect。最窄且可比的定义应是 DINO visual `model.encoder` only；proprio/action encoder、predictor、decoder 均保持 FP32。这个选择仍需 execution 前用实际 module allowlist 证明，不能凭 state-dict key 猜。
3. **每 arm 必须重建两侧 latent。** goal latent 在官方 planner 中先编码并缓存；screen 不能从 FP arm 复用 goal latent 到 Q arm，也不能在量化后只重编码 current。每 arm 都应从 pristine weights、同一 transformed pair、同一 action/noise 重新构造，才能避免 stale reference 被误读成 cancellation。
4. **不能把 task score 当 mechanism target。** 四臂只应沿用当前 objective 与 fixed paired output，报告 visual/proprio latent objective 及 2×2 contrast；不把 FP goal、masked/quantized goal 或 environment success 叫作 ground truth。`goal_source=random_state` 还会引入 goal feasibility 混杂，若未来复核应优先使用固定 file/dataset pair。

## Closest primary prior 与差异

- [DINO-WM, arXiv:2411.04983](https://arxiv.org/abs/2411.04983)：明确以 pretrained visual feature 构建 world model，并把 visual goal reaching 写成 current/goal latent 的共享 `enc` 与 MSE 对齐。它支持本地 source interpretation；没有提出 PTQ 下 current-only/goal-only/both factorial，也没有 common-mode cancellation claim。该事实使“共享 encoder 是正确坐标”更像已有方法约束，而不是新 recipe。
- [An Empirical Study of World Model Quantization, arXiv:2602.02110](https://arxiv.org/abs/2602.02110)：直接以 DINO-WM 研究 encoder/predictor 的 PTQ sensitivity、long-horizon rollout 和 planning-objective alignment，并报告 encoder 与 predictor 的不对称性。它是最接近的 QuantWM prior；当前四臂的唯一潜在新增是把 encoder PTQ 的 current/goal side interaction 单独展开，但该 interaction 在 FP predictor 下仍缺少 common-mode 的可识别 control。它不是对本候选的 exact scooping 证明。

本次只执行两条 targeted primary query：`QuantWM arXiv 2602.02110 DINO-WM goal target observation encoder quantization planning objective` 与 `official QuantWM GitHub DINO-WM quantization goal target encode planner`，并打开 DINO-WM primary page 核对原始 goal path。覆盖有限；没有把搜索结果页、二手综述或未核对的项目标签当作直接证据。

## 机制否证与 conditional salvage

最强的否证是：在 same encoder weight map、same input pair、same FP predictor 下，`Q,Q` 的 loss 改变若不能超出由 current-only/goal-only 单侧变化及普通 MSE coordinate invariance 解释的范围，就只能记为 representation consistency diagnostic；不能叫 cancellation。若 FP masked/quantized reference 本身已显著移动，任何 apparent improvement 都应先归入 objective coordinate shift。

若未来仍要保留窄 diagnostic，至少需要在 source/protocol 层冻结以下条件：

- four cells 使用完全相同的 current/goal observation pair、preprocessing、proprio/action inputs、objective `alpha/mode` 与 candidate action/noise；只切换同一 visual encoder W4 snapshot 在 current call、goal call 的位置；
- 保存 `L(F,F)`, `L(Q,F)`, `L(F,Q)`, `L(Q,Q)` 及两侧的 raw encoder deltas `e_c,e_g`，同时报告 predictor input shift；不能只留 aggregate loss；
- 加一个 encoder no-op / exact restore 对照，并检查每个 arm 的 predictor、proprio/action encoder 与 non-target state unchanged；
- 若没有同一 state 下的 matched current/goal representation pair，或不能证明 `e_c/e_g` 可比较，则结果名称固定为 `input-side quantization interaction`，不得升级为 common-mode cancellation。

这只是识别门，不是新增实验矩阵。当前现有 Wall planner assets 没有为该作用准备独立 coordinate adapter、same-state latent transform 或 predictor-equivariance evidence；按本 gate 不申请 GPU。

## 最终状态

**`identifiability_no_go / GPU=0`；novelty `insufficiently_differentiated / unresolved`。** 本地 source 证明 goal 与 current 在同一 encoder representation 中比较是 DINO-WM 的既有规划结构；四臂可描述单侧 mismatch 与双侧 shared-Q 的 loss interaction，但在 predictor 保持 FP 的约束下，不能把它唯一解释为 goal/reference shared quantization common-mode cancellation，也不能推出 task success、planner robustness 或 native low-bit deployment。
