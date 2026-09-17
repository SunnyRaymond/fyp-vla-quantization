# Q-ensemble coupled rounding：source / asset prior gate

**日期：2026-09-13**
**候选状态：`resource_blocked`（source path 已核验；未运行实验；novelty 未认证）**

## 1. 候选与边界

候选名称是 **Q-ensemble coupled rounding**：在 TD-MPC2 已有的 Q ensemble 上，只改变 W4 stochastic rounding 的随机数耦合方式。对五个现有 Q members 的对应 weight coordinate 使用同一个 uniform stream（`common-U`），与每个 member 使用独立但边际相同的 uniform stream（`independent-U`）比较。两者使用相同的 per-tensor/per-channel scale、bit-width、clipping、seed 数和量化算子；不增加 Q member，不改变 planner、loss、bit allocation、RankCal 或训练。

研究对象是“已有 ensemble 的 PTQ error covariance 是否影响实际 two-Q aggregation”，而不是把 quantization error 重新包装成 uncertainty estimator。`common-U` 与 `independent-U` 的比较只有在每个 member 的边际 rounding law 相同、同一输入/同一 Q-pair、同一 tensor granularity 时才可解释。

这是一个 mechanism screen，不是 deployment、training 或 closed-loop success claim。未来只评估 fixed cached input 上的 Q scoring；不为恢复原来的 uncertainty framing 而给 planner 增加 variance penalty。

## 2. 官方 source 的真实调用路径

使用官方仓库固定 source commit `e9f59321933cbc8e11a002b842adc7d4ffae8ff1`（本地 reading record 也记录了该 main revision）。关键原始文件如下：

- [官方 `tdmpc2.py`（pinned raw source）](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/tdmpc2.py)：`TDMPC2._estimate_value` 在 imagined latent rollout 之后调用 `self.model.Q(..., return_type='avg')`；`_td_target` 调用 `self.model.Q(..., return_type='min', target=True)`。
- [官方 `world_model.py`（pinned raw source）](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/common/world_model.py)：`WorldModel.Q` 的合法值只有 `min`、`avg`、`all`。`all` 返回所有 member；另外两者先执行 `torch.randperm(self.cfg.num_q)[:2]`，再对这两个 Q 做 `min` 或 `sum/2`。因此“random-subset”是实际的随机 two-Q 子集，而非五个 Q 的全量平均。
- [官方 `layers.py`（pinned raw source）](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/common/layers.py)：五个 Q 由 `layers.Ensemble([... for _ in range(cfg.num_q)])` 建立，并用 `torch.vmap` 批量 forward。源代码没有名为 `state_actionvalue` 的 API；对应的 state-action value 实现就是 `WorldModel.Q`。
- [官方 `config.yaml`（pinned raw source）](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/config.yaml)：默认 `obs: state`、`mpc: true`、`num_q: 5`、`compile: true`；screen 必须在 manifest 中明确设置 `compile=false`，并记录实际 override，而不是把默认配置和实验配置混为一谈。
- [官方 `common/__init__.py`（pinned raw source）](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/common/__init__.py)：单任务 `model_size=5` 使用 `mlp_dim=512`、`latent_dim=512`、`num_enc_layers=2`；未额外覆盖时 `num_q` 由 config 保持为 5。
- [官方 `evaluate.py`（pinned raw source）](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/evaluate.py)：evaluation 通过 `agent.act` → `agent.plan` → `_estimate_value`，checkpoint 由 `agent.load` 加载；这确认候选应留在原 planner scoring path。

因此，真实 source 事实是：planner terminal value 使用 live Q 的随机 two-Q `avg`；TD target 使用 target Q 的随机 two-Q `min`；训练时才通过 `return_type='all'` 得到全部 Q。不能把 `min` 描述成 planner 的 uncertainty penalty，也不能声称 planner 全程使用五 member variance。

## 3. 机制与可反驳预测

设同一个 cached state/action input 上第 (j) 个 Q member 的 FP value 为 (Q_j)，rounding 后误差为 (e_j)。相同边际的 common-U 与 independent-U 只改变跨 member 的联合分布。对 two-Q average，误差方差含有

\[
\mathrm{Var}[(e_i+e_j)/2]
= [\mathrm{Var}(e_i)+\mathrm{Var}(e_j)+2\mathrm{Cov}(e_i,e_j)]/4.
\]

这条式子本身不给出“common 一定更好”的结论：common-U 若使对应的 error covariance 为正，可能放大 avg 的 aggregate variance；若 member 的权重相位和输入敏感方向使 covariance 为负，则可能降低它。故候选的可检验命题不是一个纯代数优越性，而是：**在实际 TD-MPC2 Q members 上，rounding-coupling 改变的 covariance 是否会转化为真实 two-Q min 的 member-choice/rank stability，并且不损害 planner 使用的 avg scoring。**

预注册的方向性预测如下。

1. **`avg` planner path。** 在同一 Q-pair schedule 和同一 imagined candidate pool 上，`common-U` 的 avg error variance 应按实测 cross-member covariance 的符号变化；若 covariance 为正，common-U 不应被预先宣称改善 avg MSE。应用 gate 只接受 `common-U` 的 candidate-order disagreement 不高于 `independent-U`，并记录是否存在 covariance-driven trade-off。
2. **`min` aggregation path。** 当 member 的 FP Q gap 大于两种 rounding error 的可见尺度时，common-U 若确实产生正的 corresponding-error correlation，应减少该 two-Q `argmin` 的 accidental member flips 及由此造成的 min-score candidate-order flips；共同方向偏差也可能使两个 member 一起下移，因此必须同时检查 min absolute error，不能只报告 flip rate。
3. **反驳条件。** 如果在固定 input 上 common-U 没有产生预期的 covariance/sign pattern，或 min flip/rank 指标没有稳定改善，或 avg planner rank 明显更差，则 coupling 不是这个模型上的 load-bearing mechanism。没有 closed-loop rollout 时，不把任何分数差异写成 task-success gain。

`min` 的屏幕指标要标注为 live-Q aggregation diagnostic；官方训练 target 的实际 `target=True` 分支另行记录，不能把 live-Q 诊断冒充 target-training 结果。

## 4. closest prior 与区别

最接近的 primary prior 是 [Ex Uno Pluria: Insights on Ensembling in Low Precision Number Systems（NeurIPS 2024）](https://proceedings.neurips.cc/paper_files/paper/2024/file/f10ceee5c6979988f334058561cac89f-Paper-Conference.pdf)。它从一个 pretrained model 用 Bernoulli stochastic rounding 构造低精度 ensemble，目标是用 rounding diversity 做 classification ensemble，并讨论低精度 ensemble 的 memory cost；它没有在一个已训练好的 RL Q ensemble 内比较 member 对应 coordinate 的 common-U 与 independent-U，也没有 two-Q random-subset `min/avg` 或 planner candidate ranking。

另一个相邻 primary record 是 [Generation of Ensemble Perturbations Using Low-Precision Floating-Point Numbers](https://www.jstage.jst.go.jp/article/jmsj/103/4/103_2025-022/_html/-char/en)，其研究天气模型把 rounding error 当作 model ensemble perturbation，比较 reduced precision 运行与 conventional ensemble；它没有 TD-MPC2 Q aggregation 或跨既有 critics 的 coupling control。[SPEQ](https://ojs.aaai.org/index.php/AAAI/article/view/16839) 是 stochastic activation precision 的 self-knowledge-distillation training，也不是此候选。

所以 novelty 只能写成 **conditional / unverified**：`common-U` 本身属于 stochastic rounding/shared randomness 家族，generic novelty 有明显风险；本切面较窄的差异是“保留官方已有 Q members 和 random two-Q aggregation，控制跨-member rounding covariance，并以 planner avg 与 target min 的实际 path 做 decision-level audit”。这不是新 ensemble 建模方法，也没有完成 exhaustive prior search。

与当前 campaign 的两个旧边界必须分开：

- `ensemble-uncertainty` 的旧结论是 `structural_no_go`，因为原假设需要 planner 使用 epistemic variance/disagreement，而官方真实 path 只有 two-Q `avg`/`min`。本候选改问真实存在的 Q aggregate 是否受 rounding coupling 影响，**不重新引入 variance penalty**。
- `antithetic-rounding` 的 practical no-go 比较额外的 2-member W4 与单 member W8，成本/精度不划算。本候选不增加 member、不做 2W4 vs 1W8，不重复该成本结论；它只比较已有 members 的随机数联合结构。

## 5. source / asset gate

| 项目 | 已核验状态 | 证据与边界 |
|---|---|---|
| 官方 single-task checkpoint | online 可核验，local 未 staging | [官方 models page](https://www.tdmpc2.com/models) 说明 single-task 为 5M；[HF official repo](https://huggingface.co/nicklashansen/tdmpc2/tree/main/dmcontrol) 列出 `cartpole-balance-1.pt` 31.3 MB。 |
| 精确 checkpoint revision | online 可核验 | [HF commit `73a50e2719ed8258c72c7d1fefd23b781d66e35e`](https://huggingface.co/nicklashansen/tdmpc2/commit/73a50e2719ed8258c72c7d1fefd23b781d66e35e) 的 LFS pointer 为 `cartpole-balance-1.pt`，`size 31344610` bytes，OID `4919e562d7f22f41a11118d1db1a0ebcb0e2b4681fc7fc594d772f0d5940869b`。本轮没有下载或 hash 本地文件。 |
| task/config | source 可核验 | `cartpole-balance` 使用 state observation；官方 dataset table 给出 observation 5、action 1。[官方 dataset page](https://www.tdmpc2.com/dataset)；future manifest 还需保存实际 parser-resolved episode length、discount、compile=false 和 seed。 |
| small offline input | **未找到** | 本地 workspace 的 targeted inventory 只找到 TD-MPC2 reading/prior docs，没有该 checkpoint 或 state/action cache。官方公开 multi-task offline datasets 是 30-task 20 GB、80-task 34 GB，不是本轮可接受的 small offline input。[official dataset page](https://www.tdmpc2.com/dataset) |
| V100 / runtime | 未执行 | 不能在本机或 login/head node load model；未来必须在真实 SLURM compute allocation 先核验 `SLURM_JOB_ID`、actual hostname、owner/partition/node 和 V100，再做任何重 I/O/model load。 |

因此当前 gate 是 **`resource_blocked`**：source path 和 <500 MB 的单任务 checkpoint 是可核验的，但缺少已经存在且可 hash 的 small offline state/action input；不以 synthetic input、在线 env rollouts 或下载 20/34 GB dataset 代替。若后续仅准备 8 个 fixed state blocks 和 64 条 fixed candidate action sequences，应另写 manifest/hash 并重新打开 asset gate。

## 6. 未来最小 screen（仅在 asset gate 解除后）

预算目标是单张 V100、`<=15 min` workload；不训练、不 full rollout、不测 task success。使用 exact `cartpole-balance-1.pt`（31,344,610 bytes）和官方 `model_size=5` state config；保存 source commit、HF revision/OID/size、resolved config、quantizer version、每个 seed 的 coupling definition 和完整 weight-restore digest。

固定 8 个 disjoint cached state blocks；每块保存 64 个相同 candidate action sequences（native action dimension 1，建议 horizon 3）及对应 state/action manifest hash。所有 arm 共享同一 latent/input/candidate tensor、同一 FP32 reference、同一 Q-pair schedule 和同一 reduction order。只改 live `_Qs` 的 W4 fake/PTQ rounding：

- `FP32`：原始 live Q ensemble；
- `W4-independent-U`：每个 Q member 独立 uniform stream；
- `W4-common-U`：对应 member coordinates 共享 uniform stream；
- `W4-RTN`：matched deterministic rounding 作为 rounding-negative control。

每个 arm 先通过 `return_type='all'` 记录五个 member scalar Q，再在同一 pair schedule 上调用官方 `return_type='avg'` 与 `return_type='min'`；planner 分数另外用 `_estimate_value` 的官方 avg path、FP dynamics/reward、相同 candidate sequence 计算。不要修改 `WorldModel.Q` 语义；若为固定 pair 需要 wrapper，必须证明它只重放 `torch.randperm(... )[:2]` 的已记录 pairs，并逐项对齐官方 Q output。min 的结果只标为 live-Q aggregation diagnostic；若要覆盖官方 target min，必须同时单独声明 `_target_Qs` 的量化 transaction，不得隐式混入。

最小 raw evidence 应包含：`member_q_fp`、`member_q_independent`、`member_q_common`、`member_q_rtn`（block × candidate × 5）、每次实际 `q_pair`、`avg_scores`、`min_scores`、`planner_candidate_order`、`argmin_member`、`state/action IDs`、coupling seeds、arm names、source/checkpoint/config fingerprints、以及 no-op/restore records。summary 只存小型 gate 结果；完整 raw 远端保存。

## 7. 强 controls 与停止规则

screen 只有在以下全部成立时才可进入数值阶段：

1. exact source/checkpoint/config gate 通过：`num_q=5`、`return_type` 只为 `min/avg/all`、`torch.randperm(... )[:2]` 的 q-pair 被记录；FP32 no-op 和 arm weight restore exact；所有非-Q modules digest 不变。
2. same-input control 通过：所有 arms 使用相同 8×64 state/action blocks、相同 candidate order、same pair schedule、same FP32 reduction dtype；不得把 q-pair randomness 当 coupling effect。
3. marginal control 通过：common-U 与 independent-U 的每个 member 具有相同 scale/granularity/bitwidth/seed count 和相同 per-member marginal rounding law；`RTN` 作为 deterministic negative control；不得以不同 clipping/scale 取得“改善”。
4. 机制 gate 预先固定为：在至少 6/8 state blocks 上，common-U 的 `min` argmin-member flip rate 与 min candidate-order disagreement 均严格低于 independent-U（absolute margin `>=0.05`），且 common-U 的 `avg` planner candidate-order disagreement 不高于 independent-U；同时保存 cross-member error covariance 的方向，若方向不符合预测则标为 mechanism no-go，而不是改写假设。
5. 任何以下情况立即停止：缺少 small input 或 source identity；官方 call path 无法固定/记录 q-pair；common-U 仅改善 aggregate MSE 但不改善 min/order 指标；min 有改善但 avg planner rank 恶化；或结果只在挑选的 horizon/seed/block 出现。此时报告 `no-go` 或 `inconclusive`，不扩大 seeds、模型、任务或做 full rollout 挽救。

当前结论保持 **`resource_blocked`**，不是 empirical no-go。即使未来 gate 通过，也只能支持“在该 checkpoint、该 fixed input、该 Q aggregation path 上 coupling 改变 PTQ error/order stability”的条件性结果，不能外推为 epistemic uncertainty、通用 quantizer 或 closed-loop success 方法。

## 8. 本轮执行记录

- 仅进行 local source/document reads 与官方 web source inspection。
- 未下载 checkpoint/dataset，未读取 credentials，未连接或提交 cluster，未 load model，未进行本地数值实验。
- 本文件中的 empirical outcomes 均为空；novelty 尚未认证。
