# LpWM real-history / action-response mechanism gate

## 目的与边界

检验同一已训练 sparse `LinearDynamicsPredictor(mode=mlp_var), embed_dim=384` predictor 在 MPC replan 时加入真实近期视觉 history 和与之对齐的已执行 action blocks，是否改善固定未来候选的真实 task-relevant elite quality。这里检验的是已实现 predictor 的接口假设；不改 checkpoint、不重训、不改 loss/architecture，也不由机制结果直接推出 task-success 或 speedup。

源代码、checkpoint、planner 和任务配置固定在 [FREEZE.json](FREEZE.json)。远端 pinned source snapshot 不含 `.git` metadata；preflight 从 checkout 自带的 `SOURCE_COMMIT` marker 读取并报告实际值，不依赖 `git` CLI。此前正式 sparse 评估为 50 tasks、planning seed 99、H=5、300/30/30 CEM、最多10次 replan。其 21/50 仅描述那次 formal native harness；它不是本机制 gate baseline，也不与其他 harness 的成功率相减。

## 两种输入与时间对齐

- `cold-start` 完全复用 native 路径：每次 replan 只提供最新一帧真实视觉观测，以及当前相同的5个未来 action blocks。
- `real-history` 在有足够历史时提供真实帧 `o(t-10), o(t-5), o(t)`，以及对应的两段已执行 action blocks `[a(t-10), a(t-5)]`；之后接入与 cold-start 相同的5个未来 action blocks。每个 action block 覆盖5个 primitive env steps。predictor 仍为 visual-only，不把 proprio/state 加入输入。
- 初始 plan 没有真实过去，两臂必须保持相同 cold-start。history 帧从真实环境轨迹按 frameskip=5 决策边界采样；不能用相邻底层物理帧或重复当前帧补齐。

源码 rollout 的约定是：长度为 `n` 的观测历史输入 `n-1` 个对齐的已执行 action blocks，再接 H 个未来 blocks；总 action 长度 `n+H-1`。例如 history=3、H=5 输入7个 action blocks，rollout 返回8帧。输入中 action 按“由该帧预测下一帧”的因果约定对齐。history 输出必须裁为 `rollout[:, n-1:]`，成为 current+5 future 共6帧，才交给现有 objective/evaluator；只执行原5个未来 blocks。不能把历史 action 当 future 执行，也不能只加 observation 而不加两段历史 action。

接口门先于科学比较：history=1 必须与 official cold path 在 shape、当前帧/未来帧索引和 predictor 输出上相同；history=3 通过带时间标记 dummy predictor 核实过去 action 对齐、首个未来预测使用 A0、最后一个使用 A4，crop 后恰为当前帧+5未来帧。CPU dummy 只验证索引逻辑，不替代真实模型接口验证。GPU allocation 在 `PlanWorkspace` 创建后、native capture 前，还会用固定 task 0 的真实初始 `obs_0` 图像重复成 300×6 输入，调用真实 visual preprocessing/encoder，检查 `(300,6,384)` 输出与 finite，并恢复 torch RNG/model buffers、核对 module flags。这只验证接口，不代表真实 future prediction 的质量。

## 数据来源修正与冻结采集

已检查的 native 源码把 `planned_actions` 保存在 MPC 对象内存；`plan_targets.pkl` 只写出 `obs_0, obs_g, state_0, state_g, gt_actions, goal_H`，不保存 MPC 实际执行序列或 CEM candidates。已有 `logs.json` 是标量日志。专家 `gt_actions` 不能代替 native planner actions。

所以本任务不声称从旧成功/失败视频或 `gt_actions` 重放出了原 native replanning contexts。CPU allocation 内做窄目录/键核对；若无 action artifact，cohort 固定为原 formal 29 个 final-failure task indices 中按 ID 升序的前8个。该子批次是新诊断采集，不能要求复现旧 outcome，也不能按新机制结果补换 task。保留 planning seed 99 与每个原 50-task 0-based task index 的 `eval_seed=99*index+1`。

新 capture 记录零基 replan 1、2（只保留仍 active 的 task）的三帧真实历史、两段真实过去 action、当前 state/goal、CEM early/late candidate populations、late elites 和实际执行的 mean-sequence blocks。late 指该 context 原生早停前最后一次实际评估的 population。至少需12个 paired task-contexts、覆盖6个 task，且至少6个 task 不处于 regret at-floor，才作科学判断；不足记 incomplete，不作 NO_GO，也不按结果补 task。instrumentation 不得增加/移除 RNG 调用或改变 `eval_every=1`：每个 CEM iteration 对 mean sequence 的真实 env evaluation、全成功早停都保留并计入成本。每个 context 的 fixed bank 为300条 action sequences：iteration 0 按 `linspace(0,299,100,dtype=int)` 取100条 early proposals；从最后实际评估 population 的270个 non-elites 里按 `linspace(0,269,170,dtype=int)` 取170条；另按 native objective argsort 顺序保留全部30 elites。重复项和来源都记录。cost-only calibration 测 resource，不改变 bank 数；若超出 reviewed cap，停下请 root 决定资源，不自动缩小。candidate truth 必须从原始 init state+eval seed重放实际执行 prefix 后接候选，保留隐藏 Pymunk 速度；不能直接 reset 不含 block velocity 的7D anchor state。

## 指标与 primary gate

每个 candidate 的真实 task diagnostic cost 定义为

`(norm(goal_state[:4] - endpoint_state[:4]) / 20)^2 + (wrapped_angle_error / (pi/9))^2`。

其中前4个位置坐标是 official PushT evaluator 用于成功判断的 agent 与 T-block 位置；另单独报告 T-block object-only 2D 位置差。角度误差对应 T-block angle，并 wrap 到 `[-pi, pi]` 后单独报告。这是机制分析用的 diagnostic cost，不是 official success predicate；官方 success 按 `pos_diff<20` 且 `angle_diff<pi/9` 单独报告。不得使用混合7D norm。

primary 比较：在同一 frozen bank 内，以每臂 planner objective 排序，取模型选择 top-30 candidates 的**真实**平均 diagnostic cost；并报告相对 bank 内真实最佳30个候选的 elite regret。对同一个 task，先分别平均其可用 paired anchors 的 cold/history regret，再形成 task-level 差值。定义 relative reduction 为 `(regret_cold - regret_history) / max(regret_cold, 0.01)`，分母 floor 为0.01 diagnostic-cost units。若 `regret_cold < 0.01`，标为 at-floor，单独报告，不计入百分比改善分布或改善数。建议 support gate：非 at-floor tasks 的 median reduction 至少10%，且至少 `ceil(5/8*n_nonfloor)` 个 task 改善；8个非 at-floor task 时即至少5个改善。阈值需 root 在查看机制指标前审阅并冻结。

次级诊断为：(1) 同 frozen encoder/link 编码真实未来观测后的 H-step linked-latent rollout MSE/cosine；(2) 每 context candidate ranking；(3) 不另建 perturbation bank：以固定300-bank的 index 0（iteration-0 early proposal 0）为 reference，比较各 candidate 相对 reference 的 terminal linked-latent predicted delta 与真实 future encoder delta，报告 direction cosine 和 magnitude ratio；真实 delta norm `<=1e-8` 单列为 degenerate，cosine 不加 epsilon 造值；真实 delta 非零而预测 delta `<=1e-8` 时 ratio=0、cosine为 `null/NA`。它们用于解释 primary，不要求所有次级量同时改善才支持 primary。预测、encoder latent 或 replay endpoint 的 shape 错误/非有限值都作为技术/interface incomplete，不进入 NO_GO 计数。至少要有12个 paired task-contexts、覆盖6个 task，且至少6个非 at-floor task，才判 support/NO_GO；不足记 incomplete/inconclusive。anchor 因提前成功而缺失时不补值、不按机制结果另挑任务。8-task gate 是 exploratory mechanism screen，不估计总体成功率，不做显著性检验；candidate 不是独立样本。

真实物理 candidate endpoint 必须从原 task 的 frozen `state_0` 和对应 eval seed 开始，重放已经实际执行的 native action prefix，再接未来 candidate；或使用已验证保留完整 Pymunk dynamic state 的环境 clone。不能直接把7D anchor state reset 成新环境，因为它只有 agent x/y、block x/y、block angle、agent vx/vy，没有 block linear/angular velocity。对每个 context，重放后的 anchor observation/state 必须与 capture 侧车一致。Native CEM 的 `eval_every=1` mean-evaluation 语义另行保留并计入成本；其 reset-from-7D 只代表原 planner 的评估语义，不替代此处 task-truth prefix replay。

## 执行顺序与停止条件

1. 先运行唯一获准的 CPU-only preparation：guarded metadata/pickle-key inspection、递归列出 obs dict leaf shapes、验证 `state_0/state_g` 为有限 `[50,7]` 且 `goal_H=5`、与 `final_eval_target_states.json` 逐项 identity 比对、旧 per-task outcome/anchor 文件名摘要、窄层级旧 action/proposal filename check、确定性 dataset sample 重建与 env identity replay、deterministic dummy lag/action alignment selfcheck。此阶段不加载 checkpoint、不做 model benchmark。专家 GT 只用于验证原始 task/env identity，不作为 native planner 序列或 mechanism context。
2. GPU job `25577811.pbs101` 完成native capture和16个history1 parity contexts后，在候选维度断言处技术失败：pinned PushT packed action dim为10（`dataset.action_dim=2 × frameskip=5`），初版误假设15；未进入cost gate或质量评分。修复后的 `25578046.pbs101` 通过synthetic/native action-dimension gates和16个history1 parity contexts，在cost calibration调用generic `transform_obs` 时因visual-only true-future输入缺`proprio`而技术失败，仍未生成cost报告或质量评分。Root审阅后批准且已提交唯一接口修复job `25578290.pbs101`：真实future latent沿用 pinned `transform_obs_visual` native图像分支；GPU-entry interface probe 使用task0初始真实图像重复成300×6来验证预处理/encoder接口，不作为真实future质量证据。前两次错误都保留且不作为NO_GO；不自动追加提交、改bank或closed-loop。scheduler接受的实际资源为1 GPU/16 CPUs/110GB/90分钟；PBS源码请求为1 GPU/8 CPUs/64GB/90分钟，资源差异原因未核实。内部80分钟cap不变。
3. 修复后GPU runner仍先通过CPU identity gates，再按 native 路径 capture proposals/contexts 与保留的 mean-sequence environment evaluations；随后在第一个有效固定 context 上做一次300-bank全prefix physics replay、cold/history forwards 和真实未来 encoder cost timing，并先写 `cost_calibration.json`。估算包括已花的 load/capture/interface 时间、剩余15个物理 contexts、anchor 2 较长 prefix 的1.5倍保守因子、16组未保留的机制 model/encoder forwards、20%余量和300秒 stageout/teardown；超过80分钟内部上限则写 `INCOMPLETE_RESOURCE_LIMIT` 并停止，不计算机制质量指标，不缩 bank。成本门通过后复用首 context raw truth，再做机制评分。有效科学负结果记 NO_GO；接口或数据缺失记 incomplete。
4. 仅机制 primary gate 支持时，再向 root 提交 paired native closed-loop protocol 与资源申请。闭环须在完整相同50 task states/goals与planning seeds上只改变 history；实际 success 和 wall-clock runtime 分开报告。当前没有批准 closed-loop GPU 阶段。

任何后续 GPU entry point 都必须在重操作前检查真实 `PBS_JOBID`、hostname 非 login 且属于实际 `PBS_NODEFILE`；GPU telemetry 每5秒将 UUID、utilization、VRAM 追加到该 job 的 `job.log`。每个 context NPZ 保存 `eval_seed`、完整 executed prefix、past action blocks、history indices、current state 和原始 history observations，以支持从 init+seed 重放核对。最终 summary 在 `OUT/results/run_summary.json`，cost门报告先于机制质量输出。共享源码、venv、checkpoint、旧 artifacts 只读；新产物仅写本 experiment root。
