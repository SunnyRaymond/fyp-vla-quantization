# LeWM + PushT 接口与可复用 runner

本文只记录真实接口、已有训练数据通路和 Stage B fixed-observation 诊断的复用边界；不代表本机运行过模型或确认远端文件当前仍可读。

## 已核实的 teacher 接口

冻结 teacher 来自 lucas-maes/le-wm commit 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac，配套 Stable-WorldModel CEM 版本记录为 10c26dbd5677083fa31dba69eb738b973845e9a4。冻结源文件清单在 meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/LEWM_RECURRENT_STUDENT_FREEZE.json 的 evidence_boundary.official_files：jepa.py、module.py、PushT eval config 和 stable_worldmodel/solver/cem.py。

本地 PBS probe 通过官方 JEPA.get_cost → JEPA.rollout 取得以下实测接口：

- Policy 初始观测历史 H=1；CLS latent 为 [B,1,192]，predictor 最多使用 3 个对齐时刻。predict 接受长度 1、2、3，latent 输入输出都是 192D。
- Planner candidate 为 [B,S,5,10]。每个 10D token 按时间顺序打包 5 个 PushT 原始 2D action：[x1,y1,…,x5,y5]。Policy info.action 本身为 [B,1,2] 原始 action；不要把它误当作 10D candidate token。
- 本次官方 rollout 返回 [B,S,6,192]，含初始 latent 和 5 个未来 latent。冻结的 predictor target 因而是 z(t+1)…z(t+5)。
- Teacher 每步用相同长度、最多 3 项的 latent/action suffix 调 model.predict(history, model.action_encoder(action_window))；取最后一个预测，latent 窗口左移并追加该预测，action 窗口追加当前 candidate token。Teacher helper official_teacher_targets 用可用的真实 latent 长度（不把 H=1 teacher 输入伪造为 3）；候选 action prefix 随预测递增。不得 teacher-force 未来 latent。
- 官方预测和 objective 分两层：student 只返回未来 latent；目标 goal_emb 只在外部官方 criterion 中使用。既有 scorer 构造 predicted_emb=[initial latent, 5 个 student predictions]，然后调用 official_model.criterion({"predicted_emb": ..., "goal_emb": ...})。新模型不能接收 goal，也不能把 teacher shadow score 混入 proposal update。

权威本地证据是 meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/interface-probe/interface_probe.py 与 interface_probe/artifacts/24554356.pbs101/interface_probe.json；人类可读摘要在 interface-probe/RESULT.zh.md。Probe 仅用了两个候选，不能当 CEM、planner 或闭环结果。

## Action history 与现有 baseline 的差别

run_lewm_recurrent_student.py::prepare_detached_rows 会把 action_history 保存在 row 中，但该文件的旧 LeWMCompactRecurrentTransitionStudent.forward(latent_history, packed_actions) 不读取此字段；旧 official_teacher_targets 则按候选 prefix 构造对齐 action 窗口。故旧 balanced_base 是有用的 recipe 对照，不能单独证明历史 action 处理完全匹配，也不能用它与新模型的差值纯归因到 block decomposition。

本目录新模型使用 model(latent_history[B,T,192], packed_actions[B,H,10], initial_action_history=None) -> [B,H,192]。Action history 只包含已消费的过去 token，不含当前待执行 token；最多保留两个 past token 与当前 token 对齐。H=1 时没有 past token，省略参数即可让模型按约定 zero-left-pad；每步只追加刚消费的 action，禁止读取尚未消费的 future token。该接口由新模型模块定义；训练 arm 间需统一使用此窗口规则。Probe 中为单独验证 rolling 行为而重复 padding 的例子，不是新的训练数据接口定义。

## balanced_base 与可复用训练通路

建议区分同结构的旧初始 recipe 与被本 pilot 指定的 balanced_base：

- 结构源是 meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/run_lewm_recurrent_student.py::LeWMCompactRecurrentTransitionStudent：192D latent、10D action、h256、latest-3 latent history adapter、shared recurrent transition、192D residual output、预测 latent free-running。无 attention、无 goal input。
- 当前 balanced_base 配对训练实现由 run_lewm_state_action_prefix_gru.py::instantiate_student/train_arm/run 复用上述 baseline 类；它从共同初始化重新训练 3000 updates，512 个 temporal-balanced contexts、每 context 64 candidates，目标为 horizon-weighted latent MSE 加 0.1 × context-normalized teacher-score SmoothL1（teacher top-12 权重 2，其余权重 1）。它不是主冻结文件中较早的 1500-step recipe；新 freeze 已固定后者不可混用。
- Paired loop 可复用 instantiate_student / train_arm 的共同数据、初始化、optimizer 和 minibatch schedule 结构；新 arm 的模型构造应来自 root runner 的 factory。Quality 先看同 observation/candidate slate 的 paired teacher ranking、teacher elite overlap/regret 与 latent 误差；latency 单独报告，不能代替质量提升。
- Temporal runner 的通用入口是 run_lewm_temporal_balanced_train.py::build_balanced_train_rows/train_arm，score loss 计算在 run_lewm_score_distill.py::_student_costs_with_gradient/score_distill_loss。

## 本 pilot 冻结的 bank/cache 来源

请遵守新 FREEZE.json 的 source，尤其不要改用另一份 temporal cache：

- Train bank：/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/teacher-screening/artifacts/25213164.pbs101/prepared_balanced_rows_reconstructed.pt。它由 25213164.pbs101 从 Phase 2 prepared rows 在 compute node 重建，按 freeze 指定使用。
- 上游 Phase 2 来源：同目录 state-coverage/artifacts/24926383.pbs101/context_manifest_512.json 与 prepared_512/prepared_rows.pt。本 pilot 使用该 manifest 核对 episode identity。
- 不要混用 temporal-balanced-train/artifacts/25152151.pbs101/prepared_balanced_rows.pt；虽然结构和部分配方相近，它不是此冻结选定的 bank provenance。
- balanced_base_step3000_reconstructed.pt 是 reference/provenance 文件；freeze 明确要求当前各 arm 从同一冻结初始化重训 3000 updates，不加载这个 terminal checkpoint。
- Runtime 资产默认在 /scratch/users/ntu/yguo017/lewm-pusht-iteration/stablewm_home：pusht_expert_train.h5 和 pusht/lewm_object.ckpt；代码 checkout 在同级 le-wm，Python 在同级 venv。路径是本地 PBS wrapper 记录的默认 remote path；此 turn 没有连接远端核验其当前存在状态。

Fresh evaluation row 的可复用结构见 ema-temporal-confirm/run_lewm_ema_temporal_confirm.py::build_fresh_rows：latent_history、action_history、future_actions、teacher targets/objectives、goal_emb、episode/anchor metadata。其 builder 会在 compute node 按固定 bounds 和 seed 生成候选、teacher latent 和外部 objective；Stage B 应从其中抽 freeze 指定的 8 个 episode middle rows，使用 common innovations 运行独立的固定 observation CEM 更新轨迹。Teacher scoring 是影子评价支路，不能改变 flat/block 的 proposal。

## CEM 语义证据边界

本地已有 fixed-observation CEM 参照 adaptive-teacher-schedule/run_adaptive_teacher_schedule.py：它的显式约定是 mu=0、sigma=1、候选 0 覆盖为 pre-update mean、torch.topk(cost, k=30, largest=False, sorted=True)、elite std(unbiased=True)、不 clip，并复用各臂同一 innovation bank。阶段 1 已运行的 official-pusht-cem/PROTOCOL.zh.md 使用 pinned CEMSolver.solve 原样更新；Stage B 仅是 fixed-observation 质量诊断，不代表整个 official MPC 生命周期。

最新实际取回的远端 source_reference/cem.py 已确认本 pilot 所用 native update：top-k 选择、PyTorch std correction=1（unbiased）、init var_scale=1、无 action clipping、无 variance floor。远端 source snapshot 没有 Git metadata；因此只记录读取到的 source reference，不声称通过当前 git HEAD 证明其与上游 commit 完全一致。Stage B 实现应从 FREEZE.contingent.stage_b_native_update 读取这些显式值，不以历史实验的 selector 文本代替本次 source。

## 执行边界

Runner wrapper 是复用路径；任何 HDF5/teacher/checkpoint/row/model/tensor I/O、训练、Stage B 推理和计时都只能在真实 PBS compute allocation。沿用 run_lewm_state_action_prefix_gru.pbs 的 PBS guard、远端 root 与 GPU telemetry 模式。Login node 只做轻量连接、提交和状态查询；不下载、不安装、不编译、不读取 checkpoint/HDF5、不做 tensor smoke test。
