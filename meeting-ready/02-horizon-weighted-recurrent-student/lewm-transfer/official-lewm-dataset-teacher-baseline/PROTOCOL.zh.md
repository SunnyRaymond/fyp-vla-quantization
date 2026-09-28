# LeWM PushT：upstream-aligned teacher-only baseline

## 问题与范围

先核实官方 LeWM teacher 在 pinned upstream PushT dataset-driven protocol 下是否能超过 floor。这个独立实验只运行 teacher-only，不训练、不跑四臂，也不根据先前 random-reset 实验的 0/50 结果筛选任务。名称使用“pinned-upstream-protocol-aligned baseline”：执行遵循 pinned `eval.py` 的任务采样、预处理和 planner/evaluator 设置，但采用项目已验证的 official object checkpoint loader，且关闭无关视频输出。

## 冻结的官方设置

配置来自远端 staging 中的 `le-wm/config/eval/pusht.yaml`、`config/eval/solver/cem.yaml` 和 `eval.py`：`num_eval=50`、seed `42`、dataset `pusht_expert_train`、goal offset `25`、`eval_budget=50`、`world.max_episode_steps=100`、`swm/PushT-v1`。有效 rollout 上限仍是 50 步。PlanConfig 为 horizon/receding horizon/action block `5/5/5`；CEM 为 batch 1、300 candidates、30 rounds、top-30、variance scale 1、CUDA、seed 42。

严格复用 upstream 的 50-row selection：按所有 episode 的 `max(step_idx)+1` 算长度，以 `length - 25 - 1` 得到可用起点上限；只保留 `step_idx` 不大于该上限的 row；用 `numpy.default_rng(42)` 执行 upstream 原样的 `choice(len(valid_indices)-1, size=50, replace=False)`，再排序所选 row index。保留源代码的 `-1` 边界，不另行修正。结果目录中的 `selected_tasks.json` 在 rollout 开始前保存每条 `(row_index, episode_idx, start_step)` 与数据源元信息。

预处理严格按 upstream `eval.py`：分别对完整数据集的 `action`、`proprio`、`state` 拟合 `StandardScaler`，拟合前去掉该列中含 NaN 的行；把相应 scaler 复用于 `goal_proprio` 与 `goal_state`。`pixels` 和 `goal` 使用 ToImage、float32 缩放、ImageNet normalization、Resize 224。Policy 使用 pinned `WorldModelPolicy`；solver 从 pinned Hydra CEM config 构造。三个 `_set_state` / `_set_goal_state` callable 与 dataset、50-step evaluation budget 原样传给 `World.evaluate`。

## 唯一实现差异

upstream `eval.py` 的 `load_pretrained(cfg.policy)` 与此处 staging 的 `pusht/lewm_object.ckpt` 序列化对象格式/缓存路径不匹配：官方 HF cache 目录为空，且本实验不下载或改用 HF mirror。使用项目此前已通过 Stage 1/2 接口验证的 `run_lewm_recurrent_student.py:load_official_checkpoint`，从现存的 `stablewm_home/pusht/lewm_object.ckpt` 加载完整 teacher 对象；随后按 eval.py 设为 CUDA/eval、冻结梯度并启用 positional interpolation。它是 checkpoint loader 的唯一替换。runner 保留 upstream 的小型 `pusht_results.txt` summary 输出；`World.evaluate(..., video=None)` 是唯一 evaluator 输出层差异，用来避免写 50 个视频；episode success 向量与任务映射另存小型 JSON sidecar。

远端代码以 staging snapshot 提供，没有 `.git` 元数据。因此 freeze 中的两个 commit ID 仅作为 source provenance/reference；作业会检查导入 root、关键配置值、API 与文件位置，不声称运行时逐一验证了 commit identity。

## 结果与停止门槛

Primary outcome 是 50 个预先按 seed 42 选定任务的 episode success 数。必须有 50 个唯一 row、完整 50 个 episode outcome、有限数值 metrics，且 compute-node/source-root guard 通过。`successes >= 5/50` 只表示值得另行考虑用同一批任务开展 paired multi-arm 实验；`0–4/50` 停止并先诊断。该 5/50 是工程继续门槛，不是显著性检验。没有自动扩样、重跑、替换任务、换 checkpoint 或提交四臂作业。

保存每个 task 的 row/episode/start-step，即使 episode ID 重复，后续比较也须复用完全相同的 task list，并记录重复 trajectory ID、预冻结臂顺序；统计时按 source episode 聚类。只有达到工程门槛并另行冻结对应多臂方案后，才继续多臂实验。

## 资源与安全

单个 PBS 作业：经 `normal` route queue 提交，资源请求为 1 GPU、16 CPU、110 GB RAM、2 小时 walltime、内部 6900 秒 timeout。只读 `qstat -Qf` 显示 `normal` 路由目标包括 `gdev`；`gdev` 标记 `from_route_only=True`，允许 1 GPU 与最长 2 小时。此前一次直接对 `gdev` 的 qsub 被 ACL 拒绝、未创建 job，因此 PBS 使用授权 route 入口 `normal`，其资源请求匹配 gdev。GPU utilization 与显存每 5 秒写入该作业 `job.log`；退出 trap 直接停止并等待 telemetry PID。runner 在导入模型/数据依赖、加载 teacher 或打开 HDF5 前，要求真实 PBS_JOBID、hostname 属于 PBS_NODEFILE 且非 login/submit node。数据和 checkpoint 在 job-private cache 中仅以 symlink 引用，不复制、不下载、不解压。失败只保留证据，不自动重提。

## 方法来源

本 protocol 在结果揭晓前冻结问题、task selection、主要 outcome 与停止规则，并将 episode row 定义为配对单元；其程序性方法参考 Kassis et al., *Scientific Agent Skills*, arXiv:2609.00065 v2 (2026-09-02), <https://doi.org/10.48550/arXiv.2609.00065>。代码 staging provenance 见 `FREEZE.json`。
