# 条件式 Stage 2 草案：upstream dataset 上四臂 paired evaluation

状态：已在 teacher-only baseline 结果出现前写定的条件式设计，当前不提交。只有 baseline 完整有效且成功数至少 `5/50` 才考虑执行；`0–4/50` 就停下来诊断，不再跑四臂。该门槛是工程 floor criterion，不是统计检验，也不用于挑任务。

四臂固定为 `student_only`、`late_teacher7`、`teacher_only`、`uniform_teacher7`。样本就是那一次有效 teacher-only retry 的 `selected_tasks.json` 中同一 50 行、同一排序与 `(row_index, episode_idx, start_step)`；启动失败的 `25531848.pbs101` 不含任务选择，也不符合 baseline 有效门槛。不重抽、不删行、不按结果挑 seed。数据集、全数据拟合的 `StandardScaler`、goal scaler、ImageNet/224 transform、upstream callables、goal offset 25、50-step `eval_budget`、100-step world hard cap、pinned CEM 配置都沿用 Stage 1。学生 checkpoint 固定为既有 `treatment_step1000.pt`，teacher 为现存 `pusht/lewm_object.ckpt`；不训练、不下载、不换权重。

沿用旧四臂的策略定义：late7 在第 24–30 轮调用 teacher；uniform7 在第 4、8、12、16、20、24、28 轮调用 teacher；teacher-only 每轮用 teacher；student-only 每轮用 student。每个 arm 使用一个 fresh World 完整评估同一有序的 50 个 task。arm 顺序以 seed `2609239000` 对 arm 名称执行一次 `random.Random(...).shuffle`，冻结顺序为 `student_only → teacher_only → late_teacher7 → uniform_teacher7`，运行前写入输出。

为保留 upstream teacher baseline 的 planner randomness，每个 arm 的官方 `CEMSolver` 都从配置 seed `42` 新建，并按 pinned `self.torch_gen` 的 native stream 连续消耗随机创新，不在 solve 之间额外 reseed。每个 CEM solve 记录调用序号、调用前 generator state 与 candidate-noise shape；同 ordinal 的 state 和 shape 若不同，common-innovation validity gate 失败，不能声称随机创新配对。只直接比较小型 generator state，不哈希大型 tensor。

主比较是 late7 对 student-only 的 episode success。按选中行中的 `episode_idx` 聚类；每 cluster 的 `D_e` 是该 expert episode 内所有 paired rows 的成功差之和。使用精确 cluster sign-flip：在零假设下独立翻转各 source-episode cluster 的符号，用整数动态规划统计 `|Σ sign_e D_e| >= |ΣD_e|` 的尾部概率。假设 source expert episodes 相互独立，且 sharp null 下每 cluster 的 paired treatment labels 可交换。若每个 source episode 只有一行，该检验退化为 exact two-sided McNemar；若有重复行，绝不把 row-level McNemar p 当成独立 episode 证据。

预设门槛：late7 比 student-only 至少多 `5/50` successes 且 cluster sign-flip 双侧 `p<0.05`；late7 成功数相对 teacher-only 不低超过 `5/50`（仅 practical gap，不是 formal non-inferiority）；CUDA-synchronized CEM solve 总时间 late7/teacher-only `<=0.70`，即至少快 30%。另外按臂报告完整性、source episode cluster 数、success/step progress、episode 与总 wallclock、teacher calls、CEM solve timing、峰值显存；uniform7 是同 teacher-call budget 次要对照。整体只在完整性、配对 innovation、三项 late7 gate 都通过时判 PASS。否则如实判 FAIL/INCONCLUSIVE；无自动扩样、重跑或换 task。

baseline retry `25534994.pbs101` 已终态 `F/Exit_status=0`，50 个 frozen tasks 中 teacher 成功 49 个，达到 `>=5/50` 工程启动门槛。四臂实现沿用本文件既定 tasks、预处理、CEM seed、arm order、统计检验和结果门槛；新增的 reproducibility gate 是 teacher-only 的 50 项 success vector 必须逐项等于该 baseline，否则整体 fail-closed 并报告具体 task mismatch。具体实现和提交状态记录在 `RUN_STATUS.zh.md` 与后续作业结果文件。

Stage 2 attempts `25535684.pbs101`、`25535692.pbs101`、`25535711.pbs101` 均为 `NO_RESULT`：前两项分别因 list/tuple arm-order guard 和错误模块的 `load_modules` 调用退出；第三项已加载数据和模型并进入首个 CEM solve，但 callback 缺 `.history` 导致 solver output assembly 失败。各 job 的小型证据均保留在 `artifacts/`。当前改为复用 Stage 1 验证过的 callback class；不改变冻结的科学设置，并按持续实验授权进行一次新的有界 attempt。
