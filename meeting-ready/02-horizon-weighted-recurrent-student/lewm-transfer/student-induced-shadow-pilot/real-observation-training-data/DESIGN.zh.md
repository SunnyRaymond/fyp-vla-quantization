# Student-driven real-observation latent data：最小采集设计

## 目标

为 `25239551.pbs101 treatment_step1000.pt` 收集真实环境监督样本：每条样本保存实际 solve-start latent `z_start [192]`、随后实际执行的 5 个 packed action tokens `[5,10]`，以及每个 token 执行后的真实 poststep latent `[5,192]`。动作 token 是 5 个经官方 action transform 的二维环境动作；poststep latent 来自同一 episode、同一条实际动作轨迹。

这与 `25542215/25542467` 的 16-task 误差诊断不同：那两项只保存 forecast-error 汇总，没有保存像素/action trace；本方案使用全新的 episode IDs，并落盘可直接训练的数据。所有来源 episode 都来自 HDF5 train partition，按 episode 再分为 64 个 collection-train IDs 与 16 个 collection-validation IDs。

## Episode 选择与隔离

身份由独立的 CPU PBS metadata-only selection 阶段生成并冻结；本设计不猜写 episode IDs。候选为 `ep_len >= 26` 的 HDF5 train episode。按 `random.Random(20260925)` 对排序后的候选 ID 洗牌，排除下述集合后取前 80 个；对每个 episode 用同一 RNG 在 `[0, ep_len - 26]` 均匀抽取 start step，并验证 row/episode/step 元数据。前 64 个为 collection-train，后 16 个为 collection-validation。生成的 ID 清单必须先单独保存并冻结，之后才允许启动 capture；不得补抽、替换或跨 split 移动 episode。

选择时按来源 manifest 读取并取并集排除：

- `SEEDED_PILOT_FREEZE.json` 的 16 个 `collection_tasks`（即 25542215/25542467 的旧诊断集）。
- 同一文件的全部 16 个 `reserved_holdout_tasks`，包括旧 ranker 使用的 8 个 validation episodes。
- `onpolicy-fullbank-ranker/results/25538135.pbs101/collection_manifest.json` 中 `split=validation` 的 episode IDs；核对它们都已包含于前述 reserved set。
- `onpolicy-fullbank-ranker/FREEZE.json` 的 `task_split.untouched.tasks` 8 个 IDs；继续完整保留给未来独立 gate。
- `SELECTION_FREEZE.json` 中已有 closed-loop-50 排除集，以及按该 freeze 从 `ep_len >= 26` 重建的 prior valid-prefix-600 排除集，避免复用早先 baseline/context-selection episode。

若任一源清单缺失、schema/数量不符、排除集交叠关系异常、可选 episode 不足 80，selection 失败并停止；不放宽排除条件。每个 ID 只能出现于一个 split。16 个旧 reserved IDs 和其中的 8 个 untouched gate IDs 不会参与本次 collection、训练或 validation。

## Capture 协议

复用现有 real-observation probe 的 `RealStepCapture`、`pilot.run_episode` seeded path 与 official H=1 image encoder 接口。固定 student checkpoint、官方 LeWM/PushT pinned code 和 evaluator 语义；每个 episode reset seed 为 42，native CEM seed 为 42，沿用同一 student-only CEM 与原始 planner 参数，在全局 step 0、25 两个 solve 边界各采一条最多 25 env-step 的样本。每个 solve 使用该时刻真实 pre-action pixel 计算 `z_start`，记录 25 个真实执行的环境 action，并在第 5/10/15/20/25 步后记录实际 poststep pixel；编码为 5 个 packed action tokens 和 5 个 poststep latents。只写 H=1 observation embedding，不向 encoder 输入 action。

动作必须从实际 `env.step` 调用捕获，再通过官方 `process['action']` transform 并按每 5 个二维动作打包；逐 token 对照原 student CEM plan，最大绝对差须 `<=1e-5`。pixel 与 action 在同一个 post-step callback 中配对，因此监督目标不依赖 HDF5 的 action-to-pixel 行偏移。HDF5 仅用于选取 episode/start state/goal 与官方 evaluator 初始化。

不计算 teacher rollout、teacher candidate cost、teacher ranker labels 或 teacher shadow。Official frozen H=1 encoder 只把真实像素映射到 latent target；这不是 teacher forecast supervision。Student CEM、候选评分和实际动作执行全程不变。

每条 row 记录 `episode_idx`、split、solve start、seed、相对 env-step indices、`z_start`、实际 packed actions、poststep latents、plan/action 对齐误差及完成/终止状态。只保存训练所需 float32 数组与必要元数据，不保存原始像素或教师输出。terminal 导致的缺失 row 如实记录，不替换 episode；训练/validation 的统计按 episode 聚合，两个 solve window 不视为独立 episode。

## 样本量、成本与边界

80 个独立 episode 最多得到 160 条序列：约 128 train、32 validation；每条含 5 个 poststep latent target，即最多 640/160 个 horizon targets。单条 float32 tuple 约 4.8 KB，总数据约 0.8 MB（不含少量元数据）。这是一次最小 fine-tuning 数据可行性试验，不足以支撑 broad generalization 结论；validation episode 只用于后续单独冻结的训练诊断，不进入更新或 checkpoint 选择规则，除非另行预注册。

`25542467` 的 14-task capture 约 5 分钟；按 episode 数线性估计，80 个新 episode 约 `80/14 × 5 = 28.6` 分钟。后续 capture 预计申请 1 GPU、16 CPU、110 GB、40 分钟 PBS allocation；GPU 型号/实际时间以作业为准，若使用 GPU，每 5 秒将利用率和显存写入 job log。所有 HDF5、checkpoint 与模型读取/推理都在获批 compute allocation 内完成。此文档不提交或运行 PBS，不进行训练、CEM 变体或 holdout gate。

## 可复用性

可复用 `real-observation-rollout-probe/run_real_observation_rollout_probe.py` 的真实 step/pixel 捕获、seeded `pilot.run_episode` 路径、student checkpoint 装载和 `encode_h1`。现 runner 会加载 official model 并在汇总阶段做 teacher forecast 比较，且明确不写 raw action/pixel traces；新 runner 应保留用于 H=1 编码的冻结 official encoder，移除 teacher forecast/cost/shadow 路径，并在 capture 中直接写实际 action-to-poststep-latent rows。无需复用或重跑旧 16-task 结果。
