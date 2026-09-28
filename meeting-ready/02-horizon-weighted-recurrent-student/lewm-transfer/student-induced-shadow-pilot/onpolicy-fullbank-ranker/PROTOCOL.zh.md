# On-policy full-bank residual ranker：冻结协议

本实验只检验一个 planner-facing predictor 问题：在 student-driven 的 PushT CEM context 上，一个小型 per-candidate residual MLP 能否改善 full 300-candidate bank 的 teacher-elite ranking。使用 [FREEZE.json](FREEZE.json) 中的模型、任务 split、loss、预算和 gate；结果出来后不改阈值、不换 episode、不做 sweep。

## 数据与隔离

冻结 control 为 `25239551.pbs101` 的 `LeWMCompactRecurrentTransitionStudent h256`。训练集精确复用 seeded pilot freeze 中的 16 个 `collection_tasks`；验证集使用其 `reserved_holdout_tasks` 的前 8 个；后 8 个保持未触碰。验证集只在 1000 次更新完成后用于一次 predictor-level gate。任务 ID 不跨 split。这里保证的是 task-level isolation：官方 dataset-wide scaler 会读取完整 train columns，包括这些 reserved validation rows，因此不声称 row-level preprocessing isolation。

25537667 的 pilot costs 不能作 ranker 训练样本，因为它没有保存候选 action tensors 和 context embeddings。新 collection 必须在冻结的 base student 下，按 reset seed 42 运行训练与验证任务，在 t0、以及恰好执行 25 个真实 transition 后可用的 t25 replan，记录 CEM 第 10、20、30 轮的完整 300 候选 bank。每个候选需对齐保存 `[5,10]` future actions、student objective cost、official teacher objective cost、当前 `initial_emb[192]`、`goal_emb[192]` 及 task/state/round/candidate 身份。教师只为这 300 个候选离线打标签；不做 K-screening。

训练与验证都使用 baseline student 生成的 banks。为取得 t25 context，验证 episode 也由**冻结 baseline student** 执行真实环境动作；第一阶段不把 treatment score 接入 CEM，不执行 treatment 动作，也不以任务成功作为本阶段 gate。验证 bank 上的 MLP 输出只作 post-hoc scoring。t25 未到达的 episode 保留并记录终止状态，不替换；至少 6/8 个验证 episode 有 t25 才能继续判定 gate。

## 非干扰 gate 与训练

训练任务的前四个 frozen task 做 recording/shadow on-off 配对。比较初始 observation、reset 与 solver RNG、每次 CEM solve 的候选数组、实际动作和 post-step state trace、solve 数及 episode 终止状态；四对必须全部 exact pass，之后才处理其余任务、训练或验证。任何 mismatch 都使本轮 invalid 并停止，不替换任务。

control 使用冻结 student 的 bank-z-normalized cost。Treatment 冻结 student，只训练 `435→256→256→1` 的两层 ReLU residual MLP，输入为 `initial_emb[192]`、`goal_emb[192]`、flattened future actions `[50]` 和 bank-z-normalized student cost `[1]`；末层零初始化，初始 score 与 control 相同。每个 bank 以 teacher objective 排序，将 teacher top30 定为 300 候选上的均匀目标分布；对 `softmax(-corrected_score/0.5)` 计算 listwise cross-entropy，并加 `1e-3 * mean(residual²)`。只训练 MLP，用 AdamW，`lr=1e-3`、`weight_decay=1e-4`、1000 updates、seed `20260925`、每步均匀抽 8 个 bank；不 early stop、不扫参。

collector 使用 `collect_fullbank.py --output-dir <PBS_JOB_OUT>`，输出根目录下的 `collection_gate.json`、`train_banks.pt`、`validation_banks.pt` 和小型 `collection_manifest.json`。前四个训练 task 的 shadow-off/on 配对必须逐项通过；gate 未通过时只写 FAIL gate 与失败 manifest，立即停止，且不生成 bank 文件或继续其他 task。通过后才继续采集剩余训练任务与前 8 个 validation task。后 8 个 reserved holdout task 不会传入 `World.evaluate`。

两个 `.pt` 文件均由 `torch.save(list[dict])` 写入，一条记录对应一个 episode、一个实际到达的 replan step 和一个 captured CEM round。记录含 `split`、`selection_order`、`episode_idx`、`row_index`、`start_step`、`replan_step`、`cem_round`、`solver_seed`、`candidate_indices`，以及 CPU tensors：`initial_emb[192]`、`goal_emb[192]`、`candidates[300,5,10]`、`student_costs[300]`、`teacher_costs[300]`。`initial_emb` 是 student cost path 使用的当前历史末帧 latent；candidate 顺序保持 CEM 原始索引，`candidate_indices` 固定为 `arange(300)`。训练代码只读取 `train_banks.pt` 做 1000 次更新；完成后才读取 validation banks。

## 判定与资源

统计单位为 source episode。对每个 bank，计算 student/ranker top30 相对 teacher top30 的标准化 teacher-cost regret；先在 episode 内对可用的 t0/t25 与三个 CEM rounds 取 median，再比较 8 个 episode。GO 要求：全部配对与 finite 检查通过、t25 coverage 至少 6/8、episode-level median treatment-control delta `≤−0.05`、至少 5/8 episode 严格改善，并且 round 10、20、30 各自的 episode-median delta 均 `≤0`。其余情况为 NO-GO；coverage 不足时为 inconclusive/NO-GO。不得替换 episode。

总预算为一个最多 `02:00:00` 的单 GPU PBS compute-node job，含 collection、training、validation；GPU 利用率和显存每 5 秒记录到 job log。超出这次 allocation 即停止，不延长或追加第二个 job。大 bank 留在 compute node，只回传小型 summary、gate、job log、status 和 telemetry。该 gate 通过也只支持此 task split 上的 predictor-level ranking；不支持 treatment CEM、planner deployment、closed-loop success 或 speedup。
