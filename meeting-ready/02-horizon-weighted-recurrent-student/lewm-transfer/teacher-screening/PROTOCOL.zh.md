# LeWM balanced_base + teacher screening 冻结实验

本轮检验 student 作为候选筛选器的用途。沿用 LeWM + PushT、h256 shared recurrent
balanced_base、temporal-balanced score distillation。旧 predictor replacement
NO-GO 保留；本轮不以旧 replacement gate 替代 screening 问题，也不把筛选成功
称为 closed-loop success。

## 模型来源

已在远端轻量检查 Phase 5 `25152151.pbs101` 与 Phase 6 `25158955.pbs101` 输出目录，
两者只有 summary/log/status，没有保存 student checkpoint 或 balanced rows。
若无其他明确对应 checkpoint，允许在本次 PBS allocation 内按原始 512 contexts、
init、context schedule、candidate slate、loss 和 3000 updates 重建一次 balanced_base。
标记为 `reconstructed`，不可声称取回原权重或 bitwise 复现原模型。保存模型和
balanced rows 到本次 compute 输出，后续复用，不回传本地。不重新选择 seed、snapshot、
结构或超参数。

## Stage A：独立候选集 screening gate

沿用 selection seed `20300903` 的 valid shuffle，固定 fresh `valid[544:552]`，
排除此前全部 `valid[:544]`。8 episodes × 3 temporal anchors × 2 action seeds
(`20300947/20300948`) = 48 blocks，每 block 300 candidates。沿用原 candidate
generation、normalization、current/goal encoding 与 official cost。

比较完整 teacher top-30、student top-60/120 再经 teacher 精选的 top-30。
Teacher 始终从相同当前 encoded state 独立 rollout，不能接 student 的预测 latent。
报告 teacher top-30 recall@K、全部 elites 包含率、teacher elite mean-cost regret、
episode/stratum/block 摘要以及最差 block。Episode 为独立单位；candidate、seed、
anchor 不作为独立 replicate。该 8-episode pilot 是有界机制筛查，不作 population
success 或 statistical power 声明。

计时使用同一 GPU、CUDA synchronization、3 warmups、10 repeats，固定 seed 的交错
arm order。Teacher baseline 包含300条 rollout/cost/elite selection；hybrid 包含
student300、shortlist sort/gather、teacherK、teacher elite selection。完整 teacher
shadow 仅供诊断，不得混入 hybrid production timing。计时边界不包含 encoder 或环境。
全部48 blocks均计时，以 synchronized wall-clock milliseconds 为主指标，CUDA event
时间为辅助。Latency gate 对每block paired reduction取median，不以候选数比例推算。

每个 K 均需同时满足：overall median recall >=0.95、minimum block recall >=0.80、
early/middle/late 各 median recall >=0.95、median hybrid latency reduction >=20%，
且接口和 finite 检查有效。选择通过条件的最小 K；两个都未通过则停止，不扩大 K 或
调整阈值。记录诊断数值，即使未通过。

## Stage B：条件性 adaptive CEM

仅 Stage A 通过后实施；使用另一批 fresh `valid[552:560]`，相同三个 temporal anchors、
seeds `20300957/20300958`。三臂为 teacher-only、student-only、通过的最小 K hybrid。
每次 fixed-observation planning 使用官方 solver 语义：300 candidates、top30、30
iterations、horizon5、packed action10、action_block5，并使用 paired innovations。
保留官方 tie-breaking、candidate-zero、variance/std、bounds 和最终输出语义。

每轮 hybrid 在更新 mu/sigma 前执行 teacher 精选。Shadow teacher 不参与额外决策，
记录每轮 recall/cost regret、mu/sigma drift 与 first-action difference。生产计时另行
排除 shadow 诊断。Frozen progression gate 沿用此前 LeWM first-action normalized L2
maximum <=0.15、coordinate absolute maximum <=0.25，并要求 native latency reduction
>=20%。这只界定 teacher-fidelity，不把 action difference 等同于环境失败。

本轮不运行 closed-loop，也不同时扩展 periodic-anchor、DAgger、架构或训练 sweep。
Stage B 的实现与相关 preflight 在 Stage A 通过后完成，端点和 gate 不因 A 结果改变。

## 执行边界与停止

Login node 仅连接、轻量状态查询、提交和小型控制文件操作。所有模型、HDF5、teacher
targets、训练、benchmark 与重 I/O 必须在 PBS allocation 内；运行前检查非空
PBS_JOBID 与非 login/head/submit hostname。GPU utilization/memory 每5秒写入job.log，
结束清理monitor并退出。只回传小型报告，不做多余哈希和重复审计。

用户要求：发现主 Codex 余额突然回到100%立即停止 agent 推进，保留交接。起始主额度
usedPercent=70（剩30%）；Luna reserve 起始即100%，不算主额度重置。若已有作业运行，
记录其真实状态，避免启动后续阶段；不把未完成作业声称为完成。

方法组织参考：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents.
https://doi.org/10.48550/arXiv.2609.00065 （2026-09-22核对当前record为v2；此文献不是本实验结果证据）。
