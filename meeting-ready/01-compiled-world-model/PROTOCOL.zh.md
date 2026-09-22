# Compiled World Model：predictor-level frozen protocol

## 问题与范围

本轮只验证 `01_Compiled_World_Model_LeWM_PushT.md` 的最小结构命题：在相同 frozen LeWM encoder、相同 contexts、actions、teacher targets 和 latent-MSE objective 下，context-dependent factorization 是否优于 fixed basis，并且在 30 次、每次 300 candidates 的 multi-query 边界上不被 matched dense-prefix predictor 支配。

本轮不运行 official CEM、不执行 closed-loop PushT、不联合训练 encoder，也不增加 mixture、router、fallback 或 ranking loss。predictor-level GO 只表示值得进入下一阶段，不表示 planner replacement。

## 数据边界

- 训练复用 `25223859.pbs101` 在 compute node 留存的 512-context anchor-aligned rows；每 context 64 candidates，parent episodes 保持原 split。
- rank selection 只用旧 development slice `valid[552:560]`，8 episodes × early/middle/late × 2 seeds，共 48 blocks。
- final test 固定为此前未用于当前设计选择的 `valid[592:600]`，seeds `20301201/20301202`，同样为 48 blocks。
- final test 不参与 rank、threshold 或 architecture 选择。

## 对照

- B1：dense causal action-prefix MLP，每个 candidate 同时处理 context 与 action prefix。
- B2：`b_h(c)+B_h phi(u_1:h,h)`，`B_h` 为全局参数。
- B3：`b_h(c)+B_h(c) phi(u_1:h,h)`，context compiler 每 context 只执行一次。
- B4：与 selected B3 完全相同权重，只把 terminal squared-distance cost 改写成 quadratic contraction。

B2/B3 固定扫描 `r=32/96/192`。所有 arms 使用相同 1500-step context schedule、AdamW 和 horizon-weighted latent MSE，不使用 goal、ranking loss 或 teacher forcing。

## E0

在模型训练前验证：prefix causality、batch/chunk 一致性、context swap 会改变 B3 输出、合成 FP64/FP32 explicit/compiled cost 一致性。训练后再验证 selected B3 的 explicit terminal cost、quadratic cost 与 official criterion 的一致性和 argmin。

任何 future-action leakage、context/candidate 混用或 criterion 不可写成冻结的 terminal quadratic form，均使 B4 停止；不得以近似值冒充 exact compilation。

## 判断

rank 只按 development 的 median Spearman、median top-30 overlap、negative relative latent MSE lexicographic 选择；完全相同时选较低 rank。

final test 同时检查：absolute predictor quality、B3 相对同 rank B2 的 episode-paired 改善、B3 是否被 B1 支配、以及包含一次 prepare/compile 的 30-query timing。完整阈值在 `COMPILED_WORLD_MODEL_FREEZE.json` 中冻结。

若任一核心 gate 失败，结论为当前 factorization 的 predictor-level NO-GO；若样本或数值证据不足则为 INCONCLUSIVE。即便通过，也只允许进入后续单独冻结的 official CEM 阶段。

## 集群约束

模型加载、HDF5、训练、GPU timing 和大文件读写只在 PBS compute allocation 内进行。wrapper 必须检查 `PBS_JOBID` 和 hostname，并每 5 秒记录 GPU utilization/memory。login node 只负责小型文件上传、提交、状态查询与小型 summary 回传。
