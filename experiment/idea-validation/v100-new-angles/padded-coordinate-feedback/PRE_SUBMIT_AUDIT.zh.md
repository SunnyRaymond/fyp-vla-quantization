# Padded feedback 执行前审查

2026-09-13，GPU64772 提交前完成静态接口核对；没有读取该实验输出后修改门槛。

CPU64771 在 TC1N05 的真实 SLURM CPU allocation 中完成，耗时14s。固定新样本为 task0 episodes33/58、task1 11/19、task2 35/44、task3 45/48；frame 为各 episode length 的四分之一取整。CPU source hashing、video decoding、单写者隔离和8个raw样本均完成。base identity SHA为 be4a49ebe588a49a29bd26ed98b8a01a648a247e66d45e12a935ac7d8d0c4e64。

root 在提交前修正了 agent runner 中四个实现风险：conceptual expert module 名称不能直接用于实际 state_dict 排除；noise 的广播必须保留50个action位置；官方 float32 累加时间只对解析网格作1e-6容差检查；no-op 必须测试本次 wrapper 与原生调用。agent 完成修正，补充 V100、首步 full-state equality、engineering fail-closed 和 protocol/脚本快照。

已核对复用 Flow helper 的实际语义：严格加载 checkpoint 与 processor identity；仅 expert transformer Linear W4；每分支 restore 会逐元素检查；bypass digest 使用实际 Parameter 绑定排除 expert weights；官方 predict_action_chunk 的推理路径使用 no-grad。四臂均独立 reset 并使用同一 noise clone。padding 替换仅修改7:32 velocity，physical7不借用其他分支输出。

CPU verifier 另行从完整 raw states/velocities/actions 重建 native equality、analytic t*z、Euler递推和终点一致性，并独立计算冻结 E0/S/E1/E2/P0、binding与双reference gains。仅保存工程检查通过与数值结论相结合的状态。准备与静态检查不构成科学结果。
