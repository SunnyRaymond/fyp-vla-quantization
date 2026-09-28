# LeWM + PushT：共享线性响应校正实验

本实验只检验一个假设：先前 serial student 的 terminal action-response 误差中，是否存在能由一个共享线性映射稳定修正、并在新 episode 上改善 teacher-objective elite selection 的部分。主 student 固定为 job 25549480.pbs101 的 terminal_response step-3000 checkpoint；不重新训练 student。

## 拟合与对照

只用先前冻结的 512 个 Phase2 training contexts，每个 context 64 条候选动作及缓存 teacher targets。对每个 bank，分别将 student、teacher terminal latents 减去各自的候选均值。以 teacher bank-centered MSE（下限 1e-6）归一化，拟合一个全局 192×192 ridge 矩阵 W，ridge 强度预设为 0.1×训练 Gram 矩阵平均特征值，先验中心为 identity；不调参、不做 sweep。

推理时，对同一候选 bank 的 student terminal 输出应用：

    corrected_i = student_bank_mean + (student_i - student_bank_mean) W

前四步预测与 terminal bank 均值不变。矩阵只在旧训练数据上拟合；新 episode 上的推理不调用 teacher。identity 是主对照；从同一训练目标闭式拟合的 scalar 是诊断对照，不按新 episode 结果挑选使用。

## 新 episode 与门槛

在 PBS compute node 上复现先前固定的 PushT episode shuffle 和排除规则，核对上次 8 个 episode 的身份后，从原候选序列中跳过这 8 个，取接下来的 8 个。先保存有序 selection，再构建任何新评估 bank。每个 episode 使用 early/middle/late 三个 anchor、两个固定 action-prefix seed；各 bank 300 candidates，三臂共用相同 bank。

主比较为 linear 减 identity：先逐 bank 配对，再对同 episode 的六个差值取中位数，最后汇总八个 episode。预先继承上一实验的相对机制门槛、绝对 predictor 排序与延迟门槛、三个 stratum 的保护门槛，详见 FREEZE.json。scalar 只解释“纯缩放是否足够”，不参与选择最优臂。若结果仅改善 latent response 而未改善 elite 排序，仍判 NO-GO。

此次结果只适用于 bank-relative、predictor-level 的固定候选分布。训练/评估按 episode 隔离；官方 dataset-wide scaler 可能读取了所选 episode 的列，不能声称预处理也按 row 隔离。official CEM、planner、closed-loop 均不在本实验范围。

计算、HDF5、checkpoint、推理、拟合和完整性检查只在通过 PBS allocation 检查的 compute node 上执行。Login node 仅传小型控制文件、提交与查询作业。作业期间每 5 秒记录 GPU 利用率与显存。不得查询或使用 banked reset credits。
