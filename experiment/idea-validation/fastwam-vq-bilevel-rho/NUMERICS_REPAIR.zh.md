# 2026-10-19 分支重放修复

原 preflight 25879403.pbs101 完成，但 gate=false。参数、Adam、RNG 和恢复后的输出均通过；非空 Adam 的相同 plus 分支重复，development RMSE 从 0.0438592704 变成 0.0556151960，差 0.0117559256，大于 plus/minus 差 0.0020894539。该方向不能作为更新 rho 的可信信号。

原 job artifact 和失败 evidence 保留。控制器只允许经过验证的 terminal failed preflight 带 repair_reason 建立新 attempt，不能从 Q/R、查询失败或超时触发重提。新 attempt 使用独立 evidence_deterministic_v2.json，不会把旧结果误当作新验证。

本次候选修复开启严格 PyTorch deterministic algorithms、关闭 cuDNN benchmark，并在导入 Torch 前设置 CUBLAS_WORKSPACE_CONFIG=:4096:8。保留原 loss、rho 范围、分支步长、数据、sampler seeds、Adam 配置、16-step 最终训练及 312-query 评测分母。除了原来的输出门槛，新增相同分支 update 后参数、Adam、RNG 完全一致检查；不放宽原 gate。先用已有 outer teacher 检查新数值设置下输出一致，才继续虚拟训练。

候选修复可能改变 backward 的浮点累加顺序。因此，若本次 preflight 通过，两个固定 rho 对照和 selected rho 最终模型都从共同 initial_state.pt 训练 16 updates，统一采用 deterministic_v2。旧 rho0.1/update16 和 rho1.5/update8 checkpoint 仍只读复用于来源验证及非空 Adam 诊断，不直接作为新数值设置的最终对照。原始 C8 banks、C8 teacher、新 development teacher、历史 80-query teacher 均继续复用。

这会比原先续训复用方案多 24 gradient updates，仍在最初从零计算的 80G+400F pilot 范围内。实际成本按所有新 GPU 作业 PBS used walltime × GPU 数报告，包含失败/重试/加载。每次作业 walltime 01:59:00，GPU telemetry 每 15 秒写入 job.log。

## 已验证结果

25879908.pbs101：PBS F/Exit0、artifact exit0、PHASE_COMPLETE、对应job/protocol的完整summary均通过。initial与rho1.5/update8的状态恢复完全一致、恢复输出max_abs=0。非空Adam的plus重复分支，更新后参数/Adam/RNG完全一致，development RMSE差0；minus/plus相对差异分别11.2975%和8.1595%，均超过0.1% deadband，有限差分signal gate通过。首个复用teacher的action/future输出max_abs=0，实际新增teacher查询0。

本次5个虚拟gradient queries、25个inference queries，PBS walltime00:30:59，0.516388889 GPU-hours。加上原失败预检00:22:15，预检合计0.887222222 GPU-hours。稳定gradient step约115秒，development inference约6秒；这些是预检操作计时，不代表闭环或native kernel性能。

通过后已提交search 25880277.pbs101、train_low 25880278.pbs101、train_current 25880279.pbs101，均01:59:00。后续只按完整8轮/16updates、统一数值设置的三组endpoint及312条评测记录推进，不据初始分支方向提前确定最终rho。
