# LpWM PushT 资源调整记录

2026-09-27，root 根据已完成的 CUDA cost calibration 批准正式 sparse、dense 两臂各申请一份 12 小时 PBS allocation：每份 1 GPU、16 CPU、110 GB，提交入口仍为 `normal`，由 PBS 路由。单臂 `run_formal_gpu.pbs` 的内部 deadline 为 42,600 秒（11 小时 50 分），为 PBS 结束留出 10 分钟。

调整依据是实测 official-path training update 耗时：sparse 100 个 timed updates 均值 0.4262 秒、p90 0.4398 秒，对应 61,930 updates 约 7.33 小时；dense 均值 0.4214 秒、p90 0.4334 秒，对应约 7.25 小时。单次 validation 与 checkpoint 只需约 22 秒，正式每两轮会重复执行。native sparse 单例规划耗时 204.408 秒；乘以 50 得到约 2.84 小时的线性 proxy，不是实测 50-case walltime，也不保证是上界。Dense planning 未单独计时。12 小时给每臂保留约 1.6 小时及测量误差余量。

原先的 2 小时 GPU cap 是尚未测量训练成本时的初始资源假设。依据实际测量，root 已批准此项资源调整；总预算为两份独立 12 小时 allocation。数据量、训练 epochs、两臂官方参数、50-case evaluation、CEM/MPC budgets 均保持冻结。PBS job 仍请求 `normal` queue，不直接指定仅可路由的队列。实际执行队列、PBS `Exit_status` 与 `Stageout_status` 在各作业终态记录。

校准只证明训练成本和一个 native plan case 的运行成本，不构成 LpWM 模型结果。正式 checkpoint 必须来自各自完整的官方两 epoch 训练命令；校准权重不会进入正式结果。

PBS scheduling observation (2026-09-27): both official jobs were submitted through the approved 
ormal entry and routed to g1. The sparse run started; PBS reported User has reached queue g1 running job limit for the dense job. Keep that original job queued for automatic scheduling when the account slot becomes available; do not change the requested queue, attempt another route, or duplicate the dense run.
