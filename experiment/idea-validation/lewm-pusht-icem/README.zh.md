# LeWM PushT iCEM population-decay 实验

实验已完成。结果与结论见 [RESULTS_25547844.zh.md](RESULTS_25547844.zh.md)，原始证据见 `artifacts/25547844.pbs101/`。

本实验只借鉴 iCEM 的 **每轮候选数递减**，不复现 colored noise、elite keep/shift、best-action selection 或 clipping，因此结果称为 iCEM-inspired decay ablation，不称完整 iCEM。[iCEM 原论文](https://proceedings.mlr.press/v155/pinneri21a/pinneri21a.pdf) 给出的递减式为 `N_i=max(Nγ^-i,2K)`。

## 冻结设计

- 同一 LeWM teacher、PushT dataset、50 个已冻结任务，CEM 30 iterations、top-30、solver seeds 42/43/44。seed 42 的官方 300/30 control 必须逐项匹配有效 reference，才继续其他 arms。
- 每个 seed 依次运行固定 `300/30`（每 solve、每环境 9,000 次评分）、固定 `150/30`（4,500）、`decay_equal_4500`（首轮 300，`γ=1.0572`，总 4,500）、`decay_paper_125`（首轮 300，论文 `γ=1.25`，总 2,569）。下限均为 `2K=60`。完整 30 项 schedule 在 [FREEZE.json](FREEZE.json)。
- `decay_equal_4500` 与固定 150/30 是**同评分预算**比较，隔离预算在 CEM 各轮的分配；`decay_paper_125` 检查更激进的数量缩减，其评分总量不同，不能把两者差异单独归因于递减形状。
- 每次 `solver.solve` 记录 CUDA 同步耗时，并由 `get_cost` 实际收到的 candidate shape 统计评分次数；保留逐任务成功向量、同 seed 配对差异、整段 evaluation 时间与 GPU 利用率/显存日志。

结果只支持这 50 个固定 dataset tasks 的探索性 paired 描述；不作为 formal noninferiority、random-reset simulator 成功率或稳健速度保证。不得按结果更换任务、seed、schedule 或追加 arm。

PBS 入口为 [run_icem_decay.pbs](run_icem_decay.pbs)。它只在获批 compute allocation 内加载模型和 dataset，GPU 利用率与显存每 5 秒写入该作业的 `job.log`。实验脚本在 [run_icem_decay.py](run_icem_decay.py)，动态 solver 在 [decay_cem.py](decay_cem.py)。
