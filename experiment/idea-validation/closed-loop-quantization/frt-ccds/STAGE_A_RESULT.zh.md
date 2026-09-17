# FRT Stage A 结果

日期：2026-09-12。完整 A 作业 `64688`，CCDS `TC1N04`，Tesla V100-PCIE-32GB，SLURM COMPLETED / 0:0，实际 GPU allocation 41 秒。runner 的工程检查全部通过；独立 CPU 作业 `64689`（1 秒）复算也已通过，manifest 与 raw arrays 均无错误。

| 检查 | 实际结果 |
|---|---|
| 与原始 source 一步预测一致 | 最大绝对误差 0 |
| 相同状态、相同动作的环境重复回放 | state 与 xy 均一致 |
| 零残差的 transport | 0 |
| FP transport 独立重复计算 | 差异 0 |
| 历史与已知动作 | 原历史不变，action 残差为 0 |
| Q0 hard W4 保存/重载 | 输出误差 0 |
| 经学习的 hard W4 保存/重载 | 输出误差 0 |
| Clean / Random / FRT backward | 三者 loss 有限 |

实际 checkpoint 使用 `num_hist=1`，不是 source 配置默认的 3。权重量化仅针对 predictor 的 6 个 blocks、24 个 Linear，数值模拟 W4；本实验不证明 native low-bit 加速。

## CAL 统计与后续预算

使用 6 个新 CAL episodes，每个 2 条连续窗口记录，共 12 条。DEV 为另 6 个 episodes，不参与 Wz 或 λ 计算。

- Q0 clean weighted MSE：0.01484835334122181。
- Q0 transport weighted MSE：0.00002350226350245066。
- CAL loss 比值：631.7839700705221；按预设 [0.25, 4] 裁剪，B 冻结 λ=4。transport 项较小是本次筛选的解释限制，不能静默改权重。
- batch=2 的单步计时：Clean 0.08946 s、Random 0.08851 s、FRT 0.08857 s。这里只是短程计时，长训练可能不同。
- 峰值 allocated 显存 1.6964 GiB，reserved 1.8027 GiB。
- B 冻结 3 methods × 3 seeds × 1,000 updates，batch=2，预计训练加评测约 20–40 分钟，拟申请 1 小时上限。
- A 初步预检另用 23 秒 GPU；累计已完成 GPU allocation 64 秒，约 0.01778 GPU-hours。环境检查仅使用 CPU。

证据：`artifacts/64688/summary.json`、`artifacts/64688/manifest.json`、`artifacts/64688/slurm_status.txt`。大数组与 checkpoint 保留在 compute 工作目录 `/tc1home/UG/yguo017/frt_ccds/artifacts/64688`，不通过 login node 搬运。

A 通过只说明实现可用于后续机制筛选，不能说明 FRT 有效；机制比较属于 B，闭环任务成功率不在本轮范围。

独立核验：`artifacts/64689/verification_a.json`。复算 Q0 clean=0.014848354418275306、transport=0.000023502263472242954、λraw=631.7840167102103、λ=4；float32/float64 的微小差异在容差内。A 工程门已通过，可以执行 B。
