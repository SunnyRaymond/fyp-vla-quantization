# 第一轮结论：保留 C8 A+Z、ρ=1.5 原 baseline

本轮已完整完成。新增 teacher-anchor latent 邻域 KL **未带来 heldout 的主要指标改善**，因此不将它替换为新 baseline，也不据此进入完整 280 个闭环 episodes 或开展收益归因 ablation。

## 主要结果

所有指标越低越好；先平均同 case 的 seed，再平均 suite 内 case，最后四个 suite 等权。Heldout 为 40 cases × 2 seeds；胜/负以 40 个 case 的 seed 均值为单位。

| Heldout 指标 | 原 MSE+NMSE | 加 latent KL | 改善/变差 cases |
|---|---:|---:|---:|
| First10 motor RMSE | 0.0442968 | 0.0458895 | 14 / 26 |
| Full32 motor RMSE | 0.0475010 | 0.0493055 | 17 / 23 |
| Late22 motor RMSE | 0.0485285 | 0.0503193 | 15 / 25 |
| Future latent NMSE | 0.0608774 | 0.0614710 | 18 / 22 |
| First10 gripper RMSE | 0.0570540 | 0.0426101 | 17 / 23 |

First10 motor RMSE 在四个 heldout suite 的均值上全部变差，不能解释成只有某个 suite 拖累总体。Latent NMSE 则两个 suite 改善、两个变差。Gripper 的总体均值改善，但多数 case 没改善，不能描述为普遍或稳定的 gripper 收益；这里测的是连续 gripper 输出 RMSE，不是离散开合错误或闭环成功率。

Calibration 的 first10 motor RMSE 从 0.0421332 降到 0.0415491，latent NMSE 从 0.0544810 降到 0.0495689。直接观察是 calibration 的平均收益未转化为 heldout 收益。它与过拟合或训练目标不充分对齐相容，但单次实验不能区分这两种机制，也不能单独证明原因。

2026-10-21 补充诊断进一步限定了上述解释：calibration motor 只有 3/8 cases 改善；同一 C8 task 换 demonstration 后的 heldout motor 也退步（1/8 改善），不能只归因于未覆盖新任务。新增 KL 的 calibration 均值相对未训练初始化上升约 10%，其 latent 梯度强度也明显大于 loss 数值占比暗示的强度。完整的参数、指标及 KL 几何分析见 [DIAGNOSIS.zh.md](DIAGNOSIS.zh.md)，本次补查无新模型 forward 或闭环。

2026-10-22 的机制诊断已补齐模型参数梯度、真实回滚步与冻结 reader 干预，见 [MECHANISM_CONCLUSIONS.zh.md](MECHANISM_CONCLUSIONS.zh.md)。证据优先指向有限步优化/量化部署落差；当前 KL 的额外 motor 控制价值未稳定体现，但不能称为方向整体反了。该阶段有新增诊断 queries，原 trained baseline 未重放，仍无闭环或持久训练更新。

## 对原研究问题的含义

本轮不能认定“原 loss 中 latent 比 action 更错位”，也不能据此否定 latent loss 改进方向。我们没有新跑闭环，没有单独干预 action 与 latent 两项，也没有为该 KL 做多权重或多训练 seed 验证。

这版 KL 对齐的是 **teacher latent 局部邻居的 affinity 分布**。数学上它是有效的离散分布 KL，利用了 latent 的空间/时间结构；它并不直接测量信息瓶颈、mutual information、真实 future density 或动作重要性。局部视觉结构的概率更接近 teacher，并不自动保证控制相关信息更准确。后一句是设计上的限制，不能把它当成本次失败原因已经被证实。

Baseline 缓存缺少可用于这一定义的完整 trained future tensor，所以本轮没有 baseline anchor KL，也不能声称 KL 比 baseline 降低了多少。原始 reference_cache 的 parent 是未训练 C8 banks，只用于拟合 beta 和初始化检查，绝不冒充训练后 baseline。

因此，保留原 C8 A+Z、ρ=1.5 作为比较基础。后续若继续研究 KL，首先要说明目标分布为什么承载控制相关信息，并用 calibration 或独立开发集确定设计；不利用本次已看过的 heldout 反复调 temperature/beta 后仍称其为独立泛化验证。本轮不自动启动进一步实验。

## 运行和公平性证据

- CPU prepare：`25902832.pbs101`，F / Exit_status 0，artifact exit 0、PHASE_COMPLETE、coverage 全部通过，2m51s。
- GPU training/evaluation：`25902882.pbs101`，F / Exit_status 0，artifact exit 0、PHASE_COMPLETE、coverage 全部通过，58m44s。
- CPU aggregate：`25902885.pbs101`，F / Exit_status 0，artifact exit 0、PHASE_COMPLETE、coverage 全部通过，18s。
- 新分支从同一原始未训练 C8 banks 开始，相同顺序 8 updates；只有 codebooks/log row gains 参与训练；未加入 activation clipping/per-channel scaling。
- 初始化 action/future 与原缓存的 max_abs 和 RMSE 都为 0；最终 native infer 与 sampler 的 action parity 也为 0。
- 八步训练均为 614/614 connected、finite、nonzero Linears。GPU 利用率/显存每 15 秒写入该 GPU 作业的 job.log。
- Temperature 0.4389900267、beta 0.05137399516；channel statistics/temperature/beta 全部只来自 C8，训练前已冻结，heldout 未参与拟合。
- 88 条 baseline 查询直接复用；新增 baseline queries 0、新 teacher queries 0、新 closedloop episodes 0。新增分支总计 97 次逻辑模型查询：8 training + 88 fixed-input evaluation + 1 native infer parity；不是内部 denoiser forward 次数。
- 对照来自原 rho1.5 update8。缓存记录的 updates=0 表示在父 checkpoint 上额外训练 0 步，不表示未训练，也不表示启用了缓存目录名称中的 clipping。
- 20 分钟只读 subagent 监控已按完成状态停止。没有使用 banked reset。

完整逐 suite 数值见 [ANALYSIS.zh.md](ANALYSIS.zh.md)，机器可读结果见 `aggregate/summary.json`，实际 scheduler/artifact 证据见本实验根目录 `last_status.json`。
