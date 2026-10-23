# Action / latent 双 KL：冻结的 2×2 loss 实验

目标：保留 `L = L_action + lambda * L_latent`，检验替换 action、latent 两项 loss 的主效应和交互。训练范围为 first10 motor6 与 future latent（排除 frame0）；不加入 gripper、late22 或其他训练项。

| Arm | Action | Latent |
|---|---|---|
| mse_nmse | 原 MSE | 原 global NMSE |
| action_kl | Soft-bin KL | 原 global NMSE |
| latent_kl | 原 MSE | Soft-bin KL |
| dual_kl | Soft-bin KL | Soft-bin KL |

每组沿用原 C80、80 updates、update order、D=1/r=1 初始化、Adam、learning rates、bounds、update1 后 optimizer reset、seeds、共享 BF16 teacher。仅优化原614 ratios与1,806,863个channel log2D；books、indices、row gains、原microscaling exponents和模型参数冻结。原 mse_nmse 组复用作业25862598.pbs101已完成的 final80 checkpoint；重新播放其 calibration80 和 heldout80，要求原指标与端点记录精确一致，再补算 KL。其他三组分别从同一原始 baseline 开始训练。

原control重放的比较范围为160条记录中的原始RMSE/NMSE等标量指标。旧记录未保存完整action/future张量；本轮新训练组另行进行uncached/cached action、future张量的逐元素parity及实际infer的action parity。

概率定义：先使用全部 C80 的缓存 teacher 拟合 action 每维、future 每 channel 的固定总体 mean/std（std floor=0.02），排除 frame0 与 gripper。标准化后的 scalar 加 N(0,0.375²) 后，落入16个区间的概率；15个内边界为 linspace(-3.5,3.5,15)，两端为±inf。采用稳定 log-CDF 差，避免远离目标时因为概率截断而失去梯度。Teacher/student 使用同一冻结映射，KL方向 teacher||student，逐位置计算后平均。此分布是人为定义的数值 surrogate，并非真实的 flow policy density；不跨位置汇总 histogram。

Loss 配平仅用训练前原 baseline 在全部 C80（seed0）的输出：`alpha_a=mean(action_MSE)/mean(action_KL)`，`alpha_z=mean(latent_NMSE)/mean(latent_KL)`；替换项分别乘 alpha。原 `lambda=rho*c` 保持不变。此规则匹配初始各项平均 loss，不声称精确匹配各项梯度；保留原 optimizer/gradient clipping 并记录 gradient norm。不通过 heldout 选择 bins、sigma、权重或 checkpoint。

评估仍为原40cases×2seeds=80个 paired heldout queries，四suite等权；每组另完成 C80 端点80queries。只评固定 final80。保留原 RMSE/NMSE，并在所有四组计算相同 KL 指标。按case内平均seed，再做40case配对差、suite结果和2×2交互；80queries不当作80个独立case，每组只有一条训练轨迹。

预设工程筛选：dual_kl 的 heldout first10 motor RMSE 和 future latent NMSE 均低于 mse_nmse，motor paired-case median delta<0且至少24/40cases改善。另报告其是否优于两个 single-KL；未通过门槛仍完整报告 tradeoff，不替换门槛，不自动启动闭环。

计算：统计拟合及汇总使用 CPU PBS allocation；准备与三个训练 GPU job 每个硬上限01:59:00、软截止6840秒，每15秒把GPU利用率/显存写入所属job.log。gpt-6-luna / xhigh subagent每600秒只读监控。所有重I/O、数值和模型工作只在真实PBS compute allocation；login只处理轻量控制。共享teacher全复用，不新增teacher targets，不使用新的development set。GPU失败不自动重提。

调度依赖：三组训练仅等待完整 C80 的 loss_scaling.json 冻结，不等待原 control 的160端点重放结束。系数就绪后并行提交三组GPU job，与剩余control重放并行；最终CPU汇总仍要求准备和全部训练/评估都完整通过。该调度变化不修改任何训练或评估定义。

CPU汇总通过PBS `afterok` 依赖四个GPU job，预先提交后由调度器在全部成功退出时启动；汇总脚本仍逐项检查完整的80updates和640端点。它等待依赖时不占用GPU。

证据边界：固定输入、decoded-BF16 VQ + dynamic fakeA4、BF16 KV；不代表 native kernel、实际存储压缩收益、闭环成功率。Soft-bin KL 可能接近经过重加权的平方误差；若有收益，仍需后续区分 channel 尺度与 divergence 形状的贡献。
