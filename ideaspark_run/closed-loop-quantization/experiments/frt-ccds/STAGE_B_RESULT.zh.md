# FRT Stage B 结果

2026-09-12。结论：**工程核验通过，但本轮预注册的机制筛选未通过（mechanism_no_go）**。停止自动进入 C；这不是证明所有 FRT 变体都无效。

FRT 在固定 Q0 残差上的效果明显优于两个对照；但换成训练后模型产生的 fresh residuals 时，它与等范数随机扰动对照几乎持平。因此，本轮尚未证明“使用真实量化误差方向”带来了可迁移的额外收益。

## 实验与独立核验

- GPU 作业 64694：CCDS TC1N04，Tesla V100-PCIE-32GB，COMPLETED / 0:0，19 分 34 秒，包含最终完整 preflight 与正式训练/评测。
- CPU 独立核验 64696：COMPLETED / 0:0，14 秒。manifest、原始误差数组复算、Wz 与 A 的一致性、split/episode metadata、9 份 checkpoint SHA/hash 与 hard W4 grid 全部通过；所有 grid 重建误差为 0。
- 3 methods（Clean-only、RandomSameNorm、FRT）× 3 seeds（1201–1203），每组 1000 updates；batch=2，Adam lr=.01，λ=4。CAL 与 DEV 各 6 episodes，每个 episode 2 条 records。所有 fit 完成，没有预算耗尽或作业不完整。
- 11 个 evaluator（FP、Q0、9 个拟合模型）在共同的 11 个 residual banks 上交叉评分。每个 fresh bank 使用其固定 source context，所有 evaluator 共享该 context。主 fresh_union 指标对全部 9 个 fresh banks 求平均，避免只用模型自己的残差评价自己。
- 本轮为 predictor 六 blocks 的 weight-only numerical W4 emulation。未读取 TEST，未执行闭环任务成功率评测，也没有 native 低比特加速或压缩结果。

## DEV 结果

下表为 episode/seed 平均误差，越低越好。每个 episode 的两条 records 先在 episode 内平均；fresh_union 还平均全部 9 个 fresh banks。各 episode records 数相同。

| 指标 | Clean-only | RandomSameNorm | FRT |
|---|---:|---:|---:|
| clean weighted MSE | 0.00925354 | 0.00851265 | **0.00766778** |
| Q0 transport weighted MSE | 0.0000839704 | 0.0000649137 | **0.0000480443** |
| fresh_union transport weighted MSE | 0.000578966 | 0.000498327 | **0.000496815** |

FRT 相对 Clean 的 clean error 下降约 17.1%；Q0 transport 相对 Clean 下降约 42.8%，相对 Random 下降约 26.0%。但 fresh_union 相对 Random 只下降约 **0.30%**，未达到预设的至少 5%。

## 预设门槛判定

每项不仅检查总体平均，还要求至少 2/3 个 seeds 通过；每个通过的 seed 需平均值达标且至少 4/6 个 episodes 达标。

| 门槛 | 总体平均 | 达标 seeds | 结论 |
|---|---|---:|---|
| FRT clean ≤ 1.10 × Clean | 通过 | 3/3 | 通过 |
| Q0 transport ≤ .95 × Clean | 通过 | 3/3 | 通过 |
| Q0 transport ≤ .95 × Random | 通过 | 3/3 | 通过 |
| fresh_union transport ≤ .95 × Clean | 通过 | 2/3 | 通过 |
| fresh_union transport ≤ .95 × Random | **未通过** | **1/3** | **未通过** |

最后一项在 seeds 1201、1202、1203 的 FRT/Random 平均误差分别约为 0.976、1.116、0.920。1201 虽有改善，但不到 5%；1202 更差；只有 1203 同时满足平均改善和 episode 一致性要求。失败不只是跨 seed 稳定性不足，整体改善幅度也不足。

## 如何理解

本轮有局部改善，不应写成“FRT 所有指标都不 work”。更准确的判断是：在这套冻结配置上，FRT 对固定 Q0 误差的优势没有充分转移到 fresh residuals；它在后者上的表现接近普通随机扰动训练。按照事先约定的全部门槛，不自动扩大到更贵的 C。

证据范围仍有限：只有 6 个 DEV episodes；3 个 fit seeds 不是 3 组独立环境样本；λ 的 CAL 原始比值约 631.8，被预设规则裁剪到 4。1000 updates 完成不等于证明已收敛到最优。此结果判定的是当前 recipe 的最小筛选，不能排除其他实现，但不能据此继续用 DEV 调参来改写本次结论。

## 资源与审计

A+B 累计 GPU allocation：23 + 41 + 12 + 129 + 1174 = **1379 秒，即 22 分 59 秒，约 0.383 GPU-hours**。包含早期 probe、两个保留的工程失败尝试以及最终 preflight，远低于 6 GPU-hours 上限。环境检查、诊断与独立核验仅使用 CPU。

早期工程问题及修复：64690 的稳定 identity 误含 allocation-specific CUDA visibility，启动后停止；64691 的 batch-1 FP target 与 batch-12 评测有约 5.72e-6 的浮点差异，严格 null 检查使正式训练停止。修复为所有 evaluator 共用同 batch FP reference 后，64694 preflight 通过。没有直接清零误差、放宽 null 检查或改科学配置。

关键证据：

- `artifacts/64696/verification_b.json`：独立核验和全部 gate 数值。
- `artifacts/64694/summary.json`、`slurm_status.txt`：运行配置和完成状态。
- `manifest_b.json`、`jobs.json`：冻结配置和完整作业记录。
- 原始数组与 checkpoint：`/tc1home/UG/yguo017/frt_ccds/artifacts/64694`，留在 compute 工作目录。

监控 heartbeat `frt-b-v100` 已暂停。A、B 均完成记录；没有提交 C 或新 GPU 作业。
