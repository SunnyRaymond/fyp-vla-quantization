# Action-Response Straightening：阶段实验结论

日期：2026-09-24。范围：pinned LeWM＋PushT 的 development-only pilot。原始 [idea](../03_Action_Response_Straightening_LeWM_PushT.md) 中的机制与成功标准是待检验假说；这里仅记录已执行的实验。

## 已完成

1. **E0 数学检查**：四个纯函数例子（affine、quadratic、cubic-at-zero、cross-term）通过，说明三点曲率不能单独证明邻域 Taylor 精度。代码与运行说明见 `test_e0_math.py`、`README_E0.zh.md`。
2. **E1 frozen model pilot**：两个新的 development-only PushT reset seeds，截取官方 CEM 第 1／15／30 轮真实的 300 个候选。callback reference 与 candidate 0 对齐；官方 exact cost replay 误差为零。详情与限制见 [E1 结果](E1_PILOT_RESULT.zh.md)，原始 JSON 为 `artifacts/25537027.pbs101/e1_pilot_summary.json`。全部 GPU 计算在 PBS allocation 内完成，`25537027.pbs101` Exit status 0。
3. **E2 matched-training runtime pilot**：两臂各 5 updates，仅检验数据、模型接口及 loss 量级。接口通过，但 `λ=1e-4` 的 curvature loss 权重明显过大，且 `δ=0.02` 远离 E1 真实 CEM 的 1–9 L2 尺度。因此没有完成有解释力的 E2 训练对照，也不根据 held-out 微小差值选优。JSON 为 `e2/artifacts/25537028.pbs101/e2_summary.json`；PBS Exit status 0。
4. **E3 环境可重放性预检**：CPU PBS `25536982.pbs101` 的 seeded reset＋完整共同前缀重放检查通过，可在该路径上重建测试过的 PushT branch 起点。它不等同于任意中途状态 snapshot／restore。原始 JSON 为 `e3/artifacts/25536982.pbs101/e3_replay_probe.json`。未训练 joint model，也未生成真实反事实标签。

## 决策

**直接“每轮一次完整动作 Jacobian＋300 个线性化候选评分”路线：NO-GO（当前 pinned 模型／实现）。** 稳态 Jacobian 单项约为完整 batched 300-candidate official cost 的 6–7 倍；在 early round 的两个 context，top-30 overlap 仅 0.10。动作曲率正则可能改变局部精度，但不会自动消除每轮 Jacobian 的这笔计算，因此不延长该直接加速 recipe。

**整体研究想法：未判定为普遍无效。** 这两个 seeds 不能证明所有 context 的几何表现。late-round Jacobian 有限差分检查未通过预设 5% 门槛，其 Taylor 残差不作确认性解释。E2 的 5-update 试跑仅是量级诊断，不能用于声称训练有效或无效。

根据预设 gate 和此前“保留 NO-GO、不把失败 recipe 扩展到 planner／closed-loop”的规则，E2 的 100-update 训练、E3 joint counterfactual training、E4 surrogate adaptive CEM、E5 接触分层和 E6 end-to-end／paired closed-loop 均未运行。没有 planner speedup 或 task-success 声明。

## 复核入口

- `PROTOCOL_E1.zh.md`：E1 的原计划和首轮距离尺度修订。
- `E1_PILOT_RESULT.zh.md`：误差、覆盖、时间和 Jacobian gate。
- `e2/README_E2.zh.md`：matched-training 设计及 5-update pilot 的权重问题。
- PBS 记录：`25537027.pbs101` 与 `25537028.pbs101` 都完成且 Exit status 0；各作业的 GPU utilization／显存已写入对应远端 job log。
