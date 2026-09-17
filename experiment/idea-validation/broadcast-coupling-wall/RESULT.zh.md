# 最终结果：mechanism_no_go_stop

2026-09-17 在 ASPIRE2A 完成 frozen Experiment 1 pre-gate。最终有效 job 为 `22705367.pbs101`：1×A100，walltime `00:01:12`，`Exit_status=0`，并生成 `VERIFIED`。

Engineering gate 全部通过：episode、horizon、arms、record count、paired keys、raw treatments、model unchanged 和 A100 checks 均为 true。

Primary gate 为 H=5 的 `balanced_broadcast` vs `shared_stochastic` real-future feature MSE；预注册阈值是至少 5/6 episodes 获得不低于 10% 的 relative gain。六个 episode 的 gain 为：

```text
-0.007672, -0.006570, -0.006877, -0.000496, +0.015988, -0.030384
```

达到阈值的 episode 为 `0/6`，因此 verifier 决策为：

```text
mechanism_no_go_stop
expansion_authorized = false
```

按 frozen stopping rule，不运行正式扩展、Experiment 2、Experiment 3 或 PushT。该结论仅覆盖 initial exact-broadcast A4 fake activation quantization；后续 autoregressive predictor calls 为 FP32，不能据此声称 native low-bit、deployment gain 或优于 STaMP。

本轮三个 jobs 总 allocation 为 138 GPU-seconds，即约 `0.0383 GPU-hour`；按 64 SU/GPU-hour 估算约 `2.45 SU`。前两个 jobs 在产生 scientific records 前 fail-closed，最终 science decision 只来自第三个 VERIFIED job。
