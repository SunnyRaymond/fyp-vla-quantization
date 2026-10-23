# Fast-WAM：H/R W16A4 诊断与 R+VQ Joint8

执行顺序：完整 H/R W16A4 对照 → C8 各阶段边界 → 同起点 Joint8 → 固定最终 checkpoint 的 heldout 比较。

- 固定 C8 `[2000,2019,2021,2025,2051,2057,2066,2070]`，seed index 0；heldout 为原 40 contexts × 2 seeds，四个 suite 各 10 contexts。
- 614 个 Linears；Smooth0.5、原 signs、D=I、r=1、dynamic per-token A4 和 BF16 KV 固定。采用 BF16 decoded VQ 权重，未使用 native low-bit kernel。
- Loss 为 first10×6 motor action MSE + `0.04245452683169643 × future latent NMSE`，保留 student future 到 action 的梯度；ρ=1.5 和 normalization c 固定。
- 诊断阶段使用已完成的 R8，来源作业 `25902163.pbs101`；不重新学习该 R。保存 W16A4 两臂各完整 80 queries，以及 7 个阶段各完整 C8 指标。
- Joint8 从原 H+VQ0 banks 与 U=I 开始；R LR=1e-4，VQ LR=5e-4，Adam，8 updates、同 C8 顺序。仅 VQ Adam state 在 update1 后清空；R state 保留；两组分别 norm clip 1。
- 训练两个 stream-shared Cayley residual、614 层 codebooks 和 row gains；冻结原模型、Smooth、indices、B32 exponents。实际 forward 为 `A4((X/S)HU) @ Decode(C,g,I,e).T`。这是 compressed student 的共同适配，训练过程中不要求每一步保持原 FP 函数等价。最终直接使用编码权重，不再 refit。
- 复用 H+VQ8、R8+VQ8 的 80-query heldout、R8-W16A4 的 C8、R8+VQ8 的 C8、H+VQ0 reference cache。所有复用都校验原协议、cohort、seed 和 checkpoint 绑定。
- 记录每次 Joint update 前后同 context loss，完整初始/最终 C8 和最终 80-query heldout；不根据 heldout 选择 checkpoint 或超参数。
- Joint8 与 H+VQ8 的 VQ 起点、更新次数、学习率、样本顺序、loss 相同；Joint8 有额外 R 参数和计算，不能声称同 walltime。没有 H+VQ16 分支。

单 GPU 串行 PBS allocation，申请 normal、4 小时；实际 routing 以 qstat 为准。每 15 秒记录 GPU utilization/VRAM；subagent 每 20 分钟检查作业。所有模型/数值工作与重 I/O 均在真实 PBS compute allocation 内，login node 仅做轻量控制。

控制入口 `control.py upload/submit/status/verify/pull/resume`。单独保存 query 和训练 checkpoint，已完成单位可复用。终态成功须同时满足 scheduler Exit0、artifact exit0、PHASE_COMPLETE，以及完整结果计数。

本轮完成后根据诊断和 Joint8 结果判断后续 LR/梯度累积改进；不自动扩大 calibration、改变 loss 或进行大规模 sweep。
