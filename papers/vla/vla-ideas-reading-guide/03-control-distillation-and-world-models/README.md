# 方向三：把控制知识装进 VLA

原文：[VLA_idea.pdf p.3](../../../../idea/VLA_idea.pdf) · [三个方向总览](../README.md)

**12 个编号条目，12 份 PDF。** 两条阅读线在这里汇合：`MPC / control structure → neural policy`，以及 `task-relevant world state → predictive control`。必须明确 inference 时到底保留什么。

| # | Paper / notes + PDF | 贡献 | Inference 中的关键边界 |
|---:|---|---|---|
| 1 | [MPC-Guided Policy Search](01-mpc-guided-policy-search/README.md) | 完整 state 的 MPC 教 partial-observation neural policy | student 可以独立执行 |
| 2 | [MPC-Net](02-mpc-net/README.md) | control Hamiltonian guidance；real robot | 已有 MPC → fast neural policy 的先例 |
| 3 | [Differentiable MPC](03-differentiable-mpc/README.md) | 对 controller fixed point 的 KKT differentiation | MPC 仍是 policy class |
| 4 | [DPC + CBF](04-dpc-cbf/README.md) | differentiable objective 直接训练 policy | safety theorem 涉及 online intervention |
| 5 | [BarrierNet](05-barriernet/README.md) | 可联训的 higher-order CBF / QP layer | safety layer 保留；本地为前身 preprint |
| 6 | [ABNet](06-abnet/README.md) | explicit-barrier closed-form structure；ICML 2025 | 不需 iterative QP 不等于无 safety structure |
| 7 | [TD-MPC2](07-td-mpc2/README.md) | decoder-free control-oriented world model | latent planning 仍在 inference |
| 8 | [LaWAM](08-lawam/README.md) | compact latent visual subgoals 驱动 VLA | 避免 future video reconstruction |
| 9 | [WorldVLA](09-worldvla/README.md) | action / image generation 联合建模 | pixel-world-model 对照 |
| 10 | [GPC](10-gpc/README.md) | frozen policy + predictive look-ahead | 部署时仍做 planning / refinement |
| 11 | [DiffOG](11-diffog/README.md) | differentiable trajectory refinement | 优化层是系统组成部分 |
| 12 | [VLSA / AEGIS](12-vlsa/README.md) | CBF safety layer + SafeLIBERO | VLA-specific online safety baseline |

## Primary route（约 4–5 hours）

MPC-Net 60 min → DPC 60 min → TD-MPC2 45 min → LaWAM 45 min → VLSA 45 min。需要推导 gradient 时插入 Differentiable MPC；需要辨析 “不用 optimizer” 时读 BarrierNet → ABNet；需要 teacher baseline 时读 GPC / DiffOG。

## 两个需要在开题前纠正的地方

1. **历史边界。** MPC-Guided Policy Search 和 MPC-Net 已把训练时的 MPC 信息转入独立执行的 neural controller；TD-MPC2 与 LaWAM 已说明 world model 不必生成未来图像。因此新问题需落在 VLA、contact-relevant state、model uncertainty、控制目标或可验证的 transfer 条件。
2. **Guarantee 边界。** DPC 并非仅靠 imitation 来“蒸馏 MPC”；其直接 differentiable training 与 online CBF 是不同部件。BarrierNet/ABNet 将安全结构放入网络，不代表把结构去掉后仍继承 theorem。若 goal 是 student-only inference，要直接测 teacher-off degradation，而非引用 teacher 的保证。

## 本方向对照卡（留空）

| Paper / system | Teacher / objective | World state | Inference 还剩什么 | 保证的条件 | Student-only task success / violations |
|---|---|---|---|---|---|
| MPC-Net | | | | | |
| DPC | | | | | |
| LaWAM | | | | | |
| VLSA | | | | | |
| 我的构想 | | | | | |

不要把 smoothness、collision avoidance、contact stability 与 task completion 合并成一个“安全”指标；也不要把 empirical zero failures 写成 formal guarantee。

## 扩展阅读

原总结中的 RynnVLA-002、World Model survey、SafeFlow、Diffusion Policy 作为 [扩展入口](../REFERENCE_AUDIT.md) 保留。已有主库 [Flow Matching](../../reading-guide/papers/17-flow-matching/README.md) 与 [π0](../../reading-guide/papers/02-pi0/README.md) 可补 generative action expert 的背景。
