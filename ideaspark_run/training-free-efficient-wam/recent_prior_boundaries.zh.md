# Recent training-free WAM inference priors：重复机制边界

**核查截止：2026-10-02。** 范围为四篇 arXiv 主文；官方代码仅确认到 Efficient-WAM。文中明确机制由主文支持，复现缺口仅限本次检索范围；区分原模型训练与部署时加速。

## 机制签名速查

- **WAMachine**：跨 replan 搬运 denoising trajectory；执行动作时预计算并在真实 observation 到达后 rebinding；shallow probe 验证后复用层状态、跳过中间层。
- **Efficient-WAM**：video/action 使用不对称 denoising budget；视频前缀完成后缓存 video K/V，剩余步骤只跑 action。Compact video expert、低分辨率 future latents 属于另两项模型/表示压缩。
- **FBFM**：执行中逐步到达的真实 state latent 作为动态 masked constraint，以 endpoint-Jacobian/pseudoinverse 修正当前 flow；同时固定约束跨 chunk 的已承诺 action overlap。
- **GeoBoN**：便宜的 action–future 一致性 gate 决定是否增加采样，再以多视角 depth reprojection 几何一致性做 Best-of-N 排序。

## WAMachine — 2609.34608

**对象与训练属性：** 冻结 WAM、training-free。Trajectory Remapping 将上轮中间 latent 表成 denoised endpoint 加归一化 denoising direction，按时间槽对齐后初始化新 replan；未映射槽使用新噪声。Observation Rebinding 在动作执行期基于预测 future 预计算 denoising prefix，实测 observation 到达后过一致性门限才续算，否则回退。Residual Rescaling 用 shallow probe 验证缓存 layer state，符合条件就跳过中层。

**重复边界与证据：** 跨 replan latent 搬运、执行期 speculative denoising/rebinding、probe 后缓存 hidden/residual 并跳层，都已是明确机制签名。对照 Native、RTI-DP、RTC、VLA-Cache、BAC；Cosmos Policy/Fast-WAM-IDM/Motus 在 LIBERO 与 RoboTwin 2.0 **仿真闭环**评测；无真实机器人 task。它不改 source predictor，也不提供 physical deployment 证据。主文未见作者官方代码链接。 [arXiv HTML](https://arxiv.org/html/2609.34608)

## Efficient-WAM — 2606.10040 v3

**对象与训练属性：** 整体**不是** training-free：三阶段训练包含 WAN-2.2 视频 expert 压缩/蒸馏、action training 与 joint refinement；稀疏/低分辨率 future 是独立降本项。仅 asymmetric video-action denoising 是 training-free：视频跑 2 步并缓存各层 K/V，之后只更新 action（总 action 预算 5 或 10 步）。把该不对称 schedule、视频 K/V 复用或粗 future latent 换名都算重复。

**证据边界：** RoboTwin 2.0 对比 π₀、π₀.₅、Motus、Fast-WAM 等（主表 baseline 数字引自既有论文/报告）；另在两种双臂平台做五项真实任务，RT 65/100、Motus 64/100、π₀.₅ 60/100；RTX 4090 为 98 ms/chunk。真实 baseline 各自用相同任务数据 fine-tune；时延不含 observation acquisition 与 robot execution。source 是训练所得模型，只有采样 schedule 属 inference-only。 [arXiv v3 HTML](https://arxiv.org/html/2606.10040v3) · [作者代码](https://github.com/jiajun613/Efficient-WAM)（含三阶段训练及硬件无关 real-robot template；模板不含平台 SDK）

## FBFM — 2607.29235

**对象与训练属性：** 冻结原 WAM、training-free inference。执行上一 action chunk 时，新实测 observation 被编码成时间对齐 state latent；每次 flow solver 调用更新 state target/mask，以 masked pseudoinverse/endpoint-Jacobian 修正当前速度场；上一 chunk 已承诺的 action overlap 是固定约束。Stage-wise 与 joint-generation 分别注入，不能泛化成只刷新 history/KV 的 chunk-boundary conditioning。

**证据边界：** LingBot-VA/RoboTwin 2.0 对 Base 的 task-configuration macro 为 83.1% vs 80.1%；DreamZero/LIBERO 四套 pooled gain +0.6 pp。真实机器人部分仅回放一次 ball-stopping 录制的 120 帧 RealSense observation，比较 Wan2.2 预测画面；论文明确是 state-prediction diagnostic，**不是** real-robot closed-loop success。伪异步 clock 也不等价于真实 wall-clock latency。source checkpoint 不变，贡献全在 inference feedback；主文/arXiv 未见作者官方代码链接。 [arXiv HTML](https://arxiv.org/html/2607.29235)

## Test-Time Scaling / GeoBoN — 2607.17454

**对象与训练属性：** 冻结公开 WAM checkpoint 与 VGGT-Ω，training-free、无需 success labels。先用 optical flow 与 projected end-effector motion 检查 action–future 一致性；仅 gate 判不可靠时再采样，以跨 camera-view 的 depth reprojection inconsistency 排序候选。GeoBoN 固定预算 Best-of-N 与 Gated GeoBoN 条件加算均已覆盖；几何 verifier、action–future gate、BoN 组合不能仅改名申报。

**证据边界：** Cosmos Policy、X-WAM、LingBotVA、Motus 在 RoboCasa、LIBERO Long、RoboTwin 2.0 做仿真闭环和 offline candidate diagnostics；无 physical robot trial。单次 rollout、随机同触发率 gate、固定 N=8 作对照；Gated 只触发约 26.2%，恢复约 74.8% always-on gain，N 增大仍可能饱和/退化。source WAM 不变，额外推理有时延且排序信号可能误选。主文/arXiv 未见作者官方代码链接。 [arXiv HTML](https://arxiv.org/html/2607.17454)


