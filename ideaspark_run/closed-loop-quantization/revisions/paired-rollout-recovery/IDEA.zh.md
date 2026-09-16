# Paired-Rollout Recovery：从传播匹配转向轨迹恢复

2026-09-12。FRT no-go 后的新研究假设。**状态：有条件的 WM quantization 研究方案，尚未运行；不是已认证的新方法。** 旧 FRT 和 `experiments/frt-ccds` 的结果保留，不重写其 no-go。

## 一句话

让量化 world model 在读到自己预测偏移后的 history 时，学会接近同一动作序列下 FP world model 的正确参考轨迹；同时检验，这种恢复能否只靠 quantizer calibration 实现，还是需要有限的权重适配。

这里的“正确”仅指 **FP reference trajectory**，不是环境真值。FP 本身的模型偏差仍然存在，最终必须用环境中的规划结果检查。

## 为什么更新

旧 FRT 的目标是保持 FP 对冻结 Q0 residual 的有限差分响应。Stage B 中，它相对随机方向在 Q0 transport 上改善约 26%，但 fresh-union 只改善约 0.30%，不满足预注册门槛。这支持停止旧配方。

不过，旧 `_fresh_bank` 同时改变了 source model、history context 和预测深度；它不是一个只替换 residual direction 的实验。因此，“方向不稳定就是失败原因”仍是待检验解释。新版不依赖这个解释成立，也不靠增大旧 λ 来挽救原假设。

新的问题是：**当历史已经偏离时，模仿 FP 在这个偏离状态上的反应，是否就是我们想要的目标？** FP 可能忠实地传播这个偏移。另一个可检验选择是把下一步拉回配对的 FP 路径。

## 精确定义

记 `P` 为冻结 FP32 predictor，`Qθ` 为量化 predictor。完整 history 的 shift、action replacement 和 observation/proprio mask 沿用已验证的 DINO-WM adapter。训练起点相同，动作序列完全相同，首个 screen 只有 H=2；不允许 FP/Q 两边分别用自己的 planner 选动作来生成配对目标。

冻结一个 hard W4 donor `Q̄`，从同一起点执行同一 action prefix，得到：

\[
X^P_1=\operatorname{Advance}_{P}(X_0,a_0),\qquad
X^{\bar Q}_1=\operatorname{stopgrad}(\operatorname{Advance}_{\bar Q}(X_0,a_0)).
\]

同一个 corrupted input 有两个不同的教师目标：

\[
y_{local}=P(X^{\bar Q}_1,a_1),\qquad y_{recover}=P(X^P_1,a_1).
\]

`local` 让 student 在 donor 状态上模仿 FP；`recover` 让 student 从 donor 状态接近配对的 clean FP 下一步。二者的 input、action、batch、teacher、量化参数和训练预算完全相同，**只改 target**。

\[
L_{target}=L_{clean}+\gamma\,\mathbb E\|W_z[Q_\theta(X^{\bar Q}_1,a_1)-y_{target}]\|^2+L_{round}.
\]

`Lclean` 使用相同的 FP history/target，防止牺牲正常 dynamics。`Wz` 在新 CAL 上计算并冻结，action coordinates 为零。第一版 γ=1，不根据新 DEV 搜索；两个目标使用同一单位和同一 γ。不是沿用旧 transport/clean 的 631.8 比值或裁剪规则。CAL 必须记录各项数值和梯度是否真正生效，未有效优化只记 inconclusive。

一个解释用的恒等式是：

\[
Q_\theta(X^{\bar Q}_1,a_1)-P(X^P_1,a_1)
=\underbrace{Q_\theta(X^{\bar Q}_1,a_1)-P(X^{\bar Q}_1,a_1)}_{local\ error}
+\underbrace{P(X^{\bar Q}_1,a_1)-P(X^P_1,a_1)}_{reference\ propagation}.
\]

这解释了为什么只减小 local error 未必最有利于恢复路径。它不是新理论、因果证明或稳定性保证，也不能只靠两个项相互抵消就声称方法有效。

## 两种参数自由度，一套最终部署格式

第一种是旧实验的 **per-output-channel scales + per-weight rounding variables**，基础 FP 权重冻结。第二种在同一参数化上增加 **rank-4 LoRA**，只作用于同一 24 个 predictor Linear；两类最终均合并并重新量化为相同 signed W4 网格，部署不保留 LoRA、FP teacher 或额外反馈状态。

形成最小 2×2：

| | local teacher target | paired-clean recovery target |
|---|---|---|
| quantizer-only | A | B |
| quantizer + LoRA-r4 | C | D |

先看 B−A 是否支持修改目标；再看 D−C 是否同样支持 recovery。若只有 D 好，但 C 同样好，证据支持一般权重适配，不能归功于 recovery。只有四格都完成，才能讨论参数自由度与 target 的 interaction。

增加 LoRA 的分支属于 quantization-aware adaptation/distillation，不能把四格统称纯 PTQ。即使 D 优于 B，也只说明所测优化设置下多了一个有效适配路径，不证明 quantizer-only 在数学上不可能恢复。旧 FRT 有 per-weight rounding，表达能力不同于仅学习 scale/zero point 的方法。

## 与已有工作的真实关系

- [Hallucinated Replay / Self-Correcting Models，AAAI 2017](https://arxiv.org/html/1612.06018)：从模型生成状态预测配对参考状态已有明确先例；其理论假设不能直接搬到本 DINO-WM/CEM 配置。
- [CTEC，AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/34039/36194)：Eq.14 和 Algorithm 1 已有跨步量化纠错与 LoRA 蒸馏；Table 3 提示其 diffusion 设置下 calibration-only correction 可能恶化。它是直接方法来源与风险证据，不能称第一次让量化模型纠错。
- [AccuQuant，NeurIPS 2025](https://arxiv.org/html/2510.20348v1)：已有多步量化轨迹对齐与低内存实现，不能把短 rollout、stop-gradient 或低内存本身当作独有创新。

**候选贡献只能是受控的 WM 适配问题：在固定 hard W4 部署格式下，恢复目标与可训练自由度如何影响 action-conditioned rollout 和规划。** “hallucinated replay + quantization”或“加 LoRA”本身不构成新原理。若只复现前作现象，结果应定位为领域适配/复现实验；要发展成新的方法论文，还需要独立且可复现的 WM-specific 发现。

## 最小路线与停机

1. 用旧 checkpoint 做同 context/同 depth 的轻量诊断计划；不把旧 DEV 作为新版确认数据。
2. 新 CAL/DEV 上运行 A/B/C/D，H=2，同一 donor bank、同一动作和配对 seeds。以最终 hard checkpoint 的 **自身自由 rollout** 为主指标，固定 donor 输入误差只作诊断。
3. 有信号才补等范数随机 history 的相同恢复目标、direct two-step distillation 和 action-response 检查。不能只和旧 FRT 比较就宣称新方法成立。
4. 只有最强对照下仍有信息，才进入独立 TEST 的 CEM action/环境 endpoint；训练输入的 action replacement 不代替测试模型是否保留了动作影响。

若恢复目标没有稳定改善 held-out hard rollout，或者收益仅存在于 donor bank、被随机扰动/普通两步蒸馏解释、或压平了真实动作响应，停止该配方。失败后不自动扩大 LoRA rank、horizon 或 DEV 调参。

完整预注册草案见 [EXPERIMENT_PLAN.zh.md](EXPERIMENT_PLAN.zh.md)。首个 screen 规划 ≤6 allocated GPU-hours，最多4×A100，优先单卡；现有 V100 只能作为独立硬件分支，不能换算成 A100 时间保证。所有数字是规划，未启动实验。
