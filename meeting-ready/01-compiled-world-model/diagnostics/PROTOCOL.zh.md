# Compiled World Model：无重训机制诊断协议

## 目的

保留 job `25309513` 的 predictor-level NO-GO，不重训、不扫 rank、不改 loss。本轮只定位三个问题：

1. B3 的误差主要来自 candidate-shared common component，还是 action-dependent contrast；
2. 冻结的 `phi(u)` 是否能在一个 context 内通过局部 affine coefficients 跨 candidate bank 泛化；
3. 一次 deployable teacher reference 能否把现有 student 变成可用的 anchored residual predictor。

## 数据

- 未见 context：沿用 development `valid[552:560]`，两组原 candidate seeds，共 48 blocks。它只作探索性诊断，不重新包装成 final test。
- 训练 context 原候选：anchor-aligned training rows 的前 24 个 contexts，确保 early/middle/late 各 8 个，每个 64 candidates。
- 训练 context 新候选：相同 24 contexts，沿用相同 action distribution，用冻结的新 seeds 生成另一组 64 candidates，并在 compute node 生成 teacher labels。

## Diagnostic 1：common/contrast decomposition

对 teacher 与 student 分别按同一 context、horizon、candidate bank 计算 mean 与 centered residual。记录 common MSE、contrast MSE、relative contrast error、student/teacher contrast energy ratio `gamma`，并验证 MSE decomposition 数值闭合。

使用 official criterion 比较 raw prediction、`teacher mean + student residual`、`student mean + teacher residual`。后两者均为 oracle，只用于定位，不是部署结果。

## Diagnostic 2：冻结 phi 的跨-bank局部拟合

只在 B3 terminal horizon 上，用 bank A 的 teacher latents 对 `[1,phi(u)]` 做固定 ridge affine fit，再在 bank B 测试；随后反向。intercept 不正则化，ridge relative lambda 固定为 `1e-4`，不依据结果调参。exact cross-bank duplicate candidates 从 test bank 排除。

若 fit bank 好、cross bank 也通过冻结 gate，才支持“phi 有用而 compiler 是主要瓶颈”。它仍是 teacher-paid local oracle，不是部署方案。

## Diagnostic 3：anchored residual

参考动作为全零合法 packed action sequence，与 expert future、candidate labels 和结果无关。每个 fixed observation 只运行一次 official LeWM reference：

`teacher(c,u_ref) + student(c,u) - student(c,u_ref)`。

对 B1、B3 与旧 anchor-aligned student 使用同一 evaluator。B3 timing 包含一次 teacher reference、context prepare、student reference 和 30×300 candidate queries；不得从 timing 中删去 reference cost。

## 决策边界

- anchored quality、timing 和 common-oracle recovery 同时通过：只形成 01-v2 predictor-level GO signal，之后仍需重新冻结 official CEM。
- local phi cross-bank gate 通过但 anchored 不通过：支持 compiler bottleneck 研究，不支持部署。
- action oracle 才恢复或 local phi 无法 cross-bank：当前 action response/shared-phi 参数化仍是瓶颈。
- 没有清晰正信号：停止当前 shared-phi recipe，不加 router、mixture、fallback 或更多 rank。

本轮不运行 official CEM、closed-loop，不生成新的 confirmation claim。
