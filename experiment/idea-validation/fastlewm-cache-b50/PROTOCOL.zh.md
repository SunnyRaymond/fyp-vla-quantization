# Fast-LeWM exact cache：B=50 固定 solve 与闭环验证

## 问题

旧 cache pilot 在 8 个 B=1 fixed contexts 上实现了 bitwise-exact trace 和约 57% 的 median solve latency reduction。本实验只验证这些收益能否扩展到真实 50-environment 负载，并检查缓存完整闭环时是否保持每次 replanning 的 actions、costs 与 task outcomes。

## 冻结配置

使用官方 Fast-LeWM checkpoint revision `f95379fe193c8bfc6a59c9d8437d5052bd72ff71`、source `de3e9dac539f5bbe6ff1656a2fb00938d62a3c7d`、isolated `stable-worldmodel==0.0.6`，沿用原 50 个有序 task rows。保持 H=1、action block=25、history=1、beta=0、CEM 300 candidates / top-30 / 30 rounds、batch size 1 与 seeds 42/43/44。三个 seed 是同一批 50 tasks 的重复测量，不合并成 150 个独立样本。

缓存只覆盖同一次 `solver.solve` 内 action-free 的 goal/current encoder 结果。每次 solve 和每次 closed-loop replan 都清空缓存。每个条目同时绑定当前 solve 内的 environment slot、current/goal 语义角色和实际 pixels tensor 内容；缓存命中检查 shape/dtype/device 后逐字节比较输入。角色映射由 pinned Fast-LeWM `get_cost` / `rollout` 源路径核实，并且在运行时对输入内容作断言。不能仅依赖 encode 调用奇偶、shape 或 tensor pointer。CPU allocation source 确认 CEM 是 env-major：`batch_size=1` 时每个 environment 连续运行 30 轮；缓存随已验证的 slot 转换淘汰旧条目，最多驻留 current/goal 两项，不保留 candidate-expanded images。若 native solver 的环境调度与预检得到的映射不符，缓存拒绝继续。

## 固定 observation 阶段

对每个 seed，从官方评估第一次 `solver.solve` 捕获真实的 50-env observation 和 init action，要求 `active_environments=50`。每个 context/arm 做两次 warmup；再做六次配对计时，AB/BA 各三次。完整 trace 单独运行一对，不计时。缓存创建计入 cache 的 solve latency；warmup、输入克隆和 trace 采集在 timed interval 外。保留 native cost logging，不额外插入每次 cost 的 GPU synchronize。

每个完整 solve 检查 1500 次 native `get_cost` 调用、450,000 个候选评分、有限输出和 per-environment cache contract。Trace 对每一轮的 candidate action 与 cost tensor、最终 actions/costs 比较 dtype、shape 和原始 tensor bytes。要求全部逐位相等、paired median latency reduction 至少 10%、cache/native peak allocated memory ratio 不超过 1.10。

## Closed-loop 阶段

若固定阶段 trace fidelity 通过，就按冻结顺序完成三组 native/cache 50-task evaluation，每组都使用相同 seed 和相同 task rows。顺序为 seed42 `native→cache`、seed43 `cache→native`、seed44 `native→cache`，在三个 seed block 上尽可能平衡 AB/BA。缓存每次 solve 都重置。逐 solve 记录 active count、cost-call/candidate-score 数和最终 selected actions/costs；逐 task 保留 50 个 outcomes。每个 seed 的 native outcomes 对照已完成主实验同 seed 的50项向量。Closed-loop 的 paired actions、costs、outcomes 必须 exact；报告 raw evaluation wall time，并单独报告 runner 实测的 fixed-benchmark overhead。复用的 official runner 会在首个实际 solve 前做1次warmup和3次timed cloned solves，计入raw wall，并保留实测overhead；不估算扣除，也不从固定 solve 的加速比外推。

## 停止规则

固定阶段若 trace fidelity 或 finite 门失败，停止闭环；只允许不改变冻结配置的执行错误修复。固定 trace 通过后，即使 latency 或 memory scientific gate 未过，仍完成所有三组闭环配对；仅实际 OOM、资源状态不安全或其它技术失败可停止。不得调参追求 PASS。若 CEM encode 环境身份/角色无法在真实运行时被明确绑定，先停止并交 root 决定，不进行可能跨环境或跨 goal/current 的缓存复用。

## 证据边界

即使所有门通过，结论也仅覆盖指定 checkpoint、50 个 task rows 和本 evaluation harness。不能把 B=50 固定输入加速等同于普遍部署收益，也不能与既有不同模型的约 5× planner time ratio 相乘。此实验不涉及训练、CEM 预算变化、模型结构或 sparse kernel。
