# 有限 prior 检索记录

2026-09-13。检索词：`"quantization" "policy prior" "TD-MPC2"`、`"quantization" "proposal" "model predictive control" policy`、`"TD-MPC2" "policy" "mismatch" 2502.03550`。本次返回中未找到直接匹配“actor PTQ 改变 proposal pool、FP scorer 固定”的文章；这不是不存在先例或 novelty 认证。

[TD-M(PC)²](https://arxiv.org/abs/2502.03550)研究 learned prior 与 planner 的 policy mismatch 及 value overestimation，采用 policy regularization。它提醒本实验的 FP learned score 也可能有系统偏差；当前 screen 不把较高 J 当成真实 return。当前干预是固定 checkpoint 的数值 PTQ 和一次内部 update，不训练新 prior，也不声称解决该文的问题。

[Pinned official planner](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/tdmpc2.py)确认 policy proposals、random candidates、top-k weighted update 及最终随机 elite selection 是不同步骤。协议仅使用前三步，复评 μ 作为诊断。无需为本轮最小筛选补做广泛 survey。

与本项目已运行的 TDQ/value-head-gauge 的区别：旧实验固定候选生成，改变 Q 评分；本案所有评分保持 FP，改变24个 proposal slots，并用同24个 random slots控制。与 CEM-Update 的区别亦为干预位置，不是改一个 metric 名称。
