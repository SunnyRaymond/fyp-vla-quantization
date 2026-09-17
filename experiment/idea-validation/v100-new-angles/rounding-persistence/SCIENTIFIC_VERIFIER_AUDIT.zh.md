# `scientific_replay` 独立数学审查

## 范围与结论

本审查只覆盖 `rounding-persistence/verify_persistence.py` 的
`scientific_replay`，并与 `PROTOCOL.zh.md` 及
`new-cut-shortlist/PERSISTENCE_MATCHED_DESIGN_AUDIT.zh.md` 对照。没有运行
verifier、模型、数值实验或集群任务，也没有审查尚未提供的 producer adapter。

当前函数的数组轴、聚合公式、`S-D` 分解、无删除 gate 与 5/6 joint gate
基本忠实于冻结协议；但 raw schema 没有保存或验证 common FP path 的身份。
因此它可以重算已经给出的数值，暂时不能独立证明 `common_q_v` 确实是在
`raw_x` 的 FP trajectory 上得到的。若该证据不由 producer 的固定合同另行
提供，科学结论应保持 `implementation_inconclusive`，不能把结果解释为
frozen/cyclic persistence 机制的 preliminary go 或 no-go。

## 已核对且一致的部分

- `raw_x` 为 `(state=6, noise=2, arm=8, time=11, position=50, dim=32)`，
  `raw_v` 为对应的十步 velocity；`common_q_v` 为
  `(6, 2, draw=3, step=10, 50, 32)`。`noise`、`dt`、`schedule` 的形状分别
  为 `(2,50,32)`、`(10,)`、`(family=2,draw=3,step=10)`，与协议的两 noise、
  三 rounding draw、Frozen/Cyclic 两 family 相符。
- 初始状态通过把同一 `noise` broadcast 到所有 state 和 arm 后与
  `raw_x[...,0,:,:]` 比较；时间点为 `1, .9, ..., .1`，`dt=-.1`。Euler
  recurrence 使用 `x[...,t+1] = x[...,t] + dt*v[...,t]`，索引覆盖十个
  transition，未少算或多算一步。
- `expected_schedule` 固定 Frozen 为每个 draw 的同一 member，Cyclic 为
  `(draw+step) mod 3`。每个 step 的 `schedule` 都检查为 member `0,1,2`
  的 permutation，所以同一时间点的 member marginal 被保留。
- endpoint 使用最后一步的前 8 个 action positions 与前 7 个 physical
  coordinates，并以 arm 0 FP trajectory 为参照。`E_F`/`E_C` 对两 noise、
  三 draw、8×7 physical entries 求平均；`E_RTN` 只作报告，未被错误地
  加入协议 gate。
- common-path error 先做 `common_q_v - v_FP`，再乘 `dt`。`S` 是十步
  加权 error 的平方后平均，`D` 是逐步平方能量之和后平均，故行内的
  `S-D` 正是 cross-term（包含不同 timestep 的交叉项），不是另一个
  endpoint 指标。Frozen 与 Cyclic 的 stepwise marginal 以及 diagonal
  energy 的比较也按同一 physical slice 完成。
- `binding = (E_F > 1e-12) & (S_F > 1e-12)` 在每个 state 上计算；随后每个
  state 同时要求 `S_C <= .75*S_F` 与 `E_C <= .90*E_F`，最后以至少 5/6
  state 通过作 preliminary gate。六个 state 始终保留在 rows 中，没有按
  结果删除 state，也没有把 3 个 rounding seeds 当作独立 state。raw check
  失败优先得到 `implementation_inconclusive`，未被科学 gate 掩盖。

## 明确的科学可识别性 blocker：common FP path 未被 raw 验证

协议要求每个 `v_Q[d,t]` 在同一 `x_FP[t]` 上计算。函数却只用

```python
common_q_v - v[:, :, 0, None, ...]
```

构造 error；`common_q_v` 的 schema 没有 `common_path_x`、path fingerprint
或等价的输入 identity，`raw_x` 也只保存各 arm 的 free-running paths。
函数没有检查 common field 的输入是否等于 `raw_x[:,:,0]` 的 FP path。因此
即使 producer 把 common field 误算在另一条 path 上，schedule permutation、
marginal equality、`D` equality 和所有 endpoint recurrence checks 仍可能
通过，随后 `S`/`S-D` 的解释就不再是协议定义的 matched common-path
diagnostic。

解除该 blocker 需要在 raw contract 中提供可复核的 common path（例如
`common_path_x`，形状为 `(6,2,11,50,32)`，并与 `raw_x[:,:,0]` 逐值比较，或
提供由固定 bytes 计算且能绑定到该数组的 fingerprint）。在这项证据出现
并被 verifier 检查前，任何 `S_C` 相对 `S_F` 的改善都只能作为未绑定的
raw replay 数值，不能支持 persistence 机制 claim。该问题不是通过增加
rounding seed、删除失败 state 或调整 threshold 可以解决的。

## 较小的合同备注

函数只检查 shape 和 finite，没有强制 raw arrays 的 dtype 为协议要求的
`float32`。这不会改变当前公式的轴审查结论，但 producer 若写入
`float64`，会改变 cast 前的 subtraction/reduction 数值；建议把 dtype/生成
路径作为 engineering check 明确记录。`S-D` 当前已正确计算，无需另造
“covariance”指标；文档中应继续称它为未中心化 error product 的
cross-term。`endpoint_gain` 与 `forcing_gain` 对零分母有 `None` 保护，且
degenerate state 会先阻断决策，行为与协议一致。

## 审查判定

在补齐并验证 common FP path identity 之前：数学 replay 实现为
**formula-consistent but science-binding-incomplete**；应保留
`implementation_inconclusive`，不提交 GPU 输出的 preliminary go/no-go。
补齐后，现有 6-state、两 noise、三 draw 聚合和 5/6 joint gate 可直接用于
冻结协议，不需要扩展样本或改变主指标。
