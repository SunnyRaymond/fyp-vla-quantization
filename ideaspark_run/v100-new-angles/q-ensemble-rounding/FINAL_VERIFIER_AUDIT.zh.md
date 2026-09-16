# Q-ensemble verifier：最终静态一致性审查

**日期：2026-09-13**  
**范围：**只读核对 `verify_tdq.py`、`tdq_screen.py` 与冻结 `PROTOCOL.zh.md`；未加载模型、未运行数值、未连接集群。结论不重复此前的 RTN floor 讨论；当前协议明确使用 RTN 的 absolute MSE comparison，不把 RTN 作为分母。

## 判定

**未发现会阻止 CPU 复算或改变科学 gate 结论的 blocking bug。** Raw axis、all-pair metric、8-state joint gate、producer transaction 与 Q alias contract 当前一致，可以按现有 verifier 继续执行。

## Raw schema 与 NumPy 轴

- `tdq_screen.py:999-1006` 写入 `member_q[state, rounding_seed, arm, candidate, member]`，shape 为 `(8,3,4,64,5)`；`official_avg` 为 `(8,3,4,64)`，`official_pair` 为 `(8,3,4,2)`。`verify_tdq.py:32-48` 对这些 shape、arm/seed/order、finite 值与 CUDA RNG layout 做了 fail-closed 检查。
- `tdq_screen.py:838-857` 的 producer 临时数组是 `[state,candidate,member]`，`_copy_transaction` 在 `:880-893` 按 seed/arm 写回；因此 raw 轴顺序没有把 seed 与 arm 对调。
- `verify_tdq.py:68-72` 以每个 state/seed/arm 记录的同一 pair 重建 official average，并检查所有 arm/seed 的 pair schedule 一致；`member_q[:, :, :1]` 作为 FP reference 的 broadcasting 也与 FP/RTN 复制到三个 seed slot 的 producer 逻辑一致。

## A/M/C 聚合与 joint gate

`verify_tdq.py:74-82` 先在最后的 member/pair 轴上计算误差，再以 `mean(axis=(1,3,4))` 聚合 seed、candidate 和 member/pair，结果 shape 为 `(8,4)`（state × arm）。这与 protocol 的“每 state 内等权平均 64 candidates 和 3 rounding seeds”一致。`C` 是未中心化的 `e_i e_j` moment，且 `A=(M+C)/2` 仅作 FP64 bookkeeping check，没有被误当成独立科学证据。

`verify_tdq.py:86-114` 的 `attribution_pass` 同时要求该 state 非退化、member fairness、`A_strat<A_ind` 和 cross-term attribution≥75%；`attribution_count>=6` 因而确实是同一个 joint state 集合，不会把“改善 state”和“binding state”拼成两个不同集合。`median_gain` 在 `all_nondegenerate` 时取全部 8 states；`A_ind/M_ind≤1e-8` 会整体进入 `inconclusive_degenerate`。实际 gate 使用 8-state median、至少 6 个 joint state，未发现轴或 subset 偷换。

`practical`（`verify_tdq.py:109`）只比较 `mean(A_strat)` 与 `mean(A_RTN)`，并要求至少 6/8 个 state 的 absolute `A_strat≤A_RTN`；没有引入 RTN 分母，符合最终冻结 protocol，即使 `A_RTN=0` 也不会产生除零伪 gain。

## Producer identity 与 transaction

- `tdq_screen.py:504-562` 对 live/detach/target Q state 做一对一检查；官方 Q MLP 的 10 个 tensor entries（两层 `NormedLinear` 的 weight/bias/LayerNorm 加 final Linear 的 weight/bias）由 `verify_tdq.py:153-155` 以 alias/storage 关系核对。三组实际被量化的 Linear weight shape 在 `:151-152` 精确绑定为 `[5,512,513]`、`[5,512,512]`、`[5,101,512]`。
- `verify_tdq.py:166-195` 要求恰好 8 个 transaction label：FP32、RTN、两种 SR 各三个 seed；每个量化 transaction 必须有三个 weight records、固定 W4/grid、相同 pre-weight 与 scale hashes，并保留 SR uniform/permutation hashes。与 `tdq_screen.py:1076-1084` 的执行/复制顺序一致。
- source/checkpoint/config、FP terminal cache、manifest/hash、strict load、V100、target/bypassed digest、8 个 official API transaction 和 candidate replay 均在 `verify_tdq.py:124-165` 做 producer checks；工程检查失败时最终 `decision` 为 `implementation_inconclusive`，不会伪造 scientific no-go。

## 最终建议

当前不需要修改 root files，也不需要增加 state、seed、pair schedule 或 full validation。CPU verifier 的输出应同时读取 `numeric_decision` 与 `decision`：前者是 raw 数值 gate，后者还要求全部 producer checks 通过；若二者不同，应报告工程/复现不完整，而不是重解释科学结果。
