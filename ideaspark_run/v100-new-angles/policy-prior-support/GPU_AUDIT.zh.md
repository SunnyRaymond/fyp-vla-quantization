# Policy-prior support：GPU implementation audit

日期：2026-09-13。审查冻结 `PROTOCOL.zh.md`、`policy_support_screen.py`、`policy_support_gpu.sh`、CPU replay 与 64833 的小型 engineering/summary receipt；未运行本地数值、未连接 cluster、未读取 raw/model 大文件。

## 结论

当前 producer 的窄实验语义可执行，64833 已完成 8/8 states，receipt 显示真实 `UGGPU-TC1` 的 `tc1n02`、Tesla V100、torch 2.6.0+cu124，且 strict load、FP no-op、actor restore、non-actor unchanged、RNG pairing、μ `num_samples=1`、common-score 与 config gates 全为 true。状态仍是 **CPU verification pending**；不能据此直接给 science go/no-go。

## 更正与实际检查

先前把 `policy_support_screen.py:166` 判为 `record` 未绑定是误读：`record` 是 generator comprehension 的局部绑定，第一次 scoring 不会因此触发 `NameError`。该错误已在本文件中明确撤销，不要求重跑 64833。

静态核对通过的关键路径：`propose()` 使用官方 H3 `model.pi → model.next → model.pi`，FP/W4 policy slots 共用 seed，后 488 random slots 三臂复用；W4 只替换 `_pi.0/.1/.2.weight`，scoring 前恢复 FP；FP scorer 保留原 cfg=512，μ scorer 用 shallow cfg=1，避免 official `_estimate_value` 无条件 `[num_samples,1]` termination 的 batch 广播问题；terminal `pi`、two-Q random pair 与候选 score 每臂重置同一 scorer seed。该语义与 pinned [TD-MPC2 planner source](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/tdmpc2.py) 一致。

## 交接风险（不是 GPU 科学阻断）

producer 当前已保存 `engineering.json`、`runtime.json`、`rng.json`、`raw_policy_support.npz`、`quant_snapshot.pt` 及哈希/restore receipts；现有 verifier 草案曾要求额外 aliases 或嵌套字段，而 producer 使用独立文件与现有顶层 receipts。CPU adapter 必须按实际 producer schema 校验这些文件，或在 verifier 中建立明确的一对一映射，不能把旧 handoff 字段差异解释成 GPU 失败。CPU replay 仅复算 top64/weight/μ/mass 与 gate，不重跑 model inference。

最终边界：64833 可进入 guarded CPU verification；通过后结论仍限于 FP scorer 下的第一次 internal update，不代表 closed-loop return、泛化或 native low-bit deployment。
