# Round2 独立复核与停止决定

2026-09-13。保留agent的原始提案及来源审查；以下是root在任何实验前的判断。两条均不申请GPU，不标经验失败。

**Residual branch interaction：identifiability_no_go。** 当前 gate 不能把 endpoint non-additivity 唯一归因于所测 local attention/FF covariance。DINO-WM block 的 FF 输入已经含 attention update，单独量化 attention 会改变 FF 的输入和 downstream Jacobian。即使在同一FP状态测两个local delta，最终 `E_Q²−E_A²−E_F²` 也混合了状态路径变化、非线性传播和误差内积。token sign/permutation null还会改变token结构，保持delta范数不足以给出同成本、同功能的counterfactual。当前“byte匹配attention/FF family”要求也没有与具体module参数量绑定。

因此，现提案尚缺一个能将观察结果接到明确quantization方法或planner决策上的可反驳预测。内部non-additivity本身不是足够的研究结论。本次不追加Jacobian或跨层干预去补完整验证，按用户要求保留这个设计层面的no-go。此判断不是“所有branch interaction都不可研究”。

**Hidden-basis permutation invariance：engineering-only_no_go。** 保留为实现审计想法，但不作为科学候选执行。逐元素GELU与配对row/column permutation满足函数等变；当前per-output absmax RTN也没有预期的科学变化。通过主要验证实现正确，失败主要说明实现问题，不构成本次要寻找的新机制。

两项都进入总表，原始提案中“不进入candidate表”的建议不采用，因为用户明确要求全部保留与明确结论。原始 [ROUND2_PRIOR_GATE.zh.md](ROUND2_PRIOR_GATE.zh.md) 不覆盖、不删除。
