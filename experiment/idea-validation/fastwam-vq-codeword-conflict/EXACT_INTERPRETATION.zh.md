# 全成员梯度检查：当前解释

`analyze_exact` 作业 `25846164.pbs101` 已完成并验证：8 个 calibration contexts、4 个模块、32 个 book0 codewords。256 条 codeword/context 记录的全成员梯度之和均通过与实际 autograd codeword gradient 的一致性检查。选中 support 共 1,118,810 个 weight vectors，零 scale 成员数为 0。

## 共享子空间确实很窄，但数值小本身不是坏 assignment 的证明

下面先对全部 8 个 calibration contexts 的 decoded-weight gradient 取均值，再投影到固定 book1/scales、仅允许这些 book0 codewords 更新的子空间。报告的是 `||P g||^2 / ||g||^2`，不是实际 loss 或任务性能的保留比例。

| 模块 | 选中 vectors | 共享子空间中的梯度能量比例 |
|---|---:|---:|
| time_embedding.0 | 7,788 | 0.00226645 |
| text_embedding.0 | 113,907 | 0.0000847870 |
| blocks.24.ffn.0 | 498,894 | 0.0000185338 |
| blocks.17.ffn.0 | 498,221 | 0.00000906414 |

每个模块只放开 8 个 4D centroids，相对于大量独立 weight coordinates，自由度本来就很少。在 isotropic gradient 参考下，正交投影的期望能量比例是 `rank(P) / D`。因此不能仅凭上述比例接近零，就断言 assignment 比合理替代分组差。

全成员 cancellation 高于 32-member sample 也不能直接作为异常证据：分组成员数已从 32 增至数百至数万个。全成员结果不能直接与较小 sample 的随机分组 cancellation 作公平数值比较。全成员平均 pair cosine 仍接近零，不能解读为全部成员系统性反向。

## 有限扰动已固定，等待实际 forward

六组方向仅由 calibration 数据决定。全部使用相同 support，实际 decoded-BF16 relative norm 为：

| 目标 | Shared | Untied | Random |
|---|---:|---:|---:|
| 0.005 | 0.005006293 | 0.005000000 | 0.005000000 |
| 0.015 | 0.015001409 | 0.015000762 | 0.015000001 |

最大 norm 匹配相对误差约 0.126%，小于预定 5% 容差。零 scale 行不构成当前 untied 对照的干扰。

匹配的是全部选中 support 的总 norm，未逐模块匹配。Shared 的扰动能量几乎全部集中在 time_embedding；Untied 约三分之二在 time_embedding，其余分配到 text/FFN。这是放开共享约束后更新几何及能量分配的差异，不能把后续可能的收益单独归因于“反向成员更少”。具体能量及一阶贡献保存在原始 JSON 的 `norms.*.per_module` 中。

固定 book1/scales 且放开独立 weights 的 untied 对照增加了自由度。即使它在 held-out future/固定 reader 上更好，也不直接证明同 bits reassignment 已经有效。当前继续原定全部 60 contexts、360 conditions 的实际 native generation 和固定 BF16 reader 评估，结果尚未产生。

原始数据：[analysis.json](retrieved/analysis_exact/analysis.json)；全成员明细：[REPORT.zh.md](retrieved/analysis_exact/REPORT.zh.md)。
