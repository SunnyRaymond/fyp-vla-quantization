# 针对已生成机制的补充 prior 检查

2026-10-02；主线程在 generation 提交后做的定向 primary-source 阅读。不是 mandatory connector collision 的替代，也不更改 generation artifact。以下判断交由独立审查。

## ToPi — 2602.01609v1

- 来源：[Token Pruning for In-Context Generation in Diffusion Transformers](https://arxiv.org/html/2602.01609v1)，2026-02-02。
- 方法 §4.2–4.3：用 target-query 对 context 的 attention 与 value norm 汇总 token contribution；按累计贡献约束贪心选择 context 子集；anchor denoising steps 用全量 context 重新评分，其间复用选集；gather hidden states 和原位置索引。
- 范围：training-free Flux.1-Kontext / Qwen-Image-Edit，图像编辑；未报告 WAM action 或真机结果。
- 与当前候选的重合：context contribution ranking → 累计份额 cutoff → 后续 solver steps 复用 subset → compact execution。候选的 output-projected、按 camera/source group、action-query-only 删除与 ToPi 不完全相同；这些差别是否形成实质 WAM-specific mechanism，尚需审查。换掉评分细节或任务名不能自动支持方法新颖性。

## 其他 primary 近邻

- [VATP](https://arxiv.org/abs/2406.12335)：attention 与 value norm 联合估计 KV token importance；用于 LLM KV pruning。
- [CAPA](https://arxiv.org/html/2602.00247v1) §4.1 Eq.(2)：已把每 head 的 attention×value 经 W_O,h 投影、跨 head 求和后取 L2 norm，按这一 residual-stream contribution 裁剪视觉 K/V；另有 FFN approximation。用于 LVLM。当前候选不能把“加入 output projection”当成一般评分层面的新颖差别。
- [Contribution Weights](https://proceedings.mlr.press/v306/cunningham26a.html)：考虑 attention、value magnitude 与方向的 token influence；ICML 2026。
- [Attention is Not Only a Weight](https://aclanthology.org/2020.emnlp-main.574/)：EMNLP 2020 已用 transformed-value norm 分析 attention contribution。超出当前 48-month alias window，不能将窗口内未命中理解为这一思想从未出现。

## 对照基线所用 primary IDs

- ToPi：arXiv:2602.01609v1 — https://arxiv.org/html/2602.01609v1
- VATP：arXiv:2406.12335 — https://arxiv.org/abs/2406.12335（primary 网页与摘要已读；本轮没有经 host-ref resolver 注册，不能标为其 verified host_web record）
- CAPA：arXiv:2602.00247v1 — https://arxiv.org/html/2602.00247v1

## 待独立裁定

当前候选声称 r_lg 是贡献排序启发式，没有误差/因果保证。其需要证明的增量是 WAM-specific 的 action outcome 与净 latency，而不是 attention mass 与 value contribution 的一般区分。action queries 短、仅删除 observation QK/AV 且保留 video QKV/FFN 的结构，使可节省份额仍是未知；不得把一般 attention 方法在长序列上的速度外推为这里的收益。
