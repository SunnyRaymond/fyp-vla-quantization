# Value-head common-mode removal：独立 prior audit

## 边界与结论

本审查只使用 pinned TD-MPC2 official source、TransformerLens official documentation 和 primary papers；没有读取本候选输出，没有运行模型、集群任务或数值实验。结论分开报告：**方法新颖性 no-go；窄机制诊断 conditional-go**。把 101-bin Q head 做 softmax-invariant centering 作为新 PTQ 方法不能成立，但可以用一次小规模 screen 检验“已训练 distributional Q head 的 gauge 代表是否会改变 row-wise W4 RTN 误差”这一尚未由先例回答的 WM-specific fact。

## 1. TD-MPC2 中假设是否可辨识

在 pinned `WorldModel` 中，Q head 是输出 101 个 distributional logits 的 MLP；构造后先统一 `weight_init`，再把 `self._Qs.params["2", "weight"]` 置零。官方 `init.py` 还把每个 Linear bias 初始化为零。因此 参数初始时最后 Q weight/bias 的 output-bin common mode 为零，但这不推出训练后仍为零。[WorldModel source](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/common/world_model.py)；[init source](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/common/init.py)。

官方训练使用 `Adam` 更新 `_Qs`，value loss 对每个 Q logit 使用 `soft_ce`；该损失先做 `log_softmax`，target 是和为 1 的 two-hot 分布。[training source](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/tdmpc2.py)；[math source](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/common/math.py)。所以对一个样本，最后层 logits 的梯度满足

\[
g_j=\sum_n(p_{nj}-t_{nj})h_n,\qquad \sum_j g_j=0,
\]

bias 梯度也有同样的零和性质；全局 gradient clipping 不改变这个性质。但 Adam 的每个 output-bin 都有独立的二阶状态，更新项是 `m_j/(sqrt(v_j)+eps)`，其和一般不为零。即使从全零最后权重开始，第一步也可能因各 bin 的梯度幅度不同而生成 common mode；后续数据、target 和 Adam 状态会继续破坏该表示约束。这是源码支持的可辨识机制推导，不是已观察到的 checkpoint 事实。

官方 `two_hot_inv` 对 logits 做 `softmax`、bin expectation、再 `symexp`；官方 Q `avg` 随机取两个 member 后再解码。因此对最后 Q layer 写成 `l=Wh+b`，令

\[
W_c=W-\mathbf1\bar W^\top,\qquad b_c=b-\bar b\mathbf1,
\]

则 `l_c=l-c(h)1`。在 exact arithmetic 中所有 probabilities 和 decoded scalar Q 不变。可是 W4 RTN 若按 output row 使用 `absmax/7`，每行 scale 与整数网格会改变；这正是本 screen 唯一的 WM-specific 可检验点。`symexp` 使 decoded-value fidelity 不能由 raw-logit MSE 代替。

## 2. 最近先例与重叠

1. TransformerLens 的 `ProcessWeights.center_unembed` 已明确实现“减去 unembedding weight 的均值”，并说明 softmax 的 translation invariance 保持 log probabilities 不变；它是与本操作同一数学变换的公开 code-level 先例。[official `center_unembed` documentation](https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.weight_processing.html)。因此“发现 softmax gauge / output-bin centering”本身是 **novelty no-go**。该先例主要服务 interpretability，未回答 distributional Q head 的 W4 row-absmax 后果。

2. OEC 论文把 output embeddings 写成 `e_i^*=e_i-\mu`，并证明 mean output logit 归零、probabilities/loss 不变，同时研究 logit bound 和训练稳定性。[Output Embedding Centering for Stable LLM Pretraining, arXiv:2601.02031](https://arxiv.org/abs/2601.02031)。它是更强的 conceptual prior，但场景是 LLM training stability，不是 frozen TD-MPC2 Q head 的 one-shot PTQ；不能据此声称 Q-specific effect 已被实证覆盖。

3. Softmax Bias Correction 研究的是量化造成的 softmax output bias，并将 correction 吸收到 quantization parameters；它与“先做 exact gauge transform，再改变 weight-only grid”相邻但不等同。[Softmax Bias Correction for Quantized Generative Models, arXiv:2309.01729](https://arxiv.org/abs/2309.01729)。它进一步削弱了泛化 PTQ novelty claim，却没有直接否定上述窄诊断。

## 3. 最小可证伪 screen（仅条件建议，不是冻结 protocol）

使用尚未用于本候选的 8 个 reset states（例如 seeds `5209..5216`），固定同一 H3 imagined inputs、64 candidate actions 和 official Q 的 10 个 unordered member pairs。四臂为 `FP-original`、`FP-centered`、`W4-RTN-original`、`W4-RTN-centered`；只改最后 Q Linear weight，bias 与 FP head 均按上述公式同步 center，W4 仍使用同一 row `absmax/7` 和 RTN。每个 arm 一次 forward；部署不增加 member 或 forward。报告 decoded Q、FP common-mode 范数、每行 scale/整数变化、weight MSE，但主误差必须是 10-pair decoded-value MSE。

预注册三个门槛：

1. **FP exact-no-op gate**：`FP-centered` 与 `FP-original` 的 decoded Q 和 softmax probabilities 在全部样本上满足 `allclose(atol=1e-5, rtol=1e-6)`；失败即实现/数值 structural no-go，不解释为科学差异。
2. **binding gate**：checkpoint 最后层的 `W` 或 `b` 有超过 float32 roundoff 的非零 common mode，并且 centering 在至少 6/8 states 的 row scale 或 quantized integer 上产生可记录变化；否则没有可测试的 representation-to-grid 机制，停止。
3. **falsification gate**：在同一 8-state 输入上，`W4-RTN-centered` 的 all-pair decoded-value MSE 必须在至少 6/8 states 小于 original，且 8-state mean 严格下降；否则记为 mechanism no-go。即使通过，也只能称“该 checkpoint 的 representation-sensitive PTQ diagnostic positive”，不能推出 task success、native low-bit deployment benefit 或新方法认证。

四臂能排除最直接的 FP function-change 解释，但不能证明 centering 优于所有其他 reparameterization；若要做方法论文，还需另行设计 matched perturbation/成本控制。当前建议只保留一次 bounded diagnostic 的 conditional-go，优先用于 falsification，而不是扩展成训练或闭环 campaign。

