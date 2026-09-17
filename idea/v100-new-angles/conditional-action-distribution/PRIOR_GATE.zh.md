# Conditional action distribution：bounded prior gate

日期：2026-09-13。范围是核对 SmolVLA v0.4.4 的随机采样语义、相关 primary prior 与一个最小 distribution-shift 诊断；没有实现、下载、模型推理或集群操作。

## 判定

结论：**conditional narrow-go / statistical-risk gated**。这是区别于 flow-geometry-drift 的可识别切面：same-noise action MSE 测的是一个特定 coupling 下的 path/map fidelity，而 independent-noise distance 测的是条件边际分布 P(action | observation)。二者不能互相替代；仅用 paired MSE 判定 Q 失败，确实可能把“映射改变但分布接近”误读为分布退化。

它不是新 quantizer 或新 distribution metric。当前有限先例没有直接覆盖 SmolVLA 的 7-D action chunk 条件分布 screen，因此不是 conceptual/no-go；但高维小样本下不能声称 full-distribution preservation，统计信息不足时必须给 inconclusive。

## SmolVLA stochastic interface

官方 LeRobot v0.4.4 SmolVLA 的 inference path 在 noise 未显式传入时调用 sample_noise，sample_actions 从 standard-normal full action noise 开始做 flow-matching Euler denoising，最后再截取 physical action dimensions。其 API 也允许显式传入 noise，这正好支持一组 cloned paired seeds 与另一组 disjoint independent seeds。官方 source 见 [modeling_smolvla.py v0.4.4](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/smolvla/modeling_smolvla.py)；本地 flow screen 的 `_noise_tensors` 亦按独立 seed 建立 float32 noise（L709–714），并把同一 noise 传入各 arm。

因此必须把“native 每次 sample 新 noise”写成 `noise=None` 的默认行为，不能在 Q/FP 比较中复用隐式缓存。paired 组应显式向 FP/Q 传同一个 cloned noise；independent 组应使用不重叠、预注册的 seed namespace，并保存 noise hash/shape。RTC、cache、compile 和额外随机源要关闭或每次 reset，否则 distance 混入非 noise 随机性。

## 最小可识别设计

对每个固定 observation/condition 单独计算，不把不同 task/episode 的 action 混成一个总体分布。至少保留两种数据角色：

- paired set：FP(z_i)、Q(z_i) 使用完全相同的 z_i，报告 same-noise path MSE 或 normalized endpoint MSE；它回答 Q 是否沿用 FP 的 noise-to-action map。
- independent sets：FP-A、FP-B、Q-A、Q-B 使用四组互不重叠的 noise。FP-A 与 FP-B 估计有限样本 null，Q-A 与 Q-B 检查 Q 自身 spread/mode collapse，Q-A 与 FP-B（或等尺寸交叉组合）估计 marginal distribution distance。所有比较使用相同样本数、相同 projection/kernel 和相同 action slice。

固定-projection sliced Wasserstein 可作为一个 bounded primary metric；若担心 optimal-transport sorting 对小样本产生过度乐观的 coupling，则不要看 raw distance 单值，必须同时报告 FP-A/FP-B null。MMD 可作为非-OT sensitivity check，但 bandwidth、projection 和 kernel 必须在看 Q 输出前冻结，不能用 Q 样本调参。

推荐按 observation 得到 `W_QF`、`W_FF`、`W_QQ` 及 paired path error `M_pair`，再跨固定 observations 报 median/范围。分布接近只能在 `W_QF` 不超过同规模 `W_FF` null 加预冻结 margin、且 `W_QQ` 没有异常塌缩时报告；若 `M_pair` 大而 `W_QF` 接近 null，可标记“map drift with marginal-preserving screen”，这是本候选的核心区分。若 `W_QQ` 明显低于 FP null，即使 `W_QF` 小也只能标记 mode-collapse/inconclusive，不能称保真。

## 严格但最小 gate

1. 预先冻结每个 condition 的 sample count、paired/independent seed lists、projection matrix 或 kernel bandwidth；验证各组 seed 不重叠、noise shape/dtype 一致、FP/Q condition 与 postprocessor identity 一致。

2. 先检查 paired path metric 非退化，再以 FP-A/FP-B 的同样本 null 校准 `W_QF`。distribution-preserving 候选需满足 `W_QF <= W_FF + margin`；distribution-shift 候选需满足 `W_QF > W_FF + margin`。margin 需在看 Q 结果前冻结，不能把 raw OT distance 当作绝对阈值。

3. 同时检查 `W_QQ` 与 FP null 的相对 spread，至少标出 Q-Q 明显塌缩、极端扩散或 projection coverage 不足。Q-Q 低而 Q-F 也低不等于保真；它可能是 mode collapse。高维 projections 或样本数不足以稳定排序时，结果为 inconclusive，不降维或调 kernel 追结果。

4. 每个 condition 先完成 distance，再以预注册规则跨 observations 聚合；不得只挑选 paired MSE 较大或较小的 episodes。positive/negative 只表示 bounded screen gate，不做显著性、功效、普适性或真实 task success 声称。

为节省资源，可将 independent groups 以 batch 方式一次推理；最小可接受样本数必须由 finite-sample pilot/metric 的预注册合同给出。若只有极少数 noise（例如每组不足以形成稳定的 empirical quantiles），应直接 `statistical_inconclusive`，不能用 bootstrap 把信息不足包装成结论。

## prior overlap 与停止条件

[Q-Diffusion](https://arxiv.org/abs/2302.04304) 已指出 diffusion noise-estimation outputs 随 timestep 改变并以 FID 等生成分布质量评估 PTQ；[PTQD](https://arxiv.org/abs/2305.10657) 也以生成质量保真评估 PTQ；最新 [Q-Drift](https://arxiv.org/abs/2603.18095) 更直接把 quantization 当作逐步 stochastic perturbation，并以 marginal-distribution-preserving correction 为目标。它们使“PTQ 可能保持局部配对误差却改变边际分布”的动机有直接邻域先例，但没有直接实现 SmolVLA conditional action distribution 与 same-noise coupling 的双诊断。

因此 novelty 只可放在 VLA action 条件分布的窄测量问题，不能声称发明 SWD/MMD、noise coupling 或 distribution-preserving quantization。若文献/代码进一步发现已有方法在同一 VLA/PTQ setting 已做该 paired-versus-independent gate，则立即 novelty_no_go；在当前三组 targeted primary search 范围内尚未发现这种直接覆盖。

以下任一项都应停止而不是扩展实验：无法证明 `noise=None` 真会逐次采样新高斯、paired 与 independent seed 重叠、condition 未固定、Q/Q 或 FP/FP null 缺失、Q mode collapse 无法区分、样本/projection 数量没有预先最低合同，或把 distance 结果误写成 action success。该候选可作为一次 bounded distribution diagnostic；不应替换已有 same-noise flow/padding screen，也不需要训练或 full validation。
