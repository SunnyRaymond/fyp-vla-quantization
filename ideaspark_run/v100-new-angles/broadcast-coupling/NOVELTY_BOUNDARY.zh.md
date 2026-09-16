# Broadcast coupling：novelty boundary

审查日期：2026-09-13。范围是冻结 `RESULT.zh.md`、`ROOT_GATE.zh.md` 与 bounded primary-source review；没有运行模型、读取 raw 数组、连接集群或追加实验。

## 被审查的主张

本案固定 DINO-WM Wall checkpoint 的一个 20D state/action activation tail。该向量被复制到 196 个 visual patches；`Shared` 在复制前做一次 A4 stochastic rounding，`Spatial` 在复制后把同一组三个 SR draw 以固定 balanced offset 重排。两臂保持 scale、每个 patch 的三值 multiset、总 input MSE、weights 与 predictor 不变，只改变 patch 轴的 joint assignment，观察 one-step visual MSE。

`RESULT.zh.md` 报告 5/6 state 达到预注册 10% gain，因此支持一个窄的 application diagnostic。它不支持新 quantizer、一般 independent SR 优势、压缩/latency、native low-bit kernel、multi-step 或 task success。

## Closest primary prior

| Primary work | 直接重叠 | 不能直接覆盖的部分 |
|---|---|---|
| [Correlated Quantization for Distributed Mean Estimation and Optimization](https://proceedings.mlr.press/v162/suresh22a/suresh22a.pdf), ICML 2022 | 用 permutation 与分层 uniform 构造 correlated quantization；每个成员仍保持 uniform marginal，并以均值 MSE 为目标。 | 对象是不同 client 的 scalar/vector 与 server mean，observable 是估计误差/optimization convergence；没有 repeated token、patch-field 或 downstream nonlinear visual predictor。 |
| [QSGD: Communication-Efficient SGD](https://proceedings.neurips.cc/paper/6768-communication-efficient-stochastic-gradient-descent-with-applications-to-neural-networks.pdf), NeurIPS 2017 | 建立 stochastic quantization 用于被 broadcast 的 neural updates。 | 量化的是 gradient communication，随机性服务于 bandwidth/variance；没有相同 activation 在 token/patch 轴的 shared-versus-reassigned coupling。 |
| [Efficient Adaptive Activation Rounding for PTQ](https://arxiv.org/abs/2208.11945), 2022 | 直接研究 activation rounding 对 output/dot-product error 的影响；其 coarse-grained 实现让一个 activation 的 rounding 在多个 output-channel weights 间共享。 | 其 axis 是 activation-to-weight/channel association，且学习 rounding border；本案是 repeated activation 的 patch-axis joint law，保持 SR marginal 与 input MSE 后测 visual field。 |
| [Ex Uno Pluria](https://arxiv.org/abs/2411.14860), NeurIPS 2024 | 以 Bernoulli stochastic rounding 从一个模型构造 low-precision ensemble，说明 rounding diversity 可用于预测。 | 它量化 weights 并组合 member predictions；本案没有 ensemble/member aggregation，只对一个 predictor 的 tiled activation 改空间 assignment。 |
| [SPC-NeRF: Spatial Predictive Compression for Voxel-Based Radiance Field](https://dl.acm.org/doi/10.1145/3795528), ACM TOMM | 利用空间相关性做 predictive/residual compression，说明 spatial redundancy 可影响表示压缩。 | 它的 observable 是 voxel storage/rate-distortion，属于 predictive coding/DPCM-like compression；没有本案 SR grid、patch forward 或 nonlinear output MSE。 |

因此，三-draw cyclic/balanced assignment 不能宣称为原创 rounding construction：它是 correlated/stratified stochastic rounding 与 permutation coupling 的有限样本 specialization。当前检索没有找到同时覆盖“broadcasted activation + patch-axis assignment + matched marginal/input MSE + downstream visual error”的 primary work，但这只留下 application-level difference，不认证全局 novelty。

## Application 与 communication/DPCM 的边界

Communication prior 让多个发送者的 quantization errors 在 server mean 中抵消；DPCM-like prior 先预测邻近值、再编码 residual 以节省存储或传输。本案不传输、不求均值、不编码 residual，也不减少 activation bytes；`Spatial` 仍使用同一组三个量化值，只把它们放到不同 patch。故这些工作是 mechanism analogies，不能被写成相同 application 的直接先例，也不能用它们推出本案 visual gain；反过来也不能用本案 gain 推出 communication 或 compression benefit。

## 可保留的研究贡献与判定

* **Generic method novelty：`novelty_no_go`。** SR、shared randomness、balanced/stratified permutation 与 rounding diversity 均已有 primary prior；本案不应命名为新 quantizer 或新 decorrelation algorithm。
* **Narrow application claim：`narrow_application_unverified`。** 当前实测可以作为一个受控 case study：在该 DINO-WM checkpoint 的 tiled activation 上，保持每臂 marginal 与 input MSE 后，patch-axis joint assignment 能改变 one-step FP visual fidelity。它的可辨识 delta 是 application/diagnostic，而非 quantizer 原理。
* `ROOT_GATE` 的 preliminary go 与上述边界一致；它不升级为 deployment、planner 或 success 证据。该 campaign 已按协议 STOP，不追加 GPU/CPU 来“证明 novelty”。

本次 bounded 检索的核心词为：`correlated quantization stochastic rounding`、`spatially correlated neural activation quantization`、`broadcast repeated token quantization`、`activation rounding output error`、`predictive coding spatial quantization DPCM`、`low precision stochastic rounding ensemble`。检索范围有限，未声称 exhaustive literature clearance。

Root metadata复核：PMLR162/suresh22a为ICML2022，原AISTATS标签已纠正；AQuant title/arXiv与Ex Uno Pluria NeurIPS2024由primary metadata确认。具体构造的application novelty仍未认证。
