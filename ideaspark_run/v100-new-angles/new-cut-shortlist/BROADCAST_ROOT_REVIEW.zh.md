# BROADCAST_ROOT_REVIEW

审查日期：2026-09-13。范围只包含本地 pinned source、已有 Antithetic/Persistence 设计和最多四个 targeted primary source；不连接 cluster、不加载模型、不运行数值、不新增输入。本文修订上一版“非 native PTQ 即不可识别”的过强判断。

## 结论

该候选与 Antithetic Rounding、Rounding Persistence **不是同一干预轴**，在 C02 controlled diagnostic 下具有可识别的窄 screen 条件，建议标为 `conditional_go / novelty_unverified`，而不是预先 `identifiability_no_go`。它不应被描述为 native activation-quantization deployment 或新的 quantizer；若只想保留已有实测 STOP，则最强的反例仍可使它在首轮 preflight 前停止。

## 候选与可识别对象

固定 DINO-WM Wall epoch65、visual encoder、predictor、输入、action 和 one-step FP reference。真实 source commit 是 `0a9492fa12044b852ae9e001cc74604b79c8bb0c`，官方路径为 [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm)；固定文件为 [visual_world_model.py at the pinned commit](https://raw.githubusercontent.com/gaoyuezhou/dino_wm/0a9492fa12044b852ae9e001cc74604b79c8bb0c/models/visual_world_model.py)。`concat_dim=1` 的 `encode` 先得到 proprio/action embedding，再用 `repeat` 广播到 visual patch 轴并 `cat`；`predict` 将 `(time, patch)` 展平。当前 Wall manifest 的实际结构仍须在运行时重验 `num_hist=1`、`P=196`、`384+10+10=404`，不能用 source 默认值代替 checkpoint receipt。

只干预 tiled 后的 20 个 state/action channels，使用固定 per-coordinate A4 scale 和 stochastic rounding（SR）。比较同一 scale、同一 SR marginal 的空间耦合：

1. **SR-shared**：同一 channel 的 `U` 复制到 196 patches；
2. **SR-independent**：每个 channel×patch 独立 `U`；
3. **SR-balanced/permutation**：每个 channel 使用同一组 `U` multiset，在 patch 轴固定 permutation，使每个 patch 位置获得平衡边际；
4. **RTN-before / RTN-after**：相同 deterministic round-to-nearest grid，分别在 repeat 前和 repeat 后执行。

`SR-before` 可作为 shared 的 placement 等价实现，但必须保存实际 quantizer placement；不要把“before/after”与“shared/independent”混成一个 arm。所有 arms 的 FP embedding、per-channel scale、seed、predictor 和 visual features 相同；只报告一步 predictor 的 raw visual latent prediction error（例如 patch-wise error covariance/energy），不跑 planner、不报告 success。

## 为什么区别于已停止的两个 rounding 方向

* **Antithetic Rounding Pairs** 在 predictor **weights** 上构造两份 member/candidate rounding joint law，再平均 scores；其机制结果是 preliminary_go，但 two-W4 实际成本相对 one-W8 不划算而停止。它改变的是参数成员之间的 coupling、并使用下游 shortlist；本候选只有一个 predictor、没有 member aggregation，改变的是同一 tiled activation 在 patch 轴的 spatial joint law。
* **Rounding Persistence** 在 SmolVLA expert **weights** 的三套 snapshots 间改变 10 个 denoising calls 的时间 assignment；其窄结果是跨-call temporal coupling。它不改变一个时刻的 token/patch activation field。本候选只做 DINO-WM one-step、同一个 input、同一个 predictor，研究 spatial broadcast coupling。

因此，“误差 coupling 可能影响 nonlinear model”是共同的高层动机，但 axis、location、模型调用和 primary observable 均不同。不能把已有 Antithetic 的 practical no-go 或 Persistence 的 temporal result 直接外推为本候选 no-go。

## Identifiability 依据与最强反例

可识别性来自四个 matched 条件：

* predictor/visual weights 不变，避免 encoder-vs-predictor module confound；
* 每个 coordinate 使用同一 scale 和同一 SR marginal，避免量化误差幅度或 bias 改变；
* before-repeat 的 shared field 与 after-repeat 的 shared field 用相同 quantizer grid，RTN-before/after 提供 placement-only negative control；
* independent 与 balanced 只重排 patch 轴的 `U`，保持每 channel 的 draw multiset，比较的是 error spatial covariance，而非 aggregate MSE 的新命名。

最强反例是：DINO predictor 对 patch tokens 只使用 permutation-equivariant 或近似平均的处理，使 tiled channel 的 covariance 在 one-step output 中被完全消除；此时 shared、independent、balanced 的差异应接近 FP32 reduction noise，候选直接 `mechanism_no_go`。另一个反例是 per-patch SR 实际改变了训练时不存在的 activation distribution：这会使 screen 成为 artificial sensitivity diagnostic，而不是 native PTQ 方案；必须在报告中保留该边界，不能用它支持部署收益。

若 `P` 不是 196、20 channels 不是 state/action 的实际 slice，或 tiled tensor 在 predictor 前被另一路 normalization/reshaping 改写，立即 `structural_no_go`。若 scale、U multiset、patch permutation、RTN placement 不能写入可复核 receipt，立即 `implementation_inconclusive`，不得解释 raw error。

## Closest primary prior 与 novelty 边界

[QuantWM](https://arxiv.org/html/2602.02110) 已覆盖 DINO-WM 的 encoder/predictor、weight-only 与 weight-activation PTQ，以及 activation granularity；因此本候选不能声称“首次发现 DINO-WM 某模块更敏感”。[DINO-WM paper](https://arxiv.org/html/2411.04983) 与 pinned source 支持 action-conditioned visual prediction 和 patch broadcast 的模型语义，但没有为本候选证明 stochastic spatial coupling 的 downstream 作用。

[PTQD](https://arxiv.org/abs/2305.10657) 使“量化误差的相关性可能跨多步传播”成为已知背景；本候选不做 diffusion denoising、时间 schedule、correction 或 mixed precision。Antithetic 的 [冻结协议](../antithetic-rounding/IDEA_AND_PROTOCOL.zh.md) 是最接近的 campaign prior，但其 coupling 发生在 weight members；Persistence 的 [冻结协议](../rounding-persistence/PROTOCOL.zh.md) 是时间 coupling。Novelty 仍只可写 `unverified`，需要后续更专门的 activation spatial-correlation 检索才能升级。

## 单次最小 screen 与停止门

若 root 冻结，最小 screen 为 6 个已核验 fresh Wall states、同一 FP visual prediction、FP + RTN-before + RTN-after + SR-shared + SR-independent + SR-balanced；每个 state 使用固定 action/observation，保存 20-channel tiled input、per-channel scale、每 arm 的 U/hash、patch-wise prediction error 和 one-step output。单 V100 预计远低于 10 min，但这只是资源估计，不能替代实际 allocation receipt。

主 gate 不用 planner score 或新 scalar ranking：

* 先过 source/checkpoint/shape、finite、FP no-op、RTN placement receipt、同 scale 和 exact SR multiset gates；
* 在每 state 上比较 shared/independent/balanced 的 patch-error covariance 与 output error，同时要求 SR-before 与 SR-shared 的 expected shared field 一致到预注册浮点容差；
* 若 shared 相对 independent/balanced 没有预注册的 spatial-covariance 方向，或只改善 aggregate error 而没有 covariance binding，判 `mechanism_no_go`；不得事后挑 patch、channel 或 state；
* 若任一 arm 改变 visual/predictor weights、scale、marginal、输入 shape 或 predictor call，判 `implementation_inconclusive`；不把工程失败写成科学 no-go。

不预注册普适百分比收益门槛：本 screen 的目标是先否证“空间 joint law 可穿过 one-step predictor”，不是证明任务性能。若通过，也只能报告“该 checkpoint、该 20-channel slice、该 one-step FP predictor 上的 spatial-broadcast diagnostic preliminary signal”；不扩展到 rollout、CEM、native A4 kernel、速度、显存或 robotics success。

## 最终判定

`conditional_go` 仅表示设计层面的可识别性已达到一次窄 screen 的最低条件；`novelty_unverified` 保留。最强反例是 predictor 的 patch mixing 消除 covariance，或 after-repeat independent/balanced 被认定为训练分布外的人工干预；任一反例在 preflight 得到证据，立即 `GPU=0`。当前没有运行结果，不能把该候选写成已成立机制。
