# Prediction error vs teacher fidelity：prior gate

日期：2026-09-13。范围是先例、可识别性和最小 screen 设计；本审查没有读取真实数组、加载模型或连接 cluster。

## 判定

**`bounded_prior_go`（有条件）**：该诊断值得做一个固定的六样本 screen，但不能包装成新的 quantizer、泛化定理或真实控制收益。执行状态仍是 **`interface_pending`**：在 GPU 前必须先用 CPU allocation 生成并核验 raw current/action/future triples；若资产或索引语义不满足，标 `resource_blocked`，不换样本。

## 假设与可辨识对象

对同一记录 transition、同一 frozen FP encoder 和同一 action sequence，记未来 recorded observation 的 encoder feature 为 (z^*)，FP predictor 和 RTN-W4 predictor 输出为 (z_F,z_Q)。令 (e_F=z_F-z^*)、(Δ_Q=z_Q-z_F)，则

`MSE_Qtruth = MSE_FPtruth + MSE_QFP + 2 <e_F, Δ_Q>`

这是同坐标、同 target 下的逐样本恒等式。负 cross-term 可以显示 Q perturbation 抵消 FP prediction error；它不能证明 Q 学到了更真实的 dynamics。这里的 “truth” 必须写成 **recorded-future encoder feature**：它是观测和 frozen FP encoder 的相对参照，不是物理 state、reward 或 task success 的 ground truth。

## 先例与差异

[QuantWM (arXiv:2602.02110v1)](https://arxiv.org/html/2602.02110v1) 已在 DINO-WM 上比较 RTN、OMSE、AWQ、SmoothQuant、OmniQuant 的 weight/activation PTQ、rollout 和 planning objective/success alignment；其公开设置也讨论 predictor 与 encoder 的不对称敏感性。本次所读段落未发现将 recorded future feature 作为外部 target 并报告 `e_F` 与 `Δ_Q` signed decomposition 的对应分析；这不是全文或全部相关工作均不存在该分析的证明。因此本候选的 estimand 有区别，但只是 **diagnostic screen**，不是方法 novelty。

[Why Quantization Improves Generalization](https://arxiv.org/abs/2206.05916) 的 primary 结果是 stochastic binary-weight 网络的 NTK/generalization 分析，支持“量化在某些 held-out 误差上可能改善”这一可检验可能性；它不是 RTN、DINO-WM 或真实 future prediction 的直接先例。它不能替代本候选的 paired evidence。

本地 `dino-wm-wall` 旧协议的 Local-MSE 明确以 FP block output 为 teacher，RankCal/Score-error 也以 FP score 为参照；这些可检验 fidelity，却不能回答 FP 自身是否偏离记录未来。新诊断因此区别了“贴近 FP”与“贴近 recorded future”，但没有提出新的量化算法。

## 必须冻结的接口修订

1. 使用正确 source commit `0a9492fa12044b852ae9e001cc74604b79c8bb0c`。应从 `traj_dsets["valid"]` 的 unsliced `WallDataset/TrajSubset` 读取 source trajectory，并在 manifest 同时记录 valid-local index、底层 trajectory ID、raw frame ID、episode 边界和 source/data hashes。不得使用 `TrajSlicerDataset` 的随机排列 index 作为稳定 ID。
2. 固定六个 fresh validation IDs `124..129`；`84..95` 是 locked，不能读取或用来补缺。现有 `screen_runner.py` 的 `pilot_test` 只覆盖 fixed dataset offsets `18..41`，且 `_new_target` 从环境随机采样 initial/goal；它不能直接作为本诊断的 recorded-future target。若 124..129 的 namespace 未明确，必须先在 CPU manifest 中解析并 fail closed。
3. `num_hist=1、num_pred=1、frameskip=5` 下，H1 使用同一 trajectory 的 raw `t -> t+5` observation 和 `action[t:t+5]`；H5 使用 `t -> t+25` 和 `action[t:t+25]`，后者按五个五步 action blocks 输入 predictor。两者必须是同一记录轨迹、同一 current observation 和真实 actions，不能用 env replay、random goal、CEM candidate 或模型生成 target；不足未来帧或跨 episode 立即 `resource_blocked`。
4. FP 与 Q 必须共享已编码的 current input、actions、preprocessing、normalization 和 reduction；只量化 predictor，encoder/decoder/activation 保持 FP32。主 metric 预先固定为 visual DINO feature（patch/token 维度完整 mean squared error）；proprio 单独报告，不能事后把 visual、proprio、pixel decoder loss 混成一个数。使用 `visual_world_model.py` 的正式 one-step `predict/rollout` 路径，H5 为五次 sequential predictor transition，不能把 H5 误作一次 five-action 输入。
5. 每个样本保存 (z^*,z_F,z_Q,e_F,Δ_Q) 的 compact summary，并逐样本计算三项 MSE 与 dot term，再做六样本等权平均；另保存恒等式 residual、finite checks 和 coordinate/normalization identity。不得只保存 aggregate MSE，因为 aggregate cancellation 无法审计。RTN-W8 只作数值/scale sanity，不替代 W4 primary，也不得按 truth 结果选择 qparams。

## 最小 gate 与停止语义

所有六个样本须 finite，`MSE_FPtruth` 与 future-feature variance 高于预注册 epsilon；否则为 `inconclusive`，不能把相对改善写成零误差收益。主 gate 只看冻结的 H5：每样本 `1 - MSE_Qtruth/MSE_FPtruth >= 0.05`，至少 `4/6`，同时报告 H1、绝对误差、cross-term 符号和最差样本。若出现非 finite、target 泄漏、恒等式不闭合或两样本以上明显恶化，停止为 `no-go/inconclusive`，不靠删除样本修复。H1 只作 horizon diagnostic；通过也不推出 planning、success、return、physical dynamics 或 native low-bit deployment。

## 结论与检索边界

该问题实质上检验“FP teacher 是否是错误 proxy”，与旧 DINO-WM local fidelity / rank / CEM 失败轴不同；其独立性来自 recorded-future target 和 error-cancellation estimand，而非另一个 loss 名称。建议保留为 conditional minimal diagnostic；若 CPU manifest 无法证明 124..129 的真实 raw future triples，则工程上停止，结论写 `resource_blocked`，不能写 novelty no-go。

本次 targeted primary search（2026-09-13）仅覆盖两条 query：`2602.02110 QuantWM world model quantization task fidelity`、`2206.05916 quantization generalization`；未作全面 novelty 检索。源码核对范围为 `reproduction/dino-wm-wall/source/datasets/{wall_dset,traj_dset}.py`、`models/visual_world_model.py`、`plan.py`，以及本地 `smoke_runner.py`/`screen_runner.py`。
