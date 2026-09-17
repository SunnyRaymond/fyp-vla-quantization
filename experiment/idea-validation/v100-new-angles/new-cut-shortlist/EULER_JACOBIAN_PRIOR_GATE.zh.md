# SmolVLA Euler-step Jacobian：prior / identifiability gate

日期：2026-09-13。范围是固定 LeRobot v0.4.4、固定 observation/prefix KV、固定
FP K=10 trajectory 的一次局部诊断；不运行模型、不读取数据、不申请 cluster。

## 判定

结论为 **`conditional_go`（仅诊断）+ `generic_novelty_no_go`**。PTQ diffusion
已有 denoising error accumulation、timestep-aware quantization 和 denoiser
sensitivity 先例；本切面只有一个更窄、可反驳的模型问题：在同一个 normalized
32D action token 的 Euler map 上，expert W4 是否产生 `det(I+dt J_v)<0` 的
orientation reversal 或极小 `sigma_min`，而 FP map 保持 regular。不能把它写成
新的 quantizer、solver、稳定性保证或 deployment claim。

## source / 封闭性证据

官方 [`modeling_smolvla.py`](https://raw.githubusercontent.com/huggingface/lerobot/v0.4.4/src/lerobot/policies/smolvla/modeling_smolvla.py)
的 `embed_suffix` 对 `[B,50,32]` action chunk 设置 50 个 `att_mask=1`，
`make_att_2d_masks` 用 cumulative mask 形成 causal action-token attention
（约 L653–690、L93–119）。`sample_actions` 先缓存固定 image/language/state
prefix，逐步以 `dt=-1/10`、`x_t <- x_t+dt*v_t` 调 `denoise_step`
（约 L741–787）；`denoise_step` 的 suffix mask 同样只允许 first token 看
prefix 与自身（约 L797–819）。所以对固定 prefix、time 和后续坐标，

`J_v = ∂v[:,0,:] / ∂x[:,0,:]`、`M = I_32 - 0.1 J_v`

是有定义的 32×32 local map。这个结论只涉及**时间位置**的无 future-to-first
路径；同一 token 的 32 coordinates 仍会被 `action_in_proj`、expert 和
`action_out_proj` 混合。7D physical slice 不是封闭系统，不能把它的 7×7
principal block 称为完整 orientation map。`max_action_dim=32`、`num_steps=10`
也由 [`configuration_smolvla.py`](https://raw.githubusercontent.com/huggingface/lerobot/v0.4.4/src/lerobot/policies/smolvla/configuration_smolvla.py)
固定。

## 研究假设与最小识别

先用 FP K=10 rollout 在 step 5 前后固定 `t=.5` 的 `x_t`，再在**同一 x、同一
prefix KV、同一 t**分别运行 FP 与 expert-W4 field。用 input Jacobian（不是
有限差分近似）构造 `M_FP`、`M_Q`，保存 `sign(det M)`、normalized
`|det M|/sigma_max(M)^32`、`sigma_min/sigma_max`、`||v_Q-v_FP||`。FP regular
必须预先定义为 positive orientation 且 singular-value ratio 不低于固定
阈值；Q reversal 需同时满足 sign flip 与非数值噪声的 determinant margin，
Q near-singular 则需 ratio 低于冻结阈值。所有阈值、FP/Q arm、W4 expert
allowlist、t=.5、no-RTC、no-compile、eval 和 dtype 在读结果前冻结。

必须有 AD/固定 epsilon directional-FD 的实现自检、FP no-op、weight restore
和 prefix-cache identity。若只看到 velocity MSE 变大而没有 map spectral
变化，或只在 Q 自己的 diverged trajectory 上评估，则只能报告普通 field drift，
不能支持 reversal/near-singularity 假设。即使通过，也只说明一个 checkpoint
和一个 flow step 的 local geometry；不推断 10-step endpoint、action quality、
closed-loop success 或 physical safety。

## prior 与停止边界

PTQD（[NeurIPS 2023](https://arxiv.org/abs/2305.10657)）已把 quantization
noise 的 correlated/residual 部分、均值/方差偏移和多 denoising step 累积作为
核心对象；ADP-DM（[CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Towards_Accurate_Post-training_Quantization_for_Diffusion_Models_CVPR2024_paper.pdf)）
按 timestep 学习 rounding；Qua2SeDiMo（[AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32658)）
按 denoiser operation/block 量化敏感度做解释。这些直接覆盖“低比特改变迭代
field/敏感度”的宽主张，但没有在 SmolVLA action-token Euler map 上提出
det-sign 或 singularity diagnostic。既有 flow-step refinement 已覆盖 K=5/10/20
endpoint decomposition，flow-geometry 已覆盖 action drift/acceleration，
action-gradient geometry 已覆盖另一 world model 的 local direction mismatch；
本案若只换成另一个 scalar metric，应停止，不应包装为 novelty。

因此只有在预注册的 **large velocity-MSE、但 FP/Q map topology 可分离** 的问题
下才值得一次小 screen；若目标是通用“Jacobian-aware PTQ”方法、稳定性定理或
部署收益，直接 `novelty_no_go`、GPU=0。
