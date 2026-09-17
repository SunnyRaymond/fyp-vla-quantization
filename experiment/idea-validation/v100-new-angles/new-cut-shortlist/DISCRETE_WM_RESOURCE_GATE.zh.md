# Discrete-latent world model：numerical PTQ tokenizer-boundary gate

更新时间：2026-09-13。范围是 IRIS、DIAMOND、iVideoGPT 三个官方项目；本轮只做 source/asset gate，不下载、不加载模型、不连接 cluster。结论区分机制可识别性与当前资产可用性。

## 先行结论

当前没有候选可以冻结为本 campaign 的 GPU screen：IRIS 与 iVideoGPT 的机制值得保留为 conditional candidate，但本轮没有把官方 checkpoint、source revision 和 matching input 绑定进 workspace，因此状态是临时 `resource_blocked`，不是永久 no-go；DIAMOND 在本案要求的 discrete-tokenizer 定义下为 structural no-go。本轮不下载、不加载模型、不连接 cluster，未来是否 staging 资产须另按资源规则决定。

这不是把 VQ 的 representational quantization 当作 numerical PTQ。合格干预必须对同一个 released trained model 的 tokenizer encoder（或明确的 tokenizer projection）做 W4/W8 PTQ，保留 codebook、下游 world-model weights 和输入完全不变；FP/Q 的 token index、连续 pre-quantization latent 与下游 one-step prediction 必须同时记录。科学问题是：在匹配连续 latent MSE 后，boundary crossing 是否仍能预测下游误差，而不是把 flip rate 当作结论。

## 候选 A：IRIS（机制 conditional，当前 `resource_blocked`）

### 研究假设

IRIS 的离散 autoencoder 把图像变成 image tokens，再由 autoregressive Transformer 建模；因此 encoder 的小数值扰动可能把 pre-quantization latent 推过最近 codebook Voronoi 边界。token flip rate 可能比 latent MSE 更早显示下游 dynamics/reward prediction 的实际变化。这是 tokenizer-boundary 机制，独立于已有 DINO-WM activation scale、broadcast coupling、decoder-tail 与 temporal persistence 切面。

### source 与资产证据

- 官方 repo：[eloialonso/iris](https://github.com/eloialonso/iris)。README 明确 world model 由 discrete autoencoder 与 autoregressive Transformer 组成；官方安装要求 PyTorch 1.11.0，并警告依赖会下载 Atari ROM。
- 官方 tokenizer source：[tokenizer.py](https://github.com/eloialonso/iris/blob/main/src/models/tokenizer/tokenizer.py)。`encode` 先过 `encoder`/`pre_quant_conv`，以到 `embedding.weight` 的距离 `argmin` 生成 `tokens`，再查 codebook；这给出可绑定的 index locus。
- 官方 model loading：[agent.py](https://github.com/eloialonso/iris/blob/main/src/agent.py)。checkpoint 按 `tokenizer`、`world_model`、`actor_critic` 三个 state subset strict load，不应将不同训练 run 拼成 FP/Q 对。
- 官方 HF 资产：[pretrained_models](https://huggingface.co/eloialonso/iris/tree/main/pretrained_models)。页面显示约 3.29 GB 总仓库、单个 Atari `.pt` 约 126–127 MB（HF folder revision `c0539e9`）；体积本身满足 `<2 GB/model`，但该资产不在当前 workspace，且没有已登记的 matching episode/demo 与 pinned source/checkpoint pair。

### 当前 gate

`resource_blocked`（当前 gate）：官方 checkpoint 可公开取得，但本地没有可核验的 `.pt`、source commit、合法 Atari frame/episode 三者绑定。ROM 下载许可与运行时依赖也不能假定已满足；这只阻止当前 screen，不否定未来在合规 staging 后的可行性。

### 资产补齐后的最小 falsification screen（仅草案）

固定一个官方 game checkpoint 与至少 32 个已登记官方 frame/episode，使用同一 FP checkpoint、同一 codebook、同一 FP world-model state。arms：`FP`、仅 tokenizer encoder W4 RTN、仅 tokenizer encoder W4 SR（固定 seed）；RTN/SR 必须使用相同 weight coverage 与 scale recipe。

每个输入保存 `z_fp/z_rtn/z_sr`、归一化 latent MSE、index flip mask/rate、codebook margin/jump 与 FP world-model one-step latent/reward logits。主识别是 nearest-neighbor 配对：RTN 与 SR 的 latent MSE 比值须在 1.05 内，但 flip rate 至少相差一个 token；比较两者 downstream error。预注册至少 8 个有效配对、4 个方向一致才可称 mechanism signal；无配对即 `insufficient_mechanism_no_go`。再为每个 Q 序列构造同 flip 数量、近似 codebook jump 的替代 token 序列，送入同一 FP world model；若真实 Q 与该 distance-matched replacement 无差异，结论退回 generic token corruption；只有 Q 选择在 matched MSE/jump 下仍产生更大 error，才支持 boundary location 的非平凡作用。

`Q-encoder + FP-token-id replay` 仅用于 no-op/identity 工程检查：相同 FP Transformer 与相同 token IDs 输出相同是结构恒等式，不计作科学 gate。decoder-only W4 仍可作为另一个工程负对照，但其重构误差也不能冒充 boundary evidence。

停止规则：checkpoint/source/input 任一 identity 缺失即当前 `resource_blocked`；FP no-op、strict restore、有限值或 codebook shape 失败即 engineering stop；有效配对不足、所有输入 zero token flips，或 matched MSE/jump 下真实 Q 不优于替代 token，即 `insufficient_mechanism_no_go`。通过时只报告 matched one-step effect，不宣称 full planning/control 收益。单 V100 目标 `<10 min`，不跑完整 Atari episode。

## 候选 B：iVideoGPT（机制 conditional，当前 `resource_blocked`）

### 研究假设与实际接口

iVideoGPT 的官方接口确实把 `CompressiveVQModel` tokenizer 产生的 VQ/dynamics tokens 接到 causal Transformer；其 `tokenize → model.generate → detokenize` 路径是可观察的离散边界。action-free video prediction 已足够提供 downstream world-model 反事实，不强制要求 action-conditioned checkpoint；应研究 numerical PTQ 是否在 matched latent MSE 下改变 token ID 并放大到固定 visual prediction，而不是把 tokenization 本身当量化。

官方证据：[repo/README](https://github.com/thuml/iVideoGPT/blob/main/README.md)、[官方 predictor interface](https://github.com/thuml/iVideoGPT/blob/main/vp/ivideogpt_interface.py)、[official model collection](https://huggingface.co/collections/thuml/ivideogpt-674c59cae32231024d82d6c5)。README 列出 64×64 tokenizer 约 114M、Transformer 约 138M（medium 为 436M），因此 base 组合在参数量上有机会低于 2 GB；但 OXE released models 明确是 action-free，action-conditioned models 只列在 BAIR/RoboNet/RoboSuite/RoboDesk downstream 表中。

### 当前 gate 与停止条件

`resource_blocked`（当前 gate）：当前 workspace 没有 released tokenizer/Transformer、source revision 与 official `.npz` sample 的完整 identity。README 已给出 `inference/samples/fractal_sample.npz` 的 OXE action-free inference 路径，因此 action-free 不是阻塞项；真正缺口是当前未绑定这些文件。若未来只有 tokenizer/VQ flip 数值而无法运行固定 FP causal Transformer 的 visual prediction，才判 `insufficient_mechanism_no_go`。

若将来资产齐备，最小 screen 与 IRIS 同构：只量化 `CompressiveVQModel` encoder/projection，codebook 与 Transformer FP；固定两 context frames 与 generation RNG（runtime 若支持则用 greedy，否则逐 arm 恢复同一 sampling state）。比较 RTN/SR 的 matched latent MSE、token IDs、codebook jump 与 one-segment decoded prediction；使用 distance-matched replacement 作为 negative control。FP-token replay 只作 identity check；若只剩 token flip/Voronoi 指标而无固定 FP Transformer 的 downstream output，直接 `insufficient_mechanism_no_go`。

## 候选 C：DIAMOND（`structural_no_go`）

官方 repo：[eloialonso/diamond](https://github.com/eloialonso/diamond)，论文：[Diffusion for World Modeling: Visual Details Matter in Atari](https://arxiv.org/abs/2405.12399)。官方 README 将目标描述为 entirely in a diffusion world model，并以 Hugging Face pretrained world model + Atari ROM 下载作为 play path；本次核到的官方入口没有 VQ codebook、discrete tokenizer 或可读的 token-index interface。由此不能把 diffusion latent/noise state 重新命名为本案 discrete-latent tokenizer boundary。

因此在当前候选定义下为 `structural_no_go`，不是 checkpoint 体积判定，也不是对 DIAMOND 全部内部表示作绝对断言。即使后来发现可用 continuous VAE，仍需另立问题和反事实；不能直接纳入本 screen。官方 play path 还依赖 pretrained assets/ROM，当前无合法已登记输入，进一步不满足本轮资源 gate。

## prior 与边界

IRIS 的官方设计本身证明“离散 autoencoder → Transformer world model”路径，但不证明 numerical PTQ 的 index-flip 因果；iVideoGPT 的 official interface 证明类似 VQ-token path，但不证明当前 workspace 已绑定可复现资产。DIAMOND 的 diffusion prior 不覆盖本案 discrete-index mechanism。以上是定向 source gate，不是穷尽 novelty 检索。

与 INDEX 中已有切面的差异在于：干预 tokenizer encoder 的数值权重，并用 matched latent-MSE、matched codebook-jump 与固定 FP downstream model 的 visual/latent prediction 区分 boundary location；token-ID replay 只作 identity check。不改变 scale partition、候选排序、历史残差、decoder-tail、rounding schedule、goal/current coordinates 或 policy。若未来无法同时取得 FP/Q 同 trained checkpoint 与合法 matching input，本案保持 `resource_blocked`；若取得后缺少 matched downstream contrast，则为 `insufficient_mechanism_no_go`，不消耗 GPU allocation。
