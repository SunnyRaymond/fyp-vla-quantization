# 024. RLA-WM：Learning Visual Feature-Based World Models via Residual Latent Action

> **完整标题：** *Learning Visual Feature-Based World Models via Residual Latent Action*
>
> **本地论文：** [paper-arxiv-v1.pdf](paper-arxiv-v1.pdf)
>
> **Official resources：** [arXiv:2605.07079v1](https://arxiv.org/abs/2605.07079v1) · [HTML](https://arxiv.org/html/2605.07079v1) · [project page](https://mlzxy.github.io/rla-wm/) · [official code](https://github.com/mlzxy/rla-wm)
>
> **建议先修：** [#002 DINO-WM](../002-dino-wm/README.md)、DINO tokens、latent action、Flow Matching、Behavior Cloning、PPO。
>
> **阅读状态：** `verified-full-text`；已核对 arXiv v1 正文、appendix、本地 PDF 与公开项目入口。未安装依赖、下载 datasets/checkpoints、执行 training、inference、benchmark 或 code audit。

## 1. Paper identity

| Field | Record |
|---|---|
| Authors | Xinyu Zhang, Zhengtong Xu, Yutian Tao, Yeping Wang, Yu She, Abdeslam Boularias |
| Affiliations | Rutgers University, Purdue University, University of Wisconsin-Madison |
| arXiv | `2605.07079v1`，submitted 2026-05-08 |
| Local artifact | arXiv v1，25 pages，20,696,967 bytes，unencrypted |
| Venue / status | arXiv preprint；本次未见 peer-reviewed venue 声明 |
| Public code | `mlzxy/rla-wm`；本次记录的 `main` HEAD 为 `6f19048758699bf9a152eaed5ac6dbf1caa07c18`（2026-09-21），仅确认公开入口与文档结构，未做静态或运行时审计 |

这篇 paper 同时包含三条不同的 contribution：

1. **representation：** 从 DINO token residual 学 `Residual Latent Action (RLA)`；
2. **world model：** 在 compact RLA space 内做 Flow Matching，再 decode future DINO tokens；
3. **policy applications：** 用 RLA auxiliary target 学 actionless videos，以及在 RLA-WM 内做 visual RL。

阅读时应分别追踪三条证据链，不能用其中一条的结果替另一条背书。

## 2. 一句话抓手

不要直接 regression 或 generation 一个约 `32 x 32 x 1024 ≈ 1M` 维的 future DINO state；先把变化量 `s_{t+h} - s_t` 压成 compact latent `z`，只在这个 residual latent space 中用 Flow Matching 建模多模态 dynamics，再结合 current state `s_t` 单次 decode 出 `s_{t+h}`。

## 3. Problem：feature-based world model 的效率优势为什么会失效

DINO-WM 一类方法直接回归 future visual feature：

$$
\hat{s}_{t+h}=f_{dyn}(s_t,a_{t:t+h}).
$$

它很高效，但在 complex 3D interaction 中容易出现 regression-to-the-mean、blur 或 collapse。直接把 diffusion / Flow Matching 搬到 DINO feature space 又很昂贵：对 `512 x 512` 图像，DINOv3-L 产生约一百万维 tokens，而常见 pixel-aligned VAE latent 约为 16k 维。

作者的 hypothesis 是：absolute visual state 很高维，但 physically valid transition 所在的 manifold 更低维；因此应生成 transition representation，而不是生成完整 future feature tensor。

## 4. Method

### 4.1 RLA autoencoder：压缩 DINO residual

给定 frame pair 的 DINO patch tokens `(s_t, s_{t+h})`，先构造 residual：

$$
r_{t,h}=s_{t+h}-s_t.
$$

Encoder `f_enc` 用 learnable queries 从 residual 中提取 compact latent action：

$$
z=f_{enc}(r_{t,h}).
$$

Decoder 接收 current state 与 latent action，重建 future state：

$$
\hat{s}_{t+h}=f_{dec}(s_t,z).
$$

主设置中 `z` 为 `32 x 64 = 2048` 维；encoder 与 decoder 各有 12 个 self-attention layers、16 heads、channel size 1024。训练目标是 future feature 的 L1 + MSE，各自权重 1.0。这里的关键不是“residual 自身低维”，而是作者用 autoencoder 实证一个 compact bottleneck 仍可保留足够的 transition information。

作者把 RLA 的经验性质概括为：

- **predictive sufficiency：** `s_t + z` 的 decoder 可单次 feedforward 重建 `s_{t+h}`；
- **generalizability：** task-agnostic play data 上学到的 RLA 能迁移到未见 interaction；
- **temporal topology：** Gaussian noise 与 normalized RLA 间插值后，decoded states 近似 intermediate temporal states。

后两点主要由 qualitative examples 支持；不要自动提升为严格的 causal disentanglement 或 cross-domain theorem。

### 4.2 RLA-WM：在 compact transition space 内做 Flow Matching

Condition network 将 current DINO tokens、padded action chunk 与 learnable queries 组合成 condition。Flow network 从 Gaussian noise `epsilon` 出发，在 RLA space 预测 velocity：

$$
z_\tau=\tau z+(1-\tau)\epsilon,\qquad v^*=z-\epsilon.
$$

训练只监督 velocity MSE；inference 用 30 个 Euler ODE steps 得到 `z_1`，然后通过 frozen / pretrained RLA decoder 与 `s_t` 得到 `\hat{s}_{t+h}`。

结构要点：condition network 与 flow network 都是 8-layer、16-head、channel 1024；flow network 只处理 64 tokens（32 condition queries + 32 noisy RLA tokens），而不是在完整约 1M 维 DINO tensor 上迭代生成。

### 4.3 Minimalist WAM：RLA 是 training-time auxiliary target

标准 ResNet-18 BC policy 增加一个 linear RLA head：

- 有 action label 的少量 videos：训练 action head 与 RLA head；
- actionless videos：mask action loss，用 default proprioception token，只训练 shared backbone + RLA head；
- inference：丢弃 RLA head，仅保留 action head。

因此论文所说的 **no inference cost** 是相对于同一 BC action policy 在部署时没有额外生成 backbone；它不等于没有 RLA pretraining、DINO feature extraction 或 training overhead。

### 4.4 WMRL：在 learned world model 内做 PPO

Policy 从 pretrained BC-ResNet 初始化，加入 LoRA adapters 与 residual Gaussian action head。一次 action chunk 被视为一步 transition：

$$
(s_t,a_{t:t+h},s_{t+h}).
$$

RLA-WM 生成 future DINO tokens，再由 pretrained UNet decode 为下一步 RGB observation。`Video Aligned Reward (VAR)` 是 predicted DINO tokens 与 time-aligned reference video tokens 的负 L1 distance（Poke Cube 使用 terminal goal frame）。PPO rollouts 完全在 learned RLA-WM 内完成。

“no handcrafted reward” 不应读成“无 reward specification”：VAR 仍依赖 time alignment 与 reference videos，本质上把 offline demonstration trajectory 定义成 reward target。

## 5. Main Results

### 5.1 Future feature / frame prediction

作者在 ManiSkill 与 IWS 上从 initial frame autoregressively rollout：ManiSkill 30 steps、action chunk 10、3 个 model steps；IWS 60 steps、chunk 15、4 个 model steps。

| Model | ManiSkill LPIPS ↓ | ManiSkill SSIM ↑ | ManiSkill DINO L1 ↓ | IWS LPIPS ↓ | IWS SSIM ↑ | IWS DINO L1 ↓ | FLOPs ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| DINO-WM | 0.156 | 0.865 | 0.078 | 0.223 | 0.825 | 0.058 | 2.1T |
| RAE | 0.324 | 0.717 | 0.143 | 0.550 | 0.625 | 0.159 | 14.3T |
| FM-WM | 0.127 | 0.890 | 0.063 | 0.360 | 0.741 | 0.119 | 14.3T |
| Vid2World | 0.199 | 0.705 | 0.084 | 0.388 | 0.710 | 0.139 | 1.1P |
| **RLA-WM** | **0.071** | **0.931** | **0.030** | **0.196** | **0.847** | **0.053** | 3.5T |

RLA-WM 在所有 aggregate quality metrics 上最好，FLOPs 远低于 Vid2World，但高于 direct-regression DINO-WM。论文没有给 matched-hardware wall-clock latency、peak memory、throughput 或 30-step Euler solver sensitivity，因此不能把 `3.5T vs 1.1P` 直接当成真实 deployment speedup。

此外，DINO-WM baseline 是作者用 DINOv3 和 action chunks 重实现的版本；它并非天然等同于本项目当前 LeWM / DINO-WM checkout 的 exact code path。

### 5.2 Actionless-video policy learning

只有 5% videos 带 actions/proprioception（PushT 为 15%），其余作为 actionless videos。每个 method 都替换同一 framework 中的 latent extractor；50 evaluation episodes（seeds 42-91），结果取最后 5 个 checkpoints 平均。

| Method | PushT | Roll | Pull | Pull Tool | Poke | Avg SR |
|---|---:|---:|---:|---:|---:|---:|
| BC-ResNet | 3.6 | 42.0 | 33.6 | 7.6 | 49.2 | 27.2 |
| AdaWorld | 9.2 | 38.4 | **48.4** | 10.8 | 61.6 | 33.7 |
| **RLA** | **15.2** | **43.8** | 43.6 | **12.0** | **63.6** | **35.6** |

RLA 的 average gain 是 `27.2 -> 35.6`，但并非每个 task 都胜过所有 baseline；Pull Cube 上 AdaWorld 更高。PushT 的 absolute SR 仍只有 15.2%，所以这更像低-label regime 下的 representation-learning evidence，而不是 high-performing PushT policy。

### 5.3 WMRL large-scale evaluation

论文用 15 个 RL training seeds，每个 trial 训练 2,400 steps；最终选择 best-performing models，并用 1,500 evaluation episodes 报告：

| Method | XArm Poke | UR10e Roll | UR10e PushT | Panda Pull | Panda Pull Tool | Avg |
|---|---:|---:|---:|---:|---:|---:|
| BC-ResNet | 89.9 | 65.5 | 17.2 | **84.5** | **41.1** | 59.6 |
| **RLA-WMRL** | **95.9** | **73.1** | **20.7** | 74.1 | 39.9 | **60.7** |

WMRL 在 XArm、UR10e 上提高，但 Panda 两项下降；aggregate 是 `+1.1 percentage points`。作者给出的解释包括 Panda action dimensionality、camera occlusion、fine gripper details 与 demonstration diversity。不能只报平均值而隐藏明显的 per-robot regression。

## 6. Compute and reproducibility boundary

- RLA autoencoder 与 RLA-WM 各训练 100k steps；作者报告各自约需 `3 days on 4 x A6000 48GB`，并需要 256GB RAM。
- RLA-WM dynamics 在 ManiSkill 按 robot、IWS 按 scene/task 分别训练；论文没有把整个 model zoo 的总 GPU-days 汇总成一个数字。
- Minimalist WAM 每个 training trial 约 `1 day on one A4500 Ada 24GB`，且 policy 按 task 训练。
- WMRL 每个 trial 约 `3 hours on 4 x A6000` 或 `7 x A4500 Ada`，共有 15 seeds；内部并行 112 个 RLA-WM environments。
- Public repository 提供 configs、training/evaluation docs、Colab demo、pretrained weights 和 datasets 的下载说明；本阅读包未验证环境解析、checkpoint loading、dataset completeness 或表格复现。

这是一篇可读、可定位公开实现的 paper，但从本阅读包的证据不能宣称 turn-key reproduction 或已在 ASPIRE2A 上可运行。

## 7. Evidence Boundary / Limitations

- **质量指标不是 control metric：** LPIPS、SSIM 与 DINO L1 衡量 prediction fidelity，不直接证明 planner candidate ranking、first-action fidelity 或 closed-loop task success。
- **FLOPs 不是 latency：** headline efficiency 主要用 FLOPs 支持；缺少同卡 wall-clock、memory、batch scaling 与 ODE-step ablation。
- **autoregressive error 仍存在：** RLA-WM 对 action chunk 是 direct multi-step，但 30/60 environment-step evaluation 仍需 3/4 次 autoregressive model rollout。
- **partial observability：** 单个 `(s_t,s_{t+h})` frame pair 无法可靠表达 occluded object history；作者明确提出 multi-frame extension。
- **background motion：** eye-in-hand / humanoid setting 的 camera motion 可能占用 RLA capacity；当前 2D residual 不保证 view invariance。
- **visual-only state：** RLA-WM 不预测 future proprioception。
- **small-scale setting：** 证据来自 ManiSkill 与 IWS；internet-scale、open-world 和更广 embodiment 泛化仍未知。
- **WMRL reward scope：** VAR 依赖 time-aligned reference video；它不是 arbitrary-goal 或 free-form task reward。
- **selection boundary：** WMRL 报告 best-performing checkpoints/models；阅读时应核对 selection 与 evaluation seeds 是否完全隔离，以及 significance test 的具体定义。
- **decoder dependence：** image metrics 与 WMRL observation 都依赖额外 UNet 将 DINO tokens decode 成 RGB；“feature-only efficiency”与 downstream system cost 必须分开计时。

## 8. Why It Matters for the FYP

这篇 paper 对当前以 **LeWM + PushT** 为默认 baseline 的 FYP 有三层价值，但不是直接替代关系：

1. **新的 predictor design point：** LeWM 直接预测 future latent；RLA-WM 先生成 compact transition latent，再 decode future latent。它提出的是 representation + generative dynamics redesign，需要重新训练，不是对现有 LeWM checkpoint 的 inference-only acceleration。
2. **PushT relevance：** paper 同时报告 real/simulation PushT prediction、actionless-video BC 和 WMRL，但每条使用的数据、robot、metric 都不同。不要把 IWS PushT prediction quality、UR10e PushT WMRL success 与本项目 LeWM PushT CEM benchmark横向拼成同一表格。
3. **quantization / acceleration opportunity：** compact `z` 使 flow network 的 token count 下降，但 model 仍用 8-layer 1024-channel Transformer 和 30 Euler steps。若后续探索 low-bit 或 solver acceleration，应分别 profile condition network、Flow ODE loop、RLA decoder、DINO encoder 与 optional RGB UNet，不能只按 latent dimension 推断瓶颈。

与已有路线的边界：

| Route | Retrain? | Core object | Iteration | Primary correctness contract |
|---|---|---|---|---|
| LeWM | yes | future latent | recurrent across rollout horizon | task success / planner behavior |
| Fast-LeWM | yes | action-prefix future latents | parallel prefixes | task success + timing boundary |
| exact cache | no / ideally | action-independent activations | reuse within planning call | score/rank/elite/first-action equivalence |
| RLA-WM | yes | residual latent action | 30-step Flow ODE in compact space | prediction distribution + downstream task evidence |
| quantized RLA-WM | yes / PTQ-dependent | weights/activations in the above modules | same Flow ODE unless changed | quality + native memory/latency + downstream behavior |

## 9. Reading Route

### 20 minutes - 抓住 mechanism 与三条 claim

1. Abstract + Figure 1：写下 `DINO residual -> RLA -> future DINO tokens`。
2. Figure 2 + Section 3：区分 RLA autoencoder 与 RLA-WM Flow Matching。
3. Tables 1-3：分别标记 prediction、actionless-video policy、WMRL，禁止交叉借用 metric。
4. Section 5：记录 background motion、memory、proprioception、scaling 四个限制。

### 90 minutes - 能解释，也能质疑

1. Sections 3-4.1：追踪 `s_t, a -> condition -> z_0...z_1 -> decoder -> s_{t+h}`。
2. Table 1 + Appendix A2：核对 FLOPs、30 Euler steps、training compute、per-robot/per-task models。
3. Section 4.2：核对 action-labelled fraction、masked action loss、inference 丢弃 RLA head。
4. Section 4.3 + Figure 5：解释 VAR 如何把 reference video 变成 reward。
5. Table 3 + Appendix A3：专门解释为什么 average 提高但 Panda 失败。

### 3 hours - 形成 FYP prior-art card

1. 把 LeWM、Fast-LeWM、RLA-WM 画成三张 execution graph，标出 action horizon、autoregressive boundary 与 repeated calls。
2. 从 official code 只做 read-only navigation：定位 RLA encoder/decoder、condition network、Euler solver、UNet decode 与 WMRL rollout loop；记录 exact config，不运行训练。
3. 设计一个 predictor-level comparison schema：prediction error、candidate-order fidelity、native latency、peak memory、GPU utilization；先不要把它升级成 official CEM 或 closed-loop claim。
4. 对 30 Euler steps 做 conceptual ablation table：`steps / FLOPs / latency / DINO L1 / downstream fidelity`，明确哪些是 paper evidence、哪些是未来实验。

## 10. Reading Questions（留给你回答）

1. 为什么 `s_{t+h}-s_t` 比 absolute `s_{t+h}` 更接近低维 transition manifold？这依赖 DINO feature 的哪些局部性质？
2. Residual 是在 patch token coordinate 中逐元素相减；camera motion 或 patch correspondence 改变时，它还代表“physical displacement”吗？
3. RLA autoencoder 的 decoder 同时看到 `s_t`，它会不会把大部分信息旁路到 `s_t`，使 `z` 只编码难例或 dataset bias？
4. 2048-dimensional RLA 的真正 information bottleneck 有多强？Figure A1 的 64-dimensional example 是否有完整 quantitative ablation？
5. “temporal topology”来自 latent geometry，还是由 interpolation + decoder bias 产生的视觉平滑？什么 negative control 能区分？
6. RLA-WM 使用 30 Euler steps；若降到 1/2/4/8 steps，quality-cost Pareto frontier 如何变化？
7. Table 1 的 FLOPs 是否包含 DINO encoding、RLA decoder 与 RGB UNet？不同 baseline 的计量边界完全一致吗？
8. DINO-WM baseline 是作者重实现且升级到 DINOv3/action chunks；它与 official DINO-WM / LeWM 的 training protocol 有哪些不可比因素？
9. ManiSkill validation 每 task 只有 10 success + 10 failure episodes；Table 1 aggregate 的 uncertainty 有多大？
10. LPIPS/SSIM 是从另一个 learned UNet decoded RGB 上计算；decoder error 会如何污染 world-model comparison？
11. RLA-WM 的 stochastic samples 是否覆盖真实多模态 futures，还是只改善 single-sample regression metrics？论文缺什么 diversity/calibration 指标？
12. Minimalist WAM 的 improvement 来自 RLA physical supervision，还是仅来自更多 video frames 的 generic representation learning？需要什么 matched auxiliary-task control？
13. PushT 只给 15% action-labelled videos，而其他 tasks 是 5%；这个不对称如何影响 average 与 cross-task conclusion？
14. WMRL 的 VAR 依赖 time-synchronized reference；遇到多条同样成功但节奏不同的 trajectory 时会不会错误惩罚？
15. Table 3 选择 best-performing model/checkpoint 后再评估 1500 episodes；selection bias 与 train-seed variance如何报告才完整？
16. Panda 两项下降说明哪一层 failure：world-model error、decoded observation shift、policy optimization，还是 reward misalignment？
17. 对当前 LeWM + PushT 主线，RLA 是应该作为 architecture baseline、surrogate predictor，还是只作为 future-work prior art？什么最小 gate 能区分？
18. 若量化 RLA-WM，哪个模块最值得先做：DINO encoder、condition Transformer、30-step flow network、RLA decoder 还是 RGB UNet？需要先采什么 profile 才能回答？

## 11. Meeting Card

- **Paper：** Zhang et al., *Learning Visual Feature-Based World Models via Residual Latent Action*, arXiv:2605.07079v1, 2026。
- **Core idea：** 把 high-dimensional future DINO feature generation 分解为 compact DINO-residual latent `z` 的生成，以及从 `(s_t,z)` 单次 decode future features。
- **Strongest evidence：** Table 1 在 ManiSkill/IWS 的 LPIPS、SSIM、DINO L1 全部领先；FLOPs `3.5T`，显著低于 Vid2World `1.1P`，但高于 DINO-WM `2.1T`。
- **Downstream evidence：** actionless-video RLA auxiliary task 将 average SR `27.2 -> 35.6`；WMRL average `59.6 -> 60.7`，但 Panda 两项退化。
- **Critical caveat：** FLOPs 不等于 wall-clock；prediction metrics 不等于 planning success；WMRL reward 依赖 time-aligned reference videos 与 RGB UNet decode。
- **Compute reality：** RLA autoencoder 与 RLA-WM 各约 3 days on 4 x A6000；30-step Flow ODE 与多个 separately trained dynamics models 使它不是“只因 latent compact 就自然便宜”。
- **FYP connection：** 它是需要 retraining 的 residual-latent generative predictor baseline，不是 exact cache，也不能直接替代当前 LeWM + PushT baseline。
- **Question to bring to meeting：** 对 FYP 最有价值的是复现整套 RLA-WM，还是只把“compact transition latent”抽成 predictor-level、可证伪的 LeWM 改造？

## 12. Citation

```bibtex
@article{zhang2026learning,
  title         = {Learning Visual Feature-Based World Models via Residual Latent Action},
  author        = {Zhang, Xinyu and Xu, Zhengtong and Tao, Yutian and Wang, Yeping and She, Yu and Boularias, Abdeslam},
  journal       = {arXiv preprint arXiv:2605.07079},
  year          = {2026},
  eprint        = {2605.07079},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  version       = {v1},
  url           = {https://arxiv.org/abs/2605.07079}
}
```
