# 论文代码复现候选审计（2026-09-03）

## 结论

如果只选一个、并且优先贴合当前 VLA / simulation evaluation 的 FYP：

> **首选 `vla-eval: A Unified Evaluation Harness for Vision-Language-Action Models`，用 released OpenVLA checkpoint 复现 standard LIBERO evaluation。**

它不需要 pretraining，实验是 evaluation-only；官方仓库包含 containers、configs、model servers、tests 与 reproduction documentation。先跑一个 LIBERO suite 即可闭环，之后能自然扩展到 `LIBERO-Plus`、latency/memory profiling 和 quantized checkpoint。

如果想先复现一篇更“算法型”、变量更少的 quantization paper：

> **首选 `QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs`，先做 Llama 7B/8B 的 rotation + PTQ + perplexity/zero-shot evaluation。**

QuaRot 官方代码完整；论文报告 Llama2-70B 的 model transform 约 5 分钟、GPTQ 约 2 小时，均在单张 A100 上完成。AMD Quark 另有独立 QuaRot implementation，并公开了 Llama-3-8B 的 WikiText-2 perplexity 数值，因而比多数 2026 VLA quantization preprints 更适合作为第一次复现。

不要把 `ActQuant` 或 `QVLA` 作为第一篇：它们方向更贴近 VLA quantization，但当前 public artifact 分别存在 evaluation glue/checkpoint provenance 和 fake-quant/packed-kernel 对应关系问题。

## 审计边界

本地四套 corpus 共有 94 份 PDF。去掉 duplicate version、survey、specification、supplement 和纯 commentary 后，保留 **82 篇 substantive papers**：

| Corpus | 原始 PDF | 去重后的 paper identities |
|---|---:|---:|
| `reading-guide` | 51 | 43 |
| `sim-eval-vla-reading-guide` | 15 | 15 |
| `vla-quantization-literature-review-alternative` | 20 | 18 个新增 |
| `dan-alistarh-quantization-reading-guide` | 8 | 6 个新增 |
| **合计** | **94** | **82** |

这里没有重新计算 SHA-256，也没有重复做 PDF integrity check；这些检查对“选哪篇复现”没有增量价值。

## 判定口径

`✓` 严格通过需要同时满足：

1. 论文关键方法、config、evaluation path 和必要 checkpoint/data reference 可从公开仓库获得；只有 project page、tokenizer、fake-quant demo 或未发布的 evaluation driver 不算“完整”。
2. 不做 foundation model pretraining。允许 PTQ、QAT、LoRA/post-training、released-checkpoint evaluation，以及论文自身规模较小的 RL/policy training。
3. 一条能支撑论文核心 claim 的实验链路可在最多 `2×A100 80GB` 上完成；不能拿 toy demo 代替论文 experiment。
4. custom ASIC/RTL result、真实机器人全协议、或必须多于 2 GPU 的 headline experiment 不用“理论上能跑”蒙混过关。

`△` 表示只有限定 slice 可行，或者还差一项关键信息；不能在报告中写成“整篇论文已复现”。`✗` 表示不满足当前选择条件。

“网上有人复现”分三档：

- **独立数值复现**：第三方给出代码、protocol 和数值。
- **部分复现 / artifact audit**：跑通了主要路径，但结果有明显 gap，或发现 artifact 问题。
- **adoption only**：被 library/runtime 集成，不等于论文表格被独立复现。

## 严格通过的 11 篇

| 优先级 | Paper | 允许的复现范围与硬件 | Public code | 网上复现证据 | 建议 |
|---:|---|---|---|---|---|
| 1 | **vla-eval** | released checkpoint 的 standard LIBERO evaluation；单 A100 可跑一个 suite，2 卡可做 episode sharding | [official harness](https://github.com/allenai/vla-evaluation-harness) | 仓库报告 OpenVLA `76.2 vs 76.5`、OpenVLA-OFT `96.7 vs 96.8/97.1` 等 rerun；这是作者团队的 cross-codebase reproduction，**尚未找到完整独立复现** | **VLA/FYP 首选** |
| 2 | **QuaRot** | rotation + RTN/GPTQ + perplexity/zero-shot；论文的 70B quantization 也报告在 `1×A100` 完成 | [official code](https://github.com/spcl/QuaRot) | [AMD Quark tutorial](https://quark.docs.amd.com/release-0.8.2/pytorch/tutorial_quarot.html) 给出独立 implementation 与 Llama-3-8B perplexity；属于有数值的独立机制复现 | **Quantization 首选** |
| 3 | **GPTQ / OPTQ** | one-shot weight-only PTQ；官方实验以单张 A100 为主，OPT-175B quantization 约 4 GPU-hours | [official code](https://github.com/IST-DASLab/gptq) | [Hugging Face AutoGPTQ integration](https://huggingface.co/blog/gptq-integration) 在单 A100 给出 latency、throughput 与 peak-memory benchmark；大量 adoption，但不等于逐表复现 | 成熟、资料最多 |
| 4 | **AWQ** | 7B/13B W3/W4 search、quantization 与 TinyChat kernel；A100/4090/Orin 路线公开 | [official code](https://github.com/mit-han-lab/llm-awq) | [independent mechanism reproduction](https://daiwk.github.io/auto-research/reproductions/2306.00978-awq/)；另有 [pure-PyTorch Qwen3-8B implementation](https://konic.io/research/awq-int4-qwen3-8b) | 与 GPTQ 二选一也可 |
| 5 | **LIBERO-Plus** | released VLA checkpoint 的 robustness evaluation；无需训练，GPU 主要承担 policy inference | [official code](https://github.com/sylvestf/LIBERO-plus) | [第三方 full-run issue #61](https://github.com/sylvestf/LIBERO-plus/issues/61) 跑完 10,030 instances；多数类别接近，但 Sensor Noise 差异大 | `vla-eval` 后的第二阶段 |
| 6 | **LIBERO-PRO** | OpenVLA 等 checkpoint 的 OOD evaluation；单卡足够 | [official code](https://github.com/Zxy-MLlab/LIBERO-PRO) | [issue #20](https://github.com/Zxy-MLlab/LIBERO-PRO/issues/20) 报告 original OpenVLA 在 perturbation suites 全 0 且未解决；是失败/摩擦证据，不是成功复现 | 符合硬条件，但不宜首选 |
| 7 | **LIBERO** | benchmark 与小规模 policy/evaluation；不涉及 VLA foundation pretraining | [official code](https://github.com/Lifelong-Robot-Learning/LIBERO) | 被大量 VLA repos 与 `vla-eval` 集成；本轮未找到一份独立、逐表对齐的完整复现报告 | 作为环境 baseline，不单独做论文首选 |
| 8 | **FlashSAC** | paper recipe 的 GPU-parallel training；wall-clock 在单 RTX 5090，sim-to-real training 使用单 A100 | [official code](https://github.com/Holiday-Robot/FlashSAC) | **未找到独立成功复现报告**；官方 repo 的 `100+ tasks` 是 live capability claim，不是论文独立复现 | 算力满足，但环境矩阵较重 |
| 9 | **RaBitQ** | ANN/vector quantization，主要 CPU/SIMD 路线，远低于 2×A100 | [original code](https://github.com/gaoj0017/RaBitQ) | 后续 comparison paper 有作者谱系重叠；**未找到中立第三方完整 rerun** | 易跑，但与 VLA/LLM 有距离 |
| 10 | **Practical and Asymptotically Optimal Quantization...**（multi-bit RaBitQ） | multi-bit ANN quantization，CPU/GPU 均在预算内 | [extension code](https://github.com/VectorDB-NTU/Extended-RaBitQ) | 主要证据来自原团队后续工作，未找到中立第三方完整 rerun | 适合 vector quantization，不是 VLA |
| 11 | **Revisiting RaBitQ and TurboQuant** | matched comparison 与 Llama-3.1-8B KV-cache experiment；≤1 A100/CPU | [reproduction repo](https://github.com/VectorDB-NTU/rabitq-turboquant-comparison) | 这篇本身是 reproduction/audit，但作者与 RaBitQ 有重叠；尚无再独立复现 | 很可执行，但研究问题偏 attribution/audit |

## 接近通过，但必须限定范围

| Paper | 可做的 slice | 为什么不是严格通过 | 网上情况 |
|---|---|---|---|
| **OpenVLA** | 直接使用四个 released LIBERO fine-tuned checkpoints 做 evaluation | 原始 pretraining 为大规模多 GPU，明确排除；只能声称复现 Appendix LIBERO evaluation | [第三方 repo](https://github.com/lz-googlefycy/openvla-libero) 在单 H20 跑了 400 rollouts：Goal/Long 接近，Object 相差 `-28.4` points，属于**部分成功** |
| **π₀ / π₀.₅ / OpenVLA-OFT** | 用 released checkpoint 经 `vla-eval` 做 evaluation | 原论文 training/robot protocol 不满足本任务；只能复现 evaluation slice | `vla-eval` 有下游 rerun，不能等同于复现原 paper training |
| **FlashVLA** | released checkpoint 的 LIBERO/RoboTwin evaluation、latency profiling | 完整 RoboTwin fine-tuning 使用 `8×H200` | 官方代码完整；本轮未找到第三方成功 reproduction |
| **SmoothQuant** | OPT-30B 或更小模型的 W8A8 PTQ | 原文 OPT-66B/175B、MT-NLG 530B 的部分实验需要 4–8 GPU；不符合“整篇主表 ≤2 卡” | 有第三方 comparative evaluation 与大量 TensorRT/ONNX adoption，逐表复现证据较弱 |
| **SpinQuant** | Llama 7B/8B learned rotation | 70B path 使用多 GPU/FSDP；公开 issues 仍有 PPL/OOM/一致性摩擦 | 第三方 comparison 有 rerun；官方 issues 更多是问题报告，不是成功复现 |
| **AQLM** | 7B：约 1 天/1×A100，或约 14.5 小时/2 GPU | 70B 完整 experiment 需要多 GPU、数天 | [issue #49](https://github.com/Vahe1994/AQLM/issues/49) 有独立运行且指标接近、未完全匹配 |
| **HAQ** | CNN/ImageNet 的核心 search | 算力可行，但依赖旧 PyTorch/CUDA 与 target-hardware simulator；复现收益与当前 VLA 主线较低 | 只有 unofficial implementations/adoption，没找到可靠逐表复现 |
| **QuantVLA** | released π₀.₅/GR00T checkpoint 的 PTQ + LIBERO subset | [official repo](https://github.com/AIoT-MLSys-Lab/QuantVLA) 较完整，但论文只写 A100 GPUs，数量不明确；本轮未找到独立结果 | **先确认 GPU count，再升级** |
| **ActQuant** | OpenVLA-OFT/π₀.₅ 的 calibration、conversion、released 3-bpw checkpoint evaluation | [official repo](https://github.com/arashakb/ActQuant) 没有自带 LIBERO driver，需要改 upstream；[issue #2](https://github.com/arashakb/ActQuant/issues/2) 对 checkpoint provenance 提出未解决疑问 | 有 artifact audit，没有完整独立端到端成功复现 |
| **SIMPLER** | 单个 released policy 的 simulation evaluation | 完整论文 claim 依赖 paired real trials/system identification，只有 GPU 不足以重建整条证据链 | issues 中有 A100 Vulkan、success criteria 与 result inconsistency；未找到干净的独立完整复现 |
| **Embodied.cpp** | 单一 backend/model 的 conversion + inference benchmark | 全论文跨 heterogeneous robots/backends 需要额外设备；不是纯 2×A100 experiment | 官方 repo 可运行，未找到独立逐表复现 |

## 两个看似合适、实际应暂缓的 VLA quantization 项目

### QVLA

论文核心 LIBERO 实验用 RTX 4090，算力本身符合；但 public repo 当前主要给 sensitivity proxy、gate assignment、fake weight quantization 与 evaluation path。公开 issues [#5](https://github.com/AutoLab-SAI-SJTU/QVLA/issues/5)、[#6](https://github.com/AutoLab-SAI-SJTU/QVLA/issues/6)、[#7](https://github.com/AutoLab-SAI-SJTU/QVLA/issues/7) 讨论 code-paper inconsistency、fake quantization、缺少真实 packed low-bit kernel，以及当前代码无法重现 speedup/memory claim。因此它不满足本次“代码完整开源”。

ActQuant Appendix G 确实 rerun 了 QVLA public implementation，但 calibration budget 从原文 512 episodes 改为 60 episodes，结果也不同；这只能算 partial reproduction，不能给 QVLA 发“已成功复现”通行证。

### ActQuant

ActQuant 比 QVLA 更接近可复现，PTQ 本身也能在单 L40S/A100 级资源完成；但当前必须从 upstream 拿 LIBERO driver 并手改，released checkpoint 的 provenance 又受到公开质疑。它适合作为第二篇或 artifact-audit project，不适合作为第一个需要尽快拿到可信结果的项目。

## 82 篇逐项 disposition

下面是完整去重 corpus。`✓` 是严格通过；`△` 是仅 slice/待确认；`✗` 是不进入本轮候选。

### `reading-guide`：43 篇

| # | Paper | 状态 | 主要原因 |
|---:|---|:---:|---|
| 1 | OpenVLA | △ | released-checkpoint LIBERO eval 可做；pretraining 排除 |
| 2 | π₀ | △ | checkpoint eval 可做；原训练/robot protocol 不可复现 |
| 3 | FAST | ✗ | public artifact 偏 tokenizer；不足以闭环论文训练/evaluation |
| 4 | π₀.₅ | △ | checkpoint eval 可做；原训练排除 |
| 5 | π*₀.₆ | ✗ | RECAP/data/value pipeline 与 real-robot protocol 未完整公开 |
| 6 | ω-0 | ✗ | code/checkpoint/whole-body data 链路未确认完整 |
| 7 | BitVLA | ✗ | 约 `8×H800` 7 天 + `16×H800` 14 天，且是 native pretraining |
| 8 | FP8 Formats for Deep Learning | ✗ | datatype paper，不是完整 algorithm reproduction package |
| 9 | Microscaling Data Formats | ✗ | format/spec + native hardware 依赖 |
| 10 | Pretraining LLMs with NVFP4 | ✗ | 明确是 pretraining，且依赖 Blackwell |
| 11 | SmoothQuant | △ | ≤30B slice 可做；原文最大实验超过 2 GPU |
| 12 | AWQ | ✓ | 完整 PTQ/search/kernel 路线，单卡代表实验可行 |
| 13 | SpinQuant | △ | 7B/8B 可做；70B 原协议多 GPU |
| 14 | HAQ | △ | 计算量可行；旧 stack/target-hardware simulator 摩擦大 |
| 15 | TurboQuant | ✗ | 未建立完整、可核验的 official code-paper 对应关系 |
| 16 | LoRA | ✗ | GPT-3 headline 不开放；小模型 LoRA 只是局部 baseline |
| 17 | Outlier Suppression | ✗ | official repo 的 QAT 部分未整理完成 |
| 18 | Flow Matching for Generative Modeling | ✗ | 原论文完整 generative training/code 链路不满足本条件 |
| 19 | Self-Flow | ✗ | multimodal pretraining/大规模训练 |
| 20 | FlashVLA | △ | released-checkpoint eval 可做；完整 fine-tuning 为 8×H200 |
| 21 | REPA | ✗ | ImageNet DiT training 规模与本任务目标不匹配 |
| 22 | iREPA | ✗ | 大规模 generative training；不是轻量 post-training |
| 23 | FLARE | ✗ | VLA/WAM training、sim/robot assets 与 code 链路未确认完整 |
| 24 | Spatial Forcing | ✗ | multi-view/3D teacher/VLA training 与公开 artifact 不完整 |
| 25 | FRAPPE | ✗ | two-stage VLA training + RoboTwin/real robot，超出首复现范围 |
| 26 | FutureVLA | ✗ | predictive data/model pipeline 未完整公开 |
| 27 | VEGA | ✗ | teacher/VLA checkpoint 与训练链路未完整公开 |
| 28 | AGRA | ✗ | WAM training/action grounding pipeline 未完整公开 |
| 29 | SAM3D-Guided Alignment | ✗ | SAM3D/3D teacher/robot data 依赖，artifact 未闭环 |
| 30 | Robust-WAM | ✗ | WAM pretraining/OOD protocol 未完整公开 |
| 31 | Mind-VLA | ✗ | VAE/VGGT/3D pipeline 与 checkpoint 不完整 |
| 32 | ReconVLA | ✗ | training/data/grounding auxiliary 链路未闭环 |
| 33 | FlashSAC | ✓ | paper 训练为单 RTX5090/单 A100 级；official code 完整 |
| 34 | MEM | ✗ | code、weights、data 当前未完整公开 |
| 35 | Apprentice | ✗ | 可重写核心方法，但缺少现代、完整论文 reproduction package |
| 36 | LLaVA | ✗ | multimodal pretraining；不是本任务允许的轻量 post-training |
| 37 | OpenVLA-OFT | △ | released checkpoint eval 可做；论文训练为 8×A100/H100 |
| 38 | LLM.int8() | ✗ | BLOOM-176B headline 约 4×A100；bitsandbytes adoption 不等于逐表复现 |
| 39 | GPTQ / OPTQ | ✓ | 完整 PTQ + CUDA kernel；论文 175B quantization 也可单 A100 |
| 40 | QuaRot | ✓ | 完整 rotation/PTQ/kernel；70B quantization 报告单 A100 |
| 41 | RaBitQ | ✓ | 完整 ANN quantization code；资源远低于上限 |
| 42 | Multi-bit RaBitQ | ✓ | 完整 extension code；资源远低于上限 |
| 43 | Revisiting RaBitQ and TurboQuant | ✓ | reproduction repo + matched comparison；≤1 GPU/CPU |

### `sim-eval-vla-reading-guide`：15 篇

| # | Paper | 状态 | 主要原因 |
|---:|---|:---:|---|
| 1 | LIBERO | ✓ | benchmark/code/data 完整；小规模 training/eval 可在预算内 |
| 2 | RoboDojo | ✗ | hidden layouts、real evaluation 与多 robot protocol 无法仅靠 GPU 闭环 |
| 3 | RoboTwin 2.0 | ✗ | 50 tasks/5 embodiments/100k+ trajectory ecosystem 太大 |
| 4 | RoboTwin | ✗ | digital-twin generation、real hardware 与 training chain 较重 |
| 5 | LIBERO-PRO | ✓ | evaluation-only 且 code 完整；但已有 unresolved result issue |
| 6 | LIBERO-Plus | ✓ | evaluation-only，代码完整，并有第三方 full-run |
| 7 | LIBERO-X | ✗ | 缺 π₀.₅ fine-tuned checkpoint，training config/max_steps 仍有缺口 |
| 8 | LIBERO-Para | ✗ | paraphrase generation pipeline 尚未发布，不能复现完整 paper pipeline |
| 9 | SIMPLER | △ | simulation subset 可做；full claim 需要 paired real trials |
| 10 | VLABench | ✗ | 当前 issues 涉及 metric/evaluation/action-execution 问题 |
| 11 | RoboCasa365 | ✗ | 365 tasks、2,500 scenes、500k+ demos，不适合本预算首复现 |
| 12 | vla-eval | ✓ | released-checkpoint evaluation harness，单/双卡可闭环 |
| 13 | WorldEval | ✗ | world-model weights/training/real calibration 链路未完整开放 |
| 14 | WorldGym | ✗ | world-model checkpoint + VLM reward/OOD pipeline 未闭环 |
| 15 | dWorldEval | ✗ | diffusion evaluator training、failure data、progress labels 未完整开放 |

### Alternative VLA quantization corpus：18 篇新增

| # | Paper | 状态 | 主要原因 |
|---:|---|:---:|---|
| 1 | SQIL | ✗ | QAT + real UR5/Jetson protocol；未确认完整公开复现入口 |
| 2 | RLRC | ✗ | 约 320 GPU-hours，SFT/PPO/data pipeline 未形成可信首复现包 |
| 3 | SQAP-VLA | ✗ | CogACT/pruning/W4A4/SIMPLER 链路未确认完整发布 |
| 4 | Towards Accessible Physical AI | ✗ | real SO-101 protocol/model provenance；不是可独立闭环的软件实验 |
| 5 | QVLA | ✗ | code-paper mismatch、fake quant、packed kernel/speed claim 缺口 |
| 6 | HB-VLA | ✗ | 1-bit PTQ code/checkpoint/metadata 链路未确认完整 |
| 7 | QuantVLA | △ | repo 较完整；A100 数量未明、无独立结果 |
| 8 | LiteVLA-Edge | ✗ | 主要是 timing/runtime；缺 matched success-rate experiment |
| 9 | DyQ-VLA | ✗ | custom CUTLASS/fused kernel 与完整代码未确认公开 |
| 10 | DA-PTQ | ✗ | public template/trajectory/evaluation artifact 尚不完整 |
| 11 | ActQuant | △ | PTQ 可做；缺 bundled LIBERO driver，checkpoint provenance 待澄清 |
| 12 | HoloQ-VLA | ✗ | W4A4 full-stack code/robot pipeline 未确认完整公开 |
| 13 | Mix-QVLA | ✗ | evidence-map/calibration/OpenVLA-OFT path 未形成完整 public pipeline |
| 14 | Embodied.cpp | △ | 单 backend 可做；完整 heterogeneous hardware claim 需额外设备 |
| 15 | VQVLA | ✗ | result 依赖 custom accelerator/cycle simulation/28nm synthesis |
| 16 | FlashDrive | ✗ | paper measurement 使用 5 GPUs，超过上限 |
| 17 | Bit-Flip Attacks on VLA | ✗ | attack/real-robot/code artifact 尚不足以作为稳定首复现 |
| 18 | SpecVLA | ✗ | heterogeneous custom hardware/cycle simulation，不是 2×A100 software reproduction |

### Dan Alistarh corpus：6 篇新增

| # | Paper | 状态 | 主要原因 |
|---:|---|:---:|---|
| 1 | Optimal Brain Compression | ✗ | official repo 明示完整 YOLO/BERT integrations 未提供 |
| 2 | AQLM | △ | 7B 在预算内；70B 主实验需要多 GPU/数天 |
| 3 | HIGGS | ✗ | 当前主要是 runtime integration/预量化模型，缺独立完整 reproduction repo |
| 4 | HALO | △ | PEFT/小模型可能适配 2×A100；headline 为 4/8×4090 且无独立复现 |
| 5 | QMoE | ✗ | 1.6T model/CPU RAM/4×A6000 或 8×3090 资源超限 |
| 6 | “Give Me BF16 or Give Me Death”? | ✗ | deployment audit 涉及 405B/multi-GPU，不是单一可复现 method |

## 推荐的最小复现定义

### 路线 A：最贴合 FYP——`vla-eval`

目标不是“安装成功”，而是得到一个可审计的 evaluation result：

1. 固定 `OpenVLA-7B` released LIBERO checkpoint、commit、container 和 action preprocessing。
2. 先跑一个 suite；每 task 10 rollouts，固定 seed，并保存 success/failure video。
3. 同时记录 peak GPU memory、model latency、simulator wall time、episode timeout/crash。
4. success denominator 必须区分 policy failure 与 infrastructure failure。
5. 与 paper/harness 数值比较；差异超过 sampling uncertainty 时，先查 dependency、initial state、action un-normalization 和 termination rule。
6. 通过后才扩展到四 suites，再接 `LIBERO-Plus` 或 quantized checkpoint。

建议验收：一个 suite 的 success rate、逐 task 明细、全部失败录像、peak memory 与 P50/P95 inference latency齐全；这时才算一次完整 smoke reproduction。

### 路线 B：最干净的算法复现——`QuaRot`

1. 用 Llama 7B/8B，不要一开始上 70B。
2. 复现 FP16、RTN W4、QuaRot W4A4KV4 三个点。
3. 固定 calibration corpus/sequence count、Hadamard seed、GPTQ config 和 evaluation harness。
4. 先对 WikiText-2 perplexity，再做少量 zero-shot tasks。
5. 分开报告 fake-quant accuracy 与 real-kernel latency；二者不是同一个 claim。

建议验收：perplexity 与论文/官方 config 在合理误差内，并能解释剩余差异；随后才测 real kernel latency。

## 最终选择

- **主线第一篇：`vla-eval`。** 它最贴合现有 FYP，最快产生可信、可扩展的 VLA evaluation artifact。
- **算法第一篇：`QuaRot`。** 它的 code/compute/independent evidence 三项最均衡。
- **不要首选：`ActQuant`、`QVLA`、`FlashVLA` full training、`BitVLA`。** 前两者 artifact 尚有关键缺口，后两者完整训练超过资源边界。

这份筛选只声称 2026-09-03 的 targeted web audit；“未找到独立复现”不等于互联网上绝对不存在，只表示在 official repo、issues、paper/project page 和定向搜索中没有找到足够可信的公开报告。
