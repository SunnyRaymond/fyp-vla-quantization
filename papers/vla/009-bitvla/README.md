# BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation

> **Reading-list role**: Critical scan — VLA quantization / edge-deployment candidate  
> **Verification**: `verified-full-text`; arXiv explicitly says “Work in progress”  
> **Recommended effort**: **Critical scan**，重点审查 memory/latency measurement boundary

> **Local full text**: [paper.pdf](paper.pdf)

## 1. Paper identity

| Field | Value |
|---|---|
| Authors | Hongyu Wang, Chuyan Xiong, Ruiping Wang, Xilin Chen |
| Year / version | 2025–2026; arXiv 2506.07530, v1 2025-06-09, latest checked v2 2026-03-01 |
| Venue / status | arXiv preprint; comments: **Work in progress** |
| Primary source | [arXiv:2506.07530](https://arxiv.org/abs/2506.07530) · [DOI](https://doi.org/10.48550/arXiv.2506.07530) |
| Code / project | [Official repository](https://github.com/ustcwhy/BitVLA) |

## 2. One-sentence takeaway

BitVLA 从 native ternary BitNet 出发，再用 Quantize-then-Distill 把 vision encoder 压成 W1.58A8，在 LIBERO 以 1.4 GB model memory 达到 96.0% average success，并在 A100 setup 报告 73 ms latency，但 real-world 与 edge evidence 仍有限。

## 3. Background and prerequisites

- **Technical lineage**：OpenVLA → OpenVLA-OFT parallel action chunking → low-bit LLM/BitNet → native low-bit VLA。
- **读前知识**：ternary/1.58-bit quantization、STE、absmean weight quantizer、per-token absmax activation quantizer、knowledge distillation、representation alignment、BitBLAS、LIBERO。
- **关键 notation**：`W1.58A8` = ternary weights + INT8 activations；`PTQ` = post-training quantization；`QAT` = quantization-aware training。

## 4. Problem

- **Target setting**：在 memory-constrained robot hardware 上部署 VLA，同时保留 manipulation performance。
- **Bottleneck**：7B VLA 的 weights、activations 与 autoregressive latency 太大；deployment 后才做 INT8/INT4 PTQ 可能显著掉点。
- **Why previous methods are insufficient**：小模型未必保留能力；PTQ 与原始 optimization dynamics 不一致；1-bit LLM 向 multimodal/robot action 扩展尚未充分验证。

## 5. Method

### 5.1 System view

`images + language + proprioceptive state` → W1.58A8 `SigLIP-L` vision encoder → connector → native ternary `BitNet b1.58 2B4T` → parallel continuous action head → action chunk。

Training path：multimodal initialization → Quantize-then-Distill vision encoder → robotics pretraining on Open X-Embodiment → downstream OpenVLA-OFT-style SFT。

### 5.2 Core mechanism

- LLM weights native ternary `{−1,0,1}`；vision encoder 也被量化为 1.58-bit weights；activations symmetric INT8。
- **Stage 1**：LLaVA-style training；先用 558k image-caption samples 只训 connector，再在 10M MammoTH-VL subset 做 instruction tuning。
- **Stage 2 — Quantize-then-Distill**：BF16 vision teacher frozen；初始化 W1.58A8 student，只更新 student encoder；loss 为 task CE 加多层 hidden-feature alignment `L_aux`。
- **Stage 3 — Robotics**：约 1M Open X-Embodiment samples；先用每 dimension 256-bin next-action CE pretrain，再用 parallel action chunk + `L1` trajectory regression downstream fine-tune。
- LIBERO chunk `K=8`；efficiency test 统一 `K=25`；保留 causal mask；BitBLAS custom kernel 执行 ternary-weight/INT8-activation matrix multiply。
- compute：VLM stages 约 7 days on 8×H800；robotics pretraining 约 14 days on 16×H800。

### 5.3 What is actually new

真正贡献是 native low-bit backbone 与 teacher-guided vision quantization 被放进 VLA training recipe，而不是对 full-precision VLA 做事后 PTQ。OpenVLA-OFT action head/chunking、SigLIP、LLaVA curriculum、distillation 与 BitBLAS 都是已有 building blocks。

**术语边界。** “1-bit VLA”不表示所有运算都是 1-bit：weights 是 ternary（平均 1.58 bits），activations 是 INT8，scaling/dequantization、embeddings/heads/runtime buffers 仍可能使用更高精度。汇报应说 **native W1.58A8 backbone**。

### 5.4 Quantize-then-Distill 是不是这篇发明的？

答案要分两层：

- **Exact name and BitVLA recipe**：BitVLA 明确写 “we introduce Quantize-then-Distill”，目前核验到的 primary sources 中没有更早同名 method。它把 frozen BF16 SigLIP teacher、W1.58A8 student、answer-token Language Modeling loss 与 layerwise hidden-feature alignment 放到 VLA curriculum 的固定位置，这是 BitVLA-specific contribution。
- **Generic idea**：不是首创。2017/2018 的 [Apprentice](../010-apprentice-quantization-distillation/README.md) 已经系统研究 `full-precision teacher + low-precision student`。其 Scheme-C 先用 full-precision weights 初始化 student，再降低 weights/activations precision，并在 trained full-precision teacher supervision 下 fine-tune，与 BitVLA 的 high-level sequence 非常接近。

因此最严谨的表述是：

> BitVLA introduced the named, VLM-specific Quantize-then-Distill instantiation；quantization-aware Knowledge Distillation itself predates BitVLA。

两者差别也不能省略：Apprentice 面向 ResNet/ImageNet、主要 distill classifier outputs；BitVLA 面向 SigLIP inside VLM、只更新 quantized vision student，并通过 every-layer MSE 保护 multimodal representation geometry。把 Apprentice 称作 exact “original BitVLA method paper” 也不准确；它是 direct conceptual predecessor。

### 5.5 LLaVA 在 BitVLA 中到底负责什么？

[LLaVA](../011-llava/README.md) 是 multimodal training recipe 的来源，不是 BitVLA 的 robot policy backbone：

1. LLaVA Stage 1 冻结 vision encoder + LLM，只训 projector，学习 `visual tokens → word-embedding space` alignment。
2. LLaVA Stage 2 继续冻结 vision encoder，训练 projector + LLM 做 visual instruction following；loss 只计算 Assistant answer tokens。
3. BitVLA 复用这个 curriculum，但替换为 `SigLIP-L + BitNet b1.58 2B4T`，alignment data 用 LLaVA-1.5-558K，instruction data 用 10M MAmmoTH-VL subset。
4. 之后才加入 BitVLA-specific Quantize-then-Distill 与 robotics training。

LLaVA 的作用是先建立 stable multimodal interface，避免一开始就把 modality-alignment error 与 quantization error 混在一起。

### 5.6 bitsandbytes 在 Sec. IV-B 中做了什么？

`bitsandbytes` 是 low-bit PyTorch software library，不是一种单一 numeric format。BitVLA 用它对公开的 OpenVLA/OpenVLA-OFT checkpoints 做 **post-training baseline**：

`full-precision checkpoint → bitsandbytes INT8/“INT4” backbone loading → LIBERO evaluation`

BitVLA 自身并不走 bitsandbytes；native W1.58A8 inference 用 BitBLAS custom kernel。BitVLA citation `[9]` 对应 [LLM.int8()](../013-llm-int8-bitsandbytes/README.md)：该 algorithm 用 vector-wise INT8 处理 regular features，并把少量 outlier feature dimensions 留给 FP16 matmul。因此 `INT8` baseline 也不是所有 operation 都是 pure INT8。

更重要的是，paper 的 `INT4` row 没有给出 `FP4 vs NF4`、group size、compute dtype、double quantization、skipped modules 或 software version。Current bitsandbytes 4-bit path 暴露 FP4/NF4 options，所以 Table II 的 “INT4” 不是一个 fully specified reproducible format；不能仅从 paper 判断它究竟用了哪组 config。

### 5.7 “Each query includes three 224×224 images” 是哪三张？

这句话属于 **ALOHA-shaped inference benchmark**，不是 BitVLA 所有实验的统一 input：

| Setting | Visual inputs | State | Chunk |
|---|---|---|---|
| LIBERO, Sec. IV-B | external/third-person camera + one wrist camera | proprioception projected to one token | `K=8` |
| Franka real-world, Sec. IV-C | one global third-person RealSense view | joint/gripper state | `K=10` |
| Fig. 6 efficiency benchmark | one top-down/third-person camera + left-wrist + right-wrist | 14-D ALOHA joint state | `K=25` |

OpenVLA-OFT+ 对每张 image 使用同一个 shared vision encoder，每张产生 256 patch embeddings；three views 共 768 visual tokens，再与 one state embedding、language tokens 和 action-query slots concatenate。三张图是同一 control timestep 的 synchronized multi-view observation，不是连续 video frames，也不是三次 flow state。

BitVLA 直接采用 [OpenVLA-OFT](../012-openvla-oft/README.md) Table III 的 ALOHA input shape 与 baseline numbers 来做 Fig. 6 comparison。`341.1 Hz` 是 `K/latency` 式的 amortized action-generation throughput，不等于每秒 341 次重新看三张图。

### 5.8 BitVLA + flow matching：先纠正问题中的一个概念

π-series 的 flow matching 不是 “flow matching for continuous visual”。它生成的是 **continuous action chunk**：

`p(A_t | images, language, proprioception)`

images 只作为 fixed conditioning context。一次 query 内，solver 的 5–10 个 steps 更新 noisy actions，不会在每个 step 读入新的 camera frame。若要更频繁利用 visual feedback，需要缩短 executed action prefix、提高 replan frequency；flow matching 本身不能代替 closed-loop observation refresh。

### 5.9 Could flow matching improve BitVLA?

#### Potential upside

- BitVLA/OFT 的 L1 optimum 是 conditional median；若同一 observation 下有 multiple valid strategies，single deterministic chunk 可能落在 mode 之间或只学一个 mode。Flow matching 可以表示 joint multimodal distribution over the entire `K×D` chunk。
- 以整个 trajectory chunk 为 random variable，可以显式学习跨 timesteps 与双臂 dimensions 的 correlation；这可能帮助 contact-rich、bimanual、tool-use 与 recovery tasks。
- π₀ 的 action expert 将 continuous/noisy-action/timestep-specific computation 与 VLM semantic path 分开，这与 BitVLA “bulk backbone low-bit、sensitive action path higher precision” 的 mixed-precision design 很相容。

#### Why improvement is not guaranteed

- OpenVLA-OFT 已在 matched OpenVLA study 中发现 focused LIBERO demonstrations 上 L1 95.3% 与 50-step diffusion 95.4% 几乎相同；BitVLA average 已达 96.0%，LIBERO ceiling 很高。
- Flow model 可能学习 demonstration 中的 suboptimal modes；sampling “more diverse” 不等于 task success 更高。
- Iterative solver 增加 latency；若每一步重跑 3B BitVLA context，73 ms efficiency advantage 会消失。
- Flow inference 的 random sample 可能增加 action variance；若 control system 不允许 reranking，多 modality 反而会降低 repeatability。
- Quantization noise 若进入 iterative vector field，每一步 state update 都可能累积 error；action expert 往往比 large semantic backbone 更 precision-sensitive。

**Evaluation verdict**：这是值得做的 research hypothesis，但最可能在 **multimodal/contact-rich/recovery tasks** 上获益，而不是让 LIBERO average 自动再涨。若目标仍是 BitVLA 的 edge efficiency，直接复制 π₀ 的 300M BF16 expert + 10 steps 不是最佳第一版。

### 5.10 Recommended architecture: mixed-precision residual flow expert

我更推荐保留现有 L1 head，再加一个 compact residual flow expert，而不是直接替换：

```text
multi-view images + language + state
            │
            ▼
W1.58A8 SigLIP + ternary BitNet context encoder
            │
     ┌──────┴────────────┐
     ▼                   ▼
existing L1 head     compact flow expert
median chunk μ       noisy residual R^τ + flow time τ
     │                   │
     └────── A = μ + R ──┘
```

For clean action chunk `A`, let the current L1 head predict `μ_θ(o)` and define residual `R=A-μ_θ(o)`. Sample `ε~N(0,I)` and construct：

`R^τ = τR + (1-τ)ε`

Train an expert to predict：

`L_flow = E ||v_φ(R^τ, sg(C_o), μ, τ) - (R-ε)||²`

where `C_o` is BitVLA context and `sg` denotes stop-gradient during the first stage。

Why residual flow is a better BitVLA fit：

1. Existing 96% L1 policy remains a strong default/fallback；flow only models what deterministic head misses。
2. Residual dynamic range may be smaller，allowing a 30M–100M-class candidate expert and fewer solver steps to be tested before a π₀-scale 300M expert。
3. The backbone runs once；context K/V or projected context is cached，while 2–5 solver steps run only the compact expert。
4. When uncertainty is low，deployment can skip the expert and execute `μ`；when ambiguity/contact risk is high，invoke residual flow。

Sizes and step counts above are experiment candidates，not claims from BitVLA/π₀。

### 5.11 Training recipe combining π₀, π₀.₅ and π*₀.₆ lessons

**Stage A — preserve BitVLA pretraining**

- Keep the existing LLaVA-style multimodal stages、Quantize-then-Distill、and OXE action-token pretraining unchanged。
- This preserves the 1.4 GB backbone hypothesis and avoids rebuilding a new generalist VLA from scratch。

**Stage B — add a randomly initialized flow expert at post-training**

- Follow π₀.₅’s ordering：large-scale discrete knowledge acquisition first；continuous flow expert is introduced only for downstream continuous control。
- Start with BitVLA backbone and connector frozen or LoRA-only；train flow expert on clean action chunks with `L_flow`。
- Retain `L_L1` during warm-up：`L = L_flow + βL_L1`，preventing catastrophic degradation and keeping a deterministic path。

**Stage C — optional hybrid token + flow training**

- π₀.₅ shows why discrete FAST tokens are useful for heterogeneous pretraining；π*₀.₆ concurrently uses FAST discrete tokens and continuous flow actions。
- If more robotics pretraining is available，replace BitVLA’s naive per-dimension 256-bin stream with FAST，then optimize `L = L_FAST + αL_flow + βL_L1`。
- `L_FAST` trains backbone planning/semantic representations；flow expert handles deployment-oriented continuous chunks。This is a larger second experiment，not required for the first proof of concept。

**Stage D — Knowledge Insulation / gradient control**

- Following π*₀.₆’s Knowledge Insulation idea，initially apply `stop-gradient` from action expert to the low-bit backbone。
- Then compare：fully frozen backbone、LoRA on last BitNet blocks/connector、and unrestricted joint fine-tuning。
- If vision encoder is unfrozen，retain a small BF16-teacher alignment term to prevent W1.58A8 visual drift；otherwise Quantize-then-Distill guarantees may no longer hold after robot post-training。

### 5.12 Precision and systems design

- Keep bulk `SigLIP + BitNet` at W1.58A8 as BitVLA proposes。
- First train flow expert in BF16；then test FP8 or W8A8。Do not ternarize it first：timestep embeddings、normalization、action input/output projections and iterative vector field are likely sensitive。
- Cache visual/language/state context once per query。Latency should scale approximately as：

  `T_query ≈ T_low-bit context + N_solver · T_compact expert`

  not `N_solver · T_full BitVLA`。
- A π₀-scale 300M BF16 expert alone adds roughly 0.6 GB of weight storage；even before activations/runtime buffers，it would move the 1.4 GB headline toward about 2.0 GB。A compact expert or expert quantization is therefore central，not optional。
- After training a strong 5–10-step expert，a separate one-/two-step consistency or flow-distillation phase can recover latency；this is an extension beyond the cited π-series recipe and must be evaluated separately。

### 5.13 Minimum experiment to decide the idea

Hold fixed backbone checkpoint、data、images/state、chunk size、training steps and hardware，then compare：

1. BitVLA L1 baseline。
2. Full-action flow expert，10/5/2 solver steps。
3. Residual flow expert，10/5/2 steps。
4. Residual flow + stop-gradient vs LoRA vs joint update。
5. BF16 vs FP8/W8A8 expert；keep action projections/norms high precision as an ablation。
6. Optional flow-to-1/2-step distilled policy。

Metrics must include：

- closed-loop success/progress；perturbation recovery；language-conditioned target accuracy；
- trajectory jerk、contact failures、bimanual synchronization；
- diversity **and** success of sampled chunks，not diversity alone；
- P50/P95/P99 query latency、query rate、action throughput、actual replan interval；
- model memory、peak runtime memory、energy/action；
- success stratified by task multimodality/contact intensity and horizon。

A useful falsification criterion is：if flow only raises open-loop best-of-N action coverage but does not improve single-sample closed-loop success，or if gain disappears when replan rate is matched，then it is not a better controller for BitVLA deployment。

## 6. Experiments and main results

| Claim | Evidence | Locator | Caveat |
|---|---|---|---|
| 1.4 GB 模型接近 OpenVLA-OFT LIBERO accuracy | BitVLA 3.0B / **1.4 GB**：Spatial **96.6**, Object **99.0**, Goal **95.4**, Long **92.8**, avg **96.0%**；OpenVLA-OFT 7.7B / **15.4 GB**：avg **97.1%** | Table I, PDF p. 5 | headline 为 **11.0× smaller model memory / 1.1 points lower avg**；memory scope 未必含所有 runtime allocations |
| Native low-bit 优于强 baseline 的 PTQ efficiency trade-off | BitVLA 1.4 GB / 96.0%；INT4 OpenVLA-OFT 4.7 GB / 96.9%；INT4 OpenVLA 4.4 GB / 72.7% | Table II, PDF p. 5 | methods 训练历史不同；不是只改变 bit-width 的 controlled comparison |
| A100 inference latency 显著降低 | 100 queries，3×224² images + 14-D state，`K=25`：BitVLA **73 ms / 341.1 Hz**；OpenVLA-OFT+ **321 ms / 77.9 Hz**；π₀ **86 ms / 291.6 Hz** | Fig. 6 + text, PDF p. 7 | throughput ≠ control frequency；baseline numbers reported from OpenVLA-OFT，非全部同代码重测 |
| Distillation 对 low-bit vision encoder 关键 | BF16 encoder 0.8 GB / five-benchmark avg 53.0%；W1.58A8 0.1 GB / 51.5%；5B tokens with alignment 50.8%，without alignment 42.4% | Table III, PDF p. 8 | VQA average 不是 robot success；10B/5B data 也变化 |

Real-world section 只在 Franka 上评估 3 base tasks 与 OOD variants，主要以 bar plots 呈现；没有足够 table/interval 支撑把 “edge-ready general VLA” 当成已建立事实。

## 7. Limitations

### Authors' stated limitations

- quantization-aware training 会产生与 full-precision 不同的 parameter/activation distribution，因此不是任意 pretrained VLA 的 drop-in 1-bit conversion。
- robotics pretraining 只有约 1M samples；作者认为更广 task/environment/embodiment generalization 需要更大规模数据。
- 未来仍需 hardware-algorithm co-design，当前 arithmetic benefit 不等于所有 edge hardware 都有相同 wall-clock/energy gain。（Sec. VI, PDF p. 8）

### My critique

- **Internal validity**：native backbone、pretraining recipe、OFT head 与 kernel 同时变化，不能把全部 speed/accuracy trade-off 归因于 bit-width。
- **External validity**：real-world 只有 Franka + 3 base tasks/OOD variants；论文仍标 Work in progress。
- **Systems validity**：341.1 Hz 是 reported throughput/action accounting，不是 73 ms single-query latency 的倒数，也不是 control-loop rate；A100 不是典型 edge device。
- **Reproducibility**：需要 custom BitBLAS kernel；memory table 应进一步核对 weights、KV cache、activations、workspace 和 batch size 的计量范围。
- **PTQ baseline specification**：bitsandbytes 4-bit config 未报告，`INT4` label 不足以复现 FP4/NF4、compute dtype 与 module coverage。

## 8. Why it matters for this project

- 与 Professor Li 关注的 model compression、latency、edge deployment、software-hardware co-design 最直接对齐。
- 它提供一个重要 hypothesis：native low-bit pretraining/QAT 可能比 PTQ 更能保存 VLA performance。
- 它也提醒 project 不要只报 model size：必须同时报告 runtime memory、single-query latency、energy、control frequency 与 closed-loop success。
- 当前定位应是 **Critical scan**：适合提炼 research question，不宜作为已经成熟的 deployment recipe。

## 9. How to read it

### 20-minute route

1. Abstract + Fig. 2：区分 multimodal、Quantize-then-Distill、robotics 三阶段。
2. Sec. III-A/B（PDF pp. 3–5）：只追 W1.58A8、teacher/student frozen policy 与 action head。
3. Table I（p.5）+ Fig. 6（p.7）：核对 1.4 GB / 96.0% / 73 ms 的 measurement setup。
4. Sec. VI（p.8）：读 drop-in limitation、1M-sample scale、hardware caveat。

### 60-90-minute route

1. 复习 ternary quantization：absmean、activation absmax、STE 与 scaling overhead。
2. 画三阶段 training，逐模块标 BF16/W1.58/INT8 与 frozen/trainable。
3. 推导 `L_task + γL_aux` 如何使 student hidden features 对齐 BF16 teacher。
4. 对照 Tables I/II：区分 native low-bit 与 PTQ comparison，列出不 matched 的 variables。
5. 对照 Table III：为什么 VQA preservation 是 downstream VLA 的必要但非充分证据？
6. 解析 Fig. 6：分别定义 latency、throughput、chunk size、control loop；写出 edge-device 复现实验清单。
7. 补读 [LLaVA](../011-llava/README.md)、[OpenVLA-OFT](../012-openvla-oft/README.md)、[LLM.int8/bitsandbytes](../013-llm-int8-bitsandbytes/README.md) 与 [Apprentice](../010-apprentice-quantization-distillation/README.md)，把 inherited components 与 BitVLA contribution 分开。

## 10. Reading questions and answers

### Q1. LIBERO 96.0% 中多少来自 OpenVLA-OFT chunking，多少来自 native low-bit backbone？

Paper 不能给出可识别的 percentage decomposition。BitVLA、OpenVLA-OFT 与 PTQ baselines 的 backbone、pretraining、parameter count、mask、kernel 和 data history 都不同。当前最接近的 evidence 是：

- BitVLA without robotics pretraining 已有 94.8%，with pretraining 96.0%，说明 broad robotics pretraining 主要帮助 Long（87.6 → 92.8）。
- OpenVLA-OFT paper 在 OpenVLA 内部做过 matched ablation：original recipe 76.5 → PD+AC 90.2 → continuous L1 95.3/97.1；说明 chunking/adaptation recipe 本身贡献很大。
- BitVLA 没有报告同 architecture/data 的 BF16 backbone + same OFT head，也没有报告 low-bit backbone + autoregressive head。

真正能回答该问题的实验是 2×2：`BF16 vs W1.58A8 backbone` × `original decoding vs OFT-style chunking`，并固定 pretraining/data/parameterized head/kernel reporting。

### Q2. vision encoder 为什么需要 teacher alignment，而 ternary LLM 可以 native pretrained？

两者差别主要是 **training history**，不是 modality 的先验定律：

- BitNet b1.58 2B4T 从开始就以 native ternary dynamics pretrain；它没有经历 “把已训练 BF16 weights 突然 ternarize” 的 distribution shock。
- SigLIP-L originally learned in higher precision；BitVLA 从 full-precision checkpoint 初始化后才施加 W1.58A8 forward quantization。Without teacher，limited multimodal data 容易让 semantic geometry drift。
- Layerwise teacher MSE 比 answer-token CE 提供更密集 signal：即使 final answer 暂时正确，也要求每层 low-bit representation 贴近 BF16 teacher。

所以正确结论不是 “vision 必须 distill、language 不需要”，而是 `native low-bit pretraining` 与 `post-hoc quantization-aware conversion` 需要不同 stabilization。若 language backbone 也是从 full-precision checkpoint 直接 ternarize，同样很可能需要 QAT/KD。

### Q3. 1.4 GB 是否包含 KV cache、activations 与 runtime workspace？

Paper 只把它列作 `Memory Usage/model memory`，没有给出包括 KV cache、temporary activations、CUDA allocator、BitBLAS workspace、image tensors 和 framework overhead 的 peak-runtime protocol。因此不能把 1.4 GB 当作 end-to-end VRAM requirement。

从 quantity 也能看出它更接近 weight/model footprint：3.0B ternary parameters 的 ideal packed payload 约为 `3.0B×1.58/8 ≈ 0.59 GB`，再加 embeddings、scales、connector、action head 与 non-ideal packing，达到 1.4 GB 是合理的；runtime activations/workspace 则会随 image count、sequence length、batch 和 kernel 改变。部署报告应另外给 `peak allocated/reserved VRAM`。

### Q4. 341.1 Hz 如何由 73 ms 与 K=25 推得？它能否代表 closed-loop rate？

它是 amortized action throughput：

`25 actions / 0.073 s ≈ 342.5 actions/s`

Paper 的 341.1 差异来自未四舍五入的实际 average latency。一次 query 仍约 73 ms，所以 query rate 上限约：

`1 / 0.073 ≈ 13.7 queries/s`

如果 robot 以 25 Hz execute full `K=25` chunk，现实中每秒只 replan 一次；341.1 不是 control-loop refresh rate。它衡量 “一次 query 产出的 action 数除以 generation time”，必须与 latency、executed prefix、robot control frequency 一起报告。

### Q5. BitBLAS speedup 能否迁移到 RTX laptop、Jetson 或 NPU？

不能从 A100 result 自动推出。BitBLAS 的 published/tested support 主要是 NVIDIA CUDA GPUs，support matrix 包含 A100、A6000、V100、RTX 3090/4090 等；BitVLA 本文只在 A100/A800/H800-class setup 报告。迁移需要逐层满足：

1. target device 有 efficient packed ternary/INT8 matmul 或可生成的 equivalent kernel；
2. weight packing、scale application、Flash-Attention 与 memory layout 都被支持；
3. small-batch GEMV/GEMM shape 真能占满 device；
4. end-to-end camera preprocessing、transfer 与 control loop 不吞掉 gain。

RTX laptop 可能具备 CUDA compatibility，但 exact architecture/VRAM/kernel tuning 仍需 benchmark；Jetson 的 compute/memory balance 不同；NPU 通常需要 vendor-specific ternary primitive/compiler path，BitBLAS code 不能直接移植。最少要实测 P50/P99 latency、power 与 fallback operators。

### Q6. low-bit noise 会不会在 long-horizon closed loop 累计？

会，且 LIBERO average 接近 full precision 不能排除。Quantization error 可能造成：visual embedding drift、small systematic action bias、gripper threshold flips 与 chunk-level trajectory distortion；executing full chunk before replan 会放大这些误差。Long suite 92.8% 本身也低于其他 suites，但该差异同时受 task difficulty 与 data/pretraining 影响，不能单独归因 low bit。

应记录 error as a function of rollout time/action horizon，并比较：one-step action error、closed-loop state divergence、intervention/recovery rate、different replan prefixes，以及 BF16/W1.58A8 paired rollouts under identical initial states。若 offline L1 error 相近但 low-bit policy 在 later timesteps failure 增多，才是 compounding evidence。

## 11. Weekly meeting card

- **Problem**：怎样把 VLA 压到 memory-constrained hardware，同时维持 manipulation accuracy？
- **Key idea**：native ternary BitNet + teacher-guided W1.58A8 vision encoder + OFT-style parallel action chunks。
- **Best evidence**：1.4 GB / LIBERO 96.0% vs OpenVLA-OFT 15.4 GB / 97.1%（Table I, p.5）；A100 73 ms vs 321 ms（Fig.6, p.7）。
- **Biggest limitation**：A100 并非 edge，throughput 定义容易误读，real-world tasks 很少，且 paper 仍为 Work in progress。
- **Question for the group**：我们怎样设计一个同时报告 model memory、runtime memory、energy、latency 与 closed-loop success 的公平 VLA quantization benchmark？

## 12. Evidence boundary

- **Source claim**：architecture/training 来自 Sec. III–IV；numbers 来自 Tables I–III、Fig. 6；limitations 来自 Sec. VI。
- **My interpretation**：BitVLA 最值得读的是 native low-bit vs PTQ 这个 research framing，而不是直接相信“1-bit edge-ready”。
- **Open question**：真实 edge hardware 的 energy/latency、long-horizon error accumulation 与跨 embodiment generalization 尚未确认。
- **Primary links**：[arXiv](https://arxiv.org/abs/2506.07530) · [Repository](https://github.com/ustcwhy/BitVLA)

