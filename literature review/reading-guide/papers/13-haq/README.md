# HAQ: Hardware-Aware Automated Quantization With Mixed Precision

> **Reading-list role**: Core lineage — hardware-aware mixed-precision search / software-hardware co-design  
> **Verification**: `verified-full-text` — CVPR 2019 official CVF final  
> **Recommended effort**: Core read；方法思想比具体 CNN bit policy 更重要

> **Local full text**: [paper.pdf](paper.pdf)

## 1. Paper identity

| Field | Value |
|---|---|
| Authors | Kuan Wang, Zhijian Liu, Yujun Lin, Ji Lin, Song Han |
| Year / version | 2019; arXiv:1811.08886 |
| Venue / status | CVPR 2019, pp. 8612–8620; peer-reviewed venue final |
| Primary source | [CVF record](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_HAQ_Hardware-Aware_Automated_Quantization_With_Mixed_Precision_CVPR_2019_paper.html) · [official PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_HAQ_Hardware-Aware_Automated_Quantization_With_Mixed_Precision_CVPR_2019_paper.pdf) · [arXiv](https://arxiv.org/abs/1811.08886) |
| Code / project | [official repository](https://github.com/mit-han-lab/haq) |

## 2. One-sentence takeaway

HAQ 用 DDPG 为每层分别选择 weight/activation precision，并以 target hardware simulator 的 latency 或 energy 作为 direct feedback 而不是 FLOPs proxy；在 BISMO cloud 上将 MobileNet-V1 latency 从 151.09 ms 降至 77.49 ms，top-1 从 70.82 变为 69.97。

## 3. Background and prerequisites

- Mixed-precision quantization 与 layer sensitivity：不同 layers 不应默认相同 bits。
- Hard resource constraint 与 Pareto frontier：accuracy、latency、energy、model size 的 trade-off。
- Markov decision process、actor-critic、DDPG、continuous action、experience replay/Bellman update。
- Hardware accelerator basics：compute parallelism、memory traffic、depthwise convolution；理解 FLOPs 与 measured latency 不等价。
- KL-divergence-based clipping 与 per-layer weight/activation quantization。

## 4. Problem

- **Target setting**：给定 network、target accelerator 和 resource budget，自动确定每层 weight/activation bitwidth。
- **Bottleneck**：uniform precision 浪费 insensitive layers 的 bits；FLOPs/model-size proxy 无法反映真实 latency/energy。
- **Why previous methods are insufficient**：手工 policy 搜索空间大且 device-specific；只优化 accuracy/size 的 policy 可能在 edge/cloud accelerator 上表现完全不同。

## 5. Method

### 5.1 System view

`network layers → encode layer/hardware-relevant state → DDPG actor outputs continuous action → map to 2–8 bit integer → quantize/fine-tune one epoch → hardware simulator checks constraint and reports latency/energy → accuracy reward updates actor/critic → final full-data fine-tune`

每层做两次 decision：weight precision 与 activation precision。如果完整 policy 超 budget，algorithm 按顺序降低 bitwidth 直到满足 constraint。

### 5.2 Core mechanism

Conv state，Section 3.1, Eq. (1)：

$$
O_k=(k,c_{in},c_{out},s_{kernel},s_{stride},s_{feat},n_{params},i_{dw},i_{w/a},a_{k-1}),
$$

FC layer 用 Eq. (2) analog。Actor 输出 $a_k\in[0,1]$，Eq. (3) 映射到 bits：

$$
b_k=\operatorname{round}\left(b_{min}-0.5+a_k(b_{max}-b_{min}+1)\right),
$$

$b_{min}=2,b_{max}=8$。Quantizer，Eq. (4)：

$$
q(w,a_k,c)=\operatorname{round}(\operatorname{clamp}(w,c)/s)s,
\qquad s=\frac{c}{2^{a_k-1}-1}.
$$

$c$ 用 Eq. (5) 的 KL-divergence minimization 选；activations 用 $[0,c]$。Constraint 已由 simulator/repair 强制，所以 reward 只用 accuracy：

$$
R=\lambda(acc_{quant}-acc_{origin}),\qquad\lambda=0.1.
$$

Search 在 ImageNet-100 subset，每 episode one-epoch fine-tune；actor/critic hidden sizes 400/300，exploration $\sigma=0.5$ 每 episode ×0.99；final full ImageNet fine-tune。

### 5.3 What is actually new

关键创新是 `hardware-in-the-loop objective`：让 target simulator 的 latency/energy 决定 feasible policy，因而 edge 与 cloud 学出不同 precision allocation。DDPG、linear quantizer、KL clipping 都是已有 components；创新在 formulation 与 direct hardware feedback 的组合。

## 6. Experiments and main results

| Claim | Evidence | Locator | Caveat |
|---|---|---|---|
| Edge/cloud policy 不同且有效 | BISMO MobileNet-V1 original 8-bit：top-1 **70.82**, edge/cloud **96.20/151.09 ms**。HAQ edge **70.58/57.70** (1.67×)；cloud **69.97/77.49** (1.95×)。 | Table 3, PDF p. 6 | Feedback 来自 simulator，非直接 device measurement。 |
| BitFusion latency | Original 8/8 **70.82/20.08 ms**；HAQ flexible **70.40/11.09** (~1.81×)，或 **70.90/19.98**。 | Table 4, PDF p. 7 | 同一 table 有不同 accuracy/latency operating points。 |
| BitFusion energy | Original **70.82/31.03 mJ**；HAQ **70.37/16.30** (~1.90×)，或 **70.90/26.67**。 | Table 5, PDF p. 7 | Prose 的 16.57 对应 PACT row；HAQ exact row 是 16.30，`2×` 为 rounded claim。 |
| Tight model size | ~2-bit：MobileNet-V1 Deep Compression **37.62/1.09 MB**, HAQ **57.14/1.09**；MobileNet-V2 **58.07/0.96** vs **66.75/0.95**；ResNet-50 **68.95/6.32** vs **70.63/6.30**。 | Table 6, PDF p. 8 | 2019 CNN/ImageNet setting，不能直接外推 Transformer。 |

## 7. Limitations

### Authors' stated limitations

- 没有独立 `Limitations` section。Paper 的 formulation 明确显示 policy 是 hardware-specific；换 accelerator 需要重新搜索。

### My critique

- **Internal validity**：超 budget 后 sequentially lower bits 是 heuristic repair，不保证 global optimum；论文对 RL seed variance/confidence interval 报告不足。
- **External validity**：只覆盖 CNN/ImageNet，没有 Transformer、VLM、KV cache 或 closed-loop VLA。
- **Systems validity**：所谓 direct hardware feedback 实际由 simulator 返回，结果依赖 simulator fidelity；static budget 也不表达 thermal/battery/P99 dynamics。
- **Reproducibility**：反复 one-epoch fine-tuning 的 total search compute 未充分量化；policy 依赖 target simulator 和 search subset。

## 8. Why it matters for this project

- **VLA**：其价值是 formulation，而非复用 MobileNet bits。可为 vision encoder、multimodal projector、language/action decoder、KV cache、control head 分别搜索 precision。
- **Needed extension**：state 要加入 attention/FFN、context length、KV precision、batch/token phase、modality；reward 应加入 task success、P99 latency、energy 和 thermal constraints。
- **Professor Li's direction**：HAQ 是 model compression 与 actual hardware cost 联合优化的经典工作，直接体现 edge deployment 和 software-hardware co-design。

## 9. How to read it

### 20-minute route

1. Abstract + Figure 1，先写下“FLOPs ≠ latency”。
2. 读 state Eq. (1) 与 bit mapping Eq. (3)。
3. 读 reward Eq. (6)，解释为什么 constraint 不写进 reward。
4. 核对 Table 3 的 edge/cloud MobileNet-V1 rows。
5. 读 Table 5，注意 16.30 vs prose 16.57 discrepancy。

### 90-minute route

1. **0–15 min**：补 mixed precision、DDPG、hardware accelerator prerequisites。
2. **15–32 min**：把 layer-by-layer decision 写成 MDP：state/action/reward/transition。
3. **32–48 min**：推导 action-to-bit mapping、quantizer/clipping、constraint repair。
4. **48–60 min**：跟踪一个 episode：policy→quantize→fine-tune→simulate→reward→update。
5. **60–75 min**：核对 Tables 3–6，分别标注 latency、energy、model-size evidence。
6. **75–83 min**：审查 simulator fidelity、search cost、variance 和 heuristic repair。
7. **83–90 min**：为 VLA 设计扩展 state/action/reward，并列出 target robot measurements。

## 10. Reading questions

1. 为什么 FLOPs 不能可靠预测 accelerator latency？
2. State 中哪些 features 最可能导致 edge 与 cloud policy 分化？
3. 为什么 constraint 已强制后只用 accuracy reward；alternative constrained-RL formulation 会怎样？
4. Sequential budget repair 会错过哪类 globally better allocation？
5. Simulator mismatch 会如何改变 learned policy 的 ranking？
6. VLA reward 应如何组合 success rate、P99 latency、energy 与 thermal state？

## 11. Weekly meeting card

- **Problem**：layer-wise bits 搜索需要同时满足 accuracy 与真实 hardware budget，FLOPs proxy 不可靠。
- **Key idea**：DDPG 用 target simulator 的 latency/energy feedback 分配每层 W/A bits。
- **Best evidence**：MobileNet-V1 cloud 151.09→77.49 ms，top-1 70.82→69.97；edge 得到另一 policy。
- **Biggest limitation**：CNN-only、simulator-dependent、search cost/variance 报告不足，无法直接复用到 Transformer/VLA。
- **Question for the group**：能否定义 `module × modality × context × hardware state` 的 VLA mixed-precision controller？

## 12. Status & evidence boundary

- **Status**：peer-reviewed CVPR 2019 venue final；arXiv 1811.08886 与 CVF record 是同一 work cluster。
- **Source claim**：Eqs. (1)–(6)、DDPG/search configuration、Tables 3–6；Table 5 discrepancy 由原表与 prose 对照得到。
- **My interpretation**：将 state/reward 扩展到 Transformer/VLA，并对 robot SoC 做 runtime-adaptive precision。
- **Open question**：真实 hardware measurements、search variance、closed-loop VLA success/latency/energy frontier。
