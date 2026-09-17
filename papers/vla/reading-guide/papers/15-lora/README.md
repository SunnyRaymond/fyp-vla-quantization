# LoRA: Low-Rank Adaptation of Large Language Models

> **Reading-list role**: Core #15 - foundational Parameter-Efficient Fine-Tuning method; direct prerequisite for understanding OpenVLA adaptation
> **Verification**: `verified-full-text` - ICLR 2022 venue identity; local artifact is arXiv:2106.09685 v2
> **Recommended effort**: Core read; complete the 20-minute route before revisiting OpenVLA Table 1

> **Local full text**: [paper-arxiv-v2.pdf](paper-arxiv-v2.pdf)

## 1. Paper identity

| Field | Value |
|---|---|
| Authors | Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen |
| Year / version | arXiv:2106.09685 v2, 2021-10-16; conference publication in 2022 |
| Venue / status | ICLR 2022, peer-reviewed conference paper |
| Primary sources | [OpenReview record](https://openreview.net/forum?id=nZeVKeeFYf9) · [Microsoft Research record](https://www.microsoft.com/en-us/research/publication/lora-low-rank-adaptation-of-large-language-models/) · [arXiv](https://arxiv.org/abs/2106.09685) |
| Code | [official `microsoft/LoRA` repository](https://github.com/microsoft/LoRA) |
| Local artifact | arXiv v2, 26 PDF pages, SHA-256 `e9a0d3128767db616085dc0f4e6e455e672e89af823e8ed1282793682787395a` |

**Version boundary.** ICLR 2022 is the canonical publication identity. OpenReview returned a browser-verification challenge during unattended acquisition, so the stable local reading copy is arXiv v2 rather than an OpenReview binary. arXiv records only v1 and v2; v2 adds stronger baselines, GLUE experiments, and more adapter-latency evidence. The arXiv-issued DOI `10.48550/arXiv.2106.09685` identifies the preprint record, not a separate conference paper.

## 2. One-sentence takeaway

LoRA freezes a pretrained weight matrix $W_0$ and learns a low-rank update $\Delta W=BA$, cutting trainable parameters, optimizer-state memory, and per-task checkpoint storage while allowing $BA$ to be merged into $W_0$ for single-adapter inference with no extra model depth.

## 3. Background and prerequisites

- **Full fine-tuning**: every downstream task updates and stores a full copy of the pretrained model.
- **Parameter-Efficient Fine-Tuning (PEFT)**: adapt a much smaller task-specific parameter set while sharing a frozen base model.
- **Matrix rank and factorization**: if $\Delta W\in\mathbb{R}^{d\times k}$ has effective rank $r\ll\min(d,k)$, it can be represented by $B\in\mathbb{R}^{d\times r}$ and $A\in\mathbb{R}^{r\times k}$.
- **Transformer projections**: know $W_q,W_k,W_v,W_o$ in self-attention and the two MLP projections.
- **Adam memory**: full fine-tuning normally stores gradients and optimizer states for every trainable parameter; freezing $W_0$ removes those states for the base weights but does not remove the base weights or all activation memory.
- **Prior PEFT routes**: adapter layers add sequential modules; Prefix Tuning learns prompt-like activations and consumes part of the usable sequence/context interface.
- **VLA bridge**: OpenVLA later uses LoRA for robot-policy adaptation, but changes the target modules and rank; do not copy the original GPT-3 recipe without re-evaluation.

## 4. Problem

- **Target setting**: adapt one very large pretrained model to many downstream tasks.
- **Training bottleneck**: full fine-tuning needs gradients and optimizer states for the full model, raising the hardware barrier.
- **Storage/deployment bottleneck**: each task would otherwise require another full-size checkpoint.
- **Latency bottleneck in prior PEFT**: sequential adapter layers add extra operations and synchronization; prompt-based methods reserve tokens/activations and can be difficult to optimize.
- **Hypothesis**: the task-specific weight update has low intrinsic rank even when the pretrained matrix itself is full-rank.

The paper is not primarily a compression method for the shared base model. It compresses the **trainable update and per-task delta**. The frozen base checkpoint is still required for training and inference.

## 5. Method

### 5.1 Reparameterize the update, not the pretrained weight

For a pretrained matrix $W_0\in\mathbb{R}^{d\times k}$, full fine-tuning would learn a dense update $\Delta W$ of the same shape. LoRA constrains the update to:

$$
W_0+\Delta W=W_0+BA,
$$

where:

$$
B\in\mathbb{R}^{d\times r},\qquad A\in\mathbb{R}^{r\times k},\qquad r\ll\min(d,k).
$$

The forward path becomes:

$$
h=W_0x+\frac{\alpha}{r}BAx.
$$

- $W_0$ stays frozen.
- $A$ and $B$ are trainable.
- $A$ is initialized with a random Gaussian matrix; $B$ is initialized to zero, so $\Delta W=0$ at the start.
- $\alpha/r$ stabilizes the scale when changing rank $r$; the paper treats $\alpha$ largely like a learning-rate-related hyperparameter.

The trainable parameter count for one adapted $d\times k$ matrix changes from $dk$ to $r(d+k)$. Low rank therefore helps only when $r(d+k)\ll dk$.

### 5.2 Apply LoRA to Transformer projections

The paper studies LoRA mainly on attention weights and freezes the MLP modules. Most experiments adapt $W_q$ and $W_v$ across Transformer layers. This is an empirical design choice, not a theorem that every architecture should use only `q_proj` and `v_proj`.

The system flow is:

`load frozen pretrained model → attach low-rank A/B branches → train only A/B → save small task delta → merge BA into W0 for deployment`

### 5.3 Merge for inference

For a single selected task adapter, compute:

$$
W=W_0+\frac{\alpha}{r}BA.
$$

Inference then uses the same dense layer shape and depth as the original model. The paper's “no additional inference latency” claim is therefore a construction-level statement for a **merged adapter**. If a system keeps multiple adapters unmerged, routes adapters per sample, or mixes tasks within one batch, additional kernels, memory traffic, or batching constraints can reappear.

### 5.4 What is actually new

The novelty is not low-rank matrix factorization by itself. The new move is to represent the **downstream weight update of a frozen pretrained model** as trainable low-rank factors, then exploit the linear form for cheap task storage and mergeable inference.

## 6. Experiments and main results

| Claim | Evidence | Locator | Caveat |
|---|---|---|---|
| LoRA can match full fine-tuning on GLUE with far fewer trainable parameters | RoBERTa base: LoRA **0.3M / 87.2 average** vs full fine-tuning **125M / 86.4**; DeBERTa XXL: LoRA **4.7M / 91.3** vs full fine-tuning **1.5B / 91.1** | Table 2, PDF p. 6 | Some baselines are reported from prior work and marked `*`; setups are not uniformly author-rerun. |
| LoRA remains strong for GPT-2 generation | GPT-2 Medium E2E BLEU: LoRA **70.4** with **0.35M** trainable parameters vs full fine-tuning **68.2** with **354.92M** | Table 3, PDF p. 7 | Dataset-specific NLG result; not a general language-quality guarantee. |
| LoRA scales to the paper's GPT-3 175B setting | With **4.7M** trainable parameters: WikiSQL **73.4**, MultiNLI **91.7**, SAMSum ROUGE **53.8/29.8/45.9**; full fine-tuning uses **175,255.8M** and reports **73.8/89.5/52.0/28.0/44.5** | Table 4, PDF p. 8 | GPT-3 model/data/training system is closed; the released repository does not enable independent reproduction of this experiment. |
| Target distribution can matter more than assigning a larger rank to one matrix | Under an 18M budget, $W_q+W_v$ at $r=4$ gives WikiSQL **73.7**, versus $W_q$ at $r=8$ **70.4** and $W_v$ at $r=8$ **73.0** | Table 5, PDF p. 10 | MultiNLI's best row in the table is all four attention matrices at $r=2$; there is no universal `q+v` winner. |
| Very small rank can suffice in these tasks | For $W_q+W_v$, WikiSQL/MultiNLI at $r=1$ is **73.4/91.3**, compared with **73.8/91.6** at $r=8$ | Table 6, PDF p. 10 | The authors explicitly say small $r$ should not be expected to work for every task or distribution shift. |
| LoRA reduces GPT-3 training/storage overhead in the reported setup | Training memory **1.2TB → 350GB**; task checkpoint **350GB → 35MB** for $r=4$ on query/value; throughput **32.5 → 43.1 tokens/s per V100** | Sec. 4.2 and footnotes, PDF p. 5 | The shared 350GB base model is still required at deployment; this is not a 10,000x reduction of total serving memory. |
| Sequential adapters can add latency while merged LoRA does not add depth | Batch 1, sequence 128: Fine-Tune/LoRA **19.8 ms**, AdapterL **23.9 ms (+20.7%)**, AdapterH **25.8 ms (+30.3%)** | Table 1, PDF p. 4 | GPT-2 Medium on one RTX8000; modern fused adapter runtimes and other batch shapes may differ. |

## 7. Understanding the rank evidence

The paper does more than report task scores. It compares the learned subspaces of $\Delta W$ across ranks and random seeds:

- Figure 3 shows that the strongest singular direction overlaps between $r=8$ and $r=64$ runs.
- Table 7 argues that $\Delta W$ correlates with $W$ more than a random matrix, but amplifies directions not already dominant in $W$.
- The authors interpret this as evidence that task adaptation can use a small set of task-specific directions.

This is **empirical evidence for the tested GPT-3 layers/tasks**, not proof that all fine-tuning updates are low-rank. The factorized parameterization also changes optimization, so a good low-rank solution does not by itself establish the rank of an unconstrained full-fine-tuning update.

## 8. Limitations

### Authors' stated limitations

- Merged adapters make it non-trivial to batch samples from different tasks using different $A/B$ pairs in one forward pass.
- Experiments focus on attention projections; MLP, LayerNorm, and bias adaptation are left largely unexplored.
- Weight-matrix selection is heuristic.
- The best rank depends on task/model; the paper does not claim that $r=1$ or $r=4$ is universal.
- The mechanism behind the low-rank update remains only partially understood.

### My critique

- **Model-era boundary**: evidence centers on RoBERTa, DeBERTa, GPT-2, and closed GPT-3. It predates Llama-family GQA, Mixture-of-Experts, modern long-context systems, diffusion/flow action experts, and current VLA architectures.
- **Memory wording**: freezing weights removes their gradients and optimizer states, but activations, temporary buffers, the frozen base weights, and any trainable vision/action modules still consume memory. “3x lower GPU memory” is setup-specific.
- **Serving boundary**: no-extra-latency requires merged weights and one active adapter configuration. Dynamic multi-tenant routing or per-sample adapter selection changes the systems problem.
- **Baseline boundary**: several comparison numbers are copied from prior work; not all rows share identical code, hyperparameter search, or random-seed treatment.
- **Reproducibility boundary**: the strongest scale claim uses GPT-3 175B and Microsoft infrastructure that is not publicly reproducible from the released code.
- **Task boundary**: NLP benchmarks measure classification/generation quality, not visual grounding, action prediction, closed-loop stability, control frequency, or safety.

## 9. Direct bridge to OpenVLA and this project

| Axis | Original LoRA | OpenVLA adaptation |
|---|---|---|
| Base model | RoBERTa / DeBERTa / GPT-2 / GPT-3 | 7B VLA with fused visual encoders, Llama-2 backbone, and action-token output |
| Typical target in paper | mainly $W_q,W_v$ | rank-32 LoRA on all Linear layers in the reported downstream setup |
| Parameter evidence | GPT-3: 4.7M or 37.7M trainable parameters depending on configuration | 97.6M trainable parameters, **1.4%** of model parameters |
| Task evidence | GLUE and text generation | robot-policy success on downstream Franka tasks |
| Critical mismatch | text-only conditional prediction | visual inputs, embodiment shift, temporal trajectories, and closed-loop control |

For the Final Year Project, LoRA deserves its own core number because it separates three costs that are often conflated:

1. **Training-state cost**: gradients and optimizer states for trainable parameters.
2. **Task-specific storage cost**: the adapter/checkpoint saved per task.
3. **Base-model inference cost**: frozen weights, activations, KV/cache-like state, latency, and power during serving.

LoRA directly reduces the first two. It does **not** automatically reduce the third. That is why LoRA and quantization are complementary rather than interchangeable.

A useful VLA experiment should compare, under matched data and hardware:

- full fine-tuning;
- LoRA on only the language backbone;
- LoRA on the visual encoder/projector;
- LoRA on the action expert/head;
- component-wise rank allocation;
- LoRA plus a fixed quantized base model.

Report `closed-loop success/progress + action error + P50/P99 latency + control frequency + peak memory + power/energy`, not just trainable parameter count.

## 10. What this paper does not cover

- **QLoRA / NF4 / paged optimizers**: these are later developments and are not part of the original LoRA method.
- **Modern PEFT libraries**: current Hugging Face PEFT behavior, target-module defaults, quantized base loading, and adapter routing are implementation surfaces outside this paper.
- **Diffusion or flow-policy adaptation**: no evidence is provided for action denoisers, flow-matching experts, or temporal control loops.
- **Universal rank selection**: the paper gives empirical rank sweeps, not an automatic rank-allocation algorithm.
- **Base-model compression**: a 35MB task delta still requires the full base model.

## 11. How to read it

### 20-minute route

1. **3 min** - Abstract + Figure 1 (PDF p. 1): explain what is frozen and what is trained.
2. **5 min** - Section 4.1 (p. 4): reconstruct $W_0+BA$ and the $\alpha/r$ scaling.
3. **3 min** - Section 4.2 (p. 5): separate training memory, task checkpoint storage, and serving memory.
4. **5 min** - Tables 4-6 (pp. 8-10): inspect GPT-3 results, target matrices, and rank.
5. **2 min** - Authors' limitations (p. 5) and Conclusion (pp. 12-13).
6. **2 min** - Reopen OpenVLA Table 1 and write down which LoRA assumptions changed.

### 90-minute route

1. **0-12 min** - Review SVD, matrix rank, full fine-tuning, adapter layers, and Prefix Tuning.
2. **12-25 min** - Derive the parameter count $r(d+k)$ and calculate the reduction for one $d\times d$ projection.
3. **25-38 min** - Trace initialization, scaling, optimizer state, merge, unmerge, and task switching.
4. **38-52 min** - Audit Tables 1-4: identify which baselines are author-run and which are imported.
5. **52-65 min** - Audit Tables 5-6: distinguish target-module choice from rank choice.
6. **65-76 min** - Read Figure 3 and Table 7: separate empirical subspace evidence from causal explanation.
7. **76-84 min** - Read Appendix D for hyperparameters and Appendix B for adapter latency.
8. **84-90 min** - Design one matched OpenVLA LoRA ablation with component-specific targets and deployment metrics.

## 12. Reading questions - answer these yourself

1. Why does LoRA parameterize the update $\Delta W$ instead of factorizing the pretrained weight $W_0$?
2. Why does initializing $B=0$ preserve the pretrained model's function at step zero?
3. Under what exact deployment condition is “no additional inference latency” valid, and when can it fail?
4. Table 5 changes both target matrices and rank under a fixed parameter budget. What can and cannot be concluded from it?
5. Why does Table 6 support low-rank sufficiency for the tested tasks without proving a universal intrinsic rank?
6. Which parts of the reported 10,000x reduction apply to task-specific storage, and which parts do not apply to total serving memory?
7. Why might OpenVLA use rank 32 on all Linear layers when the original LoRA paper often uses much smaller ranks on $W_q/W_v$?
8. For a VLA, which component should receive the first LoRA budget: visual encoder, language backbone, projector, or action expert? What evidence would decide?
9. If the base model is quantized, should LoRA be trained before quantization, after quantization, or jointly? Which confounds must be controlled?

## 13. Weekly meeting card

- **Problem**: Explain why full fine-tuning becomes a multi-task storage and training-state problem.
- **Key idea**: Draw $W_0x+(\alpha/r)BAx$ and label frozen/trainable paths.
- **Best evidence**: Choose one quality table and one systems number; keep their evaluation settings attached.
- **Biggest limitation**: State which original-paper assumption breaks most clearly in a modern VLA.
- **Question for the group**: Propose one component-wise LoRA allocation experiment with a matched full-fine-tuning baseline.

## 14. Evidence boundary

- **Source claim**: method equations, initialization, target-module experiments, rank sweeps, benchmark results, and GPT-3 memory/storage/throughput numbers come from the local arXiv v2 full text.
- **Cross-paper evidence**: the OpenVLA bridge uses its local full text and Table 1; it is not claimed by the original LoRA paper.
- **My interpretation**: the three-cost decomposition and VLA experiment design are project-oriented synthesis.
- **Open question**: component-specific rank, quantized-base interaction, and closed-loop VLA behavior require new experiments.
- `verified-full-text` means the paper was checked against the local artifact; it does not mean independent experimental replication.
