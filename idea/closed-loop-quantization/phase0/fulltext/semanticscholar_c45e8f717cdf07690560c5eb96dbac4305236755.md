# MARR: Module-Adaptive Residual Reconstruction for Low-Bit Post-Training Quantization

paper_id: semanticscholar:c45e8f717cdf07690560c5eb96dbac4305236755
tier: T2
source_used: pdf_arxiv_pymupdf
warning: none

## Intro

Large language models (LLMs) [36, 41] and vision transformers (ViTs) [23, 9] have achieved re-
markable progress across diverse tasks. However, their large model sizes and dense operations incur
substantial memory, bandwidth, and inference overhead. To improve deployment efficiency, existing
studies either compress pre-trained models through pruning [34, 6], knowledge distillation [45, 13],
∗Corresponding author.
Preprint.
arXiv:2605.17997v1  [cs.LG]  18 May 2026

low-rank decomposition [42], and quantization [12, 46], or directly design lightweight architec-
tures [21] and efficient operators [29, 15]. Among them, model quantization is deployment-friendly
since it reduces the numerical precision of model while largely preserving the original architecture.
Therefore, it is a plug-and-play compression technique compatible with existing inference pipelines
and hardware kernels [11]. Moreover, pushing quantization below conventional 8-bit settings is
increasingly important, as low-bit quantization can further reduce memory and bandwidth costs,
allowing for larger models, longer sequences, or higher throughput under the same hardware bud-
get [12]. Therefore, how to push models to lower bit-widths while preserving their performance
remains a valuable and challenging problem.
Existing quantization methods are broadly divided into quantization-aware training (QAT) [24] and
post-training quantization (PTQ) [18, 37]. Compared with QAT, PTQ avoids additional training or
finetuning, making it more practical for extremely large pre-trained models. Current PTQ methods can
be roughly grouped into quantization parameter-based methods [47], redistribution-based methods [40,
3], and reconstruction-based methods [11, 19]. Among them, reconstruction-based methods have
attracted increasing attention for large models, since they efficiently reduce quantization error by
adjusting weights while maintaining low quantization overhead through closed-form updates. A
representative basic reconstruction method is GPTQ [11], which extends Optimal Brain Surgeon
(OBS) [14] and Optimal Brain Quantization (OBQ) [10] by applying Hessian-based column-wise
updates for each layer. However, under low-bit quantization, quantization errors from preceding
layers accumulate across the network, making local-only basic reconstruction insufficient. To address
this issue, recent residual reconstruction-based methods (GPTAQ [19] and ResComp [17]) account
for inter-layer activation dependencies by introducing a residual between full-precision (FP) and
quantized (Quant) activation, which improves low-bit quantization performance.
This cross-layer residual is beneficial for reducing accumulated quantization errors; however, it
may also introduce additional Hessian-approximation (HA) bias. Such a two-sided effect makes the
overall reconstruction suboptimal when overly strong residual correction is applied. In this work, to
address this issue, we introduce a scaling coefficient to modulate the residual term, as the HA bias
is directly affected by the strength of the residual correction. By doing so, residual reconstruction
can preserve effective cross-layer error compensation while reducing the adverse effect of excessive
residual strength. Moreover, the desired modulation is inherently module-dependent: the same
residual strength can lead to different reconstruction behaviors across modules. This module-wise het-
erogeneity makes a single global coefficient insufficient. Accordingly, we propose Module-Adaptive
Residual Reconstruction (MARR), which assigns a module-specific scaling coefficient to the residual
term of each module. By directly scaling the residual, this coefficient explicitly controls how much
cross-layer information is injected into each module, balancing accumulated-error correction against
HA bias. In terms of coefficient estimation efficiency, we formulate residual-strength control as a
closed-loop feedback problem, where module-level reconstruction error serves as the observable
feedback signal for a Proportional-Integral-Derivative (PID)-based adaptive update, avoiding ex-
haustive coefficient search. Figure 1 provides a conceptual comparison among basic reconstruction,
residual reconstruction, and MARR, together with representative performance improvements under
2-bit weight and 4-bit activation quantization (W2A4).
Our contributions are summarized as follows:
• We propose Module-Adaptive Residual Reconstruction (MARR), a new framework to
address the suboptimal performance of residual reconstruction-based methods under low-bit
PTQ.
• We introduce a module-specific scaling coefficient to balance the two-sided effect of residu-
als, and design a PID-based adaptive update strategy to efficiently estimate the coefficient
using reconstruction-error feedback.
• Extensive experiments on LLMs and ViTs demonstrate that compared with residual recon-
struction baselines, MARR achieves consistent improvements under low-bit quantization.
2

## Method

W/A
Method
Wiki2 ↓
C4 ↓
PiQA
ARC-E
ARC-C
HellaSwag
WinoGrande
BoolQ
Avg ↑
Llama2-7b
FP
5.47
6.90
79.00
74.60
46.50
76.00
68.90
77.70
70.50
W4A4
RTN
7.99
11.58
74.16
62.84
36.35
66.24
61.17
69.79
61.76
GPTQ
6.01
8.16
77.69
71.76
42.49
73.17
65.04
75.32
67.58
ResComp
5.88
8.04
77.86
70.46
42.69
73.72
65.23
73.06
67.17
GPTAQ
5.86
7.98
76.50
70.37
41.98
73.52
65.90
73.64
66.98
ResComp-MARR
5.84 (↓0.68%)
8.03 (↓0.12%)
76.88
69.91
42.24
73.70
66.06
74.71
67.25 (↑0.11%)
GPTAQ-MARR
5.83 (↓0.51%)
7.92 (↓0.75%)
77.04
71.46
43.77
73.57
67.96
73.85
67.94 (↑1.43%)
W2A4
RTN
7.7e3
9.0e3
50.76
25.59
28.58
26.38
48.54
37.83
36.28
GPTQ
36.37
62.87
56.80
33.38
21.76
31.77
52.33
49.79
40.97
ResComp
11.00
23.13
60.79
44.17
25.02
41.79
53.41
62.06
47.87
GPTAQ
11.24
19.47
62.13
44.15
25.68
42.70
52.64
58.59
47.65
ResComp-MARR
9.92 (↓9.82%)
21.85 (↓5.53%)
60.94
43.56
25.17
42.81
55.49
62.05
48.34 (↑0.98%)
GPTAQ-MARR
9.56 (↓14.95%)
15.53 (↓20.24%)
61.48
43.90
25.26
43.84
57.38
61.31
48.86 (↑2.54%)
Llama2-13b
FP
4.88
6.41
80.50
77.50
49.20
79.40
72.40
80.60
73.30
W4A4
RTN
5.93
8.34
77.37
74.03
44.80
73.73
68.82
76.09
69.14
GPTQ
5.29
7.37
78.24
74.83
46.67
76.90
69.53
79.14
70.88
ResComp
5.19
7.28
78.67
74.45
47.87
77.10
70.32
80.03
71.41
GPTAQ
5.18
7.22
78.24
73.91
46.84
77.10
69.46
78.23
70.63
ResComp-MARR
5.16 (↓0.58%)
7.26 (↓0.27%)
79.38
75.17
47.78
77.11
70.96
78.35
71.46 (↑0.07%)
GPTAQ-MARR
5.14 (↓0.77%)
7.17 (↓0.69%)
78.73
73.99
46.84
77.27
70.48
77.46
70.79 (↑0.23%)
W2A4
RTN
5.6e3
5.5e3
48.59
27.19
27.13
25.02
51.78
37.83
36.26
GPTQ
13.55
30.81
60.94
41.29
24.49
41.35
53.35
62.05
47.24
ResComp
8.31
17.90
63.76
51.76
26.55
47.65
54.22
60.29
50.70
GPTAQ
8.62
15.61
67.03
54.83
31.08
48.54
56.31
62.01
53.30
ResComp-MARR
7.97 (↓4.09%)
17.61 (↓1.62%)
64.74
48.27
28.75
48.67
57.62
62.91
51.83 (↑2.23%)
GPTAQ-MARR
7.80 (↓9.51%)
15.18 (↓2.75%)
66.38
51.39
30.38
51.93
59.04
64.16
53.88 (↑1.09%)
Llama3-8b
FP
6.44
9.61
80.70
77.70
53.70
79.10
73.20
81.10
74.30
W4A4
RTN
9.92
15.85
74.10
65.28
40.02
70.26
65.19
74.40
64.88
GPTQ
7.77
12.54
75.73
72.18
44.28
73.81
66.69
76.15
68.14
ResComp
7.42
12.39
78.02
72.22
47.44
74.72
67.64
76.79
69.47
GPTAQ
7.36
12.22
78.24
71.80
46.76
74.94
67.88
77.86
69.58
ResComp-MARR
7.36 (↓0.81%)
12.35 (↓0.32%)
78.18
74.62
46.16
74.97
67.40
76.70
69.67 (↑0.29%)
GPTAQ-MARR
7.31 (↓0.68%)
12.15 (↓0.57%)
77.97
73.95
47.35
75.17
68.67
76.97
70.01 (↑0.62%)
W2A4
RTN
6.1e4
5.0e4
51.09
25.63
26.62
26.87
50.20
43.15
37.26
GPTQ
133.60
196.59
53.48
32.95
21.25
30.09
50.75
46.12
39.11
ResComp
17.49
55.22
58.25
38.32
23.74
38.54
53.20
61.59
45.44
GPTAQ
19.33
34.07
58.70
40.49
23.21
38.96
54.22
62.29
46.31
ResComp-MARR
17.27 (↓1.26%)
52.83 (↓4.32%)
57.56
37.37
23.98
38.35
55.25
61.50
45.67 (↑0.51%)
GPTAQ-MARR
17.43 (↓9.83%)
31.59 (↓7.28%)
59.79
39.35
25.26
40.58
54.14
58.90
46.34 (↑0.06%)
Stopping Criterion of PID.
To avoid unnecessary reconstruction overhead, we stop the update
when the relative change of the reconstruction objective becomes sufficiently small:

Jm(α(t)
m ) −Jm(α(t−1)
m
)
Jm(α(0)
m ) + εJ
 < τ,
(14)
where τ = 10−5 controls the stopping tolerance for the relative objective change. The current α(t)
m is
then used as the final residual scaling coefficient for module m.
With the above three components, PID-based update in MARR efficiently estimates a module-specific
αm that better balances the two-sided effect of the residual reconstruction term, leading to lower
module-level reconstruction error in most modules, as shown in Figure 2. The complete update
procedure is summarized in Appendix E, and a practical discussion on the stability of the PID-based
update is provided in Appendix F.
4
Experiments
4.1
Settings
In this paper, all experiments are conducted under the same PyTorch framework on a single NVIDIA
A6000 GPU. For a fair comparison, all methods are re-run with the same calibration data, quantization
settings, hyperparameters, and evaluation protocols. For LLMs, we follow the evaluation protocol of
GPTAQ [19], reporting perplexity on WikiText2 [27] and C4 [30] together with zero-shot accuracy
on six downstream tasks. For ViTs, we follow the protocol of FIMA-Q [39] and report ImageNet [31]
top-1 accuracy. In low-bit settings, all methods adopt the same rotation-based transformation [3]
to mitigate activation outliers and avoid severe performance collapse. More detailed experimental
settings are provided in Appendix D.
7

Table 2: Compared results of MARR under weight-only quantization on three Llama-family models.
We report perplexity on WikiText2 and C4 together with the average zero-shot accuracy on six
downstream tasks under W3A16 and W2A16.
W/A
Method
Llama2-7b
Llama2-13b
Llama3-8b
Wiki2 ↓
C4 ↓
Avg ↑
Wiki2 ↓
C4 ↓
Avg ↑
Wiki2 ↓
C4 ↓
Avg ↑
FP
5.47
6.90
70.50
4.88
6.41
73.30
6.44
9.61
74.30
W3A16
RTN
146.54
114.65
37.32
48.90
54.60
43.64
39.60
50.90
47.72
GPTQ
6.10
8.70
67.67
5.36
7.71
71.30
7.55
13.08
70.62
ResComp
5.92
8.44
67.67
5.25
7.58
71.09
7.30
12.76
70.11
GPTAQ
5.89
8.36
67.44
5.22
7.50
71.34
7.28
12.57
70.30
ResComp-MARR
5.88 (↓0.68%)
8.37 (↓0.83%)
67.25 (↓0.62%)
5.22 (↓0.57%)
7.54 (↓0.53%)
70.57 (↓0.73%)
7.26 (↓0.55%)
12.71 (↓0.39%)
70.19 (↑0.11%)
GPTAQ-MARR
5.83 (↓1.02%)
8.26 (↓1.20%)
67.83 (↑0.58%)
5.18 (↓0.77%)
7.45 (↓0.67%)
71.19 (↓0.21%)
7.18 (↓1.37%)
12.57 (↓0.00%)
69.30 (↓1.42%)
W2A16
RTN
9.8e3
1.2e4
36.30
5.2e3
5.0e3
35.22
4.1e4
3.0e4
36.11
GPTQ
21.33
41.76
45.10
9.91
18.72
52.99
20.30
59.17
48.52
ResComp
9.39
18.86
51.85
7.65
16.36
56.09
13.82
43.71
50.67
GPTAQ
9.36
18.99
52.03
7.71
15.08
54.51
13.52
36.47
51.92
ResComp-MARR
8.71 (↓7.24%)
18.40 (↓2.44%)
52.13 (↑0.54%)
7.42 (↓3.01%)
16.22 (↓0.86%)
54.23 (↓3.32%)
13.70 (↓0.87%)
42.73 (↓2.24%)
48.63 (↓4.03%)
GPTAQ-MARR
8.59 (↓8.23%)
16.86 (↓11.22%)
52.52 (↑0.94%)
7.08 (↓8.17%)
13.64 (↓9.55%)
56.59 (↑3.81%)
13.28 (↓1.78%)
35.90 (↓1.56%)
50.09 (↓3.52%)
Table 3: Compared results on vision transformers. We report top-1 accuracy (%) on ImageNet under
different quantization settings, where higher values indicate better performance.
Method
W4A4
W3A3
W2A4
DeiT-T
DeiT-S
DeiT-B
DeiT-T
DeiT-S
DeiT-B
DeiT-T
DeiT-S
DeiT-B
FP
72.71
79.85
81.80
72.71
79.85
81.80
72.71
79.85
81.80
PTQ4ViT
53.95
70.81
78.53
7.69
24.57
57.32
0.68
1.15
27.14
RepQ-ViT
55.31
70.13
78.37
0.72
6.43
18.11
0.17
0.22
3.01
AdaLog
62.58
64.81
76.20
30.05
55.82
71.90
1.76
11.90
46.25
GPTQ
64.19
74.77
78.79
29.26
43.41
67.35
30.30
54.43
67.20
ResComp
63.25
75.01
78.27
29.78
40.51
67.14
30.49
55.22
69.64
GPTAQ
65.16
76.35
79.62
32.32
56.41
68.82
33.58
59.17
70.79
ResComp-MARR
63.66 (↑0.65%)
75.05 (↑0.05%)
78.68 (↑0.52%)
30.10 (↑1.07%)
41.58 (↑2.64%)
68.07 (↑1.39%)
30.50 (↑0.03%)
55.48 (↑0.47%)
70.29 (↑0.93%)
GPTAQ-MARR
65.39 (↑0.35%)
76.81 (↑0.60%)
80.11 (↑0.62%)
32.45 (↑0.40%)
59.01 (↑4.61%)
69.22 (↑0.58%)
33.82 (↑0.71%)
59.92 (↑1.27%)
71.24 (↑0.64%)
4.2
Weight-Activation Quantization Results on LLMs
To evaluate MARR on large language models, we compare it with round-to-nearest (RTN) [28],
the basic reconstruction method GPTQ [11], and residual reconstruction methods GPTAQ [19] and
ResComp [17]. We further apply MARR to GPTAQ and ResComp, denoted as GPTAQ-MARR and
ResComp-MARR. Experiments are conducted on Llama2-7b [36], Llama2-13b [36] and Llama3-
8b [1] under W4A4 and W2A4 settings. Table 1 reports perplexity on WikiText2 and C4, together
with zero-shot accuracy on six downstream tasks [4, 8, 44, 32, 7].
As shown in Table 1, MARR achieves competitive or better results than most quantization methods,
especially under the challenging W2A4 setting. Compared with RTN and basic GPTQ, residual
reconstruction methods benefit from explicitly accounting for the accumulated activation mismatch
introduced by preceding quantized modules. Building on this advantage, MARR further improves
GPTAQ and ResComp by modulating the residual contribution at the module level, which preserves
useful accumulated-error correction while reducing residual-related HA bias.
4.3
Weight-Only Quantization Results on LLMs
We further evaluate MARR under weight-only quantization, where the activations remain in 16-bit
precision. Following the same protocol, we apply RTN [28], GPTQ [11], ResComp [17], GPTAQ [19],
and our GPTAQ-MARR/ResComp-MARR to Llama2-7b, Llama2-13b and Llama3-8b under W3A16
and W2A16. Table 2 summarizes the WikiText2 and C4 perplexity together with the average zero-shot
accuracy over six downstream tasks; the per-task zero-shot accuracy is provided in Appendix G.
As shown in Table 2, GPTAQ-MARR achieves the lowest WikiText2 and C4 perplexity in all six
model–bit-width combinations, and the largest gains appear under the more aggressive W2A16
setting (e.g., 8.23%/11.22% perplexity reduction over GPTAQ on Llama2-7b). This confirms that
controlling the residual contribution remains beneficial when only the weights are quantized.
4.4
Results on Vision Transformers
We further evaluate MARR on vision transformers under the standard ImageNet PTQ setting [39].
We compare it with representative ViT-oriented PTQ methods, including PTQ4ViT [43], RepQ-
ViT [20], and AdaLog [38], as well as reconstruction-based methods GPTQ [11], GPTAQ [19], and
ResComp [17]. Besides W4A4 and W2A4, we also include W3A3, since visual representations in
8

(a) Llama2-7b, Layer 3, Key Proj.
(b) Llama2-13b, Layer 13, Query
Proj.
(c) Llama3-8b, Layer 12, Output
Proj.
Figure 4: Trajectories of the residual scaling coefficient αm and the module-level reconstruction MSE
during the PID-based adaptive update on three representative modules from Llama2-7b, Llama2-13b,
and Llama3-8b under W2A4.
ViTs often exhibit considerable redundancy, making sub-4-bit activation quantization a meaningful
stress test for low-bit PTQ [39]. Table 3 reports the top-1 accuracy of DeiT-T, DeiT-S, and DeiT-B [35]
under W4A4, W3A3, and W2A4.
As shown in Table 3, MARR achieves competitive or better performance than most baselines across
different DeiT models and bit-widths. In particular, GPTAQ-MARR improves GPTAQ in most
settings, indicating that the proposed module-specific residual scaling is also effective for vision
transformers. This suggests that controlling the residual contribution can better balance accumulated-
error correction and residual-related HA bias beyond LLMs, demonstrating the generality of MARR
across Transformer architectures.
4.5
