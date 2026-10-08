# EfficientDM: Efficient Quantization-Aware Fine-Tuning of Low-Bit Diffusion Models

paper_id: arxiv:2310.03270v4
tier: U
source_used: html_arxiv
warning: none

## Intro

Diffusion models (DM)
(
Ho et al., 2022b
;
Dhariwal & Nichol, 2021
;
Rombach et al., 2022a
;
Ho et al., 2022a
)
have demonstrated remarkable capabilities in image generation and related tasks.
Nonetheless, the iterative denoising process and the substantial computational overhead of the denoising model limit the efficiency of DM-based image generation.
To expedite the image generation process, numerous methods
(
Bao et al., 2022
;
Song et al., 2021
;
Liu et al., 2022
;
Lu et al., 2022
)
have been explored to reduce the number of denoising iterations, effectively reducing the previously required thousands of iterations to mere dozens.
However, the significant volume of parameters within the denoising model still demands a substantial computational burden for each denoising step, resulting in considerable latency,
hindering
the practical application of DM in real-world settings with latency and computational resource constraints.
Model quantization, which compresses weights and activations from 32-bit floating-point values into lower-bit fixed-point formats, alleviating both memory and computational burdens.
This effect can be increasingly pronounced as the bit-width decreases.
For instance, leveraging Nvidia’s CUTLASS
(
Kerr et al., 2017
)
implementation, an 8-bit model’s inference speed can be
2.03
×
2.03\times
faster than that of a full-precision (FP) model, and the acceleration ratio reaches
3.34
×
3.34\times
for a 4-bit model.
Therefore, it possesses substantial potential for significantly compressing and accelerating diffusion models, making them highly suitable for deployment on resource-constrained devices such as mobile phones.
Nevertheless, the challenges associated with low-bit quantization for diffusion models have not received adequate attention.
Typically, model quantization can be executed through two predominant approaches: post-training quantization (PTQ) and training-aware quantization (QAT).
PTQ calibrates the quantization parameters with a small calibration dataset, which is time- and data-efficient.
However, they introduce substantial quantization errors at low bit-width.
As illustrated in Figure
1
, when quantizing both weights and activations to 4-bit with the PTQ method
(
He et al., 2023a
)
, diffusion models fail to maintain their denoising capabilities.
In contrast, QAT methods
(
Krishnamoorthi, 2018
;
Esser et al., 2019
)
can recover performance losses at lower bit-width by fine-tuning the whole model.
However, this approach requires significantly more time and computing resources compared to PTQ method
(
He et al., 2023a
)
, as evidenced by a
2.6
×
2.6\times
increase in GPU memory consumption (31.4GB vs. 11.7GB) and a
18.9
×
18.9\times
longer execution time (54.5 GPU hours vs. 2.88 GPU hours) when fine-tuning LDM-4
(
Rombach et al., 2022a
)
on ImageNet
256
×
256
256\times 256
.
Moreover, in some cases, it may be challenging or even impossible to obtain the original training dataset due to privacy or copyright concerns.
Figure 1:
An overview of the efficiency-vs-quality tradeoff across various quantization approachs. Data is collected on LDM-8
(
Rombach et al., 2022a
)
with 4-bit weights and activations on LSUN-Churches. The GPU memory consumption is visualized by circle size.
In this paper, we introduce a data-free and parameter-efficient fine-tuning framework for low-bit diffusion models, denoted as EfficientDM, which demonstrates the capability to achieve
QAT-level performance
while upholding
PTQ-level efficiency
during fine-tuning in terms of data and time.
The foundation of our approach lies in a quantization-aware form of the low-rank adapter, as depicted in Figure
2
b.
This variant enables the joint quantization of LoRA weights with model weights, thereby obviating additional storage and calculations.
Compared to previous QAT method
(
So et al., 2023
)
, our fine-tuning process is executed in a data-free manner, accomplished by minimizing the mean squared error (MSE) between the estimated noise of full-precision denoising model and its quantized counterpart, as illustrated in Figure
2
c, thus eliminating the need for the original training dataset.
Due to quantization, the relationship between full-precision LoRA weights and quantized updated model weights becomes step-like with step size,
i.e.
, the quantization scale parameter.
Effective updates are mostly observed in layers with small scales, while other layers with larger scales do not benefit. To address this, we introduce scale-aware LoRA optimization that adaptively adjusts the gradient scales of LoRA weights in different layers to ensure an effective optimization.
Furthermore, we extend the learned step size quantization (LSQ) method
(
Esser et al., 2019
)
into the denoising temporal domain for activations, effectively mitigating quantization errors due to varying activation distributions across time steps.
Figure 2:
An overview of the proposed EfficientDM fine-tuning framework. Here,
𝐬
w
\mathbf{s}_{w}
and
s
x
s_{x}
represent the learnable quantization scales for weights and activations, respectively. Compared to QLoRA layer, both updated weights and activations in our QALoRA are quantized to enable efficient bitwise operations during inference.
Fine-tuning is performed by minimizing the mean squared error between the estimated noises of FP and quantized models.
In summary, our contributions are as follows:
•
We introduce EfficientDM, an efficient fine-tuning framework for low-bit diffusion models which can achieve QAT performance with the efficiency of PTQ. The framework is rooted in the quantization-aware form of low-rank adapters (QALoRA) and distills the denoising capabilities of full-precision models into their quantized counterparts.
•
We propose scale-aware LoRA optimization to alleviate the ineffective learning of QALoRA resulting from substantial variations in weight quantization scales across different layers. We also introduce TALSQ, an extension of activation learned step size quantization within the temporal domain to tackle the variation in activation distributions across denoising steps.
•
Extensive experiments on CIFAR-10, LSUN and ImageNet demonstrate that our
EfficientDM
reaches a new state-of-the-art performance for low-bit quantization of diffusion models.

## Method

In this section, we propose an efficient fine-tuning framework for diffusion models with both weights and activations quantized, possessing the efficiency of PTQ and the accuracy of QAT.
The proposed framework, dubbed EfficientDM, is depicted in Figure
2
.
It consists of a quantization-aware low-rank adapter and a noise distillation strategy, delivering parameter-efficient and data-free fine-tuning. Moreover, it is also equipped with scale-aware LoRA optimization and a temporal-aware quantizer for improved performance.
We elaborate each design as follows.
Quantization-aware low-rank adapter.
Low-rank adapter (LoRA) fine-tuning constrains the update of the model parameters to possess a low intrinsic rank, denoted as
r
r
. Given a pretrained linear module
𝐘
=
𝐗𝐖
0
\mathbf{Y}=\mathbf{X}\mathbf{W}_{0}
, where
𝐗
∈
ℝ
b
×
c
i
​
n
\mathbf{X}\in\mathbb{R}^{b\times c_{in}}
and
𝐖
0
∈
ℝ
c
i
​
n
×
c
o
​
u
​
t
\mathbf{W}_{0}\in\mathbb{R}^{c_{in}\times c_{out}}
, with
b
b
representing the batch size,
c
i
​
n
c_{in}
and
c
o
​
u
​
t
c_{out}
representing the number of input and output channels, respectively, LoRA fixes the original weights
𝐖
0
\mathbf{W}_{0}
and introduces updates as follows:
𝐘
=
𝐗𝐖
0
+
𝐗𝐁𝐀
,
\displaystyle\mathbf{Y}=\mathbf{X}\mathbf{W}_{0}+\mathbf{X}\mathbf{B}\mathbf{A},
(5)
where
𝐁
∈
ℝ
c
i
​
n
×
r
\mathbf{B}\in\mathbb{R}^{c_{in}\times r}
and
𝐀
∈
ℝ
r
×
c
o
​
u
​
t
\mathbf{A}\in\mathbb{R}^{r\times c_{out}}
are the learnable two low-rank matrices with
r
≪
min
⁡
(
c
i
​
n
,
c
o
​
u
​
t
)
r\ll\mathrm{min}(c_{in},c_{out})
.
Nevertheless, this approach incurs limitations when both weights and activations are quantized, denoted by
𝐖
0
^
\hat{\mathbf{W}_{0}}
and
𝐗
^
\hat{\mathbf{X}}
, respectively.
In this case,
the inner product between
𝐖
0
^
\hat{\mathbf{W}_{0}}
and
𝐗
^
\hat{\mathbf{X}}
can be efficiently implemented with bit-wise operations, whereas the operations involving
𝐁𝐀
\mathbf{BA}
and
𝐗
^
\hat{\mathbf{X}}
are computationally expensive during inference as
𝐁𝐀
\mathbf{BA}
is full-precision and has the same size as
𝐖
0
\mathbf{W}_{0}
.
To address this, we propose Quantization-aware Low-rank Adapter (QALoRA), where the LoRA weights are first merged with FP model weights and then jointly quantized to the target bit-width, as depicted in Figure
2
b.
Formally, the QALoRA is defined as follows:
𝐘
=
𝒬
U
​
(
𝐗
,
s
x
)
​
𝒬
U
​
(
𝐖
0
+
𝐁𝐀
,
𝐬
w
)
=
𝐗
^
​
𝐖
^
,
\displaystyle\mathbf{Y}=\mathcal{Q}_{U}(\mathbf{X},s_{x})\mathcal{Q}_{U}(\mathbf{W}_{0}+\mathbf{BA},\mathbf{s}_{w})=\hat{\mathbf{X}}\hat{\mathbf{W}},
(6)
where
𝐬
w
\mathbf{s}_{w}
denotes the channel-wise quantization scale for weights and
s
x
s_{x}
is the layer-wise quantization scale for activations.
After the fine-tuning process, only quantized updated model weights
𝐖
^
\hat{\mathbf{W}}
need to be saved. Notably, our approach can be readily integrated with QLoRA
(
Dettmers et al., 2023
)
by substituting
𝐖
0
\mathbf{W}_{0}
with
𝐖
0
^
\hat{\mathbf{W}_{0}}
to further reduce memory footprint.
Data-free fine-tuning for diffusion models.
Diffusion models require access to large and diverse datasets for effective training.
Obtaining such datasets can be challenging due to their sheer size, privacy concerns, or copyright restrictions.
To alleviate the dependency on the original dataset,
we propose a data-free fine-tuning approach that distills the denoising capabilities of a full-precision model into its quantized counterpart.
Specifically, we input the same noise
𝐱
t
\mathbf{x}_{t}
to both FP and quantized denoising models at denoising step
t
t
and minimize the mean squared error (MSE) between their denoising results:
ℒ
t
=
‖
𝝁
θ
​
(
𝐱
t
,
t
)
−
𝝁
^
θ
​
(
𝐱
t
,
t
)
‖
2
,
\displaystyle\mathcal{L}_{t}=\left\|{\bm{\mu}}_{\theta}(\mathbf{x}_{t},t)-\hat{{\bm{\mu}}}_{\theta}(\mathbf{x}_{t},t)\right\|^{2},
(7)
where
𝝁
θ
​
(
𝐱
t
,
t
)
{\bm{\mu}}_{\theta}(\mathbf{x}_{t},t)
and
𝝁
^
θ
​
(
𝐱
t
,
t
)
\hat{{\bm{\mu}}}_{\theta}(\mathbf{x}_{t},t)
denote the denoising results of the FP and quantized models for the denoising step
t
t
, respectively. The input data
𝐱
t
\mathbf{x}_{t}
is obtained by denoising random Gaussian noise
𝐱
T
∼
𝒩
⁡
(
0
,
1
)
\mathbf{x}_{T}\sim\mathcal{N}(0,1)
with FP model iteratively for
T
−
t
T-t
steps, as illustrated in Figure
2
c.
To facilitate the training of QALoRA, we further address the following technical challenges:
Variation of weight quantization scales across layers.
As demonstrated in Eq. (
6
), due to the quantization process, the relationship between full-precision LoRA weights
𝐁𝐀
\mathbf{BA}
and
quantized updated weights
𝐖
^
\hat{\mathbf{W}}
follows a step function, where the step size is exactly equal to the quantization scale
𝐬
w
\mathbf{s}_{w}
,
and
𝐖
0
\mathbf{W}_{0}
serves as a fixed offset.
This relationship is visually presented in Figure
3(a)
.
Consequently, while the full-precision LoRA weights are continuously optimized during the fine-tuning process, they need to be large enough to update the quantized model weights, otherwise they will be diminished by the
round
\mathrm{round}
operation within the quantization process, as referred to Eq. (
3
) and (
6
).
As shown in Figure
3(b)
and “Scale-agnostic training” in Figure
3(c)
, fine-tuning the LoRA weights with limited iterations only yields effective updates for a few layers with relatively small scales. For other layers, their quantization scales are too substantial for LoRA weights to take effect due to the
round
\mathrm{round}
operation.
Alternative approaches, such as directly amplifying the learning rate, may impede the convergence process.
To facilitate the optimization of LoRA weights, we consider the ratio of
R
=
∇
BA
ℒ
𝐬
w
¯
\displaystyle R=\frac{\nabla_{\textbf{BA}}\mathcal{L}}{\overline{\mathbf{s}_{w}}}
(8)
should be roughly consistent in each layer, where
𝐬
w
¯
\overline{\mathbf{s}_{w}}
represents the averaged weight quantization scale
across
channels.
This can be achieved by simply multiplying the gradient of LoRA weights by this average weight quantization scale during the back-propagation phase.
As shown in Figure
3(c)
, optimizing LoRA in the scale-aware approach enables effective training across the majority of layers.
(a)
(b)
(c)
Figure 3:
The motivation and effect of scale-aware LoRA optimization. Data is collected from the 4-bit LDM-4 model.
(a):
Due to step-like relationship between
𝐁𝐀
\mathbf{BA}
and
𝐖
^
\hat{\mathbf{W}}
,
𝐁𝐀
\mathbf{BA}
needs to be large enough to update model weights. Data is collected from the first channel in the
12
t
​
h
12^{th}
layer.
(b):
Significant disparity in weight quantization scales across layers.
(c):
Mean absolute value of quantized weight updates (
𝐖
^
\hat{\mathbf{W}}
-
𝐖
0
^
\hat{\mathbf{W}_{0}}
) for each layer. Most of them are zero under scale-agnostic training, indicating full-precision LoRA weights are too small to update quantized model weights. The proposed scale-aware LoRA optimization facilitates a more equitable distribution of quantized weight updates across layers.
Variation of activation distribution across steps.
Previous research on diffusion models
(
Shang et al., 2023
;
Li et al., 2023
;
So et al., 2023
)
has identified a pronounced variability in activation distributions at different time steps, which is also presented in Appendix
C
.
This variability poses substantial challenges to the quantization of diffusion models.
Notably, existing methods proposed to address this issue aim to either find a set of quantization parameters applicable to all time steps
(
Shang et al., 2023
)
or employ additional MLP module to estimate quantization parameters for each individual time step and layer
(
So et al., 2023
)
,
which can be either suboptimal or cumbersome.
Inspired by LSQ
(
Esser et al., 2019
)
, a technique where quantization scales are optimized alongside other trainable parameters through the gradient descent algorithm, we allocate temporal-aware quantization scales for activations and optimize them individually for each step, which we refer to as Temporal Activation LSQ (TALSQ):
S
x
=
{
s
x
0
,
s
x
1
,
…
,
s
x
T
−
1
}
,
\displaystyle S_{x}=\left\{s_{x}^{0},s_{x}^{1},\ldots,s_{x}^{T-1}\right\},
(9)
where
T
T
is the number of denoising steps for the fine-tuning.
It is noteworthy that recent advancements in efficient samplers have significantly reduced the number of sampling steps.
Therefore, TALSQ introduces only a few trainable parameters for a single layer, which is negligible even compared to LoRA weights (which generally have thousands of parameters per layer).
After fine-tuning, we interpolate the learned temporal quantization scales to deal with the gap of sampling steps between fine-tuning and inference.
