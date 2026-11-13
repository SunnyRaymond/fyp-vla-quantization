# Fig. 2（LAQ 方法总览）生图 Prompt · v2

v1（`2026-11-13_fig2_laq_imagegen_prompt.md`）画成了推理流水线，作废。
这一版的原则：**整张图只讲 LAQ 自己算什么、做什么，不画任何模型的推理流程。**

## 用法

1. 上传 `CVPR2027/figures/figure2_laq_unified_imagegen.png` 作为**风格参考**（只沿用它的配色、标题条、线条和公式字体，版式和内容以下面的文字为准）。
2. 发主 prompt，一次出 2–4 张。
3. 用第 3 节的短句局部修，一次只改一处。

## 1. 版式：两行六格，全部是方法

| 格 | 标题 | 这一格要讲清的事 | 画面里的对象 |
|---|---|---|---|
| (a)-1 | Reader metric | 只对 reader 反向探针，就得到"future 的哪些 token 会改变动作" | z* → R → a*，反向探针；公式 M；**一帧 future 上的 token 热力图，只有一小块亮** |
| (a)-2 | Read count | 同样能量的误差，冻结的比每步重抽的伤害大，因为它在各步上同向累加 | 两条 T 步时间轴：同向小箭头首尾相接成长箭头，对随机小箭头成短箭头；两个 T×T 矩阵 Ω（满、仅对角）；公式 n_u |
| (a)-3 | Joint allocation | 一个目标拆成两支：权重用二阶取整；激活和 cache 用同一个阈值 | 目标函数；权重支路；**token × step 网格**，上边是 φ_s，左边是 g_j，超过 μ 的格子高亮；排序柱加 μ 线 |
| (b)-4 | Token selection | 每个控制步按当前 gaze 和 μ 选 token | 当前步的 token 网格，选中的高亮；输入是上一步的 gaze 和 μ |
| (b)-5 | Second 4-bit pass | 精度从哪来、为什么仍是 4-bit | 数轴上的粗网格和放大后的细网格；q1 + q2；两个 4-bit 矩阵乘，第二个只有选中的行；cache 里选中的条目存两个码 |
| (b)-6 | Gaze update | gaze 是生成动作的副产品，带到下一个控制步 | action token 到各 token 的注意力线，少数很粗；求和得到 g_j；虚线回到 (b)-4 |

通用性靠三点保证：不画 "future token → cache → reader" 这条串行链；误差源只分"冻结"和"新鲜"两类；模型本身只在右上角用一行小字交代。

## 2. 主 prompt

```text
A method-overview figure for a CVPR paper on LAQ, a 4-bit quantization method for world action models
(robot policies that predict a future latent z and read it to produce an action a). Full-width figure
of a two-column paper, wide landscape (about 2.4:1), white page. Use the attached image ONLY as a style
reference: same palette (blue = 4-bit base, orange = 4-bit residual, purple = calibration signals),
same dark title bars, thin lines, serif italic math. Ignore its layout and content.

HARD RULES. The figure explains the METHOD only. Do not draw an inference pipeline, a model
architecture, or any block chain from observation to action. No robot photo, no camera image. The model
appears only once, as one small line of text in the top-right corner:
"WAM:  (o, l) -> future z -> reader R -> action a". Everything must hold for any world action model.

LAYOUT. Two rows of three equal panels, six panels in total, each with a numbered title bar.
Row (a), titled "(a) Offline: what to protect". Row (b), titled "(b) At every control step: how to
protect it". Inside a panel, keep arrows short; between panels, use only the two links named below.

Panel 1 "Reader metric".
Left: a small stack of frames drawn as a token grid, labeled "future z*", an arrow into a box
"reader R", an arrow out to three small squares "action a*". Dotted purple arrows run backward from a*
through R to z*, labeled "reader probes". Center: the formula "M = E_c[ J_c^T D_c J_c ]".
Right, the largest element of the panel: ONE future frame as a token grid painted as a heat map, where
a small cluster of 3 to 4 neighboring tokens is strongly colored and all others are pale, labeled
"share of M per token". Bottom caption: "future error as the reader sees it:  Z_M".

Panel 2 "Read count".
Two horizontal timelines over the denoising steps "s = 1 ... T".
Top timeline, labeled "frozen error (weights, caches, static tokens)": the same small arrow repeated at
every step, all pointing the same way; to its right the arrows are chained head to tail into one LONG
arrow. Next to it a small T-by-T matrix with every cell filled.
Bottom timeline, labeled "fresh error (tokens being denoised)": small arrows pointing in random
directions; chained head to tail they give a SHORT arrow. Next to it a T-by-T matrix with only the
diagonal filled.
Both matrices are labeled "Omega_u". Formula: "n_u = 1^T Omega_u 1 / tr Omega_u".
Two tags: "n_u > 1" beside the top timeline, "n_u = 1" beside the bottom one.

Panel 3 "Joint allocation".
Top, across the panel: "L = E_c[ A + lambda_M Z_M + beta K ]". Below it the panel splits into two
branches.
Left branch "weights": a matrix "W", a box "second-order rounding", a matrix "4-bit weights".
Right branch "activations and caches": a grid whose rows are tokens and whose columns are denoising
steps. A small bar profile along its top edge is labeled "phi_s", a small bar profile along its left
edge is labeled "g_j". The cells where both are large are orange, the rest blue. Under the grid, bars
sorted from tall to short, cut by a horizontal line "mu", with "budget B" pointing at the line; bars
above the line are orange. Formula: "n_u omega_u sigma_u^2 / c_u >= mu".
Caption: "one threshold for activations and caches".

Panel 4 "Token selection".
A token grid of the current step with a few tokens in orange. Two inputs enter it: "mu" from above,
and a dashed purple arrow from Panel 6 labeled "gaze g_j". Small note in grey: "first step: from M".
Caption: "the same tokens are refined where they are written and where they are cached".

Panel 5 "Second 4-bit pass". This is the widest and most detailed panel of row (b).
Top: a number line with a coarse grid of ticks labeled "1st pass, step s1"; one coarse cell is
magnified into a fine grid of ticks labeled "2nd pass, step s2"; a dot "x" sits between coarse ticks
and exactly on a fine tick. Formula: "x_hat = s1 q1 + s2 q2", with q1 in blue and q2 in orange.
Middle: two matrix products drawn as grids. First "Q4(W) q1" with all rows blue. Then "+", then
"Q4(W) q2" where only a few rows are orange and the others are empty, labeled "selected rows only".
Bottom: a short column of cache entries; unselected entries hold one blue code "q1", selected entries
hold a blue "q1" and an orange "q2" side by side, labeled "cached keys and values".
A tag at the corner: "all weights 4-bit, no 16-bit branch".

Panel 6 "Gaze update".
A row of squares "action tokens" at the top. Attention lines fan down to a token grid; a few lines are
thick, most are hair-thin. The tokens hit by thick lines are orange. Text: "attention of the action
tokens, summed over layers, heads and steps". An arrow to a small bar profile "g_j". From it, the
dashed purple arrow leaves toward Panel 4, labeled "gaze for the next control step".

LINKS BETWEEN PANELS, only these two: a thin purple arrow from Panel 3 down to Panel 4 carrying "mu",
and the dashed purple arrow from Panel 6 back to Panel 4.
Legend, top right, under the WAM line: blue square "4-bit base", orange square "4-bit residual",
dotted arrow "calibration", dashed arrow "gaze feedback".
No other text, no logos, no photos, no decorative icons.
```

## 3. 修改短句

- 又画出了推理流程：`Remove every block chain that goes from observation to action. The six panels only show what LAQ computes. Change nothing else.`
- 热力图不突出：`In Panel 1 the heat-map frame is the largest element; only 3 to 4 neighboring tokens are strongly colored.`
- 冻结和新鲜看不出区别：`In Panel 2 the top arrows are identical and add up to a long arrow; the bottom arrows point in random directions and add up to a short one.`
- token × step 网格画错：`In Panel 3 the grid has tokens as rows and denoising steps as columns, with the profile "phi_s" on top and "g_j" on the left.`
- 第二遍像 8-bit：`In Panel 5 the two codes q1 and q2 are two separate 4-bit codes; no label says 8-bit.`
- 格间箭头太多：`Between panels keep only two arrows: "mu" from Panel 3 to Panel 4 and the dashed gaze arrow from Panel 6 to Panel 4.`
- 太挤：`Give every panel more white space and enlarge the labels; shorten any caption longer than eight words.`

## 4. 复刻进 PPT 时核对

- 希腊字母和下标重新打：Ω_u、λ_M、β、μ、φ_s、ω_u、σ_u、ς₁、ς₂（prompt 里用 s1、s2、Omega 等写法只是为了让生图模型不乱码）。
- 四个方法名与 §4 小节标题一致；图注里的 (i)–(iv) 对应格 1、2、3 和整行 (b)。
- 不出现 Fast-WAM 的模块名和实测数字。
- 图注第一句 "A WAM writes its predicted future in a denoising loop, stores it in a key–value cache and reads it…" 要跟着新图改写。
