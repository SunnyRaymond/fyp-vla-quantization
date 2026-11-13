# Fig. 2（LAQ 方法总览）生图 Prompt

对应论文 `CVPR2027/sec/2_related.tex` 里的 `fig:method`，内容以 `sec/4_method.tex` 为准。
流程和 LAVQ 那张一样：生图模型出草图 → PPT 复刻 → 导出矢量 `figure2.pdf`。
这一版不上传参考图，风格、配色、图标全部交给生图模型。prompt 只规定结构、语义和文字。

## 1. 这张图要讲什么

一句话：**预测的 future 有"写、存、读"三个阶段，LAQ 找出动作真正依赖的那几个 token，在三个地方给同一批 token 加一遍 4-bit，全程不离开 4-bit。**

图分两层。

**底层是被量化的 WAM（一条从左到右的流水线）：**

| 位置 | 画什么 | 误差类型 |
|---|---|---|
| 输入 | 观测图像、指令 | — |
| Write | 观测 token（每个控制步只算一次）；future token 在去噪循环里迭代 T 步，写出预测的 future z | 观测 token 的误差是冻结的；正在去噪的 future token 每步重新取整，误差是新鲜的 |
| Store | 写完的 future 存进 key–value cache，每格一个 4-bit 码 | 冻结 |
| Read | reader R：action token 去噪 T 步，每步都通过注意力读 cache，输出动作 a | — |
| 两个循环下方 | 4-bit 权重，两个循环共用 | 冻结 |

**上层是 LAQ 加的四样东西，各自挂在它作用的位置上：**

| 编号 | §4 小节 | 挂在哪 | 画法 |
|---|---|---|---|
| (i) | Reader metric | 从动作沿 reader 反向指到预测的 future | 反向箭头在 future 上点亮一张热力图，只有一小簇 token 是热的。含义：只通过 reader 做反向探针，就知道 future 的哪一部分会改变动作 |
| (ii) | Read count | 每个误差源上一个小计数徽章 | 权重、观测 token、cache 标 `n > 1`（写一次，每步都被读）；循环里的 future token 标 `n = 1` |
| (iii) | Joint allocation | 流水线旁的一个小图 | 从高到低排好的柱子被一条虚线 μ 切开，线以上的柱子变成高亮色。含义：循环里的激活和 cache 用同一个阈值排队 |
| (iv) | Gaze-guided refinement | 贯穿三个站 | 被选中的 token 用同一种高亮色出现在观测 token、循环里的 future token、cache 格子三处；cache 里这些格子放两个 4-bit 码；reader 的粗注意力线正好落在这些格子上。一个小插图解释"第二遍 4-bit"：粗网格里的一格再细分 |

另有一条虚线反馈箭头：从 reader 的注意力线绕回 Write 站的 token 选择，表示这一步的 gaze 决定下一个控制步精修哪些 token。

**图里一共有三种箭头**：正向推理、离线反向探针 (i)、运行时 gaze 反馈。这三种必须能一眼分开，怎么分由生图模型定。

**不放进图里的东西**：具体数字（读次数的测量值、196 倍、预算）、公式、Fast-WAM 的模块名（video expert、action expert、frame 0）、基线名字、文本 K/V cache。§4 是通用叙述，Fig. 2 跟它一致；联合去噪的 Cosmos-Policy 由图注里的一句话覆盖。

## 2. 主 prompt

```text
A structured scientific method-overview figure for a CVPR paper: ONE left-to-right pipeline with three
stations titled "Write", "Store" and "Read", plus four numbered callouts (i), (ii), (iii), (iv) anchored
on the pipeline at the place each one acts on. It is a full-width figure of a two-column paper:
ultra-wide landscape (about 3:1), placed on a white page, and it must stay readable when printed 17 cm
wide, so use few, large, short labels. The visual style, color palette, icons and typography are
entirely your choice; make it look like a polished method figure of a top-tier computer-vision paper.
Spell every quoted label exactly and add no other text.

WHAT THE FIGURE SAYS. A world action model is a robot policy that first predicts a future and then reads
it to produce an action. Here the whole model runs at 4 bits. The predicted future has a life: it is
written, stored and read. Our method, LAQ, finds the few tokens the action really depends on and gives
the SAME tokens extra precision at all three places, without ever leaving 4-bit arithmetic.

BASE PIPELINE (the 4-bit model, the calm backbone of the figure), linked by forward arrows:
- Far left, the inputs: a small camera image of a robot arm at a table, labeled "observation", and a
  short text chip labeled "instruction".
- Station "Write": a short column of small squares labeled "observation tokens" stands outside a
  circular loop arrow; arrows from it enter the loop. Inside the loop, a larger grid of small squares
  labeled "future tokens"; the loop arrow is labeled "× T steps". The loop outputs a small stack of
  frames labeled "predicted future z".
- Station "Store": the finished future saved as a grid of cells labeled "key-value cache". Every cell
  holds one small 4-bit code.
- Station "Read": a box labeled "reader R" containing a short row of squares labeled "action tokens"
  inside a second loop arrow "× T steps". Thin attention lines fan out from the action tokens back to
  the cache cells; a few of these lines are much thicker than the others. The reader outputs a robot
  gripper labeled "action a".
- Under the two loops, one shared block labeled "4-bit weights", feeding both loops.

WHAT LAQ ADDS (clearly distinct from the backbone):
(i) "Reader metric": a backward arrow, drawn differently from the forward arrows, runs from "action a"
    back through the reader onto the predicted future and stops there. Where it lands, the future
    frames show a heat map: one small cluster of tokens is hot, all others are cold. Small caption:
    "future error as the reader sees it".
(ii) "Read count": a small counter badge sits on every place where a rounding error can live. The badge
    reads "n > 1" on the 4-bit weights, on the observation tokens and on the key-value cache, because
    their error is written once and read again at every step. It reads "n = 1" on the future tokens
    inside the loop, because their error is redrawn at every step.
(iii) "Joint allocation": a small chart of bars sorted from tall to short, cut by one horizontal dashed
    line labeled "μ". The bars above the line take the highlight color. Small caption: "one threshold
    for the loop and the cache". An arrow goes from this chart to the highlighted tokens.
(iv) "Gaze-guided refinement": the selected tokens appear in ONE highlight color, identical at all three
    places: a few of the observation tokens, a few of the future tokens in the loop, and the matching
    cells of the cache, which hold two small 4-bit codes side by side instead of one. The thick
    attention lines of the reader land on exactly these cells, and the hot cluster of (i) sits on the
    same tokens. A thin band in the highlight color links them across the three stations, showing that
    it is one token set. A small inset explains the extra precision: a coarse grid labeled
    "1st 4-bit pass", and inside one of its cells a finer grid labeled "2nd 4-bit pass".
A dashed feedback arrow, a third arrow style, leaves the attention lines of the reader and curves back
to the highlighted tokens of the "Write" station, labeled "gaze for the next control step".

A compact legend with three entries: forward arrow "inference", backward arrow "offline probe",
dashed arrow "run-time gaze".
No numbers other than the quoted ones, no equations, no logos, no people, no extra captions.
```

## 3. 修改短句（同一个对话里接着发，一次只改一处）

- 变成了上下两排卡片：`Keep a single left-to-right pipeline. The callouts (i) to (iv) sit on the pipeline at the place they act on, not in a separate row of cards. Change nothing else.`
- 高亮色不统一：`Use exactly one highlight color for the refined tokens at all three places (observation tokens, future tokens in the loop, cache cells) and for nothing else.`
- 徽章标错：`The badge on the future tokens inside the loop reads "n = 1"; the badges on the 4-bit weights, the observation tokens and the key-value cache read "n > 1".`
- 反向箭头画过头：`The reader-metric arrow starts at "action a", goes backward through the reader and stops at the predicted future. It does not continue further left.`
- gaze 箭头不对：`The dashed arrow starts at the attention lines of the reader and ends at the highlighted tokens of the "Write" station. It is the only dashed arrow in the figure.`
- cache 看起来像 8-bit：`Each selected cache cell holds two separate small 4-bit codes side by side, never one larger code. No label says 8-bit or 16-bit.`
- 太挤：`More white space. The insets of (iii) and (iv) each take at most one sixth of the width.`
- 多出文字：`Remove every text element that is not quoted in the prompt and enlarge the remaining labels.`
- 想换一种风格再看看：`Keep the layout and every label. Propose a clearly different visual style.`

## 4. 复刻进 PPT 时核对

- **四个名字与 §4 小节标题一致**：Reader metric / Read count / Joint allocation / Gaze-guided refinement，编号 (i)–(iv) 的顺序与图注一致。
- **符号在 PPT 里重新打**：R、z、a、T、M、n_u、μ。生图模型的下标和希腊字母会走样。
- **不出现 Fast-WAM 的模块名和任何实测数字**。徽章只写 `n > 1` 和 `n = 1`。
- **权重那一块**可以加一行小字 "second-order rounding"，对应 (iii) 里权重的解。草图里没放，免得字太多。
- **换成真实画面**：观测图换成 LIBERO 的真实观测帧，future 上的热力图换成 Fig. 3(b) 的画法，两张图就能对上。帧在 `experiment/idea-validation/paper-fig3-export-20261112/pulled/results/frames/`。
- **与 Fig. 1、Fig. 3 的配色**在这一步统一。生图模型自己选的颜色只当草图用。
- **字号**：画布沿用 22.6 in 宽时，正文至少 16–17 pt，缩到双栏宽（6.875 in）后约 5 pt。
- **导出后**：`CVPR2027/figures/figure2.pdf` 覆盖旧的 LAVQ 版，删掉 `2_related.tex` 图注里的红色 `\TODO` 和上面那行 placeholder 注释。
