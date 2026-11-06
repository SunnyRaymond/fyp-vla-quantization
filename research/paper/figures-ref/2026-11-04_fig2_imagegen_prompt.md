# Fig. 2（方法总览）生图 Prompt

上传两张参考图（顺序不变），都在 `fig2_prompt_refs/`：

- **Image 1** `01_fig1_style.png`：定稿的 Fig. 1，作风格蓝本。模块框、future 帧堆叠、夹爪图标、配色都照它来。
- **Image 2** `02_fig1b_seed.png`：Fig. 1 旧版的 (b) 面板。①②④ 三个阶段的小图可以直接沿用。

## 主 prompt

```text
A structured scientific method-overview figure for a CVPR paper, read left to right. Ultra-wide (about 3:1),
white background, pastel palette, flat vector style with short soft drop shadows, rounded cards, Arial-style
bold headings, italic math letters. Match the visual language of Image 1 exactly: the same module boxes with
small grids, the same stacked "future latent" frames, the same gripper icon, the same colors. Image 2 shows
earlier drafts of three stages; reuse its pictograms. Keep text short and spell every quoted label exactly.

TOP STRIP: the world action model at deployment, connected by solid gray arrows:
"observation" photo -> blue-gray box "generator G" -> orange stack of frames "predicted future z"
-> blue-gray box "reader R" -> gripper icon "action a".

BELOW: four stage cards in one row, each numbered with an orange circle and linked by a thin dashed arrow
up to the part of the top strip it acts on.
1 "Reader metric" (dashed arrow to "predicted future z"): the future-frame grid with a few orange cells,
  an orange elliptical contour labeled "M" over it, small gray text "future error as the reader sees it".
2 "Future-aware bit allocation" (dashed arrow to both module boxes): a few small bars labeled "Δg"
  (gain of one extra bit for the future), then a row of module blocks shaded light to dark by bit width,
  small gray text "same bit budget".
3 "Trellis-coded initialization" (dashed arrow to the whole top strip): a weight matrix feeding a small
  trellis of a few nodes and edges, producing a bit string "codes" that is decoded through one small
  table "C", small gray text "one decoder for all bit widths".
4 "Shared-table calibration" (dashed arrow to table "C"): the codes with a small lock icon (frozen) and
  the table "C" highlighted in orange with "+ΔCg"; three red arrows come back to the table from small
  targets "A" (action), "Z_M" (future) and "K" (reader), joined in one red label "A + λZ_M + βK".

Bottom banner in bold orange italics: "Deployed: standard trellis decoder · same bits per weight · no extra storage".
Small legend at the top right: solid arrow "inference", dashed arrow "offline quantization",
red arrow "calibration signal".

Colors: blue-gray #8A99AD for full-precision modules, orange #D9822B for the future and for LAVQ, red #C24A3E
for calibration signals, light gray for frozen parts. Stage cards use a very light warm fill (#FFF7EF) with
thin gray outlines, like the cards in Image 2.
No other text, no numbers, no logos, no people, no 3D renders, no dark background.
```

## 修改短句（同一个对话里接着发，一次只改一处）

- 箭头指错位置：`Stage 1's dashed arrow points to "predicted future z", stage 2's to both module boxes, stage 3's to the whole top strip, stage 4's to table "C". Change nothing else.`
- 公式乱了：`Write the loss exactly as "A + λZ_M + βK" on one line in italic letters; no other math anywhere.`
- 太挤：`Give the four cards equal width and more white space; the top strip takes about one third of the height.`
- 风格跑偏：`Use exactly the module-box, frame-stack and gripper styles of Image 1, and the card style of Image 2.`
- 多出文字：`Remove every text element that is not quoted in the prompt.`
- 可选，强调“观察 → 设计”：`Under the titles of cards 1, 2 and 4, add a tiny gray tag: "from Obs. (c)", "from Obs. (b)", "from Obs. (a)".`

## 复刻进 PPT 时核对

- **阶段名与 §4 一致**：Reader metric / Future-aware bit allocation / Trellis-coded initialization / Shared-table calibration。符号用 G、R、z、M、Z_M、A、K、ΔC_g。
- **不出现 QTIP，也不出现 Fast-WAM 的模块名**（video/action expert）。§4 是通用叙述，Fig. 2 跟它保持一致。② 不放 2.22 / 1.03 这类具体位宽，它们是 Fast-WAM 的结果，在附录表里。
- **数学符号在 PPT 里重新打**：生图模型的下标和希腊字母基本都会走样。
- **字号**：画布沿用 Fig. 1 的 22.6 in 宽时，正文至少 16–17 pt。缩到双栏宽（6.875 in）后约 5 pt。
