# Fig. 1（方案 A）生图 Prompt · v2

论文里现在用的是 matplotlib 初稿：`CVPR2027/figures/teaser.pdf`，脚本是 `research/paper/figures/fig1_teaser.py`。这份 prompt 只用来让生图模型出一张**草图**，提供版式和视觉灵感。定稿要在 PPT 里复刻，数字以 matplotlib 版本为准。

## 对照《论文画图指南》改了什么

| 指南里的要求 | v1 的问题 | v2 的改法 |
|---|---|---|
| AI 先出草图，再用 PPT/Visio 复刻、微调 | v1 想让模型直接画定稿，逐像素规定位置、逐个写数字 | 只规定结构、配色和关键文字。数字和柱状图后期贴真实版本 |
| 提示词先定结构，例如 “A structured, 3-part scientific diagram from left to right, clean background, pastel color palette, vector style” | v1 开头是任务描述，结构埋在后面 | 第一句就用指南的结构句式 |
| Teaser 只讲两件事：问题/观察，以及本文的优势 | v1 的 (b) 列了方法四步 ①–④，还有位宽柱状图，这些属于方法图 Fig. 2 | (b) 只留一个概念画面：LAVQ 保护 future，比特数不变 |
| 用统一风格的图标，用阴影、实线/虚线增加层次 | v1 明确禁止图标和阴影 | 允许一套统一的细线图标（相机、机械臂、盾牌）和轻微阴影 |
| 读者几秒内抓到重点，图要 self-contained | v1 图里文字太多，模型越容易拼错 | 图中文字压到 15 条左右，每条都很短 |

**字数**：主 prompt 从 1262 词降到约 340 词。参考图从 6 张减到 3 张。

---

## 1. 用法

1. 上传 3 张参考图，顺序不要变（都在 `fig1_prompt_refs/`）：
   - **Image 1** `01_layout_draft.png`：版式参考，(a)(c) 照它摆。(b) 以文字描述为准。
   - **Image 2** `02_house_style.png`：风格参考，用你之前生成的那张方法图。
   - **Image 3** `06_libero_observation.png`：真实观测帧，放进相机小方格。
2. 发第 2 节的主 prompt，一次出 2–4 张，挑版式最好的一张。
3. 用第 3 节的短句局部修。满意后在 PPT 里复刻：文字统一用 Arial，(c) 换成 `teaser.pdf` 里的真实柱状图。

---

## 2. 主 prompt（约 340 词）

```text
A structured, 3-part scientific teaser figure from left to right for a CVPR paper. Ultra-wide (about 3:1),
clean white background, pastel color palette, flat vector style with subtle soft shadows, one consistent set of
thin-line icons, Helvetica-like font. Follow the layout of Image 1 for panels (a) and (c), and the visual style
of Image 2. Keep text minimal and spell every label exactly as quoted.

Panel (a), left half. Title: "(a) 2-bit VQ breaks the predicted future".
Three rows, each a left-to-right pipeline: small camera photo (use Image 3) -> rounded box "generator" ->
a small stack of three video frames -> rounded box "reader" -> robot-gripper icon -> horizontal error bar.
Row "BF16": blue-gray boxes, clean orange frames, no error bar.
Row "2-bit VQ": pale-red boxes, frames covered with red diagonal hatching, long red error bar.
Row "2-bit VQ + BF16 future": pale-red boxes, clean purple frames, short red error bar whose missing part is an
empty dashed outline labeled in bold orange "-73%".
A thick semi-transparent purple curved arrow carries the frames of the first row down into the third row,
labeled "swap in BF16 future".
Under the rows, bold orange text: "73% of the action error comes through the future".

Panel (b), narrow middle. Title: "(b) LAVQ protects the future".
One pipeline like a row in (a), drawn smaller: pale-red boxes, clean orange frames with a small shield icon in
front of them. Below it: "same bits per weight, no extra storage".

Panel (c), right quarter. Title: "(c) Success at 2 bits".
A simple vertical bar chart with two groups, "Fast-WAM" and "Cosmos-Policy". Each group has three gray bars for
baselines and one orange bar for LAVQ, the tallest in its group, and a dashed black line above the group
labeled "BF16".

Colors: blue-gray #8A99AD for full precision, pale red #C24A3E for 2-bit, orange #D9822B for the future and for
LAVQ, purple #7A5BA6 for the swapped-in future. Thin dashed light-gray lines separate the panels.
No other text, no logos, no people, no 3D, no dark background.
```

---

## 3. 局部修改（同一个对话里接着发，一次只改一处）

- 文字错了：`Fix only this label: it must read exactly "2-bit VQ + BF16 future". Change nothing else.`
- 箭头方向错了：`The purple curved arrow starts at the frames of the first row and points down to the frames of the third row.`
- -73% 的比例不对：`In the third row, the solid red bar is about one quarter of the second-row bar; the dashed empty part covers the rest.`
- 太挤或字太多：`Make it cleaner: more white space between rows, remove any text not in the prompt.`
- 风格太花：`More minimal, like a figure in a CVPR paper: thinner lines, lighter shadows, fewer colors.`
- 想要更活泼：`Add a small sad face next to the long red error bar and a small happy face next to the LAVQ pipeline, same icon style.`（指南里 SIMPACT 那张图就是这样做的；正式版是否保留由你定。）

---

## 4. 复刻时核对（以论文为准）

- **数字**：(a) 是 `1.00 → 0.27`，即 `−73%`（3-bit 时是 85%）。(c) 用 Table 1 的 2-bit 行：
  - Fast-WAM：GPTVQ 71.7、QuIP# 72.5、QTIP 73.3、LAVQ 74.6，BF16 77.6。
  - Cosmos-Policy：GPTVQ 72.6、QuIP# 77.3、QTIP 79.9、LAVQ 80.5，BF16 83.5。
- **术语**：reader、generator、LAVQ、QTIP、QuIP#、GPTVQ、BF16、Fast-WAM、Cosmos-Policy 的拼写要和正文一致。
- **(b) 的盾牌**是概念示意，表示 LAVQ 把 future 当作保护目标。不要画成“future 完全无误差”，图上也不要加任何没有出处的数字。
- **预测的 future** 保持抽象的着色帧，不要画成逼真画面。我们还没有导出解码帧。
- **字号**：缩到论文宽度（17.5 cm）后，最小的字不小于 5 pt。
