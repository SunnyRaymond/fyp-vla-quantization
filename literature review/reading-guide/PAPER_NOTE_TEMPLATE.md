# <Paper title>

> **Reading-list role**: Core / Series / Format-family support  
> **Verification**: `verified-full-text` / `verified-metadata` / `official-technical-material`  
> **Recommended effort**: Skim / Core read / Deep read

## 1. Paper identity

| Field | Value |
|---|---|
| Authors | |
| Year / version | |
| Venue / status | |
| Primary source | |
| Code / project | |

## 2. One-sentence takeaway

用一句话说清楚：它解决什么 bottleneck、靠什么 mechanism、得到什么结果。

## 3. Background and prerequisites

- 读前需要会什么。
- 它承接了哪条 technical lineage。
- 关键术语与 symbols。

## 4. Problem

- **Target setting**：training / PTQ / QAT / inference / robot policy learning 等。
- **Bottleneck**：accuracy、latency、memory、data、generalization 或 hardware utilization。
- **Why previous methods are insufficient**：不要只复述 abstract，要指出失效条件。

## 5. Method

### 5.1 System view

先画清 input → representation → transformation/model → output。

### 5.2 Core mechanism

- 核心 equation / objective。
- 各 component 的责任。
- training / calibration / inference flow。

### 5.3 What is actually new

把真正新增的 mechanism 与 standard engineering choices 分开。

## 6. Experiments and main results

| Claim | Evidence | Locator | Caveat |
|---|---|---|---|
| | benchmark / metric / exact number | Table / Figure / Section | |

先确认 baseline、metric direction、evaluation protocol，再读 headline number。

## 7. Limitations

### Authors' stated limitations

- 

### My critique

- **Internal validity**：ablation 是否足以支持 mechanism claim？
- **External validity**：benchmark / robot / model / hardware 能否代表目标场景？
- **Systems validity**：speedup 是否来自 datatype、kernel、batch size 或更宽的 system optimization？
- **Reproducibility**：data、code、checkpoint、hardware 依赖是否公开？

## 8. Why it matters for this project

- 对 VLA / embodied deployment 的直接意义。
- 对 Professor Li 关注的 model compression、latency、edge deployment、software-hardware co-design 的意义。

## 9. How to read it

### 20-minute route

1. Abstract + Figure 1。
2. Method overview。
3. 一张主结果表。
4. Limitations / failure cases。

### 60-90-minute route

1. 补齐 prerequisites。
2. 推导或复述核心 equation。
3. 核对 strongest claim 对应的实验。
4. 读两项关键 ablation。
5. 写下一个你不同意或尚未相信的点。

## 10. Reading questions

1. 这篇论文真正改变了哪个 assumption？
2. 如果去掉最关键 component，作者预期什么会坏？证据足够吗？
3. Headline result 中有多少来自 algorithm，有多少来自 scale / data / hardware？
4. 把方法移到 VLA 时，activation、action head、control frequency 或 embodiment 会带来什么新问题？

## 11. Weekly meeting card

- **Problem**：
- **Key idea**：
- **Best evidence**：
- **Biggest limitation**：
- **Question for the group**：

## 12. Evidence boundary

- `Source claim`：来自原文，可追到 section/table/figure。
- `My interpretation`：跨段落或跨论文推断。
- `Open question`：现有 evidence 不能确认。

