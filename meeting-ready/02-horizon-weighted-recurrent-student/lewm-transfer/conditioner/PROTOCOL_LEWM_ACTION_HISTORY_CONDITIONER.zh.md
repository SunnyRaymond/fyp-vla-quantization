# LeWM PushT：Horizon-Weighted Recurrent Student Action-History AdaLN Conditioner 增量实验协议

状态：`frozen / one bounded GPU job / predictor-level only`

## 目的与假设

上一轮 `LeWMCompactRecurrentTransitionStudent` 在 latent MSE/cosine 上表现很好，但
held-out candidate ranking 很差。此次只增加一个轻量 action-history conditioner，测试
是否能恢复 action-conditioned local geometry。不是扩大 hidden size，也不是引入
attention 或 planner 改造。

官方 LeWM 源码 `module.py::ConditionalBlock` 的真实机制是：action embedding `c`
经 `SiLU -> Linear(192, 6*192)` 生成 shift/scale/gate，最后一层 zero-init，作用为
`norm(x) * (1 + scale) + shift` 后进入 attention/MLP residual branches。增量 student
仅保留其中可迁移的 feature-wise affine 部分：每一步将 causal latest-3 packed action
history `[3,10]` 展平，经 `Linear(30,128) -> SiLU -> Linear(128,512)` 产生
`[shift, scale]`，调制 baseline 原有 affine `LayerNorm` 的输出；baseline 的
`LayerNorm(256) -> Linear(256,1024) -> GELU -> Linear(1024,256)` 模块、参数初始化
和 shared residual 路径保持不变。最后一层 zero-init，要求 zero-init 时在同一 state
dict/base parameters 下与 baseline forward 完全相等；conditioner 与 recurrent cell
在五个 horizon 间共享。treatment 预期参数量为 `845,888`（baseline `775,872` 加
conditioner `70,016`）。

## 公平对照

- baseline：冻结 `24564619.pbs101` 的原始 h256 horizon-weighted recurrent student；
- treatment：同一 latent/action teacher target rows、同一 context manifest、同一
  initialization/training seeds、1500 updates、snapshots、16 held-out blocks × 300
  candidates、same CUDA latency boundary；
- treatment 只允许增加 conditioner 参数，hidden size 仍为 256；student 仍 goal-free、
  encoder-free、teacher-free、无 teacher forcing；
- 为避免重新生成 slate/targets 带来差异，GPU job 优先读取原 baseline 的
  `prepared_rows.pt`。

## 必须报告

step 1500（并保留 500/1000 descriptive snapshots）报告：held-out Spearman
median/min、top-30 overlap median/min、relative latent MSE、五个 horizon cosine、
teacher/baseline/conditioned predictor latency、parameter counts、integrity/convergence、
causality 与 predictor feasibility gate。Latency protocol 与 baseline 相同：cached H=1
latent + 5 normalized packed action tokens，batch=300，warmup=3，technical repeats=10，
CUDA synchronize；不含 encoder、CEM、goal encoding、environment。

## 停止边界

只执行 predictor-level Stage A。禁止 official CEM、planner viability、closed-loop
PushT、hidden-size/weight/step/seed/candidate-bank sweep。若 job 出现明确实现错误，最多
修复后重提一次；不以结果为理由扩展实验。
