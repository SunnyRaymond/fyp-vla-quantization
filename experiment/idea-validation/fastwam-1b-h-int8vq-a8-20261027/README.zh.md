# 1b：H 坐标 + 整数 codebook VQ + A8，A+Z+K 校准

## 目的

验证路线 3 的目标格式：去掉 U，只用冻结的 Smooth + signs + H128；activation 为 per-token INT8 网格（A8）；权重为 d4、2×256 **整数** codebook 加每行 scale。在这个格式下，用与 Q0 相同的 A+Z+reader-KL 配方校准，测 held-out 固定输入精度，并导出 engine 可用的 bundle。主臂只有这一个，消融以后再做。

## 格式

| 部分 | 设定 |
|---|---|
| 坐标 | 复用 C80 校准 banks 里冻结的 Smooth0.5 scale 和 signs，后接 signed H128；**不用 U** |
| Activation | `scale = absmax/127`，round half-to-even，clamp ±127，BF16 输出；backward 用 identity STE（与现有 A4 同构） |
| 权重 | 每层两本 256×4 整数 codebook，每个输出行一个 BF16 scale；同层任意一对码字之和落在 [−127, 127]；**不用 B32 指数** |
| 存储兼容 | 仍用 `micro_codebook` 编码：books 存 BF16 整数值，exponent 全部写 127，现有解码器无需修改即可运行 |

格式本身约 4.0 bpw（indices 4.0 + codebooks + 行 scale）。bundle 里那一组全 127 的 exponent 只是为了兼容现有解码器，额外占 0.25 bpw 的存储；`fit_stats.json` 里的 `format_bpw` 不计入它。

## 拟合（prep 阶段）

1. 原 BF16 权重换到 H 坐标，activation 用 A8（即 W16A8），在 C80 的 80 个 context 上收集每层 A8 operand 的逐通道二阶矩：先在每个 context 内平均，再对 80 个 context 等权平均。
2. 每层在“除以行 RMS”后的 d4 向量上做加权 2×256 additive k-means。度量与 `micro_codebook.fit_vq` 相同（二阶矩 × scale²），只是 scale 只有行一级。
3. 每层取一个格点步长 `δ = (max|C0| + max|C1|) / 127`，两本 codebook 都取整到 δ 的整数倍；若取整后两本的界之和超过 127，就放大 δ 重新取整。然后在整数格点上对全部向量重新分配 indices。行 scale = 行 RMS × δ。
4. 每层保存：编码、连续 codebook（作为训练 master）、δ、两本的界、weight NMSE、实际用到的最大码字和。

## 训练（train 阶段）

直接复用 loss-pair 的 `Work.train()`：一次 β 准备、8 次更新 × 10 个 context（与 Q0 同一组 batch）、Adam lr 5e-4、grad clip 1、第一次更新后清空 moments、固定取 final8、冻结 BF16 reader 的分阶段 VJP、最终 anchor 和 formal infer 一致性检查。只替换三处：
- banks：`IntBanks`，从 `TrainableBanks` 继承 snapshot/restore/refresh/gradient_report；
- `load_model`：去掉 U，换成 `IntBanks`；
- `common_endpoint`：改为 prep 阶段保存的 case 2000 初始化 anchor，训练开始前先重放并比对，误差需 ≤1e-6。

- **λ** 沿用 0.04245452683169643；**β** 按冻结规则在新初始化的完整 C80 上重算。
- **STE**：master 以“整数格点单位 / 250”参数化（初始化 = 连续 codebook / (δ × 250)），前向存储值为 `clamp(round(250 × master), ±bound)`，master 的梯度 = 250 × 整数 codebook 的梯度；行 scale 用 log 参数，前向 cast 成 BF16。
- 训练结束后：跑 held-out 40×2；导出 `results/export/inference_bundle.pt` 和 `manifest.json`。engine 启动比对用的 anchor 是 `results/arms/reader_kl/anchor.pt`（case 2000、seed index 0、eager A8 下的 action 与 future）。

## 作业与耗时

| 阶段 | 作业名 | 队列 | 资源 | 预计 |
|---|---|---|---|---|
| prep | `fw1bprep` | gdev | 1 卡，2 h（软截止 1 h 50 m） | 40–60 min：加载约 5 min、二阶矩 80 query、614 层拟合、anchor、held-out 80 query |
| train | `fw1btrn` | g1 | 1 卡，5 h（软截止 4 h 45 m） | 约 3.4 h：Q0 的 reader_kl 臂每次更新约 1300 s，加上 β 准备、held-out 和导出 |

两个阶段都可续跑：
- prep 中途中断，已保存的 `moments.pt` 和逐层 bank 文件会直接复用；
- train 中途中断，从 `checkpoint.pt` 续训，已写出的 held-out endpoint 也会复用。

## 输出

- `results/stages/{prep,train}/summary.json`、`progress.json`；train 阶段另有 `RESULTS.zh.md`（Q0 参照、VQ0′、1b 三行对比）。
- `results/fit_stats.json`：逐层 δ、bound、NMSE、bpw 汇总。
- `results/heldout/VQ0_prime.json`（未校准）、`results/heldout/H_INT8VQ_A8_KL.json`（校准后）。
- `results/export/manifest.json`。

## 命令（在项目根目录执行）

```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-1b-h-int8vq-a8-20261027/control.py --stage prep upload
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-1b-h-int8vq-a8-20261027/control.py --stage prep submit
```
prep 完成后：
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-1b-h-int8vq-a8-20261027/control.py --stage train upload
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-1b-h-int8vq-a8-20261027/control.py --stage train submit
```
查看与取回：`--stage prep|train status` 和 `--stage prep|train pull`。train 启动时会检查 prep 的 summary；如果过早提交，作业会立即失败，不会产生任何结果。

## 已知问题

- **整数格点比 BF16 粗，所以 codebook 按格点单位参数化。** δ 约为归一化单位的 0.03，若 master 用归一化单位，Adam lr 5e-4 下 8 步最多移动约 0.004，几乎没有条目能跨过取整边界。改为格点单位 / 250 后，同样 lr 下每步约移动 0.125 个整数，8 步最多约 1 个格点，与 Q0 每个条目约移动 1 个 BF16 ulp 的幅度相当。`codeword_changes` 报告实际改变的条目数。
- **界的取法偏保守。** 规则是两本各自的界之和 ≤127，保证任意码字对都在 INT8 范围内，代价是格点稍粗。按实际用到的码字对取界，再配合饱和加法，可以更细，但训练时可能越界。
- **bundle 格式名为 `fastwam_1b_h_int8vq_a8_inference_v1`。** 它没有 rotations，activation 为 8 bit。现有 `export_runtime._validate_header` 只认 Q0 的 bundle，engine 评测器需要单独接受这个格式。
