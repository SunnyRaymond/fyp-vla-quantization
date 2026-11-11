# 原始配置基线：QuaRot / SpinQuant / SVDQuant / ViDiT-Q，Fast-WAM 与 Cosmos-Policy

目的：Table 1 的四行基线按各方法公开的配置跑，不放进统一设定（统一设定属于消融，见 `../w4a4kv4-baselines-20261110`）。两个模型各跑 LIBERO-Plus 面板 A 的 280 局，与已有 BF16 面板 A（Fast-WAM 216/280，Cosmos-Policy 233/280）同一批变体和种子，逐局配对。

## 每个方法的配置（官方来源）

| 方法 | 权重 | 激活 | KV | 其他 |
|---|---|---|---|---|
| QuaRot（[fake_quant](https://github.com/spcl/QuaRot/tree/main/fake_quant)，论文 §setup） | GPTQ，每行一个对称 scale，MSE 搜 clip | A4 对称，per-token，clip 0.9 | K/V 非对称，per-head，clip 0.95；K 过 Hadamard，V 过 head 维 Hadamard | 固定随机 Hadamard，全尺寸 |
| SpinQuant（[scripts/2_eval_ptq.sh](https://github.com/facebookresearch/SpinQuant)） | GPTQ，同上 | A4 非对称，per-token，`o` 输入按 head 分组 | K/V 非对称，per-head，无 clip | R1（全尺寸）+ R2（每 attention，head 维）；Stiefel SGD，lr 1.5 余弦，100 步，在 W16A4KV4 下学习 |
| SVDQuant（[DeepCompressor int4](https://github.com/nunchaku-ai/deepcompressor/blob/main/examples/diffusion/configs/svdquant/int4.yaml)） | INT4 对残差 RTN，组 64 | A4 动态，per-token，组 64；MLP 第二层输入平移后无符号 | 无 | 每层 grid 搜 smoothing（39 个候选）；BF16 rank-32 低秩分支，最多 100 步迭代校准（早停）；q/k/v 共享输入的层共享 smoothing 与分支的输入侧 |
| ViDiT-Q（[w4a8_mixed_precision.yaml](https://github.com/thu-nics/ViDiT-Q/blob/viditq/examples/pixart/configs/w4a8_mixed_precision.yaml)） | W4A8 混合精度：attention 层 W4，FFN 与首尾两个 block W8，每行仿射，两阶段 | A8 对称，per-token | 无 | 通道平衡 α=0.5665（官方视频配置），随机符号全尺寸 Hadamard |

Fast-WAM 的 14 个接口/时间条件层保持 BF16，四个方法原版都跳过这类层。

## 必须说明的适配边界

- **旋转位置**：AdaLN 和逐维调制使全局残差旋转不成立，所以每个旋转都在 Linear 输入处局部施加并由该 Linear 自身的权重补偿。其余旋转结构（全尺寸、R3、head 维旋转、MLP 隐层 Hadamard）保持。
- **MLP 隐层的 Hadamard**：官方 R2/R4 是不带随机符号的 Hadamard，配合 LLaMA 的 SwiGLU。Fast-WAM 和 Cosmos 的 MLP 用 GELU，输出均值偏正，不带符号的 Hadamard 会把均值能量集中到一个坐标（占一个 token 能量的 4–10%），per-token A4 的 NMSE 从 0.02–0.03 升到 0.24–0.72（`diag_dc.py` 实测）。所以这一层改用带随机符号的形式（与 R1 相同）。
- **KV 位置**：Fast-WAM 量化文本 cross-attention K/V、观测帧 K/V、prefill 缓存；Cosmos 量化全部 K/V。
- **GPTQ 的 Hessian**：沿用 BF16 模型上 C80 的二阶统计，原版是逐层前向累积（上游权重已量化）。
- **校准数据**：统一用 C80 的 LIBERO 上下文，SpinQuant 的目标换成对 BF16 去噪输出的归一化误差（官方是下一 token 的交叉熵）。
- **ViDiT-Q**：官方 OpenSora 配置里的层号（blocks 11–14）依赖其自身的灵敏度搜索，不能直接移植；这里用官方 W4A8 配置里按层类型的规则，没有重跑 metric-decoupled 搜索。α 取官方视频配置的 0.5665，未扫描。

## 作业

| 作业 | 队列 | 内容 |
|---|---|---|
| `cputest` | CPU | `test_published.py` |
| `calib_*` | g1（1 卡） | C80 各层输入样本（SVDQuant 用）；SpinQuant 的 R1/R2 训练 |
| `prep_*` | g1 / g2 | 拟合四个方法，finalize，每个臂一次 anchor 查询（BF16 逐位对齐、覆盖检查） |
| `loop_*` | g3（4 卡） | 闭环，每局单独落盘，重交作业从断点继续 |

GPU 利用率和显存每 60 秒写入 `job.log`。

## 结果

### Cosmos-Policy（作业 26188483，2026-11-11 完成）

LIBERO-Plus 面板 A，280 局配对，BF16 为同一面板、同一种子的参考。p 为对 BF16 的配对 McNemar 精确检验。

| 方法 | W bpw | A | KV | 成功 | 成功率 | ±95% | 对 BF16 胜/负 | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 16 | 16 | 16 | 233 | 83.2 | 4.4 | — | — |
| QuaRot | 4.01 | 4 | 4 | 228 | 81.4 | 4.5 | 6 / 11 | 0.33 |
| SpinQuant | 4.01 | 4 | 4 | 228 | 81.4 | 4.5 | 8 / 13 | 0.38 |
| ViDiT-Q | 6.30 | 8 | 16 | 222 | 79.3 | 4.7 | 5 / 16 | 0.027 |
| SVDQuant | 4.89 | 4 | 16 | 232 | 82.9 | 4.4 | 7 / 8 | 1.0 |

只有 ViDiT-Q 显著低于 BF16。QuaRot 和 SpinQuant 在 W4A4KV4 下只掉 1.8 点，SVDQuant 与 BF16 持平。分 suite 与分扰动维度见 `pulled/results/cosmos/RESULTS.zh.md`。

### Fast-WAM

闭环作业 26188484 在 g3 排队。
