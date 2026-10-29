# 标量 W4A8 基线：GPTQ / QuaRot+GPTQ / SpinQuant+GPTQ（held-out 40×2）

## 目的

补齐论文主表里"同预算"对比组：权重 4 bpw 存储、A8 激活、INT8 计算，与我们的 VQ（1b，4.02 bpw）同类。先测固定输入 held-out 精度，最强的一两个臂再上 LIBERO-Plus 闭环。

## 臂

| 臂 | 坐标 | 权重 | 激活 |
|---|---|---|---|
| GPTQ_W4A8 | plain | GPTQ | A8 |
| QuaRot_GPTQ_W4A8 | 固定 randomized signed block-Hadamard R128 | GPTQ | A8 |
| SpinQuant_GPTQ_W4A8 | 同一初始化 × 学到的 Cayley R128 | GPTQ | A8 |
| RTN_W4A8、QuaRot_RTN_W4A8、SpinQuant_RTN_W4A8 | 同上 | absmax RTN | A8 |
| W16A8 | plain | BF16 | A8 |

- **旋转**：直接复用 [fastwam-rotation-baselines](../fastwam-rotation-baselines/PREREG.zh.md) 的 `rotation.py` 和 SpinQuant checkpoint（job 25726472，step 200，在 W4A4 RTN 下学到），每个 video/action/proprio stream 一个 R128，局部右乘 `xR`、`WR`。W4A8 行与已有 W4A4 闭环行（QuaRot 195、SpinQuant 203）旋转完全相同，差别只有 A8 与 GPTQ。
- **权重**：对称 INT4（±7）、group 128、BF16 scale，4.125 bpw。RTN 与 W4A4 基线的量化器相同。
- **GPTQ**：Hessian = C80 上 80 次 query（seed index 0）所有调用的 BF16 Linear 输入 `XᵀX` 的均值，在原坐标收集一次，旋转臂用 `RᵀHR`。damp 1%，block 128，不开 act-order。每个 group 起点在已更新的权重上做 MSE clip 搜索（shrink 1→0.8，21 档，误差范数 2.4）。逐层非顺序（所有层的输入都来自 BF16 模型）。
- **激活**：per-token `absmax/127`，与 1b 的 A8 相同。
- **评测**：held-out 40 context × 2 seed，复用缓存的 BF16 teacher，指标与 1a/1b 相同。

## 流程（单个 g1 作业）

1. 加载模型；检查：两种旋转在所有 7 种输入宽度上正交（≤1e-4）；SpinQuant 与 QuaRot 矩阵确实不同（≥1e-3）；`xR (WR)ᵀ = xWᵀ`（相对误差 ≤1e-4）。
2. 一次探测 query 测 peak 显存，按剩余显存（留 6 GB）把 614 层分成若干 pass；每个 pass 跑 80 次 C80 query 收集 FP32 Hessian，随后逐层做三种坐标的 GPTQ，codes 打包成 nibble 写盘（`results/gptq/<coord>/<idx>.pt`），并记录 RTN / GPTQ 的输出能量误差代理。
3. 七个臂依次 held-out 评测。

全部可续跑：已写盘的 GPTQ 层、endpoint 和 heldout 结果都会复用。

## 本地检查

```bash
python -B test_scalar_quant.py
```

GPTQ 在相关输入上误差低于 RTN；H = I 时 GPTQ 退化为 MSE-clip RTN；nibble 打包可逆；两种 R128 在 7 种宽度上正交。

## 命令（项目根目录）

```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-w4a8-scalar-baselines-20261028/control.py upload
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-w4a8-scalar-baselines-20261028/control.py submit
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-w4a8-scalar-baselines-20261028/control.py pull
```

## 输出

`results/summary.json`、`RESULTS.zh.md`、`checks.json`、`hessian_plan.json`、`gptq_summary.json`、`heldout/<arm>.json`。
