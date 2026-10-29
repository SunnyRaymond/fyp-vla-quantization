# 阶段 1a：Q0 的 A8 精度检查（固定输入，v2）

固定当前 Q0（`25921795.pbs101 / reader_kl / final8`），在 Fast-WAM held-out 40 contexts × 2 seeds（80 query）上对照缓存的 BF16 teacher，回答两件事：

1. A4 换成 A8 能回收多少误差（以 A16 为下界）；
2. 1b 的目标格式——只用 H（去掉 U）+ 每行 INT8 权重 + A8——在不重新训练时损失多少。

0 训练、0 新 teacher、0 闭环。decoded BF16 权重 + fake-quant activation，BF16 KV。

v1（六臂、g2 双卡，job `26028311.pbs101`）启动 1.5 分钟后按用户要求取消，没有结果；v2 精简为三个新臂，单卡走 gdev。

## 臂

| 臂 | 权重 | Activation | 坐标 | 来源 |
|---|---|---|---|---|
| `Q0_A4` | Q0 | A4 | R0（Smooth/signs/H/U） | **复用** Q-Clip `25937633.pbs101` 的 80 条 Q0 endpoint |
| `Q0_A16` | Q0 | 不量化（BF16） | R0 | 新跑 |
| `Q0_A8` | Q0 | per-token INT8（absmax/127，round half-to-even，clamp ±127） | R0 | 新跑 |
| `Q0_H_INT8_A8` | Q0 解码权重每个完整 128 块右乘 Uᵀ 搬回 H 坐标，再按行 INT8 | A8，只做 Smooth+signs+H | H | 新跑 |

`Q0_H_INT8_A8` 没有重新拟合或训练，是 1b 格式代价的上界。

另有**权重级诊断**（不跑 query）：614 层 Q0 解码权重的 INT8 行取整误差（R0、H 两种坐标）与 VQ 本身误差的比值。若 INT8 误差远小于 VQ 误差，`Q0_H_INT8_A8` 的退化就主要来自去掉 U。

## 来源绑定

- 模型装载与 Q-Clip 相同：loss-pair artifact 的 `worker.py`（作 `q0_worker.py`）+ `parent_protocol.json`，`restore(deployment.pt 的 module_masters)`。
- 启动时用原 Q0 forward 重放 case 2000，与 `reader_kl/anchor.pt` 比较，最大差须 ≤ 1e-6；另做单层随机输入的坐标改写检查。
- 每臂额外跑一次 case 2000，把 eager 的 action/future 存为 `results/anchors/<arm>.pt`，供 engine 版 LIBERO-Plus 评测器启动时逐位比较。

## 输出

`results/endpoints/<arm>/`、`results/arms/<arm>.json`、`results/weight_diagnostic.json`、`results/anchors/<arm>.pt`、`results/summary.json`、`results/RESULTS.zh.md`（主表、相对 Q0_A4、A4→A8 回收比例、权重诊断、逐 context 配对、逐 suite）。

## 运行

```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-a8-accuracy-20261027/control.py upload
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-a8-accuracy-20261027/control.py submit
```

远端 root `/scratch/users/ntu/yguo017/fastwam-a8-accuracy-20261027-v2`（v1 root 保持不动），job `fwa8v2`，`-q normal` → gdev，`select=1:ncpus=16:mem=110gb:ngpus=1`，walltime 01:30:00。GPU 利用率与显存每 15 秒写入 `artifacts/<job>/job.log`。
