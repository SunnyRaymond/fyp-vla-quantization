# W4A4KV4 基线：QuaRot / SpinQuant / SVDQuant / ViDiT-Q，Fast-WAM 与 Cosmos-Policy

> 2026-11-11：这一版把四个方法放在同一套量化器（统一 GPTQ、统一分组和 KV 量化）下运行，属于消融，不进 Table 1。Table 1 的基线改用各方法的原始配置，见 `../published-baselines-20261110/`。这里的 Cosmos 结果完整，Fast-WAM 只跑了一部分，作业已取消。

原目的：填 Table 1 的这四行基线。两个模型各跑 LIBERO-Plus 面板 A 的 280 局，与已有的 BF16 面板 A（Fast-WAM 216/280，Cosmos-Policy 233/280）同一批变体和种子，逐局配对。

## 口径

| 项 | 设定 |
|---|---|
| 权重 | INT4，每行每 32 个输入通道一个 BF16 scale（4.5 bpw 起） |
| 激活 | 动态 per-token INT4，每 32 个通道一个 scale |
| KV | 所有方法共用同一个 per-token、每 32 通道一个 scale 的对称 INT4（`kv4.py`） |
| 接口层 / 时间条件 | 与此前相同：Fast-WAM 614 层、Cosmos-Policy 280 层的目标范围不变 |
| 校准 | C80（每模型沿用 A16 的 BF16 权重与 Hessian，以及此前的通道最大值统计） |

KV4 的量化位置：

- **Fast-WAM**：两个 expert 的文本 cross-attention K/V；video 去噪循环里观测帧（前 98 个 token）的 self-attention K/V；prefill 写给 action 循环的 video K/V 缓存。
- **Cosmos-Policy**：没有跨步缓存，每个去噪步的全部 K/V（56 个 attention）都量化。

## 四个方法

| 方法 | 变换 | 权重 | 激活 | 额外 |
|---|---|---|---|---|
| QuaRot | 每个 stream 一个固定随机符号 block-Hadamard R128 | 旋转坐标下 GPTQ | 对称 | — |
| SpinQuant | 同一初始化 × 学到的 Cayley R128（沿用旧的 step 200 / update 200，不重训） | 旋转坐标下 GPTQ | 对称 | — |
| SVDQuant | 通道平滑 α=0.5 | 残差 RTN | 对称，作用在平滑后的输入上 | BF16 rank-32 低秩分支 |
| ViDiT-Q | 通道平衡 α=0.5665 + 随机符号全 Hadamard | 仿射两阶段 | 仿射 | — |

与旧 W4A4 基线的差别：组大小 128 → 32；QuaRot / SpinQuant 从 RTN 改成 GPTQ；KV 从 BF16 改成 KV4；SVDQuant 用论文的 rank 32；Cosmos 的 ViDiT-Q 不再有 W8 层。推理全部是解码后的 BF16 权重加 fake-A4（eager），不是原生 INT4 kernel，这里的延迟不用于效率表。

## 作业

| 作业 | 队列 | 内容 |
|---|---|---|
| `w4ct` | CPU | `test_w4a4kv4.py`：量化器、GPTQ、四种拟合与前向、KV4 安装、调度 |
| `w4pco` / `w4pfw` | g1（1 卡）/ g2（2 卡） | `prep.py` 拟合四个方法，再对每个臂跑一次 anchor 查询（BF16 逐位对齐、覆盖检查） |
| `w4lco` / `w4lfw` | g3（4 卡） | 闭环：Cosmos 每卡 4 个槽，Fast-WAM 每卡 2 个槽；`afterok` 依赖各自的准备作业 |

每局单独落盘，重交作业从断点继续。GPU 利用率和显存每 60 秒写入 `job.log`。

## 命令（项目根目录）

```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/w4a4kv4-baselines-20261110/control.py status
```

```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/w4a4kv4-baselines-20261110/control.py pull
```

## 输出（远端 `results/<model>/`）

`banks/<arm>/NNNN.pt`（打包码、scale、通道向量、符号、低秩因子）、`banks/<arm>/COMPLETE.json`（bpw 与误差代理）、`loops/<arm>/episodes/<index>.json`、`summary.json`、`RESULTS.zh.md`（各臂成功率、95% 区间、分 suite、对 BF16 的配对胜负）。
