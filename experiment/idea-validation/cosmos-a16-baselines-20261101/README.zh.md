# Cosmos-Policy A16 基线：GPTQ / GPTVQ / QuIP# / QTIP，W3 与 W2，LIBERO-Plus 840 闭环

目的：把 Fast-WAM 主表上的四个 weight-only 基线原样搬到第二个 WAM（`nvidia/Cosmos-Policy-LIBERO-Predict2-2B`），
给论文的跨模型结果用。BF16 与 GPTVQ / QuIP# / QTIP 的 W3、W2 各 840 局（面板 A+B+C）；GPTQ 的 W3、W2 只跑面板 A
（280 局）：它在 Fast-WAM 的 W2 上崩到 15/280，也不是主要对比对象。

## 口径（与 Fast-WAM A16 基线一致的部分）

- 激活全 BF16，只量化权重；各方法不做微调；总 bpw ≈ bits，scale、zero、码本、符号向量、padding 全部计入。
- 方法实现：直接复用 `fastwam-a16-baselines-20261028/methods/` 的四个官方移植（commit 与偏差见各自 `UPSTREAM.md`），
  每层 seed = 20261101 + 层号。唯一改动是让每个方法同时返回原生码：GPTQ 的整数码 + 每行 scale/zero，
  GPTVQ 的码本 + uint8 索引（在 upstream `lut_m_step` 里留下最终码本，标了 `[PORT]`），QuIP# 的官方 attr
  （E8P Qidxs、SU、SV、Wscale），QTIP 的 packed trellis + SU/SV/Wscale/LUT（与 `ravq_qtip` 同格式）。
  GPTQ 与 GPTVQ 在量化时就检查"码解出来逐位等于反量化权重"。
- Hessian：每个目标 Linear 的 BF16 输入 `mean x xᵀ`，C80（case 2000–2079，seed index 0）80 次 get_action
  的全部 5 次 denoiser 调用，FP32、关 TF32，所有方法用同一份。
- 闭环：反量化成稠密 BF16 权重，原地写进 280 个目标 Linear，走官方 `get_action`（5 NFE、chunk 16 执行 16、
  30 步 settling、400/700 步上限）。运行时就是 BF16 面板 A 旧作业 `25838062` 用的 `cosmos_runtime.CosmosRuntime`。

## Cosmos 特有的选择

- **量化范围 = 280 个 block 内 Linear**：28 个 DiT block 的 self/cross-attention q/k/v/output_proj 和
  mlp.layer1/layer2，共 1.762 B 权重。patch embed、final layer、time embedding、adaLN modulation 等
  conditioning 投影保持 BF16——和此前所有 Cosmos 实验（c8-w4a4、vqwam）的 target scope 相同。
  这与 Fast-WAM 的 614 层（含 time_projection 等接口层）口径不同，论文表里要注明。
- **面板与 seed**：面板 A/B/C 与 Fast-WAM 840 完全相同的 840 个 LIBERO-Plus 变体；env seed
  800000000+1000·index，sampler seed 900000000+1000·index+replan（Cosmos 运行时的默认规则，panel C 也按这个规则选的 index）。
- **复用**：BF16 面板 A 直接用旧作业 `25838062` 的 280 局（233/280）。前提是 prep 作业把面板 A 第 0/70/140/210 局
  的第一次 get_action 重放出来，与旧记录的 `first_action_chunk` 逐位相同；不相同则闭环作业自动给 BF16 加跑面板 A。
  T5 也复用：旧缓存（面板 A 用的那份）+ vqwam 缓存（C80 描述），只为面板 B/C 新出现的指令用同一个
  T5-11B 编码器补算。

## 作业结构

| 作业 | 队列 | 内容 | 依赖 |
|---|---|---|---|
| `cabct` | CPU（q 队列） | `test_cosmos_a16.py`：编译/导入、面板与 T5 缺口、四方法×两位宽小层存码解码一致、分片、调度、汇总 | — |
| `cabt5` | gdev 1 卡 | `prep_t5.py`：合并 T5 缓存并补算 B/C 指令 | — |
| `cabhs` | gdev 1 卡 | `prep_hess.py`：280 层 master、C80 Hessian、BF16 anchor（跑两次要逐位相同）、面板 A 重放检查 | — |
| `cabgq` | gdev 1 卡 | GPTQ W3 + W2 | cabhs |
| `cabqt3` / `cabqt2` | gdev 1 卡 | QTIP W3 / W2 | cabhs |
| `cabqs3` / `cabqs2` | gdev 1 卡 | QuIP# W3 / W2 | cabhs |
| `cabgv3a/b`、`cabgv2a/b` | gdev 1 卡 | GPTVQ W3 / W2，各拆两片 | cabhs |
| `cabloop` | g3 4 卡，16 h | 全部闭环 + `analysis.py` | 以上 11 个全部 afterok |

量化作业逐层续跑，到 soft deadline（6900 s 前 600 s）还没做完就以 3 退出，依赖它的闭环作业会被 PBS 撤掉，
补交该量化作业再交闭环即可。

闭环作业：每卡 5 个 worker 槽（旧 BF16 episode 单次 query 峰值 6.2 GB），共 20 槽；调度沿用 panel C 的
`sched.py`（每槽挑"未认领局数/在跑 worker"最大的臂，加入已有 worker 的臂要求每 worker 至少还剩 6 局）。
每个 worker 启动时：BF16 模型在合并 T5 缓存上复现 prep 的 BF16 anchor（逐位），装上该臂权重后取 anchor，
同臂后启动的 worker 必须与第一个逐位相同；anchor 相对 BF16 的 action 误差和 future NMSE 写进 `startup.json`。
任务在 settling 内完成的变体记为无效、所有臂一起剔除；其他未完成的 episode 释放认领后让 worker 退出，
同一槽第二次失败才写 FAILED 停掉整个作业。每局单独落盘，重交作业从断点继续。

## 运行记录

- 2026-11-01：T5 `26079891`、prep `26079892`、9 个量化作业（`26079893`–`26079916`）全部 F/Exit 0，
  每个量化作业 6–46 min。面板 A 重放 4/4 逐位相同（`reuse_check.json` passed），BF16 面板 A 复用旧的 233/280。
- 闭环 `26079917` 8 min 后 Exit 64，没有跑出任何 episode，两个原因：
  1. **T5 缓存**：官方 `init_t5_text_embeddings_cache` 在一个 30 s 超时的文件锁下读 1.8 GB 的合并缓存，并把 880 条全部
     搬上 GPU。20 个 worker 排队，多数超时后只打一行 warning 就带着空缓存继续，第一次 anchor 查询就 cache miss。
  2. **显存**：容器启动慢，每卡 5 个 worker 在同一时刻建模型，每个建模峰值约 8 GB（含 1.8 GB T5），叠加超过 39.5 GiB。
     旧 c8 作业是每卡 1 个 worker，没有验证过 5 个。
- 修复（`loop_worker.py`）：每个 worker 自己把合并缓存读到 CPU，运行时只用一条 anchor 描述的小缓存文件建；每局开始前
  只把这一局的指令搬上 GPU（同一个张量、同样的 `.to(device)`，值与官方加载器逐位相同）。建模和启动 anchor 在
  `/dev/shm` 的每卡独占 `flock` 下进行，结束后 `empty_cache()` 再放锁，同一张卡任何时刻至多一个 worker 在加载。
  `startup.json` 记录等锁时间、anchor 查询的 reserved 峰值和放锁后的 reserved。CPU 测试加了 840 个 slot 的 T5
  staging 检查和两个独立容器之间的锁互斥检查；闭环作业 afterok 依赖这次 CPU 测试。
- 复测 CPU `26080993` 全部通过（含 `LOCK_TEST PASS`）。闭环 `26080994` F/Exit 0，6 h 47 min，无重试、无 FAILED、
  无效槽位 0。每个 worker anchor 查询 reserved 峰值 5.21 GiB，每卡 5 个 worker 约 29.5 GiB；所有臂的 BF16 anchor
  与同臂 anchor 全部逐位一致。

## 结果（`pulled/results/RESULTS.zh.md`）

| 臂 | bpw | 成功 | A / B / C | 对 BF16 胜/负 | p |
|---|---:|---:|---|---:|---:|
| BF16 | 16 | 701/840 | 233 / 225 / 243 | — | — |
| QTIP_W3 | 3.020 | 695/840 | 228 / 225 / 242 | 20 / 26 | 0.46 |
| GPTVQ_W3 | 3.063 | 690/840 | 228 / 226 / 236 | 23 / 34 | 0.18 |
| QuIP#_W3 | 3.013 | 688/840 | 225 / 229 / 234 | 17 / 30 | 0.079 |
| QTIP_W2 | 2.020 | 671/840 | 223 / 218 / 230 | 21 / 51 | 5.3e-4 |
| QuIP#_W2 | 2.013 | 649/840 | 216 / 211 / 222 | 20 / 72 | 4.6e-8 |
| GPTVQ_W2 | 2.063 | 610/840 | 203 / 197 / 210 | 23 / 114 | 1.1e-15 |
| GPTQ_W3 | 3.008 | 215/280 | 215 / — / — | 5 / 23 | 9.1e-4 |
| GPTQ_W2 | 2.008 | 0/280 | 0 / — / — | 0 / 233 | 1.4e-70 |

- W3：三个 VQ/格点方法都与 BF16 无显著差异，彼此也无显著差异（p 0.43–0.90）。
- W2：三者都显著低于 BF16，且两两显著分开：QTIP > QuIP#（47/25，p 0.013）> GPTVQ（69/30，p 1.1e-4），
  顺序与逐层误差代理（0.0036 / 0.0049 / 0.0056）一致。Fast-WAM 上 W2 三个方法打平，这一点 Cosmos 不同。
- W2 的损失主要在 libero_goal（BF16 154 → QTIP 139 / QuIP# 124 / GPTVQ 110），GPTVQ_W2 另在 libero_10
  （182 → 154）和 sensor_noise（107 → 81）掉得多；libero_object 几乎不受影响。
- 最强基线是 QTIP_W2（671）。在约 60–80 个不一致槽位下，显著超过它需要净 +18–20 局。

## 命令（项目根目录）

```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/cosmos-a16-baselines-20261101/control.py upload
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/cosmos-a16-baselines-20261101/control.py submit
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/cosmos-a16-baselines-20261101/control.py status
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/cosmos-a16-baselines-20261101/poll.py --until loop
```

## 输出（远端 `results/`）

- `prep/`：`modules.json`、`masters/`、`hessians/`、`bf16_anchor.pt`、`reuse_check.json`、`text_embeddings.pkl`、
  `T5_COMPLETE.json`、`COMPLETE.json`
- `quantized/<arm>/`：`weights/`（稠密 BF16）、`codes/`（原生码）、`stats/`（bits、误差代理、配置）、`COMPLETE.json`
- `loops/<arm>/`：`episodes/<index>.json`、`anchor.pt`、`worker_*/startup.json`
- `summary.json`、`RESULTS.zh.md`：每臂成功数（分面板/suite/扰动维度）、对 BF16 的配对胜负与 McNemar p、
  同位宽方法两两对比、W3 对 W2
