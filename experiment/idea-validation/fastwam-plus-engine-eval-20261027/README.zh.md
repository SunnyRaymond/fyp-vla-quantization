# Engine 版 LIBERO-Plus 280 评测器

用推荐 engine（packed VQ 常驻 + 逐层解码 + cuBLAS + condition cache + CUDA Graph）跑历史 280 面板，替代原来每 query 约 6.7 s 的 decoded-dense eager 路径（engine 约 0.9 s）。供 1a、1b 及后续消融反复使用；本目录只是工具，还没有提交过任何臂。

## 复用了什么

| 部分 | 来源 |
|---|---|
| engine 安装、workspace、cache、Graph、decoder 选择 | `fastwam-vq-r128-c80-c-decoder-buffer` 的 `abc_runtime.install(mode="c_buffer")`，改写成 `plus_engine.install`，只多两个开关：activation 位宽、是否乘 U |
| 模型装载、anchor 检查、原子任务队列、常驻 worker | `fastwam-libero-engine-20261027/worker.py` |
| LIBERO-Plus 环境、manifest、seed、settling、步数上限、gripper 后处理 | `fastwam-vq-r128-c80-loss-pair-plus`（`environment.sh` 原样复制；episode 循环与 `vq_adapter.episode` 同构，`runner.Runtime.task/command` 原样调用） |
| 参考臂 | 直接读历史结果：BF16（rotation-baselines）、Q0/A+Z（loss-pair-plus）逐 slot 配对；其余臂总数读 loss-pair-plus 的 summary.json |

engine 依赖里的 `runner.py`、`quantization.py`、`libero_state_loader.py` 与 loss-pair-plus 用的 Plus episode runtime（`25822971`）逐字节相同（2026-10-27 已用 SFTP 核对 sha256）。

## 正确性检查（只保留两道）

1. **启动 anchor 逐位一致**：每个 worker 装好 engine 后，用 case 2000 跑一次，action 和 future 必须与该臂的 eager anchor 完全相等；第一次调用时 614 个模块的 activation operand 还要逐位等于 eager 公式（`plus_engine.reference_operand`）。
2. **首次使用的结果复现**：`Q0_A4_repro` 在 8 个 slot（覆盖 4 个 suite、7 个扰动维度：10/50/70/130/160/170/220/250）上跑 Q0+A4，成败和步数必须与历史 `25925636` 的 `reader_kl` 逐条一致。任何臂第一次提交时，`run.pbs` 会自动先跑它；通过后记录在 `results/Q0_A4_repro/summary.json`，以后不再重跑。

## A8 与 eager 逐位一致的依据

- `_a8_kernel` 是 `fused_input._a4_kernel` 的逐行复制，只把 7 改成 127。A4 kernel 已在历史作业中对 614 个模块逐位通过。
- scale 用乘倒数：PyTorch 的 `张量 / Python 标量` 在 CUDA 上同样是乘倒数（A4 当年就是靠这一点修好的），而 `fp32(1/127) == 1.0f/127.0f`（本地已核对）。
- 舍入方式（`div_rn` + 显式 round-half-even）与 clamp 与 `torch.round(...).clamp(...)` 一致。
- 运行时仍以检查 1 为准：不一致会直接失败。

## 臂表（`arms.json`，可扩展）

每个臂一个条目：`job_name`（≤15 字符）、`bundle`、`activation_bits`（4/8）、`use_u`、`anchor`、`slots`（`"all"` 或列表），可选：
- `allocation`：覆盖默认的 4 卡 / 64 CPU / 440 GB / 2 h / 每卡 2 worker；
- `shard: [k, n]`：只跑 `index % n == k` 的 slot；
- `group`：多个分片或续跑条目写进同一个 `results/<group>/episodes/`，合并出 280 条的总表。

1b 的新 bundle：只要容器格式相同（614 模块、`rotations` 含 action/video，`use_u=false` 时可存单位阵），就能直接用；非默认 bundle 只做通用格式检查，不绑定 Q0 的 source job。整数 codebook 存成 BF16 整数值、exponent 全 127，decoder 无需改动。anchor 需由 eager 路径（同一格式、同一 A 位宽、同一 use_u）生成，内容为 `dict(action, future, case_id=2000, seed_index=0, sampler_seed)`。

## 时间估计与分片

依据 loss-pair-plus 的实测：每 episode 平均约 24.9 次 query、episode 墙钟 241.9 s（其中 query 约 167 s，环境约 75 s，环境是 CPU 渲染，与 engine 无关）。

| 配置 | 每 episode 估计 | 280 条 |
|---|---:|---|
| engine，1 worker/卡，4 卡 | 约 97 s | 约 1.9 h + 装载，**超过 gdev 2 h** |
| engine，2 worker/卡，4 卡（默认） | 约 100 s | 约 1.0 h + 装载 5–10 min，首次使用再加复现约 10 min |
| engine，1 worker/卡，两个 `shard` 作业 | 约 97 s | 每个约 1 h |

默认用每卡 2 个 worker：engine 每进程 peak reserved 约 17.3 GB，两个约 35 GB，A100-40GB 放得下；GPU 每 episode 只忙约 22 s，CPU 渲染才是瓶颈，所以 64 CPU 分给 8 个 worker。若出现 OOM，把该臂改成 `workers_per_gpu: 1` 并拆成两个 `shard` 条目（同一 `group`）。gdev 全局只有 16 张卡，4 卡作业可能要排队。

walltime 到点被杀时，已完成的 episode 都留在 `results/<group>/episodes/`；加一个同 `group`、新 `job_name` 的条目重新提交即可续跑剩余 slot。

## 用法（在项目根目录）

```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-plus-engine-eval-20261027/control.py upload --arm Q0_A8
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-plus-engine-eval-20261027/control.py submit --arm Q0_A8
```
`status` / `logs` / `pull` 同样带 `--arm`。`pull` 取回 `results/<arm>/summary.json`、`RESULTS.zh.md` 与首次复现结果到 `pulled/<arm>/<job_id>/`。`Q0_A8` 需要 1a 固定输入作业先写出 `fastwam-a8-accuracy-20261027-v2/results/anchors/Q0_A8.pt`。

## 输出

- `results/<arm>/worker_<i>/startup.json`：anchor 与 614 模块 operand 逐位检查结果、GPU、复用的 decoder。
- `results/<group>/episodes/<index>.json`：逐 episode 成败、步数、每次 policy call 秒数、seed。
- `results/<arm>/summary.json`、`RESULTS.zh.md`：总表、按 suite、按 7 个扰动维度（附历史参考臂）、与 BF16 / Q0(A4) / A+Z(A4) 的逐 slot 配对胜负、policy call p50。复现臂则是 8 条逐条对照。
