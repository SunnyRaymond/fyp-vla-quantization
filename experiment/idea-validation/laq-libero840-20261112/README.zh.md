# LAQ 标准 LIBERO 闭环（840 局 × 2 模型，EEE 集群）

用户 2026-11-12 要求：把已经调好的 LAQ（W4A4KV4）Fast-WAM 和 Cosmos-Policy 拿去跑标准 LIBERO 闭环，每个模型 840 局，在 EEE 集群上跑，按余额合理用卡。结果用来填论文 Table 1 右侧 LIBERO 列里 LAQ 的 `\XX`。

## 跑的是什么

| 项 | 设定 |
|---|---|
| 基准 | 原版 LIBERO 四个 suite（spatial / object / goal / 10），不是 LIBERO-Plus |
| 局数 | 40 个任务 × 官方初始状态 0–20 = 840 局 / 模型 |
| 编号 | i = 210·suite + 21·task + trial（suite 按上面顺序） |
| 种子 | 环境 800000000 + 1000·i，采样器 900000000 + 1000·i + replan（与 LIBERO-Plus 闭环同一规则） |
| 回合 | 30 步空动作 settling；步数上限 400，libero_10 为 700；Fast-WAM 每次执行 10 步、Cosmos 16 步后重新规划 |
| 臂 | 只有 LAQ。模型、W4 权重（QuaRot 的 GPTQ bank）、分配、钩子、在线 gaze、CUDA graph 与面板 A 闭环完全相同，代码直接用 ASPIRE2A 上面板 A 闭环作业 artifacts 里的那一份（Fast-WAM 26198651，Cosmos 26204098） |

后面用户跑 BF16 和基线时用同一个 `protocol.json`，就能逐局配对。

## 与 ASPIRE2A 的差别：跨 GPU 的启动检查

ASPIRE2A 上每个 anchor 都是逐位相等，但换了 GPU 型号以后 cuBLAS 选的 kernel 不同，逐位相等做不到。所以启动时改成两条门限（`protocol.json` 的 `gates`）：

- 这张卡上 BF16 的 anchor 与 A100 上 BF16 anchor 的偏差，不超过 A100 上 LAQ 损伤的一半（Fast-WAM 比前 10 步 motor RMSE 和 future NMSE，Cosmos 比 action 相对 RMSE 和 future-image NMSE）；
- 这张卡上 LAQ 相对本卡 BF16 的损伤，是 A100 上 LAQ 损伤的 1/3–3 倍。

同一个 GPU 型号内部仍然要求逐位相等：每个型号的第一个 worker 存下 LAQ anchor，之后的 worker 必须逐位复现，eager 和 CUDA graph 回放都要。

## Fast-WAM 的 T5 提示

A16 准备阶段的 T5 缓存（`fastwam-a16-baselines-20261028/results/prep/t5_cache.pt`）只有 LIBERO-Plus 变体的提示（带 `view …`、`add 16` 之类后缀）和标定提示，标准 LIBERO 的 40 条原始提示只有 1 条在里面。`t5prep840.py`（作业 1735438，6000 Ada）用模型自己的 umt5-xxl 编码了其余 39 条，每条编码两遍并要求逐位相同（与 A16 准备同一规则），存在 `results/fastwam/t5_libero40.pt`；已有的那 1 条保留 A100 的条目。那一条在 6000 Ada 上重编码与 A100 的差：max 0.039，相对 RMSE 2.6%（BF16 T5 跨架构的正常幅度）。后面在这个协议下跑 BF16 和基线时，应该用同一个 `t5_libero40.pt`。Cosmos 的合并缓存本来就覆盖全部 40 条。

## 资源

EEE 每张 GPU 只给 4 个 CPU；6000 Ada 每卡 64 GB 内存，Pro6000 每卡 33 GB。所以：

- Fast-WAM（每个 worker 显存约 14 GiB）放 6000 Ada，每卡 2 个 worker；
- Cosmos（每个 worker 约 5 GiB）放 Pro6000，每卡 2 个 worker；
- 同一作业内模型构建串行（`build.lock`），避免两个 worker 同时加载把内存撑爆。

ug QoS 每人最多同时跑 2 个作业，每种 GPU 最多 2 张。Fast-WAM 主作业用 ug（6000 Ada ×2）；Cosmos 改走 killable，因为用户同时排着一个 ug 的 Pro6000 ×2 训练作业（`fwcg_train`），Cosmos 若占 ug 的 Pro6000 会和它抢同一份额度。`override-limits-but-killable` 不计费、最多 8 张，但可能被抢占后重排。每局单独落盘，claim 带作业号和重启次数，被抢占或重排后的作业启动时会清掉自己和已结束作业留下的 claim，所以可以随时加或撤 killable 作业。GPU 利用率和显存每 60 s 写进作业的 `job.log`（`GPU_SAMPLE`），CPU 和内存每 120 s 一次。

## 文件

| 文件 | 作用 |
|---|---|
| `aspire_pack.pbs` / `relay.py` / `unpack.sbatch` | ASPIRE2A 打包 → 本机中转 → EEE 校验解包（代码、分配、W4 bank、A16 anchor 与 T5 缓存，约 4.8 GB） |
| `probe.sbatch` | 单卡 10 分钟探测：节点 CPU / 内存、两个环境在 Ada 和 Blackwell 上能否跑 CUDA |
| `cputest.sbatch` / `test840.py` | CPU 作业：文件、导入、清单（840 行）、两套 T5 缓存覆盖 40 条指令、原版 LIBERO 环境在初始状态 0 和 20 上 settling、claim 回收逻辑 |
| `run840.sbatch` / `stack840.sh` / `slots840.sh` | GPU 作业：主机侧监控与容器内 worker 槽位 |
| `fw840.py` / `cos840.py` | 两个模型的闭环 worker |
| `l840.py` / `analysis840.py` | 编号、claim 与汇总（`summary.json`、`RESULTS.zh.md`） |
| `control.py` / `poll.py` | 本机：上传、提交、状态、拉取；后台监视 |

## 命令（项目根目录）

```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/laq-libero840-20261112/control.py status
```

```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/laq-libero840-20261112/control.py pull
```

远端结果：`/projects/yguo017/laq840/results/<model>/`（`manifest.json`、`episodes/<i>.json`、`startup/`、`workers/*/startup.json`、`summary.json`、`RESULTS.zh.md`）。

## 结果（2026-11-12 完成）

两个模型都跑满 840/840 局，全部在 RTX 6000 Ada 上完成。每局只跑过一次，没有失效局。每个产出过结果的 worker 都通过了跨 GPU 门限，同型号内的逐位 anchor 也都复现了。

| 模型 | 成功 / 局数 | 成功率 | ±95%（Wald） | Spatial | Object | Goal | Long |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fast-WAM LAQ | 825 / 840 | 98.2 | 0.9 | 98.1 | 99.5 | 98.1 | 97.1 |
| Cosmos-Policy LAQ | 829 / 840 | 98.7 | 0.8 | 98.6 | 100.0 | 98.6 | 97.6 |

失败局全部是达到步数上限（400 步，libero_10 为 700 步），没有崩溃。Fast-WAM 15 局，Cosmos 11 局。各任务的成功数见 `pulled/results/<model>/RESULTS.zh.md`；逐局记录在 `pulled/results/<model>/episodes/`。

**Fast-WAM**
- 作业 1735440（ug，6000 Ada ×2，4 个 worker），用时 4 h 31 min。
- policy call 中位数 3.08 s，每局平均 76 s。

**Cosmos**
- 主要跑在两个 killable 作业上：1735476 和 1735479，各 1 卡。两者共被抢占重排 7 次，claim 回收正常。
- 剩下的局由 ug 作业 1735477（6000 Ada ×2）补完，用时 46 min。
- policy call 中位数 2.32 s，每局平均 55 s。

计费约 2,550 SU，按 6000 Ada 每 GPU 分钟 4 SU 算，killable 作业不计费。

有 3 个 Cosmos worker（1735476.1 的两个、1735479.3 的 worker 1）没有 `startup.json`：它们在启动检查完成前就被抢占，没有产出任何一局。
`control.py pull` 拉取前会先 stat 远端文件；远端没有的文件不会在本地留下 paramiko 生成的空文件。
