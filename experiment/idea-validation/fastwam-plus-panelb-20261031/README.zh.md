# LIBERO-Plus 面板 B：对比臂各加 280 个 episode

## 为什么要做

现在每个 A16 臂只跑过一组 280 个变体（面板 A）。在这个规模下，方法与宿主之间只有约 20 个 episode 结果不同，差 4–5 个成功数就判不出显著性。再加一组 280，每臂就有 560 个 episode。主表最终用三个 VQ 宿主合并做配对检验：如果真实提升是每 280 个多成功 5 个，这个检验有 74% 的概率能测出来。

面板 B 选新的变体，而不是在面板 A 上换种子重跑，原因有三：
- 面板 A 已经被反复用来挑方法，面板 B 没人看过，是干净的测试集。
- 同一个变体换种子，结果高度相关；换新变体，样本才真正独立。
- 评测覆盖到 560 个不同的变体。

## 面板 B 怎么来的（`panel.py`）

- 来源是 LIBERO-Plus 官方的 1,400 个变体池：4 个 suite × 7 个扰动维度，共 28 格，每格 50 个。面板 A 从每格取了 10 个。
- 面板 B 在每格剩下的 40 个里再取 10 个。抽样沿用面板 A 的贪心均衡规则（原任务、子类型、任务×子类型、难度），种子改为 20261031 + 格编号。
- CPU 测试会在真实池子上用面板 A 的种子重跑这条规则，必须逐个复现面板 A，以此证明规则是原样搬过来的。
- 槽位编号是 280–559，环境种子和采样种子沿用 `seed_for(index)`，不会和面板 A 重复。
- 如果某个变体在 30 步空动作沉降阶段就完成了任务，记为无效。沉降不涉及策略，所以无效与否对所有臂都一样，这类槽位从每个臂里同时剔除。

## 运行时

与 `fastwam-a16-baselines-20261028` 完全相同：反量化后的稠密 BF16 权重，T5 缓存，condition cache + CUDA Graph。`panel_worker.py` 直接调用 A16 的 `loop_worker.build_runtime`，只换了三处：
- 结果目录指向本实验。
- 权重从 A16 的 `quantized/` 读取；BF16 臂不装权重。
- 模型加载完以后，先等面板 B 的 T5 缓存写好，再装缓存。

启动检查沿用 A16 的做法，任何一项不通过 worker 就退出：
- 无 T5 的模型配 T5 缓存，必须逐位复现完整 T5 的 BF16 case-2000 锚点。
- eager 和 cache + Graph 的输出必须逐位一致。

`t5_prep.py` 用完整 BF16 模型对面板 B 的每条指令编码两次，要求两次逐位相同。和 A16 缓存重合的指令，要与 A16 的缓存逐位一致；完整 T5 下的锚点要与 A16 的 `bf16_anchor.pt` 逐位一致。

## 作业（g4，五卡两节点，`run.pbs` + `node.sh`）

| 节点 | GPU | worker 槽位 | 启动方式 |
|---|---:|---|---|
| 0 | 4 | 0–7 | `run.pbs` 直接启动 |
| 1 | 1 | 8–9 | `pbsdsh` 启动 `node.sh`，GPU 由 `CUDA_VISIBLE_DEVICES` 或 cgroup 中可见的 GPU 确定 |

一个闭环 worker 占约 14.8 GB 显存（26069268：两个共 29.6 GB），所以每卡最多放 2 个。

**负载均衡（`sched.py`）。**
- 每个槽位挑"加入后每个 worker 平均还剩最多未认领 episode"的臂。
- 加入一个已经有人在跑的臂，前提是该臂每个 worker 平均还剩至少 6 个 episode。重载一次约 5.5 分钟，这条规则保证只在值得时才重载。
- 开局 10 个槽位会按 3 / 3 / 2 / 2 分到四个臂。先跑完的臂，它的 worker 换去剩余最多的臂，各重载一次。
- episode 用 GPFS 上的原子 mkdir（`loops/<arm>/claims/<index>`）认领，两个节点共用每个臂的同一个队列。

**T5 缓存不额外占卡。**
- 槽位 0 先建 T5 缓存，然后转成普通 worker。
- 槽位 1 与它同卡，等缓存建好后才加载。
- 其余 8 个槽位立刻加载模型，加载完在 worker 里等缓存。

**出错时**任何一处都会写 `artifacts/<job>/FAILED`，两个节点的槽位都会停下。episode 逐个落盘，重新提交后从断点继续，上一个作业中断的认领会被 `sched.py init` 清掉。

**监控**：两个节点都每 60 s 记一次 `GPU_SAMPLE`（利用率、显存），每 120 s 记一次 `CPU_SAMPLE`，都写进作业的 job.log。

## 汇总（`analysis_b.py`）

- 每个臂分别给出面板 A、面板 B、合计的成功数。
- 合计对 BF16 的配对胜负和精确 McNemar p。
- 三个 W2 宿主两两配对。
- 按扰动维度和 suite 拆开。
- 面板 B 的 policy call 耗时、episode 平均时长、每个臂起过几次 worker 及各自的加载耗时。

面板 A 的数据来源：量化臂取 A16 闭环的逐 episode 记录，BF16 取 rotation-baselines 的 `episodes.jsonl`（旧运行时）。

## 结果（26070795，g4 五卡，2026-10-31）：W2 差距在 560 上全部显著，三个宿主之间分不出高下

**检查**：10 个初始 worker 的启动检查全部逐位一致。T5 缓存只用 cache 的输出与完整 T5 相同，锚点与 A16 bf16_anchor 相同，eager 与 cache + CUDA Graph 相同。三个量化臂的锚点误差与面板 A 闭环相同（GPTVQ 0.0333124，QTIP 0.0313032，QuIP# 0.0213511）。面板 B 没有无效槽位（280/280）。作业 10:11Z 结束，exit 0。

| 臂 | 面板 A /280 | 面板 B /280 | 合计 /560 | 对 BF16 胜/负 A | B | 合计 | 合计 p |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 216 | 206 | **422**（75.4%） | — | — | — | — |
| QTIP_W2 | 204 | 198 | **402**（71.8%） | 7/19 | 18/26 | 25/45 | 0.022 |
| QUIPSHARP_W2 | 198 | 198 | **396**（70.7%） | 4/22 | 15/23 | 19/45 | 0.0016 |
| GPTVQ_W2 | 201 | 191 | **392**（70.0%） | 10/25 | 14/29 | 24/54 | 0.0009 |

三个宿主两两配对（合计）：QuIP# 对 GPTVQ 31/27（p 0.69），QuIP# 对 QTIP 27/33（p 0.52），GPTVQ 对 QTIP 30/40（p 0.28）。

**读法**
- W2 的损失是真实的：每个宿主在 560 上都显著低于 BF16，差 20–30 个成功数（3.6–5.4 个百分点）。只看面板 B 的 280，只有 GPTVQ 显著，所以单面板 280 不够。
- 三个宿主分不出高下，排序 QTIP ≥ QuIP# ≥ GPTVQ 但都不显著；面板 B 里 QuIP# 反而赢 GPTVQ 22/15。论文里三者作为同一档的 W2 参照。
- 面板 A 略高估了 QuIP# 的损失：面板 A 差 18，面板 B 只差 8（两面板输的占比 0.85 对 0.61，Fisher p 0.052）。另外两个宿主两面板一致。
- 损失集中在视觉分布偏移：camera_viewpoints 上 BF16 46/80，宿主 32–37，两个面板、三个宿主都低于 BF16，是最稳定的信号。language_instructions 和 light_conditions 上宿主不低于 BF16。按 suite 看，掉在 libero_goal（94 对 78–88）和 libero_10（106 对 96–97）。
- 面板 B 的结果不一致对数比面板 A 多：对 BF16 合计 125 对 87，宿主之间 112 对 76，两者 p 都约 0.01。宿主之间的对比在两个面板上走的是同一套 A16 运行时，所以这不是面板 A 的 BF16 来自旧运行时造成的，而是面板 B 里处在成败边缘的变体更多（BF16 在 robot_initial_states 上 A 34、B 22）。

**对后续评测的意义**：一个新量化器和 QTIP 之间，560 上大约 60–70 对结果不一致（宿主之间实测 10.7–12.5%）。要在 560 上显著赢 QTIP，需要净多 18 个成功（3.2 个百分点），接近 QTIP 到 BF16 的全部差距（20）。就算完全追平 BF16，560 上检出概率也只有约 65%，840 上约 86%。如果新方法和 QTIP 共用取整（不一致率接近 7.5%），门槛降到约净 +14/560。

**调度**：16 次 worker 启动（10 次初始 + 6 次中途加入），与模拟一致。初始启动 810–870 s，其中约 290 s 是等 T5 缓存；中途加入 100–420 s，6 次合计约占总槽位时间的 1.5%。四个臂 policy call p50 都是 0.50 s（量化臂装的是反量化后的稠密 BF16 权重，速度相同）。

结果文件：`pulled/26070795.pbs101/results/RESULTS.zh.md`、`summary.json`。

## 命令（项目根目录）

```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-plus-panelb-20261031/control.py cputest
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-plus-panelb-20261031/control.py submit
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-plus-panelb-20261031/control.py status
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-plus-panelb-20261031/poll.py --interval 300
```
