# Fast-WAM action inference 并行研究

两条固定观测 pilot 均已完成。**逐层固定 K/V cache 值得保留**：完整 infer_action 约 643→592 ms，三个观测逐步及最终输出均保持一致。跨层批处理让准备更快，但整段额外收益约 0.4%，增加峰值显存约 581 MiB。**时间并行的显存压力很小**：batch=10 相对原生只增加约 47.4 MiB / 0.20%；但全部 11 个 prefix-Picard 候选均未通过冻结的 numerical gate，当前配方 NO-GO。

所有结论限于 `FastWAMOptionalIDM / first_frame`、A100-SXM4-40GB、BF16、compile=false、10-step、32×7 action；不迁移到未来视频生成的 `idm` 或整个 LIBERO suite。

## 两个不同的并行位置

**固定 context 线**：同一个 action chunk 的文字、proprio 和 observation 不变。每层的 text/proprio K/V 都由固定 context 与该层权重决定，可提前计算；30 层的这部分投影互不依赖，可以合为较大的投影操作。Action hidden states 仍依赖上一层，30 个 transformer blocks 保持原顺序。实验先分开测文本 embedding 复用、逐层 K/V cache，以及跨层投影批处理。

**Denoising 时间线**：十个时间点上的 action states 相互依赖，不能直接把十个原生步骤当成十个独立任务。实验先猜测一个时间窗口内的 action trajectory，同一轮把不同 action 猜测与对应 timestep 放到 batch 维一起评估，再更新整段猜测，重复固定轮数。原生十个离散节点全部保留；付出的代价是更多 scalar NFEs，可能得到更短的串行等待。

## 设计与执行

| 研究线 | 关键对照 | 实际计入的成本 | 初始 PBS job |
|---|---|---|---|
| 固定 context 缓存/跨层批处理 | A0 native、A1 embedding cache、A2 逐层 K/V cache、A3 跨层 K/V 投影；另有 A2/A1 与 A3/A2 直接配对 | 每 chunk cache 构建、packing、完整十步 action；A3 额外 packed weights 与一次构造成本另报 | `25664341.pbs101` |
| Denoising 时间并行 | 原生串行十步对 W=2/5/10 的 11 个固定 prefix-Picard 组合；triangular native-rounding control；原生轨迹 batch probe 单列 | 所有 batch forwards、每轮 trajectory 更新、窗口传递、完整求解；scalar NFE 另报 | `25664342.pbs101` |

两条作业各自在原生 `libero_goal` task0/trial0 episode 捕获相同位置的三个实际 replan contexts。每个候选同观测、同 initial action noise 与原生配对；每 context 三次 warmup，四对交替 AB/BA。计时重复不等于独立任务数量。

缓存线需每步 velocity、updated latent 与最终 action 通过固定 `1e-5` allclose。时间线固定 numerical pilot gate 为 normalized action max-abs 与 relative-L2 都不超过 `1e-3`；进入完整 infer 对照还要求后处理 gripper 符号没有变化。这些只证明当前固定观测下的计算近似，不能代替候选闭环 task preservation。

显存用独立 memory passes 记录 allocated/reserved 基线和峰值。A3 的 packed weights 包含在其绝对显存中；同一 allocator 生命周期使 reserved 带有历史，不能单靠 reserved 排名判断净需求。GPU job log 每 15 秒记录 utilization 与设备 memory sampling。

## 已完成：denoising 时间并行

作业 `25664342.pbs101` 为 F / Exit_status=0，wrapper exit=0 且 `PIPELINE_COMPLETE` 存在。三个实际 replan contexts 为 0/6/12，原生 episode 重放 action 均逐元素一致；summary 共 45 项（每 context 3 个 oracle probes、11 个 prefix 候选、1 个 triangular control）。原生采样 episode 为 1/1 success，只用于取得真实 observations，未运行候选闭环。

### 显存：支持“action batch 增量占整体很小”的判断

以下为 context 0 的独立 memory passes，相对原生串行十步 action 的峰值比较；没有只按间隔 GPU sampling 猜峰值。

| 配置 | PyTorch peak allocated | 相对原生增加 | 整体增幅 |
|---|---:|---:|---:|
| 原生 batch=1 | 23,805.15 MiB / 23.247 GiB | — | — |
| Prefix window=2，R=1 | 23,811.87 MiB | 6.72 MiB | 0.028% |
| Prefix window=5，R=1 | 23,826.14 MiB | 20.99 MiB | 0.088% |
| Prefix window=10，R=1 | 23,852.55 MiB | 47.40 MiB | 0.199% |

时间点放在 batch 维，整套 video/action weights 都共享，未复制模型实例。固定 video/context tensor 通过 expand 视图广播；增加的是 action hidden states、Q/K/V、FFN、部分 mixed K/V 的 materialization 与计算 workspace。在 no-grad 推理里，通常只需当前层的主要临时 activations，不像训练那样保存 30 层反向传播所需的中间值。因此不能把 batch=10 理解为占用十份 action expert 参数，更不能理解为十份完整 Fast-WAM。

这些绝对值包含本 runner 持有的三个 frozen contexts 和模型常驻内容，不作为最小部署显存要求。原生及候选 peak reserved 都约 23.793 GiB，但 allocator 处于同一进程且带有历史；不能由 reserved 不变推断动态内存成本为零。结论只覆盖实测 W=2/5/10、当前 context/token/action shapes 和 backend。

### 延迟和精度：有批处理能力，当前求解配方没有合格加速

Context 0 的示例：原生完整 action core 约 440 ms。把**已经知道的**十个原生中间 states 同时送入模型，约 44.25 ms，对逐个 forward 为 9.97×；这是 oracle batch probe，取得这些 states 的原生成本没有计入，不能作为部署加速。

实际 prefix 求解从窗口左端 action 冷启动，不使用 teacher states。Window=10、R=1 约 44.48 ms / 9.88×，但 relative-L2=2.61、max-abs=3.53；R=5 约 222.21 ms / 1.98×，relative-L2 仍约 0.08；R=10 约 443.68 ms / 0.99×，relative-L2 约 0.00446、max-abs=0.0078125。后者仍未通过 `max_abs <= 1e-3` 且 `relative_L2 <= 1e-3` 的冻结 gate。

三 observations 的全部 33 个 prefix 求解均 gate=false，因此没有候选进入完整 infer_action 的加速比较，也未执行候选闭环。小轮数的快来自减少串行 batch calls，但未充分恢复依赖；足够多轮后串行 calls 回到十次，额外 NFEs 和更新成本使净收益消失。该 NO-GO 针对当前 cold-start prefix/BF16 配方，不证明所有时间并行方法不可行。

全部冻结候选如下。Wall 和 speedup 的括号中为三个 context 的中位数，括号前为 min–max；quality 取三个 context 的最坏值。所有行的 pilot/execution gate 均为 0/3；gripper 数量比较的是完整 32-step chunk，不能因 gripper 相同就接受其他 action 分量的偏差。

| W / R | Wall ms 范围（median） | Paired speedup 范围（median） | Worst max-abs | Worst relative-L2 | Worst 首10步 max-abs | Gripper 差异 C0/C6/C12 |
|---|---:|---:|---:|---:|---:|---|
| 2 / 1 | 220.04–232.94（220.21） | 2.000–2.014（2.001） | 0.218750 | 0.180719 | 0.187500 | 0/0/0 |
| 2 / 2 | 441.66–497.51（454.93） | 0.981–0.996（0.996） | 0.007813 | 0.003315 | 0.003906 | 0/0/0 |
| 5 / 1 | 87.95–104.74（88.19） | 4.846–5.008（4.983） | 1.171875 | 1.011188 | 1.041992 | 0/0/0 |
| 5 / 2 | 175.95–204.86（176.41） | 2.473–2.498（2.491） | 0.835938 | 0.697681 | 0.704102 | 0/0/0 |
| 5 / 3 | 264.00–300.28（269.46） | 1.567–1.667（1.640） | 0.164063 | 0.132235 | 0.135742 | 0/0/0 |
| 5 / 5 | 440.24–497.54（440.84） | 0.988–1.001（0.997） | 0.007813 | 0.003780 | 0.007813 | 0/0/0 |
| 10 / 1 | 44.48–48.74（44.49） | 9.878–10.594（9.882） | 3.531250 | 2.922401 | 2.849609 | 3/3/3 |
| 10 / 2 | 88.90–105.32（89.08） | 4.937–4.950（4.938） | 3.109375 | 2.624007 | 2.587891 | 18/19/18 |
| 10 / 3 | 132.96–158.16（133.23） | 3.303–3.312（3.304） | 1.177734 | 1.108873 | 1.177734 | 0/0/0 |
| 10 / 5 | 221.97–243.11（222.21） | 1.969–1.985（1.979） | 0.109375 | 0.088212 | 0.107422 | 0/0/0 |
| 10 / 10 | 442.91–508.66（443.68） | 0.953–0.991（0.991） | 0.007813 | 0.005220 | 0.007813 | 0/0/0 |

Oracle probes 的 W2/W5/W10 velocity max-abs 都为 0.015625，relative-L2 分别为 0.002118–0.002627 / 0.002385–0.002603 / 0.002844–0.003051，strict allclose 均失败。它们证明当前输入尺度可高效批处理，同时揭示 BF16 batch-dependent 数值差异；不作为 solver 免费初始化或加速证据。

### 数值对照

Triangular W10/R10 保留逐步 action-dtype Euler 更新分组；三 contexts 的 relative-L2 分别为 0.00280 / 0.00189 / 0.00174，max-abs 为 0.0078125 / 0.00390625 / 0.00390625。它仍有 batch-shape-dependent 数值差异：原生轨迹 batch probe 的 velocity relative-L2 也约 0.002–0.003。Prefix W10/R10 的 relative-L2 为 0.00446 / 0.00522 / 0.00521，另受 `left + prefix(increments)` 与逐步 Euler 不同的 BF16 舍入分组影响。

这些对照帮助区分未充分修正和数值实现问题，但没有做完整误差成分的定量归因。不能把所有偏差都解释为“再多迭代几轮就会消失”，也没有降低 gate 来接受结果。

## 已完成：固定 context 缓存与跨层批处理

重跑作业 `25664452.pbs101` 为 F / Exit_status=0，wrapper exit=0 且 `PIPELINE_COMPLETE` 存在。Summary 共 30 项，计时 CSV 240 行；全部三个 contexts 的原生重放一致。三种主要 cache 条件的每步 velocity、updated action latent 和最终 action 最大绝对误差都为 0；完整 infer_action 输出也 strict allclose=true。没有执行候选闭环。

### 主要结果

下表 ms 为三个 context 各自 median latency 的中位数；speedup 为各 context 的 paired-speedup median 再取中位数。它们并非同一个统计量，不能用表中 rounded ms 的比值替代 paired speedup。主要 action phase 包含每 chunk cache prepare；一次性 packed weights 构造另列。

| 对原生 A0 的候选 | Action phase 原生→候选 | Action phase paired speedup | Full infer_action 原生→候选 | Full paired speedup |
|---|---:|---:|---:|---:|
| A1 text embedding cache | 541.82→537.36 ms | 1.0016× | 641.15→642.25 ms | 0.9990× |
| A2 逐层固定 K/V cache | 533.88→480.67 ms | 1.1104× | 642.77→591.82 ms | 1.0864× |
| A3 跨层 K/V projection + cache | 532.19→477.88 ms | 1.1160× | 640.46→586.41 ms | 1.0921× |

三 contexts 的 action speedups：A1=1.00835/1.00164/1.00023，A2=1.11105/1.11039/1.11016，A3=1.11617/1.11603/1.11192。A1 的 context12 两臂同受较高时延影响，不把不同候选组的绝对 native ms 当作共同恒定基线。缓存线约534 ms与时间线约440 ms来自不同 PBS/GPU 作业；处理效应只用各自作业内的配对比较。

按三个 context 的 paired-speedup medians 做描述性线性插值，p10/p90 如下；样本只有三个同 episode observations，以下不是置信区间。

| 条件 | Action speedup p10 / p90 | Full infer speedup p10 / p90 |
|---|---:|---:|
| A1 | 1.000512 / 1.007007 | 0.998424 / 0.999464 |
| A2 | 1.110210 / 1.110917 | 1.083598 / 1.087800 |
| A3 | 1.112738 / 1.116140 | 1.091868 / 1.092348 |

### 直接消融：缓存有效，跨层 batching 的额外整段收益小

| 直接配对 | 参照→候选 median ms | Paired speedup |
|---|---:|---:|
| Action phase，A2 对 A1 | 532.72→480.34 | 1.1088× |
| Action phase，A3 对 A2 | 480.36→478.22 | 1.0038× |
| 仅固定 context 准备，A3 对 A2 | 6.227→1.303 | 4.7723× |

A2 每 chunk 先付约 6.18 ms 准备成本，随后十轮不再重复相同 K/V 投影；整段配对已计入该成本，仍取得约 10% latency reduction。A3 的跨层 projection 可以并行，但只改变准备阶段；它不并行三十层 action hidden-state 的依赖，也不并行十轮 denoising。准备阶段的约五倍加速被其在完整 action phase 中的小占比稀释，直接整段比较只剩约 0.38% speedup。四次技术重复、三个同 episode observations 不支持把这种微小差异外推为稳定的部署收益。

### 缓存显存和冷构造成本

| 配置 | 持有的 chunk cache | 额外 packed weights | Peak allocated | 相对原生 peak 增加 |
|---|---:|---:|---:|---:|
| A0 原生 | 原生已有 video cache | 0 | 23,805.15 MiB | — |
| A1 | 0.252 MiB | 0 | 23,805.17 MiB | 0.021 MiB |
| A2 | 45.604 MiB | 0 | 23,847.96 MiB | 42.815 MiB / 0.180% |
| A3 | 45.604 MiB | 360.527 MiB | 24,386.61 MiB | 581.460 MiB / 2.443% |

这些是 context0 memory pass；三个 contexts 的 allocated 峰值相同。A3 一次 constructor 为 2.368 ms；按三个被测 contexts 各使用一次示例摊销为 0.789 ms/chunk，若部署运行 N chunks 则为 2.368/N ms，不把计时重复当独立任务。各 block 的逐层 normalization 和宽投影准备同时持有较多临时 tensors，A3 相对自身（已包含 packed weights）baseline 的动态峰值为约 227.01 MiB；因此它的峰值增量不等于仅 packed weights + 最终 cache。这里的临时开销未做 allocator stack 分项归因。

就当前证据，优先保留 A2 的简单逐层 cache；A3 的额外整段收益很小，需权衡其约0.53 GiB额外峰值（相对 A2）及实现维护成本。当前只提供实验侧实现，不修改只读官方源码或发布候选 closed-loop success claim。

### 工程失败与重跑记录

初次作业 `25664341.pbs101` 为 F / Exit_status=1 / wrapper exit=1，无 `PIPELINE_COMPLETE`。已完成 embedding action-phase 一项，随后完整 infer 输出为 CPU tensor、reference 为 CUDA tensor，runner 在计时区间外比较时混用 device。共享评分 helper 已统一转 CPU，首十步比较复用该 helper；未改变算法、计时区间、候选或 gate。原失败记录保留，重跑作业 `25664452.pbs101` 完整通过；初次不完整样本不并入重跑收益估计。

## 文件与监控

- `PROTOCOL.zh.md` 为共同冻结条件；`CONTEXT_PROTOCOL.zh.md` / `TIME_PROTOCOL.zh.md` 为各线机制。
- `runner.py` / `run.pbs` 调用只读官方源；`context_cache.py` / `time_parallel.py` 为实验侧实现。
- `STATUS.json` 记录提交的 jobs；`MONITOR_STATUS.json` 由 `gpt-6-luna/xhigh` 只读监控 agent 记录当前权威状态。监控不提交或重启作业，网络超时也不当作终态。
- Remote artifacts：`/scratch/users/ntu/yguo017/fastwam-action-parallel/artifacts/<PBS_JOBID>/`。大 `frozen_contexts.pt` 留在 compute-side，不在 login node 做大传输。
- 原始失败 job 的 Exit_status 会保留；只有各线 terminal receipts、wrapper exit、测量与 PIPELINE_COMPLETE 都取得后才更新为测量完成。

两条完成作业的 PBS `Stageout_status=1` 作为 caveat 保留，不能宣称自动 stageout 已成功；已通过小型摘要读取取得 summary/CSV/self-check/exit/完成标记，完整 frozen tensors、videos 和执行时复制的源码保留于 remote artifacts/output。只读监控已结束，没有新作业或候选闭环结果等待。本轮未训练、未放宽 gate、未叠加两条优化，也未 commit/push。

终态与数据入口：

| Line / attempt | PBS state / Exit_status | Wrapper / PIPELINE_COMPLETE | Local small artifacts |
|---|---|---|---|
| Context 初次 | F / 1 | 1 / absent | `artifacts/25664341.pbs101/` |
| Context 重跑 | F / 0 | 0 / present | `artifacts/25664452.pbs101/` |
| Time | F / 0 | 0 / present | `artifacts/25664342.pbs101/` |

协议已列出 ParaDiGMS 与 materially contributed 的 Scientific Agent Skills 方法来源；论文结果不作为本地加速证据。
