# Rolling Ball / LeWM 执行记录

当前状态：正式 campaign 已完成并取回全部结果。使用一个 epoch-100 LeWM checkpoint、同一组 50 个 physical-v2 fixture，完成同步 50、固定延迟六档各 50、真实异步 50，共 400 次 episode 运行；独立初始条件仍为 50 个。三个 stage completion gate 全部通过，runner / wrapper exit 均为 0。模型服务已按计划停止。2026-09-28 00:31:15 SGT，用户授权的关机脚本返回 0，之后 SSH 连接结束；串行复查为 ConnectionRefused。控制台电源状态未独立核实，见 [关机记录](artifacts/autodl/campaign-physical-001/shutdown_receipt.json) 和 [AUTODL_RUN.json](AUTODL_RUN.json)。

这轮结果没有建立“延迟越大，成功率越低”或“一步模型更容易成功”的结论。同步基线仅 1/50 成功；真实异步也为 1/50，且 RTF 仅 0.312，没有达到真实世界 25 Hz 的 wall-clock 速度。当前应视为任务适配与并行调度 pilot，不能作为强时延因果证据。

## 正式结果

每一行均为同样的 50 个 fixture。Server / CEM 为该条件全部 request 的中位数；simulation observation age 仅统计实际执行的动作，终止前丢弃的 request 不混入年龄分布。

| 条件 | 成功 | Server median / ms | CEM median / ms | Applied sim-age median / ms | RTF |
|---|---:|---:|---:|---:|---:|
| Sync | 1/50 (2%) | 646.54 | 634.77 | 0 | 0.05297 |
| K0 / 0 ms | 1/50 (2%) | 645.80 | 633.85 | 0 | 0.05319 |
| K1 / 40 ms | 0/50 (0%) | 647.64 | 634.96 | 40 | 0.08863 |
| K2 / 80 ms | 2/50 (4%) | 667.48 | 635.73 | 80 | 0.11996 |
| K4 / 160 ms | 0/50 (0%) | 645.27 | 632.99 | 160 | 0.19085 |
| K8 / 320 ms | 4/50 (8%) | 645.58 | 632.90 | 320 | 0.25840 |
| K16 / 640 ms | 0/50 (0%) | 645.79 | 631.35 | 640 | 0.32110 |
| True async | 1/50 (2%) | 679.86 | 666.98 | 200 | 0.31163 |

完整 [SUMMARY.json](artifacts/autodl/campaign-physical-001/SUMMARY.json) / [SUMMARY.csv](artifacts/autodl/campaign-physical-001/SUMMARY.csv) 包含 count、median、p95、成功 seed、逐 fixture gain/loss、request 状态和终止阶段；[完整证据包](artifacts/autodl/formal-evidence-physical-001.tar.gz) 包含全部 episode/request 原始 JSON、三个 stage 的 GPU telemetry、退出文件和实际 control overlay。完整 fixture bank 已单独保存为 [fixtures.pt.gz](artifacts/autodl/pair-physical-50-newhost-002/fixtures.pt.gz)。未重复下载模型包或运行哈希校验。

真实异步有 216 个 request，177 个动作已执行，39 个在 native terminal 时丢弃，request error 为 0。已执行动作的 wall observation age median / p95 为 718.27 / 800.59 ms，simulation age 为 200 / 240 ms（范围 120–280 ms）；仿真确实在等待 inference 时继续推进。216/216 planner request 超过 40 ms deadline，1262/1262 control ticks 有超过 1 ms 的 wall-pacing overrun。原始 RTF 0.31163 截止最后 native physics step；计入 terminal cleanup / pending-request drain 后为 0.29768。这相当于约 7.8 Hz 的 wall-clock control，而非 25 Hz hard realtime。

同步有 1304 个 request，全部实际执行、全部超过 40 ms；wall observation age median / p95 为 653.21 / 779.22 ms，但 simulation age 为 0，因为计算期间环境冻结。固定延迟的实际 applied simulation age 全部与预设 K×40 ms 一致。固定延迟模式没有 wall-clock pacing，较高 RTF 来自更少的 planner 调用，不能称为 planner speedup。

Sync 的成功 seed 为 2026092724，K0 为 2026092737，两个无延迟条件的唯一成功没有重合。80 ms 和 320 ms 分别出现 2 和 4 次成功，也不能解释为延迟有益：延迟同时改变动作更新频率，当前基线很弱，而且 strict RGB pairing 未通过。此轮没有训练或评估一步生成模型，也没有改变 CEM 轮数。

当前是 LeWM 的 Rolling Ball 任务适配：goal cost 使用 180 个训练 episode 的最后可用 pre-step RGB，发布数据没有逐个核实的接球成功标签。它不能当作论文已发表的 Rolling Ball baseline。下一步需要先诊断目标 cost 是否反映接球成功及 action response / ranking，并解决仿真达不到实时速度的问题，再判断该 benchmark 是否适合检验规划加速。单张 4090 共享 Isaac RGB 与 LeWM/CEM；本地 reference evaluator 将 render_interval 设为 1、每个 control tick 跨 4 个 physics steps，但实际 render 调用次数与分项耗时未测量，不能把 RTF 慢全部归因于 renderer 或 GPU 争用。

## 当前 RTX 闭环证据

当前 host 为 `autodl-container-59db4f87a9-11fcb48a`，RTX 4090 / 24,564 MiB，driver 570.124.04。SDK empty-scene 003 完成 10 physics steps，native/wrapper exit 0。模型服务启动日志确认正式 epoch-100 checkpoint 和固定 source 身份；不是仅凭 `/info` 的 action shape 判定加载了正确模型。

严格 RGB v1 配对门没有通过：FXAA 加 256 纯 render 预热的最后尝试仍为 mean absolute difference 0.34320、p99 3，高于原 0.25/2 阈值。旧失败全部保留。后续在任何 policy outcome 前确定为 [physical-v2 pilot](EVAL_ADDENDUM_PHYSICAL_PAIRING_V2.json)：原生实时 RGB、原生渲染和全部物理/任务/event/counter/RNG 硬门保留，RGB 差异作诊断，不声称严格视觉配对。

Physical-002 保存了 50 个 fixture，seed `2026092700..2026092749`。同一首个动作的 scene/task/event/counter/RNG replay 全部通过，scene max absolute difference 为 0；没有保存 raw PhysX solver cache。原 `pair.json` 保存 bank 后有 summary-print 参数错误，保留该错误报告；修复日志后，bank 经 adapter 原生 loader 验证，见 [fixture_bank_validation.json](artifacts/autodl/pair-physical-50-newhost-002/fixture_bank_validation.json)。完整 gzip bank 已取回本地。

同步 smoke 003 的 native/wrapper exit 均为 0，JSON `COMPLETED`、physical paired、closed-loop evaluated，成功 0/3。每个 episode 均为 25 control ticks、1.0 simulated second，native `max_phase=terminated_phase=2`、`end_reason=terminated`，未达到 phase-4 success。具体具名 termination 未记录，不把推测原因当已确认。见 [smoke 原始结果](artifacts/autodl/sync-smoke-physical-003/sync.json) 和 [摘要](artifacts/autodl/sync-smoke-physical-003/SMOKE_SUMMARY.json)。

75 个 smoke request 的完整 server planner 中位耗时 644.95 ms，CEM 632.20 ms，client HTTP 647.02 ms；wall observation age 650.97 ms，simulation observation age 为 0，因为同步推理期间仿真冻结。40 ms deadline miss 为 75/75；aggregate RTF 为 0.05335。这些时延不能单独证明 WM、CEM 或环境哪个环节造成接球失败。

正式 campaign PID 30529 已退出，使用同一 checkpoint、同一 physical-v2 bank、CEM 300/top30/30 rounds/H5/seed1234。三个 stage 均有 native/wrapper exit 0 和 30 s GPU telemetry；每个 condition 的完整 request trace 已取回。没有按 smoke 成败挑选 seed、重训或调参。上述 completion gate 表示执行与数据完整完成，不表示基线已达到可靠任务能力。

以下为数据、训练与迁移的历史记录；当前运行状态以上述 RTX 闭环证据和交接文件为准。

## 已完成的实际作业

CPU PBS job `25578687.pbs101`：`job_state=F`、`Exit_status=0`、wrapper exit=0、runner exit=0、`preparation.status=PASS`。`Stageout_status=1` 保留为 caveat；实际小型报告已通过安全连接取回本地，未仅凭 PBS 退出码判定准备通过。

远程记录：`/scratch/users/ntu/yguo017/lewm-rolling-ball-async/runs/25578687.pbs101/`。本地证据：[preparation.json](artifacts/25578687.pbs101/preparation.json)、[info.json](artifacts/25578687.pbs101/info.json)、[JOB.json](JOB.json)。

- ReflexBench 源码固定到 `8bb931485093c6d98f8729774ad01bf824964e16`。
- 数据固定到 `9295b6e9878609a992047f0b8b65421a493299e7`。
- 合并数据中 task index 3 对应 Rolling Ball，共 200 个 episode。读取的是 source / metadata，没有下载视频、运行模型或计算成功率。
- 集群现有 LeWM Python 环境有 torch `2.8.0+cu128`、stable-worldmodel `0.1.1`、pyarrow `24.0.0`；没有 Isaac Sim / Isaac Lab / LeRobot / PyAV 的 package metadata。该查询不等于这些模型或库的运行验证。
- 旧 LeWM / StableWM 源目录没有独立 Git HEAD，因此另行下载固定版本源码，没有将旧目录假定为参考版本。

CPU PBS job `25578788.pbs101` 已完成：`job_state=F`、`Exit_status=0`、runner / wrapper exit 均为 0、`source_preparation.status=PASS`；`Stageout_status=1`。准备了 vanilla LeWM `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`、StableWM `10c26dbd5677083fa31dba69eb738b973845e9a4`，以及独立目录中的 PyAV 16.0.1。AV1 decoder 能构造；实际视频解码由下一作业验证。

数据作业 `25578909.pbs101` 已完成：F / Exit_status 0 / runner 0 / wrapper 0 / summary PASS；Stageout_status=1。固定 camera 提取 200 个 episode、5,200 帧，5200 个匹配视频帧互不重复，最大 PTS 误差约 2.86e-8 s（阈值 0.020001 s）；state 确认为真实 robot proprio，[5200,8]。划分 180 / 20 train / validation episodes。

GPU 短训练作业 `25578958.pbs101` 已完成：F / Exit_status 0 / runner 0 / wrapper 0 / summary SMOKE_ONLY；Stageout_status=1。1 epoch / 2 train batches / 2 validation batches，完整 split 的 4,140 / 460 窗口 alignment PASS；train / validation loss 均 finite。job.log 有 12 次约 30 s 间隔的 GPU 采样。这项只验证接入，baseline_checkpoint=null；spt.Manager 自动保存了一份 smoke 内部 cache checkpoint，它不能作为 formal baseline。

正式训练作业 `25578999.pbs101` 已完成：F / Exit_status 0 / runner 0 / wrapper 0 / summary PASS_epoch100 / completed_epochs 100；Stageout_status=1 保留。固定 100 epochs / batch128 / bf16 / seed0，从头训练一个模型；同步、固定延迟、真正异步评估共享正式 epoch 100 的 last.ckpt（72,263,938 bytes）。最终 validation prediction MSE 为 0.02377214，仅为 predictor 诊断，不建立规划或接球能力。job.log 有 48 条约 30 s 间隔的 GPU 采样；实际训练 elapsed 为 1230.43 s。正式 trainer 将 Manager cache 放入作业内，并关闭自动 requeue checkpoint，由 epoch callback 保存 raw model checkpoint。

完整 CEM 离线检查 `25579057.pbs101` 已完成：F / Exit_status 0 / runner 0 / wrapper 0 / summary PASS_protocol_and_planner_shape；Stageout_status=1 保留。正式 summary PASS_epoch100 / completed_epochs100 / 非 smoke / epoch100 raw checkpoint gate 已通过。检查取 validation ep638 / row52 / frame0 和真实同帧 robot proprio，运行原版 CEM 300 candidates / top30 / 30 rounds / horizon5，输出 action shape [1,1,8]，final cost shape [1,300]、finite，min 0.05834091 / max 0.06361142。单次 cold request 完整 planner 为 4.72230 s、CEM 为 4.08804 s。job.log 有 8 个 GPU 周期采样，峰值样本为 74% / 745 MiB。该项不计为 steady-state timing、闭环成功率、observation age 或 RTF 证据；也不代表未来 RTX 4090 latency。[PLANNER_JOB.json](PLANNER_JOB.json) 保留终态证据。

前三次准备失败分别由联合 dataset 布局、2.08 MB metadata 超出初始读取上限、compute 环境没有 git CLI 引起。对应作业 `25578602` / `25578648` / `25578668` 全部保留，修复由 root 执行；监控 agent 没有修改或重提作业。这些失败不代表 LeWM 性能失败。

## 迁移与历史环境准备

迁移准备：CPU export `25579500.pbs101` 已完成，F / Exit_status 0 / runner 0 / wrapper 0 / EXPORT_SUMMARY PASS，Stageout_status=1；4 CPU / 24 GB。Compact bundle 为 101,562,798 bytes / 289 files，单个 tar.gz 为 74,570,166 bytes；未通过 login node 或本机搬运大文件。[EXPORT_SUMMARY.json](artifacts/25579500.pbs101/EXPORT_SUMMARY.json) 记录路径。导出时 `eval_rolling.py`、`verify_bundle.py`、`run_bundle_smoke.pbs` 尚未纳入，后续作为小型 control overlay 补充，不重复导出模型。

GPU full-vs-compact parity `25579502.pbs101` 已完成：F / Exit_status 0 / runner 0 / wrapper 0 / PASS_full_vs_compact_planner_parity，Stageout_status=1。Export report PASS、对应 job_id 和 runner/wrapper 0 gate 已通过；同一 probe RGB/state、split、ordered goal refs、checkpoint identity 一致。Goal embeddings [180,192]、action [1,1,8] 和 final candidate costs [1,300] 的 max_abs_diff 均为 0，full/compact imports 来自各自 pinned source 目录。仅两次单次 CEM 的诊断耗时为 0.8831 / 0.8110 s；它们与早先单次 4.0880 s 不构成 speedup 或 steady-state timing 比较。GPU job 按 30 s 周期采样，短运行实际只产生一个 0% / 1 MiB 样本，不能表示整次 GPU 负载。该项不作为闭环、observation age 或 RTF 结果。

用户确认目前只有 ASPIRE2A；已使用的 normal GPU allocation 为 A100。NVIDIA [Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html) 明确不支持没有 RT Cores 的 A100 / H100。ReflexBench 的原生视觉路径使用 Isaac CameraCfg RGB，源码中没有可直接切换的非 RTX 相机 renderer。`headless` 只移除 GUI。

资源更正：NSCC [2024 Introductory Workshop](https://help.nscc.sg/wp-content/uploads/2024/05/NSCC-Introductory-Workshop-Theory-ASPIRE2A.pdf) 第 18 页另列 8 个 NVIDIA A40 visualization nodes。因此不能从 normal allocation 的 A100 推导整个 ASPIRE2A 都没有 RT-capable GPU。当前 pbs101 的一次轻量节点元数据查询只列 64 个每节点 4 GPU 的节点，queue inventory 没有 viz queue；可视化节点的当前 portal/PBS 路径、账户访问和 Isaac 工作负载许可仍待验证，不能声称已获可用 allocation。

这是一项官方要求和源码路径的核实，尚未在 A100 上启动 Isaac Sim 做 runtime failure test。没有为此安装庞大仿真环境。State-only evaluation 会使用任务内部 ball state / phase / predicted intercept，不能作为视觉 LeWM baseline 的等价替代。

评估入口 `eval_rolling.py` 已实现。stdlib 自检 PASS，执行真实 `run_episode()` 调度分支的 fakes，覆盖 sync/K0 等价、K2 精确 hold、异步等待时继续推进、terminal future drain/drop 和客户端到达/服务器完成时钟分开。AST / freeze JSON 检查通过。初始 fixture 恢复、同动作 replay、RGB rendering、共享 GPU 显存和原生任务成功判据仍需 RTX runtime 验证；这些代码检查不建立闭环或 paired runtime 结论。[EVALUATOR_ADAPTER.md](EVALUATOR_ADAPTER.md) 记录执行边界。

当前已有正式 epoch 100 checkpoint；没有同步成功率、固定延迟成功率、异步成功率、observation age 或 RTF 结果，目标尚未完成。离线与评估配置已写入 TRAIN_FREEZE.json / PLANNER_FREEZE.json / EVAL_FREEZE.json。AutoDL 1×4090 实例已接入，但用户目前主动切到无卡模式。74,570,166 bytes bundle 经用户授权的单次 SFTP 中转成功，已经在 AutoDL 解压，正式 checkpoint 和 180 张 goal RGB 均已就位；本机没有模型计算。迁移脚本在成功上传后曾因 Windows 文件锁清理返回 1，该清理问题已修复，上传未重复执行，具体保留在 `RELAY_RESULT.json`。

CPU 准备已完成：基础安装器 PID7505 native exit 0；Isaac Lab core 修复 PID8896 / exit0 / `PASS_LAB_CORE_PACKAGES`。最初 CPU package check PID10083 / exit1 的 datasets 2.14.4 与 pyarrow24 导入不兼容失败保留；PID10390 仅修复该间接依赖到 datasets5.0.1，exit0 / `PASS_CPU_POLICY_IMPORTS`，成功导入 jepa / utils / 原版 CEM，十项冻结 policy 版本全部匹配。Base Python3.12.3 / torch2.7.0+cu128 等预装包未更改；独立 simulator / policy 均为 Python3.11。22 个 USD/纹理文件的 CPU 依赖检查已通过，Kit 的 OmniPBR.mdl 已存在。该状态只表示资源和 CPU imports 就绪，未加载 checkpoint，未验证 CUDA、Kit shader、RGB、配对恢复或接球能力。现在等待用户恢复 GPU 后接续，不重新训练。

恢复 GPU 后先做 GPU/driver、task/RGB/action 与初始状态配对 smoke，再进入同步、固定延迟和真正异步的正式比较，不需要重新训练模型。正式评估使用同一组 50 个 fixture；3 episode 能力 smoke 只取该 bank 的前缀。[租卡接续](RENTAL_HANDOFF.zh.md) 记录当前接续状态。
