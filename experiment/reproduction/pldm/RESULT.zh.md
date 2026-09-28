# PLDM Two-Rooms 复现阶段记录

**状态：用户已取消正式运行，goal 按更新后的范围结束。** 官方代码、数据准备、CPU runtime 与完整 GPU 成本校准已完成。正式作业 `25572927.pbs101` 在排队期间撤销，没有开始正式训练，也没有正式 checkpoint、native evaluation 或成功率结论。

## 冻结设置与证据范围

- 官方仓库 `vladisai/PLDM` 固定为 commit `1bd7e564ecd961205bc18b23067b19e9ca24ac90`。只在 `staged/pldm` 对 Python 3.11 实际触发的 mutable dataclass defaults 作字段级 `field(default_factory=相同constructor)` 兼容修复，32 个原行和替换行写入 `staged_source_identity.json`。`upstream/` 保持原样，训练算法和冻结 config 值未修改。
- 使用官方发布的 `good_quality_data_no_images.npz` 渲染完整 dataset。磁盘数据为 3,686,400 transitions；实际训练 loader 将其构造成 3,072,000 个 length-16 windows。仓库 config 原有 `wall-visual-config_rand_expert_40-v0.npz` 未在当前 release 中找到可证实映射；因此不声称与该旧文件 bit-equivalent，也不声称逐项复现 paper v4 的约 3M/Table 2 数值。
- 官方配置 `epochs=2` 由 pinned trainer 实际执行 epoch indices `[0,1,2]`，即 3 个 passes。代码配置保留 2,000 个 MPPI samples，paper hyperparameter table 报告 500；代码中的 `eval_mpcs=20` 字段未被 pinned evaluator 使用。原生 medium evaluation 为 100 environments、200 steps、batch 20、每步 replanning、horizon 最大 96。Pinned 循环没有 early exit，完整一次 invocation 按 5 chunks × 200 steps 执行 1,000 次 `planner.plan`；这是源码规定的工作量，正式运行的实际耗时尚未测量。

## CPU preparation 结果

作业 `25571604.pbs101` 终态为 `F`、`Exit_status=0`、`Stageout_status=1`，CPU report 为 `PASS`。运行环境为 `/scratch/users/ntu/yguo017/lewm-pusht-iteration/venv/bin/python`，Python 3.11.7 / Torch 2.8.0+cu128；numpy、scipy、pandas、zarr、gym、statsmodels 等 imports 均从已记录的官方 pin overlay 或原只读 overlay 导入，共享 venv 未升级。

实测 official training loader 读入 3,072,000 个 windows，batch size 64，每 pass 48,000 updates；三 passes 共 144,000 optimizer updates。Normalizer/loader build 用时约 277.73 秒，首个 batch 用时约 2.68 秒，shape 为 `[64,16,2,65,65]`。模型和正式训练尚未在 CPU 预检中加载或运行。

Probe loader metadata 来自 official `DatasetFactory`，train 与 validation 各 156 batches/pass。predictor 20 epochs 加 encoder 30 epochs 共 50 个 training passes、7,800 training batches；两项 validation 共 312 batches，另有两个官方 shape-probe batches。CPU gate 只读取长度，没有生成或迭代 probing batch。

## GPU 成本校准与正式实验

`25571685.pbs101` 的 PBS 终态为 `F`、`Exit_status=2`、`Stageout_status=1`，实际 walltime 13:15、CPU time 00:00:39、memory 1,898,524 KB。校准报告状态为 `PARTIAL`：完成 10/10 warmup 和 97/100 timed updates，共 107/110；内部 720 秒校准时限触发停止，`training_error=null`，末尾写出 `GPU_CALIBRATION_PARTIAL` 并以 exit 2 结束。它没有执行 native planner cost probe，也没有保存 calibration checkpoint。完整原始证据保存在远端 `reports/calibration-25571685.pbs101/CALIBRATION.json` 和 `job.log`。

已完成的 97 个 timed updates 报告 mean 2.9990 秒、median 2.6056 秒、p95 3.3656 秒；GPU training peak allocated/reserved 分别为 7,038,008,320/7,111,442,432 bytes。GPU 身份核对为 `CUDA_VISIBLE_DEVICES` UUID `GPU-e6f4ed92-545b-476f-d00e-491441df423a`，对应 NVML index 0 与 Torch `cuda:0` 的 NVIDIA A100-SXM4-40GB；作业按 5 秒间隔记录了 per-GPU 利用率和显存，多数观察样本 utilization 为 0%，偶尔为 15%/19%。由于计划的 110 更新未全部完成且 planner cost 未测，报告中的整段训练和 probing proxy 外推不用于正式资源决策。

compile 计时也有明确边界：staged `pldm/train.py` 的 `compilation finished after 18.593s` 计量的是 `torch.compile(self.model)` 调用本身；该代码段没有执行模型 forward，也没有计量首次调用后的完整编译。标准 `torch.compile` 流程会在首次调用时惰性编译；首个 warmup update 总耗时 16.73 秒，但其中还包含 loader、forward/loss/backward、optimizer/EMA/logging 和 CUDA synchronize，不能分离为纯编译时间。[PyTorch compile-time documentation](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.reducing_compile_time.html)

第二次 calibration `25572720.pbs101` 的 PBS 终态为 `F`、`Exit_status=0`、`Stageout_status=1`，请求 30 分钟后于 05:34 结束，walltime 05:06、CPU time 01:07、memory 32,086,060 KB。normal queue 的短时 GPU allocation 路由到 gdev，实际运行节点为 `x1000c1s0b0n1`。完整原始报告与 GPU job log 已保存在远端 `reports/calibration-25572720.pbs101/`，本地副本为 `CALIBRATION.25572720.json` 与 `job.25572720.log`。

存储 gate 使用官方 eager `lazy_load=false` 路径。states 数组是 host `uint8` ndarray，shape `[40960,90,2,65,65]`、31,150,080,000 bytes；Trainer 初始化61.64秒，gate 后进程 RSS 32,107,233,280 bytes。64个确定窗口包含首尾与多个 trajectory 边界；raw 和 Normalizer 输出的六个 `WallSample._fields`（`states/locations/actions/bias_angle/wall_x/door_y`）均逐字段 `torch.equal`，RNG 状态前后不变。该结果支持正式使用 native eager storage 路径，数据和值未改变。

第二轮完成 10 warmup 和 100 timed updates。Timed update mean/median/p95 为 0.044064/0.044065/0.044233 秒；其中 `next(iterator)` loader wait mean/p95 为 0.005848/0.005944 秒，update body（含现有 CUDA completion sync）为 0.038215/0.038346 秒。官方每 pass 48,000 updates，epoch indices 0/1/2 共144,000 updates；按完整 timed mean估计 training updates 用时6,345.19秒，另外Trainer初始化61.64秒。该成本估计基于通过存储等价 gate 的 eager 数组表示，不是已完成的完整训练。

Probing 的8,114个 batch-work 没有逐批计时；按训练 update mean 换算的357.53秒只是 planning proxy，不是测量值或 guaranteed upper bound。另一个 cost-only probe 使用 `model.eval()`、B=20、H=96、S=2,000，完成1 warmup+3 steady `MPPIPlanner.plan` calls；稳态均值1.29437秒，输出 shapes 和 finite gate PASS。Pinned native final evaluation 每次预计执行1,000个 planner calls，故 planning-only H=96 proxy为1,294.37秒，latent-transition线性 proxy为986.96秒；两者均未计环境step、prober位置输出映射、probing、绘图、汇总与I/O。Probe没有执行 native evaluator、没有读取 success，也没有产生质量指标。

训练 updates、初始化、probing proxy 与 planning-only proxy 的组件和约8,058.7秒（2小时14分钟）。Root 已批准 formal allocation 为 normal request、1 GPU、16 CPUs、110 GB、4小时，`train.py` 内部 timeout230分钟并预留10分钟结束/Stageout。最终 PBS 脚本已由 root 审阅并批准一次提交；作业会从零训练，关闭resume、initial/mid-training validation，只运行 official native final validation，不加载 calibration checkpoint。完整资源计算及估计限制见 [RESOURCE_AMENDMENT.zh.md](RESOURCE_AMENDMENT.zh.md)。

Calibration-only checkpoint `/scratch/users/ntu/yguo017/pldm-reproduction/calibration_output/tworooms-calibration-25572720.pbs101/epoch=0_sample_step=7040.ckpt`（26,675,552 bytes）已保留，禁止用于正式训练/评估。正式训练完成后才会从 native trainer/evaluator summary、checkpoint metadata、planning result/report 和实际终态 PBS 状态采集正式证据。预计正式 checkpoint 文件名为 `epoch=2_sample_step=9216000.ckpt`，其中 sample_step 是累计 training windows；该路径目前仅是预期命名，不代表文件已生成。


## 正式作业状态

正式 GPU 作业 `25572927.pbs101` 曾提交一次并处于 `Q`。2026-09-27 用户取消该实验，`qdel` 成功；随后 `qstat` 确认 `F` 且 comment 含 `terminated`。该查询未提供 `Exit_status` 或 `Stageout_status`，因此不填造退出码。正式训练未开始；监控已停止，不重提、不采集正式结果。此前资源批准与预期产物仅作为设计历史保留。

机器可读的 job/source/data 状态见 [SUBMISSION.json](SUBMISSION.json)。

实验设计流程参考：Kassis, T.; Agarwal, V.; He, Y.; Patel, D.; Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. [doi:10.48550/arXiv.2609.00065](https://doi.org/10.48550/arXiv.2609.00065)。
