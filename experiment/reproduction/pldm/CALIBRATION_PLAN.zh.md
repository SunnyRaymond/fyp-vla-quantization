# PLDM GPU 成本校准

状态：CPU preparation `25571604.pbs101` 已通过（`Exit_status=0`、`Stageout_status=1`）。首轮 GPU cost calibration `25571685.pbs101` 保留为 107/110 updates 的 PARTIAL；第二轮 `25572720.pbs101` 已 PASS，完成存储等价 gate、110 updates 与 1 warmup+3 steady planner cost calls。Root 已批准正式资源/storage amendment：4小时 GPU allocation、230分钟 inner timeout，完整科学预算不变；正式作业 `25572927.pbs101` 已提交一次；当前 `Q`，请求队列 normal、调度器实际队列 g1，comment 为队列运行名额限制。保留原 handle 只读监控，不重提。

真实 loader 确认完整 released 数据形成 3,072,000 个 sequence windows，batch 64、每 pass 48,000 updates；官方 epoch indices 0/1/2 共 144,000 updates。原生 probing 的 train/val loader 各 156 batches；20+30 个 training passes 共 7,800 batches，另有 312 validation batches 与 2 个 shape-probe batches。以上是完整配置的工作量元数据，尚未证明 GPU 成本或科学质量。

校准估计冻结官方配置下的训练、初始化/保存和 native planner 计算成本，不生成科学质量结论，也不把 calibration checkpoint 用于正式评估。

## 冻结入口与运行边界

- 官方源码：`vladisai/PLDM`，commit `1bd7e564ecd961205bc18b23067b19e9ca24ac90`；从独立 `staged/pldm` 入口导入，并逐个记录模块来源。
- 数据：官方 released `good_quality_data_no_images.npz` 渲染的完整 `good_quality_data_memmap`。要求 preparation report、render metadata 与 CPU loader probe 都通过，且正式数据为完整 3,686,400 帧、`uint8` states。
- 环境：复用现有只读 Python/Torch runtime；PLDM source 与 task-specific NumPy 1.26.4/SciPy 1.10.0、Pandas 2.0.1、Statsmodels 0.14.4/Patsy 1.0.1、Zarr/numcodecs 依赖 overlay 优先导入。上述新 scientific/dataframe wheels 仅为已观察 import/API mismatch 按官方 pins 隔离，不改共享环境。
- 配置：官方 `configs/wall/icml/seqlen90_3M.yaml`，seed 101、`epochs=2`、`quick_debug=false`、sequence length 16、VICReg + IDM、CUDA、100 个 eval environments、200 environment steps、batch 20、每步 replan、MPPI 2000 samples、`eval_mpcs=20`。首轮 `25571685.pbs101` 使用 `lazy_load=true` 并以107/110 updates终止；第二轮 `25572720.pbs101` 使用 native default `lazy_load=false`，完成64窗 eager uint8 ndarray/read-only memmap equality gate。其他只覆盖 released 数据路径、W&B/resume/eval 开关和隔离输出目录。

## 校准方法

作业在启动时检查 `PBS_JOBID`、`PBS_NODEFILE` 与当前 hostname；不满足 allocation 检查就退出。首轮实际作业使用 normal queue、1 GPU、16 CPU、110 GB、15 分钟并以部分状态终止。第二轮资源为 normal request、1 GPU、16 CPU、110 GB、30 分钟；scheduler 将其路由到 gdev（该时长的预期行为）。训练 deadline 为1200秒、planner deadline1710秒、outer timeout29分钟，作业在 `reports/calibration-<jobid>/job.log` 每 5 秒按 GPU index/UUID/name 记录 utilization 和显存。启动时打印 `CUDA_VISIBLE_DEVICES`，并核对 PyTorch `cuda:0` 名称与该可见 token 对应的 NVML 设备；`CALIBRATION.json` 也记录同一 identity mapping。

先运行完整官方 `Trainer` 初始化，再用只转发真实官方 dataloader 的有界代理完成 10 个 warmup updates 和 100 个 timed updates。计时包括 loader wait、官方 forward/loss/backward/optimizer/EMA/logging 路径，并在每个 update 后 `torch.cuda.synchronize()`。模型、loss、optimizer 和数据不替换；不调用 native validation/evaluation。只有完整 110 updates 才写入 job 专属 calibration 输出目录，记录保存时间和文件大小；该 checkpoint 明确禁止作为正式训练或结果使用。报告记录初始化、配置解析、每步时间分布、峰值显存、CPU loader 报告中的每 epoch update 数及三次 pass 的总 update 数。

CPU loader metadata gate 同时通过 official `DatasetFactory._create_wall_probing_datasets` 记录 probing train/val loader lengths，不迭代或生成额外 probing batch。官方 eval 配置对 predictor prober 训练 20 epochs、encoder prober 训练 30 epochs，共 50 个 training passes；另按 flags 记录 validation passes 和两个 shape-probe batches。GPU calibration 不训练 prober；报告以实际 probing batch-work 与 batch-size 比例乘 measured full training update mean 作为明确标注的保守 planning proxy，不当作实测 probing 时间，并为正式资源保留额外 margin。若该 proxy 与正式预算间差距仍不清楚，可在 root 审阅后复用 calibration checkpoint 做 bounded prober-only cost probe。

训练校准完成且 15 分钟 allocation 留有足够时间时，同一 job 继续运行 cost-only 的官方 `MPPIPlanner.plan` probe：固定首个真实 loader batch 的 20 个 current/goal observations，目标 encoding 在每次调用前重置，H=96、S=2,000，先 1 次 warmup、再最多 3 次 steady calls；planner 的 nominal MPPI 状态按 native planner 行为保留。它只计运行时间、输出 shape、finite 与显存，不触发 evaluator、success/prober quality 或其它质量值。Pinned `projected_cost=false`，因此 `prober=None` 不改变 MPPI running cost；这项 probe 明确不计 official trained-prober 的预测位置输出映射开销。

第二轮 `25572720.pbs101` 终态 PASS，Exit_status=0、Stageout_status=1，完成10 warmup+100 timed updates和1+3 native planner cost-only calls。eager uint8 states数组31,150,080,000 bytes，Trainer初始化61.64秒，gate后RSS约32.11GB。六个运行时 `WallSample._fields` 在64个确定窗口上raw与normalized均逐字段严格相等，RNG unchanged。Timed mean为0.044064秒/update（loader wait均值0.005848秒，update body均值0.038215秒）；planner steady mean为1.29437秒/`B=20,H=96,S=2000`。完整训练、probing和native evaluation的成本边界与限制见 `RESOURCE_AMENDMENT.zh.md`。Root 已批准正式资源/storage amendment：normal queue request、1 GPU、16 CPUs、110 GB、4小时，230分钟 inner timeout、10分钟收尾；root 已审阅并批准最终 PBS 脚本的一次提交；正式作业 `25572927.pbs101` 当前排队于 g1。

## 正式 native evaluation 的成本边界

按 pinned evaluation 配置，每次 native evaluation invocation 使用 100 个 environments、200 个 steps、每 20 个 environments 一批、每步 replanning，因此是 200 × 5 = 1,000 次 planner calls。MPPI 为 2,000 samples，规划 horizon 上限 96。依据 `sum(min(200-i, 96), i=0..199)=14,640`，一次 invocation 约对应 2.928 billion candidate latent-transition evaluations（100 × 2,000 × 14,640）。`eval_mpcs=20` 只是在 `TrainConfig` 中定义、未被 evaluator 使用的字段，不代表 20 次调用。

第二轮实测 steady H=96 call均值1.29437秒；按 latent transitions线性换算，一次 native invocation约等于762.5个H=96 call。1000次H=96 call是planning-only proxy，不是完整 evaluation 时间上界：不含environment step、trained-prober projection、native probing、I/O或调用间开销。完整成本组件约8058.7秒（2小时14分钟），包含6345.2秒训练updates估计、1294.4秒planner proxy、357.5秒未实测probing proxy和61.6秒初始化。Root据此批准4小时资源，预留完整native evaluator所需额外margin；这些proxy仍不是end-to-end upper bound，不得降低配置或把cost-only calibration称作质量结果。

## 通过条件

仅当 CPU preparation 为 `PASS`、来源与 isolated dependency import gates 通过、官方配置逐项匹配、真实 optimizer updates 与 timing records 均为 110、其中 timed 为 100、validation 调用数为 0、calibration-only checkpoint 成功保存，且 native planner probe 有 1 warmup + 3 steady samples 时，报告标记 `PASS`。若 planner probe 因时间不足而跳过或不足 3 个 steady samples，训练成本仍会完整记录但总状态为 `PARTIAL`。其它终态保留实际状态，不自动重试，不提交正式训练。
