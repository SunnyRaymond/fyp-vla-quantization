# LpWM PushT 最小官方复现协议

## 目的与边界

官方仓库没有在 README 或 GitHub Releases 中提供可直接运行的 LpWM checkpoint；作者 publication page 与定向 Hugging Face 搜索也未找到公开 LpWM 权重。因此本协议从头训练，不把阅读或代码检查算作复现。

本次只跑官方 `scripts/reproduce_pusht.sh` 中 `mlp_var, D=384` 的 sparse 与 dense 两个 cell。它检验论文中间容量 predictor 路线的最小端到端流程。它不是完整 Figure 1(b) 网格，也不是三评估种子的论文主结果。

## 固定代码与数据

- Paper：arXiv `2608.22764v1`。
- 官方代码：`YilunKuang/lpworldmodel` commit `bdd812d9432cccda8c350086006401b436f91982`。
- 数据来源：DINO-WM 官方 README 所指的 OSF PushT archive（item `k2d8w`），原始 `official-pusht-noise.zip`（2,785,304,515 bytes）保留于 `/scratch/users/ntu/yguo017/dino-wm-wall/artifacts/pusht-assets-23952308.pbs101/downloads/`。旧 `PUSHT_ASSETS_READY` marker 指向的 `23952308.pbs101` 在 extraction 前失败，不能作为完整解包证据。CPU PBS `25570558.pbs101` 从同一 archive 恢复了全部缺失 train clips：18,685 个 archive episode ID 与 metadata trajectories 一一覆盖；保留32个同size已有文件、不覆盖，新增18,653个 clip（2,163,166,894 bytes）。官方 train/validation slicer `__getitem__` 与 episode 9956、2930、18684 的 `get_frames` 均实际读取成功。
- 该 archive 的官方 loader 读取 train 18,685 条轨迹、validation 21 条。DINO-WM 论文文本/表格列出18,500 samples；我们保留这个计数差异，不把它当作数据转换证据。
- 不下载或复用第三方 LpWM checkpoint；两臂都使用官方 from-scratch encoder 与 predictor。

## 训练

两臂共同固定：PushT，`frameskip=5`，`num_hist=3`，scratch ViT CLS encoder，`D=384`，`mlp_var` predictor，RDMReg，2 epochs，batch 64，20 data workers，训练 seed 0。具体超参数严格取 pinned 官方复现脚本在对应 cell 中的值：

CPU PBS 在 pinned source 上调用官方 `load_pusht_slice_train_val` 并构造 batch-64 `DataLoader`：train 为 18,685 trajectories、1,981,721 windows、30,965 batches/epoch，即每臂 2 epochs 共 61,930 updates；validation 为 21 trajectories、2,115 windows、34 batches。CPU-only job `25570558.pbs101` 从同一官方 OSF archive 补齐 18,653 个缺失 train clips，并通过 train/valid slicer 实际 `__getitem__` 和三个 `get_frames` 读取门；PBS `Exit_status=0`、`Stageout_status=1`，小型 recovery manifest 已取回。成本校准 job `25571046.pbs101` 每臂完成严格 110 个真实 updates（10 warmup+100 timed）、一次 validation 与 checkpoint 保存，得到均值 update 0.4262 秒（sparse）/0.4214 秒（dense）；更新耗时外推分别约 7.33/7.25 小时。native planning 成本 job `25571238.pbs101` 在 sparse 校准权重上完成一个完整 case，耗时 204.408 秒；50×线性 proxy 为 2.839 小时，不是实测 50-case walltime 或保证上界。parent 根据实测成本在 `RESOURCE_AMENDMENT.md` 批准正式每臂独立 12 小时 allocation（1 GPU、16 CPU、110 GB），内部 deadline 42,600 秒；科学规模与参数未缩减。校准模型权重只用于成本测量，不是正式结果。

| Arm | Link / target | `reg_weight` | `mup_lr` | Run name |
|---|---|---:|---:|---|
| Sparse LpWM | `reprelu`, `p=1`, `mu=0` | 0.1 | 5e-4 | `repro_sparse_pusht_mlp_var_pd384` |
| Dense control | `identity`, `p=2` | 0.01 | 5e-5 | `repro_dense_pusht_mlp_var_pd384` |

逐臂命令（在 pinned repo 根目录执行；`CKPT_BASE` 指向本次独立目录，`DATASET_DIR=/scratch/users/ntu/yguo017/dino-wm-wall/data`）：

```bash
PREDICTOR=mlp_var PROJ_DIM=384 MUP=1 MUP_LR=5e-4 REG_WEIGHT=0.1 MU=0 SEED=0 RUN_NAME=repro_sparse_pusht_mlp_var_pd384 REGULARIZER=rdmreg CKPT_BASE=/scratch/users/ntu/yguo017/experiment/reproduction/lpwm/runs WANDB_MODE=offline scripts/train.sh pusht 5 3 2 64 reprelu cls 1 b
PREDICTOR=mlp_var PROJ_DIM=384 MUP=1 MUP_LR=5e-5 REG_WEIGHT=0.01 SEED=0 RUN_NAME=repro_dense_pusht_mlp_var_pd384 REGULARIZER=rdmreg CKPT_BASE=/scratch/users/ntu/yguo017/experiment/reproduction/lpwm/runs WANDB_MODE=offline scripts/train.sh pusht 5 3 2 64 identity cls 2 b
```

两臂训练超参数按官方 reproduction grid 各自的 tuned cell 取值，因此 `reg_weight` 与 `mup_lr` 并不相同。结果只能解释为这两个官方 cell 的运行结果，不能把它说成固定训练超参数下的单因素因果对照。

## 规划评估

对两个 checkpoint 分别运行官方 `scripts/plan.sh plan_lewm.yaml <run_name> latest 50 10`，显式设 `SEED=99`、相同的 data root 和 checkpoint root。官方配置语义是：每次 CEM solve 使用 300 candidates、top-30 elites、30 inner optimization steps、`H=5`、initial variance scale 1；MPC 最多重规划 10 次，每次执行 5 个 actions。`max_iter=10` 是外层 MPC 上限，不是 CEM 的 inner steps。

```bash
SEED=99 CKPT_BASE=/scratch/users/ntu/yguo017/experiment/reproduction/lpwm/runs DATASET_DIR=/scratch/users/ntu/yguo017/dino-wm-wall/data WANDB_MODE=offline scripts/plan.sh plan_lewm.yaml repro_sparse_pusht_mlp_var_pd384 latest 50 10
SEED=99 CKPT_BASE=/scratch/users/ntu/yguo017/experiment/reproduction/lpwm/runs DATASET_DIR=/scratch/users/ntu/yguo017/dino-wm-wall/data WANDB_MODE=offline scripts/plan.sh plan_lewm.yaml repro_dense_pusht_mlp_var_pd384 latest 50 10
```

按官方 `goal_source=dset` 使用该 LpWM 数据 pipeline 中的 evaluation goals。不得换成项目内 paired CEM task list，也不得按中途结果减少训练 epochs、数据规模、CEM budget 或 eval trajectories。

## 资源与停止规则

- CPU prep 最多 2 小时；首个 CPU allocation 采用已验证的 `normal` queue，1 node、8 CPUs、64 GB、1 小时。
- 初始 2 小时 GPU cap 已由 parent 根据实测成本在 `RESOURCE_AMENDMENT.md` 正式调整：sparse 与 dense 各用独立 12 小时 PBS allocation（1 GPU、16 CPU、110 GB），内部 deadline 42,600 秒，保留 10 分钟 teardown。入口仍请求 `normal`，由 PBS 分配实际队列。
- 已完成不超过 15 分钟的 GPU 成本校准：两臂各严格 110 个真实 updates（10 warmup+100 timed），保留官方训练步骤；另对 sparse 校准权重运行一次完整 native plan case，仅测运行成本，不读取成功率或用于质量选择。两份校准 checkpoint 与正式权重隔离。
- GPU allocation 内每 5 秒采集 GPU utilization 与 VRAM used/total，写入该作业的 `job.log`。
- 每臂 12 小时到限而未完成即记录为 incomplete。不得减少 epochs、data、CEM/MPC budget 或 evaluation count 后续跑来追求通过，也不得把局部输出称作成功复现。

## 通过条件与可比性

一个完整有效 run 要求：两臂 train 与 plan 命令均退出 0；两个 checkpoint 存在；每臂恰有 50 条 ordered native outcomes，且与 native `final_eval/success_rate` 一致；每臂另保存原生 `state_0`/`state_g` 50 项 sidecar；PBS `Exit_status=0`；GPU telemetry 以约 5 秒间隔贯穿运行。只有两臂 state sidecar 逐项相同才称 paired。失败、超时、数据结构不匹配或 telemetry 缺失均保留为 incomplete/invalid。

允许报告这两个官方 `mlp_var, D=384` cell 的训练与评估结果。不要宣称复现整张论文图、三 seed 均值、稳健的 sparse gain 或 predictor replacement。官方 dense arm 用 identity link + Gaussian target，但同样使用 RDMReg/SWD，不等于原 LeWM 的 SIGReg 训练 recipe。

其 PushT 绝对 success rate 不直接与本项目现有 JEPA PushT 结果比较：目标选择、data/task selection、模型训练与评估 harness 必须先证实完全相同。此协议不评估推理延迟、内存、稀疏 kernel 或加速。
