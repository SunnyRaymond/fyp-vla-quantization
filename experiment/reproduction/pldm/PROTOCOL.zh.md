# PLDM Two-Rooms reproduction

本任务复现官方 PLDM 的 **Two-Rooms / Wall** 设置，不将 PushT 当作 PLDM benchmark。source 固定为 `vladisai/PLDM` commit `1bd7e564ecd961205bc18b23067b19e9ca24ac90`，官方训练配置为 `pldm/configs/wall/icml/seqlen90_3M.yaml`，seed 101。配置字段 `epochs=2`、offline dataset、VICReg + IDM；原样保留该配置和官方 `Trainer.train()`。Pinned trainer 从初始 `self.epoch=0` 执行 `range(self.epoch, epochs+1)`，因此更新步数按 3 个 epoch index（0、1、2）估算并如实报告，不改配置以适配措辞。最后一个 epoch 由官方 trainer 调用原生 `validate()`，保留 medium planning 的100个环境、200步、MPPI 2000 candidates、最大horizon96、replan_every=1及batch/chunk20。源码的5个chunks各执行200步且无early exit，合计1000次native `planner.plan`调用。YAML中的`eval_mpcs=20`仅在TrainConfig定义，pinned Python源码没有使用该字段；不能把它解读为20次MPC调用。此项为训练/评估启动前的执行语义澄清，未更改官方预算。除scratch输出路径、数据路径和关闭wandb外，不缩减配置参数。`quick_debug=false`保持官方值。

官方仓库没有 GitHub Release，未找到官方预训练 checkpoint；因此从头训练是本设置的必要部分。Two-Rooms 数据通过官方 `download_all.sh` 指向的 Google Drive archive 获取。Pinned wall README 以 `good_quality_data_no_images.npz` 为例，并要求先渲染；CPU allocation 确认该官方发布成员存在，NPZ headers 给出 3,686,400 个 transition。该成员的生成配置为 `good_quality_data.yaml`：`n_steps=91`、`cross_wall_rate=0.60`、`expert_cross_wall_rate=0`、固定布局、65 像素图像；paper v4 将 best-case 概括为约 3M transitions。Pinned training config 原来指向未发布的 `wall-visual-config_rand_expert_40-v0.npz`，当前 source/archive 没有将旧名称映射到该 released member 的证据。为继续官方 released-data pipeline，只对 offline data path 做 override，使用完整 `good_quality_data` member，不裁剪；最终报告会明确其 count 为 3,686,400，且不声称与旧文件 bit-equivalent 或复现 paper v4 的精确 3M/Table 2 数值。CPU header probe 对 `len_17/33/65` 和 `ds_size_1500K` 的结果只用于区分 sensitivity variants，不选用这些较小/不同 sequence 数据。若官方 renderer 的内存占用过高，chunked renderer 将逐块调用同一 `WallDataset.render_location` 和 `render_walls` 并写入 `.npy` memmaps；官方 renderer 保存 `images.numpy()`，因此状态图像保持 `uint8`，不转为 `float32`。以 3,686,400 帧、65×65、两帧通道估算，状态 memmap 约 31 GB；动作与位置数组沿用原 dtype。首轮 CPU prep 和 GPU calibration 使用 `lazy_load=true`。源码核对确认 native default `lazy_load=false` 只将完整 states 数组作为约31.2 GB 的 `uint8` host ndarray 保留，`__getitem__`仍按样本切片、转 tensor 并送到 device，不整体转为 `float32` 或放入 CUDA。第二轮 `25572720.pbs101` 在64个确定窗口上通过eager/mmap exact gate；六个 raw与normalizer输出的 `WallSample` 字段均相等且RNG状态不变。Root据此批准正式使用`lazy_load=false`。CPU loader预检仍在隔离 config copy里设置`device=cpu`，formal GPU device保持CUDA。`PREPARATION.json`记录实际数据表示与相对pinned config的差异。

CPU preparation 请求 normal queue、4 CPUs、24 GB RAM、最长 2 小时。工作脚本在下载、解压、导入/依赖检查和渲染前，必须验证 `PBS_JOBID`、`PBS_NODEFILE`、当前 hostname 在节点列表内且不是 login/head/submit node；工作在自身期限内结束。依赖只读检查已有 `/scratch/users/ntu/yguo017/lewm-pusht-iteration/venv`，不升级该环境。若 Python 3.11 导入时确实因 mutable dataclass default 失败，只在独立 `staged/` copy 中按报错字段改为相同 constructor 的 `field(default_factory=...)`，逐项记录原行和改后行；`upstream/` pinned source 原样保留。本次额外实际触发 `EvalConfig.probing: ProbingConfig = ProbingConfig()`，resume 会只对这一字段应用 `field(default_factory=ProbingConfig)`，并写入 source identity patch record。已观测 `gdown` 缺少 `bs4` 时，只在本任务 isolated `runtime_overlay` 安装 `beautifulsoup4==4.11.2` 与 `soupsieve==2.4`。官方 requirements 固定 `zarr==2.14.2`、`numcodecs==0.12.1`、`asciitree==0.3.3`、`fasteners==0.18`，因此这些依赖仅在隔离 overlay 中按 pinned versions 安装。真实 import 先观察到 `statsmodels.api` 从继承的 SciPy 1.17.1 导入 `_lazywhere` 失败；修复后又观察到 statsmodels 0.14.4 与继承 pandas 3.0.6 的 `deprecate_kwarg` API 不兼容。官方 `requirements.txt` 固定 `pandas==2.0.1`，且继承 runtime 缺少其官方 pin `pytz==2022.7.1`，故在下一次 15 分钟 CPU resume 中仅用 binary wheels、`--no-deps` 装入 task-specific overlay；NumPy 1.26.4 + SciPy 1.10.0 也留在独立 overlay。当前 resume 对最多 6 次 fresh-import 逐次保留失败报告；只对已观测 Python 3.11 mutable dataclass default 作字段级兼容修复，或对最多 3 个已观测且能在 official requirements 中核实精确 pin 的小型 pure-Python 缺失依赖（`python-dateutil==2.8.2`、`pytz==2022.7.1`、`tzdata==2024.2`、`six==1.16.0`）在 compute allocation 中 binary-wheel、`--no-deps` 安装到独立 overlay。其他错误立即停止；后续 GPU scripts 从 `PREPARATION.json` 读取这些已核实 overlay，并验证实际 import origin/version。上述 repair 均不触碰共享 runtime。不盲装完整 210 项 `requirements.txt`，不修改共享 venv。

Chunked render 会先计时最多 8 个 batch（2,048 帧），按实测吞吐估算完整输出时间；若预估超过当前 allocation 可用时间的 90%，就停止并把耗时与规模记入该次报告，不继续全量渲染。

GPU 训练和原生最终验证使用一个 normal-queue allocation；原始 2 小时资源值保留为 calibration 前的 planning placeholder。完整成本 calibration 后，root 批准 formal allocation 为 1 GPU、16 CPUs、110 GB RAM、4 小时，`train.py` 内部 timeout 230 分钟并预留 10 分钟收尾/stageout（`formal_resource_review_status=APPROVED_AFTER_CALIBRATION`）。Root 已审阅并批准一次正式提交；正式 PBS ID 与终态将单独记录。任何模型训练/加载与评估只在获批 allocation 执行。每 5 秒把 GPU UUID 对应设备的 utilization 和 memory 使用量记入 job log。训练保持 pinned 官方入口与原生最终 evaluator：

```bash
cd <pinned-source>/pldm
python train.py --configs configs/wall/icml/seqlen90_3M.yaml --values \
  output_root=<remote-root>/checkpoints \
  output_dir=tworooms-seqlen90-3M-seed101 \
  data.offline_wall_config.offline_data_path=<prepared-dataset> \
  data.offline_wall_config.lazy_load=false \
  resume_if_possible=false eval_at_beginning=false eval_during_training=false \
  seed=101 wandb=false
```

唯一 config 改动是数据/输出路径、日志开关、经 exact equality gate 验证的 native eager `lazy_load=false` storage-only 设置，以及显式禁止恢复/初始或中途 validation，以保留 scratch 训练后官方 final `validate()`。训练输出目录必须在启动时不存在；训练算法、数据规模、`epochs` 配置值、网络、planning 与评估环境数均保持官方设置。Pinned code/config 的 MPPI 为 2,000 samples；paper v4 的 Two-Rooms hyperparameter table 报告 500 samples，最终报告会区分 code/config reproduction 与 paper value。Calibrated resource amendment 已由 root 批准；最终 PBS script 经 root 审阅后才提交。

作业超过获批 allocation walltime、allocation/网络/依赖失败、官方评估未完成或没有最终 checkpoint/summary 时，记录实际状态并停止，不缩小 benchmark 后沿用正式复现标签。最终 `SUBMISSION.json` 与 `RESULT.zh.md` 记录 PBS job ID、PBS `Exit_status` / `Stageout_status`、source/data identity、日志、训练 checkpoint、官方 summary，以及完成范围。源码/配置检查仅属于准备证据，不作为 reproduction 成果。

Pinned README 使用 `--config`，而 pinned `ConfigBase` 注册的是 `--configs`；argparse 默认允许唯一长选项前缀，因此 singular `--config` 是有效缩写。runner 显式使用完整 `--configs`；CPU preparation 会分别 composition-check 两种拼写，并确认它们加载相同冻结 YAML 与 dotlist 覆盖。

官方来源：

- [PLDM repository](https://github.com/vladisai/PLDM)
- [PLDM paper, arXiv v4](https://arxiv.org/abs/2502.14819v4)
- [Official Two-Rooms training instructions](https://raw.githubusercontent.com/vladisai/PLDM/1bd7e564ecd961205bc18b23067b19e9ca24ac90/pldm/readme.md)
- [Frozen Two-Rooms config](https://raw.githubusercontent.com/vladisai/PLDM/1bd7e564ecd961205bc18b23067b19e9ca24ac90/pldm/configs/wall/icml/seqlen90_3M.yaml)
- [Official dataset preparation instructions](https://raw.githubusercontent.com/vladisai/PLDM/1bd7e564ecd961205bc18b23067b19e9ca24ac90/pldm_envs/wall/README.md)
- [Official dataset download script](https://raw.githubusercontent.com/vladisai/PLDM/1bd7e564ecd961205bc18b23067b19e9ca24ac90/pldm_envs/wall/presaved_datasets/download_all.sh)
- [Official dataset render script](https://raw.githubusercontent.com/vladisai/PLDM/1bd7e564ecd961205bc18b23067b19e9ca24ac90/pldm_envs/wall/presaved_datasets/render_all.sh)
- [Official image renderer](https://raw.githubusercontent.com/vladisai/PLDM/1bd7e564ecd961205bc18b23067b19e9ca24ac90/pldm_envs/wall/render_images.py)

## Storage-only amendment gate

首轮 `25571685.pbs101` 保留为 `lazy_load=true` 的 PARTIAL 校准（107/110 updates），不覆盖。第二次校准 `25572720.pbs101` 的 native eager `lazy_load=false` exact gate PASS：64个实际窗口 raw 与 normalized 的全部六个 `WallSample._fields` 均逐字段相等，Python/NumPy/Torch CPU/CUDA RNG 状态不变；实际 states uint8 host ndarray 为31,150,080,000字节，实测进程 RSS 约32.11GB。该结果批准正式配置使用 `lazy_load=false`，不改变任何数据值或科学参数。该校准另完成110次 cost-only updates 和1 warmup+3 steady native planner calls，无 native validation 或 success measurement；完整成本、proxy限制和4小时资源理由记录在 `CALIBRATION.25572720.json` 与 `RESOURCE_AMENDMENT.zh.md`。正式 PBS 启动前需快速读取并要求CPU preparation与此校准报告均PASS，同时复核source SHA；作业将快照配置文档、source identities、校准报告和最终结果collector。

## 最终结果采集

正式作业终态后，先用只读 PBS 查询确认实际 `Exit_status` 和 `Stageout_status`；不能把 training subprocess 的返回码当成 PBS 终态。`collect_native_result.py --job-id <actual-job-id> --pbs-exit-status <actual-exit-status>` 只针对冻结的正式 run，读取该 job 的小型 provenance/状态文件、日志首尾各 64 KiB，以及 checkpoint 的文件元数据，不加载权重或数据集。结果写入该 job report 下的 `RESULT.json`。PBS `Exit_status=0`、Stageout 成功和完整 native 证据是分别核实的条件；collector 不替代 Stageout 查询。

CPU 实际 loader 为每 pass 48,000 updates、batch 64；epoch indices 0/1/2 共 144,000 updates，官方最终 `sample_step=9216000` 和 `custom_step=143999`。最终文件为 `epoch=2_sample_step=9216000.ckpt` 与 `summary_epoch=2_sample_step=9216000.json`。Pinned `Trainer.validate()` 在 `Evaluator.evaluate()` 返回后才写 summary，因此要求 summary 新鲜、最终 step 正确且包含六个 `wall_medium` 字段；仅 checkpoint 存在或 process exit 为零不能单独证明完成。长日志中的 eval-start/MPC 耗时消息仅作辅助证据。

`status=COMPLETE/INCOMPLETE` 表示运行证据是否完整；`quality_status` 另报原生指标的 finite/range 一致性，不按成功率挑选结果。零成功率保留为完整运行的真实结果；原生 cross-wall 分母可能为零，非有限值保留并注明质量限制。Calibration checkpoint 不进入正式结果采集。Self-check 已覆盖长日志中间消息不可见、非零 PBS exit、零成功、非有限指标和缺最终 summary；这只是收集逻辑的验证，尚未证明真实正式复现完成。
