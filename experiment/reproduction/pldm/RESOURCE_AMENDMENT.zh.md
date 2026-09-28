# PLDM 正式运行资源修订

## 决策

正式复现资源从最初占位的 2 小时调整为 4 小时 PBS allocation，提交设置为 normal queue、1 GPU、16 CPU、110 GB RAM。作业内部训练与原生最终评估的 deadline 为 230 分钟，预留 10 分钟用于保存、日志收尾和 stage-out。该配置已在成本校准后获批；校准是资源规划依据，不是 PLDM 质量结果，也不代表完整正式运行已完成。

| 项目 | 正式设置 |
|---|---|
| GPU allocation | 1 GPU，A100 型号由 PBS 实际分配决定 |
| CPU / RAM | 16 CPU / 110 GB |
| PBS walltime | 04:00:00 |
| 内部运行时限 | 03:50:00，信号中断后最多 30 秒强制结束 |
| 收尾与 stage-out 余量 | 00:10:00 |

## 科学工作量保持不变

正式作业从头开始训练，不加载或续训 calibration checkpoint。继续使用 pinned PLDM source/config、seed 101、`quick_debug=false`、完整 released `good_quality_data_no_images.npz` 数据集、模型、loss、`torch.compile` 与 native evaluation。配置中的 epochs 保持为 2；官方训练循环从初始 epoch 0 运行到 2（含 0、1、2），即 3 passes、144,000 次 optimizer updates。数据不裁剪，native final validation 也不缩减：100 个环境、每批 20 个环境、每环境最多 200 步、MPPI 2,000 samples、horizon 96、每步 replanning；官方实现对应 1,000 次 planner 调用。原生 predictor/encoder probing 的 20/30 epochs 保持不变。

运行采用 `data.offline_wall_config.lazy_load=false`。这是唯一获批的 storage/materialization override：官方默认即为 eager 模式，states 保持 uint8 host ndarray，不将整个数据集转成 float 或加载到 GPU。正式训练前的实际 Trainer 检查测得 states 数组为 `[40960, 90, 2, 65, 65]`、31,150,080,000 bytes；同一个 Trainer/Normalizer 下，64 个跨首尾及轨迹边界的窗口在 eager ndarray 与只读 mmap 表示之间，`WallSample` 六字段的原始值和归一化值均相等，Python、NumPy、Torch CPU/CUDA RNG 状态未变化。该门禁通过后才采用 native eager 路径。

数据使用官方公开的完整 released member，共 3,686,400 transitions。它不是未公开的历史 `wall-visual-config_rand_expert_40-v0.npz` 文件；两者的等价关系未证实。论文称数据约 3M，因此此运行不宣称与论文 Table 2 的数据规模或数值逐项复现。

## 校准证据与限制

成本校准作业 `25572720.pbs101` 终态为 Exit 0、Stageout 1，实际使用 `gdev` queue。存储等价门禁通过。完整 calibration-only 训练段包含 10 次 warmup 和 100 次计时 update；训练 update 均值约 0.0441 秒，训练 loader 等待均值约 0.00585 秒。按 144,000 次正式 updates 线性估算为约 6,345 秒；实际 Trainer 初始化约 61.6 秒。该 calibration checkpoint 与正式训练隔离，不会复用。

成本 probe 在 `model.eval()` 下对官方 MPPI planner 完成 1 次 warmup 和 3 次 steady cost-only 调用，输入为 B=20、H=96、S=2,000，steady 均值约 1.294 秒，输出形状及有限值检查通过。按 1,000 次全 horizon-96 调用计算的 planning-only proxy 约 1,294 秒；真实评估的 horizon 会变化，并包含环境推进、trained-prober projection、probing、报告和其他工作，因此这不是完整评估时长上界。

官方 probing metadata 对应 20 个 predictor epochs、30 个 encoder epochs，并报告 8,114 个训练/验证/形状探测 batch-work 单位。以训练 update 成本换算的 probing proxy 约 358 秒，未实测，且不是上界。将训练估算、planner proxy、probing proxy 与初始化相加约为 8,059 秒（约 2 小时 14 分钟），仍未完整计入环境启动、训练中的编译波动、完整 prober 和绘图、最终 checkpoint、原生评估及 I/O。4 小时 allocation 留有约 1 小时 46 分给这些未覆盖部分；最终是否完成以实际 PBS 与原生输出为准。

## 正式运行与结果判定

作业使用新 scratch 输出目录和固定 seed，从头执行官方训练，保留 native final validation。禁止自动缩短数据、epoch、probing 或 native evaluation；若 230 分钟内部 deadline 到期，保留当时真实 checkpoint、日志和阶段状态，不把部分结果标为完整复现。

结束后先以只读 PBS 终态确认真实 `Exit_status` 与 `Stageout_status`，再用 `collect_native_result.py` 读取正式 run 的最终 summary、checkpoint metadata 和必要的小型日志证据。完整复现需要正式终态成功及官方 final summary 中的预期 `custom_step` 和六项 `wall_medium_*` 指标；校准报告不能替代这些证据。PBS、校准、CPU preparation 与本次正式运行的记录均保留各自身份和路径。
