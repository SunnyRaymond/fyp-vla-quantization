# AgileX 实机数据上的 BF16 Fast-WAM 微调

目标：用 AgileX Piper 两个 pick-and-place 任务（柠檬放进盒子、碗放到平台）的示教数据，从 LIBERO Optional-IDM
checkpoint 微调出实机用的全精度 Fast-WAM。它和 LIBERO 上所有量化实验用的是同一个结构，后续的量化流水线可以直接套用。

## 数据

- 本地 `agilex pepper training data/` 下有两个 run，各 40 条；两个 run 的 `episode_000039.hdf5` 都是 0 字节，
  实际可用 78 条、7,359 帧，10 Hz。
- `action[t] == state[t+1]`：动作是下一帧的绝对关节位置（6 个关节，单位 rad）加夹爪开口（0–0.08 m）。
- `convert_agilex.py` 把数据转成 LeRobot v2.1：parquet 存 `observation.state.{joints,gripper}` 和
  `action.{joints,gripper}`，固定相机存为 `observation.images.image`、腕部相机存为 `observation.images.wrist_image`，
  视频为 10 fps 的 mp4，GOP 为 2。

## 训练设置（`configs/`）

- 双相机各缩放到 224×224 后横向拼接；每个样本 33 帧，对应 32 步动作和 9 帧视频，和 LIBERO 完全相同。
- 关节动作相对当前状态（`RelativeJointTransform`），夹爪保持绝对值；归一化用 min/max；state 补零到 8 维，
  以匹配 checkpoint 的 proprio 输入。
- 初始化：`libero_optional_idm_2cam224.clean.pt`。由 `fw_agilex.create_model` 在 DeepSpeed 包装模型之前加载，
  并严格核对键名。上游的 `resume=<.pt>` 在 `accelerator.prepare` 之后才加载，ZeRO 的 fp32 主权重仍是随机初始化，
  第一步更新会把权重写回随机值。
- g3 节点 4×A100-40GB，ZeRO-2；优化器拆成 16 个参数组（每次只物化一组的 fp32 梯度）。
  ZeRO-2 常驻约 28 GiB（bf16 权重、fp32 主权重、Adam），而单个样本经过 30 层联合 MoT 的激活值超过剩余的 8 GiB
  （实测 batch 1 也 OOM）。上游 MoT 自带的检查点被禁用（"not supported by the compiled tensor core"，冲突在
  torch.compile 路径，这里没开），所以由 `fw_agilex` 给每次 `_forward_joint_layer` 调用套上非重入式
  `torch.utils.checkpoint`，代价是计算量多约三分之一。
- 即使加了检查点，不卸载时也只有每卡 batch 1、不做梯度累积才能放下：ZeRO-2 在非累积边界的 micro-step 上会额外复制
  一份本卡负责的梯度（`all_grad_tensors`），batch 越大激活值也越多。所以把优化器卸载到 CPU（ZeRO-Offload，
  `contiguous_gradients` 必须开）：fp32 主权重和 Adam 状态放在节点内存里，用 DeepSpeedCPUAdam 更新（现场 JIT
  编译，失败时退回 torch AdamW），GPU 上只留 bf16 权重、梯度桶和激活值。
- 全局 batch 32，每卡先用 8×1，遇到 OOM 依次退到 4×2、2×4。共 10k 步，lr 1e-4（cosine，5% warmup），
  每 2k 步保存一次权重；walltime 20 h。

## 作业

1. `control.py upload`，然后 `control.py upload-data`（78 个 HDF5，约 1.7 GB，附 sha256 清单）。
2. `control.py submit --jobs prep train`：
   - prep 是 CPU 作业：转换数据、校验视频解码并选出解码后端、在 CPU 上算 T5 提示词缓存、导出组合后的配置、
     计算统计量并检查样本、在 CPU 上构建模型并核对 checkpoint；
   - train 是 GPU 作业，以 afterok 依赖 prep。
3. `poll.py --until train` 在后台监控。

结果写在 `/scratch/users/ntu/yguo017/fastwam-agilex-bf16-20261106/runs/agilex_optional_idm_2cam224/`：
`checkpoints/weights/step_*.pt` 和 `dataset_stats.json`（实机推理时反归一化要用）。

## 结果（作业 26129524，2026-11-06 16:19 → 11-07 08:39 SGT）

- 一次跑通：每卡 8×1，全局 batch 32，10,000 步（约 43 个 epoch），walltime 16:24:47，平均约 5.9 秒/步。
  每卡显存约 22.3 GB / 40 GB，GPU 利用率均值 54%（包括 CPU 端 Adam 更新和存盘时的空闲）。
- loss 按每千步取均值：

  | 步数 | loss | action | video |
  |---|---|---|---|
  | 1–1k | 0.344 | 0.0821 | 0.262 |
  | 1k–2k | 0.218 | 0.0337 | 0.184 |
  | 4k–5k | 0.112 | 0.0144 | 0.098 |
  | 7k–8k | 0.060 | 0.0047 | 0.055 |
  | 9k–10k | 0.045 | 0.0025 | 0.043 |

  全部记录在 `pulled/loss_curve.tsv`。没有划验证集（`val_set_proportion 0`），loss 只反映对 78 条示教的拟合程度。
- 权重：`checkpoints/weights/step_{002000,004000,006000,008000,010000}.pt`，每个 12,041,737,379 字节，
  格式与初始化用的 `libero_optional_idm_2cam224.clean.pt` 相同（`mot` + `proprio_encoder`）。
  `checkpoints/state/step_010000/` 是最后一步的 DeepSpeed 完整状态（82 GB），可以从这里接着训。

## 开环检查与实机交付包（2026-11-07）

- `control.py race-openloop` 提交同一个 `openloop.pbs` 两份：一份走 gdev（walltime 1.5 h），一份走 g1（walltime 3 h）。
  - 两份作业启动时抢 `control/openloop_lock_<id>`，先拿到锁的那份用 qdel 取消另一份；`poll_openloop.py` 作为后备也会取消。
  - 这次 g1 那份 26137762 先启动（gdev 因 GPU 配额排队，被取消），用时 15 分钟。
- 作业在 `release/agilex_fastwam_bf16/` 组装交付包：
  - `agilex_policy.py`、`openloop_check.py`、`policy_config.json`（由 composed config 导出）、`dataset_stats.json`；
  - 两条指令的 umT5 文本特征，所以实机端不需要加载 T5；
  - Wan2.2 VAE 和 `step_010000.pt`（集群上是符号链接）；
  - 两条原始样例示教和自检参考值。
  - 开环检查完全从这个包加载，所以同时验证了包本身能用。
- 开环检查：每个任务 8 条原始 HDF5 示教（编号 0, 5, …, 35），每 10 步查询一次，共 149 次，预测 32 步，与示教动作比较。
  - 默认 idm 模式：前 10 步关节 MAE 0.0074 rad，"原地不动"基线 0.099 rad，比值 0.075；夹爪 0.39 mm，基线 9.4 mm。两个任务都通过（判据是比值 ≤ 0.5）。
  - `sigma_shift=1.0` 与默认无差别；`first_frame` 关节 MAE 0.0092 rad、0.51 秒/次（idm 是 0.99 秒/次）。
  - 换随机种子带来的差异约 0.004 rad。
  - 显存峰值 12.8 GiB（nvidia-smi 13.3 GiB）。全部结果在 `pulled/results/openloop/report.json`。
- `manifest.pbs`（CPU 作业）：生成 `SHA256SUMS`，并统计 78 条示教的每步关节变化。99% 分位 0.07–0.09 rad，最大 0.104 rad；结果在 `step_delta.json`。
- `fetch_bundle.py` 把交付包下载到 `deliver/agilex_fastwam_bf16/`（已加入 .gitignore），逐个文件核对 sha256。
  实机用的说明是交付包里的 `README.md`，源文件在 `release/README.md`。
