# 师兄三任务 π0.5 全参微调（2026-11-14）

用 openpi 在 ASPIRE2A 上对 π0.5（`pi05_base`）做全参微调，数据是师兄的三个 AgileX Piper 任务，三个任务合训一个模型。做法和 Fast-WAM 三任务一样：10 Hz、6000 步、全局 batch 32。

远端目录：`/scratch/users/ntu/yguo017/pi05-agilex-3task-20261114`
环境：`/scratch/users/ntu/yguo017/openpi-pi05`（11-09 建，openpi 215abfb，JAX 0.5.3，见 `experiment/reproduction/openpi-pi05-env-20261109/`）

## 数据

| 任务 | 指令 | 原始 episode | 原始帧数（30 Hz） | 10 Hz episode | 原始数据位置 |
|---|---|---|---|---|---|
| cluster_grasp | place the red apple and green cube to the box | 80 | 28,179 | 240 | Fast-WAM 上传的 `fastwam-agilex-3task-20261110/raw` |
| cabinet_lemon | open the bin and put the lemon in it | 50 | 13,944 | 150 | 本目录 `raw/`（11-14 SFTP 上传） |
| circle | place the cup in the tape | 50 | 9,969 | 150 | 本目录 `raw/` |
| 合计 | | 180 | 52,092 | 540 | |

两个上传目录各有 `manifest.json`（每个文件的 sha256），准备作业在转换前全部核对一遍。

## 转换语义（`convert_openpi.py`）

- 30 Hz 降到 10 Hz，方法和 Fast-WAM 相同：每条原始 episode 按起始帧 0、1、2 拆成三条 10 Hz episode，三个相位正好覆盖全部原始帧，一帧不丢。
- 10 Hz 第 k 帧对应原始帧 t = p + 3k。状态取 `state[t]`，动作取 `action[min(t+2, T-1)]`，也就是 `state[t+3]`（下一个 10 Hz 时刻的状态）。
- 状态和动作都是 7 维：6 个关节（rad）加夹爪开口（m）。
- 图像保持 240×320，以 PNG 形式嵌进 parquet（LeRobot 的 `image` 特征，和 openpi 的 LIBERO 示例一样）。训练时不需要解码视频，容器缺 FFmpeg 的问题也碰不到。缩放到 224×224（保持长宽比、补边）由 openpi 自己做。
- 输出是 LeRobot v2.1 数据集 `local/agilex3_10hz`，位于 `data/lerobot/local/agilex3_10hz`。

## 训练配置（`agilex_openpi.py` 里的 `pi05_agilex3`）

- 模型：`Pi0Config(pi05=True, action_horizon=10)`，即每次预测 10 步、1 s 的动作块。状态按 π0.5 默认方式离散化后放进 prompt。从 `pi05_base` 起训。
- 输入输出沿用 openpi 的 LIBERO 变换：
  - 固定相机进 `base_0_rgb`，腕部相机进 `left_wrist_0_rgb`，右腕槽位置空并屏蔽。
  - 动作块中 6 个关节转成相对当前状态的增量，夹爪保持绝对值（`extra_delta_transform=True`，openpi 处理绝对关节目标的标准做法）。推理时自动转回绝对关节目标。
  - 归一化用 π0.5 默认的分位数归一化，统计量由准备作业从这批数据算出。Piper 不在 pi05_base 的预训练机型里，所以不能复用预训练的统计量。
- 超参参照官方真机配置 `pi05_droid_finetune`：
  - batch 32，AdamW（b1 0.9，b2 0.95，梯度裁剪 1.0），EMA 0.99。
  - 学习率：warmup 1000 步到 2.5e-5，再余弦衰减，在第 6000 步降到 2.5e-6。官方配置是 20k 步，衰减区间设到 30k；我们按用户的决定只训 6000 步，并把衰减终点挪到 6000。
- 精度：openpi 的 JAX 训练把可训练权重、AdamW 两个矩、EMA 都放在 fp32，前向和反向用 bf16 计算，也就是标准的混合精度全参训练。
- 6000 × 32 = 19.2 万个样本，每个 10 Hz 窗口约被看到 3.7 次。

## 排卡

ASPIRE2A 的 GPU 节点全是 4×A100-40GB，openpi 的训练脚本不支持多节点。3.35B 参数 × 每参数约 16 字节（fp32 权重 + Adam 两个矩 + 梯度），再加 EMA，常驻约 67 GB，所以只能用 g3 单节点 4 卡，`fsdp_devices=4`，每卡状态约 17 GB。g2（2–3 卡）切两份后每卡约 34 GB，放不进 40 GB 的卡；g4 需要跨节点。

| 作业 | 队列 | 资源 | walltime | 内容 |
|---|---|---|---|---|
| prep（`prep.pbs`） | CPU | 16 核，96 GB | 3 h | sha256 核对 → 转换 → `compute_norm_stats` |
| train（`train.pbs`） | g3 | 4×A100-40GB，64 核，440 GB | 8 h | afterok prep；FSDP-4 训练 6000 步 |

按 FLOPs 估算的速度是 1–1.5 s/步。实测 1.0–1.1 s/步，整个作业 1 h 56 min，8 h 的 walltime 远远够用。超时后重新提交会从最新的 checkpoint 自动续训（`--resume`）。

GPU 利用率和显存每 30 s 写进作业的 `job.log`。JAX 预先占用每张卡 90% 的显存，所以 `memory.used` 显示的是预留量，不是实际占用。

## 操作

```bash
"../../../nscc-access/.venv/Scripts/python.exe" -Xutf8 -B control.py upload
```

```bash
"../../../nscc-access/.venv/Scripts/python.exe" -Xutf8 -B control.py upload-data
```

```bash
"../../../nscc-access/.venv/Scripts/python.exe" -Xutf8 -B control.py submit --jobs prep train
```

```bash
"../../../nscc-access/.venv/Scripts/python.exe" -Xutf8 -B poll.py --until train --interval 300
```

`control.py status` 查看进度，`control.py pull` 把小结果拉到 `pulled/`（包括 `loss_steps.txt`，即每 20 步一行的 loss）。

## 运行记录（11-14）

| 作业 | Job ID | 用时 | 结果 |
|---|---|---|---|
| prep | 26247497 | 23 min | exit 0：sha256 全部通过，生成 540 条 episode、52,092 帧，parquet 共 7.84 GiB；三个任务的动作与下一时刻状态的最大差都是 0 |
| train | 26247498 | 1 h 56 min | exit 0：checkpoint 2000、4000、5999 都已存好 |

各任务的 10 Hz episode 长度：cluster_grasp 87–181 帧，cabinet_lemon 76–131 帧，circle 47–95 帧。

训练时间线（SGT）：
- 12:03 作业开始。
- 12:10:45 训练状态初始化完成。启动用了约 7 分钟，包括读数据、初始化和 JIT 编译。
- 13:57:34 最后一份 checkpoint 存完。

6000 步实际用了 1 h 47 min，每次异步存 checkpoint 约 20 s。训练期间 4 张卡的利用率一直是 99%，CPU 实际只用到约 4.4 核。g3 每张卡强制配 16 核，所以申请的 64 核大部分闲着。

训练 loss 是 flow-matching 损失，每 500 步取一次平均（`pulled/loss_steps.txt`，每 20 步记一行）：

| 步数 | 0–500 | 500–1k | 1k–1.5k | 1.5k–2k | 2k–2.5k | 2.5k–3k | 3k–3.5k | 3.5k–4k | 4k–4.5k | 4.5k–5k | 5k–5.5k | 5.5k–6k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| loss | 0.0313 | 0.0194 | 0.0171 | 0.0142 | 0.0122 | 0.0108 | 0.0090 | 0.0078 | 0.0065 | 0.0055 | 0.0048 | 0.0044 |
| 梯度范数 | 0.312 | 0.138 | 0.115 | 0.107 | 0.096 | 0.096 | 0.089 | 0.089 | 0.079 | 0.074 | 0.073 | 0.071 |

- loss 从第 0 步的 0.070 降到最后的 0.0037。
- 第 1000 步之后梯度范数最大只有 0.152，全程没有尖峰。
- 到第 6000 步 loss 还在下降。这是训练集上的 loss，没有留验证集，不能据此判断真机效果。

## 输出

checkpoint 在 `checkpoints/pi05_agilex3/bs32_6k/{2000,4000,5999}/`。openpi 把最后一步存成 `num_train_steps - 1`，所以第三份是 5999。每份包含：

- `params/`：EMA 权重，fp32，12 GB。推理用这个。
- `train_state/`：优化器状态，31 GB，只在续训时用到。
- `assets/local/agilex3_10hz/norm_stats.json`：归一化统计量。

部署时起 policy server：

```bash
python agilex_openpi.py serve policy:checkpoint --policy.config=pi05_agilex3 --policy.dir=<.../bs32_6k/5999>
```

机器人端用 openpi-client 走 websocket。每次请求发 `observation/image`（固定相机 RGB）、`observation/wrist_image`（腕部相机 RGB）、`observation/state`（7 维）和 `prompt`（任务指令），返回的 `actions` 是 10×7 的绝对关节目标加夹爪，按 10 Hz 执行。

## 交付包（11-14）

给师兄的包是 `release/agilex3_pi05`（集群）。下载到本地的副本在 `deliver/agilex3_pi05`，这个目录被 gitignore 了。机器人侧的说明是 `release/README.md`，它会被上传进包里。

| 文件 | 内容 |
|---|---|
| `checkpoint/params`、`checkpoint/assets` | 第 5999 步的 EMA 权重（fp32，12 GB）和归一化统计量 |
| `openpi_cache/big_vision/paligemma_tokenizer.model` | 分词器，推理不用联网 |
| `agilex3_pi05.py` | 包内自带推理配置，提供三个入口：`AgileXPi05Policy.predict`、`selftest`、`serve`（openpi websocket server） |
| `samples/` | 每个任务第 0 条原始示教 |
| `selftest_reference.json` | A100 上 30 次预测（固定噪声）的参考结果 |
| `training_code/` | `agilex_openpi.py` 和 `convert_openpi.py` |
| `env/` | 集群环境的依赖版本 |
| `SHA256SUMS` | 校验和，共 27 个文件、12.7 GB |

`deliver.pbs` 负责打包，作业 26249369 跑在 gdev，用时 22 min。几个阶段：

- 拷贝 checkpoint、分词器、脚本和样例。
- 用 `release_tools.py` 统计环境版本，并统计全部示教的关节范围和 10 Hz 单步变化，结果写进 `results/deliver/robot_stats.json`。
- 自检。容器里只有包和 openpi venv，不设 `OPENPI_DATA_HOME`，工作目录设成包本身。

自检的结果：

| 任务 | 次数 | 关节误差 (rad) | 原地不动 (rad) | 比值 | 夹爪 (mm) |
|---|---:|---:|---:|---:|---:|
| cluster_grasp | 11 | 0.0148 | 0.1171 | 0.126 | 1.13 |
| cabinet_lemon | 13 | 0.0153 | 0.1095 | 0.139 | 2.54 |
| circle | 6 | 0.0199 | 0.1456 | 0.137 | 1.27 |
| 合计 | 30 | 0.0160 | 0.1195 | 0.134 | 1.77 |

性能和复现性：

- 加载 44–75 s。第一次调用要做 XLA 编译，两次测到 38 s 和 196 s。
- 之后单次推理的中位数是 0.081 s，其中模型 56 ms。显存峰值 6.4 GiB。
- 第二遍按说明里的命令带 `--reference` 重跑，预测和参考平均相差 2.6e-7 rad，结果 PASS。

这些示教都在训练集里，所以这项自检只能说明推理链路正确，说明不了泛化。

操作：先运行 `control.py submit --jobs deliver`，作业完成后运行 `control.py readme` 上传说明，最后运行 `control.py fetch` 下载，下载时逐个文件核对 sha256。
