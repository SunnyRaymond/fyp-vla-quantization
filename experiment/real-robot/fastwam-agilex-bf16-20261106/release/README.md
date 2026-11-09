# AgileX Piper 实机推理：BF16 Fast-WAM

这个包里是一个可以直接上 AgileX Piper 单臂的 Fast-WAM 策略，会做两件事：

- 柠檬放进盒子：`place the lemon to the box`
- 碗放到平台上：`place the bowl on the platform`

策略的输入和录数据时一样：固定相机一帧、腕部相机一帧，加上当前 7 维状态（6 个关节角 + 夹爪开口）。输出是接下来 32 步的绝对关节目标和夹爪开口，按 10 Hz 执行。

## 1. 模型来历

- 结构：Fast-WAM Optional-IDM。视频分支是 Wan2.2-TI2V-5B 的 5B DiT，动作分支是 1B 的 ActionDiT，两者在每一层做联合注意力，总共约 60 亿参数，权重为 BF16。
- 初始化：用上游发布的 LIBERO checkpoint（`libero_optional_idm_2cam224`）。
- 微调数据：上面两个任务的遥操作示教，每个任务 39 条，共 78 条、7,359 帧，10 Hz，相机分辨率 240×320。
- 训练：全局 batch 32，1 万步（约 43 个 epoch），lr 1e-4（cosine），4×A100-40GB 上跑了 16.4 小时。
  最后 1000 步的平均 loss 是 0.045，其中动作 loss 0.0025。
- 交付的权重是第 10000 步：`step_010000.pt`。

## 2. 包里的文件

| 文件 | 大小 | 作用 |
|---|---|---|
| `step_010000.pt` | 12.0 GB | 模型权重，包含 `mot`（视频+动作 DiT）和 `proprio_encoder` 两部分 |
| `model-cache/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors` | 1.4 GB | Wan2.2 VAE，用来把当前画面编码成潜变量 |
| `dataset_stats.json` | 53 KB | 状态和动作的 min/max，用于归一化和反归一化 |
| `text_embeds/*.pt`、`prompts.json` | 各 2 MB | 两条指令预先算好的 umT5 文本特征。实机端不需要加载 11 GB 的 T5 |
| `policy_config.json` | 小 | 模型结构参数、输入尺寸、默认推理参数 |
| `agilex_policy.py` | 小 | 推理封装：`AgileXFastWAMPolicy.predict(...)` |
| `openloop_check.py` | 小 | 开环检查和自检脚本 |
| `samples/*.hdf5` | 约 41 MB | 每个任务一条原始示教，供自检用 |
| `selftest_reference.json` | 小 | 我们在 A100 上对这两条示教的自检结果，作为对照 |
| `env/` | 小 | 集群上的 `pip freeze` 和版本信息 |
| `SHA256SUMS` | 小 | 所有文件的校验和 |

拷贝完之后先校验文件完整：

```bash
cd agilex_fastwam_bf16 && sha256sum -c SHA256SUMS
```

## 3. 机器和环境

- **显卡**：NVIDIA，显存 16 GB 以上，推荐 24 GB（RTX 3090/4090、A5000/A6000 等）。在 A100 上实测：PyTorch 峰值占用 12.8 GiB，算上 CUDA 上下文，`nvidia-smi` 峰值 13.3 GiB。同一张卡上不要再跑其他大模型。
- **内存和磁盘**：内存 32 GB 以上（权重以 mmap 方式读入后再搬上显卡），磁盘留 15 GB。
- **加载时间**：构造策略对象约 40–110 秒，取决于硬盘速度和系统文件缓存。只在程序启动时加载一次。

安装（和上游 FastWAM 一致，Linux + CUDA 12.8 驱动）：

```bash
conda create -n fastwam python=3.10 -y
conda activate fastwam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/yuantianyuan01/FastWAM.git
cd FastWAM && git checkout 7faa71108368fbb3b6885649f112af607427a2d4 && pip install -e . && cd ..
pip install h5py
```

- 必须用这个 commit。模型结构和 checkpoint 的键名都是按它对齐的。
- 我们集群上的依赖版本在 `env/pip_freeze_cluster.txt` 和 `env/versions_cluster.json`（Python 3.11.7、torch 2.7.1+cu128、torchvision 0.22.1），装不上时可以对照。
- `torchvision` 要 0.17 以上。图像缩放用的是带抗锯齿的双线性插值，和训练时一致。

## 4. 第一步：自检（上机器人之前必做）

自检只读包里两条示教，不碰机器人。它会把整个推理流程跑一遍：加载权重、处理图像和状态、生成动作、反归一化、把相对关节量加回当前状态。然后把预测和示教里真实的关节动作比较。

```bash
cd agilex_fastwam_bf16
python openloop_check.py --bundle . --episodes samples/*.hdf5 --configs idm \
  --reference selftest_reference.json --out selftest_local.json
```

- 最后一行打印 `OPENLOOP_PASS` 就说明环境装对了。
- 判定标准：预测误差至少比"原地不动"低一半；并且关节误差不超过我们 A100 上结果的 1.5 倍再加 0.005 rad。换显卡后 BF16 的数值会有细微差别，这个余量足够覆盖。
- 我们在 A100 上的自检结果：
  - 两条示教共 17 次推理，前 10 步关节平均误差 0.0070 rad（约 0.40°），"原地不动"是 0.068 rad；
  - 夹爪误差 0.31 mm，"原地不动"是 7.0 mm；
  - 单次推理耗时中位数 1.09 秒。
- 输出里的 `latency median` 就是这台机器上的单次推理耗时，第 5 节安排执行节奏时以它为准。

如果打印 `OPENLOOP_FAIL`，或者报错，先看第 9 节"常见问题"，不要上机器人。

## 5. 接入机器人

```python
import sys; sys.path.insert(0, "/path/to/agilex_fastwam_bf16")
from agilex_policy import AgileXFastWAMPolicy

policy = AgileXFastWAMPolicy("/path/to/agilex_fastwam_bf16", device="cuda")
instruction = "place the lemon to the box"          # 或 "place the bowl on the platform"，必须一字不差

while not done:
    fixed = camera_fixed.read()      # (240, 320, 3) uint8，和录数据时 realsense_fixed 存进 HDF5 的数组完全一样
    wrist = camera_wrist.read()      # (240, 320, 3) uint8，对应 realsense_wrist
    state = robot.get_state()        # (7,) float：joint_1..joint_6（rad）+ 夹爪开口（m）
    actions = policy.predict(fixed, wrist, state, instruction)["actions"]   # (32, 7)
    for a in actions[:10]:           # 执行前 10 步，然后重新观测、重新推理
        robot.command_joint_position(a[:6], gripper=a[6])
        sleep_until_next_tick(hz=10)
```

要点：

1. **相机画面要和录数据时一模一样。** 我们训练时直接把 HDF5 里存的数组当 RGB 用。录数据的程序如果存的是 BGR（比如用 OpenCV 读的），推理时也必须原样传 BGR，不要自己转换颜色。分辨率必须是 240×320，相机的摆放位置、朝向和录数据时一致。
2. **状态的单位和顺序**和 HDF5 里的 `observation/state` 相同：前 6 维是关节角（rad），第 7 维是夹爪开口（m，0 到 0.08）。
3. **输出的含义**：`actions[k]` 是 k+1 步之后（即 0.1×(k+1) 秒后）的目标状态，和录数据时的 `action[t] == state[t+1]` 一致。关节是绝对角度（策略内部已把相对量加回当前状态），夹爪是绝对开口。
4. **执行节奏**：每次执行 10 步（1 秒）就重新推理，和我们在 LIBERO 上的设置相同。推理期间机械臂停在上一个目标点等待，最简单也最安全。A100 上默认模式单次推理约 1.0 秒（中位数 0.99 秒，p90 1.05 秒），也就是每执行 1 秒停顿约 1 秒。嫌停顿太长，可以换第 6 节的 `first_frame` 模式（约 0.5 秒）。
5. **指令**只支持上面两条，字符串必须一字不差。换其他说法需要重新算文本特征（要加载 T5），请联系我们。

## 6. 推理参数

默认值写在 `policy_config.json` 里，也可以在构造时覆盖：`AgileXFastWAMPolicy(bundle, mode=..., sigma_shift=..., num_inference_steps=...)`。

我们在 16 条示教、149 个时刻上比较了三种设置（A100，误差是前 10 步的平均绝对误差）：

| 设置 | 做法 | 关节误差 (rad) | 夹爪误差 (mm) | 单次推理 |
|---|---|---|---|---|
| `idm`（默认） | 先生成未来 9 帧视频的潜变量，再据此去噪出动作 | 0.0074 | 0.39 | 0.99 秒 |
| `idm` + `sigma_shift=1.0` | 同上，换一种噪声调度 | 0.0074 | 0.39 | 0.96 秒 |
| `first_frame` | 只看当前帧，不生成未来视频，直接出动作 | 0.0092 | 0.51 | 0.51 秒 |

- **建议先用默认的 `idm`**。`sigma_shift=1.0` 的结果和默认几乎一样，不必改。
- `first_frame` 快一倍，误差大约多 25%，但仍不到"原地不动"的十分之一。默认模式的停顿影响任务时再换它：
  `AgileXFastWAMPolicy(bundle, mode="first_frame")`。
- 去噪步数是 10，和训练时的评测设置相同。
- 随机种子固定为 0，所以同样的输入给出同样的输出。换种子后，前 10 步的关节目标平均变化约 0.004 rad，这是模型本身的采样随机性，比上表的误差还小。

## 7. 安全建议

- 头几次让人守在急停旁，并把机器人侧的速度上限调低。
- 发送前做一次限幅。示教数据中关节角的范围如下，超出太多说明输入出了问题：

  | | joint_1 | joint_2 | joint_3 | joint_4 | joint_5 | joint_6 |
  |---|---|---|---|---|---|---|
  | 最小 (rad) | -0.773 | -0.003 | -1.316 | -0.531 | -0.610 | -1.071 |
  | 最大 (rad) | 0.265 | 1.935 | 0.001 | 0.953 | 1.167 | 0.568 |

  夹爪开口限幅到 [0, 0.08] m。
- 检查每一步要走多远。我们统计了 78 条示教共 7,281 步，每个关节每步（0.1 秒）的变化：

  | | joint_1 | joint_2 | joint_3 | joint_4 | joint_5 | joint_6 |
  |---|---|---|---|---|---|---|
  | 中位数 (rad) | 0.009 | 0.033 | 0.014 | 0.010 | 0.017 | 0.010 |
  | 99% 分位 (rad) | 0.068 | 0.094 | 0.081 | 0.088 | 0.089 | 0.090 |
  | 最大 (rad) | 0.094 | 0.101 | 0.095 | 0.099 | 0.104 | 0.100 |

  示教里任何关节一步都没有超过 0.105 rad（约 6°）。建议机器人侧做两件事：
  - 把单步关节变化限制在 0.12 rad 以内；
  - 如果 `actions[0]` 和当前状态的差，或者相邻两步之差，超过 0.2 rad，就停下来，这多半是相机或状态接错了。

  夹爪是开合式动作，一步可以从全开变到全闭（0.08 m），不需要限速。
- 场景尽量和示教一致：同一张桌子，物体和容器放在示教时的大致区域，光照相近。只有 78 条示教，离示教分布太远时策略不可靠。

## 8. 开环检查结果（集群 A100）

每个任务取 8 条示教（编号 0, 5, …, 35），每隔 10 步让策略看一次当时的原始相机画面和状态，预测 32 步，再和示教里真实执行的动作比较，共 149 次。"原地不动"指一直保持当时的状态，作为对照。

| 任务 | 次数 | 前 10 步关节误差 (rad) | 原地不动 (rad) | 比值 | 前 10 步夹爪误差 (mm) | 原地不动 (mm) |
|---|---|---|---|---|---|---|
| 柠檬放进盒子 | 77 | 0.0077 | 0.108 | 0.072 | 0.38 | 8.8 |
| 碗放到平台 | 72 | 0.0070 | 0.089 | 0.079 | 0.40 | 10.1 |
| 合计 | 149 | 0.0074 | 0.099 | 0.075 | 0.39 | 9.4 |

- 32 步全程的关节误差是 0.0084 rad（原地不动是 0.203 rad）。误差随预测步数缓慢增大：第 1 步 0.0057 rad，第 10 步 0.0083 rad，第 32 步 0.011 rad。
- 夹爪预测值落在 [-0.001, 0.081] m，基本都在 [0, 0.08] 以内，越界部分直接限幅即可。
- 所有输出都是有限值。
- **这些示教都是训练集里的**。这项检查证明两件事：一是推理链路是对的（图像处理、归一化、相对转绝对都和训练一致）；二是模型确实学会了这 78 条示教。它不能说明换了物体摆放位置之后成功率怎样，那只能在实机上测。

## 9. 常见问题

- **`checkpoint/model key mismatch`**：FastWAM 不是 `7faa711` 这个 commit，或者 `policy_config.json` 被改过。
- **找不到 VAE / 尝试联网下载**：没有找到 `model-cache/`。策略默认把 `DIFFSYNTH_MODEL_BASE_PATH` 设为包内的 `model-cache`；如果环境里已经设置了这个变量，就会用环境里的路径，unset 掉即可。
- **显存不够（OOM）**：见第 3 节的显存要求。另外，同一张卡上不要同时跑别的大模型。
- **第一次推理特别慢**：CUDA 初始化和 cuBLAS 选算法，属于正常现象。可以在开始任务前先用 `samples/` 里的一帧空跑一次。
- **动作明显偏、机械臂往奇怪的方向走**：最常见的原因有三个，依次检查：
  1. 两路相机接反了（固定相机必须传第一个参数）；
  2. 颜色通道顺序和录数据时不一致；
  3. 状态单位不对（关节用 rad，夹爪用 m）。
