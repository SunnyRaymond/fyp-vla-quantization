# AgileX Piper 实机推理：三任务 π0.5（全参微调）

这个包里是一个可以直接上 AgileX Piper 单臂的 π0.5 策略。一个模型同时会做三件事：

| 任务 | 内容 | 指令（必须一字不差） |
|---|---|---|
| cluster_grasp | 从一堆物品里挑出红苹果和绿方块，放进盒子 | `place the red apple and green cube to the box` |
| cabinet_lemon | 打开箱子，把柠檬放进去 | `open the bin and put the lemon in it` |
| circle | 把杯子放到胶带圈里 | `place the cup in the tape` |

策略的输入和录数据时一样：固定相机一帧、腕部相机一帧，加上当前 7 维状态（6 个关节角 + 夹爪开口）和任务指令。输出是接下来 10 步的绝对关节目标和夹爪开口，**按 10 Hz 执行**（共 1 秒）。录数据是 30 Hz，见第 5 节第 3 点。

## 1. 模型来历

- **结构**：π0.5（Physical Intelligence 的 openpi），PaliGemma 视觉语言模型加动作专家，约 33.5 亿参数。动作用 flow matching 生成，10 步去噪。
- **初始化**：openpi 官方的 `pi05_base`。
- **微调数据**：三个任务的遥操作示教，共 180 条、52,092 帧（30 Hz，相机 240×320）。

  | 任务 | 示教条数 | 帧数 |
  |---|---:|---:|
  | cluster_grasp | 80 | 28,179 |
  | cabinet_lemon | 50 | 13,944 |
  | circle | 50 | 9,969 |

  每条示教按起始帧 0/1/2 降采样成三条 10 Hz 示教，一帧都没丢。10 Hz 第 k 帧的动作是下一个 10 Hz 时刻（0.1 秒后）的状态。
- **输入输出处理**：沿用 openpi 的 LIBERO 设置。
  - 固定相机进 `base_0_rgb`，腕部相机进 `left_wrist_0_rgb`，右腕槽位空着。图像在模型内部补边缩放到 224×224。
  - 6 个关节在模型内部用相对当前状态的增量表示，输出时已经加回当前状态，所以拿到的是绝对目标。夹爪一直是绝对开口。
  - 状态和动作的归一化统计量是从这批数据算的，已经放在 `checkpoint/assets/` 里。
- **训练**：openpi 的 JAX 训练脚本，全参微调。
  - 全局 batch 32，6000 步。
  - 学习率先 warmup 1000 步到 2.5e-5，再余弦衰减到 2.5e-6。
  - EMA 0.99，权重 fp32，计算 bf16。
  - 4×A100-40GB 上跑了 1 小时 47 分钟。
  - 最后 500 步平均 loss 0.0044。全部示教都用于训练，没有留测试集。
- **包里的权重**：第 5999 步，也就是最后一步，取 EMA 权重。

## 2. 包里的文件

| 文件 | 大小 | 作用 |
|---|---|---|
| `checkpoint/params/` | 12 GB | 模型权重（fp32 EMA，openpi/orbax 格式）。加载时自动转成 bf16 |
| `checkpoint/assets/local/agilex3_10hz/norm_stats.json` | 小 | 状态和动作的归一化统计量 |
| `openpi_cache/big_vision/paligemma_tokenizer.model` | 4 MB | PaliGemma 分词器。放在包里，推理时就不用联网下载 |
| `agilex3_pi05.py` | 小 | 推理封装 `AgileXPi05Policy`、自检和 policy server（**实机用它**） |
| `samples/*.hdf5` | 约 205 MB | 每个任务第 0 条原始示教（30 Hz），供自检用 |
| `selftest_reference.json` | 小 | 我们在 A100 上对这三条示教的自检结果，作为对照 |
| `training_code/` | 小 | 训练配置 `agilex_openpi.py` 和数据转换 `convert_openpi.py`，只在想接着训时才用 |
| `env/` | 小 | 集群上的依赖版本和 openpi commit |
| `SHA256SUMS` | 小 | 除本 README 外所有文件的校验和 |

拷贝完之后先校验文件完整：

```bash
cd agilex3_pi05 && sha256sum -c SHA256SUMS
```

## 3. 机器和环境

- **显卡**：NVIDIA，显存 12 GB 以上比较稳妥（RTX 3060 12G、4070 及以上、3090/4090、A5000/A6000 等）。
  - A100 上实测推理峰值占用 6.4 GiB，算上 CUDA 上下文约 7 GB。
  - JAX 默认会预先占掉显卡 75% 的显存。8–10 GB 的卡按 75% 算放不下，要先设 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 关掉预分配再运行。
  - 同一张卡上不要再跑其他大模型。
- **系统**：Linux + CUDA 12 驱动，Python 由 uv 管理（openpi 要求 Python 3.11）。
- **耗时**：
  - 加载只在程序启动时做一次。A100 节点上构造策略对象 44–75 秒，取决于硬盘速度。
  - 第一次推理要编译（JAX/XLA），我们两次测到 38 秒和 3.3 分钟。编译时间受 CPU 和机器负载影响很大，等它完成就好。
  - 之后每次推理（A100）中位数 0.081 秒，p90 0.085 秒。其中模型本身 56 毫秒，其余是输入输出处理（图像缩放、分词、反归一化）。换显卡后以自检打印的耗时为准。

安装 openpi。必须用这个 commit，模型结构和数据处理都是按它对齐的：

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi
git checkout 215abfb217dbac7d5f1273282331b9b1866c0479
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

- 之后都用 openpi 的 Python 运行：`/path/to/openpi/.venv/bin/python`，下文简写成 `python`。
- 包可以放在任何位置，不需要放进 openpi 目录。
- 我们集群上的依赖版本在 `env/pip_freeze_cluster.txt` 和 `env/versions_cluster.json`，装不上时可以对照。

## 4. 第一步：自检（上机器人之前必做）

自检只读包里三条示教，不碰机器人。它会把整个推理流程跑一遍：加载权重、处理图像和状态、生成动作、反归一化、把关节增量加回当前状态。然后把预测和示教里真实的动作比较：

```bash
cd agilex3_pi05
python agilex3_pi05.py selftest --reference selftest_reference.json --out selftest_local.json
```

- 最后一行打印 `SELFTEST_PASS` 就说明环境装对了。
- 自检时去噪用的初始噪声是固定的，所以同一台机器每次结果都一样，换机器后也只有数值上的细微差别。
- 判定标准有三条：
  - 每个任务的预测误差至少比"原地不动"低一半；
  - 关节误差不超过我们 A100 上结果的 1.5 倍再加 0.005 rad；
  - 每次预测和我们 A100 上同一时刻的预测，关节平均相差不超过 0.02 rad。
- 我们在 A100 上的自检结果（三条示教，每 10 步推理一次，比较完整的 10 步动作）：

  | 任务 | 推理次数 | 关节误差 (rad) | 原地不动 (rad) | 比值 | 夹爪误差 (mm) | 原地不动 (mm) |
  |---|---:|---:|---:|---:|---:|---:|
  | cluster_grasp | 11 | 0.0148 | 0.1171 | 0.126 | 1.13 | 12.7 |
  | cabinet_lemon | 13 | 0.0153 | 0.1095 | 0.139 | 2.54 | 13.8 |
  | circle | 6 | 0.0199 | 0.1456 | 0.137 | 1.27 | 10.2 |
  | 合计 | 30 | 0.0160 | 0.1195 | 0.134 | 1.77 | 12.7 |

  我们在同一台 A100 上用上面这条命令又跑了一遍，30 次预测和参考结果平均只差 2.6e-7 rad，判定为 `SELFTEST_PASS`。
- 输出里的 `latency median` 就是这台机器上的单次推理耗时，`peak GPU memory` 是显存峰值。

如果打印 `SELFTEST_FAIL`，或者报错，先看第 9 节"常见问题"，不要上机器人。

## 5. 接入机器人

两种接法任选一种。

**接法一：在机器人程序里直接调用**（推理和控制在同一台机器上）

```python
import sys; sys.path.insert(0, "/path/to/agilex3_pi05")
from agilex3_pi05 import AgileXPi05Policy

policy = AgileXPi05Policy()             # 加载包里的 checkpoint/，只做一次
instruction = "place the cup in the tape"   # 三条指令之一，必须一字不差

while not done:
    fixed = camera_fixed.read()      # (240, 320, 3) uint8，和录数据时 realsense_fixed 存进 HDF5 的数组完全一样
    wrist = camera_wrist.read()      # (240, 320, 3) uint8，对应 realsense_wrist
    state = robot.get_state()        # (7,) float：joint_1..joint_6（rad）+ 夹爪开口（m）
    actions = policy.predict(fixed, wrist, state, instruction)   # (10, 7)
    for a in actions:                # 执行完 10 步（1 秒），再重新观测、重新推理
        robot.command_joint_position(a[:6], gripper=a[6])
        sleep_until_next_tick(hz=10)
```

**接法二：policy server**（GPU 机器跑模型，机器人电脑通过网络请求）

GPU 机器上：

```bash
cd agilex3_pi05 && python agilex3_pi05.py serve --port 8000
```

机器人电脑上装 openpi 自带的轻量客户端（只依赖几个小包，不需要 GPU，也不需要装整个 openpi 环境）：

```bash
pip install -e /path/to/openpi/packages/openpi-client
```

```python
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(host="GPU 机器的 IP", port=8000)
out = client.infer({
    "observation/image": fixed,          # (240, 320, 3) uint8，原样发送，不要自己缩放
    "observation/wrist_image": wrist,
    "observation/state": state,          # (7,) float32
    "prompt": instruction,
})
actions = out["actions"]                 # (10, 7)
```

server 模式不检查输入的尺寸和类型，发送前请自己确认和接法一要求的一样。

要点：

1. **相机画面要和录数据时一模一样。** 训练时直接把 HDF5 里存的数组当图像用。录数据的程序如果存的是 BGR（比如用 OpenCV 读的），推理时也必须原样传 BGR，不要自己转换颜色。分辨率必须是 240×320，相机的摆放位置、朝向和录数据时一致。两路相机不能接反：`observation/image` 是固定相机，`observation/wrist_image` 是腕部相机。
2. **状态的单位和顺序**和 HDF5 里的 `observation/state` 相同：前 6 维是关节角（rad），第 7 维是夹爪开口（m）。输出的关节是绝对角度，夹爪是绝对开口。
3. **10 Hz，不是 30 Hz。** 录数据是 30 Hz，但模型是在 10 Hz 的数据上训练的：`actions[k]` 是 0.1×(k+1) 秒之后的目标状态。机器人控制回路如果是 30 Hz，就在相邻两个目标之间线性插值，或者每个目标保持 3 个控制周期。不要把 10 个目标按 30 Hz 连续发出去，那样动作会快 3 倍。
4. **执行节奏**：最简单的做法是执行完 10 步（1 秒）再推理下一次，推理期间机械臂停在最后一个目标点等待。A100 上单次推理约 0.08 秒，停顿很短。想让动作更灵敏，也可以每执行 5 步就重新推理一次（openpi 的 LIBERO 示例就是这样做的）。
5. **指令必须一字不差。** 模型只见过上面三条指令。换说法、加标点，效果都没有保证。

## 6. 推理参数

- 用 openpi 的默认设置：10 步去噪。
- 每次推理的初始噪声由策略内部的随机数生成器给出，同样的输入多次推理，结果会有细微差别，这是 flow matching 的正常现象。
- 想要完全可复现的输出，可以像自检那样给 `predict` 传固定的 `noise`（形状 (10, 32) 的标准正态数组）。

## 7. 安全建议

- 头几次让人守在急停旁，并把机器人侧的速度上限调低。
- 发送前做一次限幅。三个任务全部 180 条示教中关节角的范围如下，超出太多说明输入出了问题：

  | | joint_1 | joint_2 | joint_3 | joint_4 | joint_5 | joint_6 |
  |---|---|---|---|---|---|---|
  | 最小 (rad) | -1.028 | -0.004 | -1.944 | -0.828 | -0.353 | -1.706 |
  | 最大 (rad) | 0.915 | 2.351 | -0.002 | 0.952 | 1.225 | 0.537 |

  夹爪开口限幅到 [0, 0.08] m。自检时模型输出的夹爪开口在 0.001–0.0805 m 之间，偶尔会略超 0.08。
- 检查每一步要走多远。按 10 Hz 统计全部示教（每条三个相位）共 51,552 步，每个关节每步（0.1 秒）的变化：

  | | joint_1 | joint_2 | joint_3 | joint_4 | joint_5 | joint_6 |
  |---|---|---|---|---|---|---|
  | 中位数 (rad) | 0.021 | 0.042 | 0.017 | 0.016 | 0.014 | 0.015 |
  | 99% 分位 (rad) | 0.094 | 0.097 | 0.078 | 0.092 | 0.085 | 0.093 |
  | 最大 (rad) | 0.100 | 0.105 | 0.097 | 0.098 | 0.101 | 0.100 |

  示教里关节一步最多走 0.105 rad（约 6°，joint_2）。建议机器人侧做两件事：
  - 把单步关节变化限制在 0.15 rad 以内；
  - 如果 `actions[0]` 和当前状态的差，或者相邻两步之差，超过 0.25 rad，就停下来，这多半是相机或状态接错了。

  夹爪是开合式动作，一步可以从全开变到全闭，不需要限速。
- 场景尽量和示教一致：同一张桌子，物体和容器放在示教时的大致区域，光照相近。模型没见过的物品、摆法和指令，效果没有保证。

## 8. 自检结果说明

- **测法**：每个任务第 0 条示教，按 10 Hz 读（从第 0 帧起，和训练时的降采样一样）。每隔 10 步让策略看一次当时的原始相机画面、状态和指令，预测 10 步动作，再和示教里真实执行的 10 步比较，共 30 次。"原地不动"指一直保持当时的状态，作为对照。
- **结果**：三个任务的关节误差都只有"原地不动"的 1/7 到 1/8（比值 0.13–0.14），夹爪误差 1–2.5 mm，所有输出都是有限值。
- **能说明什么**：推理链路是对的，包括图像怎么喂、状态和动作的单位、关节增量怎么加回去、归一化统计量、10 Hz 的时间对齐。
- **不能说明什么**：这三条示教都在训练集里（这次没有留测试集），所以误差小不代表换了物体摆放之后也能成功。成功率只能在实机上测。

## 9. 常见问题

- **尝试联网下载 `paligemma_tokenizer.model`**：环境里设了 `OPENPI_DATA_HOME`，而那个目录里没有分词器。`agilex3_pi05.py` 只在没设这个变量时才指向包里的 `openpi_cache/`。unset 掉这个变量即可，或者把包里的分词器拷到 `$OPENPI_DATA_HOME/big_vision/`。
- **报错找不到 `params` 或 `norm_stats.json`**：`checkpoint/` 不完整，先跑 `sha256sum -c SHA256SUMS`。
- **`fixed image must be (240, 320, 3) uint8`**：相机画面的尺寸或类型和录数据时不一样。按录数据程序的处理方式缩放和转换成一样的数组再传。
- **显存不够（OOM）**：见第 3 节的显存要求。同一张卡上还有别的程序时，可以设 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`，或者设 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 关掉预分配。
- **第一次推理特别慢**：JAX 第一次调用要编译，属于正常现象。可以在开始任务前先用 `samples/` 里的一帧空跑一次。
- **动作明显偏、机械臂往奇怪的方向走**：最常见的原因有四个，依次检查：
  1. 两路相机接反了；
  2. 颜色通道顺序和录数据时不一致；
  3. 状态单位不对（关节用 rad，夹爪用 m）；
  4. 按 30 Hz 发送了 10 Hz 的目标（见第 5 节第 3 点）。

## 10. 其他

- 第 2000、4000 步的 checkpoint 还在集群上。如果第 5999 步在真机上表现得过于死板，可以要来对比。
- 想在这个模型上接着训，或者加新任务重训，`training_code/` 里有训练配置和数据转换脚本。训练要 4 张 40 GB 以上的卡。
