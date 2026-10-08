# DreamZero 与 LingBot-VA：公开权重和真机复现可行性

**范围：**仅核对官方论文、源码和模型卡；未下载权重或数据、未运行模型或机器人。状态是公开 artifacts 的核查结论，不表示已运行或可保证复现。

| 模型 | 联合 policy 权重与许可 | 真机证据 / 可复现入口 | 采样与 attention 插入点 | 推理硬件与原训练成本 | 判断 |
|---|---|---|---|---|---|
| **DreamZero** | `GEAR-Dreams/DreamZero-DROID` 为公开 14B BF16 joint video/action checkpoint，模型卡标 **CC-BY-NC-4.0**；源码 Apache-2.0。另有用于新 embodiment post-training 的 `DreamZero-AgiBot`（模型卡 Apache-2.0）。无需申请 gated access 的提示。 | 论文报告 DROID/Franka 单臂与 AgiBot G1 真机；另有 YAM few-shot 适配。公开源码提供分布式 WebSocket inference server/test client、仿真入口及 RoboArena integration，但本次未找到可独立复现论文真机结果的 robot-side client/controller、标定文件或 matching Franka/AgiBot 配置。论文只确认结果，未交付这些真机接线 artifacts。仿真托管 API 需申请访问。 | 论文与源码均为 16-step 联合 video/action 去噪，基础配置共用 timestep；每次 joint forward 产出两种 velocity，视频和动作分别更新。QKV attention 与 modality mask 是明确插入点。论文称多视角图像拼为单帧，源码未见显式 camera/source group ID；可用 token 区间加分组 mask，但须自行改造并测 fidelity。 | 官方 README：分布式推理最低 2 GPU，测试 GB200/H100；H100 开启 cache 后约 3 秒，未公布显存最低值。复训 DROID 为 100K steps、global batch 128；新 embodiment 需自有约 30 分钟 play data 并 post-train。直接评测已有 checkpoint 不需重训。 | **条件可用，优先候选。**架构最贴近 joint attention 稀疏化；满足目标的真机复现仍缺机器人控制/标定入口及硬件。DROID 权重有 NC 限制。 |
| **LingBot-VA** | 官方公开 `robbyant/lingbot-va-base`、`lingbot-va-posttrain-robotwin`、`robbyant/lingbot-va-posttrain-libero-long`；模型卡 Apache-2.0。RoboTwin 与 LIBERO 有匹配的 task policy 和官方评测代码。 | 论文报告六项物理机器人任务，称每项使用 50 个 real demos，并做 500-step、LR 1e-4 微调；没有确认到对应真实任务 checkpoint、机器人 client/controller、标定或可识别的硬件配置。官方 repo 的 server-client 入口覆盖 RoboTwin/LIBERO 仿真，不构成真机控制入口。 | 官方推理先以 **3 步**生成 video（积分到 s=0.6），再以 **10 步**由预测 visual transition 推断 action（积分到 s=1.0）；不是同一 sampler 内共同更新当前 future action 与 video。训练 MoT 的 token/mask 路径在 `wan_va/modules/model.py`（`FlexAttnFunc`），能加入 token grouping；论文将多视图沿宽度拼接，源码 mask 未见显式视图 ID。行动影响视频主要经历史 action/FDM 路径，目标若特指 action-query→future-observation 稀疏注意力，机制匹配较弱。 | 官方 README 报 RoboTwin 单 GPU、VAE/text encoder offload 时约 **24 GB VRAM**；image-to-video-action 约 18 GB。未给出具体 GPU 型号。预训练 1.4T tokens，训练硬件/总时长未披露；RoboTwin checkpoint 已公开（论文设置 50K steps、27,500 demos），直接评测无需重训。论文真机 task adaptation 需 50 demos，公开 matching policy 未确认。 | **仿真可用；真机条件不可确认。**适合做第二 backbone 的 simulation-only 适配；若研究假设锁定 action→future-view attention，不宜直接当等价架构证据。 |

## 结论

DreamZero 与 LingBot-VA 都有可用的公开 joint-WAM 权重；LingBot-VA 的 RoboTwin 仿真基线 artifacts 较完整。**两者均未达到“凭公开 artifacts 即可独立复现论文真机闭环”的门槛**：缺少 matching physical setup 的完整 controller/client 与 calibration/config；LingBot-VA 还缺论文真机 task checkpoint。DreamZero 值得在未来作为有条件的独立 adaptation 候选；LingBot-VA 可先承担仿真跨骨干验证。当前不要把任一模型记作已确认可真机复现。

## Primary sources

- DreamZero：[论文](https://arxiv.org/abs/2602.15922)、[官方源码与部署说明](https://github.com/dreamzero0/dreamzero)、[DROID checkpoint](https://huggingface.co/GEAR-Dreams/DreamZero-DROID)、[AgiBot checkpoint](https://huggingface.co/GEAR-Dreams/DreamZero-AgiBot)、[joint sampler 源码](https://github.com/dreamzero0/dreamzero/blob/main/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py)。
- LingBot-VA：[论文](https://arxiv.org/abs/2601.21998)、[官方源码与仿真评测入口](https://github.com/robbyant/lingbot-va)、[base checkpoint](https://huggingface.co/robbyant/lingbot-va-base)、[RoboTwin checkpoint](https://huggingface.co/robbyant/lingbot-va-posttrain-robotwin)、[attention/mask 源码](https://github.com/robbyant/lingbot-va/blob/main/wan_va/modules/model.py)。
