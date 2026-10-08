# Training-free Efficient WAM：基线与真机复现可行性

**资料截止：2026-10-02。** 只查了论文、官方代码与官方 checkpoint 页面；没有下载权重、运行模型或提交作业。文中将论文/源码明确写出的内容标为“已确认”，跨来源推断标为“推断”。

## 结论

- **FastWAM-Joint：源码和 LIBERO 评测入口公开，但官方权重页没有列出同名 Joint checkpoint。** 它提供较清晰的 training-free 插入点，论文设置是 10 步联合 denoise；直接复现其 Sparse-WAM 数字仍需取得/训练匹配的 Joint 权重。
- **Cosmos3-Edge-Policy-DROID：官方 4B checkpoint、Cosmos Framework 和 RoboLab server/client 路径公开。** 可按官方文档部署仿真基线；Sparse-WAM 的 Cosmos 结果是在 RoboLab-120 上，不是真机。论文所报 4 步 denoise 很短，额外加速空间和 FastWAM-Joint 不宜按同一模型规模直接比较。
- **Sparse-WAM 的真机行是 FastWAM-Joint + AgileX Cobot Magic。** 作者先用自采数据对 FastWAM-Joint 做 post-training，再比较 dense 与 sparse inference。post-training 是部署该任务策略的前置成本；Sparse-WAM 本身不更新权重，二者不能合并成“加速方法需要训练”或“真机复现零成本”。
- 两个模型共同可做的 training-free 计算改动对象是**未来视频 latent tokens 的 Transformer 计算**。Sparse-WAM 用 action-to-future attention 选择未来 token，仅保留 observation/action tokens 全量计算；它的关键不是增加一个通用 temporal feature/KV cache。方法还会复用 token 选择及未保留区域的速度预测，这是为 token pruning 服务的跨步状态。

## 基线对照

| 项目 | FastWAM-Joint | Cosmos3-Edge-Policy-DROID |
|---|---|---|
| 架构 | **已确认（论文/源码）**：Wan2.2 VideoDiT + ActionDiT 双 expert，由 MoT 混合注意力一起预测未来视频与动作。Joint mask 允许 action queries 读完整 video token 序列；video queries 仍按视频 mask 计算。配置各有 30 层，video/action hidden dim 分别为 3072/1024。 | **已确认（官方模型卡/源码）**：4B Cosmos3-Edge MoT；文本走 autoregressive tower，图像/视频/动作等连续生成走 diffusion tower。Sparse-WAM 将其作为共享 Transformer 序列的联合未来视觉—动作 denoise 基线。 |
| 每 chunk 输入/输出 | 当前图像、instruction 的 prompt 或预编码 context、可选 proprio；Joint sampler 同步更新未来 video latents 与 32-step action chunk。Sparse-WAM 配置为 2 个未来 latent frames、98 tokens/frame、2 个 camera views。 | DROID policy checkpoint 接收 instruction 和多视角视觉 conditioning，action chunk size 为 32。Sparse-WAM 设置为 3 视角、8 个未来 latent frames、340 tokens/frame。 |
| steps / sampler | Fast-WAM 论文与 Sparse-WAM 设置均为 **10 步、CFG=1.0**。源码是连续 Flow-Matching scheduler，按 shift 构造时间步，再以 sample + velocity × delta 更新。当前 Joint YAML 默认 video/action shift 为 5/1；Sparse-WAM 论文统一设 shift=5，因此复现其表格时应显式传相同 shift override，不能只依赖 YAML 默认值。 | Sparse-WAM 报告 **4 步、guidance=3.0、shift=5.0**。Cosmos Framework 的 OmniMoT 源码使用 Rectified Flow，并支持 UniPC/EDM sampler；Sparse-WAM 论文没有写明 Edge policy 该实验选择了哪种 sampler，HF checkpoint metadata 也没有给出 config/experiment，故具体 sampler 尚未从公开材料钉定。 |
| observation / conditioning cache | infer_joint 接口接收 raw input_image，每次请求 VAE encode 一次；可直接传入预编码 context/context_mask，且首帧 latent 在每步更新后固定回原 observation。没有确认到可跨 denoise 调用复用的 FastWAM-Joint observation/KV cache API。源码的 legacy video-KV-cache 路径标明供 optional IDM 用，不等同 Joint baseline。 | Sparse-WAM 论文写明其评测中 instruction representations 跨 action chunks 缓存，而视觉 conditioning 随每个新 observation 更新。公开 server/client 提供策略服务接口；没有确认到一个可由调用方传入、跨 denoise 复用 observation KV 的稳定 API。 |
| 可确认的改动路径 | FastWAMJoint mask：**src/fastwam/models/wan22/fastwam_joint.py**；联合 denoise loop 与 image/context API：**src/fastwam/models/wan22/fastwam.py**；MoT attention：**src/fastwam/models/wan22/mot.py**；shift schedule/velocity step：**src/fastwam/models/wan22/schedulers/scheduler_continuous.py**；架构/shift 配置：**configs/model/fastwam_joint.yaml**。 | 已确认的基线模型/采样入口：**cosmos_framework/model/generator/omni_mot_model.py**；RoboLab policy server：**cosmos_framework/scripts/action_policy_server_robolab.py**。**推断**：稀疏选择要接入 action-to-future attention 与生成 token packing；Sparse-WAM 源码未公开，故具体函数/层号无法确认。 |
| 公开可运行物 | 官方 FastWAM repo 有训练/评测代码和 LIBERO 数据入口；官方 HF 权重页列的是 FastWAM action-only、Optional IDM 和 RoboTwin action-only checkpoint，没有列出 FastWAM-Joint checkpoint。Optional IDM 的 future-imagination 路径不是 Joint 架构的替代 checkpoint。 | 官方 HF 有 nvidia/Cosmos3-Edge-Policy-DROID checkpoint；Cosmos Framework 文档给出 server 启动路径，NVlabs/RoboLab 给出 client 入口。可按公开文档部署仿真模型；本次没有实际运行验证。 |

## Sparse-WAM 的证据与训练边界

**论文设置（已确认）**：核心算法先用 dense conditional forward 统计 action-to-future attention，再为每个未来帧保留 action-relevant core tokens 和 shared spatial anchors；后续步骤对压紧后的未来 token 序列计算，所有 observation/action tokens 仍保留。未进入计算的 future positions 继续通过最近一次 dense velocity 预测参与 sampler 更新；动作预测每一步重算。

| 评测 | Dense → Sparse-WAM | 读数边界 |
|---|---:|---|
| FastWAM-Joint / LIBERO | 98.75% → 98.45%；论文报 1.98×，保留 31/98 future tokens 每帧 | 10 steps、2 future latent frames；这是仿真结果。 |
| Cosmos3-Edge / RoboLab-120 | 22.90% → 23.00%；相对 dense eager 报 1.85×，相对同样开启 compile/CUDA Graph 的 dense 为 1.56×；保留 184/340 future tokens 每帧 | 加速数字受 backend 优化影响；相同后端的 1.56×更接近 token sparsity 本身带来的比较。 |
| 真机 FastWAM-Joint / AgileX Cobot Magic | 平均成功率 77.78% → 75.00%；501 ms → 242 ms（2.08×） | 三个任务为 object packing、stacking cups、battery assembly；平台有一个主相机和两个腕部相机。 |

**Training-free 的边界**：Sparse-WAM inference 不训练、不改 checkpoint；但真机表格使用先行 fine-tune 的 FastWAM-Joint。附录给出 8×H800、每卡 batch 8、global batch 64、LR 1e-4、BF16、15k/20k/15k steps。论文只说数据由作者采集，未给演示数量或公开数据链接；也未找到 Sparse-WAM 官方代码/权重链接。因此这组真机实验无法仅凭公开 artifacts 原样复现。

作为区分，Fast-WAM 原论文的真机实验是 Galaxea R1 Lite towel-folding，采集 60 小时 teleoperation demos；这不是 Sparse-WAM 的 AgileX 实验，也不能替代其真机 checkpoint/data。

## 可用评测平台与复现判断

| 平台 | 可做什么 | 当前缺口 |
|---|---|---|
| LIBERO + FastWAM | 官方代码有 LIBERO evaluation manager，FastWAM 训练用数据集有公开入口；适合验证 FastWAM 系列和 inference 实验。 | 没核实到官方 FastWAM-Joint evaluation checkpoint。Sparse-WAM 自身代码未公开；须另行实现其 action-guided sparse 路径。 |
| RoboLab / RoboLab-120 + Cosmos3-Edge | 官方 Cosmos policy server 和 RoboLab client 可运行；Sparse-WAM 论文给出 120 tasks（64 simple / 39 moderate / 17 complex，每项 10 rollouts）的 protocol。Edge 权重可公开获取。 | 公开 server 示例是 RoboLab task client；要逐项对齐论文 RoboLab-120 任务/seed 和 Sparse-WAM 稀疏 kernel，还需拿到/重写对应评测脚本与方法实现。 |
| AgileX Cobot Magic 真机 | 论文的 Sparse-WAM 真机结果平台；有明确任务、相机和 latency/success 数字。 | 需访问同型硬件并重采任务 demonstrations；自采数据量、真机训练数据、fine-tuned Joint checkpoint、Sparse-WAM inference code 均未在论文给出公开下载入口。 |
| DROID 真机平台 | Edge checkpoint 面向 DROID data/embodiment；官方 Cosmos Framework server 说明 client 可接模拟或真实 robot。 | Sparse-WAM 没在 DROID 真机报告结果。不能把 RoboLab-120 或 Cobot Magic 数字表述成 Edge/DROID 真机证据。 |

**优先顺序（推断）**：若先验证“同一方法能否跨 WAM backbone”，Cosmos3-Edge 更容易建立公开权重的 dense 仿真基线；若要复现 Sparse-WAM 论文的 FastWAM-Joint 表，先解决 Joint checkpoint。若目标是真机复现 Sparse-WAM 表，当前最大阻碍是其专用源码、真机演示数据与 matching policy checkpoint 均未确认公开，而不是 acceleration method 需要额外训练。

## 官方主要来源

1. [Fast-WAM 论文](https://arxiv.org/abs/2603.16666)；[官方源码](https://github.com/yuantianyuan01/FastWAM)；[官方权重页](https://huggingface.co/yuanty/fastwam)；[LIBERO-FastWAM 数据页](https://huggingface.co/datasets/yuanty/LIBERO-fastwam)。
2. [Sparse-WAM 论文 HTML](https://arxiv.org/html/2609.38984)；arXiv 当前条目未列代码仓或 checkpoint 链接。
3. [Cosmos3-Edge-Policy-DROID 官方权重页](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID)；[Cosmos Framework DROID policy server 文档](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/action_policy_droid_server.md)；[官方 RoboLab](https://github.com/NVlabs/RoboLab)。
4. [Cosmos Framework OmniMoT 源码](https://github.com/NVIDIA/cosmos-framework/blob/main/cosmos_framework/model/generator/omni_mot_model.py)；[RoboLab policy server 源码](https://github.com/NVIDIA/cosmos-framework/blob/main/cosmos_framework/scripts/action_policy_server_robolab.py)。
