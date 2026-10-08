# 备用第三 baseline：Motus

## 结论

**首选 `motus-robotics/Motus_robotwin2` 作 RoboTwin 2.0 仿真跨骨干后备。**官方仓库公开代码、Stage-3 RoboTwin checkpoint 和 `eval.sh`/`auto_eval.sh` 评测入口；model card 标明 50+任务、14-D 双臂动作。其 MoT 联合 video/action/understanding experts，官方 `inference_step` 同时返回 predicted frames 与 actions，故可在不增加训练的前提下验证同一推理期方法在第二种架构上的适用性。它不能冒充 FastWAM-Joint，或替代 FastWAM 的同模型复现；若论文方法要求至少两种 joint-WAM backbone，Motus 可补上该方法级跨骨干证据。来源：[论文](https://arxiv.org/abs/2512.13030)、[代码及评测入口](https://github.com/thu-ml/Motus)、[匹配 checkpoint](https://huggingface.co/motus-robotics/Motus_robotwin2)。

## 联合推理与可改计算对象

源码把 action 与 future-video latent 从噪声同步初始化；每个 NFE 都经过 trimodal joint attention，再各自预测 velocity，并以 Euler 更新两条 latent 轨迹。采样步数默认 50，checkpoint API 示例用 20；这是运行时可改的 NFE，不需额外训练。可共同研究的静态计算对象是 observation/prompt conditioning：条件帧 VAE latent 在循环外编码，T5 embedding 接口允许预编码；相对地，joint attention 中的 action/video tokens 随步更新，不能当静态缓存。代码还在循环内重复调用 `extract_und_features(vlm_inputs)`，可作为待实测的静态特征复用候选，不能仅凭源码声称已获加速。[推理实现](https://github.com/thu-ml/Motus/blob/main/inference/robotwin/Motus/models/motus.py)

## 真机边界与复现风险

官方 real-world 示例支持 AC-One、Agilex-Aloha-2 配置，并能对三视角图像输出未来帧和动作；但文档明确是“无机器人环境”的单图推理示例。公开 checkpoint 清单中的目标任务权重是 RoboTwin2；未核实到配套公开的 AC-One/Aloha 真机任务 checkpoint、采集演示数据或复现论文真机 post-training 配方。因此可直接复现的匹配平台是 **RoboTwin 2.0 simulation**；真机闭环需目标 embodiment 的模型/数据及控制接口，不能声称开箱可跑。baseline 原有 post-training 是其已发布 checkpoint 的前置成本；本研究新增方法仍可限定为 training-free。另有公开 issue 报告 RoboTwin 随机任务成功率与论文值不符、action chunk 配置有歧义；这是用户报告而非维护者结论，复现实验应冻结版本、NFE、chunk 和 seed，并单独声明结果范围。[真机推理说明](https://github.com/thu-ml/Motus/blob/main/inference/real_world/Motus/README.md)、[issue #46](https://github.com/thu-ml/Motus/issues/46)

## Cosmos3 DROID 接口补充

此处核对的是 Cosmos3 DROID，不是旧的 Cosmos Policy（arXiv:2601.16163）。Sparse-WAM 论文称 Edge Policy 存在 joint-future 能力；公开 DROID policy server 的 deployment 接口则以 action streaming/返回为主。**server 仅返回 action 不能单独证明模型内部没有生成 future**；本次未核实该匹配 checkpoint 的 joint-video/action 推理与评测 recipe，故不将公开 deployment 接口等同于已验证的联合评测基线。来源：[Sparse-WAM 论文](https://arxiv.org/abs/2609.38984)、[DROID policy server](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/action_policy_droid_server.md)、[Action API 专家接口说明](https://github.com/NVIDIA/cosmos/blob/main/cookbooks/cosmos3/nim/action.md)。
