# 后续 VLA / WAM 候选的已查先例

2026-09-13，轻量 primary-source 检索；不是系统综述或 novelty 认证。以下用于避免反复提出已经充分覆盖的泛化点子。未据此运行模型实验。

| 已覆盖切面 | 核实来源 | 对本轮的约束 |
|---|---|---|
| Action-guided tensor bit allocation 与 action-aware scale | [ActQuant v3](https://arxiv.org/abs/2605.24011v3) | 单纯把action importance加入PTQ，不足以形成新切面；论文性能为作者报告，本轮未复现 |
| Attention temperature / output interface scale balancing | [QuantVLA v4](https://arxiv.org/abs/2602.20309v4) | 不将一般attention或output scale修正包装为新方法 |
| 跨denoising步骤的误差传播与蒸馏纠正 | [CTEC, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/34039) | 与本项目旧FRT/PRR一起排除普通temporal error correction重命名 |
| 调整noise scheduler吸收quantization error，含flow-matching路径 | [DNS-AQE官方代码](https://github.com/ZephyrYoung-eYuan/DNS_AQE) | 改solver/scheduler也有强先例，不能仅靠换VLA任务主张新颖 |
| Timestep-dependent activation quantizer分组 | [CVPR 2024论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Towards_Accurate_Post-training_Quantization_for_Diffusion_Models_CVPR_2024_paper.pdf) | 单纯early/late denoising scale分组不是空白 |

轻量资源备选：[SmolVLA官方base checkpoint](https://huggingface.co/lerobot/smolvla_base)，约0.5B参数、continuous flow-matching action expert。官方定位是供task-specific fine-tuning的base；这不能证明LIBERO成功率，也不能证明当前CCDS已安装或V100运行兼容。本轮优先核对既有OpenVLA-OFT/Fast-WAM资产；不为宽泛点子先下载新模型。
