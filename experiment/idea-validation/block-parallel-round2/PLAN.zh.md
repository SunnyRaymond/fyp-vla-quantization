# 第二轮：toy oracle 与 C-SWM 视觉 pilot 并行

用户授权：同时执行已知正确坐标的 toy oracle 与 C-SWM 视觉 pilot，沿用此前冻结、配对、subagent、PBS 和监控要求。两条实验线独立推进，视觉实验不等待 toy 结果或质量门槛。

## Toy oracle

沿用第一轮三种真实动力系统和数据、三个 training seeds、1500 updates、全部 horizon 与 action-response。仅把可学习坐标换成生成器的已知正确正交坐标并固定。相同 seed 的局部 MLP 和 message projection 初始化保持相同。执行三个 oracle arms × 三种系统 × 三个 seeds，另加 lowrank oracle_global16 三个 seeds，共30次训练。

第一轮原始 learned/dense checkpoints 在新的 compute allocation 内重新加载、复现逐 episode 误差并重新计时；用配对误差判断原先困难是否主要来自表示学习。正确坐标属于额外已知信息，不能将 oracle 结果称为可部署方案；冻结 Q 也会减少可训练参数。

## C-SWM visual pilot

2D Shapes 图像输入，遵循官方 block-pushing 动作和碰撞语义。官方来源固定为 commit `e944b24bcaa42d9ee847f30163437a50f0237aa0`。standalone 环境和现代依赖兼容差异需要明确记录，不声称原论文逐配置复现。

每个 seed 首先训练 encoder 与 C-SWM-style GNN，再固定该 encoder，让全部后端读取相同 slots。后端均从头配对训练，比较完整 GNN、local、local4、global4、global16、六层 Transformer 和参数预算接近 global4 的 flat MLP。六层 Transformer 是本接口的新架构对照，不是已训练的 LeWM checkpoint。位置真值仅用于环境验证和辅助评估，不参与视觉 encoder 或 dynamics 训练。

比较重点是 global4 对等参 local4 的通信收益，多步 latent ranking、自由移动/物体阻挡/边界阻挡、action-response，以及相对 GNN、Transformer 和小 flat MLP 的实测延迟。视觉收益不等同于 LeWM、CEM 或闭环收益。每个分支的具体冻结配置和判断依据保存在对应目录的 FREEZE.json 与 PROTOCOL.zh.md。

## 执行与交付

执行 subagents：gpt-6-luna / xhigh。CPU PBS allocation 负责数据生成和必要下载；GPU allocations 负责机制测试、训练与benchmark，并每30秒在 job log 记录 GPU 利用率和显存。login node 只连接、提交、查状态和传输有界小控制文件，模型和大数据留在 compute。不得动用 banked reset，不重复提交排队或运行作业。

交付包含两条实验的实际结果、PBS退出证据、配对比较、负结果与结论范围；报告基于结果，而不是仅凭 DONE 或测试通过宣称方法有效。持续活跃的 Luna 监控属于本会话监控，不声称建立了可唤醒已暂停 goal 的定时自动化。本 goal 不自行暂停。

方法引用：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. https://doi.org/10.48550/arXiv.2609.00065 （本轮已核对当前v2）。
