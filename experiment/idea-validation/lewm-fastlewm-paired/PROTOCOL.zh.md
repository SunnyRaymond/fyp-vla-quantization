# LeWM / Fast-LeWM × CEM budget 配对实验

2026-09-26 冻结；主配置为 `FREEZE.json`。本实验接续已完成的 Fast-LeWM checkpoint 复现，不恢复 Goal-query projected+tail 已失败的 recipe。

问题是：两种 pretrained models 在共同运行环境下，对 CEM 10/30 轮的成功率损失和规划耗时是否有不同反应？模型和搜索预算构成 2×2 设计，solver seeds 固定 42/43/44，共 12 arms。每臂使用相同 50 个任务，来自作业 `25569241.pbs101` 的已保存 manifest。它们已参与 baseline 验证，因此本次是探索性配对诊断，不是新任务集上的 confirmatory generalization test。

LeWM 使用 H=5、action_block=5、receding=5；Fast-LeWM 使用 H=1、action_block=25、receding=1。两者预测与执行均为 25 primitive steps，history=1、goal offset=25、eval budget=50，50 个 environments。Fast consistency beta=0。两者共用 stable-worldmodel 0.0.6 的 native CEM，300 candidates、top30、batch_size=1、var_scale=1；保留该版本 topk、unbiased std、candidate-zero 和 warm-start 行为。两者均安装官方 buffered-action fast path，避免仅执行缓存动作时重复处理图像。

复用同一 PushT HDF5 与现有 compatible Torch runtime，不更改共享环境。全数据 action/proprio/state StandardScaler 在 CPU allocation 一次拟合并序列化，逐 arm 不重跑全列扫描或任务抽样。该 scaler 沿用官方 dataset-wide 定义，所以不能声称 preprocessing 对 evaluation episodes 隔离。不同模型的 `jepa/module/utils` 顶层模块隔离在独立子进程；source、checkpoint、backend 和实际运行版本写入结果。

每臂第一次 native solve 的输入作为固定起始 observation。独立 solver 在该 observation 的 fresh copies 上 warm up 一次，再计时三次；不消耗实际闭环 solver 的 RNG，不把 model loading、数据准备和环境创建计入规划时间。记录实际 active B、candidate scores、计时边界和 finite 检查边界；native solve 自带的 CPU cost logging 保留。结束后在真实 policy 的 solver 上执行原生 evaluate_from_dataset，保存完整 50 项 outcomes。固定臂顺序的前两个 seed blocks 反序，第三固定置换；没有完全位置平衡，重复计时共享 contexts，也不是独立任务 replication。

接受条件首先是接口和执行有效：12 arms、每臂 50 outcomes、有限最终 costs/actions、完整时序记录。Fast 30 轮 seed42 与之前 49/50 的逐任务向量对照；不一致时先诊断 harness，不根据成功率调参。LeWM 30 轮各 seed 至少 45/50 才作为 usable baseline 讨论 model×budget 机制；未达则保留负结果并定位 backend 变化。旧 backend 的 LeWM 49/50 与旧 instrumentation 时间不能作为本次公平比较值。

分析以 50 source tasks 为单位，solver seed 是重复测量。报告每 seed 成功数、配对预算差、model×budget 差和任务层 uncertainty；不能把 150 个重复 outcomes 当作 150 IID tasks，也不能从不显著推出等价。延迟只比较相同起始 observations 和相同实际负载的 steady samples，冷启动与实际闭环耗时另列。不同 latent spaces 的 cost 大小不可直接比较；不能将模型差异唯一归因于 action-prefix architecture。native RNG 随预算消耗不同，本协议不声称后续 environments 或 replans 共用相同候选。

CPU 准备上限 1 小时，单 GPU 主实验上限 2 小时。真实 PBS allocation guard 位于重操作之前；login node 不执行下载、模型/数据读取、安装或计算。GPU 利用率与显存每 5 秒写入 job.log。失败记录保留，只允许修复执行错误，不调整科学配置；任何 insight 的新初步实验需单独冻结且在本次完成之后开展。主额度剩余低于 40% 时，完成当前已启动实验后总结并暂缓扩展，不使用 banked reset。

实验设计参考流程：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. [DOI](https://doi.org/10.48550/arXiv.2609.00065)，2026-09-26 核对当前 arXiv v2；此引用说明流程来源，不为方法有效性背书。
