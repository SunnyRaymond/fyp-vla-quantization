# Fast-WAM Smooth + Hadamard → Weight VQ 两阶段验证

状态：2026-10-08，preflight 作业 25721014.pbs101 已成功完成（Exit_status=0，21 次查询，614/614 Linear 覆盖）。正式 full 作业 25722477.pbs101 已确认正常启动：数值自检通过、MODEL_READY、已完成至少 25 次模型查询并进入 phase1_identity。按用户要求暂停目标及监控，等待通知；正式实验的最终结果尚未验证。启动证据保存在 [work_status.json](work_status.json) 和 [full progress.json](results/full/progress.json)。

Preflight 的单个评估输入、两个 sampler seeds 上，BF16 weights/A4 的 action RMSE 从 Identity 的 0.270035 降至 Smooth α=1 + Hadamard 的 0.045414；同一变换下 Scalar W4A4 为 0.046645，VQ+A4 为 0.057953。变换的 BF16 对照漂移为 0.001424。Scalar 与 VQ 的有效存储分别为 4.1370 与 4.0154 bits/weight（计入变换 tensor 元数据）。这些结果只证明程序可运行，不能据此宣布 VQ 优于 Scalar 或泛化收益；full 仍按原协议独立选择参数。原始汇总见 [preflight summary.json](results/preflight/summary.json)。

复用上一轮已经保存的 22 个固定观察，不执行预测动作。模型是 released Optional-IDM clean checkpoint，IDM 中 Video 和 Action 各 10 个 denoising steps，KV 保持 BF16。目标 Linear 的范围与原量化诊断相同。

第一阶段只改变输入坐标与 activation 精度：Identity、Hadamard-only、三个 Smooth 强度加 Hadamard，以及 Smooth-only。变换后的 weights 为 BF16，activation 为 A4。同时运行未量化的变换对照，量化收益需与浮点变换自身的数值漂移一起阅读。

第二阶段在 Identity 和三个 Smooth + Hadamard 配置下分别重新拟合 Scalar W4 和 additive Weight VQ。VQ 用四维向量、两本各 256 entries 的 BF16 codebook、两个 uint8 indices；index 成本为 4 bits/weight，另计 codebooks、变换元数据、tails 和文件容器成本。VQ 使用实际 Q4 输入的 diagonal second moment 拟合局部 Linear-output 误差；最终 action sensitivity 与敏感方向保护留待第三阶段。

8 个输入的第一 sampler seed 只用于 calibration；4 个输入、各两个 seeds 用于选择 Smooth 强度与码本配置；最后 10 个输入、各两个 seeds 作为 test。三个集合互不重叠。第一阶段、第二阶段的 Scalar 和 VQ 分别选配置，选择完成后写入 frozen winners，再运行 test。Test 中两种 weight quantizer 都评估 Identity 与两个已选变换的并集，确保可在相同变换下配对比较。重复 seeds 是同一输入内的配对重复，最终均值先对每个输入的两 seeds 取平均。

主要指标是 normalized action 前 10 步、前 6 个 motor 坐标相对自身 BF16 的 RMSE。Gripper 单独报告。Activation 重建、weights 重建和最终 action 误差均报告；不能根据 activation 更平滑单独判断成功。

先运行缩小的 preflight：一个校准输入、另一个评估输入的两 seeds，两种变换，两个 weight quantizers。它只验证实现和计算资源，不参与完整实验的参数选择。完整实验开始前保留同一份 [protocol.json](protocol.json)。

所有模型计算、码本拟合、数值自检和汇总均在获批 PBS compute allocation 内执行。GPU 作业记录利用率和显存；本地只做脚本编辑、语法检查和小型控制/结果文本传输。两阶段采用 decoded BF16 reference 运算，量化表示可实际保存，尚不能据此宣称 native VQ kernel、部署加速或闭环成功率。

Full runner 每 10 分钟输出一次 faulthandler stack snapshot，用于定位执行位置。其 `Timeout` 前缀本身不代表作业失败；完成与否仍以 scheduler、exit code 和结果文件判定。

设计流程使用 `experimental-design` skill 的分组、重复与独立评估规则；软件参考：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065)，当前 arXiv v2（2026-09-02）。

连接恢复后的入口是 `control.py`：先 `upload`，再 `submit --phase preflight`。Preflight 必须同时满足 scheduler terminal state、Exit_status=0、PIPELINE_COMPLETE、exit_code.txt=0 和非空 summary.json；检查该数值结果后才提交 `submit --phase full`。重复使用同一提交 attempt 会恢复既有 job handle，未知提交结果不会自动重投。完整执行保留原 BF16 weights 的 CPU 副本用于各变换的公平拟合，GPU 推理使用重建后的 BF16 weights；该工作内存不等于低位部署显存。
