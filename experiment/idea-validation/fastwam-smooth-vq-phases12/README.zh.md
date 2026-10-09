# Fast-WAM Smooth + Hadamard → Weight VQ 两阶段验证

Weight / activation 可视化已完成：GPU 采集 `25727294.pbs101`、最终 CPU 绘图与汇总 `25728448.pbs101` 均为 F、Exit_status=0。3 条 BF16 teacher inputs 各覆盖 614/614 Linear，10 张最终图已逐张检查。实测热点与清晰读图说明见 [VISUALIZATION.zh.md](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-smooth-vq-phases12/VISUALIZATION.zh.md>)，固定方法见 [VISUALIZATION_PLAN.zh.md](VISUALIZATION_PLAN.zh.md)。复用 existing codebooks，没有做恢复实验或重新拟合；这是已知输入上的机制诊断，不改动下文正式 held-out 结论。

状态：2026-10-08，Fast-WAM 两阶段验证已完成。正式 full 作业 25722477.pbs101 和 CPU 汇总作业 25725256.pbs101 均为 F、Exit_status=0；444 次查询完整，614/614 Linear 覆盖，原始 query 重现结果均值。完整解释、逐 case 对比、实际存储成本和另外两个 WAM 的架构迁移讨论见 [RESULTS.zh.md](RESULTS.zh.md)。

Held-out 10 个输入、每个两个 seeds 上，BF16 weights/A4 的 motor RMSE 从 Identity 的 0.337242 降至 Smooth α=0.5 + Hadamard 的 0.045289。同变换下 Scalar W4A4 为 0.057447，VQ+A4 为 0.078310，VQ 在 10/10 cases 更差。选择集冻结的 VQ winner 是 α=1 + Hadamard，其 test RMSE 为 0.104092；不能根据 test 把 α=0.5 改称正式 VQ winner。本轮支持 activation 变换的收益，但当前简单 codebook 尚未胜过 Scalar。结果源为 [full summary.json](results/full/summary.json)、[analysis summary.json](results/analysis/summary.json) 和 [frozen winners](results/full/frozen_winners.json)。

Preflight 作业 25721014.pbs101 也已成功结束。它的单个评估输入、两个 sampler seeds 上，BF16 weights/A4 的 action RMSE 从 Identity 的 0.270035 降至 Smooth α=1 + Hadamard 的 0.045414；同一变换下 Scalar W4A4 为 0.046645，VQ+A4 为 0.057953。变换的 BF16 对照漂移为 0.001424。这些只用于验证实现，不参与 full 的配置选择。原始汇总见 [preflight summary.json](results/preflight/summary.json)。

复用上一轮已经保存的 22 个固定观察，不执行预测动作。模型是 released Optional-IDM clean checkpoint，IDM 中 Video 和 Action 各 10 个 denoising steps，KV 保持 BF16。目标 Linear 的范围与原量化诊断相同。

第一阶段只改变输入坐标与 activation 精度：Identity、Hadamard-only、三个 Smooth 强度加 Hadamard，以及 Smooth-only。变换后的 weights 为 BF16，activation 为 A4。同时运行未量化的变换对照，量化收益需与浮点变换自身的数值漂移一起阅读。

第二阶段在 Identity 和三个 Smooth + Hadamard 配置下分别重新拟合 Scalar W4 和 additive Weight VQ。VQ 用四维向量、两本各 256 entries 的 BF16 codebook、两个 uint8 indices；index 成本为 4 bits/weight，另计 codebooks、变换元数据、tails 和文件容器成本。VQ 使用实际 Q4 输入的 diagonal second moment 拟合局部 Linear-output 误差；最终 action sensitivity 与敏感方向保护留待第三阶段。

8 个输入的第一 sampler seed 只用于 calibration；4 个输入、各两个 seeds 用于选择 Smooth 强度与码本配置；最后 10 个输入、各两个 seeds 作为 test。三个集合互不重叠。第一阶段、第二阶段的 Scalar 和 VQ 分别选配置，选择完成后写入 frozen winners，再运行 test。Test 中两种 weight quantizer 都评估 Identity 与两个已选变换的并集，确保可在相同变换下配对比较。重复 seeds 是同一输入内的配对重复，最终均值先对每个输入的两 seeds 取平均。

主要指标是 normalized action 前 10 步、前 6 个 motor 坐标相对自身 BF16 的 RMSE。Gripper 单独报告。Activation 重建、weights 重建和最终 action 误差均报告；不能根据 activation 更平滑单独判断成功。

先运行缩小的 preflight：一个校准输入、另一个评估输入的两 seeds，两种变换，两个 weight quantizers。它只验证实现和计算资源，不参与完整实验的参数选择。完整实验开始前保留同一份 [protocol.json](protocol.json)。

所有模型计算、码本拟合、数值自检和汇总均在获批 PBS compute allocation 内执行。GPU 作业记录利用率和显存；本地只做脚本编辑、语法检查和小型控制/结果文本传输。两阶段采用 decoded BF16 reference 运算，量化表示可实际保存，尚不能据此宣称 native VQ kernel、部署加速或闭环成功率。

Full runner 每 10 分钟输出一次 faulthandler stack snapshot，用于定位执行位置。其 `Timeout` 前缀本身不代表作业失败；完成与否仍以 scheduler、exit code 和结果文件判定。

设计流程使用 `experimental-design` skill 的分组、重复与独立评估规则；软件参考：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065)，当前 arXiv v2（2026-09-02）。

控制入口是 `control.py`，已有 preflight/full/analysis 三个成功作业及持久化 handles，不需重新提交。完成判据同时检查 scheduler terminal state、Exit_status=0、PIPELINE_COMPLETE、exit_code.txt=0 和非空 summary.json。重复使用同一提交 attempt 会恢复既有 job handle，未知提交结果不会自动重投。`analysis.py` 的结果汇总通过 `analysis.pbs` 在 CPU allocation 内执行。完整执行保留原 BF16 weights 的 CPU 副本用于各变换的公平拟合，GPU 推理使用重建后的 BF16 weights；该工作内存不等于低位部署显存。
