# Fast-WAM 固定 codebook 的 action proxy 诊断

目标是先比较真实 action 在原四个方向上的投影与训练 proxy，再分别改变 probes 和 token rows，依据结果决定下一版 codebook 目标。没有进行 codebook 拟合。

来源是完整作业 `25743693.pbs101`。固定八个 calibration cases `[0,2,4,6,10,12,14,16]`，每个使用原 sampler seed index 0，保留四套已冻结 banks：row-scale、局部输出拟合、action-sensitive λ=0.1 和 λ=1。全部 614 个目标 Linear、Smooth α=0.5、signed Hadamard128、A4 和存储格式均保持原配置。

CPU 阶段从已保存 action 和 BF16 baseline 计算真实 60 维 motor action 误差、原四个方向的实际投影、它们张成空间内的误差能量，并计算四组新的随机投影。旧日志只保存 proxy RMS，故 CPU 阶段比较相同方向的 RMS；不会从 scalar RMS 推测 signed proxy。

GPU 阶段在相同 teacher trajectory 上重新采集原四个 probes 加四组各四个新 probes，共 20 个方向。每次 Linear 调用独立抽取两套各两个 token rows，采集其并集，再分别按各自 N/n 计算。跨调用和跨层贡献仍先求和再平方。小 token 数的调用可能得到相同 row subsets，实际数量单独报告。新旧 probes、rows 构成四个组合；新 probes 的四组属于同一 case 内的 Monte Carlo 重复，不计为新场景。

按用户 2026-10-10 的排队安排，同时提交 case0 的完整 614-Linear preflight 和八个 cases 的 full；full 不再等待另一作业完成，先在自身 case0 上执行相同的 BF16 parity、梯度覆盖、原 rows 精确复现和原 proxy RMS 复现检查，失败即停止。数值测试、分析、模型加载和重 I/O 均在真实 PBS allocation 内；GPU 利用率和显存每 15 秒记录到 job log。

2026-10-11 的首个 preflight 已确认重新采集的 adjoints 与冻结原 capture 存在差异，尽管全部原采样 inputs、teacher outputs 和 BF16 action 均一致。因此修订后的严格原 proxy 复现使用冻结原 capture；新 capture 独立用于全部四个 factorial 组合。逐 case 和 summary 同时记录冻结原 capture 的 signed proxy 对真实 action 投影，以及相同原 probes、原 rows 下新旧 capture 的 signed prediction difference，避免把 capture 漂移归因于换 probes 或换 rows。容差没有放宽；新的 capture 不声明逐元素梯度复现通过。

新旧 capture 的 proxy drift 还包含原 LayerData 与新 evaluator 数值求和顺序的差异，不将该指标解释成纯梯度噪声。当前重提 handles 为 preflight `25754358.pbs101`（已核实成功）、full `25754359.pbs101`（仍排队），输出按 job ID 隔离，旧失败 attempt 的结果保留。成功 preflight 只覆盖 case0，不能代替八个 cases 的 full 结果。

关联监控 `fast-wam-proxy` 使用 `gpt-6-sol / xhigh`，每 30 分钟核对两个 phase 的最新 job handle。用户授权在失败时修复共享根因、同步两阶段脚本，并保留失败 artifacts 后用新 attempt 重提；运行中的作业保留其已冻结快照，后续按实际问题处理。没有实质变化时保持安静，成功或需要原任务继续分析时唤醒原任务。

解释顺序：

1. 原四个真实投影若仍大，优先判断采样/teacher 线性近似失配；不能只归因于未覆盖方向。
2. 原四个真实投影若小、其他方向和完整 action 误差仍大，支持方向覆盖不足的解释。
3. 固定 rows 换 probes 与固定 probes 换 rows，分别检验对两种训练采样的依赖；不据此重新训练或在已看过的 test 上选方法。
4. 综合 signed prediction error、四组新方向的变化和两套 rows 的差异，提出一个受限的 codebook 目标改动。

本轮只诊断固定输入 calibration 的机制。没有新方法效果、closed-loop 成功率、其他 baseline 或 native VQ latency 结论。有限 probes 和两行采样也不是完整 Jacobian。

追加的 CPU-only 小比较为 `25755155.pbs101`，申请 8 CPU、64GB、1小时，已核实完成：PBS F/Exit_status=0、exit receipt=0、PIPELINE_COMPLETE 和匹配 complete summary 齐全，用时13分15秒，四项自检及12项 CPU/GPU 差分核对通过。它复用成功 preflight 的 case0 capture，不加载模型或重跑 BF16/VJP，保持614模块、20directions和两套rows。以局部输出拟合为基准，分别计算 row-scale、λ=0.1、λ=1 三套冻结权重改动的分层投影与跨调用/跨层抵消；shared bias 和 teacher output 在 bank 差分中抵消。净能量/分项能量比可以大于1，不裁剪成百分比，不将有效抵消自动判错。每个组合另与保存的 GPU bank 差分核对，独立容差为 absolute1e-4 或 relative1%；原 proxy 复现容差不变。详细模块记录留在 compute storage，只提取小摘要。该实验不拟合新 codebook、不计算绝对逐层输出误差，不能直接证明改进后的 action 精度；完整八个 cases 的 GPU full 仍保留。
