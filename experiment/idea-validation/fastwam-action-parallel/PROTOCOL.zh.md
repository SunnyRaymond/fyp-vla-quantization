# Fast-WAM action inference：两条独立并行实验线

冻结日期：2026-10-03。目标是分别测清固定 context 重复投影的可消除成本，以及 denoising 时间并行的收敛、延迟与显存代价。先完成一个有界机制 pilot；不训练、不换 checkpoint、不增加任务、不以 teacher 付费初始化或纠错冒充部署加速。

## 共同基线

- 官方 revision `7faa71108368fbb3b6885649f112af607427a2d4`，源码只读。
- 现有 `libero_optional_idm_2cam224.clean.pt` 与对应 stats，沿用已确认文件，不重复 hash。
- `FastWAMOptionalIDM / first_frame`，BF16、compile=false、seed 42、sigma_shift=1、CFG=1；32×7 action，10 个原生 Euler 节点。
- `libero_goal` task0 / trial0，30 waiting steps、replan_steps=10、OSMesa CPU rendering。
- 两条独立 PBS GPU 作业均先运行同一原生 episode，从 replan 0、6、12 保存真实输入和核心状态。若 episode 不足 13 次 replan，使用首、中、末并明确记录；不伪称是指定 0/6/12。
- 两条线分别以自己作业内的同观测原生执行为参照；跨 GPU/作业的绝对 ms 不用于估计处理效应。若节点不是同型号 A100，报告型号并限定比较范围。

## 实验 A：固定 context 缓存与跨层批处理

四组：A0 原生；A1 缓存 action text embedding；A2 加各层 text/proprio cross-attention K/V 的逐层预计算；A3 把这部分固定 context K/V 投影跨层批处理。每个 chunk 都重新构建对应 context cache；只能在同一观测 chunk 内复用。A3 额外 packed weights 的持久显存、一次构造时间另报，不能隐藏为免费成本。

同一初始 action noise 和 schedule，比对每步与最终 normalized action；语义保持 gate 为 finite 且 `allclose(atol=1e-5, rtol=1e-5)`。不通过则记录 numerical drift，不降低阈值、不宣称精确替代。A2 对 A1 衡量 K/V cache 收益；A3 对 A2 衡量跨层批处理净收益。

主要时延为十步 action phase（包含每 chunk cache prepare）；另测完整 infer_action 以包含 VAE、encode_prompt、video prefill 与返回 CPU。原生与候选每 context 3 warmups、4 对 AB/BA，逐候选配对。没有逐层 hooks，边界 CUDA synchronize，独立记录 wall/CUDA event。

## 实验 B：denoising 时间并行

保持原生十个离散时间点。候选 window=2/5/10；修正轮数分别为 1/2、1/2/3/5、1/2/3/5/10。以窗口左端 action 初始化猜测，禁止 teacher trajectory 初始化。主研究为 prefix-Picard；保留 native-rounding triangular 对照解释数学固定点与 BF16 舍入的差别。

先把原生轨迹上的 action states/time levels 批处理，单独测 batch-forward 时间与输出差异；这是离线计算能力诊断，不是部署加速。再测实际无 teacher 的完整并行求解（包含 trajectory updates），报告 scalar NFE、batch forward 次数、轮数、窗口数量。

主参照是原生串行 action core，不叠加 A 线的 context cache；这样不会把缓存收益算成时间并行收益。每 context 3 warmups、4 对 AB/BA，候选固定，不根据结果扩大搜索。pilot fidelity gate：finite、全 32×7 normalized action `max_abs <= 1e-3` 且 `relative_L2 <= 1e-3`；另外记录首 10 个 action 最大误差及后处理 gripper 符号差异。此 gate 仅筛选计算近似，不证明 closed-loop task preservation。

## 显存、有效性与停止条件

- 每组独立记录基线 allocated/reserved、`max_memory_allocated`、`max_memory_reserved`；显存测量与主时延样本分开，不用间隔 nvidia-smi 采样充当 forward 峰值。
- GPU 每 15 秒采样 utilization/memory 写入 job log。所有模型/数据 I/O、推理、环境运行、toy 自检和结果汇总在真实 PBS compute allocation；login 只传小控制文件、提交、查询状态和读取小型摘要。
- 每个观测是一个 block；同 block 内交替 AB/BA。4 个计时重复是技术重复，不当作 4 个独立任务；报告各观测结果和配对比值，不报告全 suite success rate。
- 原生重放应匹配捕获的 episode action；不匹配则暂停该线性能主张并定位。OOM 记录对应固定候选失败，不悄悄减 batch。质量失败或没有净加速也是有效 NO-GO，不追加训练、降阈值或扩大任务。
- 两条线各自需 terminal PBS state/Exit_status、wrapper exit、测量 JSON/CSV、PIPELINE_COMPLETE。失败原始终态保留；修复工程错误后新 job 另列，不抹除前次失败。
- 本 pilot 不包含候选闭环成功率测试。若后续要声称可替代机器人策略，需另行冻结独立 episode 级比较；当前只报告 fixed-observation evidence。

## 方法来源

- Shih et al. (2023), *Parallel Sampling of Diffusion Models*, https://arxiv.org/abs/2305.16317 。采用并行 Picard 思路，不迁移其论文加速数字到本模型。
- 实验布局参考 K-Dense `experimental-design` skill：同观测 blocking、配对平衡顺序、技术重复与任务重复分开。Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, https://doi.org/10.48550/arXiv.2609.00065 （冻结时核对最新记录为 v2）。
