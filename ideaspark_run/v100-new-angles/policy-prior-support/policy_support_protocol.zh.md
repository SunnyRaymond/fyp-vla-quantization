# Policy-prior proposal support：冻结最小协议

2026-09-13，任何本案新 observation/model output 产生前冻结。ResearchStudio-Idea C02 controlled diagnostic + C04 decomposition；不做完整 pipeline 或完整 planner 验证。

假设：actor 的 W4 数值量化可能削弱 policy proposals 对 FP planner 第一次 update 的贡献。只改变候选生成，评分模型始终 FP。此 screen 与历史固定候选池 score fidelity/CEM update 的干预不同。Novelty 未认证。

## 身份与输入

复用 TD-MPC2 commit e9f59321933cbc8e11a002b842adc7d4ffae8ff1、compatible seed3 checkpoint SHA256 0e2c0eada8f160dd65c8faaa5e4ca2f8f496104e21433e334d716b73257a52c2，parent manifest SHA256 9240979105b919f6051f27261d00ff33652105f962d039728638caabb042d369。复用 tdq_screen.py，LF SHA256 3688d97e4c6e2bed4da4ff7aa552605690c3571e49dcf4c026e955b3c642eb11。

Cartpole-balance 8 个 fresh reset seeds 5217..5224，DMControl reset only，obs [8,5]；不 step、不 render、不选样本。CPU allocation 准备且冻结 manifest hash 后才允许 GPU。不会读取 Wall locked test 或更换 checkpoint。

实际 resolved config 已从旧 64807 runtime receipt 核对：horizon=3，num_pi_trajs=24，num_samples=512，num_elites=64，iterations=6，temperature=.5，max_std=2，episodic=false，action_dim=1。保留配置本身，明确仅截取 t0 的第一次 update，initial mean=0/std=2；不称完整官方 planner。

## 三臂与随机数

固定顺序 FP-policy、W4-policy、random-replacement。每个 state i=0..7：policy seed=7501+i；common488 random seed=7601+i；replacement24 random seed=7801+i；pool scorer seed=7701+i；update-mean scorer seed=7901+i。CPU/CUDA torch seeds 每次重置。FP 与 W4 proposals 均执行官方 model.pi 的 Gaussian sampling + squash、FP dynamics 递推 H3，使用相同随机数起点；保留原始 policy noise receipt 或等价 RNG 前后状态证据。

前24 slots 分别放 FP policy、W4 policy、random24；后488 slots 必须逐元素相同，所有 random 为 N(0,1)*2 后 clamp[-1,1]。只在 W4 proposal generation 期间替换 model._pi 中全部 Linear weight：逐输出通道 symmetric RTN qmax7，scale=maxabs*(1/7)，zero row scale=1，round/clamp 后 FP32 dequant。Bias 和所有其他参数不变。每次 scoring 前恢复 FP actor 且实际读回验证；保存原 weight、scale、integer codes、dequant readback 与完整受影响层名。无量化则随机数配对应产生完全相同 proposal（一次同输入 FP replay guard）。

## Official update 与观测量

使用 pinned TDMPC2._estimate_value（轻量 proxy，避免构造训练 optimizer）评分全部512候选，terminal pi 和随机 two-Q pair 每臂同 scorer RNG。episodic=false。每个 arm 取 official top64，score=exp(.5*(value-max))/sum，得到 mean μ 与 std（保留但不进入第二次 update）。保存完整 actions [8,3,3,512,1]、values [8,3,512]、elite indices/weights、μ [8,3,3,1]、std、前24slot inclusion/mass。

每个 μ 用 FP 同一 scorer、batch1、相同独立7901+i seed 复评 J。模型与planner配置 num_samples=512 保持。源码 line117 无条件创建 [cfg.num_samples,1] termination，即使非 episodic，batch1 会广播成512；故仅 μ scorer 的浅拷贝 proxy.cfg.num_samples=1，以匹配scratch tensor形状，其他字段完全一致且 model.cfg 不改，输出必须 [1,1]。该 shape adapter 在任何本案数据产生前明确冻结。G_FP=J(μ_FP)-J(μ_random)，G_Q=J(μ_Q)-J(μ_random)。共同488的 score 必须逐元素一致或 allclose(atol=1e-5,rtol=1e-6)；否则 implementation_inconclusive。保存实际最大差，不能把数值0误当失败的布尔 gate。

Binding state：G_FP>1e-4 且 FP first24 weight mass > random first24 mass+1e-4，且所有值有限、top64 cutoff无精确tie。少于4/8 binding => inconclusive_binding。

在 binding state 中，G_Q<=0.9*G_FP 且 Q first24 mass<=0.9*FP mass 为 joint damage；至少4个 => scope_limited_preliminary_go，否则 mechanism_no_go。同时报告 elite inclusion、action drift、clamp saturation，joint mass/value effect 不声称 causal mediation。负结果不调阈值/seed、位宽或任务。

## 证据与预算

GPU single V100 <=10min，内部480s/外部540s。CPU preparation/replay 各<=5min。所有重 I/O、数值运算及 checkpoint/source hash 在真实 SLURM compute allocation，脚本检查 allocation/job/host/partition；head仅小控制文件和scheduler。无本地数据计算。

保存 raw arrays、quantization snapshot、resolved config、source/checkpoint/runtime/runner/protocol/input identity、RNG配对 receipt、所有完整性与guard状态；独立 CPU verifier 重算 top64/weights/mean/std/mass、J差与冻结结论，并验证 raw hash、维度、有限性、非actor不变、FP restore/readback。模型 J 本身只由已绑定 FP GPU scorer 产生，CPU 不伪称复现模型推理。

完整性或预算失败 => 对应 implementation/resource/budget inconclusive；不推断科学结果。无论 go/no-go/inconclusive 都归档 STOP，不作 closed-loop return、泛化、部署 INT4 或速度收益声明。
