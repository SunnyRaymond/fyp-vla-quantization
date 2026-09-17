# Reference-branch rounding coupling：冻结最小诊断

2026-09-13。Adapted ResearchStudio-Idea C02/C04。新候选；原goal-coordinate四臂主张保留prior-only，本实验不修复任何已完成的经验no-go。独立设计审查和root gate通过后才执行；尚无科学输出。

## 问题与可证伪预测

前面的temporal/spatial实验关注误差聚合，去相关可能有益。本候选研究另一种计算结构：predicted current feature减去goal reference时，同一encoder rounding map的误差是否在经过冻结predictor后仍有利于抵消。不是提出新的SR算法；generic common-random-numbers/correlated-quantization已有先例，application novelty未确证。

冻结同一个DINO-WM Wall epoch65、FP32 predictor与所有非目标参数。对visual encoder 12个Transformer blocks内的48个Linear.weight做W4 symmetric per-output-row SR；bias、LayerNorm、patch embedding、proprio/action encoders与predictor均FP32。每个draw在current/goal调用时可复用，不能共享两侧activation或误用缓存。

三份draw d=0,1,2。对每个固定current/recorded-next pair与同一记录action，记录实际 propagated visual output y_d=P_F(E_d(current),action)、goal feature g_d=E_d(goal)，以及FP y_F,g_F。所有g与y均196×384。一次前向的weights保持冻结；没有ensemble prediction averaging。

比较完整3×3 pairing matrix：Shared取(d,d)三格；Different取(d,e),d≠e六格。两臂current端和goal端各自的draw边际频率严格相同，故两侧单独的feature误差预算也相同。Different是有限三draw的off-diagonal参考，不冒称本次已估计全部独立SR分布。

对每格定义 residual r_de=y_d−g_e，FP r_F=y_F−g_F。primary A_de=mean((r_de−r_F)^2)。先在每state各自对Shared三格、Different六格取mean，得到A_S,A_D。预测为Shared的A更小，且这种改善同时体现在scalar visual objective fidelity：B_de=(mean(r_de^2)−mean(r_F^2))^2，聚合为B_S,B_D。

Primary不是“量化objective更小”；FP本身不是环境oracle。它测量reference residual与scalar objective相对FP的误差，不测planner排序、action success或真实future预测改善。

## 为什么不要求predictor等变

定义实际输出delta Δy_d=y_d−y_F和Δg_e=g_e−g_F，则A_de=mean(Δy_d²)+mean(Δg_e²)−2mean(Δy_d Δg_e)。使用实际传播后的Δy，这个分解对任意冻结非线性predictor成立，不需假定输入误差被原样传递。

分解恒等式只作CPU一致性检查，不作科学发现。科学未知是共享同一weight draw是否使该checkpoint的实际跨分支误差有更有利的相关性，以及其作用是否延续到scalar objective fidelity。不能把符号结果推广成任意坐标变换不变性或新quantizer。

## 数据、量化与基线冻结

- 复用已登记teacher-bias六条Wall validation124..129（underlying1035/1534/1158/203/1837/1095），明确是新机制的复用诊断，不叫fresh independent confirmation。当前frame0，goal为同trajectory primitive frame5，使用已经dataset-normalized的第一个10D action block；只做H1。
- manifest `/tc1home/UG/yguo017/v100_newangles_ccds/teacher_bias_ready2/manifest.json` SHA256 `6dae677a0e7763896d735495500aea78a9254e59b67fa47817bb9bb9bb1f5c14`。不打开84..95 locked，也不扩取frames/seeds。
- CPU torch float32实施quantizer，逐模块按完整name词典顺序；三个CPU Generator seeds2701/2702/2703，每draw遍历所有48个模块，每权重一次torch.rand(shape,float32)。row scale=maxabs/7，零行scale1；v=FP/scale，code=clip(floor(v)+1[U<v−floor(v)],−7,7)，dequant=code×scale。weights复制到GPU，只做FP32 numerical emulation。
- 另保留encoder-only W4 RTN（同48个module与scale）以及FP exact-copy/restore no-op。RTN是背景对照，不参与Shared/Different的matched primary，也不根据它更改阈值。
- 每draw只需对每state编码current/goal并做一次FP predictor。保存current视觉feature、goal视觉feature、propagated视觉feature及实际predictor输入readback。Shared/Different pairing全部由CPU从保存的单branch outputs重建，不额外跑9次相同网络。

## 事前gate与停止

所有source/checkpoint/input identity、V100/allocation、eval/FP32、48目标Linear、quantizer重建、实际weight readback、non-target state不变、FPcopy/restore、shape/finite/completeness均须通过。CPU独立复算全部raw与3×3分解，并核对三draw RNG/code/scale与producer保存的weights。

每state先要求A_D>1e-12且B_D>1e-12以避免near-zero相对改善；不满足记nonbinding。至少4/6binding才允许科学gate解释，否则inconclusive_binding。binding state若(A_D−A_S)/A_D>=10%且(B_D−B_S)/B_D>=10%，记joint positive。至少4/6 joint positive才scope_limited_preliminary_go；工程通过且有至少4binding但不足4joint则mechanism_no_go。

10%与4/6是本次小规模经济性阈值，不是统计显著性声明。不从substate/单draw另挑成功结果，不调整seed、目标帧、bitwidth或module。无论go/no-go均STOP；不增加候选集、horizon或闭环任务。

## 执行与证据

仅真实CCDS SLURM UGGPU-TC1单V100；GPU allocation5min、producer内部180s，CPU verifier allocation5min。无新下载/环境安装/训练；复用已验证teacher helper及DINO runtime。head只提交/查状态/小控制文件，重I/O/hash/model/data/数值均在allocation，检查真实SLURM_JOB_ID/hostname/scontrol ownership，禁止伪造PBS或绕过guard。

Producer单独保存engineering.json、runtime.json、quant_snapshot.pt、raw.npz、protocol/source快照与SHA。CPU保存独立verification.json。无native W4速度/显存压缩、共享map额外成本或task outcome主张。每项结论保留失败原件，不用后来的成功替换失败记录。
