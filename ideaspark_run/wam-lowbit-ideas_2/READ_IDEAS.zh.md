# WAM W4A8 量化：三个候选与筛选判断

2026-10-03。用户要求 ResearchStudio-Idea，目标 W4A4/W4A8，single precision 或 mixed precision 均可。初次 idea 生成共用真实文献 grounding，分别生成三案，再独立检查程序逻辑与先例；当时尚未运行模型实验。下列保留 research proposals，后续仅候选2完成了有界的粗粒度 W4A8 最小反证。

**后续实验结论：候选2当前粗粒度 recipe 停止扩展。** 作业 `25666715.pbs101` 完成448条 TEST记录，退出码0。Learned phase 相对预注册主比较中最强的 independent dither 降低 action MSE **5.28%**，8/8留出轨迹改善，但未达到冻结的 **10%** 扩展门槛，记为 `NO_GO_FOR_EXPANSION`。这是给定冻结 draws 的小幅正信号，不否定整个 phase 家族；native kernel、W4A4 和量化策略闭环均未验证。完整证据与 `Stageout_status=1` 交付 caveat 见 [最小反证结果](<D:/Downloads/Final Year Project/ideaspark_run/wam-lowbit-ideas_2/mechanism-screen/RESULT.zh.md>)，数学依据见 [推导](<D:/Downloads/Final Year Project/ideaspark_run/wam-lowbit-ideas_2/mechanism-screen/MATH_GATE.zh.md>)。

**用户另行授权的配对 follow-up 已完成。** 完整输出误差分解 `25667993.pbs101` 与单LIBERO-goal task0四arm、8组配对闭环同时启动；闭环首个作业 `25667994.pbs101` 在episode前因配置检查错误退出，修复后同批cases由 `25668140.pbs101` 完成。两个有效作业均F/Exit0。新draw组learned相对RTN的action MSE改善2.58%，主要体现为交互项减小；闭环BF16/RTN/independent/learned均8/8成功，未观察到成功率增益。原10%门槛与NO-GO保持不变，不追加phase搜索。完整证据及stageout caveat见 [双实验报告](<D:/Downloads/Final Year Project/ideaspark_run/wam-lowbit-ideas_2/paired-followup/RESULT.zh.md>)、[独立冻结协议](<D:/Downloads/Final Year Project/ideaspark_run/wam-lowbit-ideas_2/paired-followup/PREREG.zh.md>) 和 [运行记录](<D:/Downloads/Final Year Project/ideaspark_run/wam-lowbit-ideas_2/paired-followup/RUN_STATE.json>)。

| 候选 | 精度主案 | 当前定位 | 审阅状态 |
|---|---|---|---|
| 1. Controller-Prefix Command PTQ | 目标 denoiser Linear W4A8 | 工程 pilot / 应用增量 | coherence 0 blocking；prior audit advance，应用定位保留 |
| 2. Phase-Controlled Subtractive Dither | 目标 denoiser Linear W4A8 | 有明确干预自由度的高风险机制探索 | proposal复核pass；粗粒度真实模型pilot完成，当前recipe `NO_GO_FOR_EXPANSION` |
| 3. Quantized-History Quantizer Calibration | 目标 denoiser Linear W4A8 | on-policy calibration 辅线 / 对照 | coherence 0 blocking；prior audit revise→dev/test 修订；post-revision needs_work→明确离线 test-reference replay→第二轮 pass |

三案的 W4A8 指目标 Linear 的权重/激活位宽。Norm、Softmax、RoPE、非线性、residual、scheduler、controller 等仍保留 BF16/FP32；encoder、VAE、缓存、scale 和其他模型组件必须在真实实现中单列。没有任何一案完成 native kernel 验证，也没有全图4/8 bit或加速结论。正式字段以各 candidate JSON 为准。

初次研究选择中，候选1适合最小工程 pilot；候选2适合有明确止损条件的机制筛选；候选3作为 on-policy calibration 对照，不宜仅凭 DAgger 数据聚合包装成主贡献。后续候选2筛选已完成，按冻结门槛停止当前粗粒度 recipe；候选1/3仍未实验，不据此自动转入其他方向。Fixed-input action/command fidelity、closed-loop success、native latency 需分别建立，不能由本次 fake-quant 结果推断 A4 可行。

**1. Controller-Prefix Command PTQ：按机器人实际执行命令校准。**

WAM 生成的 action chunk 记为 a，但 evaluator 实际送给控制器的是 u=C_c(P_h a)：P_h 取已执行 prefix，C_c 包括现有反归一化、clip、gripper threshold 或实际存在的 ensembling。对相同 observation、seed、controller snapshot 比较 BF16 与 W4A8 的 u。量化器的 scales/clipping/合法 zero-point 以两级词典序目标校准：先减少 controller branch mismatch，再减少连续 command 误差。模型参数冻结，部署不需要 teacher 或额外 correction branch。

与 [Q-WAM v1](https://arxiv.org/abs/2609.33269v1) / [SteerQuant v1](https://arxiv.org/abs/2609.39056v1) 的差异在于固定 controller 后的 executed command 对象；[PreDE v1](https://arxiv.org/abs/2609.19441v1) 已有固定输入的 action deviation 筛选，因此不能把本案称为一般新 PTQ 原理。最适合先回答一个工程问题：full-chunk error 较小的配置，是否可能在 gripper/clip 或已执行 prefix 上更差。

最小反证使用 trajectory-disjoint fixed-input records，比较同位宽 local PTQ、action-sensitive PTQ 与本案；测 command error 与 branch mismatch。原案负对照置换 observation 对应的 controller state/prefix，实际 evaluator 若使这些量恒定，就必须先识别该对照无作用的情形。程序检查的合成结果不建立真实收益。不同 prefix 长度下采用 per-record 还是 per-command 聚合，以及恒定 command 维度的归一化策略，实施前须明确。校准成本依赖全模型坐标搜索的候选数，尚未得到 GPU-days。

**2. Phase-Controlled Subtractive Dither：通过量化器调节联合残差。**

局部近似下，最终 action perturbation 可写成 δa≈Σ_r J_r e_r，其平方含有不同 site 间的 cross terms。Q-WAM 的推导明确近似舍去这些项，SteerQuant 在 FP upstream inputs 上估计局部影响；这构成可检验的切入点，但非零 cross terms 本身不是新算法。

本案给每个 `(module, stream, denoising step)` 一个 phase φ_r。同一 sample 的 base draw U 经 d_r=frac(U+φ_r) 构造 activation dither；候选直接在完整 W4A8 path 上校准 phase table。对每个 token row，q=round(x/Δ+d_r)，反量化为 x_tilde=Δ(q−d_r)。冻结 packed W4 weights W_hat 时，Linear 输出可分成 Δ q W_hat−Δ d_r(1ᵀW_hat)，第二项由预存的 output-column-sum vector 在 GEMM epilogue 中处理。额外成本包括 RNG、phase/scales、column-sum state 和 correction arithmetic；不需要第二套高精度权重，但融合与耗时尚未验证。

核心假设是：这些 phase 自由度能改变真实联合误差，并在新 observations 上改善精确 endpoint action MSE。固定输入、独立 dither、无饱和时的经典条件性质，不会自动延伸到整个量化网络，因为 downstream input 依赖同一 U。跨站相关性、clipping 与 nonlinear propagation 都要在完整路径测；不提供无偏或稳定性保证。实施时必须补齐零 activation row 的旁路定义（输出零 activation，保留 Linear bias），并固定校准与评估的 round tie 规则。即便中间张量依赖 phase，最终 action 也可能因 downstream projection 为零而不变。

先比较 learned phase、同 scale/clip 的 RTN、independent dither、phase-site permutation 与同校准预算 direct endpoint PTQ。建议 trajectory-disjoint 评估另配新 draws，并在各 arm 内配对这些 draws，排除固定随机样本的拟合收益。如果 phase 只改变局部诊断量、无法降低 held-out full-path action MSE，或简单 endpoint PTQ 能匹配它，就停止这个 recipe。全体 phase 同时平移是 uniform U 下的 gauge symmetry；有些置换同样不改变 joint noise law，不能作为有效的负对照。后续已补充数学推导并运行完整 first_frame WAM 路径的8组绑定 phase 筛选，详见上方结果。原 proposal 的 0.5–1 个80GB-class GPU-day 仍是未经实测的较大 pilot 估计；本次有界筛选实际使用 A100 40GB、01:49:36，不含 native kernel / quantized closed loop。

**3. Quantized-History Calibration：在量化策略自己访问的 observations 上重拟合。**

当前量化策略与固定 evaluator 交互，生成 D_Q(α)。在这些相同 observations/history、相同 denoising randomness 上查询冻结 BF16 checkpoint，以其 action chunk 作为 calibration target；只更新各 layer/stream/step 的 A8 clipping α，W4 weights 与 scales 不变。更新后由新量化策略收集下一轮 history。部署是独立 W4A8 路径；teacher queries 用于校准，以及冻结配置后的离线 reference 测量，二者开销均不能归入或隐去部署耗时。

[QuantWAMs v1](https://arxiv.org/abs/2607.28405v1) 的 quantized-state profile 是 shift diagnostic，schedule proposal 仍由 FP16 reference replay 得到；PreDE 使用固定 logs 筛选。本案把量化候选自己的 state distribution 接入 quantizer 更新，但本质上属于 DAgger-style/on-policy calibration 应用，不把数据聚合本身当新贡献。

最小反证比较 static reference-history PTQ、one-pass on-policy calibration、multi-round refit 与 generic adversarial PTQ。负对照把更新信号换回 BF16 reference-policy histories。在每个 D_Q record 上，Q/BF16 比较是同 observation 的条件动作差异；但不同候选会访问不同 D_Q，直接比较这些均值可能奖励容易的状态。需另用固定的公共 replay set 配对比较，并单列各候选自身访问分布上的指标。停止条件只看 development split，最终 test trajectories 留到配置冻结后：先独立运行全部 Q-policy test episodes，保存 observation/history/cache、seed 与完整 action chunk，再离线查询 BF16 reference 计算 R_act。这些 labels 不用于拟合、停止、选择配置，也不影响已发生的 test action 或 occupancy。closed-loop success 另报；较低 action MSE 不建立更好的任务成功率。若 multi-round 不能优于 one-pass，就作为简单基线保留并停止扩展。建议 pilot 估计2–4个80GB-class GPU-days，未实测，不含大量 kernel port。

**W4A4 与 mixed precision 怎么接。**

三案正式主案均为 W4A8；本轮没有通过 W4A4 方案的同等审阅或实测。A4 可作为下一阶段的位宽变量，保持同一 checkpoint、机制与对照，先检查较大 activation error 是否仍可被当前自由度控制。

候选2有一个具体但尚未独立审阅的 signed INT4 算术扩展：非零 row 取 Δ=max|x|/6，同样 d∈[0,1)，则 x/Δ+d∈[-6,7)，nearest rounding 的整数 q∈[-6,7] 可落在 signed INT4 范围内；零 row 仍单独旁路。这只是避免 dither 引起整数饱和的范围构造，代价是更大的 Δ 和量化误差，不是 W4A4 fidelity 或 kernel 可用性的证明，亦不等同于 NVFP4 配方。

若少量 conditioner/action 输出路径必须 A8，则应报告为 mixed W4A4/W4A8，并统计升精度对象、比例、额外 kernel dispatch、weight packing 和完整内存。不能由 W4A8 成功直接推断 W4A4 可行。

**可用资源与证据范围。**

[FastWAM 官方仓库](https://github.com/yuantianyuan01/FastWAM) 提供 checkpoint 与 LIBERO evaluator；必须锁定 revision、checkpoint、inference mode、scheduler 与 controller。W4A4 可参考 [Nunchaku](https://github.com/nunchux-ai/nunchaku)，W4A8 可参考 [QServe/OmniServe](https://github.com/mit-han-lab/omniserve)，它们是工程参考，FastWAM 适配需另做。沿用历史 OTC/RankCal/CEM-Update recipe-specific NO-GO 作约束，本轮未重跑，也不把它们泛化为整个 PTQ 方向的失败。

Phase0 有31篇真实 connector records（10 core、21 adjacent），四篇 WAM 锚点全文取得。Phase3 collision 使用 arXiv/OpenAlex 的 signature 10-month 与 alias 48-month channels；candidate1/2/3分别有217/161/220条筛后命中。Semantic Scholar timeout/429、OpenReview 未覆盖，初始全文池中断后收窄为四个锚点；有限检索没有建立完整 novelty 证明。完整 workflow scope、正式 candidate、coherence/audit 与检索来源都保留在本目录。Phase4 manuscript expansion、双语三卡与PDF未运行；后续有界GPU机制筛选单列在 `mechanism-screen/`，不改写原 proposal/audit。

正式方法定义：[候选1](<D:/Downloads/Final Year Project/ideaspark_run/wam-lowbit-ideas_2/candidate-1/phase2_generate/phase2_generate_output.json>)、[修订后候选2](<D:/Downloads/Final Year Project/ideaspark_run/wam-lowbit-ideas_2/candidate-2/phase3_revise/final_candidate.json>)、[修订后候选3](<D:/Downloads/Final Year Project/ideaspark_run/wam-lowbit-ideas_2/candidate-3/phase3_revise/final_candidate.json>)。初审 critique 的 `revise` 与后续修订分别保留，不能改写为从未发现问题。

候选3第一次 post-revision 的测量问题保存在 [round-1 审阅记录](<D:/Downloads/Final Year Project/ideaspark_run/wam-lowbit-ideas_2/candidate-3/phase3_revise/post_revision_audit_round1.json>)。修订后的 [候选2复核](<D:/Downloads/Final Year Project/ideaspark_run/wam-lowbit-ideas_2/candidate-2/phase3_revise/post_revision_audit.json>) 与 [候选3第二轮复核](<D:/Downloads/Final Year Project/ideaspark_run/wam-lowbit-ideas_2/candidate-3/phase3_revise/post_revision_audit.json>) 均为 pass；这些是程序逻辑与 proposal 边界的审阅结果。后续模型证据仅以独立的 `mechanism-screen/` 报告为准；没有 task success 或 native speed 结论。
