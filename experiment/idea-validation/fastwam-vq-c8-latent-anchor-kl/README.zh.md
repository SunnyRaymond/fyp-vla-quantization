# C8 A+Z、ρ=1.5 加 latent anchor KL

**状态：第一轮已完成。** Heldout first10 motor RMSE 从 0.0442968 变为 0.0458895，future latent NMSE 从 0.0608774 变为 0.0614710，当前 KL 未改善主要指标。保留原 baseline。结论见 [results/CONCLUSIONS.zh.md](results/CONCLUSIONS.zh.md)，完整配对数据汇总见 [results/ANALYSIS.zh.md](results/ANALYSIS.zh.md)。

**Follow-up：机制诊断已完成。** 完整 16 组参数梯度、22 个回滚更新条件和 8 个等 NMSE reader 对照齐全；GPU 25905057.pbs101 与三项 CPU 作业均 F/0，artifact/coverage 验证通过。主要证据支持优先处理有限步/量化部署的优化落差；beta 降十倍不稳定，当前 affinity KL 的额外 motor 价值也未被证明。结论见 [results/MECHANISM_CONCLUSIONS.zh.md](results/MECHANISM_CONCLUSIONS.zh.md)，完整数值见 [results/MECHANISM_ANALYSIS.zh.md](results/MECHANISM_ANALYSIS.zh.md)。设计见 [MECHANISM_DESIGN.zh.md](MECHANISM_DESIGN.zh.md)，补充 CPU 几何的事后分析范围见 [MECHANISM_POSTHOC_OPTIMIZER.zh.md](MECHANISM_POSTHOC_OPTIMIZER.zh.md)。原 baseline 继续复用，0 持久训练更新、0 新闭环；原 RUN_STATE.json 仍保留第一轮训练/评估状态，本阶段状态见 MECHANISM_STATE.json。

本轮遵循用户更正后的范围：复用原 MSE+NMSE 对照，只运行新增 KL 分支；暂不做 280 个闭环 episodes，不引入 activation clipping 或 per-channel scaling。

对照是原始 C8、1 epoch / 8 updates 的 rho1.5 checkpoint。新分支从与对照相同的**未训练 C8 banks**出发，使用原作业 `25814264.pbs101` 保存的训练实现、相同顺序的八个 case、Adam 5e-4、gradient clip 1、update1 后一次 optimizer reset。只训练 614 个 Linear 的 codebooks / log row gains；保持普通 dynamic A4、BF16 KV、固定 indices / B32 exponents / Smooth0.5-H128。

新增 loss 为 `action_MSE + 1.5*c*future_NMSE + beta*KL(p_teacher || q_student_to_teacher_anchors)`，其中 `c=0.028303017887797623`。KL 在每个 latent token 的局部邻居集合上定义概率：同相机同时间的八邻域，以及同位置相邻未来时间；teacher anchors 停梯度。排除 clamped frame0、self 和跨相机边。每个相机单独做 2×2 pooling，channel 使用固定的 teacher 标准差归一化。

仅用八个 C8 teacher future 拟合 channel mean/std；temperature 是 C8 正 teacher 邻居距离的中位数。`beta=0.1*(1.5*c)*mean(initial_parent_NMSE)/mean(initial_parent_KL)`，完全用原始缓存计算并在训练前冻结。0.1 是预设的初始 loss 数值占比，不代表梯度占比或最优权重。heldout 不参与参数选择。

对照的 8 calibration + 80 heldout 查询从既有 baseline 分支逐条复用，保留原始记录并附来源路径；缓存目录虽含 activation-clipping，所取 baseline 本身没有 clipping。原始 reference_cache 的 parent 是**未训练初始 banks**，只用于初始化检查和 beta，绝不作为训练后对照。

新增分支只做 8 次训练更新、8 个 calibration 查询、40 cases × 2 seeds 的 heldout 查询，以及一次 native infer parity。无重试时预计 97 次逻辑模型查询；不是内部 denoiser forward 次数。新 baseline 查询 0、新 teacher 查询 0、新闭环 episodes 0。新分支保存 action/future tensor，后续可复用；baseline 没有完整 tensor 的 KL 不补跑，明确标为 unavailable。

先在 case 内平均 seed，再在 suite 内平均 case，最后四个 suite 等权。主要比较 first10 motor RMSE 和 future latent NMSE，并报告 full32 / late22 motor、gripper RMSE 与逐 suite 取舍。相同任务上的 distinct demonstrations 仅提供现有固定输入泛化检查，不能声称 unseen-task 泛化或闭环改善。KL 是 latent 邻域 affinity 分布的蒸馏，不能称为真实 future density 或 mutual information 的直接估计。

运行阶段仅 `prepare → anchor_kl → aggregate`。CPU 4 cores / 32GB / 45min；GPU 1 GPU / 16 cores / 110GB / 6h，并每 15 秒采集 GPU 利用率/显存写 job log。计算、tensor 读取和聚合只在真实 PBS allocation；监控每 20 分钟只读检查精确 handle，Q/R 无实质变化保持安静。

旧 C80 per-channel 实验已被用户本次更正取代，相关五个未完成作业已撤销，记录见旧目录 `USER_SCOPE_CHANGE_CANCELLATION.json`。
