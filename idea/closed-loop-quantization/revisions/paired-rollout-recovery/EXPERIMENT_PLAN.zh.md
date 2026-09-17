# PRR 最小实验草案

2026-09-12；只准备方案，不授权或启动集群。先过接口/公平性检查，再冻结执行 manifest。

## 旧证据与新数据

旧 FRT A/B 保留 no-go。旧 CAL 60–65、DEV 66–71 已参与研究决策，不能作为新版独立验证。原 0–59 也属于历史保留范围。

新范围暂定 CAL 72–77、DEV 78–83、锁定 TEST 84–95，必须在执行前用轻量 manifests 验证这些 **底层 episode/initial-state fingerprints** 从未使用且互不重叠；dataset index 不等于独立 episode。若范围已用或两个 windows 属于同一原始轨迹，应顺延并记录，不能只改标签。新 CAL/DEV 各6 episodes、每个2个起点，3 fit seeds。它是资源筛选，不是有充分功效的确认实验；frames/actions/seeds 不增加独立环境样本数。

## R0：区分旧诊断混合的变量

可复用旧 hard checkpoints，但所有模型运行/大数组处理在真实 allocation 内。取同一原始 history 和同一 horizon，交叉比较 Q0 与旧拟合模型：在共同 FP history 上分别算 residual；再在共同指定的 quantized history 上算 residual。不能把不同 horizon/context 的 cosine 解释成纯方向漂移。该分析解释旧失败，不重新给旧方法打通过分，也不参与新 DEV 参数选择。

## R1：唯一的主实验是 target × 参数自由度

四格 A/B/C/D 定义见 IDEA。每格3个 seeds，初始计划1000 updates、batch2、H=2。共享 new CAL Wz、scales/rounding initialization、seeded minibatch schedule、hardening schedule；LoRA 两格使用相同初始化、rank4、同一学习率策略。拟合前可按 CAL smoke 统一缩减 step 数以适应预算，但不得按 DEV 为某个方法单独选 steps。

所有格使用 **相同冻结 donor histories**：Q0 和已有 `clean_seed_1201` 两个 hard models 在新 CAL 上各生成一条同 action prefix 的 history，各占1/2。旧 Clean checkpoint 只用来生成输入，不能作为新 benchmark 的结果；若不能确认其 CAL-only provenance 或文件完整性，则删除第二 donor，并在所有格共同改成 Q0-only。donor 不随候选参数变化，不在线更新 bank；这使 target effect 可辨识。是否需要 refresh 是后续问题，不作为本轮另一个可调因素。

比较 A↔B、C↔D 时，corrupted input、clean anchor、FP reference、action、Wz、γ=1、数据和参数自由度完全相同。A/B 间只改 target。四格使用相同更新/样本预算，单独报告各格实际 GPU 时间，不能把相同步数写成相同 FLOPs。

按独立审查补充：CAL 中记录各格 target norm、target-loss/clean-loss 比例、量化参数及 LoRA 的梯度范数；相同 γ 不代表相同有效梯度强度。B−A、D−C 先解释为所测 target/parameterization 的效果，不能直接称恢复机制证明。不能根据 DEV 单独重标某格 loss 来放大结论。

**参数化硬检查：** signed W4 范围[-7,7]、per-output-channel scales、相同24个 Linear。LoRA 必须先形成 `W_eff=W+BA`，再执行相同 hard quantization；不能训练一个未量化 bypass 后直接宣称 W4。合并后丢弃 B/A 和 rounding 参数，从序列化的整数/scale 重新加载评估，计入未量化参数及 metadata/padding。

旧 `soft_round_weight` 对 floor 路径 detach，不能直接用于 LoRA 后假定梯度能流入 B/A。新实现需声明统一的 STE/surrogate：例如训练时 `b_ST=x+stopgrad(floor(x)-x)`，`x=W_eff/s`，再加 soft rounding 并 clamp；硬化仍为 `clip(floor(W_eff/s)+1[h>=.5],-7,7)`。两种自由度使用同一 surrogate，仅 LoRA 开关不同；CAL smoke 检查 LoRA 与量化参数梯度、hardening/reload、FP null。不得把 soft 结果作为主结果。

**history/action 硬检查：** 延续实际 `num_hist=1`，只按 source 允许的 history 更新。每个 action 先参与它应影响的预测，再由 `replace_actions_from_z` 写入下一已知 action；不要把一个新替换、尚未被 predictor 使用的 action 当作动作敏感性测试。paired branch 同 initial state、同实际动作序列和 RNG。仅 mask 已知 action coordinates，不能混入 teacher action 选择或以修改 action 输出来降低损失。

## 指标与可解释结论

主指标是最终 hard model 从共同初始观测 **自身自由 rollout** H=2 的 terminal observation/proprio weighted error，相对同动作 FP 路径。评估输入不能停留在固定 donor histories，否则只有 teacher-forced recovery evidence。另记录 clean error、共同 donor-bank recovery error、逐 episode/seed结果；所有格使用相同 FP batch reference，保留旧 batch-dependent null 检查的修复。

资源决策门槛暂定：B 相对 A 或 D 相对 C 的总体平均至少改善5%；至少2/3 seeds各自的episode平均也改善≥5%，且该seed至少4/6 DEV episodes方向改善；clean error 不劣化超过10%。这是筛选规则，不是显著性。完整四格即使 A/B 未改善也跑完：否则无法检验“quantizer-only 无信号而 LoRA recovery 有信号”的分支。CAL 工程失败/梯度无效/预算不完整属于 inconclusive，不判科学 no-go。

- B优于A：支持当前 quantizer-only recovery adaptation，尚未证明量化残差专属机制。
- D优于C、B不优于A：与恢复需要额外权重适配的解释一致；不能推出普遍表达能力下界。
- C、D同样优于A、B：更像一般权重适配收益，没有 recovery-specific 支持。
- 四格均无稳定方向：停止此恢复配方。

interaction 使用同 episode/seed 的 `(E_D-E_C)-(E_B-E_A)` 报告原始值及方向；不以四个 aggregate 排名代替 interaction。small pilot 不作显著性交互声明。

## R2：有信号后必须补的近邻对照

在胜出参数自由度下，补齐：① 相同 target 的 Wz-norm-matched random history corruption，保留原 action 槽；② 真正的 H=2 unrolled trajectory distillation，含它自己的输入梯度链；③ clean-only 对照。使用同 CAL/DEV、参数与样本/更新预算，并计入单独 GPU 开销；如果内存不允许 unroll，记录 unavailable，不能声称优于它。

固定 donor 的 PRR loss 是 detached trajectory-reconstruction 的适配，不是数学上新发明的多步目标。与 full unroll 的区别包括 history 生成器和跨步梯度；不能把比较结果仅归因于其中一个。AccuQuant 的低内存路线是直接近邻，不能用 naive unroll 太贵来忽略它。

action-response 检查：对相同初始状态构造两条合法、不同 action prefixes，其余 suffix相同；分别自由 rollout FP/Q，比较非-action outputs 的预测差异，检查是否把不同动作压成近似同一结果。先用 FP response 大于 CAL 冻结数值 floor 的 pairs，再评估，不能拿已知 action embedding 的一致性冒充 dynamics 一致性。只降低 latent MSE而压平actionresponse，不推进。

R1+R2最多6 allocated GPU-hours，含12次主拟合、必要对照、collection、失败与验证；先单卡估算整体账本。若完整必要对照不能落入预算，保持方案未验证，不删去不利对照。最多4×A100并发；旧 V100 计时仅用于规模参考，不证明新 LoRA/backward 或 A100时长。禁止自动增 rank/horizon/预算。

## R3：独立 TEST 与闭环（仅形成下一阶段计划）

R2有信号才冻结方法，使用未打开的新 TEST：相同 checkpoint、source、CEM5/300 candidates/30 elites/H5 与 objective_fn，比较实际 first-action、goal progress、环境 xy drift/success。只改善 FP rollout fidelity 而不改善环境 endpoint，降级为数值适配结果。R3另行预算，不包含在R1+R2的6 GPU-hours中，不能宣称6小时完成全部研究。

## 集群与证据边界

CCDS 仅在真实 SLURM allocation 中核验 job RUNNING、UserId、实际TC1N hostname/NodeList和GPU；ASPIRE2A 则核验真实PBS_JOBID、实际非-login hostname和allocation/GPU。所有模型、hash、传输、数组校验与计算都在 compute 内，login仅轻量控制；保留host-key verification，不记录credentials，单目标单写者。原大数组继续留在compute，不为此次文档修订下载或重跑。

本设计使用 experimental-design skill 的配对/分组原则；方法工具来源：[Kassis, Agarwal, He, Patel, Brueckner (2026), Scientific Agent Skills](https://arxiv.org/abs/2609.00065)。该引用仅说明工作流来源，不作为 PRR 有效性证据。
