# 本轮范围与约束

用户原始请求：wam quantization使用researchstudio-idea给我几个idea，最好量化层级能到W4A4/W4A8，混合精度或者单一精度都可以,

本轮为 ResearchStudio-Idea / idea-spark 的精简多候选流程。共用真实 connector 文献检索与 full-text grounding，保留 Phase 1 bottleneck、每个候选的 Phase 2 selection/generation、独立 coherence trace、Phase 3 collision/critique 与最小证伪。用户先前明确要求省去用不到的 pipeline 部分，故不做 Phase 4 论文式扩写、三语卡片与 PDF 渲染。保留所有候选及其明确 verdict，不能把停用的阶段写成已完成。

优先级与解释：
- 对象为机器人 World Action Models 的模型权重/激活量化；不是 action tokenization，不是单独 latent world model+CEM。
- 目标 W4A4 或 W4A8，可采用混合精度，必须交代所有高精度例外、有效位宽/显存与 operator 类型。
- 默认先考虑 frozen-weight PTQ；若引入量化器参数校准、轻量 QAT 或 weight update 必须明确分类和成本。用户没有排除 QAT。
- 当前只做 idea 生成与文献/逻辑筛选，未授权也不执行任何 GPU 实验。不以软件 emulation 声称 native INT4 speedup。
- GPU 硬件和 campaign 预算尚未由用户指定；不能把 skill 的 factory default 150 GPU-days 当用户实际资源。试验预算仅给建议上限与需测量的未知项。
- 当前日期 2026-10-03，Asia/Singapore。

已知近期 prior art（必须读原文，不仅摘要）：
- Q-WAM: 4-Bit Quantization of World Action Models with Action-Subspace Protection, arXiv:2609.33269v1，2026-09-27：Action Observability Gramian + action-sensitive low-rank high-precision protection。
- SteerQuant: Steering Quantization Error with Action-Guided Scaling in World-Action Models, arXiv:2609.39056v1，2026-09-30：stream/layer/denoising-step final-action sensitivity + shared channel scaling + stream activation modulation + Rudder kernels。
- QuantWAMs: Calibrating at the Right Granularity for World Action Models, arXiv:2607.28405v1：shared-basis outlier calibration + joint video/action empirical Fisher + closed-loop replay precision schedule。
- PreDE: Predict Before You Deploy, arXiv:2609.19441v1：policy-specific offline action-deviation configuration screening; needs closed-loop development labels。
- SVDQuant, arXiv:2411.05007：low-rank outlier absorption and fused real low-bit inference，直接WAM baseline而非WAM独有机制。
- DSAQuant, arXiv:2609.04031：denoising-stage-aligned video QAT；QVGen, arXiv:2505.11497：auxiliary low-rank QAT with rank decay；仅可迁移参考，video fidelity不等于robot control。

历史负约束（来自已完成本地实验的历史总结，未在本轮重跑）：
- OTC-PTQ on Fast-WAM Optional IDM `idm`: discrepancy C_r=d_a+d_o 被 observation 分量主导约99.77%，与Local MSE选同一top-2，原recipe为NO-GO。不要换个action-aware loss名字复述它。
- RankCal与CEM-Update原recipe在DINO-WM为NO-GO；这不否定所有decision-aware量化，但不能套用failed allocator并声称新方向。
- Fast-WAM `first_frame`、`idm`、joint inference以及Cosmos action-only/jointvideoaction路径必须分别锁定，不能把已省略的video分支当推理瓶颈。
- 保持 predictor/固定输入action诊断/closed-loop/部署延迟 evidence 分开。局部补偿或teacher-paid repair只作诊断，部署方案不能每次调用BF16 teacher。

操作约束：
- 不使用 banked reset。不写入或显示 credentials。不修改 skill、.env、旧实验或旧ideas。
- 本轮文献和脚本控制在本机进行，不接入集群。若以后做 ASPIRE2A 实验，login node 仅连接、提交、状态和轻量控制，重I/O、环境安装、模型加载、推理和benchmark必须在真实PBS compute allocation，检查PBS_JOBID与非login hostname；GPU作业需记录利用率和显存。
- 每个独立 LLM phase 使用文件输入的隔离 subagent。遵照用户当前 AGENTS 配置 gpt-6-luna，详细生成/审查使用 max，机械任务使用 xhigh。
