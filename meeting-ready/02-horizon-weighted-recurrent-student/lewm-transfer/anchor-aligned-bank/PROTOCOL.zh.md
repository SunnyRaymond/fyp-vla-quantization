# Anchor-aligned training action bank 配对实验

本轮只检验 training state 与 action-bank 中心的匹配。旧 temporal-balanced 实验为控制
变量，把 state/goal 移到 temporal anchors 后仍复用 anchor0 bank；这不是实现错误。
新实验将 current-anchor bank 作为唯一 treatment，不改模型、loss、训练预算或 contexts。

## 配对与来源

Control 复用 job25213164 的 reconstructed balanced_base terminal checkpoint 和
prepared balanced rows，保留历史训练来源，不声称本轮重新训练了 concurrent control。
Treatment 从同一初始化开始，沿用512 contexts、ordinal temporal anchor规则、h256、
AdamW、batch8、64 candidates、3000 updates、原 horizon latent MSE + score-distill loss。
不从control terminal权重fine-tune，以免改变训练预算。

按原 dense.prepare_dense_rows 的 CPU RNG consumption 顺序重放每context的两次单候选、
64候选、再H×action_dim的noise draw。在64候选前复制generator state，分别围绕旧anchor0
和当前anchor动作生成候选。必须在PBS内确认重放的旧bank与cached bank相同；不得通过
clip后action相减恢复noise。两臂使用同一pre-clipping innovations，仅候选中心改变。
Treatment teacher targets和cost重新计算。原state/goal/anchor/context schedule保持一致。

## Fresh evaluation

固定shuffle seed20300903的valid[552:560]，排除valid[:552]；该slice原为未运行StageB
预留，本次首次使用。8 episodes×3 anchors×2 fresh seeds20300967/20300968=48 blocks，
各300 candidates。两臂共享当前anchor的evaluation bank与完整teacher score。
Episode是真replicate，anchor/seed为nested measurements；这是有界mechanism pilot，
不作population success或正式power结论。旧失败样本仅形成假说，不进入本轮fresh test。

Primary为recall@120，recall@60仅描述，不按结果挑K。报告Spearman/top30、relative latent
MSE、full elite containment、raw teacher elite mean-cost regret，以及除以同block全部
300个teacher costs population std（floor1e-6）的standardized regret。

每episode对6blocks取median，计算treatment-control。Mechanism gate需全部满足：
recall120 paired delta的median>0；至少5/8episodes严格改善；recall<0.8的block数减少；
worst-block recall不低于control；standardized regret paired delta的median<=0；
finite/interface/paired-noise有效。另独立报告absolute screening gate：median recall120
>=0.95、minimum>=0.8、每temporal stratum median>=0.95。相对机制改善不等于replacement。

## 小型耗时拆分

同一GPU缓存相同state/goal；全部48blocks，3warmups/10repeats，seed20300969交错顺序，
CUDA同步wallclock。记录teacher300/120/60、两student-only300、两hybrid120。
Teacher小batch取common bank前K条，仅用于计时，绝非质量对照。所有路径包含对应cost
和elite selection；hybrid包含student300、sort/gather、teacher120、teacher精挑30。
完整teacher shadow只用于诊断，不混入production timing。耗时结果独立于训练机制gate，
不得仅因训练指标改善就声称hybrid加速。

## 执行与停止

单个有界GPU作业；PBS guard、每5秒GPU utilization/memory写job.log，正常或失败及时退出。
Login仅连接/小控制文件/提交/轻量状态。模型HDF5、训练、teacher推理、benchmark、重I/O
全部compute allocation内执行。模型与rows留cluster，仅回传small summary/log/status。
旧NO-GO不改写。无official CEM、closed-loop、warm-start或新架构实验；不改阈值补救结果。

实现与PBS监控由Luna xhigh负责；主agent不重复轮询。主Codex当前used83（剩17%），若
突然used0（剩100%）立即停止agent推进并交接；Luna reserve原本100%不触发。

设计组织参考：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents.
https://doi.org/10.48550/arXiv.2609.00065 （本会话已核对current record为v2；不是实验结果证据）。
