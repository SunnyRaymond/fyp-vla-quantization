# VQ 表示、action/latent 监督与机制诊断

返回 [总交接](../HANDOFF.zh.md)。本部分的旧 22-case 面板、旧 Spatial C8 与后来的四-suite C8/C80 是不同数据合同。除明确标为闭环的条目外，指标都是相对 BF16 teacher 的固定输入 fidelity，不是环境成功率。

## fastwam-vq-action-aware

**问题与实现。** 从简单 additive VQ 依次增加 BF16 row gain、local Linear-output fitting 和 action-sensitive proxy。表示为 d4、两本 256-entry BF16 books、uint8 indices；共享 Smooth0.5/H128、dynamic A4、BF16 KV。所有新拟合从相同 row-scale bank 开始。Calibration8、selection4×2、test22×2；action-sensitive lambda 只在 selection 选择。

**结果。** Test first 10 motor RMSE：原始 VQ0.07131、row-scale0.05705、local-output0.05726、action-sensitive λ0.1 为 0.06103，Scalar 参考 0.05499。Row-scale 相对原始约改善 20%、19/22 cases 更好；之后的 proxy 优化没有继续改善实际 test。

**关键失败。** Action-sensitive 训练 proxy 降到初始约 0.0048%，但 calibration 真实 motor 仍 0.04851，差于 local-output0.04671，说明失配在训练输入就存在，不只是泛化问题。当前方法不继承该 proxy 作为训练主目标；保留 row-scale 表示，并转向真实 sampler 末端监督。

**接手入口。** [RESULTS.zh.md](../idea-validation/fastwam-vq-action-aware/RESULTS.zh.md)、[README.zh.md](../idea-validation/fastwam-vq-action-aware/README.zh.md)。`25743693.pbs101`为主要 GPU 来源；不要混淆这里的 lambda0.1 与后续 latent loss 的 rho0.1。

## fastwam-vq-proxy-diagnostic

**问题。** 区分 proxy 失效是 probes 没覆盖最终 60D action，还是连已观察方向也预测错，并检查固定 token rows/probes 的依赖。第一阶段只读 8calibration×4 banks 的已保存 actions，不重跑模型；后续重 capture 与 old/new rows×old/new probes 交叉比较。

**已经确认。** λ0.1 版本训练 proxy RMS 约 0.00192，真实原四方向投影 RMS 却 0.04963，完整 motor0.04851。原方向上的真实误差没有随 proxy 消失，因此“其余 56 维没覆盖”不能单独解释。成功 case0 preflight 及 CPU 差分确认采样组合依赖：同一 bank 的 old/old 很小，换 rows 或 probes 后 proxy 大幅回升。

**完成范围。** CPU 第一阶段`25749368`、case0 后续差分`25755155`有完整结果；根报告最后仍把 full8-case `25754359`列为未完成。该 Q 是旧快照，不是今天的实时状态，本交接不把 full8 伪装成成功。下一步若重启，应先核实原 owner 的最终 receipt，而非按旧 handle 盲目 resume。

**结论限制。** 修正抵消原误差，不等于模块修正之间异常互相抵消；不应据此引入“禁止跨层抵消”的新损失。当前主线已使用真实 output loss，不依赖补完这批旧 proxy。

证据：[RESULTS.zh.md](../idea-validation/fastwam-vq-proxy-diagnostic/RESULTS.zh.md)、[README.zh.md](../idea-validation/fastwam-vq-proxy-diagnostic/README.zh.md)。

## fastwam-vq-action-distill

**问题与设计。** 放弃间接 proxy，对完整 student sampler 的最终 first 10×6 motor 直接蒸馏。训练 614 层 books 与 row gains，indices 和底模固定；从 row-output parent 开始，C8×2epochs=16 updates，4-case selection 选 epoch，完整 22-case test 只在冻结后评测。

**结果。** Parent→selected epoch2：test first 10 motor **0.057256→0.058110（+1.49%）**，full 32 **0.066828→0.069572（+4.11%）**，late 22 **0.069300→0.073342（+5.83%）**；calibration 则 0.046706→0.044329。13/22 cases 改善但少数大退化使 primary mean 更差，不能用中位数或胜例数替换主指标。

**实现与恢复。** 16 次更新均有 614/614 connected finite nonzero targets。首次 legacy checkpoint 缺 Adam state，恢复时显式 reset 一次，后续保存完整 optimizer/RNG；不能宣称与不中断 16 步 Adam 完全等价。无新增 teacher queries，0 episodes，decoded-BF16。

**当前决策。** 该 selected checkpoint 不替换 parent；这轮促成 future-latent 诊断和 A+Z。不能将 late 22 退化单因归为未监督 horizon，因为还没做因果干预。

证据：[ANALYSIS.md](../idea-validation/fastwam-vq-action-distill/ANALYSIS.md)、[RESUME.md](../idea-validation/fastwam-vq-action-distill/RESUME.md)。

## fastwam-vq-latent-diagnostic

**问题。** Action-only 更新是否破坏了想象的 future，以及使用 teacher future 能否修复 action。固定 parent 和 distilled epoch2，不训练；同一 60 contexts 比较 BF16、parent/self、parent/teacher-future、distilled/self、distilled/teacher-future。

**实施。** final clean future 不等于视频最后一帧；排除 clamped frame0，记录整体/逐 latent 时间位置/temporal-delta 误差。替换 future 后重新 prepare/prefill K/V，保持 action 初始 noise 和下游 checkpoint，不能只替换一个已经无效的缓存变量。两 shards 均有 self-future 替换 parity。

**状态与结论。** GPU `25787685/25787687`与 CPU `25790422`完整结束，60 contexts、5 arms、300metric rows，22testcases 全部保留，self-replay 差异 0。结果支持检验 A+latent 组合：calibration 的 action 拟合可伴随 latent 恶化；但本轮没有训练 latent-only，也不能从相关性指定某一语义特征为中介。详细指标见该目录分析产物，不在缺少对应表的根 README 中补造数值。

证据：[README.zh.md](../idea-validation/fastwam-vq-latent-diagnostic/README.zh.md)。

## fastwam-vq-action-latent

**问题与实现。** 在与 action-only 相同的 parent、C8、16 updates 和 selection 规则下，加入 teacher clean-future NMSE；student 仍使用自身 future，没有 teacher forcing。训练前用 C8 确定 w=0.0029918822946277566，使初始 latent loss 数值约为 action 的 10%，这不是 10%的梯度占比。

**结果。** Test first 10 motor：parent0.057256、action-only0.058110、A+Z **0.052951**；full 32 为 0.061332、late 22 为 0.063866；future NMSE 从 parent0.084435、action-only0.093011 降至 **0.071787**。Gripper 反而从 parent0.018110 到 0.018826；更远 horizon 仍不及部分 Scalar/row-scale 参考。

**含义。** 支持保留 action 和 latent 双目标。修复部分严重 case 拉动均值，不能说所有 case/horizon 全面改善；已见 test 属于 recovery diagnostic。该旧 row-output 实验的 w 不能直接复制到后来 Microscaling/C80，必须在各自起点按 calibration 重算并冻结。

证据：[INTERPRETATION.zh.md](../idea-validation/fastwam-vq-action-latent/INTERPRETATION.zh.md)、[README.zh.md](../idea-validation/fastwam-vq-action-latent/README.zh.md)。GPU`25798439`、CPU`25799647`均完成。

## fastwam-vq-microscale

**问题。** Row-scale 的单个 output-row gain 不足以匹配行内幅度变化；在相同 d4/two-book 容量上加入 B32 power-of-two scale。一个 B32 覆盖 8 个 d4 vectors；额外 uint8 exponent 为 0.25 bpw。拟合按真实 scale²和 A4 输入 diagonal second moments 加权，不只是先归一化后无权 K-means。

**冻结比较。** 同 Smooth0.5/H128/A4/BF16 KV，614 Linears、8calibration、4selection×2、22test×2；不训练末端 loss，不跑闭环。

**结果。** Row→Micro test first 10 motor **0.05705206→0.05240395（−8.15%）**，14/22 cases 改善，通过 mean 下降≥5%且≥12/22 改善的工程 gate。Selection 反而+1.11%，所以不称普遍稳定改善。含 transforms 的 target effective bpw **4.020873→4.270878**，payload 约 3.024→3.212 GB。Gripper 比 row 更好，但仍不及历史 Scalar。

**接手意义。** 这是后来 MXVQ 表示的起点；B32 是 weight microscaling，不是 activation block quantization。Indices/exponents 在后续 distillation 中通常冻结。

证据：[RESULTS.zh.md](../idea-validation/fastwam-vq-microscale/RESULTS.zh.md)、[micro_codebook.py](../idea-validation/fastwam-vq-microscale/micro_codebook.py)。

## fastwam-vq-plus

**问题。** 将旧 row-scale 与 Micro 的已冻结 banks 放到同一 280-variant 闭环面板，验证固定输入改善是否转成控制收益。两臂各 280，复用 1120 已有 reference slots；没有重拟合。4 卡各一个 persistent worker。

**结果。** 续跑`25771275.pbs101`完整结束：row-scale **202/280**、Microscaling **204/280**；旧 BF16 为 216/280。两 VQ 的后续 query mean 约 4.731/4.783 s，是当时 decoded-dense 实现，不能代表后来 layer-decode/Graph engine 速度。

**判断。** Micro 在这个面板净多 2 个成功，与固定输入正向结果一致，但不足以支持显著性或广泛优越性。它们都不是 native VQ GEMM；当时常驻 decoded weights 使 storage 压缩不自动转成显存收益。旧作业被取消后复用完成 episodes 的记录保留，不能把两个 attempt 当独立重复实验。

证据：[README.zh.md](../idea-validation/fastwam-vq-plus/README.zh.md)、[完整结果](../idea-validation/fastwam-vq-plus/results/RESULTS.zh.md)。

## fastwam-vq-micro-latent-search

**问题与设计。** 将 A+Z 叠加到共同未训练 Micro 起点，扫描 rho；lambda=rho×c，c=0.03033027132999191 只由初始 8calibration 确定。五个 R1 候选 rho0.1/0.5/0.7/1/1.5 各 16 updates；selection4 cases×2 seeds，late 22/gripper/P90 guards 为 parent 的 1.2 倍。按最低 first 10 的 1%near-tie 优先小 rho、早 epoch。

**最终范围。** 用户在第一轮结束后取消第二轮并停止。因此只完成 5 候选×3checkpoints 的 15selection points，不能沿用旧 README 称“两轮 optimal lambda 已完成”，也没有 rho0 matched action-only control。

**结果。** 协议选 rho0.1/epoch2，lambda0.00303302713，selection first 10 **0.0418016**，相对 parent0.047929 下降 12.78%。严格最小是 rho1.5/epoch1 的 0.0417929，但 near-tie 规则选小 rho；future 最小是 rho0.7/epoch2 的 0.0226788。三种“最好”对象不同，必须同时保留。

**当前用途。** 保留小样本 loss tradeoff 证据，不将 selection 数字当独立 test 或闭环。接手不自动恢复已取消 R2，也不把这里旧 Spatial C8 系数替换成四 suite C8 的 c。

证据：[ROUND1_SUMMARY.zh.md](../idea-validation/fastwam-vq-micro-latent-search/ROUND1_SUMMARY.zh.md)、[CONTRACT.md](../idea-validation/fastwam-vq-micro-latent-search/CONTRACT.md)。

## fastwam-vq-c8-action-latent-plus

**问题与设计。** 用四个 Original LIBERO suites 各 2 contexts 的 C8，训练 A+Z MXVQ，再完整评测冻结 280 面板。C8 是`2000,2019,2021,2025,2051,2057,2066,2070`，seed0；c=0.028303017887797623。rho0.1 训练 16 updates/2epochs，rho1.5 训练 8 updates/1epoch，二者预算不同。

**结果。** `25822971.pbs101`完成 560 新 episodes 与 1680references：rho0.1 **201/280**，rho1.5 **206/280**；旧 Micro204、SpinQuant203、BF16216。Source/action replay 差异 0，batched H/A4 通过 bitwise checks。

**解释与恢复。** 此结果支持保留 rho1.5 作为后续 incumbent 完整配置，不能说“只把 rho 调大”就多 5 个成功，因为 epochs/update 也变了，旧 Micro 也不是同 C8 未训练对照。训练后置 metadata 校验失败由 CPU 补验证，不重训，原 GPU 失败 receipt 保留。报告顶部和完整 results 优先于正文旧“待评测”状态。

证据：[README.zh.md](../idea-validation/fastwam-vq-c8-action-latent-plus/README.zh.md)、[结果](../idea-validation/fastwam-vq-c8-action-latent-plus/results/RESULTS.zh.md)、[验证修复](../idea-validation/fastwam-vq-c8-action-latent-plus/VALIDATION_REPAIR_25821658.zh.md)。

## fastwam-vq-future-reader

**问题。** A+Z 收益是否只因它自己的 student 下游与 future 共同适配。冻结同一 BF16 reader（video prepare/prefill、ActionDiT、proprio），仅替换 BF16/parent/action-only/A+Z 的 final future；随后完整做两个 student readers×两种 futures 的 2×2 交叉。

**覆盖与结果。** 原 60 contexts、test22 cases×2 seeds。固定 BF16 reader 的 first 10：parent **0.036144691**、action-only **0.041243219**、A+Z **0.035083670**；A+Z 对 action-only 有 13/22 cases 改善，进入交叉阶段。Action-only reader 读 A/A+Z 为 0.058110442/0.053481176；A+Z reader 为 0.058676505/0.052951009。60 次 BF16 future replay 和 120 次 student self-future replay 差异 0。

**含义。** 在相同下游里，更换 future 本身能改变 action，因此收益不完全依靠 student reader 共适配。但两个 student reader 的 case 异质性明显，不能指定语义区域或宣称 latent 质量完全决定 action。所有缓存/teacher-assisted 替换都是诊断，不是可部署需要 teacher 的 policy。

证据：[README.zh.md](../idea-validation/fastwam-vq-future-reader/README.zh.md)。

## fastwam-vq-future-nmse-matched

**问题与对照。** 将 action-only future 误差方向保留，只围绕 teacher 缩放其幅度，使实际 BF16 future NMSE 匹配 A+Z；用同一 BF16 reader 比较 A、匹配幅度 C、A+Z 的 B。Alpha 只按 future error 确定，不按 action 结果选；保留全部 60 contexts。

**结果。** Test future NMSE A/C/B 为 0.093011/0.0717863/0.0717869；first 10 motor **0.0412432/0.0360742/0.0350837**。C 已经重现大部分 A→B 平均收益。B−C mean 略好−0.00099049，但 median 为正，仅 11/22 cases 改善，leave-one-case-out 范围跨 0；full 32/late 22 也更差。

**决策。** 预设完整推进 gate 未通过，停止后续时间片/空间替换与搜索。不能将“保护特定有害 latent 方向”写成已证明机制；也不等于证明只有 NMSE 有用或 latent loss 无效。C 是 teacher-assisted counterfactual，不是可部署 arm。所有 context 的 NMSE 匹配误差均小于预设 0.01%，保留 BF16 离散化处理来源。

证据：[RESULTS.zh.md](../idea-validation/fastwam-vq-future-nmse-matched/RESULTS.zh.md)、[README.zh.md](../idea-validation/fastwam-vq-future-nmse-matched/README.zh.md)。

## fastwam-vq-bilevel-rho

**问题。** 能否通过独立 development 上的虚拟 Adam 有限差分，自动调 rho，而非反复看旧 held-out。Inner 原 C8；outer8 与 confirmation8 contexts 来自与 C80/held-out 不同 demonstrations；8rounds/16 updates，固定 rho0.1/1.5 对照从共同状态重训。

**结果。** 完整 pilot312endpoint records，rho_hat 约 2、lambda 约 0.05660604；相对 development 冻结 rho1.5，confirmation motor **+27.44%**，旧 held-out **−2.77%**，未过 gate。另做阶段 1 保留 adaptive 路径的 endpoint 诊断，相对 rho1.5 仍有 motor/latent 退化。

**阶段 2 最终状态。** rho1 与 2.25 从相同初始 master/Adam/RNG 训练 16 步，H1/H8/H16 完整 outer 比较；2.25 始终 motor 稍好于 1，未出现总体排序翻转。但 final16 相对 rho1.5，motor 分别+6.18%/+5.99%，future+51.07%/+37.82%，两候选均不合格，**不进入阶段 3**，保留 rho1.5。不能把曾经“阶段 2 已授权”的正文当作仍待自动提交的任务。

**接手意义。** 单步 warm-state 目标与从头固定 rho 训练不是同一优化问题；该路线未替代当前 rho1.5。确定性修复、state-restore 与 nonempty-Adam replay 是这里的重要工程资产。

证据：[README.zh.md](../idea-validation/fastwam-vq-bilevel-rho/README.zh.md)、[最终阶段2结果](../idea-validation/fastwam-vq-bilevel-rho/results/stage2_compare/RESULTS.zh.md)、[数值修复](../idea-validation/fastwam-vq-bilevel-rho/NUMERICS_REPAIR.zh.md)。

## fastwam-vq-codeword-conflict

**问题与设计。** 检验 shared codeword 是否形成异常梯度冲突、阻碍 latent 保护。扫描 614 Linears occupied words 后，仅按 C8 选 4modules×8book0 words，共 1,118,810 个 d4 vectors。相同 support 比较 shared codebook 更新、untied 独立 decoded-weight 更新、random，两个有限步长 0.005/0.015；不发明新的 assignment。

**结果。** 小步 shared/untied 的 test future NMSE **0.0832547/0.0730161**，fixed-reader motor **0.0391120/0.0333009**，有正信号。大步 untied 却显著恶化：future0.1218054、motor0.0590210，均差于 shared0.0777560/0.0359131。两个步长都需通过的 gate 为 false。

**解释。** Exact 成员梯度求和与 autograd 一致，但 observed cancellation 接近 random grouping，不足以证明异常 assignment 冲突。Untied 放开了容量，不能当成同 bits 可部署结果；只挑小步长宣称解除共享有效会违反原 gate。下一步先做模块范数匹配，已由下一条完成。

证据：[FINAL_REPORT.zh.md](../idea-validation/fastwam-vq-codeword-conflict/FINAL_REPORT.zh.md)、[EXACT_INTERPRETATION.zh.md](../idea-validation/fastwam-vq-codeword-conflict/EXACT_INTERPRETATION.zh.md)。

## fastwam-vq-module-norm

**问题。** 上一轮 untied 收益究竟来自方向自由度还是模块间更新能量分配。固定同 4modules/32words/support，用 shared 和 untied 原有预算分别构成 S-budget/U-budget，每模块对 shared/untied/random 匹配实际 decoded-BF16 更新范数；不新增 backward。

**结果。** 小步 S-budget shared/untied motor 为 0.0391120/0.0388714；U-budget 为 0.0408482/0.0347709，说明方向与模块预算存在交互。大步 0.015 的 untied 在两种预算下仍明显失败。小步 untied 对 U-budget random 的 motor 只多改善约 0.000581，不能脱离 random 实例和已见 test 过度解释。

**存储与决策。** 若将 untied 保存成 BF16 overlay，增量约 8.53645 MiB，按全部 target 权重为 0.01190246 bpw，但在被选 support 上约 16 bpw。相对现有零新增 bytes A+Z checkpoint 的边际收益不足以直接采用；不开发 conflict-aware assignment 或新 overlay 部署。

**工程注意。** BF16 离散台阶使一个原范数目标不可达，在任何新 query 前修订到最近可实现共同目标；修订记录与原失败保留。这类 norm matching 需以实际 decode 后值衡量。

证据：[FINAL_REPORT.zh.md](../idea-validation/fastwam-vq-module-norm/FINAL_REPORT.zh.md)、[README.zh.md](../idea-validation/fastwam-vq-module-norm/README.zh.md)。

## fastwam-vq-residual-capacity

**问题与设计。** 同一预算下，将第二本 residual book 的容量按未来/reader 损伤重新分配，是否优于 uniform、reconstruction 或 Fisher。冻结 action-only16update checkpoint，不训 books、不改已有 index 值；4096 个 d4 vectors/tile，0/50/100% residual slots；tags、padding、metadata 都计入 bytes。完整 17 arms×60 contexts=1020 记录。

**结果。** 统一 target 预算 2,272,020,342 bytes。Uniform 的 fixed-reader/native first 10/future 为 **0.068938/0.092265/0.181304**；action-Fisher **0.040995/0.056460/0.099004**；future+reader signed-damage 组合 **0.174579/0.189466/0.367432**，22/22 cases 均差于 uniform 的 action。Action 目标的 signed-damage 也差，不能只责怪 future 目标。

**结论与接手。** 否定的是这套有符号一阶评分与大幅容量移除的组合；不能据此否定 A+Z 训练。Fisher 是较好的本轮排序 proxy，不等于完成完整 GPTVQ 算法。此 layout 是容量核算、decoded-BF16 执行，尚无稀疏 packed 部署 kernel。随后做单 tile 与多 tile 实测，见下一条。

证据：[README与完整结果](../idea-validation/fastwam-vq-residual-capacity/README.zh.md)、[17arm报告](../idea-validation/fastwam-vq-residual-capacity/retrieved/results/analysis/REPORT.zh.md)。

## fastwam-vq-tile-intervention

**问题。** Residual 分配失效是单 tile 评分已经不准，还是多 tile 同时改动才产生交互。固定 4 个 video/action 早晚 FFN 模块、每模块 7 个 tiles，8calibration cases；28single+24multi 组合与首尾 baseline，共 432 配对记录，每条包含 native 与 fixed-reader 两个 query。

**结果。** 全部 864 模型查询完成，无新 training/backward。单 tile signed prediction 的 Spearman 弱甚至为负，例如 native motor/video late FFN 约−0.331；某 reader 高正分 tile 预测+9.40e-7，实际 MSE 变化−2.98e-4。多 tile 的真实 output 也不满足可加性：2tiles 的相对交互中位数约 0.629，有 43/144 方向翻转。

**判断。** 失败不能只归因于一次移除太多 tiles，亦不能只说遗漏平方 loss 曲率：真实线性 output 项与梯度预测已偏离。没有区分 STE 误差、网络有限步长非线性和 curvature 的唯一贡献。当前没有据此继续 allocation 搜索；其价值是约束 paper 机制表述和避免依赖不可靠 saliency。

证据：[README.zh.md](../idea-validation/fastwam-vq-tile-intervention/README.zh.md)、[primary_summary.json](../idea-validation/fastwam-vq-tile-intervention/retrieved/results/analysis/primary_summary.json)。
