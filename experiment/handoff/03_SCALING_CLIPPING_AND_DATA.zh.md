# Activation scaling、clipping、数据覆盖与控制诊断

返回 [总交接](../HANDOFF.zh.md)。本部分最重要的经验是：fixed-input motor、latent、gripper 与 closed-loop 可以给出不同排序。所有“通过/未通过”只指该实验事先声明的工程 gate；不表示统计显著性。

## fastwam-vq-calibration-suites

**问题与数据。** 将旧 Spatial-only 诊断扩展到 Original LIBERO 四 suite。C8 为每 suite 两个不同 tasks 的 contexts，嵌套在 C80；C80 覆盖 40 tasks×2distinct demonstrations/task×1snapshot/demo，共 80，四 suite 各 20，phase 每 suite7early/7middle/6late。统一 seed20261013，源 dataset revision 记录在 data receipt；不含 Plus 训练输入。

**方法与结果。** 对 C8/C80 分别完成 614 层 MXVQ 拟合，比较同 40held-out×2 seeds。四 suite macro first 10 motor：C8 **0.0464021562**、C80 **0.046636498（+0.505%）**；full 32 **0.0523299512→0.050788815（−2.945%）**，gripper 约不变。C80 有更广覆盖，但未在 primary first 10 稳定胜出。

**状态与接手。** Teacher`25787525`、C8`25789685`、C80`25791907`与 CPU`25793267`均已有完成证据，无待续实验。保留 C8 用于快速开发、C80 用于正式跨 suite 比较，不说 C8 已被证明足够/最优。C8 仅 8/40 tasks、一个 calibration draw，paired sampler seeds 不是独立 calibration 抽样。旧监控文档的 wake 失败不代表实验未完成。

证据：[README.md](../idea-validation/fastwam-vq-calibration-suites/README.md)。Plan、source receipt 与完整比较由该 README 定位；恢复时先找 plan 再找 teacher/banks。

## fastwam-vq-original-episode-c80

**角色。** 本地只见`protocol.json`的协议/配置保留目录。内容登记 Original demonstration state/XML 恢复、early/middle/late snapshot 规则、C80 与 held-out ID 空间，以及 calibration/clipping 源 roots。

**完成状态。** 没有该目录独立的 worker、controller、结果报告或完成 receipt，不能因为有完整 protocol 就宣称跑完了“Original episode C80”实验，也不能给它补一套成功率。实际已完成的数据准备、拟合和 clipping 结果分别归于其他目录。

**接手方式。** 将它作为历史合同参考，先比较后续实际实验的 protocol 再决定是否需要恢复；不能直接提交一个并不存在的完整执行入口。保留 snapshot 恢复而非偷偷改用 initial state0 的要求。

证据：[protocol.json](../idea-validation/fastwam-vq-original-episode-c80/protocol.json)。

## fastwam-vq-activation-clipping

**问题与设计。** 以四 suite C8 rho1.5/update8 checkpoint 为共同起点，比较 baseline、额外 8 步 weight_only、clip_only、joint。每 Linear 一个 ratio，token/denoising 共享，ratio 从 1 限制到[0.5,1]；weights LR5e-4、ratio LR1e-2，固定 final8。训练仍 A+Z；原 40held-out×2 seeds，共 80/arm。

**结果。** 相对原 baseline，held-out motor/latent：weight_only **+2.410%/+3.179%**，clip_only **+2.569%/+12.884%**，joint **+0.517%/+0.211%**。均未净胜原 checkpoint，joint 只是缓解继续 weight 训练的退化，预设 channel-scaling gate 为 false。

**恢复。** Teacher/baseline GPU`25848310`写完数值后因错误`runtime.close()`退出 1；CPU`25848931`只验证现有 80 records，不重跑模型。原 GPU 失败保留，不能将 CPU 验证伪装成 GPU 正常结束。后三臂及 CPU`25850887`完成，0 闭环。

证据：[README.zh.md](../idea-validation/fastwam-vq-activation-clipping/README.zh.md)、[结果](../idea-validation/fastwam-vq-activation-clipping/results/RESULTS.zh.md)。

## fastwam-vq-activation-ablation

**问题。** Joint 权重是否只在其共同训练的 clipping 下有效；训练集目标究竟有没有下降。固定 checkpoint，不重训，比较 joint weights+ratio1、baseline weights+joint ratios，并补四个完整 C8 端点。

**结果。** Held-out motor/latent 相对 baseline：joint W+ratio1 **+2.762%/+4.144%**，baseline W+joint ratio **+1.452%/−3.758%**。拆开也没有双指标净改进。C8 端点相反明显变好：clip_only motor/latent **−8.648%/−33.702%**，joint **−5.307%/−33.707%**。

**解释。** 小 C8 拟合与 held-out 分歧确实存在，不是“训练目标没降”。但不能从一次运行认定所有退化都由过拟合造成。两个 GPU 与 CPU`25852915`完成 192 新评估 records，无训练/新 teacher。它为更广 C80 覆盖提供动机，不能只按 C8 loss 选 ratio。

证据：[README.zh.md](../idea-validation/fastwam-vq-activation-ablation/README.zh.md)、[完整结果](../idea-validation/fastwam-vq-activation-ablation/results/RESULTS.zh.md)。

## fastwam-vq-c80-activation-clipping

**问题与设计。** 从同一 C8 rho1.5 起点，只训练 614ratios，扩到 C80/80 updates；books/indices/row gains/exponents/Smooth/H 固定。用户取消新 development，继续用原 40×2held-out。不能把已取消的数据准备算入本轮。

**结果。** 原 baseline motor/latent **0.044296774/0.060877371**；C80 clip-only **0.043848437/0.054848124**，分别−1.01%/−9.90%，23/40motor、30/40latent 改善。Full32/late 22/gripper 则分别+5.50%/+6.61%/+3.97%。两 primary 通过不等于全输出改善。

**状态与解释。** GPU`25853617/25854128`及 CPU`25854912`完成；最终 deployment 主要保存 ratios 与冻结 source 引用。相对旧 C8 同时改变 coverage 和 updates，不能单因归为“增加数据量有效”。这个 clip-only80 后来有 207/280 闭环，是长期复用 comparator。

证据：[README.md](../idea-validation/fastwam-vq-c80-activation-clipping/README.md)。

## fastwam-vq-c80-clip-joint-plus

**问题。** 更低 fixed-input error 的 joint 能否提高闭环；clip-only 与 joint 均从同一 C8 checkpoint，C80、80 updates，但 joint 同时更新 books/log gains 和 ratios。

**结果。** 两臂各 280，共 560 新 episodes、2240references。Clip-only **207/280**，joint **202/280**；joint 对 clip 为 9 胜/14 负/257 平，camera 维度净少 5 成功。C8 初始化为 206/280。不能按 joint 更低 RMSE 宣布更好的 policy。

**工程记录。** Clip-only GPU`25858029`四 workers 完成后汇总失败，CPU`25860507`验证恢复完整数据；joint`25858606`正常完成。训练 final80 源`25858027`。当时 batched-H128/decoded-BF16 full query 约 4.99/4.96 s、allocated peak 约 28.45 GB，和后来 packed engine 不是同一内存路径。

**接手用途。** 保留两 checkpoint，供状态覆盖、gripper 和机制诊断；没有删除 clip-only comparator。后续所有新路线应清楚是否从 C8 起点还是从已训 clip-only 继续。

证据：[RESULTS.zh.md](../idea-validation/fastwam-vq-c80-clip-joint-plus/RESULTS.zh.md)、[README.zh.md](../idea-validation/fastwam-vq-c80-clip-joint-plus/README.zh.md)。

## fastwam-vq-c80-joint-rmse

**角色。** 对上条 joint final80 做独立 fixed-input evaluation，不重训，不新增 teacher。源绑定到实际完成的`25858027`，新 160 端点，复用 baseline/clip320，共 480。

**结果。** GPU`25858590`、CPU`25859109`完成。Held-out joint motor **0.0418468372**、latent **0.0489733362**；比 baseline−5.53%/−19.55%，比 clip-only−4.56%/−10.71%。源 reference action parity0。后续只读追加闭环 202 对 207 及 camera 损失，没有重跑 episode。

**意义。** 这是 proxy-to-closed-loop mismatch 的干净例子：同协议固定输入明显更准，成功数反而少。源 final80 是固定选择，不能看完 held-out 改选 checkpoint。追查部署 loader 时要保持 clipping forward 分支，即使所有 grad 已冻结。

证据：[README.md](../idea-validation/fastwam-vq-c80-joint-rmse/README.md)、[后续分析](../idea-validation/fastwam-vq-c80-joint-rmse/results/ANALYSIS.zh.md)。

## fastwam-vq-b32-scaling-clipping

**问题与实现。** 在 activation Z 上每输入 B32 group 学习 D∈{0.5,1,2}，配对 weight compensation；D 从 1、ratio 从 1，冻结 VQ，C80/80 updates。整数 k 能折入原 weight B32 exponents；前向必须实际取整，不能用 shadow 连续值冒充部署。

**结果。** GPU`25855238`、CPU`25856260`完成 160 新端点、320reference；相对 clip-only80 motor **+0.3154%**、latent **−1.2700%**，双指标 gate 失败。只 13/56466 groups 变化，涉及 3/614 Linears。

**解释。** 离散自由度几乎没动值得检查，不等于连续 scaling 也无效。其 backward 将 dynamic absmax 视为常数，是 LSQ/STE surrogate。后续先检验连续 B32，再检验耦合，不直接把失败归因于“B32 粒度太粗”。

证据：[README.md](../idea-validation/fastwam-vq-b32-scaling-clipping/README.md)。

## fastwam-vq-continuous-b32-scaling-clipping

**问题与实现。** D=2^log2D，log2D∈[−1,1]，每 32 channels 共享；从原 C8 起点训 80 步，不从离散 arm 继续。真实 forward 始终`Linear(Q_A4(Z/D;ratio), BF16(Wq*D))`，连续 D 额外保存 FP32 metadata，不能说精确折进整数 exponents。

**结果。** GPU`25856825`、CPU`25857520`完成。Held-out motor **0.044588993**、latent **0.060690613**；相对 clip-only+1.6889%/+10.6521%，四 suite 均退化，gate 失败。

**接手意义。** 比较的是未补齐 absmax 链式项的连续版本，不应与 coupled 版本混用。BF16 weight compensation 自身会 round，评估已核对 uncached/cached 完整输出 parity；它不是精确实数等价，不应忽略 rounding 账本。

证据：[README.md](../idea-validation/fastwam-vq-continuous-b32-scaling-clipping/README.md)。

## fastwam-vq-b32-absmax-coupling

**唯一干预。** 补齐 dynamic absmax 对 input/D 的链式项；相同连续 B32 范围、80 updates、loss、LR 与共同起点。Max ties 按 amax 等分、零输入有旁路。Ratio 仍用既有 surrogate，不能把它称为硬 round/BF16 函数的精确梯度。

**结果。** GPU`25861303`、CPU`25861881`完成；held-out motor **0.043394818**、latent **0.054067248**，比 clip-only−1.0345%/−1.4237%，比未 coupled 版本−2.6782%/−10.9133%，完整主 gate 通过，suite 仍有取舍。

**意义。** 梯度实现本身会影响结论，前一版连续 scaling 失败不能直接归因于表示。建议的 D=1 coupled clipping 归因对照只是建议，不自动声称完成；后续实际 per-channel 路线单独记录。

证据：[README.md](../idea-validation/fastwam-vq-b32-absmax-coupling/README.md)。

## fastwam-vq-per-channel-scaling-clipping

**问题。** 将 coupled B32 的 D 细化到每物理 input channel，共 1,806,863 个 FP32 log2D；ratio 仍每 Linear 一个，不是逐 channel clipping。共同 C8 起点、C80/80 updates，VQ 冻结。完整 coupled input/D surrogate 和实际 BF16 compensation 保持。

**结果。** GPU`25862598`、CPU`25863316`完成。Held-out motor **0.031641466**、latent **0.055377875**；相对 coupled B32 **−27.0847%/+2.4241%**，motor 有 39/40 cases 改善，但双 primary gate 仍失败。Latent 问题主要在 Goal/Libero10。

**解释。** 有明确 motor 收益和 latent tradeoff，不可写成全面提升。之后虽有独立闭环 203/280，但这里 fixed-input 结果本身不包含闭环。D 需独立 metadata/weight compensation，不属于原始约 4.27 bpw 固定表示的免费修改。

证据：[README.md](../idea-validation/fastwam-vq-per-channel-scaling-clipping/README.md)、[分析](../idea-validation/fastwam-vq-per-channel-scaling-clipping/results/ANALYSIS.zh.md)。

## fastwam-vq-joint-per-channel

**问题。** 在 per-channel D+ratio 上同时放开 books/log row gains，能否减少 latent tradeoff。共同 C8 起点、80 updates，indices/exponents/Smooth/H 固定；weights/ratio/D LR 分别 5e-4/.01/.02。新增 weight adjoint 要从实际 BF16 compensated Linear 回到 Wq，避免重复算 D 补偿梯度。

**结果。** GPU`25864060`、CPU`25865058`完成。Motor/latent **0.032668161/0.048842217**；相对 per-channel motor+3.245%、latent−11.802%；相对旧 joint motor−21.934%、latent 仅−0.268%。Gripper0.080485551，比旧 joint+81.89%。Calibration 两项也比 per-channel 更差，不是典型训练集更好而 held-out 更差。

**决策。** 当前是联合优化取舍，无新闭环/无需按 held-out 再选 best。不能把相对旧 joint 的变化全部归因于 D，因为 absmax coupling 实现也不同。

证据：[README.zh.md](../idea-validation/fastwam-vq-joint-per-channel/README.zh.md)。

## fastwam-vq-c80-per-channel-plus

**问题。** 对已固定 per_channel_clip final80 单独闭环，不重训。新增 280 episodes，复用 2800reference slots；source scalar metrics、uncached/cached action/future 与实际 infer 均检查。源没保存 final action vector，不能称为直接 saved-action replay。

**结果。** Per-channel **203/280**，clip-only207、C8 rho1.5 为 206、BF16216。Suite 为 Libero10 47/70、Goal46、Object56、Spatial54；motor fixed-input 大幅改善没有转换成总体成功率收益。

**接手用途。** 这是分析 gripper/visited states 的动机，但不能事后将 4 个净损失全部归因于某个控制事件。保留实际 source、280 面板和对应执行路径，不能用另一 checkpoint 的 engine parity 给它补精度证明。

证据：[RESULTS.zh.md](../idea-validation/fastwam-vq-c80-per-channel-plus/RESULTS.zh.md)、[闭环分析](../idea-validation/fastwam-vq-c80-per-channel-plus/CLOSED_LOOP_ANALYSIS.zh.md)。

## fastwam-vq-rollout-c80

**问题与设计。** 原 C80 的 40 偶数 cases 保留 anchor，40 奇数替换为固定 C8 student 从 demo snapshot 执行最多 10actions 后的 visited states；四 suite 各 20。保持 80 contexts/80 updates 和原 jointrecipe，采集/teacher 成本另算；不按失败或 held-out 选择 states。

**结果。** 新 joint motor/latent **0.042925590/0.053785088**，均差于原 joint **0.041846837/0.048973336**；Object 改善，Spatial/Goal 退化。Gripper 相对原 joint−20.715%，只是 secondary。40 新 visited teacher 与 40anchor 旧 teacher 绑定 plan 和 seed。

**结论。** 局部短程 student 覆盖没有在原分布 fixed-input 上胜出；不能据此证明它对长期闭环无用，也不能声称已解决闭环退化。本轮没有新 closed-loop。所采 40states 被后续 visited 诊断复用，不自动回灌其他训练。

证据：[README.zh.md](../idea-validation/fastwam-vq-rollout-c80/README.zh.md)、[结果](../idea-validation/fastwam-vq-rollout-c80/results/RESULTS.zh.md)。

## fastwam-vq-visited-diagnostic

**问题与输入。** 在 40 个既有 visited states 上，以同输入/seed 比较 BF16、clip-only、joint，避免把轨迹分叉当成局部模型响应。每 state 有 base、raw 双相机 brightness±1、raw EEF 对角方向±5mm，再走原 preprocessing；共 600 端点，566 新 queries，0 训练。

**结果。** GPU`25863989`与 CPU`25864600`完成，全部 reference/repeat parity0。Base normalized motor clip/joint **0.045833804/0.044575345**；gripper 实际开合错误 **6/400→3/400**。但 joint proprio 方向响应误差变差，且 case2045 motor 更准却漏掉前 10 步释放事件。

**意义。** 小平均 motor error 不能替代 gripper 时机或局部 response。两条输入方向不是完整 Jacobian，也不代表真实 camera viewpoint 扰动；这些 states 不是此前 camera 失败轨迹，不能作跨实验因果归因。Switch mismatch 需结合事件方向，漏 open 不能被误算成更小延迟。

证据：[README.zh.md](../idea-validation/fastwam-vq-visited-diagnostic/README.zh.md)、[ANALYSIS.zh.md](../idea-validation/fastwam-vq-visited-diagnostic/results/ANALYSIS.zh.md)。

## fastwam-vq-per-channel-visited-diagnostic

**设计。** 在完全相同 40states×5inputs 上只增加 per-channel200 端点，复用 600 旧端点。共 205 新 queries，1 个 fresh BF16 仅用于 parity。旧完整 image tensors 没持久化，核对 inputmetadata 与 fresh BF16 replay，不宣称全部逐像素重验。

**结果。** GPU`25868350`、CPU`25868746`完成。Base normalized motor **0.034658006**，比 clip/joint 都好；actualcommand motor0.027482118。Gripper 连续 RMSE0.115395849、开合错误 6/400，漏 open2 次；与 motor 改善不同步。相对 clip normalized motor37/40 cases 改善。

**决策。** 这支持单独补 first 10gripper 监督的有界试验，不支持同时添加 response/bias loss，也不解释全部闭环净损失。后续 gripper 实验已完成且没有通过，见下一条。

证据：[README.zh.md](../idea-validation/fastwam-vq-per-channel-visited-diagnostic/README.zh.md)、[ANALYSIS.zh.md](../idea-validation/fastwam-vq-per-channel-visited-diagnostic/results/ANALYSIS.zh.md)。

## fastwam-vq-per-channel-gripper

**唯一干预。** 保持 per-channel 原 C80/80 updates、motor/latent 系数与优化器，从共同 C8 起点加 first 10 第 7 维 MSE，lambda_g=1/6。只训练 D/ratio，VQ 冻结；不是对已训 per-channel 继续微调。原 held-out80 和 visited200 端点均完整评估，新增 query 预算 445。

**结果与状态。** GPU`25869007`、CPU`25869987`完整结束。Visited base 开合错误从 **6→9/400**，漏 open 仍 2 次，并有 motor/latent 代价。当前 checkpoint 不支持进入闭环；未向暂停中的闭环任务发恢复消息。

**接手判断。** 增加连续 gripper 监督不等于解决 sign/event 时机。若再设计事件 loss 需另立协议；不能偷偷修改当前 motor6 mean 系数，或让 late 22 也获得未声明监督。现有 visited states 不回灌本轮训练。

证据：[README.zh.md](../idea-validation/fastwam-vq-per-channel-gripper/README.zh.md)、[完整分析](../idea-validation/fastwam-vq-per-channel-gripper/results/ANALYSIS.zh.md)。
