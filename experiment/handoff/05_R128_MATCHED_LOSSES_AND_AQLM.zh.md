# R128、联合优化、当前 matched loss 与 AQLM

返回 [总交接](../HANDOFF.zh.md)。本部分串起当前主模型的来源：W16A4 学 R0 → 同 R0 拟合 VQ0 → 冻结 R0、matched A+Z/reader-KL → 闭环 → inference-only 优化。`R128`在这里是正交变换的 block 宽度，不是 SVD rank128，也不是 VQ 向量长度。

## fastwam-vq-learned-r128-c8

**问题与流程。** 固定 Smooth，比较 H+VQ8 与先 W16A4 学 R8、收集 moments/refit VQ、再冻结 R 训 VQ8。C8、614 targets、普通 A4，loss 仍 A+Z；candidate 多出 R 学习与 refitting，不能称 equal-compute。

**结果。** GPU`25902163`完成。H/R held-out motor **0.044296774/0.046408990**，latent **0.060877371/0.065187050**，mixed **0.004598652/0.004980800**；分别+4.77%/+7.08%/+8.31%，保留 H+VQ8。CPU 核对 R 实际变化，U−I 约 0.015–0.016，正交误差约 1.91e-6，不是 R 未接入。

**下一步含义。** 不能将失败一概归于 VQ refitting，应先看无 VQ 的 W16A4 R/H 诊断；该诊断后来完成并发现进入 VQ 前已退化。成功任务 walltime44m49 s，不是 native-kernel latency。

证据：[README.zh.md](../idea-validation/fastwam-vq-learned-r128-c8/README.zh.md)、[结果](../idea-validation/fastwam-vq-learned-r128-c8/results/25902163.pbs101/RESULTS.zh.md)。

## fastwam-vq-r128-joint-c8

**问题。** 比较 sequential R→VQ 与同起点 Joint8，并补 H/R W16A4 诊断及 C8 阶段边界。Joint 从 H+VQ0、U=I 开始，R LR1e-4、VQ LR5e-4，8 updates；仅 VQ 在 update1 后 reset，R moments 保留。共同训练 R 和 books/gains 时，compressed weights 不自动配对补偿 R。

**结果归属。** 完整总结在后续 joint-refine 目录：Joint8 held-out motor **0.045524040**、latent **0.060773535**、mixed **0.004718190**；比 sequential 有改善，仍差于原 H+VQ8 的 motor/mixed。H/R W16A4 比较显示 R8 在 held-out 先已退化。

**实现边界。** 本 Joint 是 compressed-student 共适配，训练中不保证固定 VQ 时 R-only 保持原浮点函数；不能凭正交性说这就是等价 rotation 优化。没有 H+VQ16 对照、没有闭环。后续 lower-LR conditional arm 另记下一条。

证据：[README.zh.md](../idea-validation/fastwam-vq-r128-joint-c8/README.zh.md)、[后续总表](../idea-validation/fastwam-vq-r128-joint-refine-c8/README.zh.md)。

## fastwam-vq-r128-joint-refine-c8

**问题与条件执行。** 基于 C8 而非 held-out 触发一档 R LR1e-5，并增加同实现 H_currentpath（U=I 冻结）控制，核对实现路径差异。

**结果。** W16A4 H/R held-out motor **0.042604556/0.044585121**、latent **0.038380617/0.040343879**，说明 R 在压缩前已泛化退化。Joint R1e-5 motor/latent/mixed **0.045226322/0.065914399/0.004893894**，相对 H_currentpath mixed 只−0.003%，没有可靠额外收益；full 32 更差。原 H+VQ8 仍保留。

**工程发现。** LowerLR 让同 case update 后 mixed 下降从 3/8 到 7/8，却未改善 held-out；下降次数不是收敛/泛化证明。H_currentpath 也未精确重现旧 H，CUDA `index_add_`非确定性只是候选解释，未确认为全部差距根因。两个新增 GPU`25903221/25904715`完成，无新 teacher。

证据：[README.zh.md](../idea-validation/fastwam-vq-r128-joint-refine-c8/README.zh.md)、[完整结果](../idea-validation/fastwam-vq-r128-joint-refine-c8/results/25904715.pbs101/RESULTS.zh.md)。

## fastwam-r128-improvements

**三个问题。** 无 VQ 的 W16A4 下，分别检验最终 R 幅度收缩 t=0/.25/.5/1、C8 顺序×LR 的 2×2，以及 C80 coverage recipe。保持 Smooth/signs、D1、ratio1、A4、BF16 KV，仅两个 stream-shared Cayley U128 可训。

**C80 来源。** 80distinct contexts 分 8 个 batch10、平均梯度 8 updates，R LR1e-4，固定 final8；不是 C80×80 updates。C80 对 H 的 held-out motor/latent/mixed 为 **−0.38%/−10.50%/−5.36%**，但 gripper **+25.08%**，37/40 cases 变差。C8 新 arms 均未双指标全面超过 H。

**当前用途。** `25905203.pbs101 / coverage_c80 / C80_once_lr1e4`生成当前主模型的 R0。采用它作为固定来源不等于宣称 R0 本身带来显著闭环收益；coverage、LR、平均梯度方式共同改变。Path/optimization/coverage 均完成，零新 teacher；相同首步梯度数值差需保留其描述性边界。

证据：[README.zh.md](../idea-validation/fastwam-r128-improvements/README.zh.md)、[最终分析](../idea-validation/fastwam-r128-improvements/results/final/ANALYSIS.zh.md)。

## fastwam-r128-w16a4-plus

**问题。** 单独量化 activation、不做 weight VQ 时，C80 R0 能否改善闭环。两臂 H-W16A4、R0-W16A4 各 280；原 BF16 weights 只配对 transform 并一次缓存，D1、ratio1、无新增训练。

**结果。** `25910102.pbs101`完成 560 新 episodes、3640references。H **204/280**，R0 **203/280**，16 胜/17 负/247 平，净−1。Object/Spatial 有所改善，Goal/Libero10 退化；当前 cohort 没有总体成功率优势。

**意义。** 把 activation-only 损失与 weight 压缩代价分开。后面 R0-MXVQ+readerKL 的 208 高于 R0-W16A4 的 203，是完整配置结果，不能推导“量化权重天然比 BF16 权重更好”。各 source scalar replay 与 cache/infer parity0，但旧 source 未保存完整 tensor，不称直接原 source tensor replay。

证据：[RESULTS.zh.md](../idea-validation/fastwam-r128-w16a4-plus/RESULTS.zh.md)、[README.zh.md](../idea-validation/fastwam-r128-w16a4-plus/README.zh.md)。

## fastwam-vq-r128-c80-dual

**问题与共同起点。** 复用上条 R0，R0 坐标下只拟合一次 VQ0，再比较冻结 R 的 Seq8 与 warm-start Joint8。相同 8batch×10 contexts、80 exposures、VQ LR5e-4；Joint 额外 R LR1e-5。614 编码、indices/exponents 固定；用独立 Runtime 与严格 replay/determinism gate。

**结果。** `25912619.pbs101`完整完成。共同 R0+VQ0 held-out motor/latent **0.048127906/0.059216201**；Seq8 **0.044207680/0.051400843**；Joint8 **0.045749947/0.059470054**。Joint calibration 略好而 held-out 更差，Seq8 作为后续方法候选。

**关键实现问题。** 实际 Joint 的 R 只有 activation 梯度路径，weight 没有随 R 配对补偿；固定 VQ 下更新 R 仍改变底层函数，不是单纯换正交坐标。需要 transport 诊断，不能把所有回退归为普通过拟合。Gripper 平均与 case 胜负可能反向，需完整报告。

**传承资产。** 这里的**共同未微调 banks/VQ0**和 source/repro receipts 被后续 matched loss-pair 复用，而不是把这里的 Seq8 或 Joint8 当新 loss 起点。两次准备失败与最终成功分开记录；成功 allocation 约 3h41m47 s。

证据：[README.zh.md](../idea-validation/fastwam-vq-r128-c80-dual/README.zh.md)、[JOINT8_DIAGNOSIS.zh.md](../idea-validation/fastwam-vq-r128-c80-dual/JOINT8_DIAGNOSIS.zh.md)、[结果](../idea-validation/fastwam-vq-r128-c80-dual/results/25912619.pbs101/RESULTS.zh.md)。

## fastwam-vq-r128-c80-transport

**两个问题。** 固定某个 Seq/Joint 端点函数，换 R 并运输 decoded weights，评估 A4/noA4；再从共同 R0/VQ0 训练带可微 weight 补偿 Comp8，与原 Joint8 比较。

**Probe 结果。** Seq 函数由 R0 补偿换 RJ，held-out mixed **0.004175274→0.004404567（+5.49%）**；Joint 函数由 RJ 换 R0，mixed **0.004682388→0.004460030（−4.75%）**。Calibration 的排序反向；不能说“RJ 方向本来好，只是 weight 没跟上”。NoA4 残差 gate 是聚合主指标条件，不等于完整 action 逐位等价，存在较大 gripper/最大坐标差异。

**Comp8 结果与范围修改。** 用户停止 Uncomp8 同期重跑，复用原 Joint8 作为 reference；Comp8 保留已完成 3 updates 后单卡继续到 8，`25921642`完成。相对原 Joint8 mixed−5.251%，收回 gap **48.488%**，低于冻结 50%effect gate，仍比 Seq8 差 6.256%。不能四舍五入成通过，也不能宣称具备原计划同期完整 replay 的归因强度。

**诊断约束。** Seq/Comp 轨迹已分叉，跨轨迹 final 差不是纯 R 效应。每步更新里 R/VQ 贡献应靠同端点 2×2swap 或有界 counterfactual；“7/8loss 下降”不等于更稳泛化。Decoded transport 也不解决旋转后的 weight 是否还能原样表达为 d4/B32 VQ 的问题。

证据：[README.zh.md](../idea-validation/fastwam-vq-r128-c80-transport/README.zh.md)、[ANALYSIS.zh.md](../idea-validation/fastwam-vq-r128-c80-transport/ANALYSIS.zh.md)、[逐更新诊断](../idea-validation/fastwam-vq-r128-c80-transport/COMP8_UPDATE_DIAGNOSIS.zh.md)。

## fastwam-vq-r128-c80-loss-pair

**当前最干净的 loss 设计。** 固定 R0、同一原始 VQ0，从共同初始化分别训练 A+Z 和 A+Z+frozen-reader-KL。C80 八批十 contexts、8Adam updates、80 exposures、LR5e-4、VQ 首步后 reset，indices/exponents/全部 transforms 固定；只改变 reader-KL。两个 arms 同一个双 GPU allocation 并行，各一张卡。

**Loss 与 beta。** A=first 10×6motor MSE；Z=teacher clean future NMSE 排除 frame0；lambda **0.04245452683169643**。Reader 用原始 BF16 future→prepare/prefill→ActionDiT，teacher parameters 冻结但 student future 可导，16-binsoft-KL。完整初始 C80 一次标定 **beta=0.006083668576461056**，不从旧 KL checkpoint warmstart，也不用 held-out 选 beta/final 步数。

**实际结果。** `25921795.pbs101`summary 完整：两臂 8 updates、614 targets、frozen_rotation、initial/formal parity0、finalanchor，new teacher0。AZ 记录 82studentqueries；KL242student 与 162readerqueries，体现额外 reader 成本，不把零新 teacher targets 当作零 BF16 reader 计算。本轮**没有新 held-out fixed-input 评估**，不能凭 calibration 目录补造一张 held-out 结果表；下游闭环另记。

**代码入口。** `worker.py`负责共同来源绑定、VQ 装载、staged reader cotangent、batch 梯度均值、checkpoint；`finalize.py`聚合；`control.py`管理唯一 handle、intent 与 verify。运行时依赖旧 artifact 源码 snapshots，不能只复制这个目录假设自足。

证据：[README.zh.md](../idea-validation/fastwam-vq-r128-c80-loss-pair/README.zh.md)、[protocol.json](../idea-validation/fastwam-vq-r128-c80-loss-pair/protocol.json)、[worker.py](../idea-validation/fastwam-vq-r128-c80-loss-pair/worker.py)、[成功summary](../idea-validation/fastwam-vq-r128-c80-loss-pair/pulled/25921795.pbs101/summary.json)。

## fastwam-vq-r128-c80-loss-pair-plus

**问题。** 只增加 reader-KL，在严格 matched 起点/预算后是否改进控制。Frozen280 面板，两臂各 280，复用 4200reference slots；不重训，不新增 baselineforward，不追加 held-out RMSE。

**结果。** `25925636.pbs101` F/0，四 workers 各 140；AZ **204/280（72.86%）**，AZ+readerKL **208/280（74.29%）**；14 胜/10 负/256 平，净+4（+1.43pp）。BF16 **216/280**，仍高 8 成功。KL camera14→19，Goal44→47，但 robot initial33→31，非所有扰动受益。

**当前定位。** **reader_kl final8 是当前 Fast-WAM 算法 Q0/reference**，后续 runtime 与 Q-Clip 基于它。它是单一冻结 cohort 的点估计，不能称统计显著、SOTA 或跨训练 seed 稳定。首次启动因容器缺 CUBLAS_WORKSPACE_CONFIG 失败，0 episodes；修复入口后复用原 checkpoint，不能把修复算新训练。

证据：[RESULTS.zh.md](../idea-validation/fastwam-vq-r128-c80-loss-pair-plus/RESULTS.zh.md)、[summary.json](../idea-validation/fastwam-vq-r128-c80-loss-pair-plus/summary.json)。

## fastwam-vq-r128-c80-qclip

**问题。** 冻结上条 Q0 全部 weights/VQ/R，仅追加 614activation ratios，C80 一遍 80 单 contextupdates，LR.01、ratio[.5,1]，原 A+Z+readerKL 系数不变。固定 final80，无 held-out 选模。与旧 C80 clip-only 不是同 checkpoint。

**结果。** `25937633`完成。同一 Q0/Q-Clip 的 held-out motor **0.045606257→0.043983056（−3.56%）**，future **0.061737563→0.051682642（−16.29%）**，full 32/late 22/gripper 均值也降。Calibration motor 却+0.70%、gripper+38.85%；要保留这些反向指标。Full endpoint reader-KL 未追加，不能把逐 case 更新前日志冒充最终共同 KL 表。

**当前意义。** Fixed-input 有净收益，但部署是否采用应看下一条闭环。FrozenR/VQ 与 r1source parity 均 0；zero new teacher，仍有 80readerqueries。此研究按指定只做该 arm，没有自动扩展 D 或重新训 books。

证据：[RESULTS.zh.md](../idea-validation/fastwam-vq-r128-c80-qclip/RESULTS.zh.md)、[deployment与断点来源](../idea-validation/fastwam-vq-r128-c80-qclip/protocol.json)。

## fastwam-vq-r128-c80-qclip-plus

**问题与结果。** 将 Q-Clipfinal80 直接评 280，复用 17 arms/4760 slots。`25939858`完成，Q-Clip **207/280**，父 Q0 **208/280**；11 胜/12 负/257 平，无总体闭环提升，故 Q0 继续作当前 reference。旧 C80 clip-only 恰好也 207，不代表同 outcomes 或同 method。

**执行与边界。** 四 workers 各 70、全部 source/anchor/inferparity0，无新 training。旧 decoded-dense 完整 querymean 约 6.969 s、allocatedpeak28.478 GB，不是当前 engine 结果。LIBERO10 净少 2、Goal 多 1，不能以部分 case 改善替换整体主指标。

证据：[RESULTS.zh.md](../idea-validation/fastwam-vq-r128-c80-qclip-plus/RESULTS.zh.md)、[FINAL_COMPLETION_EVIDENCE.json](../idea-validation/fastwam-vq-r128-c80-qclip-plus/FINAL_COMPLETION_EVIDENCE.json)。

## fastwam-aqlm-loss-pair-20261026

**问题与对照。** 将同一任务末端 loss 转移到另一种 quantizer，检验收益能否超出自家 MXVQ 表示。共同初始化使用 pinned official `Vahe1994/AQLM@e79a896ed6656fe4ed06193d42d004e7d0bbdbb2` 的 AQEngine：input-aware XTX、residual K-means、continuous Adam 与 discrete beam-search；g8、4 个 K256 additive codebooks、BF16 scales，共同冻结 Smooth0.5/H128/R0。Nominal indices4 bpw，weight encoding **4.018892 bpw**，含共同 preprocessing **4.031080 bpw**。这是官方 AQLM 的 Fast-WAM adaptation，不是原 LLM 全流程，也不是官方 block fine-tuning。

**训练与评估合同。** C80 全 contexts、seed0、全部 sampler Linear calls；每 call 确定性均匀抽最多 16 token rows，完整 C80 XTX 不缩减。两臂只训练 codebooks/scales，冻结 indices、preprocessing、底模；均 batch10、8 Adam updates/80 exposures、LR5e-4、clip1、首步后 reset moments、fixed final8，不按 held-out 选模。`reconstruction` 在同一 A4 operand 比较原 transformed BF16 weight 与 decoded AQLM weight 的 Linear 输出 MSE，先 calls→module→614 modules→batch 平均；`ours` 用 A+lambda Z+beta reader-KL，不额外叠 reconstruction。Lambda **0.04245452683169643**，beta **0.005507108948679781** 由初始完整 C80 冻结。每臂 final calibration80 与 held-out40×2 均完整，聚合 seeds→context→suite→四 suite 等权；共同 `mixed_loss=A+lambda Z` 不含 KL。

**共享拟合实际预算。** Prepare `26005386.pbs101` 已完成全部 614 banks，最终为 **22 heavy_v1、12 light_v2、580 minimal_v3**。Minimal K-means≤5 iterations、max_points_per_centroid16、单次残差 codebook 聚类最多 4096 vectors；所有 weight groups 仍 assignment；≤1 epoch×5 Adam updates、beam1、relative tolerance.005。已完成旧预算 banks 原样复用，两臂共享同一混合初始化，不能把 614 层描述成统一 minimal，也不能仅以混合来源解释两臂分化。Context archives/批量 GPU→CPU copy/compute host cache 只改 I/O；官方 CUDA index_reduce 的 deterministic override 仅在共同 quantize 范围，finally 恢复，两臂训练仍 deterministic。

**完成与恢复来源。** Pair `26008761.pbs101` 的 summary/两臂 complete 均为 true：8 updates、80 exposures、80 calibration+80 held-out、final anchors 已保存，initial source parity0。原 `26006936` 在首个更新前 reconstruction OOM；`26007418` 提交 reconstruction update1 后 reader backward OOM；`26008308` 在 reference cache 与 Adam/decode refresh 峰值重叠处失败。最终修正按层搬原 BF16 reference、reader 阶段将完整 padded decoded cache park CPU、batch 梯度完成后释放 reference 再 Adam/refresh；恢复已提交的 reconstruction 第一步，不多做一次。失败 attempts 原样保留，不能改写为所有训练都在一个无失败 job 内完成。

**Fixed-input 最终结果。** 这是负结果。Held-out 全 40 contexts，以下各项 ours 均 **0 胜/40 负/0 平**：

| 指标 | Reconstruction final8 | Ours final8 | Ours 相对变化 |
|---|---:|---:|---:|
| first10 motor RMSE | 0.051755376 | 0.116873852 | +125.82% |
| full32 motor RMSE | 0.054076112 | 0.115703907 | +113.96% |
| late22 motor RMSE | 0.054653070 | 0.112910300 | +106.59% |
| first10 gripper RMSE | 0.035122972 | 0.127694358 | +263.56% |
| future NMSE | 0.069989021 | 0.239592169 | +242.33% |
| mixed=A+lambda Z | 0.005762744 | 0.025813018 | +347.93% |

**校准拟合也失败。** 既有初始化缓存→reconstruction final8→ours final8，C80 motor RMSE **0.048906→0.048310→0.118843**；future NMSE **0.057176→0.061066→0.250011**；mixed **0.004960→0.005010→0.028407**。Ours 相对初始化 80/80 mixed 变差，motor/future 各 79/80 变差。这不是只在测试集或闭环才出现的退化。

**训练/参数诊断。** CPU `26020010` 使用原初始化预测匹配各更新前实际 cases，第一次更新后，下一 batch 的含 KL 总 loss 已为其初始值 **6.030×**、motor MSE **8.335×**。八批输入不同，不能把原八个 loss 直接画成固定 batch 收敛曲线，也不能据此断言第一步在原训练 batch 上逆真实梯度更新。最终 books 相对 L2 漂移 reconstruction **2.057%**、ours **8.903%**（4.33×）；每层中位 **2.795%/13.279%**、最大 **8.455%/37.139%**。Ours 更新 RMS .001530 远大于 BF16 舍入 RMS .00002824，单靠最终存储舍入难解释退化。部署 row scales 无符号翻转、变化很小，不优先归因 scale 爆炸。

**已排查与未知。** 两臂 614 模块梯度 finite/connected、staged action/future replay0、四闭环 workers 对应 source anchor0；最终 1228 个 Adam 参数状态 step7 符合首步 reset 后七步。源码检查未见明显 loss 符号、batch 重复除法或 cotangent 重复添加错误，但这不证明完整参数梯度的真实数值方向。优先候选是 codebook 绝对 LR 与参数尺度/不同目标更新步幅不匹配；各 loss 分量的独立梯度及固定 batch 小步下降尚未验证。不能将其写成“已定位 KL bug”或“loss 不可迁移”。

**当前决策。** Fixed-input 阶段已完成；它本身 closed-loop episodes=0，后续环境结果见下一条。整体按用户要求停止，保留全部模型、初始化及缓存，不新增训练/forward/sweep，不自动恢复。旧 `remaining_required_work` 列表有历史待办；最新 stop receipt 和状态优先。

证据：[最终 fixed-input 报告](../idea-validation/fastwam-aqlm-loss-pair-20261026/pulled/26008761.pbs101/results/report.zh.md)、[两臂 summary](../idea-validation/fastwam-aqlm-loss-pair-20261026/pulled/26008761.pbs101/pair/summary.json)、[comparison.json](../idea-validation/fastwam-aqlm-loss-pair-20261026/pulled/26008761.pbs101/results/comparison.json)、[停止后训练诊断](../idea-validation/fastwam-aqlm-loss-pair-plus-20261026/STOPPED_ANALYSIS.zh.md)、[RUN_STATE](../idea-validation/fastwam-aqlm-loss-pair-20261026/RUN_STATE.json)。

## fastwam-aqlm-loss-pair-plus-20261026

**原目标与配对。** 在源 AQLM 两臂 final8 和全部 fixed-input 评估完成后，各评冻结 280 Plus variants，共 560 新 episodes。沿用 4 suites×7dimensions×10variants、state0、环境 seed `800000000+index*1000`、sampler seed `900000000+index*1000+replan`；10 动作执行、400/700 cap。四 GPU 单 allocation，worker0/1 为 reconstruction 偶/奇片，2/3 为 ours；原计划各 140，20 秒错开，15 秒 GPU telemetry。External reference slots0，两臂产生自身新 episodes，不复用其他 VQ 评分。

**来源和部署。** `bind-source` 已核实真实 pair、两臂 8 updates/80 exposures、calibration80/held-out80 及 anchor 后绑定。614 Linears、AQLM g8/M4/K256、冻结 Smooth/H/R0、dynamic fake A4、BF16 KV、decoded BF16 dense；没有 native 低比特 kernel 结论。四 workers 各进行完整 source anchor 精确 replay 和 formal-infer action≤1e-6 检查，实际完整 action/future source replay 均 0；不新增训练/BF16 teacher。

**用户停止及验收状态。** 闭环 `26010239.pbs101` 在 525/560 条时按用户指令取消，PBS **F/Exit141**、walltime **09:33:11**。Reconstruction **280/280**、ours **245/280**；缺 35 条不记失败，原 560 目标明确未完成。CPU 只读分析 `26020010.pbs101` **F/0**、00:02:37，新增 training updates0、model forwards0，分析完成不代表 560 评测完成。最新完成 receipt 为 **2026-10-27 13:57:09 SGT**，监控已停止，automatic_resume=false。

**闭环结果。** Reconstruction 完整 **202/280（72.14%）**。严格共同 245 条上，reconstruction **179/245（73.06%）**，ours **125/245（51.02%）**，差 **−22.04pp**；ours 逐 episode **5 胜/59 负/181 平**。

| Suite | 共同 episodes | Reconstruction | Ours | Ours 差值 |
|---|---:|---:|---:|---:|
| Spatial | 70 | 55/70 | 46/70 | −12.86pp |
| Object | 70 | 53/70 | 42/70 | −15.71pp |
| Goal | 70 | 46/70 | 25/70 | −30.00pp |
| Long | 35 | 25/35 | 12/35 | −37.14pp |

**解释与接手边界。** 245 条 overall 按 episode 汇总，Long 仅半数；不是四 suite 等权的完整 280 结果，也不是随机停止得到的无偏总体效应或显著性证据。不能把 ours125/245 直接与 reconstruction202/280 作配对差，也不能写 ours125/280。训练阶段已发生显著拟合退化，问题不应只归为环境不适配。详细初始化、C80 漂移、参数变化与未知原因见上一条及 STOPPED_ANALYSIS；若以后明确重开，先验证优化方向，再考虑重训，不自动补剩余 35 条。

**资产。** 本地停止分析含 closedloop_stopped/calibration_drift/parameter_summary 与 completion receipt；全部 614 层参数明细和原 episode 副本保留于 compute 端 `/scratch/users/ntu/yguo017/fastwam-aqlm-loss-pair-plus-20261026/user_stop_analysis/26020010.pbs101/`。原两臂 checkpoints 不覆盖、不重拟合。

证据：[STOPPED_ANALYSIS.zh.md](../idea-validation/fastwam-aqlm-loss-pair-plus-20261026/STOPPED_ANALYSIS.zh.md)、[停止receipt](../idea-validation/fastwam-aqlm-loss-pair-plus-20261026/user_stop_receipt.json)、[CPU完成receipt](../idea-validation/fastwam-aqlm-loss-pair-plus-20261026/stopped_analysis/26020010.pbs101/completion_receipt.json)、[机器可读分析](../idea-validation/fastwam-aqlm-loss-pair-plus-20261026/stopped_analysis/26020010.pbs101/summary.json)、[冻结协议](../idea-validation/fastwam-aqlm-loss-pair-plus-20261026/protocol.json)、[当前RUN_STATE](../idea-validation/fastwam-aqlm-loss-pair-plus-20261026/RUN_STATE.json)。

