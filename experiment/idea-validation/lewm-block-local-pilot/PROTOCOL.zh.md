# LeWM 分块 predictor：冻结配对 pilot

## 问题与证据边界

目标是检验：在与既有 cheap predictor 接近的参数预算下，保维换坐标、局部 residual predictor 和低维跨块消息，是否改善真实 LeWM action-conditioned prediction，尤其是规划需要的 elite 排序。速度是计算预算约束，不能替代质量证据。

官方 LeWM encoder/predictor/checkpoint 只读。真实接口是 192D CLS，policy 初始历史 H=1，预测过程中最大历史 3，horizon 5，packed action 每 token 10D。学生不接收 goal，不含 teacher；goal 仅进入外部官方 criterion。每步反馈自己的预测。此实验不把 C-SWM object slots 当作 LeWM 的现有表示，不减少状态维数。

## 模型与预定比较

每个 training seed 从头训练五个 arms，各 3000 updates：

1. `balanced_base`：已有 h256 shared recurrent、无 attention recipe 的当前重新训练对照。
2. `flat_adapter`：正交坐标 adapter，加读取全部三步历史的小 Flat MLP。
3. `block_adapter_global16`：同类正交 adapter，192D 分成 6 个 32D 块；每块读取自身三步 latent 历史、最近三枚已消费的 action tokens（含当前、缺失过去以零左填充）和共同的 16D 全局消息；六个独立的小 MLP 以 batched 运算更新。
4. `block_adapter_local16`：与 global16 严格同参数、同局部函数和投影初始化；16D 摘要只来自本块，不存在跨块状态信息通路。不是删掉参数的容量消融。
5. `block_identity_global16`：坐标固定为 identity 的诊断 arm，其余结构及初始权重同 global16。报告少了可训练 adapter 参数的限制，不能把该对照单独解释成纯容量匹配。

正交 adapter 用 Cayley transform，Q 初始 identity，s=zQ，预测 delta 后通过 Q transpose 回到原 latent。训练每个 rollout 计算一次 Q；训练后冻结并物化 Q，推理仍计入所有实际坐标旋转、消息和反变换开销。Flat 与分块都有该 adapter，防止把 adapter 收益误归给分块。新的四个 arms 都显式保留相同的最近三步已消费动作窗口；旧 balanced_base 仅直接读取当前 token，是历史 recipe 参考，不能将与它的差异纯归分块。局部 hidden=271，Flat hidden=550，参数预算包含动作窗口。没有 reconstruction 任务、额外位置标签或额外 response loss。

主比较是 `block_adapter_global16` minus `flat_adapter`。通信比较是 global16 minus local16；坐标和现有 recipe 比较都是预定次要比较。不能用次要比较取代失败的主比较。

## 数据与训练

直接使用 teacher-screening job `25213164.pbs101` 的 reconstructed balanced rows：512 个 train episodes/contexts，64 条固定 action candidates/context，早/中/晚 anchors 为 171/171/170。验证 Phase2 `24926383.pbs101` manifest，禁止混用 temporal-balanced cache。所有 arms 使用同一 bank、teacher targets、外部 criterion、minibatch schedule、optimizer 和 update budget。

三个 training seeds 的初始化/训练/schedule分别为 (20300901,20300902,20300904)、(20300911,20300912,20300914)、(20300921,20300922,20300924)。不同结构不声称全部参数同初始化；global/local/identity 的共同参数严格配对。Arm 执行顺序按 seed 轮换，避免把单个 seed 与顺序绑定。

AdamW lr=3e-4，weight_decay=0.01，betas=(0.9,0.999)，eps=1e-8。每 update 8 contexts×64 candidates，五步自回归。目标维持 balanced_base：mean-normalized horizon-weighted latent MSE（weights 1/3,2/3,1,4/3,5/3）加 0.1 context-normalized teacher-score SmoothL1（teacher top12 weight2，其余1）。固定 step3000 为主 checkpoint，禁止 early stopping、结果依赖选择或宽度/学习率 sweep。

## 新 held-out 评价

在真实 PBS compute node 上读取 HDF5 ep_len 元数据，按 random.Random(20300903) 排列 valid episodes，验证前520与旧8 heldout+512train完全一致。跳过前1000；排除 training、旧 heldout、seeded32、real-observation80 和 closed-loop50 manifests，再取前16个新 episodes。IDs 必须在任何新 held-out模型推理之前写入 selection.json；不足则停止，不能缩减或重复样本。

每 episode 早/中/晚三个 anchors，每 anchor 两个新的 action-prefix seeds20301027/20301028，每 bank300 candidates、elite30，合计96 blocks。Gaussian scale/bounds和 logged candidate-zero 继承既有 bank builder，不把该固定 bank 评价称作官方 CEM运行。

主指标是 standardized teacher elite regret 和 top30 recall；另报告 Spearman、bank-centered terminal action-response MSE/cosine/scale、relative latent MSE、worst blocks及 temporal strata。对照按 episode/anchor/action seed严格配对；先在每 episode的六个 blocks取差值median，再在三个training seeds取该episode差值median，最终跨16episodes汇总。95% paired bootstrap以episode为单位，三个training seeds作为固定实现复现，不把candidate当独立样本、不外推训练随机性总体。

## 预先冻结的判据

相对质量 primary gate：主比较的 episode-median elite regret delta <= -0.05，top30 delta >= +0.05；至少10/16 episodes regret严格改善；aggregate bootstrap regret CI upper<0且top30 CI lower>0；三个training seeds分别在两项指标上均无负方向整体退步。Any response denominator floor、非有限值、因果/接口检查失败时，质量gate不能GO。

绝对 fidelity gate 继承旧量级：每seed median Spearman>=0.95、minimum>=0.80；median top30>=0.75、minimum>=0.50；median relative latent MSE<=0.25；每早/中/晚 stratum median Spearman>=0.95及top30>=0.75。没有因相对改善而放松绝对门槛。

计算预算：报告完整五步 predictor B=1/300延迟（同GPU、float32、TF32off、20warmup、60repeats，交替顺序、median/p90），encoder/CEM/environment不计入；所有adapter和message运行开销计入。保持global/local严格相同参数，Flat含adapter与既有775872预算差距<=2%。Cheap约束为B300相对官方teacher至少20%延迟降低。Flat更快不否定分块质量改进；未达速度约束则不能称便宜替代。

## 后续分级与停止条件

Stage A 完成全部15fits及配对评价。若相对主质量gate不通过：记录NO-GO/inconclusive并停止，不继续C-SWM、不追加训练或CEM/闭环。

若相对质量gate通过且cheap/机制检查通过，自动进入预定Stage B：在前8个新episodes的middle anchor上，固定使用第一个training seed的terminal checkpoint（不择优），比较teacher、flat和global16的300/30、30 rounds，shared innovations。当前远端源码已经确认：初始mean0/std1，candidate0为更新前mean，torch.topk(largest=False)，std默认correction1，无clip/std floor；精确源码快照已保存，但远端没有可核实的Git HEAD。它是保留原生update的fixed-observation诊断，不声称完整官方solver/MPC生命周期。Teacher shadow仅诊断，不能混进student proposal；报告elite、mu/sigma、first-action和teacher regret轨迹。Regret用student更新后mean的teacher score减teacher更新后mean的teacher score，除teacher第一轮候选score的population std（floor1e-6），防止后续收缩改变归一化尺度。具体控制脚本和contingent freeze均在接触结果前完成。

闭环Stage C只在绝对fidelity与Stage B稳定性条件均通过时执行：teacher/flat/global16同新任务、环境seeds和CEMbudget的配对PushT pilot；任务身份/数量/判据先冻结，再运行。若触发，必须完成而不是只留下建议。否则明确NOT_RUN_GATE_FAILED，不声称任务成功率改善。本协议不承诺任意训练结果都运行闭环。

## 集群约束与交付

所有tensor tests、checkpoint/HDF5/model/data读取、训练、benchmark、bootstrap在真实PBS compute allocation，重操作前验证PBS_JOBID、PBS_NODEFILE与当前非login hostname。Login只连接、提交、查询与<=200KB小控制文件传输；无下载、安装、递归扫描、重hash。GPU每30秒采集utilization/显存写job.log，结束kill本作业monitor并退出。

交付FREEZE/PROTOCOL、models/checks/runner、PBS job身份、15训练记录/checkpoints（远程）、selection、逐block/episode对照、主/次gate、计时及中文结果解释。失败原样保留。模型由gpt-6-luna/xhigh委派实现；不动banked reset。

实验设计使用已读取的experimental-design skill的配对、blocking及独立实验单位原则；没有生成因子搜索或增加依赖。

方法来源：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. [arXiv / DOI](https://doi.org/10.48550/arXiv.2609.00065)。2026-09-26查验当前记录为v2；这是实验设计流程来源，不是分块机制的实验证据。

