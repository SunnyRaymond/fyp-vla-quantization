# Action-gradient geometry：冻结的最小screen

2026-09-13，未采集新输出。裁剪pipeline：primary prior、C02受控诊断、冻结gate与独立审查。旧prior brief位于 `idea/v100-new-angles/action-gradient-geometry/IDEA_PRIOR_GATE.zh.md`，不移动或覆盖。

问题：predictor-only W4 RTN是否在保留候选objective全局结构时，破坏DINO-WM可供GD使用的局部action gradient？这是不同于CEM elite更新与residual拟合的诊断；不提出gradient-aware PTQ新算法，不推断真实环境梯度。QuantWM已覆盖WM PTQ与objective-success错位，2402.05290已讨论world-model gradient；generic gradient-aware framing不新。

使用已验证Wall epoch65 checkpoint/source，保持num_hist1、encoder FP32。两臂FP32与predictor-only W4 signed[-7,7] per-output RTN dequant FP32；全部parameter不求梯度，只有normalized action leaf求梯度，无STE，无训练。

固定6个新dataset indices118..123；env seed=970000+i，candidate seed=980000+i（i为0..5）。不读84..95 reserved states；通过dataset映射证明与先前indices0..117底层episode不重叠。每episode固定64个标准正态action candidates，H2×actiondim10。使用与既有screen相同的官方visual终端objective；明确不新增物理成功指标。梯度anchors固定为pool indices0,1,2,3，禁止按score/error筛选。

保存FP/Q全部64 scores以及4 anchors的action gradients。global binding：平均rank处理exact ties的Spearman>=0.90，且centered score NRMSE<=0.25；NRMSE=RMS((sQ-mean(sQ))-(sFP-mean(sFP)))/std(sFP)，分母<=1e-8为no_binding。只检查objective全局结构，不能写成latent output误差小。

局部比较采用每个gradient的单位L2方向，在20维normalized action中走同一个长度0.10的负梯度步；不裁剪、不加噪声、不线搜索。用FP32模型评价两种step前后的objective，保存原score、FP-step score、Q-step score。4 anchors的FP/Q gradient norm均须>1e-8、FP-step改善均须>1e-7，否则该episode的局部gate no_binding。坏局部几何定义：4个gradient cosine中位数<=0.50，且Q-step平均FP改善<=FP-step平均改善的0.50。差方向和坏一步必须同时出现，不能仅凭norm或sign比例决定。

总gate：至少4/6 episodes同时满足global binding和坏局部几何，才为conditional_signal；若不足4个global/local可识别episodes，记inconclusive_binding；可识别足够而联合signal不足4，记mechanism_no_go。数值比较仅1e-12容差。逐episode与逐anchor原值全部保留。阈值是经济筛查，不是显著性或novelty证明；尤其不能把负结果标为novelty_no_go。

工程gate：真实V100+allocation、source/checkpoint/helper hash、weight restore、finite raw arrays、明确action graph；同一第一anchor用中心差分核查FP和Q的directional derivative，epsilon固定0.005，容差`abs(FD-dot(g,u)) <= 0.1*abs(dot(g,u))+1e-4`。失败为implementation/inconclusive，检查原因后只做同协议修复。记录该检验实际数值，不使用STE补救失败。

最多1个V100作业20分钟及1个CPU raw-metric复算，不跑环境rollout，不换H/bit/seed/threshold，不深验。所有模型/数据/IO/hash/统计在真实compute allocation；head仅小控制和scheduler。所有结果保留。

## 执行前独立审查澄清（尚无本候选输出）

保持既有官方objective不变：alpha=1、base=2、mode=last，实际为terminal visual MSE + terminal proprio MSE；上文“visual终端objective”是简写不准确，不能解释为visual-only，不改成alpha=0。读取实际import的planning/objectives.py并记录hash。

118..123为validation dataset indices（取layout），初始/目标state由固定env seeds经env.prepare产生；只调用screen._new_target的初始化/渲染路径并把target metadata goal_H设为2，禁止smoke._make_explicit_targets及其25-step env.rollout。模型rollout horizon仍严格H2。64候选均来自固定normal RNG，不将candidate0改零。

每episode第一个anchor做FP/Q各自FD，u=g/||g||；raw依序存L_m(a+0.005u),L_m(a-0.005u)，比较中心差分与正||g||。负步a-0.10u的两条结果均用FP模型评分，gain=L_FP(a)-L_FP(step)。只使用torch.enable_grad内的独立action leaf、parameters全部requires_gradFalse、每candidate独立loss梯度，不沿用runtime全局禁用grad。NRMSE逐episode，population std ddof0。上述是无输出前修正接口/语义，不改变筛查阈值或样本。
