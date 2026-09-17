# 新切面筛选总表

状态更新：2026-09-13。所有idea均以最小feasibility screen为止；pipeline采用用户授权的裁剪版本。

先读[结果阅读指南](READ_RESULTS.zh.md)。14个已实测screen的机器可读目录见[SCREEN_REGISTRY.json](SCREEN_REGISTRY.json)，其余条目为prior/source/resource gate，不计经验实验数量。

| 候选 | 实质变化 | 当前状态 | 证据入口 |
|---|---|---|---|
| Antithetic Rounding Pairs | 两个冻结W4 predictor的joint rounding coupling；无训练或bit-map搜索 | mechanism preliminary_go / practical_no_go；GPU64733+CPU64734已完成，停止扩验 | [结果](antithetic-rounding/RESULT.zh.md) |
| Semantic Input Scales | visual/proprio/action接口的固定activation scale partition | mechanism_no_go；GPU64738+CPU64740，primary仅改善0.3385%，停止 | [结果](semantic-input-scales/RESULT.zh.md) |
| Flow Geometry under PTQ | 复用flow几何信号，诊断Q-only信号能否排序FP/Q action drift | inconclusive_provenance；backbone数值子结果scope_limited_preliminary_go，expert mechanism_no_go；GPU64763/CPU64764，停止 | [结果](flow-geometry-drift/RESULT.zh.md) |
| Action-gradient geometry | 全局候选objective保真与局部GD action方向的分离 | mechanism_no_go；GPU64767/CPU64769，4/6可识别、0/6联合signal；停止 | [结果](action-gradient-geometry/RESULT.zh.md) |
| Recency-Split Latent-History Cache | 按history age保留精度 | model_structure_no_go；现checkpoint只有1个slot，未跑无意义GPU干预 | [结论](recency-history-cache/RESULT.zh.md) |
| Batch Scale Coupling | candidate co-batch 对动态activation scale的影响 | novelty/applicability no-go；已有直接先例，当前路径未发现该机制，GPU 0 | [结论](batch-scale-coupling/RESULT.zh.md) |
| OFT continuous/discrete interface | 同checkpoint两种action输出的量化敏感性 | objective-mismatch no-go；L1-only配置不能证明discrete head是可靠基准，GPU 0 | [结论](continuous-discrete-interface/RESULT.zh.md) |
| TD-MPC2 ensemble uncertainty | PTQ对planner epistemic-disagreement信号的污染 | structural_no_go；真实planner没有假设所需variance路径，GPU 0 | [结论](ensemble-uncertainty/RESULT.zh.md) |
| MOPO dynamics uncertainty | PTQ对实际rollout reward penalty使用的ensemble disagreement影响 | resource_blocked；机制存在但没有核验checkpoint，不训练补资产 | [审查结论](../../../idea/v100-new-angles/used-uncertainty/IDEA_BRIEF_AND_GATE.zh.md) |
| Instruction contrast | paired language intervention下的PTQ response差分 | identifiability_no_go；原模块grounding主张混入扰动预算与指令合法性，GPU 0 | [结论](instruction-contrast/RESULT.zh.md) |
| Gripper decision margin | 小连续误差跨执行threshold的离散翻转 | insufficient_mechanism_no_go；当前gate主要是threshold crossing的代数条件，GPU 0 | [结论](gripper-decision-margin/RESULT.zh.md) |
| Flow step refinement | FP/Q不同Euler步数的误差分解 | novelty_no_go / diagnostic_conditional_go；无冻结实验，不计经验no-go | [审查](../../../idea/v100-new-angles/flow-step-refinement/PRIOR_GATE.zh.md) |
| Padded-coordinate feedback | 用已知zero-target解析更新替换未执行坐标的flow velocity | method_no_go；GPU64772/CPU64774，8/8binding、3/8正向，双median gain −1.15%/−1.32%；停止 | [结果](padded-coordinate-feedback/RESULT.zh.md) |
| Conditional action distribution | 分开检查same-noise映射误差与条件action marginal | statistical_inconclusive；GPU64778/CPU64779，0/8正向、仅1/8完整negative gate；停止 | [结果](conditional-action-distribution/RESULT.zh.md) |
| Q ensemble common-U初始案 | 共同randomness改善two-Q min的设想 | prior-method_no_go；min不是planner主路径，原案保留，未跑实验 | [结论](q-ensemble-rounding/COMMON_U_DECISION.zh.md) |
| Existing-Q stratified rounding | 五个已有critics的stratified SR，针对planner实际two-Q avg | inconclusive_binding；GPU64799/CPU64801，0/8可比binding，median gain−29.02%，停止 | [结果](q-ensemble-rounding/RESULT.zh.md) |
| Simplex latent quantization | SimNorm概率组的整数总量约束与latent transition | novelty_no_go / mechanism-claim structural_no_go；标准projection收益不能证明transition composition，GPU0 | [审查结论](../../../idea/v100-new-angles/simplex-latent-quantization/PRIOR_GATE.zh.md) |
| Bellman consistency | reward/dynamics/value的量化误差抵消 | identifiability_no_go；内部residual保真不能识别value收益，且coupling规则未建立，GPU0 | [复核结论](../../../idea/v100-new-angles/bellman-consistency/ROOT_DECISION.zh.md) |
| Value-head gauge | softmax-invariant head centering后再做W4 RTN | implementation_inconclusive；GPU64807/CPU64809，W4 median改善62.38%但FP no-op未过；原screen停止 | [结果](value-head-gauge/RESULT.zh.md) |
| Residual branch interaction | attention/FF量化误差的内部non-additivity | identifiability_no_go；local covariance与endpoint混杂尚未分离，GPU0 | [复核结论](../../../idea/v100-new-angles/new-cut-shortlist/ROOT_DECISION.zh.md) |
| Hidden-basis permutation invariance | 函数保持变换下的RTN等变性 | engineering-only_no_go；预期null是实现自检，不独立跑GPU | [复核结论](../../../idea/v100-new-angles/new-cut-shortlist/ROOT_DECISION.zh.md) |
| Action-chunk suffix feedback | 物理action时间后42位置的误差是否反馈到前8位置 | structural/identifiability_no_go；固定SmolVLA causal action mask缺少该路径，GPU0 | [source审查](../../../idea/v100-new-angles/new-cut-shortlist/ACTION_CHUNK_PREFIX_PRIOR_GATE.zh.md) |
| Prefix KV reuse vs suffix A8 | 持久condition cache与逐步suffix的量化误差记忆 | identifiability_no_go；改变locus/shape/Jacobian，不能隔离reuse，GPU0 | [root gate](../../../idea/v100-new-angles/new-cut-shortlist/ROUND3_ROOT_DECISION.zh.md) |
| Q decoder-tail rounding | decoded-value reconstruction指导101-bin head rounding | insufficiently_differentiated；未提出超出data-aware rounding邻域的独立机制，GPU0 | [root gate](../../../idea/v100-new-angles/new-cut-shortlist/ROUND3_ROOT_DECISION.zh.md) |
| Denoising-call rounding persistence | matched draw multiset下frozen/cyclic weight schedule | scope_limited_preliminary_go / generic_method_novelty_no_go；B2 GPU64825/CPU64831，6/6joint；De-biasing Diffusion直接prior已补入，STOP | [结果与边界](rounding-persistence/RESULT.zh.md) |
| Timestep constant folding | 将time-only affine contribution缓存为K10 table后量化data half | novelty_no_go；TFMQ-DM等已覆盖temporal feature cache，不为模型专属branch sensitivity追加GPU | [root gate](../../../idea/v100-new-angles/new-cut-shortlist/TIMESTEP_ROOT_DECISION.zh.md) |
| Temporal residual correction初始检索 | 已有FRT/PRR高度重叠 | not_pursued / incomplete_prior_review；非经验no-go | [检索状态](../../../idea/wm-trajectory-residual-ptq/STATUS.zh.md) |
| Recorded-future error cancellation | 区分贴近FP predictor与贴近真实记录future的固定encoder feature | mechanism_no_go；GPU64828/CPU64830，0/6 H5 gate通过，停止 | [结果](teacher-bias/RESULT.zh.md) |
| Camera redundancy | 双camera availability与expert W4的交互 | identifiability_no_go；现有scene/wrist不是matched redundant views，GPU0 | [审查](../../../idea/v100-new-angles/new-cut-shortlist/CAMERA_REDUNDANCY_PRIOR_GATE.zh.md) |
| Shared goal/current quantization | 两侧encoder量化与reference坐标的一致性 | identifiability_no_go；FP predictor下四臂不能识别common-mode cancellation，GPU0 | [审查](../../../idea/v100-new-angles/new-cut-shortlist/GOAL_COORDINATE_PRIOR_GATE.zh.md) |

## 读取顺序

1. [Campaign范围与停机规则](../../../idea/v100-new-angles/CAMPAIGN.zh.md)
2. [历史no-go审查](../../../idea/v100-new-angles/PRIOR_NO_GO_AUDIT.zh.md)
3. 各候选协议、独立审查和RESULT文件（产生后链接）

## 数据登记

历史及保留区间：0–83已经使用；84–95为PRR未打开的test_locked，不允许本轮占用。Antithetic预留96–101，namespace930000/940000；batch-scale候选预留102–107；Semantic预留CAL108–111、DEV112–117，namespace950000/960000（各candidate seed加10000）。新候选不得重用已查看DEV进行确认。索引仅为第一层隔离，实际底层episode与state fingerprint仍需在compute预检。

Action-gradient预留validation dataset indices118–123，env970000+i/candidate980000+i；使用这些trajectory的layout构造新初始/目标state，不误称直接读取dataset state作测试。

## 状态含义

`preliminary_go`仅表示所测最小前提有支持，不是完整机器人任务验证；`mechanism_no_go`仅否定所测recipe；`implementation_failure`或`resource_blocked`不产生科学否定结论；`novelty_no_go`与经验可行性分开。

| 后续候选 | 实质变化 | 当前状态 | 证据入口 |
|---|---|---|---|
| Policy-prior proposal support | 只量化actor生成的24个候选；同FP scorer与488个shared candidates | mechanism_no_go；GPU64833/CPU64834，8/8binding、0/8joint，停止 | [结果](policy-prior-support/RESULT.zh.md) |
| Action-token reinjection | 重复FP token替换与量化动态误差清除 | identifiability_no_go；token替换只移除扰动，不能清除其他recurrent state误差，GPU0 | [root gate](../../../idea/v100-new-angles/new-cut-shortlist/ROUND5_ROOT_DECISION.zh.md) |
| Success-conditioned masking | mask/quantization与成功条件的交互 | identifiability_no_go；真实outcome与动作/branch混杂，GPU0 | [root gate](../../../idea/v100-new-angles/new-cut-shortlist/ROUND5_ROOT_DECISION.zh.md) |
| Physical-state readout | visual PTQ误差与固定2D readout-visible分量的区分 | identifiability_no_go；full latent已有真实state对应proprio路径，probe nullspace不等于task不可观测，GPU0 | [审查](../../../idea/v100-new-angles/new-cut-shortlist/PHYSICAL_READOUT_PRIOR_GATE.zh.md) |
| Null-input columns | 静态零输入列影响state projection的row-wise W4 range | insufficiently_differentiated / engineering-only；非exact prior认证，GPU0 | [root判定](../../../idea/v100-new-angles/new-cut-shortlist/NULL_INPUT_ROOT_DECISION.zh.md) |

Euler Jacobian：**mechanism_no_go**；GPU64835/CPU64836，6/6 FP eligible、0/6 W4 geometry flag，全部实现检查通过，STOP。[结果](euler-jacobian/RESULT.zh.md)。

Broadcast activation coupling：**scope_limited_preliminary_go**；GPU64838/CPU64839，6/6binding、5/6 gain>=10%，输入MSE严格匹配，STOP。[结果](broadcast-coupling/RESULT.zh.md)。原encoder/predictor模块比较与CA/SA原案保持prior-only/GPU0，见[原案](new-cut-shortlist/FINAL_BOUNDED_CUTS.zh.md)及[复审纠正](new-cut-shortlist/BROADCAST_ROOT_REVIEW.zh.md)。

Action-token support：source_path_confirmed，但严格mask是直接工程修复，AR checkpoint未建立可运行manifest；prior/resource gate，GPU0，未声称量化已触发越界。[审查](new-cut-shortlist/ACTION_TOKEN_SUPPORT_PRIOR_GATE.zh.md)。

Broadcast contribution边界：generic correlated/stratified rounding method novelty no-go；narrow application unverified，原5/6数值signal保留。[审查](../../../idea/v100-new-angles/broadcast-coupling/NOVELTY_BOUNDARY.zh.md)。

Discrete-tokenizer WM：IRIS/iVideoGPT为design_not_frozen / assets_not_bound，DIAMOND仅在discrete-tokenizer定义下structural_no_go；GPU0，不把缺少当前下载当作永久资源否定。[root结论](../../../idea/v100-new-angles/new-cut-shortlist/DISCRETE_WM_ROOT_DECISION.zh.md)。

Reference-branch rounding coupling：mechanism_no_go，GPU64840/CPU64841；6/6binding、0/6联合gate，STOP。[最终结果](reference-branch-coupling/RESULT.zh.md)。[协议](reference-branch-coupling/PROTOCOL.zh.md)与[设计边界](../../../idea/v100-new-angles/reference-branch-coupling/ROOT_GATE.zh.md)。旧goal-coordinate四臂仍为prior-only；不重开已完成screen。

最终停机：实时Codex额度used99% / remaining1%，全部14个经验screen停止；无待运行job。
