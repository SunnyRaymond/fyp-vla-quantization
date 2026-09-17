# RankCal × DINO-WM：Wall 实验协议 v0.1

日期：2026-09-09。状态：设计完成，尚未实现/运行；不是 pipeline 输出的实验结果。

## 1. 决策与研究边界

- 主 benchmark：官方 Wall / wall_single，random-state goal task。复用公开 checkpoint，不训练模型。
- 第二 benchmark：PushT，仅在 Wall 通过机制验证后实施；需单独核验 checkpoint、环境、success predicate，不复用 Wall 的阈值。
- 第一轮检验：在固定 weight-bit budget 下，site-only candidate-order influence 是否比 block-output reconstruction error、planner-score error 和随机 mapping 更好地保留完整量化模型的闭环 success。
- 第一轮为 numerical weight-only PTQ pilot：FP32 reference，W4/W8，activations 为 FP32，量化权重 dequantize 到 FP32 运算。不得宣称真实低比特 kernel、memory reduction 或 speedup。
- 部署验证是后续独立阶段：只用实测 backend-native 配置，在 measured PeakBytes 和实际 latency 下重新比较。
- 本方案明确修改原 proposal 的 inferred DreamerV3/TD-MPC2 + DMC、FP16 background、mixed W/A、完整 nuisance grid 和全局逐 tensor sites。这里使用 DINO-WM + Wall、FP32、weight-only、block-group intervention 和一个主 planner setting。原文件保持不变；本 pilot 不等于验证完整原方案。

选择 Wall 的理由：本地已有 source、官方 checkpoint 身份和评估适配记录；QuantWM 也在 Wall 上做过量化对照；可先隔离 quantization/planner 问题，降低重新搭建 benchmark 的成本。Wall 单任务成功不能外推到 manipulation、其他 world models 或真实机器人。

## 2. 证据账本

| 状态 | 内容 |
|---|---|
| 本地 source 已核对 | CEM 根据 loss 的 top-30 候选更新 mean/std，返回最终 mean action sequence，并非 top-1 候选。 |
| 本地 source 已核对 | objective = terminal visual MSE + alpha × terminal proprio MSE；alpha=1。 |
| 本地 source 已核对 | Wall success 使用位置前两维 Euclidean distance <4.5。 |
| 已有记录，非当前重跑结果 | 旧 FP32 Wall run 在第7轮人为停止，日志47/50；没有完整最终 artifacts，不能当本实验正式 baseline。 |
| 假设 H1 | 排序影响在未用于估计的新状态上有可重复性。 |
| 假设 H2 | FP32-background 的单 block 信号对联合量化配置仍有分配价值。 |
| 假设 H3 | 排序/elite/action fidelity 改善能转化为 held-out success 改善。 |
| 未验证 | 真实 checkpoint 当前可用性、runtime site counts/shape、低比特 backend、完整 replay、数值稳定性和耗时。 |

## 3. Baseline 身份与 planner

已存身份：DINO-WM source commit `0a9492fa12044b852ae9e001cc74604b79c8bb0c`；DINOv2 source `7764ea0f912e53c92e82eb78a2a1631e92725fc8`；checkpoint `outputs/wall_single` epoch 65，记录大小368656057 bytes；DINOv2 encoder `dinov2_vits14`；predictor depth 6。实施前核对文件存在、checkpoint metadata、shape 和 source diff，不重新训练、不更新依赖或上游 revision。

| 项目 | 主实验固定值 |
|---|---|
| reference | FP32 / eval / no gradients；decoder=None，所有方法相同 |
| inputs | 官方 preprocessing / normalization，224图像输入；不改内部resize |
| goal source | random_state，官方分布；环境布局来自固定的dataset metadata manifest |
| prediction horizon H | 5 model actions |
| execution interval E | 5 model actions，每个包含frameskip=5个原始动作，即每次25个环境step |
| candidate count | 300，candidate 0 仍是当前mu |
| elite count | 30 |
| CEM optimization | 固定10 iterations，var_scale=1 |
| outer MPC | 最多12轮，最多300个真实环境steps；仅执行后的success允许提前结束 |
| objective | 官方mode=last，alpha=1，base=2；保持visual+proprio loss |
| tie rule | E2保留exact ties；elite selection用stable candidate-index排序 |
| processing | 每次一个episode；候选batch保持300。不得以方法为单位改变batch/chunking |

**两项统一的 planner adapter 修改必须披露并做 FP32 bridge：**

1. CEM 内层传入 `evaluator=None`，不在每个iteration执行真实环境来触发success早停；固定10次模型规划。外层MPC evaluator保留。旧代码会根据内层真实环境success提前break，且batch组成会影响停止。此修改让planner budget固定，所有量化/FP32方法都使用修改后的版本。
2. `torch.argsort(loss, stable=True)` 统一ties；非finite score直接标记模型失败并保留episode，不丢弃或随机jitter。

这不是对官方论文主表的严格复现。先在独立smoke目标上比较原始与adapter FP32，解释预期差异，检查无额外行为改动；再把adapter FP32作为唯一正式reference。禁止根据哪个adapter更有利而选择结果。

## 4. 数据与 randomness

设计规模（是预算受控pilot规模，不声称统计power已足够）：

| split | episodes | 作用 |
|---|---:|---|
| smoke | 2 | metadata、reset/replay、量化旁路no-op、计时和FP32 adapter bridge |
| calibration | 24 | 所有3种allocation signal共用；前8个作工程耗时预检 |
| development | 20 | 量化强度筛查、held-out influence诊断、停止/继续决策 |
| pilot_test | 50 | 冻结方案后的首次完整对照；仅作exploratory evidence |
| confirmatory_test | 待pilot后独立确定 | 全新episodes；样本量与功效先定再生成结果 |

- 每个episode显式保存ID、environment seed、CEM seed、初始状态、目标状态、wall/door layout、dataset trajectory ID、normalization identity。不能只用全局seed标识。
- 原 `eval_seed=[seed*n+1,...]` 在所有run复用1；改成manifest里的互不重叠显式seed，检查所有split的seed和完整(start,goal,layout)无重复；记录任何预先排除的重复，不因困难或失败剔除。
- 预分配seed namespaces：smoke 100000起，cal 200000起，dev 300000起，pilot_test 400000起；confirmation另用500000起。env与CEM分别记录，后续replicate另分namespace。
- 原 `plan_targets.pkl` 没有environment seed/layout/RNG，`goal_source=file`也不自动恢复这些；新增sidecar并显式恢复环境。只有seed不同不足以证明targets不同。
- calibration每episode取前两个实际发生的MPC planning points，各取CEM iteration 1/5/10的300候选；最多144个scoring records。成功早停后不人为续跑补记录；先episode内平均，避免长失败episode主导signal。
- 每个record包含raw/transformed current+goal observations、proprio、候选actions、mu/sigma、CEM/MPC index、score、stable IDs、RNG state、model/preprocessor revision。
- 若量化encoder，必须从相同原始观测重新编码current与goal；不能复用FP32 latent并声称测了encoder影响。predictor-only probe可缓存FP32 encoder结果，但须验证cache与完整路径等价。
- 当前观测与goal均使用同一个被干预模型的encoder，保持deployment语义。固定FP32 goal latent属于另一项ablation，不混入主实验。
- CEM innovations按(episode,MPC round,CEM iteration)单独生成，与simulator RNG隔离；相同setting中各方法用相同standard-normal innovations，但各自mu/sigma和candidate pool可分叉。保留candidate0=mu。
- 为避免observer/candidate顺序造成偏差，site probe用固定随机顺序，全部signal共享记录；校准数据、qparams在all-sites evaluation中冻结。

## 5. 量化与site定义

第一轮选block-group sites，降低probe成本，也让相同shape的block之间可直接交换bits：

- encoder：`encoder.base_model.blocks.<i>`中所有实际执行的Linear weights；预期12 blocks，必须runtime确认。
- predictor：`predictor.transformer.layers.<i>`中Attention/FeedForward的Linear weights；checkpoint配置6 blocks。
- 每个group一次干预其中全部eligible Linear weights；这是一个group intervention，不冒称单个tensor干预。
- Patch embedding Conv、action/proprio encoders、LayerNorm、bias、positional parameters、softmax及所有activations保持FP32；decoder不参与。完整量化指全部已声明groups都分配，不等于全模型每个参数都低比特。
- operator：symmetric per-output-channel RTN；b=4或8，qmax=2^(b-1)-1；scale=max(abs(W_row))/qmax，Q=clamp(round(W/scale),-qmax,qmax)×scale；全零row直接返回零，round语义固定PyTorch；scale FP32；无learned clipping/rounding/QAT。
- 保存integer值/scale用于逻辑存储记账，mechanism execution将其dequantize到FP32。不得把文件大小或FP32运行PeakBytes称为native INT4/INT8 memory收益。
- metadata manifest记录group路径、每个Linear shape/numel、scale数和quantized coverage比例。若group-count或family内shape不一致，先修订资源strata，不能运行后再改。

主budget：12个encoder blocks中3个用W8，其余W4；6个predictor blocks中2个用W8，其余W4。该配额是预先选定的pilot设计，不是已知最优值。它固定encoder/predictor各自precision配额，只测试family内放置位置，不测试跨family最优资源分配。

只有family内weight numel和scale/packing规格相同才能声称相同logical weight bytes。否则按真实相同shape/cost建strata并修订配额。完整memory包含未量化参数；记录logical packed weight bytes与运行PeakBytes两个不同字段。

## 6. 三种signal与相同allocator

对每个group g，分别做site-only W4和W8，其余全部FP32；共预期36种probe，加FP32 reference。输入和候选相同。

1. **RankCal**：原E2 sign agreement，所有unordered candidate pairs；exact ties保持0。先每record平均，再CEM/point内、episode内平均，最后episode等权得到I_g(b)。排序cost越低越好。300候选=44850 pairs，它们不是44850独立样本。
2. **Local-MSE allocation（自建对照，不称官方OMSE allocator）**：每block用FP32 teacher输入，比较量化与FP32 block输出，采用mean squared error / (FP32 output mean square +1e-12)。encoder包括current和goal branch且两branch等权；predictor对rollout各步等权。再按record/episode层次平均。这样measure局部reconstruction而不是累计输入漂移。
3. **Score-error allocation**：相同候选池的quantized与FP32 scalar score差的MSE，除以该池FP32 score mean-square +1e-12；再按相同episode权重平均。分母在g/b间固定。此对照区分ranking-specific benefit与一般downstream fidelity。

三种signal均使用相同配额和同一选择算法：计算upgrade benefit Delta_g=S_g(W4)-S_g(W8)，在encoder family选最大的3个、predictor family选最大的2个；相等按site ID。负benefit也保留，因配额固定仍按同一规则选取。该算法精确最小化本配额下的可加surrogate，但不证明联合实际discordance或success最优。

随机mapping：独立uniform抽样5个非重复配额匹配配置，固定seed 71001–71005；如碰到已选相同map按同一seed流继续抽，保存实际结果；不得挑表现差的random作为展示对象。

高低单site影响只是假设的allocation signal；正式测完整assigned checkpoint，不能把36行single-site结果相加当all-sites实测值。calibration bootstrap可离线检查selected sites稳定性；pilot默认只有一份calibration fit，限制结论，后续再独立calibration重复。

## 7. 分阶段执行与结果冻结

### G0 工程门槛

2个smoke episodes，检查reference reproducibility、state/layout/RNG restore、旁路hooks不改score、W4/W8数值有限、完整结果可逐episode落盘。先测一个scoring record×一个group×一个bit和一轮CEM成本。禁止按历史整任务耗时直接乘出GPU-days。

### E0 development筛查

在20个dev episodes跑FP32、all-eligible-W8、all-eligible-W4（均相同adapter planner）。保留全部结果。若W4几乎无损，继续完成离线机制测量，但不自动扩充昂贵test；若W8也崩溃或无效值，先定位operator/coverage问题。不得根据test改bit-width；W3/activation quantization必须成为新协议，不作为偷偷替代。

### E1 probe与held-out诊断

先8个cal episodes验证耗时，工程通过后按同一规则完成24个。dev record上评估信号稳定性、elite/action fidelity，以及all-sites实际discordance与可加surrogate的关系。失败时保留negative result，不选择有利horizon/site子集后继续称原假设成立。

### E2 完整闭环pilot

冻结sites、qparams、3个signal allocations、5个random maps、planner、metrics、test manifest后跑50个新的paired episodes：FP32、all-W4、all-W8、RankCal、Local-MSE、Score-error、5 random，共11个模型配置；最多550个model-episodes。这是工作量清单，不是全部提交授权或GPU预算。

每episode配置顺序随机化/分块，同硬件软件，异常记录并使用同一恢复策略。基础设施中断重跑原ID与seed；模型NaN等方法失败不作为基础设施错误删除。

### E3 confirmation与扩展

pilot全部报告后再确定独立confirmatory sample size：主要比较RankCal-Local-MSE的paired success difference，建议以+10 percentage points作为待确认的实际有用收益目标，敏感性分析+5/+10/+15 pp；它们是设计阈值，不是预测。用discordant-pair概率范围做exact McNemar power计算/模拟，目标80%、双侧alpha=.05；若预算不足明确exploratory，不能把50 episodes称充分power。不能在pilot_test反复加样直到显著。

支持ranking独特性还需RankCal-Score-error对照；若将两项都设confirmatory，预先指定Holm控制familywise错误并据此power。random结果报告整组分布，不把5 maps×50 episodes当250个独立环境样本。

确认后再做：独立calibration repeat；H=10且E=5固定的horizon stress test（重新收集/校准H10 pool，各方法相同预算，不改E）；第二memory quota；PushT。原plan.py会把H和E都覆写成goal_H，必须先明确adapter解除此耦合，否则不能把结果叫纯horizon effect。次级结果单独标记。

## 8. Metrics与解释

- **Primary**：12轮/300环境steps预算内，在实际执行的MPC边界首次满足官方distance<4.5的success；保留官方成功后mask/action_len语义。不是CEM内层hypothetical success，也不改成任意中间frame首次碰到目标。
- **Secondary outcome**：预算终点/成功停止点goal distance；失败计满300 steps的capped steps-to-success。不可只报告成功episodes的平均时间。
- **Mechanism（共同state/pool才可比较）**：E2 disagreement、top-30 overlap=intersection/30、top1 ID agreement、同一pool的elite-mean sequence RMSE、first primitive action及完整执行chunk在denormalization后差异。连续action不用exact-equality作为主指标。
- CEM最终output是反复更新后的mu，另外在共同state跑完整CEM，比较最终mu/chunk；单次pool elite mean不冒称闭环最终action。
- 闭环分叉后主要比较outcome，不直接比较各自pool的candidate IDs。若需on-policy diagnostic，在量化访问state上额外做只读FP32 shadow scoring，不能用该结果修改planner或重新校准test模型；单列FP visitation/on-policy strata。
- 全部episode-level paired differences、95% cluster/bootstrap CI；success主比较可用exact McNemar，报告双方成功/失败四格表。每个episode内的records先汇总。bootstrap重采样单位为episode，同一episode所有方法及random maps保持成组。
- Rank更好但success不变：只支持fidelity。RankCal不胜过score-error：ranking特有优势未证明。random相当且CI宽：证据不足，不能直接断言机制被推翻。可靠的等效结果/多次复制才更能限制site mapping解释。
- native backend不可用是deployment可行性限制，不是H1–H3的反证。

## 9. 部署验证（不并入numerical主表）

锁定支持的W4/W8或其他backend-native operator、packing/grouping与accumulation dtype，先验证与数值版本的误差；不假定A100有符合本shape的native kernel。若转FP16，新增FP16 reference/bridge并重新校准相应signal，不能直接沿用FP32结论。

为每种完整map用fresh process测load-to-completion peak allocated、peak reserved、model storage、workspace和planner buffers。固定batch=1、300 candidates、相同harness。方法比较匹配histogram且measured peak差距不超过预注册1%；同时报告精确bytes，不能用人为padding假造公平。原proposal要求exact match，此处1%是独立deployment protocol的明确修改；exact strata另列，无匹配则unmatched。

资源审计另外用相同预先保存输入序列/相同最大planning work做固定工作量测量，避免成功早停导致的work差异；native闭环真实peak也单独报告。两种测量不可合并。warmup与timing测量分开，GPU同步，报告p50/p95 CEM与端到端planning latency，不用单次best timing。

## 10. 实施工作清单与交付

当前只生成设计文档与design manifest，没有runner、quantization实现、GPU结果。

1. 新建隔离experiment runner，复用现有loader/preprocessor/evaluator；不要覆盖旧reproduction artifacts。
2. 加episode/layout/RNG manifest与逐episode原子写盘，验证官方env replay。
3. 实现上述两个统一planner adapters；保存diff和FP32 bridge。
4. 枚举group manifest与量化覆盖率，实现RTN/no-op验证。
5. CEM scoring hooks写candidate IDs、action、score、elite与mu；不改变采样顺序/RNG。
6. 实现三signal与同一quota allocator，离线复核sum-surrogate选择和matched bytes。
7. E0/E1后冻结实际design manifest再执行E2；确认集需另行锁定。

输出文件：source_manifest.json、episode_manifest.jsonl、site_manifest.json、quantizer_manifest.json、probe_scores、influence.csv、allocations.json、per_episode_results.jsonl、paired_results.csv、resources.csv、selected_videos、REPORT.zh.md。probe原始大tensor分批保存/按需缓存，不把全部rollout latents常驻GPU。

至少三张结果图：local-error vs rank-influence；各allocation的success与paired CI；elite/action fidelity vs closed-loop outcome。native部署通过后另画memory-success Pareto；fake quant不画deployment Pareto。

ASPIRE2A：仅allocated GPU node计算；一次先跑bounded smoke，测耗时后计算next-stage上限（scoring calls×实测耗时 + env/I/O margin）；每episode落盘，每个job有walltime与内部deadline，不启动原proposal 12 GPU-days完整grid，不新增训练，不在本设计阶段SSH或qsub。

## 11. 可追溯来源

- 官方仓库及checkpoint入口：https://github.com/gaoyuezhou/dino_wm （2026-09-09读取）
- QuantWM §3.1（OMSE calibration），Tables 7–8（module ablation）：https://arxiv.org/html/2602.02110v1
- 本地 source 根：`experiment/reproduction/dino-wm-wall/source/`
- `planning/cem.py:114–134`：elite update、inner evaluator、返回mu。
- `planning/objectives.py:31`：visual+proprio objective。
- `planning/mpc.py`：执行chunk、success mask、action_len、outer max_iter。
- `plan.py:132`：seed公式；`:194`：H/E耦合；`:299`：不完整target replay文件。
- `env/wall/wall_env_wrapper.py:61`：success阈值。
- `experiment/reproduction/dino-wm-wall/README.md`：source/checkpoint/runtime与旧运行边界。
- `experiment/reproduction/dino-wm-wall/artifacts/16170841.pbs101/checkpoint-config-original.yaml`：frameskip=5、predictor depth=6。

待实施门槛：当前checkpoint可用性/metadata、DINOv2实际block数与shape、cal/test唯一性、no-op/replay、FP32 adapter bridge、实测时间和native backend均不能以本文取代验证。
