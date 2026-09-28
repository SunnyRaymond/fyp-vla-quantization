# 推荐 1 / 2 后续实验结束记录

2026-09-27。用户授权将 FastLeWM cache 验证与 LpWM real-history 诊断分开进行，遵循冻结实验规则。两项均已完成各自的停止阶段；没有重新训练模型，没有恢复 LpWM dense 或 PLDM formal，也没有启动本记录建议的下一轮实验。

| 实验 | 实际结果 | 结论边界 |
|---|---|---|
| FastLeWM exact encode cache | 50 active environments 的 fixed-observation solve，3 seeds 全部 byte-exact；matched median latency 降低 56.31%–56.58%；closed-loop 每个 seed 的第二次 solve 出现 actions/costs 差异 | Fixed 部分 PASS；整体与 closed-loop fidelity NO_GO。成功 outcomes 相同不能替代动作一致性门 |
| LpWM sparse checkpoint + real history | 8 tasks、16 contexts 全部完成；real-history elite regret 中位降幅 5.63%，4/8 tasks 改善 | 样本门满足，但未达到冻结的 10% / 至少5个任务改善门，有效 NO_GO；停止在机制阶段，未运行 history closed-loop |

## 1. FastLeWM cache：工程收益成立，闭环证据尚未成立

同一 solve 的 50 个 active environments 沿用 native `batch_size=1`；encoder 实际逐 environment 处理 B=1，本次不是 vectorized encoder batch B=50。

| Seed | Native median solve | Cache median solve | Paired median reduction |
|---|---:|---:|---:|
| 42 | 24.043 s | 10.475 s | 56.42% |
| 43 | 24.263 s | 10.589 s | 56.31% |
| 44 | 25.027 s | 10.871 s | 56.58% |

每个 seed 覆盖1,500组 candidate actions/costs、每臂450,000 scores，完整 trace 和返回 actions/costs exact；每次 solve 的3,000次 encode 请求压至100次实际计算。Peak allocated memory ratio为1.01193。

Closed-loop 每个 seed 的第一次 solve exact，第二次分别有11、20、10个 task 的 costs及对应动作出现差异。两臂的逐 task success outcomes 仍相同，分别49/50、49/50、48/50。现有记录缺少每个 replan 的输入图像和 RNG states，不能判断差异来源。Raw eval wall 包含 benchmark overhead，且 fidelity 未通过，不作为部署加速证据；这组比例也不能与历史约5×相乘。

最值得改进的是配对诊断：在第一处分歧记录 observation/goal bytes、solver RNG state及候选序列，在完全相同 context 下复核 cached/native。当前结果足以保留 fixed-input cache baseline，不能证明闭环中可直接替换。

详情见 [cache 结果](../fastlewm-cache-b50/RESULT.zh.md)。

## 2. LpWM：训练、这次诊断和它们的区别

原 sparse 作业 `25571461.pbs101` 从零联合训练 ViT CLS encoder 与 `mlp_var` predictor，D=384、最大 history=3、frameskip=5；2 epochs / 61,930 updates，训练 seed0。Sparse recipe 是 RepReLU + rectified-Laplace RDMReg，regularizer weight0.1、muP LR5e-4。它保留384维，通过出现exact zeros改变表示分布，没有测得稀疏 kernel 加速。

训练使用4帧真实观测，对前3个 causal positions 的 next-step latent prediction 做 MSE，再加0.1×RDMReg。Target encoder 同样回传梯度，没有 EMA teacher 或 decoder loss。原 native planning 使用 latest checkpoint、50 tasks、seed base99、CEM300/top30/30、future H5；每个 model action block包含5个 primitive actions，MPC最多10次，每次执行5 blocks。结果21/50，与其他 LeWM harness 的成功率不直接比较。原训练和评估详情见 [sparse 正式记录](../../reproduction/lpwm/RESULT.zh.md)。

该 `mlp_var` predictor 略去bias可写为 `ẑ_next = W ReLU(A₀z_t + A₁z_{t−1} + A₂z_{t−2} + B e(a_t))`，其中e(a_t)是action block的embedding。它包含非线性readout，不能仅凭类名称为纯线性模型。Cold-start缺少的真实lags在后续rollout中由预测补入；真实history既改变历史项，也可能改变ReLU激活及对action的响应。因此history是否有益需要实际对照，不能仅凭“训练history=3”认定规划入口是bug。

**本次没有训练。** 它复用这个 checkpoint，比较 native 单帧 cold-start 与3帧真实近期 history。Native rollout 会用预测逐步补齐 history；real-history arm改用实际过去观测及对齐的两个历史 action blocks，未来候选及执行窗口仍为5 blocks。Cold 对照保留 predictor最大history=3，在16 contexts / batch300上与native objective数值完全一致，最大差0。

冻结 cohort 为旧29失败任务中ID升序前8个：0、2、3、4、5、8、9、12，每任务取第2和第3次重规划前的状态。旧输出没有保存 native actions / proposal bank，因此本次重新采集诊断轨迹；不是旧失败轨迹的逐步重放，也不要求旧 outcomes 一致。

每个 context 的固定300候选由100 early proposals、170 late non-elites和30 late elites组成，保留重复及来源。两个模型 arm评估相同候选。真实终点从原始 seed/init state重放完整已执行prefix后再执行future actions，避免将不含物体速度的7D状态当作完整物理状态。

Primary真实终点分数为前4个目标位置坐标误差除20后平方，加wrapped angle误差除π/9后平方。它是诊断分数，不替代官方 success predicate。Elite regret为模型选出的30候选的真实平均分数减去同一bank中真实最佳30的平均分数，越低越好。先平均同任务两个anchors的regret，再计算相对降幅。

## 3. Real-history 完整结果

| Task ID | Cold regret | History regret | 降幅；负值为恶化 |
|---|---:|---:|---:|
| 0 | 2.5520 | 1.6921 | 33.70% |
| 2 | 0.6222 | 1.6887 | −171.40% |
| 3 | 2.2710 | 1.7869 | 21.32% |
| 4 | 0.0686 | 0.0548 | 20.07% |
| 5 | 2.0994 | 2.2616 | −7.73% |
| 8 | 1.2308 | 1.2546 | −1.94% |
| 9 | 0.3978 | 0.3992 | −0.35% |
| 12 | 2.3507 | 2.0775 | 11.62% |

16 contexts / 8 tasks / 8 non-at-floor tasks满足样本门；中位降幅5.63497%，改善4个任务，两个支持条件均未达到。无需以缺样本或接口错误解释本次 NO_GO，也不通过追加任务或降低标准追门。

Secondary作描述性分析：每个task先平均两个anchors，再取8个task的中位数；response ratio在每context中先取候选中位数。

| 描述性指标 | Cold | Real history |
|---|---:|---:|
| Future frames 1–5 latent MSE | 0.007542 | 0.007917 |
| Terminal action-response direction cosine | 0.7222 | 0.7083 |
| Terminal predicted / true response magnitude ratio | 1.5563 | 1.6156 |

仅2/8 tasks的latent MSE降低，2/8 tasks的direction cosine提高。所有contexts在truth response非零时，predicted-zero response count均为0。Response以bank index0的同一早期候选为reference，比较不同future actions所造成的terminal latent变化；它不是物理位移的倍数，也不能直接推出物体过推。

## 4. 对下一步的启示

**补真实history不足以稳定修复这个checkpoint。** 个别任务改善真实elite选择时latent MSE反而升高；例如task0、3、12。Task9的MSE降低但regret略恶化。预测MSE、action response和真实任务选择确实需要分别检验。

**当前信号更接近响应幅度偏大且方向质量因context而异。** 它不支持把这批失败概括为“模型完全忽略action”。下一轮若研究LpWM，优先分解terminal action-response误差：沿真实响应方向的gain、正交误差、contact/history条件与latent objective对真实终点的排序关系。幅度ratio大并不证明纯gain校准足够；也没有证明encoder、predictor或latent cost哪一个是根因。

最小后续设计应先使用冻结任务划分，区分“oracle诊断显示可修”与“无需true future的可部署校准”。任何使用真实终点或teacher拟合的收益先作为机制上限，不称为可部署控制或加速。当前不重训、不加新loss或架构、不扩展history实验。

工程上先解决cache闭环配对缺失，已有fixed-input收益最明确；研究上优先定位LpWM的action-response幅度/方向与真实elite选择的关系。两者独立推进，不把一项的性能比与另一项相乘。本次数据没有建立新的文献novelty claim。

## 5. 执行记录与证据

- Cache GPU `25577481.pbs101`：F / PBS Exit4 / wrapper4 / Stageout1，wall00:33:37；全部冻结pairs完成，Exit4对应closed-loop科学门失败。
- History GPU前两次 `25577811.pbs101`、`25578046.pbs101`：分别packed action维度误设15而实际10、generic observation预处理缺proprio而失败。两次均未产生机制质量结果，保留为技术ERROR。
- History修复后的 `25578290.pbs101`：F / PBS Exit0 / wrapper0 / runner0 / Stageout1，wall00:24:27；runner机制阶段1345.482s。科学NO_GO通过summary明确记录，Exit0只表示完整执行。
- 所有模型、physics及benchmark在真实PBS allocation；5秒GPU telemetry写入job log。Stageout1保留，shared-scratch结果已实际取回核对。

本次仅是8个旧失败任务、每任务2个固定contexts的机制screen；候选不独立，tasks可能共享dataset轨迹，不作总体显著性或全50闭环结论。没有history task-success或speedup claim。

实际证据：[history 完整summary](HISTORY_SUMMARY-25578290.json)、[cost gate](HISTORY_COST-25578290.json)、[capture manifest](HISTORY_CAPTURE-25578290.json)、[入口接口检查](HISTORY_RUNNER_PREFLIGHT-25578290.json)、[冻结协议](../lpwm-real-history/PROTOCOL.zh.md)、[attempts](../lpwm-real-history/SUBMISSION/attempts.md)。
