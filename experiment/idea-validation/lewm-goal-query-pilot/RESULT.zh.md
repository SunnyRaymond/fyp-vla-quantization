# LeWM Goal-query pilot：有效执行，NO-GO / STOP

日期：2026-09-26。PBS job `25568103.pbs101`；计算节点 `x1000c0s1b0n1`，NVIDIA A100-SXM4-40GB。

四个训练臂都没有通过 dev 的候选排序质量门，未选出 deployable arm。按冻结协议结束本次 preliminary screen，不推进 adaptive CEM 或 closed loop。这个 NO-GO 只适用于本次 r32、h256、1500 updates 和指定数据分布，不否定所有 goal-query models，也不验证 novelty。

## 实验回答了什么

固定 LeWM encoder、teacher、H5 和官方 terminal squared-distance criterion，从可信当前 latent 与完整动作序列直接预测 goal query。比较 direct scalar、scalar_aux、projected＋goal-dependent tail、small full-latent predictor；scalar_aux 取得相同 projected/tail 辅助监督，但部署不执行辅助头。四臂共享 teacher labels、训练数据和 schedule，两个 initialization seeds，均只读取 step1500 checkpoint。

这次没有测试 `g=g0+Pq` 的 affine goal-family 变体，P 是训练 terminal targets 的 PCA，不能将其结果解释为 goal-family 可压缩性验证。

训练：512 parent episodes ×64 action candidates；dev/test 各8 parent episodes、3个物理时间 anchors、2个 fresh action banks、300 candidates、own＋3个同split donor goals。每个模型每个split有192个嵌套 query banks。这些不是192个独立样本；共享 donor goals 也使 episode 摘要存在依赖。本报告只作描述性工程筛选，无显著性或 population CI。

## Dev 选择结果

每格两数分别对应 initialization seeds `20302611 / 20302612`。Spearman 和 top30 是全部 query banks 的 median。

| 臂 | Spearman | top30 overlap | native scoring p50 / ms |
|---|---:|---:|---:|
| scalar | 0.634 / 0.534 | 0.433 / 0.333 | 0.395 / 0.397 |
| scalar_aux | 0.650 / 0.625 | 0.433 / 0.417 | 0.393 / 0.395 |
| projected_tail | 0.657 / 0.645 | 0.400 / 0.400 | 0.554 / 0.558 |
| full_latent | 0.664 / 0.605 | 0.433 / 0.367 | 0.330 / 0.332 |

冻结门要求两个 seeds 分别达到 median/min Spearman ≥0.95/0.8、median/min top30 ≥0.75/0.5，以及 p50 scoring 降低≥20%。四臂速度门通过，但四个质量条件全部失败：各臂最差 top30 都是0，最差 Spearman 为负。停止依据是 dev 的 `NO_DEV_ARM_QUALIFIED`，没有借 test 再选模型。

## 预先约定的 test 诊断

Dev selection 写入后才生成 test rows。下面是预先约定的全臂诊断，不是通过选择门后的 confirmatory replacement test。

| 臂 | Spearman | top30 overlap | native scoring p50 / ms |
|---|---:|---:|---:|
| scalar | 0.545 / 0.536 | 0.400 / 0.333 | 0.388 / 0.387 |
| scalar_aux | 0.479 / 0.461 | 0.300 / 0.300 | 0.399 / 0.399 |
| projected_tail | 0.433 / 0.558 | 0.300 / 0.367 | 0.556 / 0.556 |
| full_latent | 0.522 / 0.553 | 0.333 / 0.333 | 0.331 / 0.329 |

Own/donor 分组也没有隐藏一个可接受的 projected_tail 替换：其 own-goal median Spearman 为0.458/0.658、top30为0.367/0.400；donor分别为0.429/0.534和0.267/0.367。Donor goals 是 teacher query 泛化诊断，不能假设物理可达。

Tail signal 为 PASS：在 exact teacher projected y 的 oracle 条件下，加入 learned tail 相比 exact projected-only 的 paired episode top30 delta median 为 +0.0583/+0.0917，两个 seeds 均改善7/8 episodes。它说明 tail head 学到了一部分会影响候选选择的信息。这个 oracle 付了 teacher 成本，没有部署计时，而且其全bank Spearman median 只有0.789/0.769、top30只有0.583/0.600，仍不够质量门；因此不能把失败全部归因于 learned y。

Structural signal 为 FAIL：fully learned projected_tail 相比 scalar_aux 的 paired episode top30 delta median 为 −0.0583/+0.0250，分别改善1/8和4/8 episodes，未达到冻结的+0.05及≥6/8门。Native p50 比 scalar_aux 慢39.4%/39.5%。这不构成结构化输出的稳定独立收益，也不意味着 scalar 在所有指标和所有 seeds 上严格支配它。

部署参数量 scalar/scalar_aux=243,685、projected_tail=251,909、full_latent=177,828；scalar_aux 训练期367,110。33个输出没有让 structured predictor 比192维 full-latent predictor 更小或更快，goal-conditioned tail head 与额外 scoring 操作的成本必须实际计算。

## 有效性、计时和限制

官方 criterion 与直接 terminal 平方距离一致：训练32,768、dev14,400、test14,400条 own-goal costs 均 max error=0。Projection/self checks、finite training、step1500、episode/goal来源隔离均通过。训练 FP64 exact decomposition max error=4.55e-13；实际 FP32 recovery max error=1.22e-4，relative max=5.73e-7，通过冻结的 atol=1e-4、rtol=1e-5；FP32 orthogonality max error=1.19e-7。没有工程错误被误判成研究 NO-GO。

计时从 cached H1 context/goal＋300 normalized H5动作到最终scores，包含前后CUDA同步的CPU wall time；每split取前6个不同row的bank0/own-goal，各臂warmup3、每block10 repeats，平衡运行顺序，两seeds单独计时。Test teacher p50为20.219/20.219 ms。这里只比较 reference 官方五步predict＋criterion，缓存的是输入latent/goal，未应用within-rollout iteration cache。快速的近似输出在质量失败后不构成planner加速或teacher replacement。

未测 encoder、adaptive CEM、first action 或 closed-loop success。候选来自logged-action Gaussian banks，未覆盖method-induced adaptive proposals。沿用原teacher/dataset预处理，全数据scaler可能接触evaluation episodes；仅声明本run的episode与goal来源隔离，且不声明全项目所有旧实验从未接触这些episodes。Stage是无.git的source archive，commit仅为历史declared provenance，不能声称本次重新核验HEAD。

## 执行证据与结束状态

PBS `F`、`Exit_status=0`；wrapper和runner exit均为0，`run_summary=PREDICTOR_LEVEL_COMPLETE`、`stage1_gate=NO_GO_STOP`。PBS `Stageout_status=1` 保留为调度层输出stageout异常；实验自行写入scratch的报告、退出文件及job.log已成功取回，不能把它隐藏或解释为训练失败。每15秒的GPU利用率/显存记录见job.log，采样最高43%和759 MiB，并非保证捕获瞬时peak。

所有模型/数据I/O、PCA、训练、标签生成与计时在guarded PBS allocation执行；本机只处理源码和小报告。没有重提交、参数扫、延长训练、banked reset或新安装。`gpt-6-luna` live monitor已终态交付并结束；没有创建定时heartbeat或宣称可指定模型的自动唤醒闭环。

证据：[冻结协议](PROTOCOL.zh.md)、[FREEZE](FREEZE.json)、[run_summary](artifacts/25568103.pbs101/run_summary.json)、[selection](artifacts/25568103.pbs101/selection.json)、[stage1_gate](artifacts/25568103.pbs101/stage1_gate.json)、[退出状态](artifacts/25568103.pbs101/job_status.txt)、[job.log](artifacts/25568103.pbs101/job.log)。Checkpoints和rows留在compute storage，未取回。

停止本次recipe；保留tail的部分学习信号作为诊断。没有足够证据将Goal-query compression升级为已成立的研究gap或有效加速方法。
