# 等 future NMSE 对照：完整结果

结论：本轮没有通过预先冻结的推进门槛，不启动后续时间片替换或训练配置搜索。Action+latent 的 first10 平均优势在控制 future NMSE 后大幅缩小，而且不稳定；目前不足以声称 latent protection 稳定修正了某种超出总体误差大小的有害 drift。

## 假说与对照

问题是：action+latent future 的 action 收益，是否超出“future 总体误差更小”所能解释的部分？

- T：同输入、同 sampler seed 的原始 BF16 teacher future。
- A：action-only epoch2/16 updates 的 future。
- B：action+latent epoch2/16 updates 的 future。
- C：尽量保留 A 相对 T 的误差方向，调整其幅度，使每个 context 的实际 BF16 future NMSE 匹配 B。概念上为 `T + alpha * (A - T)`；BF16 离散化修正的记录和边界见 README。

三组使用同一整套原始 BF16 reader，包括 video prepare/prefill、ActionDiT 和 proprio；每次重新生成 K/V。观察帧 frame0、输入、action 初始噪声和 sampler 保持一致。Alpha 和 BF16 舍入修正仅根据 future 误差确定，没有查询 action 后选择对照。

完整保留原来的 60 contexts：8 calibration、4 selection cases × 2 seeds、22 test cases × 2 seeds。下面 test 结果先平均同一个 case 的两个 seeds，再对 22 cases 等权平均；不是将 60 contexts 当作 60 个独立样本。

## Test 结果

数值越低越好，action 误差相对原始 BF16 teacher action。

| 指标 | A：action-only | C：A 的误差幅度匹配 B | B：action+latent |
|---|---:|---:|---:|
| Future NMSE | 0.093010999 | 0.071786295 | 0.071786945 |
| First10 motor RMSE，主指标 | 0.041243219 | 0.036074160 | 0.035083670 |
| Full32 motor RMSE | 0.052907799 | 0.039701241 | 0.043542794 |
| Late22 motor RMSE | 0.056283144 | 0.040488611 | 0.045684393 |
| First10 gripper RMSE | 0.001630140 | 0.001858871 | 0.001868341 |

仅缩放 A 的误差得到 C，first10 motor 平均 RMSE 就下降了 0.005169059，16/22 cases 改善，中位数差为 -0.002652972；逐一移除任意一个 case 后的平均差仍为负。原来 B 相对 A 的平均改善为 0.006159549，13/22 cases 改善。因此，这个 teacher-assisted 对照已经重现了原来平均收益的很大一部分。

关键的新比较是 B−C：

| 预先冻结的推进条件 | 完整 22 test cases 的结果 | 是否通过 |
|---|---:|---|
| First10 motor RMSE 平均差 < 0 | -0.000990490 | 是 |
| 逐 case 差值的中位数 < 0 | +0.000042169 | 否 |
| 至少 12/22 cases 改善 | 11 改善、11 变差 | 否 |
| 移除任意一个 case 后，平均差仍 < 0 | 范围 -0.002209508 至 +0.000594217 | 否 |

最后一行是 leave-one-case-out 敏感性范围，不是 confidence interval。推进门槛是工程决策规则，不是统计显著性检验。完整结果未通过，不能因为平均值略好就改用较弱的推进标准。

Secondary metrics 也有明显 tradeoff：B 相对 C 的 full32 motor 平均 RMSE 增加 0.003841553，late22 增加 0.005195782；两项均为 9/22 cases 改善、13/22 变差。不能将 first10 的微小平均优势概括成整个 action horizon 都更好。

## 可以如何解释

本轮支持较保守的解释：在这个已见 cohort 上，latent protection 的原有 action 平均收益，很大一部分可以通过总体 future 误差幅度的改变来重现。当前结果没有提供稳健的额外证据，支持 B 在相同 NMSE 下具有普遍更好的误差结构。

这不等于证明 action 只由 NMSE 决定，也不等于 latent protection 没有效果。B 和 C 在各个 cases、action horizons 上仍存在不同影响；原来 B 相对 A 的改善也仍然存在。区别在于，现在还不能把这种改善上升为“发现并修复了某一类特定的有害 world-representation drift”。

C 使用 teacher future 构造，是机制诊断对照，不是可直接部署的方法。本轮没有新 future generation、训练或 closed-loop episodes；没有验证真实未来准确度、独立泛化、native 低位 kernel 或 VQ 相对 scalar quantization 的优越性。

## 完成证据与资源

- 所有 60 contexts 的实际 BF16 注入均通过逐 context NMSE 匹配；最大相对差 0.009923824%，小于冻结门槛 0.01%。最小误差方向 cosine 为 0.999918561。18 个需要放大 A 误差的 contexts 也全部保留。
- 完整 180 条逻辑记录：120 条复用的 A/B reader 输出，加 60 条新 C 输出。CPU 分析从保存 action 重算指标并核对原记录。
- 新增 reader queries 共 68 次：60 个 C 和 8 次首尾 A/B 重放。8 次重放均逐数值完全一致，max_abs=0。
- GPU shard0 `25852095.pbs101`：F / Exit 0 / PIPELINE_COMPLETE，30/30 contexts，walltime 00:20:21。
- GPU shard1 `25852096.pbs101`：F / Exit 0 / PIPELINE_COMPLETE，30/30 contexts，walltime 00:19:58。
- CPU preparation `25851967.pbs101`：F / Exit 0 / PIPELINE_COMPLETE，walltime 00:05:28。
- CPU analysis `25852727.pbs101`：F / Exit 0 / PIPELINE_COMPLETE，walltime 00:05:56。
- 较早的合成自检作业 `25851867.pbs101` 以 Exit 1 结束，在真实 donor 准备和 reader queries 前停止。修复记录保留在 README 与 prepare_selfcheck_repair.json，失败作业不计为完成结果。
- 每个 GPU 作业上限 2 小时，运行期间每 15 秒将 GPU 利用率和显存采样写入 job.log。所有数值工作均在实际 PBS compute allocation 内执行。
- 监控 subagent 按约 10 分钟间隔只读检查；所有本轮作业验收后已停止，无后续作业待提交。

机器可读完整结果见 [analysis.json](retrieved/results/analysis/analysis.json)，逐 context 记录见 [paired_contexts.csv](retrieved/results/analysis/paired_contexts.csv)。Calibration、selection、全部 secondary metrics 和逐 case 差值均保留在完整结果中。
