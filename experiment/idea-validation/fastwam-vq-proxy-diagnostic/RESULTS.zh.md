# Fast-WAM action proxy 诊断：第一步结果

第一步 CPU 作业 `25749368.pbs101` 已成功完成，PBS 为 F、Exit_status 为 0、walltime 为 3 分 7 秒；对应 exit receipt、PIPELINE_COMPLETE 和匹配 job/phase 的 summary 均已核对。四项数值自检通过。范围是八个原 calibration contexts、每个原 sampler seed index 0、四套冻结 banks，共 32 条保存的 action。没有重跑 student，也没有重新拟合 codebook。

**目前最重要的结论：原四个方向上的真实误差没有随着 proxy 一起消失。因此，当前失配不能只用“另外 56 维没有被 probes 覆盖”解释。**

| 冻结版本 | 完整 60 维 motor RMSE | 训练日志 proxy RMS | 原四方向上的真实投影 RMS | 新 16 方向上的真实投影 RMS | 原四方向 span 内的真实误差能量占比 |
|---|---:|---:|---:|---:|---:|
| Row-scale | 0.05215 | 0.27232 | 0.05273 | 0.05248 | 8.25% |
| 局部输出拟合 | 0.04671 | 0.23510 | 0.04652 | 0.04914 | 7.50% |
| Action-sensitive λ=0.1 | 0.04851 | 0.00192 | 0.04963 | 0.04592 | 8.44% |
| Action-sensitive λ=1 | 0.05145 | 0.00096 | 0.03958 | 0.04935 | 4.80% |

各列均先在 case 内计算，再对八个 cases 等权平均。新方向是四组各四个 unit-norm Rademacher probes；它们是同一 case 内的 Monte Carlo 重复，不是 32 个独立场景。投影使用同一 normalized action 单位，与原训练 proxy 的投影定义一致。旧 probes 不保证正交，span 能量使用 SVD 正交基计算，不能直接将四个投影的平方相加当成子空间能量。

对 λ=0.1，真实原四方向的 RMS 与完整 motor RMSE 大致同量级，且比局部输出拟合的对应真实投影更高；但训练日志的 proxy 几乎归零。表中真实投影 RMS 均值与 proxy RMS 均值之比约为 26。这是“已观察的方向也预测不准”的直接证据，而非仅仅未覆盖方向的风险。

λ=1 的真实原四方向 RMS 比局部拟合低，但完整 motor RMSE 更高，原 span 能量比例也更低；这与部分方向取舍相容。不过它的真实原四方向 RMS 仍约为 proxy 的 41 倍，所以方向覆盖不足也不能单独解释该版本。

Row-scale 和局部输出拟合的 proxy 本身又明显大于实际对应投影。失配已存在于敏感度拟合之前；拟合进一步将该 proxy 压低，也没有保证真实 action 同步改善。现阶段仍不能区分 N/n token-row 采样误差、teacher 一阶近似失效和采样下的误差抵消各占多少。

**第二步尚未完成。** GPU preflight `25749380.pbs101` 已实际运行并以 F/Exit_status=1 结束，walltime 为 16 分 14 秒。Case0 的 20 probes、614 个目标和 bank 投影已完成，但新 capture 的原 proxy 复现检查未通过。BF16 baseline replay RMSE 为 0，6,436 次调用的原 row IDs 均匹配。失败记录和逐 case 结果已保留；这不是完整八个 cases 的通过结果。

PBS CPU 比较作业 `25754337.pbs101` 为 F/Exit_status=0，exit receipt 为 0；其 compute-node 日志表明：6,436 次调用中原选中 rows 的 inputs、teacher outputs 全部一致，6,411 次调用的 adjoints 存在差异，最大绝对差为 0.001953125。该诊断脚本没有产生 PIPELINE_COMPLETE 或标准 summary，依据是匹配该作业的 PBS 终态、exit receipt 和日志；不能宣称已通过不存在的 marker。

原 full `25750571.pbs101` 在启动前被 hold 并取消。共享修订使用冻结原 capture 与原 LayerData 算法严格复现原训练 proxy，保留新 capture 独立计算四个 factorial 组合，不放宽原容差。补充报告冻结原 signed proxy 对真实 action 的预测误差，以及原 probes、原 rows 下重采集 proxy 对冻结 proxy 的 signed drift。后一个量同时包含 capture 差异和 evaluator 的数值求和顺序差异，不能直接当作纯梯度噪声的大小。

修订后的 preflight `25754358.pbs101` 已严格核实通过：PBS F/Exit_status=0、job exit receipt=0、PIPELINE_COMPLETE 和匹配 job/phase 的 complete summary 均齐全，walltime 为 16 分 35 秒。覆盖 case0、全部 614 个目标、4 old+16 new probes、两套各2 rows；冻结原 capture 的四个 arm proxy 均在原容差内复现。Signed source/replay 报告已实际产生，job log 持续记录了 A100 40GB 的利用率和显存。新 capture 的逐元素梯度一致性没有被声明通过。

完整八个 cases 的 full 为 `25754359.pbs101`，最近由实时 PBS 确认为 Q，comment 为 g1 用户 running-job limit。Full 申请 1 GPU、12 小时，无需等待独立 preflight 完成。新旧 probes、rows 仍保持现有 books、indices、scales、Smooth、activation、输入与 seeds 不变；结果目录按 job ID 隔离。关联监控继续使用 `gpt-6-sol / xhigh` 每 30 分钟跟进待完成任务，并获授权修复共享问题、保留失败记录后以新 attempt 重提。

失败 attempt 留下的 case0 数值可作为有明确边界的初步诊断。对 λ=0.1，在同一个新 capture 内保持原四个 probes 不变，旧 rows 的 proxy RMS 为 0.001501，新 rows 为 0.276613；真实相同四个方向的投影 RMS 为 0.030332。保持旧 rows 换为四组新 probes，proxy RMS 依次为 0.152584、0.211552、0.156059、0.070250。这支持对固定 rows 和固定 probes 的强依赖，但还不能用一个 case 推定八个 contexts 的总体表现，也不能声称已证明改进方法有效。

成功 preflight 的 case0 对照再次出现同类模式。下表都是该 case 的 proxy RMS；new probes 列对四组各四个 directions 的 RMS 等权平均，不是四个新场景。

| 冻结版本 | Old probes / old rows | New probes / old rows | Old probes / new rows | New probes / new rows |
|---|---:|---:|---:|---:|
| Row-scale | 0.42002 | 0.50104 | 0.31704 | 0.18125 |
| 局部输出拟合 | 0.34960 | 0.44616 | 0.25962 | 0.17027 |
| Action-sensitive λ=0.1 | 0.00174 | 0.14697 | 0.27849 | 0.17732 |
| Action-sensitive λ=1 | 0.00216 | 0.13705 | 0.27681 | 0.17214 |

λ=0.1 的原四个实际 action 投影 RMS 为 0.030332；冻结原 capture 的 proxy RMS 为 0.001963，signed prediction error RMS 为 0.031729。新旧 capture/evaluator 的 signed proxy drift RMS 为 0.002141，虽然与这个很小的训练 proxy 同量级，但远小于换 rows 后的 proxy RMS。数值漂移没有解释掉这里对采样的强依赖，也没有令原四个方向上的真实 action 误差消失。

同一 case 的原 probes、原 rows 下，λ=0.1 的“各模块投影平方之和的均值”为 0.08401，row-scale 为 0.04206；λ=0.1 的全局 proxy 却更小。这与跨模块抵消相容，不能直接将该量当真实 action MSE。有效的一阶抵消本身允许；结合换 rows/probes 的结果，目前应进一步检查这种抵消在新采样下是否稳定，而不是认定所有抵消都有问题。

成功 preflight 的原始摘要见 [summary.json](retrieved/artifacts/25754358.pbs101/summary.json)，signed 逐 case 记录见 [case_0.json](retrieved/results/preflight-25754358.pbs101/case_0.json)，终态证据见 [preflight_terminal_evidence.json](preflight_terminal_evidence.json)。完整八个 cases 仍未完成。

CPU-only 差分定位 `25755155.pbs101` 已完成：PBS F/Exit_status=0，walltime 13分15秒，exit receipt=0、PIPELINE_COMPLETE 和匹配 job/phase 的 complete summary 齐全。覆盖成功 preflight 的 case0 和全部614目标；四项内置自检、三套 banks × 四种采样组合的12项 CPU/GPU 差分核对均通过。没有重新拟合 codebook、加载模型或重跑 BF16/action。监控 `fast-wam-proxy` 已恢复并重新核实为 `gpt-6-sol/xhigh`、每30分钟；CPU 结果由原任务处理后，仅完整 full 仍待完成。

### CPU 差分定位：修正抵消原误差，不等于修正之间相互抵消

本次量的是每个冻结 bank 相对局部输出拟合 bank 的变化，而不是该模块相对 BF16 的绝对误差。令 d_l 为模块 l 的权重差经量化输入和 teacher adjoint 得到的方向投影；同一模块的各次调用先相加。净变化 RMS 为 RMS(sum_l d_l)，跨模块能量比为 mean((sum_l d_l)^2) / sum_l mean(d_l^2)。比值小于1说明这些变化净抵消，大于1说明净增强；它不是百分比，也不是精度指标。

| Action-sensitive bank | Probes / rows | 净变化 RMS | 跨模块能量比 | 模块合并后 / 逐调用能量比 |
|---|---|---:|---:|---:|
| λ=0.1 | old / old | 0.35106 | 2.683 | 1.838 |
| λ=0.1 | new / old | 0.35613 | 1.735 | 1.596 |
| λ=0.1 | old / new | 0.04125 | 0.420 | 0.697 |
| λ=0.1 | new / new | 0.08196 | 0.590 | 0.824 |
| λ=1 | old / old | 0.35068 | 2.528 | 1.800 |
| λ=1 | new / old | 0.36920 | 1.790 | 1.554 |
| λ=1 | old / new | 0.04955 | 0.400 | 0.813 |
| λ=1 | new / new | 0.09023 | 0.649 | 0.860 |

new probes 的 CPU RMS 将16个方向合并计算，前面 GPU proxy 表则先计算四组 RMS 再取平均，二者不能直接混用。逐调用比值涉及所有重复调用，不应一概称为跨 denoising steps。

在训练 rows 和训练 probes 上，λ=0.1 的各层变化整体增强，净修正 RMS 为0.35106。相同 capture 中局部输出拟合的绝对 proxy RMS 为0.34960，action-sensitive 为0.00174；由于投影可线性相加，整体修正近似对消局部拟合原本的 proxy 误差。这支持“组合修正补偿基准误差”，不支持“各层修正互相抵消”。此前绝对模块误差能量与全局 proxy 的比较和此处 bank 差分比较衡量不同对象；两者并不矛盾。

保持原 probes，换 rows 后，同一组权重改动的净投影仅为0.04125，并出现变化之间的净抵消；此时绝对 proxy 从局部拟合的0.25962变为 action-sensitive 的0.27849，没有改善。换 probes 而保留 rows，修正仍大，但最终绝对 proxy 为0.14697，说明修正大小不能保证在新方向上正确补偿。λ=1 呈现相同结构；仅增加 λ 未解决这一单 case 的采样依赖。

λ=0.1 在 old/old 下变化能量最大的模块依次为 video_expert.time_embedding.0、text_embedding.0、time_embedding.2、time_projection.1、text_embedding.2。换到 new/new 后，最大项变为 time_projection.1，随后是 time_embedding.0、blocks.10.self_attn.k、time_embedding.2 和 blocks.22.ffn.0。这里定位的是投影变化大的模块，不是绝对误差最大的模块，也不能据此直接判定这些模块必须保留高精度。

下一版受限候选优先考虑：在相同存储预算下增加训练 token rows 的代表性，并在拟合中刷新 rows，保留独立 held-out rows/probes；同时给局部输出重建保留约束，避免为少数采样点形成过大的补偿。首先单独比较 rows 改动，再决定是否增加局部约束，避免同时改多个因素。真实 action 仍需验证，因为现有一阶 proxy 对原方向的真实误差也存在失配。不能仅凭这次差分结果把逐层禁止抵消作为主要目标，也不应直接继续加大 λ。

这是一个 case、冻结 banks 的诊断，不是新方法优于基准的证据。完整八个 cases 的 `25754359.pbs101` 仍为 Q，实时 comment 为 g1 running-job limit；尚不能给出跨 contexts 的最终机制结论。小摘要见 [CPU summary](retrieved/artifacts/25755155.pbs101/summary.json)，大模块记录留在 compute storage。

下一版 codebook 目标暂不定案。若换 rows 后 proxy 明显反弹，优先处理固定 rows 下的过拟合和估计方差；若换 probes 后反弹，处理方向采样依赖；若两种采样下都无法预测对应真实 action，优先处理 teacher-path 一阶近似失配，考虑真实 student trajectory 的 codebook 参数拟合，而不是仅继续增加 λ 或 probes。

本结果仅用于固定输入 calibration 的机制诊断，不是新方法效果、test 选型、closed-loop 成功率或 native VQ 延迟结果。

原始小摘要见 [summary.json](retrieved/results/original/summary.json)，逐 case 数值见 [per_case.json](retrieved/results/original/per_case.json)，终态证据见 [original_terminal_evidence.json](original_terminal_evidence.json)。
