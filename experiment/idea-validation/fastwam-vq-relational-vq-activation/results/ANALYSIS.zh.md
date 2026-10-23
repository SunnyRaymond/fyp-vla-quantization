# VQ relational KL + activation 固定输入结果

## 主要结果

在固定 C8/8 VQ updates 与 C80/80 activation updates 的预算下，两个阶段都改用 soft-bin + relational KL，没有超过当前的 MSE-VQ8 + relational-activation80。最终模型在完整 C80 和原 held-out 上的八项指标均较当前 relational 参照变差。Held-out composite 为 0.0061686663（+7.40%）、first10 motor RMSE 为 0.034438064（+9.54%）、future latent NMSE 为 0.057364387（+5.65%）；composite 有 25/40 cases 变差，median paired delta 也为正。

这次差距同时出现在四项 KL、统一 objective 和 RMSE/NMSE 上，不能只归结为不同评价指标的排序错配。C80 composite 也增加 12.68%，所以仅用 held-out 过拟合解释全部退化不足。单条训练轨迹不能确定具体原因；当前证据只支持这套冻结系数、学习率与 8-update VQ 预算下的替换效果较差，不能推断 KL 普遍不适合 VQ。

这里的 C80 是 activation 的完整训练 cohort；VQ 仍只用其中原 C8 的 8 个 context。因此，C80 也退化不能排除 VQ 阶段的 C8 过拟合，只能说明差距并非仅发生在 activation 从 C80 到 held-out 的泛化环节。当前结果也不能区分 VQ 的欠优化、C8 过拟合或 loss/optimizer 标定不适配。

上游 VQ-only 的同预算对照已显示差距：KL-VQ8 的 held-out RMSE/NMSE 较 MSE-VQ8 高 10.33%/12.32%。后续 activation KL 仍有效：相对新 VQ-only，held-out composite 降低 32.86%、RMSE 降低 29.54%、NMSE 降低 16.10%，但没有追上当前参照。VQ 阶段只有 8 次更新，且系数沿用原 activation 问题的标定；更新预算、优化设置与不同参数化的梯度分配是否适配，仍是待验证的原因，不能由本轮直接区分。

当前保留 MSE-VQ8 + relational-activation80 作为这些固定输入指标上的 reference。本轮不根据 held-out 结果调整参数或追加训练。

本实验只测 fixed-input teacher/student 输出，不执行环境闭环。原始共享 teacher 全部复用；新增 teacher query 为 0。每个新 arm 在 C80 calibration 80 cases 和 held-out 40 cases × 2 seeds 上评估，旧 softbin/relational 端点直接复用。

公平性约束：四组按同一 case、seed、plan_row 和 sampler_seed 配对；先在 case 内平均 seed，再对四个 suite 等权。held-out 的配对单位是 40 个 case，不把 80 条 query 当成独立样本。本轮新 VQ 阶段从原 C8 bank masters 开始，未套用 MSE-aware masters；旧 softbin/relational 两组的上游仍是原 MSE-aware VQ final8。

训练分两阶段：`vq_kl` 将 action+latent VQ 上游原 C8 的 8 updates 从 MSE 改为冻结 softbin 与 relational KL；`vq_kl_activation_kl` 在该 C8 KL 结果上再按当前 activation 代码训练 D/r 80 updates。目标沿用冻结定义：

VQ 只新训练一次；其 final8 同时用于 VQ-only 端点评估和 activation 初始化。本轮总新增为 8 个 VQ updates + 80 个 activation updates，两阶段顺序执行。VQ 阶段更新 codebooks/log row gains；activation 阶段冻结这些 VQ 参数，仅更新原有 D/r。

`L = αa·(action_kl + βa·action_temporal_kl) + (ρc)·αz·(latent_kl + βz·latent_spatiotemporal_kl)`

冻结值来自 source_relational/source_dual：αa=0.01230247928, αz=0.3129789522, βa=0.6196851092, βz=27.14823378, ρc=0.04245452683。温度、softbin 定义和 scaling/clipping 沿用 source_relational；没有新 dev，也没有 held-out 调参。

下表的百分比均为同一 metric 下 `vq_kl_activation_kl` 相对相应对照组的变化（负值表示下降）。`softbin_objective` 与 `relational_objective` 分别在同名 metric 内跨 arm 比较；不把两种不同 objective 的原始值互相比较。

## C80 calibration values

| metric | vq_kl | vq_kl_activation_kl | softbin | relational | vq_kl_activation_kl_vs_relational | vq_kl_activation_kl_vs_softbin | vq_kl_activation_kl_vs_vq_kl |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| action_kl | 0.15519296 | 0.08899535 | 0.071938882 | 0.07147981 | +24.50% | +23.71% | -42.66% |
| latent_kl | 0.16800515 | 0.16480344 | 0.14394428 | 0.14685875 | +12.22% | +14.49% | -1.91% |
| action_temporal_kl | 0.11762129 | 0.037776069 | 0.04180668 | 0.034701079 | +8.86% | -9.64% | -67.88% |
| latent_spatiotemporal_kl | 0.0058017689 | 0.0050700768 | 0.0050667305 | 0.0047080519 | +7.69% | +0.07% | -12.61% |
| softbin_objective | 0.0041416053 | 0.0032846683 | 0.002797668 | 0.0028307459 | +16.04% | +17.41% | -20.69% |
| relational_objective | 0.007131175 | 0.0054015826 | 0.0049441032 | 0.0047936246 | +12.68% | +9.25% | -24.25% |
| first10motor_rmse | 0.045238011 | 0.034271113 | 0.030608191 | 0.030427245 | +12.63% | +11.97% | -24.24% |
| future_latent_nmse | 0.052607889 | 0.051566234 | 0.045182512 | 0.046244474 | +11.51% | +14.13% | -1.98% |

## Held-out values

| metric | vq_kl | vq_kl_activation_kl | softbin | relational | vq_kl_activation_kl_vs_relational | vq_kl_activation_kl_vs_softbin | vq_kl_activation_kl_vs_vq_kl |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| action_kl | 0.18799386 | 0.092725382 | 0.083403036 | 0.075975663 | +22.05% | +11.18% | -50.68% |
| latent_kl | 0.22054357 | 0.18446535 | 0.17095114 | 0.17523083 | +5.27% | +7.91% | -16.36% |
| action_temporal_kl | 0.10958743 | 0.041127923 | 0.040418109 | 0.034484781 | +19.26% | +1.76% | -62.47% |
| latent_spatiotemporal_kl | 0.0086198199 | 0.0062742686 | 0.0063970953 | 0.006147169 | +2.07% | -1.92% | -27.21% |
| softbin_objective | 0.0052432353 | 0.0035918121 | 0.0032975557 | 0.0032630464 | +10.08% | +8.92% | -31.50% |
| relational_objective | 0.0091881094 | 0.0061686663 | 0.0059133057 | 0.0057434071 | +7.40% | +4.32% | -32.86% |
| first10motor_rmse | 0.048873691 | 0.034438064 | 0.03232986 | 0.031438487 | +9.54% | +6.52% | -29.54% |
| future_latent_nmse | 0.068375369 | 0.057364387 | 0.053033005 | 0.054295895 | +5.65% | +8.17% | -16.10% |

## Held-out 按 case 配对

| Comparison | Metric | Improved / 40 | Worsened / 40 | Tied / 40 | Median case delta |
|---|---|---:|---:|---:|---:|
| vq_kl_activation_kl_vs_relational | action_kl | 16/40 | 24/40 | 0/40 | +0.00456523 |
| vq_kl_activation_kl_vs_relational | latent_kl | 14/40 | 26/40 | 0/40 | +0.0143453 |
| vq_kl_activation_kl_vs_relational | action_temporal_kl | 15/40 | 25/40 | 0/40 | +0.00490756 |
| vq_kl_activation_kl_vs_relational | latent_spatiotemporal_kl | 14/40 | 26/40 | 0/40 | +0.000308597 |
| vq_kl_activation_kl_vs_relational | softbin_objective | 17/40 | 23/40 | 0/40 | +0.000169332 |
| vq_kl_activation_kl_vs_relational | relational_objective | 15/40 | 25/40 | 0/40 | +0.00027368 |
| vq_kl_activation_kl_vs_relational | first10motor_rmse | 15/40 | 25/40 | 0/40 | +0.00267123 |
| vq_kl_activation_kl_vs_relational | future_latent_nmse | 14/40 | 26/40 | 0/40 | +0.00461449 |
| vq_kl_activation_kl_vs_softbin | action_kl | 18/40 | 22/40 | 0/40 | +0.00232724 |
| vq_kl_activation_kl_vs_softbin | latent_kl | 17/40 | 23/40 | 0/40 | +0.0110293 |
| vq_kl_activation_kl_vs_softbin | action_temporal_kl | 20/40 | 20/40 | 0/40 | -0.00030502 |
| vq_kl_activation_kl_vs_softbin | latent_spatiotemporal_kl | 25/40 | 15/40 | 0/40 | -0.000401181 |
| vq_kl_activation_kl_vs_softbin | softbin_objective | 19/40 | 21/40 | 0/40 | +0.000114676 |
| vq_kl_activation_kl_vs_softbin | relational_objective | 19/40 | 21/40 | 0/40 | +0.000145666 |
| vq_kl_activation_kl_vs_softbin | first10motor_rmse | 16/40 | 24/40 | 0/40 | +0.00200266 |
| vq_kl_activation_kl_vs_softbin | future_latent_nmse | 17/40 | 23/40 | 0/40 | +0.0037654 |
| vq_kl_activation_kl_vs_vq_kl | action_kl | 38/40 | 2/40 | 0/40 | -0.0875195 |
| vq_kl_activation_kl_vs_vq_kl | latent_kl | 26/40 | 14/40 | 0/40 | -0.0184098 |
| vq_kl_activation_kl_vs_vq_kl | action_temporal_kl | 38/40 | 2/40 | 0/40 | -0.0656216 |
| vq_kl_activation_kl_vs_vq_kl | latent_spatiotemporal_kl | 32/40 | 8/40 | 0/40 | -0.00122596 |
| vq_kl_activation_kl_vs_vq_kl | softbin_objective | 38/40 | 2/40 | 0/40 | -0.00134388 |
| vq_kl_activation_kl_vs_vq_kl | relational_objective | 37/40 | 3/40 | 0/40 | -0.00260607 |
| vq_kl_activation_kl_vs_vq_kl | first10motor_rmse | 38/40 | 2/40 | 0/40 | -0.0130977 |
| vq_kl_activation_kl_vs_vq_kl | future_latent_nmse | 25/40 | 15/40 | 0/40 | -0.00610346 |

## Held-out suite 异质性

以下为主要比较 `vq_kl_activation_kl` vs `relational` 的 suite 内相对变化；四个 suite 分开呈现，不推断总体显著性。

| Metric | libero_10 | libero_goal | libero_object | libero_spatial |
|---|---:|---:|---:|---:|
| action_kl | -10.64% (7/10) | +14.61% (4/10) | +28.06% (4/10) | +58.52% (1/10) |
| latent_kl | +11.16% (4/10) | -13.76% (5/10) | +30.60% (1/10) | +0.42% (4/10) |
| action_temporal_kl | -8.66% (4/10) | +25.35% (3/10) | +16.66% (6/10) | +36.61% (2/10) |
| latent_spatiotemporal_kl | +10.45% (3/10) | -24.79% (4/10) | +38.75% (1/10) | +13.66% (6/10) |
| softbin_objective | +3.63% (5/10) | -6.85% (5/10) | +29.97% (3/10) | +18.92% (4/10) |
| relational_objective | +5.19% (5/10) | -13.95% (4/10) | +32.11% (2/10) | +17.76% (4/10) |
| first10motor_rmse | +0.36% (5/10) | +9.18% (2/10) | +18.36% (5/10) | +11.11% (3/10) |
| future_latent_nmse | +10.76% (4/10) | -12.81% (5/10) | +32.08% (1/10) | -0.79% (4/10) |

## 训练与复用证据

| Group | PBS job | VQ updates added this run | Activation updates in checkpoint | Endpoint queries | New teacher queries | Receipt / checks |
|---|---|---:|---:|---:|---:|---|
| vq_kl | 25886544.pbs101 | 8 | 0 | 160 | 0 | complete; gradient/frozen/parity checks passed |
| vq_kl_activation_kl | 25886572.pbs101 | 8 | 80 | 160 | 0 | complete; gradient/frozen/parity checks passed |
| softbin | 25884637.pbs101 | 0 | 80 | 160 | 0 | reused complete source endpoint |
| relational | 25883750.pbs101 | 0 | 80 | 160 | 0 | reused complete source endpoint |

两条新 arm 合计新增 320 个 student endpoint query；旧 softbin/relational 的 320 个 endpoint query 复用。四组共享既有 teacher 输入。每个新 arm 只有一条训练轨迹，因此 case 计数用于描述配对一致性，不代表独立训练重复或一般化显著性。

旧两组的 VQ 上游均已有 8 个 MSE updates；表中 VQ=0 只表示本轮未重训它们。两个新 endpoint 共用同一次 KL-VQ8 训练，不构成两次独立 VQ 重复。VQ 的全部 8 updates 和 activation 的全部 80 updates 均通过 614 模块的梯度连接/有限性检查；VQ books/row gains 实际改变，activation 阶段这些参数保持冻结。初始化、compensated-weight cache 与实际 infer 的一致性检查均为 max-abs 0。

## 完成与资源

三个作业均已通过 PBS F/Exit_status=0、artifact exit_code=0、PHASE_COMPLETE 和匹配作业 ID 的完整 summary。VQ 运行 00:32:34，activation 运行 01:01:15，CPU 聚合运行 00:00:31；两个 GPU 作业各申请 01:59:00，实际 GPU allocation 合计 01:33:49。排队时间不计作 GPU allocation。

两个 GPU 作业的 job.log 持续以约 15 秒间隔记录 utilization/memory；gpt-6-luna/xhigh 按十分钟监控并已结束。完整验收记录见 `../completion_evidence.json`。执行路径为 decoded BF16 VQ + dynamic fake-A4，KV 保持 BF16；本报告不提供 native low-bit kernel、显存节省或速度结论。

## 历史 MSE activation final80 参照

历史 MSE 端点 `25878493.pbs101`（80 updates）只报告它原有的四项：

| Metric | Value |
|---|---:|
| action_kl | 0.074794248 |
| latent_kl | 0.17942314 |
| first10motor_rmse | 0.031641466 |
| future_latent_nmse | 0.055377875 |

该参照没有 relational 或 composite 输出，本报告不补造这些指标。

## 同预算 VQ8 上游对照

以下复用原 MSE-aware rho1.5 C8 VQ final8、D=1/r=1 的固定输入 summary。summary 中 `updates=0` 表示没有额外 activation 更新；其 parent 是 rho1.5 C8 VQ checkpoint（8 updates）。因此这是上游 VQ 阶段自身的同预算参照，和上面的 MSE activation final80 参照分开。这里只读既有 summary JSON，没有新增 forward、训练或 held-out 选参。

source root：`/scratch/users/ntu/yguo017/fastwam-vq-c80-activation-clipping-20261017`；protocol：`fastwam-vq-c80-activation-clipping-v2-original-heldout`；PBS job：`25853617.pbs101`。

| Split | Metric | MSE-aware VQ8 baseline | VQ-KL final8 | VQ-KL vs baseline |
|---|---|---:|---:|---:|
| calibration | first10motor_rmse | 0.044146961 | 0.045238011 | +2.47% |
| calibration | future_latent_nmse | 0.054735372 | 0.052607889 | -3.89% |
| heldout | first10motor_rmse | 0.044296774 | 0.048873691 | +10.33% |
| heldout | future_latent_nmse | 0.060877371 | 0.068375369 | +12.32% |

该历史 baseline 只有这两项诊断值；没有 KL relation 或 composite 指标，本报告不补造，也不据此声称这些目标的优劣。该对照仅说明固定输入下 VQ 上游阶段的变化，不代表 activation final80 的比较。

范围限制：单次训练轨迹、冻结 held-out cohort 的 fixed-input 证据；不报告 statistical significance、未见任务泛化或 task success。

方法流程参考：本报告沿用 `experimental-design` workflow 组织冻结 cohort、配对单位与证据边界。参见 Kassis et al., *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents* (2026), [arXiv](https://doi.org/10.48550/arXiv.2609.00065)。本实验 loss 与系数来自冻结 source protocol/prepare artifacts。
