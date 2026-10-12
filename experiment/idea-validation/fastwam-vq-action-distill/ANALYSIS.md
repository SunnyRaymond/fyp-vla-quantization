# Fast-WAM VQ final-action distillation：完整结果

本次真实 final-action distillation 跑完了。Calibration 和 original test cases 有改善，但整个 frozen test cohort 的平均 motor 误差略有退化，当前 checkpoint 不应替换初始 `row_output_vq_a4`。这说明本次配置没有获得整体净增益，不能据此判定 distillation 普遍无效。

## 结果与分母

全部数值是相同 observation、相同 sampler seed 下，相对 cached BF16 teacher 的 normalized action RMSE，越低越好。Test 的 22 个 cases 每个有两个 sampler seeds：先在 case 内平均两个 RMSE，再对 cases 等权平均。44 contexts 是计算覆盖量，不是 44 个独立样本，更不是 44 个 closed-loop episodes。

| 指标 | 初始 VQ | Distillation，epoch 2 | 相对变化 |
|---|---:|---:|---:|
| Test：前 10 步 motor，22 cases / 44 contexts | 0.057256 | 0.058110 | +1.49% |
| Test：前 10 步 gripper | 0.01811027 | 0.01810996 | 基本不变 |
| Test：完整 32 步 motor | 0.066828 | 0.069572 | +4.11% |
| Test：后 22 步 motor，补充诊断 | 0.069300 | 0.073342 | +5.83% |
| Calibration：前 10 步 motor，8 cases / 8 contexts | 0.046706 | 0.044329 | −5.09% |

Calibration 的 8 个 cases 中，7 个改善、1 个退化；其 full32 motor 平均误差也有小幅改善，0.061930 → 0.061159。训练确实吸收了部分校准信号。

Selection 只用于选择 checkpoint，不能代替 test：

| Selection checkpoint | 前 10 步 motor RMSE |
|---|---:|
| Epoch 0，初始 VQ | 0.042646 |
| Epoch 1 | 0.049628 |
| Epoch 2 | 0.041507 |

Epoch 2 按既定规则胜出：选择集分数低于初始值，且 test 查询之前已经冻结。第一轮变差、第二轮回落，说明选择分数并非随训练单调改善；这个现象本身不能证明学习率过大或过拟合。

## 为什么多数 case 改善，平均值仍退化

前 10 步 motor 的配对结果是 **13/22 改善、9/22 退化**；case RMSE 的中位数从 0.049646 降至 0.045088，配对差值的中位数也小于零。可是较大的退化抵消了多数改善，所以 primary mean 仍变差。中位数改善不能取代预先使用的 case-mean 指标。

| Case | 域与扰动 | 初始 first10 motor | Distilled first10 motor | 相对变化 |
|---|---|---:|---:|---:|
| 1016 | Plus / robot | 0.098388 | 0.169303 | +72.08% |
| 1015 | Plus / camera | 0.050859 | 0.071612 | +40.81% |
| 1025 | Plus / background | 0.103292 | 0.114219 | +10.58% |
| 1017 | Plus / robot | 0.115799 | 0.085322 | −26.32% |
| 1022 | Plus / light | 0.062944 | 0.043425 | −31.01% |

Case 1016 尤其值得定位：full32 motor 从 **0.090051 → 0.316298**，后 22 步 motor 从 **0.085086 → 0.363706**。但同为 robot 扰动的 1017 明显改善，所以不能把结果写成“robot 域全部失效”。1015 的 first10 退化而 full32 改善，也说明不同 horizon 的误差不能混为一谈。

不会删掉 1016 重报 primary mean。这是实际需要保护的失败输入，删除它只能隐藏问题。

![配对结果、逐 case 变化和 selection 曲线](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-vq-action-distill/analysis/paired_recovery.png>)

左图：虚线下方表示改善。中图：正值表示退化。右图：checkpoint 只按 selection 选择，曲线不包含 test 调参。

## Selection 的域覆盖限制

整个 selection 与 test 都在 **LIBERO Spatial**；差异在 original 与 LIBERO-Plus 扰动域，并不是另一个 benchmark suite：

| 范围 | Cases | 初始 first10 motor | Distilled first10 motor | 相对变化 |
|---|---:|---:|---:|---:|
| Original test | 10 | 0.048122 | 0.045234 | −6.00% |
| LIBERO-Plus test | 12 | 0.064867 | 0.068841 | +6.13% |

Selection 的 4 个 cases **全部为 original**，没有 Plus case。它能发现 original 上的改善，却缺少扰动域的直接反馈。本次“original 改善、Plus 平均退化”与这个覆盖限制相容，但 subgroup 分析是看完结果后的探索，不能据此断定退化原因。未来需要在训练之前冻结一个覆盖 original 与扰动域的 selection set；不能把当前已看过的 test cases 临时改成 selection 后继续称作 holdout。

## 本次方法与 teacher

Student 初始参数来自既有 `row_output_vq_a4`。只更新全部 614 个 target Linears 的 FP32 codebook masters 和 log row scales；forward 使用存储为 BF16 的 books/scales。VQ indices、Smooth alpha=0.5、signed Hadamard block=128、A4 absmax/qmax=7、bias 和其他模型参数保持固定。

对一个校准 observation 和固定 sampler seed，student 运行原有的 10 次 video 与 10 次 action denoising，并通过 video KV prefill 等完整路径反向传播。Teacher 是同 observation、同 seed 的缓存 BF16 最终 action，不重新运行 teacher，也不训练 teacher。实际 loss 为：

\[
L_{\mathrm{KD}}(\theta)=\frac{1}{60}\sum_{t=1}^{10}\sum_{d=1}^{6}
\left(a^{\mathrm{student}}_{t,d}(\theta)-a^{\mathrm{BF16}}_{t,d}\right)^2.
\]

因此，这是实际最终 action 的 distillation；本次 loss 中没有随机投影 proxy，也没有局部 Linear-output loss。它没有直接约束后 22 步或 gripper。A4 及 BF16 decoder 使用 STE；STE 自检只确认实现符合设定，不证明 surrogate gradient 在任务上完全准确。

总共 8 个 calibration cases × 1 seed × 2 epochs = **16 updates**。另有 2 次 epoch selection × 8 contexts、8 次 final calibration 和 44 次 test，共 **68 个评估 contexts**。16 次更新都记录了 **614/614 connected、finite、nonzero targets**。最大 global gradient norm 为 0.028428，低于 clip 阈值 1.0，记录没有显示梯度裁剪触发。

存储后的 BF16 参数确实变化：**683,483 个 book scalars** 和 **340,375 个 row scales**。master 参数总量分别为 1,257,472 和 2,073,799；row-scale 总量包含固定 inactive/zero rows，不能直接用它推断有效参数更新比例。综上，当前负结果不能解释为“梯度没接通”或“BF16 舍入导致所有更新消失”。

## 已知事实与仍待验证的机制

已经验证：校准拟合改善；selection 只覆盖 original；test 少数大退化抵消多数改善；未监督的 late horizon 平均误差更高；参数和梯度确实生效。

仍是假设：学习率 5e-4 太大、8 个校准状态覆盖不足、共享 codebook/scale 更新对不同输入有冲突、固定 assignments 限制优化空间、STE 梯度存在任务相关偏差、只监督 first10 使 late horizon 缺少保护。上述机制不能仅凭一条训练运行和当前平均误差被区分。

有一个可直接推导的覆盖限制：本轮约 333 万个 master scalars（包含 inactive rows），而 8 个校准输入的 first10 motor teacher 只提供 8 × 10 × 6 = 480 个标量输出约束。在固定参数附近按 STE surrogate 线性化，校准输出矩阵为 \(J_{\mathrm{cal}}\in\mathbb{R}^{480\times P}\)，因此 \(\operatorname{rank}(J_{\mathrm{cal}}^\top J_{\mathrm{cal}})\le480\)。这个 Gauss–Newton 近似中的低秩结构不是精度退化的证明，也不保证训练会走入某个特定坏方向；它说明少量最终 action 配对本身无法约束全部参数方向。另一个 observation 的 action-sensitive directions 可能没有被这些输入覆盖。重复两轮相同输入没有增加输入覆盖；增加 teacher 覆盖、限制相对初始 checkpoint 的漂移，或补充未监督 horizon，分别针对这个问题的不同部分。

因此不建议马上加很多 epoch 或同时改变 transform、indices、codebook、loss 与数据范围。本轮结果最直接支持调查的是 **tail protection 与扰动域覆盖**。

## 下一步的有界比较

推荐先测试一个额外的 late-horizon teacher 约束，保留实际执行的 first10 motor loss 权重：

\[
L'=L_{\mathrm{first10,motor}}+\lambda_{\mathrm{late}}L_{\mathrm{last22,motor}},
\qquad \lambda_{\mathrm{late}}=0.1.
\]

0.1 是待验证的初始实验选择，不是从当前结果证明的最优系数。这一次只新增该项，不同时调 learning rate、重分配 indices 或学习 rotation；使用同一 frozen cohort、seeds 和 selection 规则。这样能检验 late supervision 是否降低 tail 退化，以及是否伤害 first10 指标。它不会自动修复 first10 的 robot-variant 退化；如果实验没有帮助，应再单独研究校准覆盖或更新幅度。

若尽量复用本轮执行历史，可从同一个已保存的第 1 次更新后 master 起跑，采用与本轮相同的 Adam 重置，再比较剩余 15 次更新；此时第 1 次 prefix-only 更新是两个 arms 的公共 warm start，需要明确记录。不要把“只更换 loss”与“另一个初始化/另一条 Adam 历史”混成一个因素。本建议尚未提交新的训练任务。

长期更贴近“保护敏感方向”的目标，可以在 final-action metric 中加 task-relevant channel weights 或 tail-risk term，再用 teacher loss 优化共享 codebooks。但必须单独测方向/扰动下的敏感性；某个 case 的量化误差很大并不等同于已经测出了 Jacobian 的敏感方向。

## 执行与证据边界

`25755802.pbs101` 首次 timing-guard 退出，保存了第 1 次更新后参数，但没有 Adam state。`25756079.pbs101` 明确重置 Adam 一次后继续，在 01:58:47 保存 16 updates/2 epochs/20 evaluation contexts 的 resumable checkpoint。`25764781.pbs101` 恢复完整 checkpoint 后在 01:03:57 完成剩余评估。新段没有重复已完成的 calibration/selection 评估。

本轮 cached teacher inference queries = **0**。CPU 分析 `25765002.pbs101` 使用旧 actions/checkpoint，运行 **00:01:17**：没有重新加载 Fast-WAM 做推理，也没有新增训练。数值分析全部在 PBS CPU allocation 内完成；本地只读取小型结果和编写报告。

完整实验门槛已经核实：PBS F、Exit 0、`PIPELINE_COMPLETE`、匹配 job 的 complete summary、614 targets、2 epochs/16 updates、22 cases/44 test contexts、`test_queries_before_freeze=0`。CPU 分析也有对应 Exit 0 与 completion marker。

本次只支持已有 cohort 的 **paired action-fidelity recovery diagnostic**：test inputs/results 以前已经看过，不能作为新的独立泛化证据；episodes=0，没有 closed-loop success；计算使用 decoded BF16，没有 native W4A4 kernel 或端到端加速结论；只有一条训练运行，没有训练随机性重复实验。由于 first-step legacy Adam reset，本轮还不能声称完全等同于连续未中断的 16-step Adam 运行。

来源：`full_terminal_evidence.json`、`analysis/analysis.json`、`analysis/paired_cases.csv`，以及远端各段原始 checkpoint/summary。数值汇总交叉核对了保存 action 的 first10 RMSE 和原 summary 的全部 test means。

分析流程参考：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065)。使用 `statistical-analysis` 的配对单位与探索/验证边界指导；采用标准库进行描述性计算，没有进行显著性检验或安装新的统计环境。
