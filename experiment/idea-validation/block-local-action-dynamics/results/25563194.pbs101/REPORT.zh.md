# 第一轮受控系统实验结果

本轮完整执行 3 个 conditions × 5 个 arms × 3 个 training seeds，共 45 次训练。这里只回答受控 64D 系统中的结构与计算问题，不支持视觉任务、CEM 或 closed-loop 结论。

模型用完整正交变换保留 64D 状态，分成四个 16D 块。每个块接收同一个已知 8D action；global4 另接收四维全局状态摘要。local4 的参数量与输入宽度相同，摘要只读取本块。

训练固定 1500 steps；最后 checkpoint；train/dev/test episode 分离；同 seed 配对 minibatch draws。下表是三个 training seeds 各自 episode 汇总值的平均，区间请读后面的逐 seed 配对结果。

| Condition | Arm | h10 error | h20 error | Action-response error | Params | Quality seeds | B1 ms | B300 ms | Each-step B300 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| independent | dense | 0.12495 | 0.07205 | 0.0963 | 32552 | 3/3 | 1.250 | 1.339 | 1.361 |
| independent | random_block | 0.87518 | 0.31472 | 0.1279 | 27200 | 0/3 | 2.306 | 2.535 | 2.849 |
| independent | learned_block | 0.24762 | 0.11265 | 0.0966 | 31296 | 0/3 | 2.295 | 2.538 | 2.851 |
| independent | learned_global4 | 0.18745 | 0.09115 | 0.0953 | 32576 | 0/3 | 2.613 | 2.850 | 3.144 |
| independent | learned_local4 | 0.25745 | 0.11702 | 0.0961 | 32576 | 0/3 | 2.662 | 2.921 | 3.239 |
| lowrank_coupled | dense | 0.12627 | 0.07314 | 0.0967 | 32552 | 3/3 | 1.233 | 1.324 | 1.345 |
| lowrank_coupled | random_block | 0.92850 | 0.33390 | 0.1292 | 27200 | 0/3 | 2.304 | 2.536 | 2.856 |
| lowrank_coupled | learned_block | 0.26551 | 0.11988 | 0.0971 | 31296 | 0/3 | 2.288 | 2.533 | 2.850 |
| lowrank_coupled | learned_global4 | 0.19695 | 0.09471 | 0.0961 | 32576 | 0/3 | 2.609 | 2.836 | 3.118 |
| lowrank_coupled | learned_local4 | 0.27918 | 0.12673 | 0.0969 | 32576 | 0/3 | 2.629 | 2.876 | 3.188 |
| dense_coupled | dense | 0.13121 | 0.07639 | 0.0978 | 32552 | 3/3 | 1.247 | 1.335 | 1.357 |
| dense_coupled | random_block | 0.98584 | 0.36951 | 0.1330 | 27200 | 0/3 | 2.302 | 2.535 | 2.861 |
| dense_coupled | learned_block | 0.28932 | 0.13351 | 0.0983 | 31296 | 0/3 | 2.302 | 2.699 | 2.855 |
| dense_coupled | learned_global4 | 0.19995 | 0.09812 | 0.0968 | 32576 | 0/3 | 2.622 | 2.852 | 3.141 |
| dense_coupled | learned_local4 | 0.28590 | 0.12869 | 0.0981 | 32576 | 0/3 | 2.661 | 2.912 | 3.243 |

h10/h20 为 train delta energy 归一化的末步 MSE；Action-response 列以真实 action 扰动造成的状态差分 energy 归一化，越小越好。它是诊断，不替换 primary h10。延迟含初始变换、native rollout、最终重构；每步重构接口另列。Q 在部署前冻结缓存，缓存准备不计入每条 rollout。

## 相对 dense 的门槛

质量门槛为 h10 <= dense + max(0.1*dense, 0.02)。速度门槛为 complete rollout 至少快 20%，B1/B300 分开判断。

| Condition | Arm | B1 reduction | B300 reduction |
|---|---|---:|---:|
| independent | dense | 0.0% | 0.0% |
| independent | random_block | -84.5% | -89.3% |
| independent | learned_block | -83.7% | -89.5% |
| independent | learned_global4 | -109.1% | -112.8% |
| independent | learned_local4 | -113.0% | -118.1% |
| lowrank_coupled | dense | 0.0% | 0.0% |
| lowrank_coupled | random_block | -86.9% | -91.5% |
| lowrank_coupled | learned_block | -85.6% | -91.3% |
| lowrank_coupled | learned_global4 | -111.6% | -114.2% |
| lowrank_coupled | learned_local4 | -113.2% | -117.2% |
| dense_coupled | dense | 0.0% | 0.0% |
| dense_coupled | random_block | -84.7% | -89.8% |
| dense_coupled | learned_block | -84.6% | -102.1% |
| dense_coupled | learned_global4 | -110.3% | -113.5% |
| dense_coupled | learned_local4 | -113.4% | -118.0% |

## 逐 training seed 配对对照

差分 = left - right，负值表示 left 更好。Bootstrap 以 episode 为单位，在单个 training seed 内作描述；三个 training seeds 不支持对训练随机性的总体推断。

| Condition | Seed | Left - Right | Mean difference | Episode bootstrap 95% CI |
|---|---:|---|---:|---|
| independent | 1101 | learned_global4 - learned_local4 | -0.064484 | [-0.073193, -0.056076] |
| independent | 1101 | learned_global4 - learned_block | -0.065079 | [-0.073930, -0.055808] |
| independent | 1101 | learned_block - random_block | -0.630329 | [-0.659851, -0.598605] |
| independent | 1102 | learned_global4 - learned_local4 | -0.075193 | [-0.082578, -0.067235] |
| independent | 1102 | learned_global4 - learned_block | -0.063564 | [-0.072749, -0.055029] |
| independent | 1102 | learned_block - random_block | -0.625505 | [-0.652382, -0.598404] |
| independent | 1103 | learned_global4 - learned_local4 | -0.070326 | [-0.077804, -0.062744] |
| independent | 1103 | learned_global4 - learned_block | -0.051884 | [-0.059377, -0.044033] |
| independent | 1103 | learned_block - random_block | -0.626842 | [-0.655531, -0.600628] |
| lowrank_coupled | 1101 | learned_global4 - learned_local4 | -0.080936 | [-0.090559, -0.072169] |
| lowrank_coupled | 1101 | learned_global4 - learned_block | -0.085815 | [-0.096919, -0.073865] |
| lowrank_coupled | 1101 | learned_block - random_block | -0.662324 | [-0.694832, -0.631343] |
| lowrank_coupled | 1102 | learned_global4 - learned_local4 | -0.086975 | [-0.095103, -0.078357] |
| lowrank_coupled | 1102 | learned_global4 - learned_block | -0.058810 | [-0.067419, -0.050194] |
| lowrank_coupled | 1102 | learned_block - random_block | -0.675891 | [-0.707896, -0.642718] |
| lowrank_coupled | 1103 | learned_global4 - learned_local4 | -0.078766 | [-0.088257, -0.070562] |
| lowrank_coupled | 1103 | learned_global4 - learned_block | -0.061065 | [-0.071250, -0.050965] |
| lowrank_coupled | 1103 | learned_block - random_block | -0.650734 | [-0.680280, -0.620789] |
| dense_coupled | 1101 | learned_global4 - learned_local4 | -0.074582 | [-0.086106, -0.063445] |
| dense_coupled | 1101 | learned_global4 - learned_block | -0.092804 | [-0.103821, -0.081763] |
| dense_coupled | 1101 | learned_block - random_block | -0.689280 | [-0.722335, -0.656655] |
| dense_coupled | 1102 | learned_global4 - learned_local4 | -0.089840 | [-0.098718, -0.080954] |
| dense_coupled | 1102 | learned_global4 - learned_block | -0.089730 | [-0.101135, -0.078665] |
| dense_coupled | 1102 | learned_block - random_block | -0.694436 | [-0.725175, -0.664783] |
| dense_coupled | 1103 | learned_global4 - learned_local4 | -0.093433 | [-0.105033, -0.082934] |
| dense_coupled | 1103 | learned_global4 - learned_block | -0.085584 | [-0.098119, -0.074134] |
| dense_coupled | 1103 | learned_block - random_block | -0.705846 | [-0.734928, -0.676182] |

## 后续门槛与解释限制

低秩条件 dev 上，global4 对 local4 和 learned-block 在全部三个 seeds 一致改善：是，触发预设 global16 附加对照。

坐标 subspace overlap 仅作描述，块的排列与块内旋转不唯一；它不能代替 heldout dynamics error。参数量为优化器可训练张量元素数，包含正交参数化的冗余元素，不等于有效自由度或 FLOPs。

独立与 rank-4 条件由设计保证存在局部结构。稠密条件覆盖更广的状态交互，但没有证明所有坐标系下都不可分。一次固定训练预算的失败可能来自优化，也可能来自结构容量；本轮保留 NO-GO/inconclusive，不增加训练步数或放宽门槛。

## 运行与复现

PBS job：25563194.pbs101；compute host：x1000c3s1b0n1。

必要机制测试见 mechanism_tests.json；完整配置见 FREEZE.json；逐 run 数值、延迟和 seed 对照见 summary.json。Checkpoints、per-episode arrays、training.tsv 与源代码 snapshot 保存在该 PBS run 目录。GPU 利用率和显存每 30 秒写入 job.log。

实验设计参考：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065
