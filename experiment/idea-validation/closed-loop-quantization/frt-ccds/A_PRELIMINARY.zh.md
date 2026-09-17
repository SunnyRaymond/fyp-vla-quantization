# A 初步 V100 实测

2026-09-12。Job 64687，TC1N04，Tesla V100-PCIE-32GB，SLURM COMPLETED 0:0，实际分配 23 秒 = 0.00639 V100 GPU-hours。

真实 checkpoint epoch65；history shape `[1,1,196,404]`，因此实际 `num_hist=1`，不能用源码默认的 3 代替 checkpoint 配置。选择 valid dataset index60 的真实连续记录，未使用 DEV。

核心数值检查通过：

- wrapper 对齐 source rollout：第一步和第二步输出最大误差均为 0。
- FP append 加 slot residual 与 Q0 append 差异仅 1.1921e-7；action residual 严格为零。
- 零残差 transport 和 FP 重放差异均为 0；原始 history 未修改。
- 实际 quantizer scale/rounding backward 梯度有限且非零。
- hard-W4 保存和重载输出差异为 0，soft alpha 已从 model parameter tree 移除。

8 次 batch1 backward，其中后6步平均 0.08734 秒/步；峰值 allocated 1,715,683,328 bytes（约1.60GiB）、reserved 1,807,745,024 bytes（约1.68GiB）。该预检使用无 Wz 的 clean+transport loss 和短8步更新；不是正式拟合速度、收敛或机制证据。

完整 A 仍需 physical xy deterministic replay、冻结 CAL Wz/random control、正式 loss/schedule 计时与独立核验。此文件仅记录已完成的初步结果，不作为 B 自动放行文件。

原始小结果：`artifacts/64687/core_probe.json`；完整 hard checkpoint 和代码快照保存在 CCDS 对应作业目录。无 C/TEST 或闭环收益声明。
