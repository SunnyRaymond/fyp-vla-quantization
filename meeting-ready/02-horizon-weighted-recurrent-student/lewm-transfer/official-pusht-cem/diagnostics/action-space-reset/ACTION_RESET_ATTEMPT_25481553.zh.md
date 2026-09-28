# Action-reset 诊断入口失败：25481553.pbs101

PBS 作业于 2026-09-23 在 `x1001c6s4b1n1` 结束，状态 `F`、`Exit_status=1`，walltime `00:01:57`。作业通过 PBS allocation guard 并导入 stable-worldmodel，但 runner 在创建任何 `World` 或执行 reset 前，于读取冻结执行顺序键时抛出 `KeyError: 'arm_order_seed_base'`。因此没有诊断样本，也没有对 action-space 根因作出结论。

原始 job log、PBS 状态和本次使用的 FREEZE 均保留在 `artifacts/25481553.pbs101/`。冻结修订版只补入来自 Stage 1 冻结文件的 `arm_order_seed_base=2609222000` 与 `fidelity_order_seed_base=2609223000`；reset seeds、六条路径标签、采样/seed 观测、资源和 Stage 1/2 科学设置均未改变。该作业是诊断脚本启动失败，不是环境或实验结果。
