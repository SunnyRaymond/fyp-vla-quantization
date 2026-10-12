# SpinQuant native W4A4：复用 action-aware VQ 的 action RMSE cohort

直接复用已经训练并选中的200-step SpinQuant-adapted bank，运行原280-episode实验的 native INT4 实现。本轮不重新训练、calibrate 或选择rotation，也不增加 decoded-BF16 SpinQuant 对照。

固定复用 action-aware VQ 作业 `25743693.pbs101` 的 observations 和 BF16 action reference。4个selection observations与22个test observations各有2个sampler seeds，新增52次SpinQuant query。执行前用其中一个输入重放BF16，以确认缓存reference与当前pipeline一致。

主指标是 normalized action 前10步、6个motor维度相对同输入同seed BF16的RMSE；先将2个seeds在case内平均，再对22个test cases等权平均。另报告gripper、完整32步motor误差和Original/Plus分项。现有Scalar及VQ结果从缓存action复算，不重新运行这些配置。

1个PBS作业、1张GPU、walltime为2小时；每次allocation只加载1次模型。614层使用原packed W4 codes、BF16 G128 scales及learned R128，激活动态A4，KV BF16。逐次确认614层调用及native INT4/PTX证据。

每个case/seed完成后原子保存action与指标；接近时限时正常退出并标记checkpointed，下一次allocation跳过已完成queries。GPU utilization与显存每15秒写入job log。

本轮没有环境动作执行或closed-loop episodes。对照反映不同完整配置的固定输入action fidelity，不能将差异单独归因于rotation或训练目标，也不作为native latency结论。

当前作业：`25756064.pbs101`。本地状态保存在RUN_STATE.json，完成后结果写入results/full/summary.json及RESULTS.zh.md。
