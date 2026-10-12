# 两小时分段续跑

原任务 `25755802.pbs101` 在第一次更新之后因整轮预算预测退出；已保存 614 个 Linear 的 FP32 books 和 log row scales。新版从该次更新后继续，不重复 case 0 的第一次更新。旧版没有保存 Adam，因此第一次接续明确重置一次 Adam；后续接续恢复完整 optimizer state。

每个 PBS allocation 仍为两小时。只预测下一次训练或评估操作是否会接近本段时限，并预留 90 秒保存；不再预测整轮训练与全部评估必须在本段完成。冻结配置仍是两轮、16 次更新、4 个 selection cases × 2 seeds，以及 22 个 test cases × 2 seeds；已完成的输入不会重复评估。

每次 optimizer update 和每个 evaluation context 完成后，以临时文件加 atomic replace 写入 `results/full-<job>/checkpoint.pt`。其中保存当前 master 参数、Adam moments/step、Python/NumPy/Torch/CUDA RNG、best checkpoint、最终训练参数、训练 cursor、全部已完成评估及 selection history。保留旧文件直到新文件写完；如被硬时限中断，下段最多重做尚未提交到 checkpoint 的当前操作。

正常达到本段预算时返回 `summary.status=resumable`、Exit 0 和 `RESUME_READY`。这表示可接续，不是完整实验成功。控制器 `continue` 只在确认当前 job 终态并验证 checkpoint 后提交新段，记录 predecessor，更新 `full_handle.json`；普通错误需先修复，不反复提交。两小时硬终止且已有有效 checkpoint 时也可接续。每半小时的现有监控跟随当前 handle 并执行此接续流程。

完整成功需要 PBS F/X、Exit 0、`PIPELINE_COMPLETE`、匹配当前 job 的 complete summary、2 epochs/16 updates、22 cases/44 test contexts，以及 `test_queries_before_freeze=0`。最终选中的 overlay 和最后训练参数均保存；不能以部分结果代替完整 cohort。

PBS 开始数值实验前运行小型自检：保存/恢复之后的下一次 Adam 更新必须与连续运行完全相同；模拟写入失败必须保留旧 checkpoint；训练和评估 cursor 不能重复已完成项。所有数值自检和模型操作都在真实 PBS compute allocation；本地只做语法检查与轻量控制。
