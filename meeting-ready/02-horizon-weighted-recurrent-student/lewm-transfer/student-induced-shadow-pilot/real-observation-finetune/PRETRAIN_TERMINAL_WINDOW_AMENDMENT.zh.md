# 训练前的 terminal-window 处理

`25543623` 的采集汇总在任何 finetune 或 validation 预测前显示：64 个预选 train episode 中 58 个有完整 25-step 训练窗口，6 个因提前终止没有完整窗口；16 个 validation episode 均有 t0 完整窗口，14 个也有 t25 完整窗口。采集协议本来禁止填补、替换或补抽提前终止的任务。

原 finetune runner 却要求 64/64 train episode 都有完整窗口，和上述采集协议冲突。现将采样规则明确为：只在已冻结的 64 个 train IDs 中，对实际至少有一个完整窗口的 ID 做均匀 episode 抽样；episode 内再均匀抽可用的 t0/t25 window。记录预选与可用 episode 数，不补抽或移动 split；validation 仍只在 500 updates 和 terminal checkpoint 保存后使用一次。此修改只依据窗口可用性，不依据 latent 误差或模型改进结果；训练目标、优化器、更新数和 predictor gate 均不变。
