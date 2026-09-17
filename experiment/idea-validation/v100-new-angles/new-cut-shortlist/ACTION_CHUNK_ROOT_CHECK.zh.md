# Root source复核

已核对 LeRobot v0.4.4 [embed_suffix/mask](https://raw.githubusercontent.com/huggingface/lerobot/v0.4.4/src/lerobot/policies/smolvla/modeling_smolvla.py)：action mask 每position递增，故该固定路径无 future-to-earlier attention。不是所有 flow-VLA 都有相同mask，不能泛化。

接受本候选结构性 no-go，不跑GPU。另明确 first8 是本轮 diagnostics 的固定评价窗口，不能自动称为该 checkpoint 实际部署执行 horizon；n_action_steps 是外部 queue 设置。本结论仅涉及固定网络路径；若人为引入跨时间共享activation scale，可能产生量化器的额外耦合，但那属于已审查的 shared-scale coupling 家族，不能偷偷替换当前假设。
