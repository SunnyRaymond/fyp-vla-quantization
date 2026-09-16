# NULL INPUT COLUMN：SmolVLA `state_proj` 的 prior gate

审查日期：2026-09-13。结论：**novelty / standalone-GPU no-go；最多保留为
asset preflight 的 engineering self-check**。本文件只读 source 与小型既有
metadata，不运行模型、读取大数据、连接 cluster 或修改任何 screen。

## Source 与可成立的结构命题

LeRobot v0.4.4 的 `SmolVLAPolicy._get_action_chunk` 先调用
`prepare_state`，再把结果交给 `sample_actions`。官方 `prepare_state` 从
`observation.state` 取最后一个 observation（若有时间轴），调用
`pad_vector(state, max_state_dim)`；`pad_vector` 在右侧创建 zero tensor 并拷贝
原始坐标。`VLAFlowMatching.__init__` 的 `state_proj` 是
`Linear(max_state_dim, VLM hidden size)`，所以输入列的结构路径确实是
`state -> right zero padding -> state_proj`，而不是 action-flow 的
padding-coordinate 递归。对应 source 是 [v0.4.4 modeling_smolvla.py]
(https://raw.githubusercontent.com/huggingface/lerobot/v0.4.4/src/lerobot/policies/smolvla/modeling_smolvla.py)，
其中 `prepare_state` 在 427–431 行、`state_proj` 在 519–520 行。

这只证明“若 processor 输出的 active state 维数为 d，列 d:32 在送入
`state_proj` 前为零”。source 的 `prepare_state` 本身不做 normalization；
normalization 由外部 LeRobot processor 完成，随后才进入该 padding。故不能
把 raw dataset shape、config 的旧值或一个 checkpoint 的经验映射直接当成
当前 checkpoint 的 exact zero proof。

本地已有小证据显示 dataset `observation.state` shape 为 8，normalizer header
中的 state statistics 也为 shape 8；但同一 metadata probe 的
`normalizer_processor` feature config 记录过 shape 6。这个冲突没有在本审查中
用模型加载解决，因而不能静态宣称 active columns 是 `0:8`，也不能把
`8:32` 当作已验证的 zero range。若 runtime preprocessor 实际输出 6 维，则
候选边界应是 `6:32`；若输出 8 维，才是 `8:32`。任一 mapping/adaptation
使这些列非零，命题立即失效。

## 机制与最近 prior

在 active state 确认后，完整 row-wise RTN 的 scale 是该输出 row 的
`max(abs(W[row, :]))/7`。永远为零的输入列仍参与 max，可能让 active columns
使用更粗的 grid。把这些列先置 zero 再对同一个 `state_proj` 做相同 RTN，FP
函数是 exact no-op，而 active-column quantization error 可能改变；两臂参数
bytes、bitwidth、forward 次数和 downstream model 都相同。这是一个可写成
代数的 range-dilution 诊断。

但“删去恒为零输入 / structured pruning 后再量化”属于经典 dead-input
pruning 工程路径。[Deep Compression](https://arxiv.org/abs/1510.00149)
把 pruning 与 trained quantization 组合，[Network Slimming]
(https://arxiv.org/abs/1708.06519) 则系统化了结构化稀疏/通道裁剪；它们不是
SmolVLA-specific causal evidence，却足以使本项不能作为新的 quantizer 或
方法 novelty。现有 campaign 的 padded-coordinate feedback 研究的是 action
flow 的动态 7:32 坐标与 expert 反馈；本项是 state prefix 的静态 zero-input
列，机制对象不同，但这个差异只支持一次工程审计，不能升级为独立研究切面。

## 若作为 preflight，唯一可接受的最小检查

在不产生科学结果的情况下，先由 checkpoint 的实际 preprocessor 输出记录
`d_active`、归一化后 state tensor 的 exact `d_active:32` zero 检查，并记录
`state_proj` 实际 weight shape/name。然后比较同一 FP checkpoint 的两次
state-projection RTN：

1. `full-row-RTN`：原始 32 列参与每个 output row 的 absmax；
2. `zero-input-masked-RTN`：只在计算 scale 前将已证明 zero 的列置 zero，仍
   保存完整 32-column tensor，使用相同 signed `[-7,7]` grid。

两臂的 FP state projection 输出必须 exact/allclose 到预注册 arithmetic
tolerance；weight storage、forward 数、后续 modules 和 input batch 完全相同。
若 `d_active`、zero proof、state_proj binding 或 FP no-op 任一失败，记录
`structural/identity_no_go`。若只发现 active-column grid 改变，也只能记为
`engineering_range_audit`，不能用 action MSE、task success 或较少参数宣称
收益。

**最终 prior-only 判定：** source path 存在，但当前 asset mapping 有 6/8
冲突，且方法本质是已知 dead-input pruning；不申请 standalone GPU，不改
`policy-support` 或其他历史 pipeline。只有后续独立 asset preflight 证明
exact zero 时，才可作为 quantizer implementation self-test 保留。
