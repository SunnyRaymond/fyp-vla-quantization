# Action-chunk temporal prefix：prior gate

审查日期：2026-09-13。本文只检查一个切面：SmolVLA 的 50 个 action positions 中，实际执行的 first 8 与后续 42 个是否存在“后缀量化误差反馈到前缀动作”的模型路径。只读取 pinned LeRobot v0.4.4 source、相关 prior 和现有 campaign 文档；没有运行模型、数值、代码、集群或下载。

## 结论

**`identifiability_no_go`，本轮不值得 GPU screen。** 在 v0.4.4 的官方 attention mask 和 inference call path 中，action positions 是 causal 的：第 `j` 个 action token 只能看同一 chunk 中不晚于 `j` 的 token，以及已缓存的 observation prefix。位置 8–49 不能影响位置 0–7。`n_action_steps` 只控制执行队列取多少个已经生成的动作，不会改变 action expert 的 temporal mask。因而“future42 的 precision demand/quantization error 通过时间轴反馈改变 executed first8”在当前固定架构中没有可识别路径。

## v0.4.4 source 事实

官方 `SmolVLAConfig` 固定默认 `chunk_size=50`、`n_action_steps=50`、`max_action_dim=32`、`num_steps=10`。[configuration_smolvla.py](https://raw.githubusercontent.com/huggingface/lerobot/v0.4.4/src/lerobot/policies/smolvla/configuration_smolvla.py)

`SmolVLAPolicy._get_action_chunk` 让 `sample_actions` 生成完整 chunk；`predict_action_chunk` 返回完整结果。只有 `select_action` 把 chunk 放入 queue 时，才按 `n_action_steps` 取前若干动作。[modeling_smolvla.py](https://raw.githubusercontent.com/huggingface/lerobot/v0.4.4/src/lerobot/policies/smolvla/modeling_smolvla.py) 所以 first8 是一个外部执行窗口，不是模型内被标记为 prefix 的 action token 区域；官方代码里的 `prefix` 指 image/language/state conditioning。

在 `embed_suffix` 中，`action_in_proj`、时间 embedding 和两个 time MLP 都逐 action position 处理 `[B,50,32]`，没有跨时间 pooling。50 个 action token 的 `att_masks` 全为 1；`make_att_2d_masks` 用 cumulative mask 产生 `mask[query,key] = (cumsum[key] <= cumsum[query])`，所以 action query `j` 只能 attend `0..j`。[modeling_smolvla.py](https://raw.githubusercontent.com/huggingface/lerobot/v0.4.4/src/lerobot/policies/smolvla/modeling_smolvla.py)

在 inference 的每个 denoising step，image/language/state 的 key/value 先通过 `embed_prefix` 和 `forward(..., fill_kv_cache=True)` 缓存；随后 `denoise_step` 只把 action suffix 作为 query。cross-attention layer 使用 action query 对 cached prefix 做 attention，self-attention layer 使用 prefix KV 加上 causal action KV。两种路径都不执行 prefix query 对 future action key 的读取；因此位置 8–49 没有回写位置 0–7 或 cached prefix 的路径。`x_t = x_t + dt*v_t` 也逐 position 更新，下一步的 first8 state 仍只依赖此前 first8 state 与固定 condition。

这里需要和此前的 `padded-coordinate-feedback` 分开：32 维坐标在同一 action position 内由 `action_in_proj`/`action_out_proj` 混合，可能产生 physical7 与 padded25 的**坐标**耦合；本候选讨论的是时间位置 0–7 与 8–49，官方 source 没有相反方向的时间耦合。

## Prior 与重叠边界

1. [SmolVLA](https://arxiv.org/abs/2506.01844) 的 primary paper 将 chunked action generation 与 asynchronous inference 作为提高 responsiveness 的执行架构，但没有提出 action-position quantization 或 future-to-past precision feedback。它支持“first actions may be executed before later actions”的执行语境，不支持本候选的机制假设。

2. [Adaptive Action Chunking at Inference-time for VLA Models](https://arxiv.org/abs/2604.04161) 研究用 action entropy 选择 chunk size，直接覆盖 chunk 长度与重规划频率的 inference 设计；它没有把未来 action positions 的 PTQ error 作为回写前缀的因果机制。把 `n_action_steps=8` 与 50 混用会改变 replanning/执行分布，不能作为本候选的 matched quantization control。

3. [QuantWAMs](https://arxiv.org/abs/2607.28405) 的 fixed-intervention auditing 针对 reachable state 与 denoising step 的保护安排；其 step 是 flow denoising 的 `num_steps` 轴，不是 SmolVLA chunk 内的 action-position 轴。它不能为 8-versus-42 的时间后缀反馈提供直接先例。[QuantVLA](https://arxiv.org/abs/2602.20309) 的 action-head PTQ、[Ω-QVLA](https://arxiv.org/abs/2605.28803) 的 per-step scaling，以及 [ActQuant](https://arxiv.org/abs/2605.24011) 的 action-guided bit/scale 分配，也没有在 causal action chunk 中建立 future-token 到 earlier-token 的路径。

因此 prior 已覆盖 chunk execution、denoising-step precision 和 action-head PTQ 的邻域，但没有把它们合并成一个可成立的机制。更关键的是，source-level causality 已经否定当前 checkpoint/架构中的目标路径；不能因为 prior 没有同名论文就把结构性缺失包装成 novelty gap。

## 不能用的替代解释或实验

- 若量化共享的 `action_in_proj`、expert layer 或 `action_out_proj` 后 first8 与 FP 改变，这是共享参数扰动或同 position feature mixing；它无法证明 future42 对 first8 的反馈。量化所有 action positions 的同一模块也不会产生时间方向反转。
- 将 first8 action 固定、替换后42的 noise，或在每一步拼接另一路 future velocity，会测试不同的 input distribution、padding/coordinate 或 oracle field；它不再是固定 v0.4.4 causal path 下的 temporal PTQ screen，也违反不使用 FP oracle 和不混淆 H/预算的边界。
- 把 `n_action_steps=8` 当作“只执行前缀”的模型开关是不准确的：它只截取 queue；`sample_actions` 仍计算完整 50-position chunk。改变该值还会改变何时重新观测和重新采样，不能和原设置构成只改 precision 的 control。

## 停止规则与 claim 边界

本切面在当前 v0.4.4 固定 source 上直接停止为 `identifiability_no_go`，不写 runner，不申请 GPU，不新增 sample、noise、denoising step 或 action-chunk length。若未来更换为明确的 bidirectional action mask、非 causal temporal mixer，或出现 action-token 之间的后向状态缓存，必须以新 source identity 重新审查；那将是新的架构问题，不得复用本结论。

即使某个共享模块的量化让前 8 个输出更敏感，也只能报告“executed prefix 的 output sensitivity”或“replanning-window effect”。本审查不支持 temporal precision allocation、future-to-prefix feedback、任务成功、部署收益或新的 quantizer claim。
