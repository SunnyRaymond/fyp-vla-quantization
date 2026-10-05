# FastWAM 固定 context 缓存协议

状态：冻结，限于当前 `first_frame` action denoising 实验。实现只在实验侧临时替换 `forward`，不改官方 FastWAM 源码。

## 固定条件与问题

- 官方源码：`experiment/reproduction/fastwam-smoke/FastWAM`，revision `7faa71108368fbb3b6885649f112af607427a2d4`。
- 模式：FastWAM `Optional IDM`、`first_frame`、BF16、`compile_action_infer=false`、10 个 Euler steps。视频 K/V 按官方路径预先填充。
- 输入：固定 raw context `[1,129,4096]`；action query 的 batch 记作 W（目标形状 `[1,32,7]`）。context batch 保持 1，缓存只通过零拷贝 `expand` 供 W 个 query 读取，不把 context batch 扩展当作另一种缓存条件。
- Action expert：30 层、hidden 1024；cross-attention 为 24 heads × 128 dim。所有条件使用同一 checkpoint、context、video K/V、mask、Euler schedule、device、dtype 和 cloned 初始 action noise。
- 唯一处理因素是 action context 的计算/复用方式。Action transformer 层仍按原顺序执行；不并行 action 层，不改 planner、候选、视频缓存或采样算法。

## 条件定义

| 条件 | 操作 |
| --- | --- |
| A0 native | 不启用 `ContextCache`，逐次运行官方 text embedding 和每层 cross-attention K/V。 |
| A1 embedding | 每 chunk 预计算一次 action `text_embedding`；每层仍按官方 cross-attention 实时算 K/V。 |
| A2 kv | 每 chunk 预计算一次 embedding；逐层调用该层原生 `k`、`norm_k`、`v`，保存 K/V。读取时按 query batch W 扩展共享缓存。 |
| A3 batched_kv | 每 chunk 预计算一次 embedding；将 30 层 stack 权重展平为一个宽 `linear`，一次计算全部层 K/V，再恢复为连续的 `[layer,batch,sequence,dim]`。每层保留自己的 K/V weight、bias、`norm_k.weight` 和 `norm_k.eps`；K 先线性投影（bias 融入 `linear`），再逐层按原 RMSNorm 公式归一化。各 action layer 仍顺序执行。 |

A1/A2/A3 通过 `ContextCache(model, mode)` 创建。对每个独立 chunk，在 `cache.activate()` 内调用一次 `cache.prepare(context)`，随后完成完整 10-step 原生 denoising。完整 `infer` 也须在首个 denoise 前用当次实际 context 调用 `prepare`。完成采样并记录数据后调用 `cache.clear()`；它释放该 chunk 的 context embedding 和 K/V，A3 的 packed weights 保留至 cache 对象销毁。

`prepare_latency_ms` 仅是 CPU enqueue 时间，设备完成时延以 runner 在完整边界同步计时为准；`cache_bytes` 是当前 chunk 缓存张量的逻辑字节数；`packed_weight_bytes` 是 A3 额外堆叠的 K/V/norm 参数副本字节数；`packed_weight_build_ms` 是建 cache 对象时同步计时的 A3 一次性构造成本。A3 的 packed storage 与构造时间都必须单独报告；不能并入 native baseline 或说成免费。A0 不建 cache，A1/A2 的 `packed_weight_bytes` 为 0。

源码语义锚点：FastWAM `_denoise_action_with_video_cache` 调用 `action_expert.prepare` 后进入 `mot.forward_action_with_video_cache_tensor`；`ActionDiT.prepare` 调用 action `text_embedding`；`CrossAttention.forward` 的 K 路径是 `norm_k(k(ctx))`，V 路径是 `v(ctx)`，随后通过官方 `flash_attention`。此 helper 复用该 attention 函数；这只表示沿用官方 attention 调用路径，不构成 FlashAttention backend 证明。

## 正确性门槛

每次配对从相同初始 noise 开始，保存 10 个 Euler step 的模型 action prediction、每步更新后的 action latent 和最终 action。A1/A2/A3 各自与 A0 比较；A2/A1 与 A3/A2 另做直接配对。所有 step 与最终输出均须满足 `torch.allclose(atol=1e-5, rtol=1e-5)`。不要求 bitwise 相同。

任一被测条件未过门槛时，保存误差与输出，标记 `numerical_drift`，停止该条件的性能结论；不放宽阈值、不更换输入来追过门槛、不叠加其他改动。不得把单步局部相等、只比最终结果或 planner outcome 当成逐步语义 parity。

## 时延与显存测量

1. 在 PBS compute allocation 内加载模型和数据、预热；login node 只做轻量作业控制/状态查询。记录实际 GPU 利用率和显存占用到该 PBS job log。三个真实 context 固定后，全实验使用同一模型实例和相同 checkpoint。
2. 对每个 context、每个对比先对两臂各做 3 个完整 warmup chunk，再做 4 个配对重复，运行顺序交替为 AB、BA、AB、BA。固定的五组对比为 A0/A1、A0/A2、A0/A3、A2/A1、A3/A2。每个配对使用相同 context 和 cloned 初始 noise；A2/A3 是独立成组的直接比较。统计单位是 context 内的配对重复，不把 10 个 step 或同一 pair 的两次运行当独立样本。
3. 每次计时从 chunk context/cache 准备前开始，到第 10 个 Euler step 完成后由 runner CUDA synchronize 结束；报告端到端 chunk 时延和 `prepare_latency_ms`（仅 CPU enqueue 时间）。A3 另报 raw `packed_weight_build_ms`、`packed_weight_bytes`，并按实际使用 chunk 数列出一次性构造成本摊销值。报告每个 context 的配对时延差及跨 context 汇总中位数、p10/p90。
4. 峰值显存使用独立 memory pass，在相同模型实例上按条件分组测量；每组前调用 `cache.clear()`，记录 prepare 前 allocated/reserved baseline，重置 CUDA peak 统计，执行一个完整 10-step chunk 后记录 absolute peak allocated/reserved 及相对 baseline 增量。各组测后清理 chunk cache。A2/A3 memory pass 单独运行，不与 timing pass 混在一起；不要求重启进程或重载模型。A3 baseline 包含持久 packed weights。reserved 会受同一 allocator 生命周期和已保留内存影响，需连同 baseline 报告，不能只用 reserved 峰值增量解释缓存成本。
5. 所有配对使用相同的三个 context 和 noise 配对，失败门槛不变。时延结果只支持当前固定 context 与 query batch 的 predictor-level 计算成本结论，不外推为闭环成功率、通用吞吐或 FlashAttention backend 结论。

## 实现自检

`context_cache.py` 提供 `self_check()`（兼容别名 `toy_self_check()`）：先验证 `PBS_JOBID`、`PBS_NODEFILE`、当前 hostname 在 nodefile 中且 hostname 不含 login，再用小型真实 `CrossAttention` 检查 A1/A2/A3 对 native 的 `1e-5` parity、异常退出后 forward patch 恢复，以及 `clear()` 释放 chunk cache。此检查应在 PBS compute allocation 运行；它不加载正式 checkpoint，也不替代正式输入的逐步 parity gate。
