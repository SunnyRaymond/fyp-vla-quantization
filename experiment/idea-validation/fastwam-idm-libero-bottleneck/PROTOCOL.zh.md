# Fast-WAM Optional IDM LIBERO 延迟剖析

日期：2026-10-03。该目录仅保存同 checkpoint 的 `idm` 新测量；原 `fastwam-libero-bottleneck` 结果保持不变。

## 固定设置

- 官方 FastWAM revision `7faa71108368fbb3b6885649f112af607427a2d4`，模型 `FastWAMOptionalIDM`，checkpoint `/scratch/users/ntu/yguo017/fastwam-smoke/checkpoints/libero_optional_idm_2cam224.clean.pt` 和原 dataset stats。
- task config `libero_optional_idm_2cam224_1e-4`；BF16、seed 42、sigma shift 1、text CFG 1、`compile_action_infer=false`。
- `libero_goal` task 0 / trial 0，30 waiting steps，每 10 步重规划，10 inference steps。`num_video_frames=(33-1)//4+1=9`，VAE latent frames 为 3；future pixels 不解码。
- 一次真实 native IDM episode。OSMesa CPU rendering，推理使用 PBS 分配的 NVIDIA A100-SXM4-40GB；单 episode 结果只作运行与计时证据。

## 测量

- IDM 官方实现每个 `infer_action` 先进行 10 次 `_denoise_video` 和 10 次 video scheduler step，再冻结完整 denoised video 为 condition、prefill video K/V，最后进行 10 次 action denoise 和 10 次 action scheduler step。
- 从该 episode 实际首、中、末三个 replan observation 取样；固定 observation、proprio、task text 和 RNG。每种模式 warm 3 次；`first_frame` 与 `idm` 按 ABBA 次序做 4 对 native 整体调用，记录 CPU wall 与 CUDA event，不加细分 hooks。
- IDM 每个 context 再做 4 对 native/fine、2 对 native/thin，均 ABBA；检查 finite/allclose。`first_frame` 另做每个 context 一对 native/fine instrumentation parity。固定 observation replay 不用于比较闭环成功率。
- 单次 `torch.profiler` trace 只用于定位 op/kernel 与输入 shape。`FW/` scope 的 CUDA total 只有在 CPU span 非零且 parent chain 有效时才单列为 inclusive scope 总量；它不等于单个 kernel 且彼此可能重叠，不与 kernel 表相加。CPU 为零的 synthetic scope 记录会滤除。

## 来源与归因

- `video_expert.prepare` 的原生返回值给出实际 tokens-per-frame；保存其 token shape、per-frame token 数和 cache sequence 长度。Action mixed attention 按 current observation frame、generated future video frames、current action tokens 分别记录 K/V 数量、来源和 QK+AV MAC2。该调用只有一个 fused SDPA timer，各来源的耗时不是实测拆分。
- IDM latent frame 0 来自当前 observation，另外两个 latent frames 是被 denoise 的 future video。预期运行 shape 是 98 observation tokens、196 future-video tokens、32 action tokens；profile 会以实际返回值和 attention tensors 验证。
- Video self-attention、video text-cross-attention、MoT cache prefill 与 MoT action mixed attention 分开记形状、来源、理论 QK+AV MAC2 和 stage。Per-layer video QKV/FFN/text-cross 数据保留 stage，明确区分 `video_denoise` 与 `video_cache_prefill`。
- 所有模块区间为 inclusive，不能相加。Native、thin、fine、CUDA event 和 profiler kernel 各自分开报告；只有 native/thin 用于主要低扰动耗时比较，细分结果用于定位。

## Allocation 和完成门

PBS 请求 1 GPU、16 CPU、110 GB RAM、30 分钟。Login node 只同步控制脚本、提交/查询作业并读取小型日志/JSON。Python/model initialization、模型与数据读取、render、推理、profiling 和产物汇总均由脚本 guard 后在 allocation 内执行。每 15 秒记录 GPU 利用率与显存。

需要 `Exit_status=0`、`PIPELINE_COMPLETE`、非空 episode results JSON/MP4、profile summary、成对模式计时、所有内部 parity gates 及 runtime cache geometry。真实 episode 可失败；其结果会单独标记，不把失败解释为 pipeline timing 无效。此范围不涉及 full suite、training、quantization、ACSM 或 epsilon sweep。

## 实际后处理恢复记录

GPU `25663625.pbs101` 已完成 episode、全部配对重放与 trace，`profile_summary.status=complete`，但简报汇总器读取 nested parity 字段的键名错误，原始 wrapper/PBS Exit_status=1 且没有 PIPELINE_COMPLETE。CPU-only `25663764.pbs101` 读取同一批 measurement 产物，修正字段映射并通过原有输出一致性、pair-count 和 runtime geometry gates，以 exit=0 与 DIAGNOSTIC_POSTPROCESS_COMPLETE 完成诊断后处理；不改写 GPU 原始失败终态。中间 CPU `25663741.pbs101` 的额外报告字节限制已取消，该限制不是模型/质量 gate。完整失败历史与最终证据见 RESULT.zh.md。

若 GPU profile 数据已完整、仅 compact 汇总未完成，`finalize_metrics.pbs` 使用 CPU-only allocation 读取现有 `profile_summary.json` 并运行同一组 gates；汇总作业有独立 terminal receipt，原 GPU job 的 exit status 与 `PIPELINE_COMPLETE` 状态保持原样记录。
