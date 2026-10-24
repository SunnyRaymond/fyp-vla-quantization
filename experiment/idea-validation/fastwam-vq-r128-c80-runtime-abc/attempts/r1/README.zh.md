# C80 R128 A+Z+KL：runtime A → B → C 与 torch.compile

冻结 checkpoint `25921795.pbs101 / reader_kl`，不重训。仅运行新增版本，已有 A100 对照直接复用。

| Mode | 新增改动 |
|---|---|
| runtime_a | H/U/A4 复用 FP32/FP32/BF16 workspace；保留原始算术及 cuBLAS |
| runtime_b | A 加 query 内 context projection、各层 cross-attention K/V 缓存 |
| runtime_c | B 加 video step、video prefill、action step 三个手动 CUDA Graph |
| runtime_compile | B 加三个区域的 torch.compile/Inductor，关闭自动 CUDA Graph |

主对照：`25954757.pbs101 / layer_decode_fp32`。BF16/eager/input_fused 复用 `25943391.pbs101`。
每个候选沿用 contexts 3000–3039 × seeds 0/1，共 80 queries；3 warmups，batch 1，32×7 action，9 video frames，10 video + 10 action steps。

A 与 B 先通过 full80 action 和独立 anchor action/future 的 `max_abs <= 1e-6` gate，再进入下一步。C 和 compile 分别基于 B；其中一个运行失败不妨碍另一个尝试。性能回退仍如实记录。

Policy latency 从 prepared GPU datum 到 infer_action 返回并同步；外部 observation preprocessing、环境及机器人运动另计。主计时无 profiler；额外 trace 仅用于诊断。报告 query peak、load peak、setup peak、常驻 workspace/cache 和 compile/capture 准备耗时，避免只展示 steady-state 开销。

长任务 6h、短任务 1h59 同时排队；compute allocation 内原子选胜者，监控精确取消另一份。每 15 秒 GPU 利用率与显存写入 job log。任何模型、tensor I/O、GPU 编译及推理仅在经过 PBS/hostname/nodefile 检查的 compute allocation 内执行。

本地验证仅用 Python 标准库，覆盖缓存生命周期、Graph 输入刷新与 gate；CUDA 数值、显存和性能以获批 GPU 作业结果为准。
