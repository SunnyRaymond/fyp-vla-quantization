# Fast-WAM + LIBERO：冻结基线的瓶颈测量

日期：2026-10-03。用户已确认没有 Joint checkpoint，先测现有 Fast-WAM。

## 目的与范围

测量真实 LIBERO episode 与固定观测重放中的延迟组成，尽可能细分到预处理、VAE、当前画面特征、每个 action denoising step、各层 attention/projection/FFN、动作后处理、环境 step 和视频保存；判断 ACSM 的 action-query→observation QK/AV 是否覆盖主要耗时。

模型是已发布 Optional IDM checkpoint 的 `first_frame` / Fast-WAM direct-action 模式。这里不生成未来视频，不是 FastWAM-Joint；结论只适用于本 checkpoint、模式、配置和设备。Joint 的延迟组成仍需要匹配权重实测。

## 冻结项

- 官方源码 revision：`7faa71108368fbb3b6885649f112af607427a2d4`；不修改源码、权重、mask、solver 或训练。
- Checkpoint：`/scratch/users/ntu/yguo017/fastwam-smoke/checkpoints/libero_optional_idm_2cam224.clean.pt`，沿用已验证的 12,041,735,545-byte 文件及对应 dataset stats；不重复大文件 hash。
- 官方 task config：`libero_optional_idm_2cam224_1e-4`；`first_frame`、BF16、seed 42、sigma_shift 1.0、CFG 1.0、compile_action_infer=false。
- `libero_goal` task 0 / trial 0，原 initial state；官方默认 30 个等待 steps、每次执行前 10 个动作再重规划、10 个 denoising steps。实际 action horizon 和 token 数由 resolved config/runtime shapes 记录。
- 单 env、单 episode；OSMesa CPU rendering + PBS 分配的单 CUDA GPU。不能作为 EGL/GPU 渲染性能或全 suite 成功率。

## 测量顺序

1. 获批 PBS allocation 内加载原环境和模型，记录设备、resolved config、源 revision、模型加载及初始化时间。
2. 运行一个原生闭环 episode；轻量记录 CPU 外层阶段和实际 replan contexts。保存官方 JSON、非空 MP4、任务 outcome 和管线 exit。
3. 从该 episode 的首、中、末 replan 选择最多三个实际观测 context。冻结观测、proprio、指令及 RNG，按每个 context 先做 3 次 warmup，再做 4 对无细分 hooks / 有细分 hooks 的重放，交替 AB/BA 顺序。
4. Native 完整调用在边界同步，以 wall time 和 CUDA event 测量；细分 CUDA events 成对异步记录，结束后集中读取，不在每个子操作同步。验证配对输出 parity；失败则不发布 instrumentation attribution。
5. 同三个 context 各做 2 对 native / 轻量计时重放，仅记录 VAE、文本、prefill、每轮 action denoise 与 mixed attention；不挂逐层 linear/FFN/QKV hooks。单独报告这组计时的输出 parity 和扰动。
6. 少量独立 torch.profiler pass 记录算子、CPU launch、GPU kernel 和输入 shape。其时延只用于诊断，不替代 native 稳态 latency。

## 归因规则

- 冷启动、稳态 action chunk、CPU 环境/渲染与视频输出分别报告。
- CPU enqueue wall time、CUDA event 区间和 profiler kernel time分别标注；嵌套 inclusive 区间不能相加。
- 细分 hooks 的开销必须与无 hooks 重放比较；受扰动的时间不能直接冒称原生各项独占时间。
- fused attention 中 action→observation 子矩阵没有可直接观测的独占 timer。先报告 native mixed-attention 整体区间、真实 Q/K/V shapes、理论边计算量，以及该整体路径的理想上限；不能把人为拆开的 matmul 计时当作 native 子路径耗时。
- ACSM 保留 QKV、FFN、video-query 路径和 solver。只有目标 QK/AV 可能删除；评分、packing、首步 dense 会减小实际收益。参数多、FLOPs 低或 attention 图都不足以证明净加速。
- 本次不执行 ACSM 裁剪，不扫 epsilon，不增加任务或改后端来放大收益。

## 计算规则与验收

所有模型/数据 I/O、加载、渲染、推理、profiler 和产物处理在 PBS compute allocation。脚本检查 PBS_JOBID、PBS_NODEFILE、非 login hostname 和获批 GPU UUID；每 15 秒把该 GPU 的利用率和显存写入 job log。Login 只做轻量脚本同步、提交、qstat 和小型日志/JSON读取。

验收需 PBS terminal state/Exit_status、任务 JSON/MP4、实际 native 重放计时、instrumentation parity/overhead、细分阶段与真实 shape证据，以及中文报告指出最大延迟和当前 idea 的可优化上限。单 episode 的成功仅是管线证据，不提供统计任务成功率或 Joint 迁移证明。
