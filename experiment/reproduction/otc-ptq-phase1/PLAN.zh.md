# OTC-PTQ 第一阶段：预注册范围与运行记录

本阶段验证 measurement feasibility 与 one-transition signal；不验证联合量化 success gain，不重跑 IdeaSpark pipeline。

## 固定范围

- FastWAM Optional IDM，固定源码 revision `7faa71108368fbb3b6885649f112af607427a2d4`；本地/远端小型源码按换行归一化比较。
- 复用已验证 checkpoint 和配套 stats；运行前检查 metadata 与历史 VERIFIED marker。本阶段不重新计算大 checkpoint hash。
- `idm`、sigma shift 1.0、compile disabled，保持官方 action horizon / denoising steps / replan / gripper transformation。实际 reference dtype 必须记录；BF16 不写作 FP16。
- LIBERO-Goal tasks 0、1，按 suite order 预选；每 task initial-state indices 0、1 为 CAL，2、3 为独立 CHECK。
- 每 episode 目标 4 states；从 reference replan boundaries 选择，覆盖 robot/task contact 附近和 free motion，缺失 strata 明确标记。不是 IID frames。
- 8 个预选 Linear weight sites，跨 action/video branch 和深度；具体 registry 在 scoring 前保存。single-site intervention 在所有 denoising calls 生效。
- W4 symmetric per-output-channel fake quant，范围 [-7,7]，ties-to-even，zero point 0。所有其他 target sites 保持原 precision。

## 三个实验

1. Reference-repeat、identity-hook、single-site 对照。恢复完整 physics/controller/observable/RNG 状态，核对模型 cache 生命周期。测量 action、next-observation、physics 自然噪声；恢复失败即停止解释 quantization effect。
2. 同一 traces 计算 Local、Action-only、Observation-only、OTC；分 contact/free，记录 first-action 与 full-chunk 差异；检查 effect 是否超出重复噪声。
3. CAL-only 归一化和 ranking，top-2 protection 候选在 CHECK episodes 上诊断稳定性。独立 physics/contact 指标只作诊断，不加入 score。不同排名不是 task relevance 的证明。

## Go / no-go interpretation

- Replay 失败：instrument invalid，先修复。
- 数值扰动无可测效应：当前 intervention/范围 inconclusive，不否定整个 idea。
- OTC 与 action-only 选择一致：当前没有支持 simulator 额外成本的证据。
- OTC 特有选择在 CHECK 不稳定：弱证据，暂缓扩大。
- 可重复的环境敏感 sites，且 action-only 漏掉、独立 physics 变化支持：进入后续联合量化闭环测试的 provisional go，仍不是 success gain 证明。

## 资源与安全

首次 probe 单 GPU，20 分钟上限；paired probe 与后续作业按实际速度分配，不因可并行就复制同一工作。采集与评分使用互不重叠输出。
Login 只做轻量查询、PBS 提交和小型控制文件操作。所有 model/physics/render、依赖或 heavy I/O 在获批 compute allocation；脚本检查真实 PBS_JOBID、hostname、PBS_NODEFILE 与 scheduler R 状态。OSMesa CPU render + scheduler UUID CUDA，保留 SSH host-key verification。CPU 准备不用 GPU；失败及时退出。

## 当前作业

- `16180521.pbs101`：首次源码身份 gate 因 Windows CRLF / Linux LF 差异停止；未加载模型。随后实际比较内容确认归一化相同，修正 gate。
- `16180526.pbs101`：官方 `idm` task0/trial0 entrypoint probe 成功，Exit_status=0，allocated walltime 00:02:07；task0 1/1 success，evaluation duration 41.1788s。PBS GPU monitor maxGpuMemoryUsed=24718MB。不是 paired replay 或 quantization 证据。小型 JSON、日志、identity check 与 qstat 已取回 artifacts/16180526.pbs101。

既有 smoke `16180074.pbs101` 仅为 Optional IDM checkpoint 的 first_frame 运行成功，不能作为本次 idm feasibility 证据。

## 配对 probe 已验证与后续执行调整

- `16186136.pbs101`：Exit0，allocated walltime 574s；repeat / identity action max abs 与 next-image L1 均为0。单video d00 self_attn.q W4：action chunk MSE 1.27097e-6，next-image L1[0,1] 3.84472e-4。预热后推理约1.16–1.22s，peak allocated约25.14GB。仅证明该state/site测量可行。
- `16184340.pbs101`：早期CPU恢复检查在task0 t30/t40的原始下一步与恢复下一步obs/qpos差异均0。
- 增补FlatStove visual flag与object-state registry检查后，`16186266.pbs101`/`16186267.pbs101`分别对task0/1进行CPU-only复核。
- 后续每task四个episodes共用一次model load；episodes仍各自reset/seed并写独立文件。采集轨迹不回退RNG，直接继续已执行第一步后的native状态。
- Contact标签实际为排除support/robot-self后的candidate contacts，含object-object；不声称全部是任务相关。persistent contact与first-action transition分开报告。

在任何正式score结果读取之前冻结分析补充：CAL bootstrap在每个固定task内部按完整episode重采样，避免task组成变化；将原已记录的first_action_mse作为secondary ranking diagnostic，检查full chunk与first transition时间范围不匹配是否足以解释OTC/action-only差异。主要OTC仍lambda=(1,1)，ratio sweep仅{0.5,1,2}。

## 首次score作业的恢复失败与修复

- `16186342.pbs101`/`16186343.pbs101`均Exit1（540s/536s），尚无score rows。新进程reset后的gripper.current_action为shape(1,)，采集snapshot在30-step warmup后为(2,)，严格shape gate中止。
- 修复仅让score入口完成与collection相同的30 dummy steps，再恢复snapshot；不放宽shape/physics/observation gates，不改已采集数据或任务选择。
- 重试前新增CPU-only全量collected replay：从新env恢复每个episode的4个selected states，执行已记录的第一action，对照原始next images/physics/contact/terminal。GPU model与CUDA RNG不属于此CPU检查；它们仍由正式score的reference重复与原始action gate验证。

## Derived contact cache 的恢复边界修正

全量CPU replay先发现integer dummy action导致recent_actions buffer dtype变化；已改为保留snapshot历史dtype。随后部分state在sim.forward后原始contact point数不同（40→39、26→27）。这是派生缓存边界：官方MuJoCo文档说明mj_step在更新state后结束，position-dependent mjData仍对应上一步；本地固定robosuite rs_base.py:391在每次控制前主动sim.forward。因而raw contact cache不应作为integration-state exact invariant。

保留native integration/model/controller/observable严格恢复及原始动作/下一图像/qpos/qvel/接触/终止状态的functional replay；记录而不隐瞒forward前后raw contact cache差异。只有全32-state原始next-step对照通过才允许正式score。来源：[MuJoCo consistency/reproducibility](https://mujoco.readthedocs.io/en/stable/computation/index.html#consistency-in-mjdata)。这不是允许下一步结果误差的阈值放宽。
# 完成状态（2026-09-09）

第一阶段已完成：8 个 episode、32 个状态、256 条评分全部通过。结论为暂不按当前 OTC 配方扩大实验；详见 [RESULT.zh.md](RESULT.zh.md)。以下保留原设计及执行中的修订记录，不代表仍有作业待运行。
