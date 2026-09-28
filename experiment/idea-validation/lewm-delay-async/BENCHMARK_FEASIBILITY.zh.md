# 现有 LeWM benchmark 的固定延迟与 true-async 可行性

四个任务都可在原环境外加控制调度器：继续使用同一 checkpoint、dataset start/goal rows、episode budget、成功判据和 CEM，只改变从观测到动作应用期间环境是否推进。这是对原 benchmark 的新评估协议，不能把它直接称为论文报告的原始成功率。官方 [LeWM 仓库](https://github.com/lucas-maes/le-wm)列有 PushT、Cube、TwoRoom、Reacher 的预训练 checkpoint；[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel)提供对应环境。以下按当前冻结的 LeWM / stable-worldmodel 源码判断。

| Benchmark | 固定 K control ticks | true-async | 主要需核实的量 |
|---|---|---|---|
| PushT | 已实现，官方 50-env K0 已逐项复现 49/50 | 已实现调度器，待 N=1 基线与 GPU 运行 | control tick 为 0.1 s；首次无动作时的中性动作；RTF 与 deadline overrun |
| Reacher | 已实现调度器，官方 K0 待运行 | 已实现调度器，待 GPU 运行 | 从实际 `dm_control` control timestep × wrapper action repeat 取得 wall period；动作空间中点与环境步进时长 |
| OGBench-Cube | 可复用冻结 dataset start/goal rows、在动作应用前推进 K 步；尚未实现/实测 | 技术上可加同一调度器，尚未定义/实测可信 wall period | MuJoCo wrapper 的每 control tick 物理时长、无计划时 hold/neutral 动作、goal 必须仍来自 dataset |
| TwoRoom | 可复用冻结 dataset start/goal rows、在动作应用前推进 K 步；尚未实现/实测 | 技术上可按声明的外部周期调度，尚无已验证的原 benchmark wall period | control tick 的物理/语义含义、hold/neutral 动作是否合理、chosen wall period 的解释范围 |

Cube 的 goal 不能随意改成环境随机生成：`stable-worldmodel` 的[相关 issue](https://github.com/galilai-group/stable-worldmodel/issues/224)指出无 dataset 的 test-time goal 目前缺失。因而延迟协议应沿用原 evaluator 的 dataset goal，与同步基线严格配对。

`stable-worldmodel` 有一个[async environment stepping PR #290](https://github.com/galilai-group/stable-worldmodel/pull/290)，截至 2026-09-28 仍为 open，不能假设它已进入本项目固定的源码版本。当前 PushT/Reacher 使用外部调度器且不更换基础环境实现。固定 K 是仿真步年龄，不等于 wall-clock realtime；只有 true-async 还需报告实际 observation age、RTF 和目标周期超期率。若 RTF < 1，成功率仍可报告，但不得称达到该目标实时频率。

优先级：先完成 PushT 与 Reacher 的同步门槛、N=1 基线及固定/async 配对结果；若两者显示可辨认的延迟效应，再考虑 Cube。TwoRoom 适合做按步控制调度的补充，但必须先定义有依据的 wall period。
