# LeWM / Fast-LeWM × CEM 10/30：冻结实验结果

2026-09-27。实验完成，技术有效性 PASS；model × budget 的成功率交互仍不确定。本次不扩展第二个模型/预算网格。

## 结果

| Model | CEM rounds | seed42 / 43 / 44 成功数（各50任务） | 三个 seed 的平均成功率 | steady solve 中位数，秒 |
|---|---:|---|---:|---:|
| LeWM | 10 | 48 / 47 / 47 | 94.67% | 41.00 |
| LeWM | 30 | 48 / 46 / 47 | 94.00% | 122.97 |
| Fast-LeWM | 10 | 48 / 47 / 47 | 94.67% | 8.26 |
| Fast-LeWM | 30 | 49 / 49 / 48 | 97.33% | 24.51 |

时间列先取每 arm 的三次 steady solve 中位数，再取三个 seed 中位数；实际 active environments=50、native batch_size=1，300 candidates/top30。相同预算下 LeWM/Fast-LeWM 的每 seed 耗时比中位数为 4.96×（10轮）和 5.02×（30轮）。同一模型从30减至10轮，耗时约降至三分之一。时间包含 native solver 的 cost logging，排除加载、环境初始化和输入复制；这是固定初始 observations 的规划耗时，不能替代完整闭环系统延迟。

| 配对成功率差，percentage points | 点估计 | 任务层 bootstrap 95% interval |
|---|---:|---:|
| LeWM：10 − 30 rounds | +0.67 | [−6.00, +7.33] |
| Fast-LeWM：10 − 30 rounds | −2.67 | [−7.33, +1.33] |
| 交互：(Fast 10−30) − (LeWM 10−30) | −3.33 | [−10.00, +3.33] |
| Fast − LeWM，10 rounds | 0.00 | [−6.00, +5.33] |
| Fast − LeWM，30 rounds | +3.33 | [−2.67, +10.00] |

三个 solver seeds 是同一50任务的重复测量；10,000次 cluster bootstrap 重抽 source tasks，保留每任务的 seed 重复。不能把150 outcomes 当150个独立任务。任务已用于 Fast baseline，属于探索性诊断；区间不是 multiplicity-adjusted confirmatory inference，也不是 noninferiority/equivalence 证明。10轮时两个模型的总成功数相同，但逐任务 outcomes 并不完全相同。

## 已完成的后续初步实验

Fast-LeWM 的速度优势在本次共同 backend 与实际负载下稳定；30轮的成功率点估计最高。保留30轮，单独验证其 CEM 迭代内重复的 goal/current encoder 调用能否精确复用，是合理的有限工程实验。旧 LeWM 已有 exact iteration cache，因此这个后续不作为新的 research gap 或 novelty claim。缓存实验独立冻结，只验证成本、候选、actions 的 bitwise fidelity 和 fixed-observation latency；不从主实验转移闭环成功率保证。

该有限实验已完成，见 [Fast-LeWM exact cache 结果](../fastlewm-exact-iteration-cache/RESULT.zh.md)。8个 B1 fixed contexts 全部保持逐轮 candidates、costs 和最终 actions 的 bitwise fidelity；配对耗时降幅中位数为57.132%，最大 peak allocated memory 比为1.00。每solve的60次encoder调用降为2次实际计算与58次命中。此结果只覆盖B1固定输入；主实验的B50及closed-loop缓存收益尚未验证，不能把它的约2.33×与这里约5×的模型耗时比直接相乘。该pilot已收尾，本次不扩展第二个预算网格。

当前数据不足以证明 Fast-LeWM 比 LeWM 更依赖搜索预算，也不足以把 LeWM 降至10轮认定为无损替代。模型的训练、encoder、checkpoint 和 action interface 均有差异，5×不能唯一归因于 action-prefix architecture；不同 latent space 的 raw cost 不能直接比较。

## 执行与身份

- 主 GPU：`25569974.pbs101`，host `x1000c0s0b0n0`，A100-SXM4-40GB；PBS `F / Exit_status=0 / Stageout_status=1`，walltime `01:07:23`。12/12 arms PASS。
- Fast seed42/30 的49/50逐任务 outcomes 与先前官方复现完全一致；LeWM/30 的48、46、47全部通过预先冻结的至少45/50 usability gate。
- CPU准备：`25569830.pbs101` PASS。第一次 GPU `25569928.pbs101` 在模型加载前因 Python import scope 错误失败；修复执行错误后原冻结配置重跑，未据成功率调参。
- Fast source：`de3e9dac539f5bbe6ff1656a2fb00938d62a3c7d`；checkpoint `naiverer/fast-leworldmodel@f95379fe193c8bfc6a59c9d8437d5052bd72ff71`。
- LeWM historical source identity：`8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`；使用既有 official PushT object checkpoint。
- backend：isolated `stable-worldmodel==0.0.6`，共用 compatible Torch runtime；不是逐项重建论文环境。
- LeWM H5/block5/receding5；Fast H1/block25/receding1、beta0。均预测并执行25 primitive steps，history1、goal offset25、eval budget50。不同预算消耗 native RNG 不同，不声称后续任务或 replans 的 candidates 相同。
- 全数据 StandardScaler 沿用官方定义，不声称 preprocessing 对 evaluation episodes 隔离。GPU 利用率/显存每5秒写入 job.log；重操作全部在真实 PBS compute allocation。

完整设计见 [PROTOCOL.zh.md](PROTOCOL.zh.md) 和 [FREEZE.json](FREEZE.json)。机器可读结果位于 `artifacts/25569974.pbs101/paired_analysis.json` 与各 `arms/*/result.json`；这些分析在 GPU allocation 结束前完成。

实验设计流程参考：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. https://doi.org/10.48550/arXiv.2609.00065。流程引用不为本方法的有效性或 novelty 背书。
