# Week ending 2026-10-04：周报实验索引

范围：2026-09-28 至 2026-10-04；依据 [Notion 周报](https://app.notion.com/p/12844c7da96583aab809818763fa7f0a) 的 Main results 和 Progress 整理。只保留其中提及的本周实验；历史周报实验及所有 reproduction 目录继续保留。

## 实验与结果

| 周报位置 | 实验与入口 | 核心结果 | 简要分析 |
|---|---|---|---|
| Main results 1 | [first_frame profiling](fastwam-libero-bottleneck/RESULT.zh.md) | 暖态完整调用约 637–640 ms；10 次 action forwards 合计 523.12 ms，约占 82% | 优先优化完整 action loop；细分计时有扰动，原生调用决定总延迟 |
| Main results 1 | [同观测 first_frame / IDM 比较](fastwam-idm-libero-bottleneck/RESULT.zh.md) | 完整调用中位数 643.19 / 1158.59 ms；配对比值 1.804× | IDM 额外执行 10 次 video denoising；属于同一 Optional IDM checkpoint 的模式比较 |
| Main results 2 | [固定条件 K/V cache](fastwam-action-parallel/RESULT.zh.md) | 642.77→591.82 ms，配对 1.0864×；逐步和最终输出最大差异 0；峰值增加约 42.8 MiB | 把固定 text/proprio K/V 移出重复 denoising loop 有效；每 chunk 准备成本已计入 |
| Main results 2 | [K/V 跨层 batching 对照](fastwam-action-parallel/RESULT.zh.md) | 准备 6.227→1.303 ms；相对简单 cache 的 action-phase 配对加速仅 1.0038×，额外峰值约 539 MiB | 准备阶段占比小，简单逐层 cache 更值得保留；text embedding cache 单独无整段收益 |
| Main results 3 | [Prefix-Picard 时间并行](fastwam-action-parallel/RESULT.zh.md) | 11 种 window/round 配方均未过 numerical gate；W10/R1 约 9.88×但误差大，W10/R10 约 0.99×仍未过 gate；batch10 仅多约 47.4 MiB | 少轮修正不能恢复依赖，多轮失去时间收益；BF16 运算分组与 batch shape 也引入差异 |
| Main results 4 | [W4A8 phase 初筛](fastwam-phase-dither/mechanism-screen/RESULT.zh.md) | MSE ×10⁻⁴：RTN 3.85256、independent 3.78977、shared-zero 3.75674、learned phase 3.58978 | 相对 independent 改善 5.28%，是冻结 draws 下的小信号；未达到预注册 10% 扩展门槛 |
| Main results 5 | [新 dither draws 的 A8 对照](fastwam-phase-dither/new-draw-a8/README.zh.md) | 同一 TEST observations，新 draws 2101–2104：independent 3.67186、locked phase 3.75312（×10⁻⁴） | Phase 相对 RTN 的收益从 6.82% 降为 2.58%，且比 independent 差约 2.21%；跨 draw 组不稳定 |
| Main results 5 | [完整 action 输出误差分解](fastwam-phase-dither/paired-followup/RESULT.zh.md) | RTN/phase 的 weight-path 项均 3.61510；activation 项 0.13482/0.14112；两倍交互项 +0.10264/−0.00310（×10⁻⁴） | Phase 的 activation 增量稍大，但减少正向耦合；输出级恒等式不能唯一识别逐 site 抵消机制 |
| Main results 6 | [完整 LIBERO-Goal 比较](fastwam-phase-dither/libero-goal-suite/RESULTS.zh.md) | 10 tasks × 50 states × 4 arms，共 2000 episodes。BF16 487/500（97.4%）、RTN 486/500（97.2%）、independent 483/500（96.6%）、locked phase 481/500（96.2%） | Learned 对 RTN 赢7/负12/平481，对 independent 赢7/负9/平484；当前 phase 配方没有成功率优势，小差值不构成统计显著性或普遍方法失败的结论 |

## 配置与证据边界

FastWAMOptionalIDM 固定 revision `7faa71108368fbb3b6885649f112af607427a2d4`，主要模式为 `first_frame`，A100 40GB、BF16、compile=false，输出 32×7 action chunk，闭环执行前 10 步后 replan。Profiling/cache/Picard 使用 10 action-denoising steps；PTQ 使用 20 steps，不跨协议直接比较时延。

固定观测时延、完整 action 数值误差和自驱闭环成功率分别报告。全部 PTQ 为 fake quantization + floating-point GEMM，没有 native low-bit kernel 加速证据。每个保留目录包含运行源码、冻结协议、配置/摘要、结果和相关作业日志；模型、环境及被 `.gitignore` 排除的二进制输出不包含在 Git 中。

## 整理范围

- PTQ 数据集中到 [fastwam-phase-dither](fastwam-phase-dither/README.zh.md)，保留 Q-WAM 动机、phase proposal、机制初筛、新 draws A8 对照、输出分解和完整 suite。
- 未在最新周报中提及的 810 配方 branch-output reuse、W4A4 结果及 task0 32-episode pilot 移到仓库外可恢复归档：`D:\Downloads\FYP-unreported-experiments-20261005`。混合 A4/A8 原始汇总保留在该归档，Git 中只保留其 A8 字段提取及明确来源。
- Task0 pilot 的 `closedloop.py` 继续作为完整 suite 的共享执行 helper；保留通用源码不表示保留该 pilot 的结果。
- 原 `ideaspark_run/` 中其余候选生成材料属于 research ideas，保留在本地，不纳入此次实验提交。与本周实验无关的嵌套仓库改动不提交。

下周的 weight/activation sensitivity analysis 和 WAM QAT 与 Yanlong 的讨论仍是计划，不列为本周已完成实验。
