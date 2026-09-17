# RankCal / DINO-WM Wall fast screening 结果

2026-09-09。状态：全部 17 个 jobs 完成，200 条完整 model-episode 轨迹（其中 held-out 168 条）、probe metrics、joint fidelity 与冻结配置均通过核验。

## 研究判断

最终观察为 weak/mixed signal；资源决策为 **no-go：按快速筛选策略不扩大当前方案**。RankCal 比自建 LocalMSE comparator 的成功率高，但与 ScoreError 持平；两个固定 random allocations 分别为 20/24、17/24，RankCal 的 19/24 位于两者之间。RankCal 在联合量化下的全候选排序误差较低，却没有同时领先 LocalMSE 的 elite/action fidelity。FAST_SCREEN.zh.md 预先声明的方向一致且超过同预算对照的证据标准未满足。

这是对本轮具体实现的 go/no-go 判断，不是证明所有 ranking-aware quantization 都无效。

## 已冻结的实际实验

- 官方 DINO-WM Wall checkpoint，epoch 65；source revision `0a9492fa12044b852ae9e001cc74604b79c8bb0c`。仅 Wall / wall_single，无训练。
- FP32 reference。CEM 5 iterations、300 candidates、30 elites，horizon / execution interval 都为 5 model actions，frameskip 5，最多 12 MPC rounds / 300 environment steps。所有方法使用相同 planner。
- success：执行 MPC 边界上的 xy goal distance 严格小于 4.5；失败案例保留完整轨迹。
- weight-only、symmetric per-output-channel RTN；FP32 activations。12 encoder blocks + 6 predictor blocks 内的 eligible Linear weights，其他参数不变。
- 五个 mixed allocations 各保留 encoder 3 blocks、predictor 2 blocks 为 W8，其余 eligible groups 为 W4。共同 eligible logical weight cost 为 26,817,472 bytes，含 scales。该数字不包括所有模型参数，也不等于实测 device memory。
- calibration 8 episodes 决定配置；development 8 episodes 用于资源门槛及机制检查；held-out 24 episodes 使用独立 target indices / seeds。五种 mappings 互不重复，测试提交前已保存 TEST_FREEZE.json。
- 两个 random allocations 使用预先固定的 seeds 71001 / 71002，没有按表现选择 seed。

LocalMSE 是本实验自建的 block-output NMSE allocation comparator，不能说成 DINO-WM 原文采用的 quantization allocator。ScoreError 使用 planner scalar-score NMSE；三个信号均比较 site-only W4 与 W8，按收益和固定 3/2 配额分配。

## Held-out 闭环结果

同一批 24 targets，配对比较；test 没有用于配置拟合。

| 方法 | 成功 / 24 | 成功率 |
|---|---:|---:|
| FP32 | 22 | 91.7% |
| all-W4 | 16 | 66.7% |
| RankCal | 19 | 79.2% |
| LocalMSE | 15 | 62.5% |
| ScoreError | 19 | 79.2% |
| Random1 | 20 | 83.3% |
| Random2 | 17 | 70.8% |

RankCal 相对 LocalMSE 为 5 wins / 1 loss / 18 ties，净提升 16.7 percentage points；保守 95% paired-difference CI 为 [-17.8, 45.0] pp，exact two-sided McNemar p=0.21875。相对 ScoreError 为 2 wins / 2 losses / 20 ties，净提升 0 pp，CI [-29.1, 29.1] pp，p=1。

相对 Random1 为 1 win / 2 losses，净 -4.2 pp；相对 Random2 为 5 wins / 3 losses，净 +8.3 pp。两个 random maps 是两个固定配置，不能把它们与 RankCal 当作大量独立 allocation 重复，也不能据此断言随机分配普遍更强。

这些是 exploratory 个别比较，没有多重比较后的显著性结论；CI 宽，不能由未显著推断等效。RankCal 相对 all-W4 的净提升为 12.5 pp，但它使用更多 W8 权重，仅这个提升不能证明 allocation signal 的独特价值。

## 联合量化后的机制检查

在 development FP32 轨迹的 26 个固定 candidate pools 上，完整应用各 allocation；每个 visited MPC point 包含 CEM iterations 1 和 5，各层按 pool → point → episode 等权聚合。下表是 8 个 development episodes 的描述性均值，不把 26 pools 或 44,850 candidate pairs 当成独立实验样本。

| 方法 | 全候选 E2 disagreement ↓ | Elite overlap ↑ | Elite-mean action MSE ↓ | Score NMSE ↓ |
|---|---:|---:|---:|---:|
| FP32 | 0.00% | 100.00% | 0 | 0 |
| all-W4 | 42.33% | 25.52% | 0.081242 | 3585.962 |
| all-W8 | 1.31% | 97.08% | 0.001719 | 0.000519 |
| RankCal | 25.35% | 42.40% | 0.044385 | 8.351 |
| LocalMSE | 25.92% | 43.75% | 0.043846 | 18.434 |
| ScoreError | 27.46% | 39.69% | 0.049966 | 29.732 |
| Random1 | 27.80% | 38.85% | 0.047278 | 91.185 |
| Random2 | 37.47% | 31.35% | 0.065962 | 2461.839 |

E2 比较所有 unordered candidate pairs 的 score-difference signs，保留 exact ties。Elite overlap 比较 stable top-30 集合；action MSE 比较其均值 action sequences，因为该 CEM 执行 elite mean。Score NMSE 用每个 FP32 pool 的 score mean-square 归一化，与 block-output NMSE 不是同一个量。

RankCal 的全局 E2 在五个 mixed allocations 中最好，但比 LocalMSE 仅低 0.565 percentage points，elite overlap 则低 1.354 pp，action MSE 高约 1.2%。因此不能声称全候选排序目标与实际 planner 选择完全一致。相对 ScoreError，RankCal 的这四项离线指标都较好，但本轮 held-out success 没有净改善。

## 假设、已有证据、待定项

| 原研究假设 | 本轮证据 | 仍不能推断 |
|---|---|---|
| 排序影响比 local reconstruction 更适合 allocation | RankCal 比 LocalMSE 的全局 E2 略低，held-out success 多 4 例；elite/action 指标没有同时领先 | 排序信号普遍优于 LocalMSE 或 ScoreError |
| 单点信号在联合量化后仍有用 | 完整 allocation 的 E2 与 Score NMSE 有方向性支持；calibration LOO 三个信号均 8/8 mapping 不变；RankCal 的 dev/cal top-sites 为 5/5 overlap | site effects 可加、联合最优、跨配置稳定 |
| 收益迁移到新 episodes 的闭环表现 | 独立 24 targets 上观察到相对 LocalMSE 的改善，但对 ScoreError 没有净提升，Random1 也达 20/24 | 跨任务、跨 layout、跨 checkpoint 的泛化；确认性统计结论 |

本轮已经实现数值 emulation、冻结 allocation、闭环评估及独立核验。Native low-bit kernels、真实压缩部署、CEM10、更多 calibration 重复、更多 benchmarks 与 confirmatory 样本均未执行。按用户的快速止损偏好，不自动增加这些实验。

## 工程核验与资源

Development FP32 / W4 / W8 分别 7/8、5/8、6/8，通过测试前的资源门槛。Calibration 24 pools、development 26 pools，共 1,800 site/bit rows；独立 NumPy 重算 probe metrics 与 allocations 通过。Joint-fidelity 8 methods × 26 pools 独立重算通过，FP32/reference 与 restore 检查通过。

闭环检查包括所有 episode 的 targets、seeds、checkpoint identity、动作/状态形状、有限性、执行步数和成功 predicate；最终 7-method held-out verification 为 verified=true。Freeze audit 为 audit_pass=true、7/7 jobs、missing=[]、errors=[]，检查实际 bit mapping 与归档代码；source 比较使用 normalized text，未做 SHA-256 检查。

本轮实际 **9,565 allocated GPU-seconds = 2.65694 A100 GPU-hours**；含上一轮 smoke 的累计值为 **2.70139 GPU-hours**。本轮 17 个 jobs 全部 Exit_status=0，每个 job 为 1×A100，最多 4 jobs 并发。GPU-hours 按分配 GPU 数 × PBS 实际 walltime 累计，包含启动和目标准备，排队不占用 GPU-hours。

2026-09-09 Singapore / cluster local time：首个 job 12:42:21 启动，最后 job 14:01:50 结束，实验执行跨度 **79 分 29 秒**；14:03 已完成最后的本地冻结配置与资源核验。这段跨度包含阶段间排队、传输与分析，**不包含此前工程实现时间**。本轮低于原先预留 5–10 GPU-hours，主要由于多数 episodes 提前成功、实际只产生 50 个 cal/dev pools，且不同方法在独立 GPU 上并行；不能把此次时间保证推广到其他任务。

| 阶段 | Allocated GPU-seconds | GPU-hours |
|---|---:|---:|
| Targets 准备 + development + calibration reference/probes | 3549 | 0.98583 |
| 7-method held-out + development joint fidelity | 6016 | 1.67111 |
| 本轮合计 | 9565 | 2.65694 |

分工：subagents 负责 probes/stability、runner 与独立 verifier、PBS 监控归档和结果审读；parent 负责调度、冻结决策、交叉核验和最终解释。没有重新运行 IdeaSpark pipeline。

本次 PBS 已完成 development 闭环 jobs 的 sampled device memory maximum 为 6564 MB、平均 sampled SM utilization 93–100%；calibration probes 为 12820 MB、99%。这些是 PBS 采样指标，不是 PyTorch allocated memory，也不能当作理论 FLOPS 利用率。此前 smoke 中同卡两个 workers 的吞吐为单 worker 的 0.888 倍，因此保持每卡一个 worker。

量化在 FP32 中数值模拟；不作 native W4/W8 加速或显存节省结论。各方法闭环耗时还受成功早停和实际 MPC rounds 影响。

## 证据入口

- `TEST_FREEZE.json`：预先冻结的 allocations 和 held-out targets。
- `jobs.json`、`artifacts/screen/resource_summary.json`：PBS 状态及实际资源。
- `artifacts/screen/verify_pilot_test.json`：最终配对结果、CI、逐 episode success。
- `artifacts/screen/joint_verification.json`：独立重算的完整 joint metrics。
- `artifacts/screen/freeze_audit.json`：配置、targets 与源文件一致性审计。
- `artifacts/screen/cal_probe_verification.json`、`dev_probe_verification.json`、`stability.json`：校准与稳定性证据。
- `artifacts/screen/artifacts/<jobid>/`：实际运行代码、stage spec、PBS 记录、日志、逐例轨迹与数值输出；raw tar archives 保留在 artifacts/。
