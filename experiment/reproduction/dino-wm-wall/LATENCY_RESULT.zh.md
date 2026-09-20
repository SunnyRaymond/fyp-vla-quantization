# DINO-WM Wall latency profiling 结果

## 结论

PBS job `23486763.pbs101` 在一张 `NVIDIA A100-SXM4-40GB` 上正常完成，`Exit_status=0`，PBS walltime 为 `00:01:26`。两个 Wall cases 都在第二个 MPC round 后成功；完整被测 pipeline 为 `65.949 s`，Python 进程观测时间为 `66.129 s`。

本次最重要的结果是：planner 内部的主要 latency 不是 sampling、objective 或 top-k，而是 candidate world-model rollout；其中 predictor 占绝大部分。当前 cache idea 若要产生明显收益，应该直接减少或复用 predictor 的重复计算，并继续用 planner ordering / selected first action 作为正确性边界。

## 完整流程

| 阶段 | 时间 | pipeline 占比 |
|---|---:|---:|
| workspace 与 target 初始化 | 30.272 s | 45.90% |
| 完整 planning / final evaluation | 29.470 s | 44.69% |
| checkpoint 与 model load | 4.918 s | 7.46% |
| environment 创建 | 0.996 s | 1.51% |
| dataset load | 0.269 s | 0.41% |
| final videos | 0.251 s | 0.38% |

`workspace_and_targets` 是冷启动最大单项，但本次没有继续拆分。根据 source 调用链，它包含 validation dataset 的可用轨迹扫描、random-state target/layout 准备、environment prepare、planner 构造和 `plan_targets.pkl` 写出；其中哪一步占主导仍是待验证项，不能只凭这次记录断言。

## Planner 分解

- MPC total：`29.032 s`
- CEM total：`28.746 s`，占 MPC `99.01%`
- 11 个 CEM optimization steps：合计 `28.631 s`，平均 `2.603 s/step`
- candidate world-model rollout：`27.328 s`，占 CEM `95.07%`
- CEM diagnostic evaluator：`1.219 s`，占 CEM `4.24%`
- 22 次 objective：合计 `20.70 ms`
- 22 次 top-k / distribution update：合计 `24.42 ms`
- 22 次 candidate sampling：合计 `2.85 ms`

第一轮 MPC 执行完整 10 个 CEM steps，用时 `26.261 s`。第二轮在第一个 CEM diagnostic 已达到两个 cases 全成功，因此 early-stop，用时 `2.769 s`。这说明 task latency 会被 CEM early termination 强烈影响，不能只报告单一平均值而忽略 round/iteration 数。

## Candidate world-model rollout

22 次 candidate rollouts 分别来自两个 cases，每个 case 11 次；两者合计时间分别为 `13.702 s` 和 `13.626 s`。

| block | 调用数 | 总时间 | candidate rollout 占比 |
|---|---:|---:|---:|
| autoregressive predictor | 88 | 18.621 s | 68.14% |
| terminal predictor | 22 | 4.645 s | 17.00% |
| initial observation/action encode | 22 | 3.966 s | 14.51% |
| action replacement、concat、separation | 其余 | 0.097 s | 0.35% |

autoregressive 与 terminal predictor 合计 `23.266 s`，占 candidate rollout `85.13%`。因此 planner latency 优化应优先针对 predictor，而不是 `randn` sampling、objective 或 elite selection。

## Evaluator 与 environment

14 次 evaluator calls 合计 `1.680 s`：

- 11 次 CEM diagnostic：`1.219 s`，平均 `110.83 ms`
- 2 次 MPC execution：`279.29 ms`，平均 `139.64 ms`
- 1 次 final evaluation：`181.37 ms`

其中 environment rollout 合计 `1.161 s`，约占 evaluator `69.1%`；imagined rollout 合计 `380.35 ms`，约占 `22.6%`；achieved-state encoding 合计 `72.46 ms`。在当前 Wall/A100/300-candidate 设置中，environment execution 不是 planner 的端到端主瓶颈。

## Task 结果与资源

- cases：2/2 success
- deterministic env seeds：`1`、`100`
- 两个 cases 都执行 50 environment steps
- final goal distances：`2.7864`、`1.0331`；官方 success threshold 为 `<4.5`
- peak CUDA allocated：`3.994 GiB`
- peak CUDA reserved：`5.906 GiB`
- JSONL 写出开销：`39.29 ms`

PBS 的 `resources_used.ngpus=0` 与 `gpu.txt` 中的 A100、CUDA memory records 及成功 GPU inference 不一致，因此该字段只能视为 scheduler accounting anomaly，不能用来否定 GPU 执行。

## 证据边界

- 仅两个 Wall cases，P50/P95 只作 descriptive profiling，不代表稳定分布。
- 没有额外 warmup task；第一轮保留 cold-start 行为。
- 细粒度 GPU block 使用 `torch.cuda.synchronize()`，绝对 latency 包含 profiling perturbation，不应直接当作 production control frequency。
- decoder 已禁用；这里只测 official planning objective 所需路径和真实 environment/final videos，不包含 imagined-video decoding。
- 这是 FP32 checkpoint profiling，不是 quantized/native low-bit latency。
- 本次结果支持“predictor 是优先优化对象”，但不证明任何 cache 的 planner invariance、closed-loop success 或实际 speedup。

原始结构化结果位于 `artifacts/23486763.pbs101/`。
