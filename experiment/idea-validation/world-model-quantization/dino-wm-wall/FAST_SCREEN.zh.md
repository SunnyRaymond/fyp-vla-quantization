# RankCal / Wall 快速 go-no-go screening v0.2

执行状态更新：该计划现已获用户授权并执行，以下“无提交/仅设计”措辞描述编写时状态。实际协议、冻结门槛与执行证据以 `SCREEN_EXECUTION.zh.md`、`TEST_FREEZE.json`、`jobs.json` 为准。

2026-09-09：用户优先快速判断是否值得继续，允许negative/no-signal后换idea。此文件作为下一轮默认计划，替代v0.1完整pilot排期；旧文件保留作为较完整验证的参考。不运行pipeline、训练或GPU实验。

## 保留与缩减

| 项目 | v0.1 | 当前screen |
|---|---:|---:|
| CEM iterations | 10 | 5（所有方法统一） |
| Candidate / elite | 300 / 30 | 300 / 30 |
| Horizon / execution interval | 5 / 5 | 5 / 5 |
| MPC round cap | 12 | 12 |
| Calibration episodes | 24 | 8 |
| Development episodes | 20 | 8 |
| Held-out episodes | 50 | 24 |
| Calibration pools / MPC point | iterations 1,5,10 | iterations 1,5 |
| Random allocations | 5 | 2 |
| Held-out model configurations | 11 | 7：FP32、all-W4、RankCal、Local-MSE、Score-error、Random1、Random2 |

2个smoke保留。all-W8只在development筛查，暂不加入held-out。量化operator、block sites、encoder3/predictor2的W8配额、FP32 activations、split隔离、tie规则和CEM禁用内部env early-stop均沿用v0.1。阶段数据使用原namespace内更小的前缀，但manifest尚未生成，必须去重。旧50cases历史结果仅供耗时参考。

不同时缩短MPC cap或减少candidate数，以免把任务进一步变难而不易解释。CEM5可能改变reference与量化差距，先在dev检查FP32有足够成功空间；不能从screen宣称标准CEM10场景也成立。

## 逐级止损

1. smoke：确认数值、replay、输出和计时正确。FP32 adapter与旧实现的差异解释清楚；不直接把旧94%当reference。
2. 8个dev episodes：FP32、W4、W8可并行。若FP32本身明显失效，或W4与FP32几乎没有差距，标记screen不具辨别力，不自动投入后续实验。小样本相同成功率不能证明无量化损害；此门槛只决定研究资源是否继续。
3. 8个cal episodes：每episode前2个实际MPC points，CEM iterations1/5，最多32 pools。36site/bit probes共享3个signal计算；dev同样至多32 pools作稳定性检查。冻结3个allocations与2个random maps。
4. 24个新的paired held-out episodes，7个完整模型配置，共168个model-episodes。逐episode落盘，保存全部success/失败。report paired wins/losses及CI；不为显著性追加test样本。
5. 决策：RankCal若在success、elite/action fidelity上有方向一致的优势，且优于Local-MSE/Score-error并非仅一个随机map偶然较差，建议继续。无可见优势或分配极不稳定，则建议停止当前方案/换idea；这是go/no-go资源决策，不是对所有RankCal变体的数学否定。CI宽时写“没有足够信号，按快速筛选策略不继续”，不写“证明无效”。

## 新预算

旧实测635.79秒/(50cases×10 CEM iterations)；screen按CEM5线性近似6.3579秒/(episode×MPC round)。仅为换算，不保证端到端减半：较少CEM可能导致更多MPC rounds。

- 7×24×12 rounds 的闭环CEM参考成本：3.56 GPU-hours。
- 8×3×12 development：0.51 GPU-hours。
- calibration reference轨迹8×12：0.17 GPU-hours。
- cal+dev probes：(32+32)×36 scoring calls×1.2716秒≈0.81 GPU-hours。
- 上述合计约5.05 GPU-hours，未包含全部hooks/IO/启动；实际episode提前成功可减少计算。预留screen总量 **5–10 GPU-hours**，不是硬保证。
- 4张A100可并行且无排队、runner已就绪：约 **3–6小时**，含串行数据/分配依赖；仅1张卡约 **5–10小时以上**。若排队或smoke显示较慢，更新估计。
- 从当前尚无runner出发：工程实现与验证规划 **1–2个工作日 + 排队**；不能把3–6小时说成从现在起一定拿到结论。
- 不自动做confirmation、H10、独立calibration重复、PushT或native backend。

## 显存低的代码依据与性能预检

- `source/planning/cem.py`先循环CEM iterations，再`for traj in range(n_evals)`逐episode评分；每次rollout仅300候选。50cases不会同时形成15000-candidate batch。
- `evaluate_wall.py:30`全局禁用grad，CEM也使用no_grad，因此无训练backward graph与optimizer states。
- `evaluate_wall.py:56`禁用visualization-only decoder；使用DINOv2小型encoder+6层predictor和H5，不是大型video diffusion model。
- 旧4.06 GiB为PyTorch `max_memory_allocated`累计峰值，不是nvidia-smi设备总占用，也不表示GPU算力只用10%。allocator reserved、CUDA上下文/库等要另测。

smoke新增低成本吞吐检查：同一scoring workload分别1与2个独立worker共用一张GPU，进程绑定同卡、保持各自300候选与随机数；先检查输出等价，再比较总records/second、peak reserved、设备显存与GPU utilization。只有实测总吞吐改善才采用。显存能容纳不代表算力不饱和；不根据4GiB推断10倍提速，也不引入昂贵跨episode向量化重构。

默认仍每个job 1×A100。双卡job用两个独立单卡worker，不做DDP模型切分。记录每个job的分配GPU数和实际时长，正确累计GPU-hours。

状态：仅设计修订，无GPU job提交，无实验结果。
