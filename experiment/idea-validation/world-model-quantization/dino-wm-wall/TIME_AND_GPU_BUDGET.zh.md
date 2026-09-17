# Wall 时间与 GPU 预算预估

2026-09-09；正式运行前估算，不是新benchmark结果，不提交作业。用户允许多个job并行，单个job最多2×A100；1×A100是默认。下述日历时间假设同时可获得4张A100（4个单卡job或2个双卡job），不是已核实的NSCC可用容量或全局配额。

## 实测参照与模型

来源：`artifacts/16170841.pbs101/planning_times.jsonl`，7个完整CEM记录；同目录`stopped_summary.json`与`qstat_after_stop.txt`。旧运行FP32、50cases、H5、300candidates、top30、10CEM iterations、单A100。

- 7轮CEM合计4450.5304秒，平均635.7901秒/50-case MPC round。
- 摊到一个case一轮CEM为12.7158秒；摊到一次300-candidate scoring约1.2716秒。
- 后一个数字包含旧CEM的采样、elite更新、内部env evaluator等摊销，不是单独测过的model forward latency。
- 50cases×12轮=2.1193 GPU-hours CEM；11配置×50cases×12轮=23.3123 GPU-hours CEM。
- cal probe上限24episodes×2points×3CEM pools×36site/bit interventions=5184 scoring calls，旧速度摊销1.8311 GPU-hours。
- dev probe若覆盖同样36interventions：20×2×3×36=4320 scoring calls，旧速度摊销1.5259 GPU-hours。
- cal/dev合计9504 scoring calls不含reference、block-output hooks、统计计算与IO。3个signal尽可能在相同probe中提取，不机械乘以3。

新runner使用episode batch=1并可独立成功早停，移除CEM内部env evaluator；旧runner持续处理整个50case batch。这些变化可能降低时间。反过来，hooks、CPU/GPU拷贝、保存、quantized weight切换、启动与环境replay可能增加时间。W4/W8 emulation提前materialize权重，禁止每次forward重复量化。预算使用约6–12实际MPC rounds与约1–1.5倍旧单case耗时的情景，再加启动/诊断余量；不是保证的上下界。checkpoint加载复用到同一worker的多个episodes，不为每个episode重建runtime。

## 第一轮分项（GPU-hours）

| 阶段 | 预估 |
|---|---:|
| Smoke、FP32 bridge、replay/no-op GPU验证 | 0.5–1.5 |
| Dev: FP32/all-W4/all-W8，3×20episodes | 1.5–4.5 |
| Cal FP32轨迹、36组probe、dev信号诊断、必要all-sites诊断 | 5–9 |
| Held-out pilot: 11配置×50episodes | 14–38 |
| 第一轮合计（圆整规划范围） | **25–55** |

未预先计入任意次数的失败重跑；新增重大bug或GPU OOM会使此范围失效。该预算不是一次性提交全部作业的计划：先smoke，再用实测成本刷新下一阶段。

## Wall 总范围与日历时间

为消除“整个实验”的歧义，分成以下累计里程碑：

| 里程碑 | 范围 | 累计GPU-hours | 4卡可同时获得时的GPU阶段日历时间（含依赖间隔，不含排队/实现） |
|---|---|---:|---|
| 工程/量化筛查 | Smoke + dev FP32/W4/W8 | 2–6 | 1–3小时 |
| 第一轮完整pilot | Cal+probes+50个held-out test episodes全对照 | 25–55 | 10–24小时 |
| Wall核心确认 | 第一轮 + 新200–400episodes/配置的独立确认情景 | 80–240 | 2–5天 |
| Wall数值完整实验 | 核心确认 + 2份新calibration fits的复验 + H10/E5小规模stress test | 120–320 | 3–7天 |

confirmation的200–400是工作量情景，不是已经算出的充分样本量；假定仍评估最多11配置，约追加50–190 GPU-hours。真实N需由预设最小有用收益、paired discordance和多重比较power决定，不能根据test显著性增加样本。若只有2–3pp微弱差异，N和总预算可能远超过此表；若现象清楚且pilot反证充分，可提前不推进后续阶段。

复验情景：2份独立24episode calibration，在新50个evaluation episodes分别测RankCal、Local-MSE、Score-error（FP32/control reference能在完全相同条件下复用）；连同probe约追加20–35 GPU-hours。H10情景：新calibration/同规模probe与至多6配置×50episodes，约追加20–50 GPU-hours；不假设耗时精确随H翻倍。两者是后续资源情景，尚未冻结独立protocol。

从现在起还需runner、量化/record hooks、replay和测试分析实现，规划**1–3个工作日工程时间**；复杂问题可能更久，这不是测得的开发工时。

- 首个可讨论的works/no-signal结果：规划**2–4天 + NSCC排队时间**。
- 较可靠的Wall核心结论：规划**4–8天 + 排队时间**。
- Wall完整数值实验与报告：规划**5–12天 + 排队时间**。
- 若代码已验证就绪，分别只看上表GPU阶段时间。

如果只有1张卡可获得，GPU阶段总时间接近累计GPU-hours（第一轮约25–55小时，Wall完整约5–14天），另加实现、分析和排队；2卡理想计算下限是GPU-hours/2，4卡是/4，但有串行依赖，不能将它们当承诺。

## 并行策略

1. 一个1×A100 smoke，核验后再扩展。
2. FP32/W4/W8 development可开3个独立单卡job；calibration FP32轨迹可作为第4路，在smoke通过后并行生成。
3. calibration数据冻结后，按site/bit或record shard分发probes；各job共享相同manifest与随机数定义，不混用临时结果。
4. allocation冻结后，按model map或episode shards并行跑test；每个配置使用相同episode IDs与CEM innovations。每个job内部按预先定义block混合/轮换可比方法，记录node与run order，避免method固定绑定硬件/时间。
5. 默认1×A100。双卡job若采用两个独立worker，每个绑定一张GPU并处理独立shard，两个worker都必须持续有工作。不假设原DINO-WM代码具备双卡加速；不得申请2卡后只启动一个单卡进程。
6. GPU-hours=allocated GPU数×job实际walltime的求和；多个job并行会减少elapsed time，不自动减少GPU-hours。若后台账单单位不同，应与NSCC使用记录分开核对。

## 结论边界与native部署

第一轮可以发现明显收益或有价值的失败现象，但50episodes很可能给出“尚不确定”。确认结果也允许三种结论：支持机制、不支持当前实现/假设、证据不足。不能保证在某个日期得到二元works/not-works。

本表“Wall完整”限定为W4/W8 numerical allocation机制、独立确认与指定robustness；不包含PushT、重新训练、activation quantization、多backbone，也不包含真实低比特backend工程。没有核实可用kernel/layout前，无法可靠估算native适配工期。

若后续native backend已经可用且无需自写kernel，4个关键配置×50episodes与数值一致性/资源测试可暂留**额外10–30 GPU-hours、1–3天工程与核验时间**的低置信度预留；这是有条件预算，不是总预算上限。若需要kernel适配，这个范围不成立。

下一次估算刷新点：smoke给出新runner的per-CEM/per-record latency、reference loading、IO、实际MPC轮数；用它们替换本文件的旧速度摊销，再决定每个job的walltime。不要自动采用原proposal的12 GPU-days。
