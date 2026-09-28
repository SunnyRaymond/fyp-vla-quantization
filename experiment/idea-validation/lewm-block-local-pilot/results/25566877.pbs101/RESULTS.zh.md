# LeWM 分块 predictor 配对结果

PBS `25566877.pbs101`，NVIDIA A100-SXM4-40GB；完成 15/15 fits。

**主相对质量 gate：NO-GO。**
指标均为真实LeWM固定候选动作bank的predictor评价，不是任务成功率。

## 结果解释

本轮没有支持“在 LeWM 的现有表示上，通过正交换坐标与小块 predictor，提高同预算预测质量”的猜想。主比较 Global16 versus 参数匹配 Flat 的配对 Top30 recall 差值为 -0.83 percentage points（95% CI -3.33 至 +1.67）；standardized elite regret 差值为 -0.0165（95% CI -0.0551 至 +0.0523，越低越好）。两项区间均跨0，未达到冻结的 +5 points / -0.05 改善门槛。Regret 只有9/16 episodes改善，另外两个training seeds的整体配对方向也退步。

动作响应是更明确的负面信号：Global16 的 bank-centered terminal response MSE 比 Flat 高0.0837（95% CI +0.0514 至 +0.0930），response cosine低0.0797（95% CI -0.0970 至 -0.0631）。这表示它对不同动作候选造成的变化拟合更差。分块的全状态 relative MSE 约0.092–0.099，却仍有top30 overlap为0的blocks、负的排序相关性；较小的状态误差不能替代规划需要的动作区分能力。

通信与换坐标的次要对照在response MSE上有改善方向：Global16 versus Local16为 -0.0246，learned Q versus identity为 -0.0150；但两项对照的elite regret区间均跨0，没有形成可靠的主排序改善。它们是探索性的机制线索，不能替代失败的主比较；identity对照还少了36864个可训练参数。

速度问题这次已用官方heavy predictor实测：A100上B300的完整五步预测，Global16为2.31–2.73 ms，官方teacher为对应19.08–20.35 ms，满足cheap门槛；Flat为1.39–1.40 ms。因此分块确实比官方teacher快，但速度优势不能归因于分块，而且没有解决质量问题。这里不是完整CEM或闭环延迟。

实现检查11/11通过，15个trained models的因果检查均通过；learned Q已偏离identity且训练后保持正交。训练loss明显下降并满足预定下降比例检查。这些证据排除了本轮发现的接口/因果性/正交性错误，但不证明优化已达到全局最优，也不证明学到了正确因素分解。

严格结论限定于冻结encoder的192D CLS、线性正交adapter、6×32D块、16D线性共享消息、当前训练目标和3000-update预算。不能据此否定重新学习视觉表示或非线性因素分解。按冻结停止条件，本轮结束于Stage A，未执行Stage B/C，因而没有成功率改善或下降的实测结论。

以下逐模型表是各seed内96个blocks的边际median；主比较先对episode内blocks及固定seeds作配对差值median，两者不是同一个统计量，不能通过相减边际median重建主比较。

192D状态保维正交换坐标、6×32D局部更新、16D消息；Flat与分块保留相同三步latent与已消费action窗口。旧balanced_base仅直接读当前action，因此与旧模型的差异不能纯归分块。

|Seed|模型|Top30 recall median|Elite regret median|Response MSE median|B300 predictor ms|
|---|---|---:|---:|---:|---:|
|20300901|balanced_base|0.5667|0.3613|0.9343|1.544|
|20300901|flat_adapter|0.5167|0.3579|0.8980|1.391|
|20300901|block_adapter_global16|0.5000|0.4523|0.9467|2.313|
|20300901|block_adapter_local16|0.4333|0.5522|0.9609|2.369|
|20300901|block_identity_global16|0.4333|0.6159|0.9588|2.328|
|20300911|flat_adapter|0.5333|0.3939|0.8950|1.401|
|20300911|block_adapter_global16|0.5000|0.4854|0.9471|2.728|
|20300911|block_adapter_local16|0.4667|0.4770|0.9618|2.380|
|20300911|block_identity_global16|0.3667|0.7294|0.9530|2.325|
|20300911|balanced_base|0.5333|0.4055|0.9066|1.552|
|20300921|block_adapter_global16|0.4667|0.5583|0.9480|2.324|
|20300921|block_adapter_local16|0.3833|0.6264|0.9565|2.378|
|20300921|block_identity_global16|0.4000|0.6227|0.9598|2.325|
|20300921|balanced_base|0.5667|0.3507|0.9351|1.561|
|20300921|flat_adapter|0.5333|0.3779|0.8813|1.405|

## 主比较与诊断对照

每个episode先对六个blocks取配对差值median，再对三个固定training seeds取median；以下bootstrap以16个episode为单位。candidate不是独立实验单位。

|对照（treatment-control）|Top30 delta / 95%CI|Elite regret delta / 95%CI|
|---|---|---|
|block_vs_flat|-0.0083 / [-0.033333333333333326, 0.016666666666666663]|-0.0165 / [-0.05513580143451691, 0.052277058362960815]|
|communication|+0.0000 / [-0.008333333333333331, 0.033333333333333326]|-0.0034 / [-0.03288847813382745, 0.028369277715682983]|
|coordinates|+0.0083 / [0.0, 0.033333333333333326]|-0.0079 / [-0.042215511202812195, 0.004424825310707092]|
|old_recipe|-0.0417 / [-0.07499999999999998, 0.0]|+0.0569 / [0.03137636464089155, 0.11261293292045593]|

## 冻结判据与边界

- regret_effect: FAIL
- top30_effect: FAIL
- regret_ci: FAIL
- top30_ci: FAIL
- improved_episodes: FAIL
- all_seeds_nonworse: FAIL
- no_response_floor: PASS
- integrity: PASS

Stage B: NOT_RUN_GATE_FAILED；Stage C: NOT_RUN_GATE_FAILED。
未运行的CEM/闭环不能由latent误差或候选bank排序替代。相对改善不替代绝对fidelity门槛；固定三个training seeds不代表训练随机性总体。

正交adapter参数包含在预算；identity诊断arm少36864个可训练参数。推理计时计入实际旋转、消息和反变换，Q作为冻结权重物化；排除encoder/CEM/environment。

数据：直接复用25213164 reconstructed balanced rows；新的16个episodes来自冻结shuffle1000之后，排除训练及所列旧任务manifest。
Episode/task-level only. The official dataset-wide scaler reads train columns that can include selected evaluation episode rows; do not claim row-level preprocessing isolation.

Reuse existing staged official checkpoint and interface probe. Remote staging has no .git; historical commit provenance is not a currently verified HEAD. Exact current CEM source is saved as source_reference/cem.py and runtime source snapshots.

原始summary.json、逐arm evaluation JSON、training JSON/checkpoints和fresh_rows.pt均保留于本PBS run目录；本地仅取回小型摘要/报告。

方法来源：Kassis et al. (2026), [Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065)，当前v2，实验设计流程来源。
