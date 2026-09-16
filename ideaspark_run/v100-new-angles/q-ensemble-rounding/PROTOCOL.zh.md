# TD-MPC2 existing-Q stratified rounding：冻结最小协议

2026-09-13，尚无本候选模型输出。完成独立设计审查后冻结；CPU真实reset输入和小checkpoint准备独立进行。

目标是控制五个已有trained critics的joint rounding law，改善实际planner random-two-Q avg的误差，不增加member、不改planner、不重新引入不存在的variance penalty。`min`不是主指标。Stratified SR只是一种既有shared-randomness模式，novelty未认证。

固定官方TD-MPC2 source `e9f59321933cbc8e11a002b842adc7d4ffae8ff1`，checkpoint `dmcontrol/cartpole-balance-1.pt`，HF revision `73a50e2719ed8258c72c7d1fefd23b781d66e35e`、31,344,610bytes、SHA256 `4919e562d7f22f41a11118d1db1a0ebcb0e2b4681fc7fc594d772f0d5940869b`。model_size5、num_q5、state observation5、action1、compile=false、FP32、eval、no-grad。官方api_model_conversion如被需要必须记录，严格完整加载，不任意删key。

CPU输入：seed5201..5208各创建fresh official DMControl cartpole-balance env，只reset一次，按官方wrapper flatten顺序保存obs5。不调用step、不render、不训练或评估完整任务。实际dependency与source hash落盘；复用Python3.10/torch2.6的运行环境与官方Docker不同，不能称exact-environment reproduction。

一次FP缓存：CPUtorch.Generator seed6201生成8×64×3×1个Uniform[-1,1] candidate actions；各state从FP encode开始用FP next沿3步action sequence到terminal latent。固定seed7201+state_index只调用一次FP pi并缓存terminal action。四个arms、三个roundingseeds全部复用相同terminal latent/action。输入来自真实reset后FP模型可达状态，不是环境H3 rollout或ground truth trajectory。

Arms顺序：FP32、W4_RTN、W4_independent_SR、W4_stratified_SR。只改live `_Qs` 三个Linear的weight，实际shape必须[5,out,in]；每member、每output row独立absmax/7 scale，signed grid[-7,7]。bias、LayerNorm、encoder、dynamics、reward、pi和target Q均FP32。detach Q与live Q共享storage，按真实alias绑定检查，不把它当额外member。

Stochastic rounding用floor与fractional probability。independent为5个对应coordinate独立Uniform；stratified先对每coordinate抽U、独立随机permutation pi∈S5，使用 `(U+pi(j)/5) mod 1`。所有member边际Uniform，scale/grid完全一致。roundingseed4101/4102/4103；每个seed固定一套量化weights覆盖全部8states，不逐state重采或选seed。RTN/FP只计算一次并在raw三个seed位置重复。

`Q(return_type='all')`返回logits，必须用official two_hot_inv先逐member解码成5个scalar。primary对全部10个unordered member pairs分别求平均值的FP误差平方，再平均；这是官方均匀random-two subset误差的exact finite expectation，不能替换成5-member总体均值的误差。所有quantized arms与FP使用同一pair。

实现路径对照：每arm/state用固定seed8201+state_index设置并保存同一实际CUDA device的RNG state、在该device抽torch.randperm(5)前2个，然后恢复该device RNG调用official Q(avg)，对照已decode的对应两个member均值，逐元素allclose(atol=1e-5, rtol=1e-6)，保留max_abs记录。不能仅恢复CPU RNG。此容差在任何本候选Q输出前确定，用于FP32 value解码的正常roundoff；科学MSE仍独立FP64重算。此对照只验证原API语义，不作为额外统计样本；raw保存official输出及pair。所有weight restore逐元素exact，target与非Q参数digest不变。

科学单位为8个reset states。每state内先对64candidate、3roundingseed平均。记M为5member单独error²均值，C为10unordered pair的error product均值，A为10pairs的avg error²均值；独立CPU核对A=(M+C)/2。这个恒等式只用于拆分误差来源，不是科学预测。

冻结gate：8个state全部满足A_ind>1e-8且M_ind>1e-8才进入相对改善判断，任一退化则inconclusive_degenerate并保留全集。每state M_strat/M_ind∈[0.9,1.1]为finite-sample member fairness；少于6/8 fair则inconclusive_binding。机制支持必须同时满足：全集8-state median(1−A_strat/A_ind)≥25%，且同一个至少6-state joint集合同时满足fairness、A_strat<A_ind、以及(C_ind−C_strat)/2至少解释75%的(A_ind−A_strat)。有足够fairness但机制条件未过则mechanism_no_go。

只有机制gate通过后再判同成本RTN：全部8states的mean A_strat≤mean A_RTN，且至少6/8 A_strat≤A_RTN，两者均过才preliminary_go；否则mechanism_positive_practical_no_go。RTN误差不作为任何比值分母；即使A_RTN=0，绝对MSE比较仍有效，不因零baseline删除state或改为统计不可识别。原独立审查建议为RTN设置floor，root未采用此项，原因是这里没有除以RTN的计算。全部state仍保留。

只做一次single V100≤15min、内部≤12min screen；CPU准备≤10min，CPUraw replay≤5min。不会因go扩任务、因no-go调整bits/seeds/quantizer或进行完整闭环。资源/工程失败保留原job后按具体缺陷处理，禁止login-node workload与本机数值实验。
# Pre-output checkpoint amendment：CPU 64795

GPU 64793 的 seed 1 在 strict load 阶段失败，未产生任何 Q-value。CPU 64795 按预定 seed 2→3 顺序，只依据 strict-load compatibility 检查：seed 2 不兼容、seed 3 通过；没有 inference、没有按性能选择。原失败保留。

下文旧 seed 1 身份仅为原始冻结历史；执行 pin 现在为 `nicklashansen/tdmpc2@73a50e2719ed8258c72c7d1fefd23b781d66e35e/dmcontrol/cartpole-balance-3.pt`，31,344,610 bytes，SHA256 `0e2c0eada8f160dd65c8faaa5e4ca2f8f496104e21433e334d716b73257a52c2`。Manifest SHA256 `9240979105b919f6051f27261d00ff33652105f962d039728638caabb042d369`。Source commit `e9f59321933cbc8e11a002b842adc7d4ffae8ff1`、8 states、seeds、arms、gate 与资源上限不变。该修订发生在任何 Q-value 输出之前。
