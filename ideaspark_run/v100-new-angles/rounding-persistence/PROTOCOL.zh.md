# Matched temporal rounding persistence：冻结最小协议

2026-09-13，任何本候选模型输出之前冻结。采用适配pipeline的C02 matched control与C04 cross-term分解；独立设计审查见 ../new-cut-shortlist/PERSISTENCE_MATCHED_DESIGN_AUDIT.zh.md。多步误差积累本身已有PTQD/AccuQuant先例，不宣称新quantizer。

研究假设：同一个有限W4 SR draw set在10-step flow中的时间assignment，可能改变coherent forcing与最终action drift。只改变时间coupling，不改变bits、scale、每个step的draw multiset、模型结构或单条trajectory的forward数量；不平均不同trajectory的actions。

资产：SmolVLA checkpoint `lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de`，weights SHA256 `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`；LeRobot0.4.4/torch2.6.0+cu124，沿用已验证runtime/source identity与不可修改的flow_screen.py（SHA256 `ddf6e02ceb6b27a86ac479b54a6b24b5351342876916f39709504034efdb45a9`）。Dataset `lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`。额外现有video002身份来自conditional-marginal的已核验extension，需绑定其parent chain。

样本预选规则：LIBERO task0..5各1个episode，按episode ID升序取首个未用者，排除 `[0,18,22,1,4,5,2,3,34,6,38,40,33,58,11,19,35,44,45,48,85,88,21,37,59,80,49,50]`，取原prep约定的中间frame。仅用现有video000/001/002；若该固定episode所需文件缺失就resource_blocked，不为缓存可得性改选。CPU准备登记确切episode/frame/文件SHA，GPU前补冻结manifest identity。保留真实8D normalized state；模型max_action_dim32、chunk50，全部积分但评价仅physical7维前8个positions的**normalized action coordinates**，不称该checkpoint部署执行horizon。

同一初始noise用于所有arms及draws，两个noise seed **2201/2202**，各用独立CPU torch.Generator产生float32 Normal[50,32]，跨state复用相同两noise以配对。K=10，t=1,.9,...,.1，dt=-.1，与official Euler一致。保持同一raw condition、processor、prefix cache和完整32D状态更新。

量化仅现有helper的expert组（必须112个实际Linear），不改VLM/prefix权重、bias、norm、action input/output projections或其他参数。每weight per-output-row absmax/7，signed[-7,7]；zero row scale1/code0；SR为floor(w/scale)+Bernoulli(frac)，clamp后FP32 dequant。预先固定3套SR weights，seed **2101/2102/2103**，各独立CUDA Generator按排序module名连续抽样，整次screen复用，不按state/time调scale或重抽。RTN按同grid torch.round。保存每个draw/layer原weight、scale、integer code、dequant identity与实际绑定/写回receipt；实际draw snapshots保留在compute artifact。

8个free-running arms：FP32、RTN、Frozen0/1/2、Cyclic0/1/2。Frozen d在每个step都用draw d；Cyclic d在step t用draw `(d+t)%3`，t从0开始。每条都从同一noise起积分10步并保留各自x_t/v_t，不能把不同trajectory endpoint平均成输出。每step，三条frozen与三条cyclic调用同一个draw multiset，保存实际schedule/digest核验。

另外在同一FP path x_FP[t]上，用每个draw各计算一次field，保存v_Q[d,t]；定义e[d,t]=v_Q[d,t]−v_FP[t]。这组30calls/condition用于共同路径诊断，不能替代各臂free-running轨迹。Common path上的逐step marginal field MSE由exact multiset设计匹配，不做事后fairness挑选。

评价单位是6个state，先在每state内平均2个noise及3个draw。Primary `E_F/E_C` 是各自free-running endpoint相对FP endpoint的MSE，平均前8×7坐标。Common-path `S_F/S_C` 是按各schedule对dt*e求时间和后平方，再平均noise/draw/前8×7；`D`为同一误差的sum_t dt²e²，必须由CPU核对两schedule相等，`S-D`只是off-diagonal cross-term分解，不把forcing直接当nonlinear endpoint。

所有6state的E_F和S_F均须>1e-12，否则inconclusive_degenerate，不删state。至少 **5/6相同state** 同时满足 `S_C<=.75*S_F` 与 `E_C<=.90*E_F`，才给scope_limited_preliminary_go；不满足为mechanism_no_go。RTN误差照实列出，不能用RTN比较替代主门槛。无论结果如何均停止，不改draw、noise、step count或threshold。

工程对照预先固定在第一个state/第一个noise：manual FP与official sample_actions(noise固定) allclose(atol=1e-6,rtol=0)；RTN复用与每step重新装载同一RTN snapshot allclose同容差。全体输入/noise/time/schedule/hash一致、finite、prefix cache无污染、实际quantized参数readback与每臂restore exact。工程失败不能写科学no-go。

资源上限：CPU prep5min、single V10010min（内部最多540s）、CPU raw replay5min。预计6×2×(8+3)×10=1320 suffix calls，另20个工程control calls；不得因为不够时间扩大walltime做完整验证。3套weights的存储/切换开销须保留，只是fake quantization，不宣称native kernel加速、真实cache节省、task success或通用调度收益。

