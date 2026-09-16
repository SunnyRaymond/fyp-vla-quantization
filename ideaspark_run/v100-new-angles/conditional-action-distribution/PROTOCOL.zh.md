# Conditional action marginal：最小初筛协议

2026-09-13，所有本候选模型输出前制定。已完成独立审查并解决末尾所列三项澄清，本协议冻结，可据此提交一次最小GPU screen。

研究假设：W4 可能明显改变同一 Gaussian noise 的 action mapping，却未在预设低维有限样本 screen 中产生相应的 conditional marginal shift。此处不证明分布等价，不提出新 SWD/energy 方法，也不把不同 observation 的 action 池化。

复用 LeRobot v0.4.4、SmolVLA LIBERO checkpoint31d453f7edd78c839a8bbc39744a292686daf0de、dataset a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4。FP32 与 expert-transformer-only per-output-channel RTN W4 为两个 model arms，所有其他模块保持 FP32。num_steps10、完整action horizon50、noise dimension32。无训练、rollout、native low-bit performance 或 success 声称。

八个 fresh conditions：task0..3各排除旧 Flow manifest3个 episode 与 padded manifest2个 episode，再按episode ID取最早两个未用者；固定 frame=floor(length/4)。先按metadata选择再检查已有视频，缺失即resource_blocked，不换样本。新 assets 独立写入 conditional_marginal；固定两份排除 manifest、source IDs、frame、原始输入与源文件hash。

这里的dataset是 `lerobot/libero`，与总表中DINO-WM WallDataset的0–83已用、84–95 test_locked属于不同namespace，不能把Wall indices当作LIBERO episode ID排除。本候选只读取LIBERO资产；Wall test_locked完整保留。

每个 condition 与 model arm 使用同一64个预生成noise：CPU torch.Generator，A block seed1901一次生成32×50×32，B block seed1902独立生成32×50×32，再以A后B拼接；各condition复用相同固定noise集合。A/B之间无共享draw。每condition内FP/Q same-noise配对；跨组 marginal比较只用Q_A对FP_B与Q_B对FP_A。总共8×2×64=1024个action samples。noise重复不当作独立episode。

执行可用固定microbatch4，8个condition分开处理。首condition首4noise分别在FP和Q模型比较batch4结果与逐条结果，保存原始差值，每arm max_abs<=1e-5才通过batch实现gate；不通过不得改门槛，标implementation_inconclusive。每调用reset cache，同一batch排序与显式noise clone，完整FP输出与Q输出分别保存。量化按arm执行一次，分组前后保存真实Parameter绑定的bypass digest和exact restore gate，避免每个sample重复大范围hash。完整runtime/source/checkpoint/processor identity必须先落盘。

主切片固定为模型返回的 normalized physical first action7，不用后处理或挑选某个坐标。投影矩阵为7个坐标轴及9个NumPy default_rng(1903) Gaussian单位方向，所有condition共用，CPU验证保存实际矩阵和hash；不用看Q输出后选方向。

所有SWD、energy、Dpair、Dnoise及translation都使用同一 normalized first-action 7D tensor；MSE明确对7坐标取mean。num_steps=10指Euler denoising steps，projection count=16是距离的固定方向数，两者无关。

每condition分别计算，距离均使用FP64：SWD1为16个投影上等样本排序后绝对差均值；energy用非负有偏V-statistic `2 mean||X-Y|| - mean||X-X'|| - mean||Y-Y'||`，包括self-pairs。`W_FF`/`E_FF`来自FP_A/FP_B；`W_QF`/`E_QF`为两个cross-block Q/FP距离的均值；`W_QQ`来自Q_A/Q_B。FP reference floor为W_FF>1e-4、E_FF>1e-8；低于floor标reference_degenerate，不用clamping伪造稳定ratio。

`Dpair`为64个same-noise Q-FP physical first-action MSE的均值；`Dnoise`为全部32×32个FP_Ai/FP_Bj cross-pair MSE均值，用作FP-only effect scale，不把1024个距离视作独立样本。mapping binding要求Dnoise>1e-6且Dpair/Dnoise>=0.25；否则此condition不能支持“大mapping drift与small marginal shift分离”的假设。

translation sensitivity control：给FP_B的每个7D action加固定向量 `sqrt(7*0.25*Dnoise)*e1`，对FP_A计算SWD/energy，并除以同一FP-FP reference。要求ratio分别>=1.50和>=1.25。此平移仅由FP-only effect scale决定，per-coordinate MSE等于最小mapping drift门槛；不根据Q结果调整。此项仅检验对固定方向、预定量级位置变化的灵敏度，不证明一般统计power。

初步门槛（非显著性或等价检验）：candidate在W_QF/W_FF<=1.25且E_QF/E_FF<=1.25时落入bounded reference band；W_QF/W_FF>=1.50且E_QF/E_FF>=1.25时为shift。中间或两metric相反为inconclusive。16个固定投影的Q/FP IQR比以各arm64个draw计算，FP IQR>1e-6的valid投影至少13个；支持bounded band还要求至少13个valid投影的比值落在[0.5,2]。若至少8个valid投影ratio<0.5且W_QQ/W_FF<0.5，标collapse。Q-Q小而没有这种spread证据仅为sampling-null warning。

跨condition gate：至少6/8同时满足mapping binding、reference/spread/sensitivity可识别、bounded band，且其余condition无明显collapse，才为bounded_dissociation_preliminary_go。至少6/8在mapping binding且reference、spread、translation sensitivity可识别时为shift，则mechanism_no_go。其余statistical_inconclusive。Translation sensitivity不通过的condition只能标sensitivity_inconclusive，不能进入科学no-go计数；reference退化也一样。每condition全部保留，不隐藏未binding样本。

预算：CPU准备10min；一张V100最多15min、内部14min deadline；CPU raw replay5min。不因结果扩episode/seed/threshold，不追加完整distribution validation。OOM、超时、未完成1024sample如实保留为resource_blocked；具体工程错误先审查原job，不静默减少样本。

冻结前复核：独立FINAL_PROTOCOL_REVIEW确认统计结构可用。root明确了LIBERO/Wall数据namespace、normalized MSE单位和denoising/projection两个计数，以上均为文档澄清，未改变查看输出前的门槛。原始审查保留在DESIGN_AUDIT，不据此进行完整统计验证。

资源准备修订（任何本候选模型输出前）：CPU64775保留resource_blocked记录，固定选择需要两段未缓存视频。CPU64776从同一pinned dataset revision核验两段共64,412,505bytes及LFS SHA256，低于预定500MB补齐上限。允许一次CPU allocation下载这两段到独立conditional_marginal_ready/source_extension，扩展identity保留旧identity parent hash，不改旧资产与失败lock。再次准备严格使用64775的同一8个episode，不更换样本。仅撤销“任何缺缓存都停止”的资源假设；两arms、noise、距离、门槛和GPU预算不变。若补齐失败保留原文件，不退回login执行。
