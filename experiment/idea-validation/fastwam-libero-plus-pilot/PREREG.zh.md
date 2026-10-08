# Fast-WAM Optional-IDM / LIBERO-Plus paired pilot

冻结范围：4 base suites × 7 perturbation dimensions × 每 cell 50 个不同 variants × 4 arms，共 5,600 episodes。每个 variant 取官方 first initial state，单次 trial；同一冻结 manifest 和 env/sampler seeds 用于 bf16、w4a8、w4a4、w4a4kv4。按原 task 与 perturbation subtype 分层抽样，seed=20261005。只使用 test variants 做评估，不使用 Plus training dataset，也不根据模型结果重抽样。

Variant 与 task_id 使用固定 Plus source 的官方 task_order_index=0 字典顺序；canonical task 必须与 Plus 中的原任务精确匹配，按最长完整前缀识别扰动后缀。Objects Layout 的 levelN_sampleN 为真实布局扰动；无扰动的 canonical 原任务单列 control，不混入每 cell 的50个 stress variants。清单同时记录完整类别数量、可采样数量与排除的 original controls。

模型固定为已发布 libero_optional_idm_2cam224.clean.pt 与原 dataset stats；FastWAM source revision=7faa71108368fbb3b6885649f112af607427a2d4，action_infer_mode=idm，BF16 baseline，10 video / 10 action denoising steps，sigma_shift=1，CFG=1，compile=false，action chunk=32×7、执行前10步后replan、30 no-op settling steps。Spatial/Object/Goal active-step cap=400，Long=700。图像旋转/双相机拼接、proprio normalization、gripper处理使用原 FastWAM helper；Plus uses its own get_task_init_states and environment, assets, instructions.

W4：video expert、action expert、proprio_encoder 下所有 nn.Linear weights，signed symmetric RTN [-7,7]、每 output row 按 input channels G128；其他 params 保持原精度。A8/A4：对应 Linear input，dynamic per-token-row symmetric RTN [-127,127]/[-7,7]，zero row 映为0。三种低比特 arms 共用同一 W4 bank，不使用 dither、QAT、额外校准或 layer exceptions。

KV4：仅完整 video-conditioning prefill 返回的所有 layers K/V，包含 observation+generated future tokens。完整 prefill 保持本 arm 的 W/A 精度；prefill 完成后 K/V 一次量化并实际 packed uint8 双nibble存储，action attention按layer读取解包为BF16。K post-RoPE 每 head/channel 沿 token 轴 G64；V 每 head/token 沿 channel 轴 G64；group min/max affine 16 levels、zero span guard。无完整 FP residual cache。Action 当步 K/V、text cross-attention K/V、video denoising 临时 K/V 不属于此 KV4 范围。

用户明确要求最终5,600 episodes 使用 real quant，并优先使用原生 INT4。W4A4/W4A4KV4 需要 packed signed INT4 weights 和 packed INT4 activations，执行 SM80+ 原生 S4×S4→S32 Tensor Core MMA，再按 G128 尺度累加为 BF16 输出；最终 episode 必须有 native_int4_gemm_calls 的实际运行增量。W4A8 使用 packed W4 与真实 INT8 activation，在 kernel 内将 W4 扩展为 INT8 后执行 INT8×INT8→INT32 dot，明确标注其算术路径。所有低比特 arm 均不存储或调用完整 BF16 dequant weight bank，不用 floating GEMM fake quant 结果替代最终评分。KV4 实际 packed 存储、逐 layer 读取解包后仍用 BF16 attention。预检必须通过数值对照，确认实际 packed buffers、native INT4/INT8 GEMM 和 KV4 读取路径；SM80 以下不得通过 fallback 混入最终 A4 结果。Storage 通过实际输出 packed weight tensors+scale+metadata 与未量化 denoiser parameters 的 artifact 测量，报告 tensor payload 和 disk bytes，scope 固定为 denoiser core（video/action/proprio），不含 T5/VAE；BF16 使用相同 scope 的16bit artifact。仅 storage export，不在 episode 热路径写权重。

Lat.(ms)：一次 replan 中 video scheduler schedule 开始至最后一次 action scheduler update 的 CUDA-event elapsed；包含 video denoising、video-conditioning preparation/prefill、KV量化及实际读写、action denoising 与 sampler updates，排除 image/text encoding 与环境 rendering。阶段事件只用于归因，primary 连续区间不由 nested spans 相加。另存 native full action-query synchronized wall time 和 cold/warm 标记。Spd.=同 GPU / 同固定 observation+seed 的 BF16 median Lat. ÷ 本 arm median Lat.；不同闭环 observation 分布上的均值不能冒充 paired speedup。Peak：reset_peak_memory_stats 后真实 max_memory_allocated，包含整个 native query；单位 decimal GB (10^9)，同时报告 baseline allocated/reserved 及 GPU型号。Storage 单位 GiB (2^30)。

每个 timing job 使用同一固定 reference observation，warmup后至少3次测量，各 arm 使用相同noise seed；4arm闭环仍各自驱动后续 observation。汇总主要报告 paired固定输入timing；episode实测timing另列distribution。CPU wall/CUDA event/allocated/reserved不相加。FP已有权重先执行并export，之后替换packedW4 modules释放FP权重，不保留第二份GPU FP权重。weight conversion使ARM顺序受限，固定输入timing报告顺序和限制。

timeout 为有效policy failure；import/render/model/load/NaN或allocation异常为未完成，不计入policy失败，不重抽。已有完成slot（包括有效timeout）不重跑；失败作业保留artifact，只修复缺失明确slots。逐episode写入result JSONL/progress，以variant_id+arm唯一标识。完整汇总需要1,400 unique variants，每cell50，每arm1,400，5,600 unique terminal slots；部分结果不得标成完整评分。

所有下载、安装、manifest处理、storage导出、rendering、模型推理、quantization和聚合均在获批PBS allocation内。Login仅SSH、小控制文件、qsub/qstat、小日志；PBS_JOBID+nodefile+hostname guard 必须成立。GPU job每15秒采样allocated GPU UUID、utilization和memory至job.log。先preflight测每episode保守时间与load成本，再按2h walltime留余量拆分gdev，array并发上限初设10（服从scheduler）。数据准备优先CPU allocation。监控使用gpt-6-luna；无实质变化保持安静，终态/失败/需处理时反馈主chat。严禁banked reset。
