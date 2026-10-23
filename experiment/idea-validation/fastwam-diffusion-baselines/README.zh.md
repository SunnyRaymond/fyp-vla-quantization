# Fast-WAM：SVDQuant / ViDiT-Q 的 W4A4 闭环比较

两种方法各沿用 frozen **4 suites × 7 perturbation dimensions × 10 variants = 280 episodes**。总计新增560例，复用八个已有配置的2240例作为配对 reference。环境、initial state 0、environment/sampler seeds、10/10 denoising steps、32-step horizon、执行10个actions和400/700-step cap均保持原协议。

## 资源安排

只提交一个四卡24h PBS task。每卡加载一次模型，四卡各处理20个 original-LIBERO calibration contexts，合并80例的 per-channel activation maxima。随后四卡按 `module_index % 4` 分担两种方法的权重准备，每个 bank 只有一个写入者；完成后 worker0/1分别评测SVDQuant的偶数/奇数140例，worker2/3对应ViDiT-Q。无需安装新 ML dependencies。所有模型、校准、权重分解、数据读取和聚合都在获批PBS allocation中执行。

每15秒将GPU utilization和显存写入job.log。由gpt-6-luna/xhigh subagent按30分钟间隔监控，正常排队/运行时保持安静；完成或失败时通知owner继续处理。

## 方法与预算

两个arm均覆盖相同614个canonical target Linears，主路径W4A4、weight group128、FP32 weight scales、BF16 KV。推理使用decoded BF16模拟；不把这份实现的latency或显存当作native low-bit kernel表现。

- **SVDQuant adapted W4A4**：固定alpha0.5迁移activation outliers，分解平滑后的weight；BF16 low-rank branch保护主成分，量化residual走W4和逐token signed A4。采用固定种子的randomized truncated SVD（oversampling8、power iterations4），不进行梯度训练。与DeepCompressor完整配置的smoothing grid search和最多100次low-rank refinement不同，本轮采用论文核心的直接SVD分解，并显式保留这个适配边界。
- **ViDiT-Q adapted W4A4**：使用作者代码约定alpha0.5665，固定channel mask，再施加随机Hadamard；逐token affine minmax A4。遵循作者Linear实现的两次weight quantization：先量化scaled weight，再rotation并重新量化；只保存最终编码。对power-of-two和12×power-of-two宽度使用完整structured Hadamard；没有实数full Hadamard的物理宽度（例如7）使用dyadic blocks，并在bank manifest中列出。没有启用其官方mixed-precision示例的W8A8或FP保护层。

SVD的BF16 factors计入budget。按实际614层shape，在任何量化质量或闭环结果之前，从全局rank `[32,16,8,4,2,1]` 中选择满足以下条件的最大值：两arm的target effective-bpw相对差≤5%，且各自≤4.60。每层rank取全局rank和物理矩阵较短维的较小值。若没有非零rank满足约束，作业退出，不静默移除low-rank branch。

计入存储的项目包括实际padded W4 nibbles、FP32 group scales、ViDiT uint8 zero points、FP32 smoothing vectors、int8 rotation signs，以及BF16 low-rank factors。统计分别报告target effective-bpw与whole-model parameter-bpw；模型加载时的decoded matrix、KV和workspace占用另行测量。

## 启动与完成条件

GPU数值检查包括packed W4独立算术、A4零值/非恒等、rotation的dense reference/逆变换/正交性，以及SVD low-rank和ViDiT forward路径。每个worker在原C80 case2000/seed0上先验证BF16 teacher action，再检查同arm两个worker从相同bank读取后的完整action一致性；正式query必须执行614个目标模块以及对应的low-rank/rotation分支。

完成需要PBS exit0、artifact exit0、完成标记、4×140 worker receipts、每arm280/总560个新episodes，以及2240条reference一致。soft deadline时保留完整episode前缀。该实验不使用Plus测试结果选择alpha、rank或calibration contexts，不将单次cohort的小幅差异称为稳定优势。

## Primary sources

- [SVDQuant paper v4](https://arxiv.org/html/2411.05007v4)：smoothing、high-precision low-rank和low-bit residual。
- [DeepCompressor default configuration](https://github.com/nunchaku-ai/deepcompressor/blob/main/examples/diffusion/configs/svdquant/__default__.yaml) 与 [INT4 configuration](https://github.com/nunchaku-ai/deepcompressor/blob/main/examples/diffusion/configs/svdquant/int4.yaml)：官方rank32、group64及校准选项；本轮group/rank适配见上文。
- [ViDiT-Q paper](https://arxiv.org/html/2406.02540) 与 [author Linear implementation](https://github.com/thu-nics/ViDiT-Q/blob/viditq/quant_utils/qdiff/viditq/viditq_quant_layer.py)：固定channel mask、rotation与动态token quantization。
- [ViDiT-Q PTQ](https://github.com/thu-nics/ViDiT-Q/blob/viditq/examples/pixart/ptq.py)、[OpenSora alpha configuration](https://github.com/thu-nics/ViDiT-Q/blob/viditq/examples/opensora1.2/configs/w4a8_mixed_precision.yaml)、[W4A4 mixed-precision example](https://github.com/thu-nics/ViDiT-Q/blob/viditq/examples/pixart/configs/w4a4_mixed_precision.yaml)。
- [Author Hadamard utility](https://github.com/thu-nics/ViDiT-Q/blob/viditq/quant_utils/qdiff/quarot/quarot_utils.py)：H12 basis；本轮用structured butterfly实现相同类型的固定orthogonal rotation，不保存dense Q。

## 已提交任务

25878591.pbs101 / dfbase0001，2026-10-19 19:25 SGT提交。4 GPU、24h；scheduler queue为g3，并按站点规则扩为64 CPU / 440GB。监控agent为/root/diffusion_long_watch（gpt-6-luna/xhigh，30min）。

实际准备结果：target参数6,016,317,440；SVDQuant rank16、4.425034381 bpw；ViDiT-Q 4.324954219 bpw，差2.3140%。ViDiT rotation为249个full-power2、334个full-H12×power2、31个dyadic fallback，具体宽度计数见RUN_STATE.json。四个worker的BF16 teacher及cross-worker action max abs均0，全部GPU selfchecks通过。

Rotation适配的具体边界：31个dyadic fallback中，30个input-width14336使用H8192/H4096/H2048分块，另1个width7使用H4/H2/H1。作者通用utility可为14336提供H28基底；本轮预登记实现仅包含H12与power-of-two基底，因此不声称完整官方full-H28 recipe复现。此配置在采集闭环结果前已经固定，运行期间不按结果改rotation。

最终完成：25878591.pbs101，PBS/artifact exit0，560/560新episodes（各280），2240references，四worker各140且rc0。实际08:48:26。SVDQuant adapted 108/280（38.57%）；ViDiT-Q adapted203/280（72.50%）。结果与适配边界详见RESULTS.zh.md；监控已停止。
