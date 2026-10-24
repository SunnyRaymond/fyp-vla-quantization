# Profiling 全量收据分析与 BF16 精度选择

分析日期：2026-10-23，Asia/Singapore。仅复核既有小型 JSON、log 与代码，没有新增 forward、训练、闭环 episode 或 PBS 作业。数值均来自最终成功 job `25938083.pbs101`，不与前一轮部分结果混合。

结论：当前 decoded export 的 policy-call p50 为 6.714 s，BF16 reference 为 1.481 s；query peak allocated 分别为 25.139 GB 和 25.131 GB。Packed-resident 诊断把 query peak 降至 17.088 GB，但逐层完整 decode 后耗时 10.717–10.875 s。现阶段应优先融合输入变换，再实现 tile decode + BF16 GEMM。全面改成 BF16 不具备数值等价保证。

## 计时与证据范围

同一 A100-SXM4-40GB、Torch 2.7.1+cu128、CUDA 12.8；每模式 fresh process、3 warmups；batch=1、32×7 action chunk、9 video frames、10 video + 10 action steps；compile 和 CUDA graph 关闭。

三个主组各为 40 contexts（3000–3039）× seeds 0/1，共80 query。stream_decode 仅有 case2000/seed0、case3000/seed0 两条诊断，不属于同一个完整主面板。

`full_query_wall_ms` 从 prepared GPU datum 到 `infer_action` 返回并完成同步，包含内部 text/prompt encoding、输入图像 latent encoding、noise/schedule、denoising 和返回 action。排除外部 observation 加载/转换、动作反归一化/controller command、环境与机器人执行。一次返回32步 action，不代表完成32步机器人运动。

`preprocessing_wall_ms` 是 cached observation 读取与 GPU datum 转换，单独计时。每个 context 的两个 seed 复用 datum，因而80行中 prep 只有40次独立观测。prep+query 的逐行相加仅是两段计时的派生总和，不是连续采集的机器人闭环 E2E；缓存、执行顺序和文件 I/O 影响 prep，不能当作量化 kernel 收益。

`denoising_latency_ms` 是覆盖连续 denoising 区间的两个 CUDA events；它包含 GPU stream 等待间隙，不等于活跃 CUDA kernel 时间之和。逐query wall减此区间的均值：BF16 65.778 ms、legacy 69.703 ms、decoded 68.369 ms。主要回归发生在 denoising 区间内。

主表 speedup 定义为 BF16 full-query p50 / candidate full-query p50。legacy=0.209084×、decoded=0.220594×，即分别耗时4.783×、4.533×。单次固定模式顺序不足以把约5%的 legacy→export 变化称作稳定统计收益。

## 内存与存储解释

GPU GB 使用10^9 bytes，storage GiB 使用2^30 bytes。Query peak 是每次重置 allocator peak 后测得的峰值；主表取80次中的最大值。Warm resident 是 warmup、GC、empty_cache 后的 allocated。Reserved 包含 allocator 缓存块，不是再加到 allocated 上的一份内存。

Decoded export 清理重复编码等状态，query peak 从28.475降至25.139 GB（-11.71%）；GPU 仍保存完整 decoded BF16 weights，所以没有相对 BF16 的显存节省。Packed 的 warm resident=16.174 GB、query peak=17.088 GB；共享 case3000/seed0 对 BF16 peak 的下降为32.00%，但这是两条诊断证据。Packed load-stage peak仍24.832 GB：factory先构造dense模型再替换。17.088 GB不是当前启动所需显存上限。

`resident_after_warmup` 和 query baseline 不同，后者每模式多602,624 bytes；收据没有给这项 allocation 独立命名，不将两者混淆。Warm snapshot 中的历史peak也不是warm当前resident。

614个量化 Linear 共有6,016,317,440个weight值，占原dense唯一tensor bytes约48.494%；剩余未量化参数为11.902288 GiB。目标encoding为2.982873 GiB，加Smooth/sign/U为2.991409 GiB。整模型唯一tensor payload为14.893702 GiB，实际archive为14.895614 GiB；原dense唯一tensor payload为23.108552 GiB。目标encoding压缩3.756869×，整模型tensor压缩1.551565×，下降35.55%。

编码bits/weight：indices约4.000001、B32 exponents约0.250005、codebooks约0.003344、row gains约0.005515；共4.258865，含transforms为4.271052。不能称为整模型严格4-bit。

共享storage只计一次。Bundle实际archive不含外部cached VAE/text/tokenizer资产，不能当作完整可搬运部署目录总大小。Load seconds分别592.628（legacy）、455.291（BF16）、87.970（dense）、81.088（packed），包含的恢复/anchor工作不同，不报告cold-start speedup。

KV cache唯一payload为108,380,160 bytes（103.36 MiB），BF16；相对于17–28 GB的总显存，它不是本轮主要显存来源。

## 瓶颈定位

614个目标Linear每query调用6436次：video3365、action3070、proprio1。Video主要层每query调用11次（10次video denoise及cache prefill），action主要层10次。每次目标Linear都运行Smooth/sign/H、learned U、whole-row A4，再执行BF16 GEMM。

Legacy `DecodedWeight.forward` 返回已有dense view，因此legacy的慢不能解释为每次weight decode。Export dense也不在forward重解码。只有stream_decode逐Linear完整解码：还把indices提升到int64、生成FP32 vectors/scales、最后写出BF16整层weight；`check_encoding`含GPU→host布尔判断，会同步。这条路径用于证明packed常驻可行。

输入H128以7层butterfly执行，每层有add/sub/stack；底层有cat与临时张量。Detailed query的aten::cat=45,653、add=52,850、sub=45,035、copy_=22,129、mm=6,593、addmm=6,436。大量小kernel、host dispatch、allocation和读写强烈指向fusion收益；尚无Nsight硬件counter，不能精确声称其中多少比例是launch-bound或bandwidth-bound。

Detailed CUDA-event spans：Smooth/H=5923.336 ms、A4=2362.363 ms、U=1293.875 ms；全部F.linear GEMM envelopes（含非target）=1358.499 ms。Detailed video denoise=6558.829 ms、action denoise=6520.898 ms、cache prefill=644.358 ms。这是另一个带大量hooks/events的intrusive query，不是主表6.714 s的分解。不要相加nested spans、除以无hook延迟求百分比，或做Amdahl预测。

底层profiler同样支持小算子判断：aten::cat self-device≈493.220 ms、self-CPU≈844.066 ms；aten::mm self-device≈154.201 ms，而U的event envelope≈1293.875 ms。两者统计边界不同；不能把U的1.29秒全看作FP32数学运算。aten::cat累计self-device-memory约125.223 GB，是全query多次分配/释放的统计，绝非峰值或实测HBM流量。

另一个值得优先核查的来源是deterministic memory fill：本轮`freeze_numerics`显式启用`torch.use_deterministic_algorithms(True)`，trace有101,513次`aten::fill_`，对应约221.410 ms self-device和353.203 ms self-CPU。PyTorch 2.7默认在deterministic模式填充未初始化内存，官方明确指出这会影响性能。实际`fill_uninitialized_memory` flag没有单独保存，缺少stack也不能把所有fill都归因于它，故这里只把它列为有依据的候选。后续若确认不存在未初始化读取，可以保持deterministic algorithms，仅关闭这项填充做配对比较；BF16/VQ两臂需同设置，不能把调试填充的移除冒充VQ算法收益。[PyTorch 2.7说明](https://docs.pytorch.org/docs/2.7/notes/randomness.html#filling-uninitialized-memory)。

Trace出现`ampere_bf16_s16816gemm_*`和BF16 CUTLASS kernels，确认主Linear已经使用BF16 Tensor Core路径；出现`ampere_sgemm_*`对应仍有FP32 GEMM。Attention与normalization也有成本，但当前证据优先指向target Linear输入路径。详细operator receipts仅保存原大文件的完整前缀记录，不能当作完整kernel总表或相加CPU/ATen/CUDA多层记录。

Shape重点为video M294、action M32、text+proprio context M129；不能照抄LLM M1 decode kernel。参数量最大的不一定是端到端延迟最高的路径：action只有约17%的target weight值，却有约48%的target calls，固定per-call开销显著。FFN大矩阵适合Tensor Core吞吐；小M和大量投影更需注意dispatch和decode开销。

## 为什么BF16参数仍有FP32运算

应区分：持久存储dtype、GEMM operand dtype、算术/accumulator dtype、输出dtype。主权重是BF16，并不要求中间都舍入到BF16。当前Smooth scales本来是FP32（7,227,452 bytes），两个128×128 U也是FP32（131,072 bytes），signs是int8，indices/exponents是uint8。Frozen U已导出，inference无需每次重新求Cayley线性系统。

当前输入路径：BF16输入→FP32 Smooth除法/sign/H128→FP32 U乘法→FP32整行absmax/7、safe divisor、round/clamp、乘scale→BF16 GEMM operand。

当前weight解码：BF16 codewords读入FP32，两本相加，乘FP32的`ldexp(BF16 row_gain, exponent-127)`，最后舍入为BF16 weight。BF16输入/weight送Tensor Core、保留FP32 accumulator是标准高效路线，不是把主GEMM改为普通FP32 GEMM。A100原生BF16 MMA的accumulator为FP32。

BF16只有7位fraction bits；FP32为23位。简单反例：三个数1、1/256、1.5都能BF16精确表示。先在FP32计算`(1+1/256)*1.5=1.505859375`，最后转BF16得到1.5078125；若先把加法结果舍入BF16，`1+1/256`在nearest-even下变成1，最终得到1.5。同样的BF16输入和输出，改变中间精度仍会改变结果。此例正对应codeword相加再乘row/block scale，不是该checkpoint实测误差。

A4还具有离散阈值：归一化值1.499若先舍入BF16变1.5，则round-to-nearest-even可能从code1跳为code2。U舍入、H的逐级舍入和scale舍入都可能影响这些阈值。改变U还会改变已有quantized weight所匹配的输入基底。因此不能因末端本来A4就认为前面的BF16误差一定被吸收。

reshape/view不损失精度，有限值符号翻转通常可精确；正常范围内某些2的幂缩放也可精确。BF16扩展到FP32本身精确。需要防止的是过早向BF16截断与改变运算顺序，而不是看到cast就一律删除。

本地FastWAM源码也明确在RMSNorm、T5LayerNorm、T5 softmax使用FP32中间量；noise与schedule还使用FP32。把这些一并改掉会扩大数值变化范围，不属于当前transform kernel的简单替换。`.to(bfloat16)`也不会自动覆盖源码的`.float()`或改变底层accumulator类型。

将全部persistent Smooth/U转BF16，理论仅节约3.679 MB；主要目标应是减少FP32中间张量反复写global memory，而不是省这几MB参数。全BF16或BF16-U可以作为独立候选，但是否保持action/closed-loop准确度必须实测，不能预先保证。

## 修订后的kernel顺序

正式kernel之前，先记录实际determinism/fill/reduction flags，评估上述不改算术dtype的memory-fill开销；旧job没有完整policy对照；后续job已做单shape microbench（见下文），仍不能外推整模型收益。

1. Fusion-preserving input path：BF16 load后在寄存器中转FP32，融合Smooth/sign/H128，保持现有butterfly与rounding顺序；保留现有FP32 U GEMM；融合whole-K A4 reduction、safe scale、round/clamp/dequant和BF16 store。精确division、round ties、tail、零行及编译器reassociation都要与现有operand对照。
2. Packed VQ tile decode + BF16 Tensor Core GEMM：读uint8索引/指数，FP32局部解码，MMA之前舍入BF16；只保留tile，不向global memory写整层decoded weight；FP32 accumulation、原bias/output dtype。按M32/M129/M294分别选择tile；K按32对齐，d4边界和tail有mask。格式检查从每forward移到install一次。
3. Loader直接构造空/meta target weight并安装encoding，解决packed startup仍约24.83 GB的问题。其收益与query latency分开报告。
4. Fusion后重测再决定BF16-U或更大范围BF16 arithmetic。单独比较input operand误差、A4 code flip rate、最终80-query action和full-future anchor；数值改变后需另验证closed-loop。当前没有这些候选的新结果。

首版仍用Triton原型，必要时再转CUDA/CUTLASS SM80。一次融合全部H/U/A4/GEMM会遇到whole-K scale reduction、寄存器压力、tile边界与不同M的问题，不必作为首版前提。相同输入的Q/K/V若Smooth/sign不同也不能直接共享量化operand。Prompt缓存、静态context投影缓存可另评估，并对BF16和VQ使用同一缓存协议；本轮没有实测这类收益。

现有parity=0指export相对当前quantized legacy checkpoint，不是quantized模型相对原BF16模型。未来更换GEMM算法可能改变accumulation顺序，即使保留BF16 operands和FP32 accumulation也应报告数值差异。

参考：[PyTorch 2.7 numerical accuracy](https://docs.pytorch.org/docs/2.7/notes/numerical_accuracy.html)、[NVIDIA Ampere Tensor Core types](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#improved-tensor-core-operations)、[Triton dot](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)。PyTorch的BF16 reduced-precision reduction配置可能允许部分中间截断，本轮未单独记录该flag，不能声称对库内部每一步累加做过精度审计。

以下附表仅从既有JSON receipts派生。所有intrusive值只用来定位，不是主延迟分解。

## 后续 fused 原型的局部结果：单独报告

后续 job `25942669.pbs101` 使用另一张 A100（UUID `6a22557e-ff02-e2d1-c523-9294d2dedb53`），不能与上表的旧 job 跨卡计算 speedup。它通过了全部614个实际 weight decoder 的 BF16 原始 int16 位模式比较；synthetic Smooth/H 和 A4 也通过。但真实模型 `input_fused` 的 anchor 在 `video_expert.blocks.0.self_attn.k` 的 A4 operand 位模式检查失败，PBS F/Exit1；没有完成 fused/packed/direct-loader 的完整80查询，因而没有这些版本的 E2E latency/peak memory 结论。

该 job 的 eager comparator 已完整完成80查询：p50 6.958742 s、p95 7.529546 s、query peak allocated 25.139422 GB、warm allocated 24.874590 GB、load-stage peak allocated 24.901912 GB；source action/future anchor 均为0。这是独立重复的 eager 实现测量，不是 fused kernel 结果。

局部 microbench（实际 checkpoint weight、random BF16 input，CUDA event，3 warmups/10 repeats）：

| scope / M,N,K | eager or dense ms | fused or packed ms | interpretation |
| --- | ---: | ---: | --- |
| Smooth/sign/H128, M294 K3072, deterministic fill=True | 0.466944 | 0.086118 | 局部5.42×，仍保持FP32 arithmetic |
| Smooth/sign/H128, M294 K3072, fill=False | 0.450150 | 0.073216 | 仅局部诊断；主面板仍fill=True |
| VQ GEMM 294,3072,3072 | 0.101376 | 0.617267 | 首版packed慢6.09× |
| VQ GEMM 294,14336,3072 | 0.223232 | 2.380186 | 首版packed慢10.66× |
| VQ GEMM 294,3072,14336 | 0.195686 | 3.091456 | 首版packed慢15.80× |
| VQ GEMM 32,3072,1024 | 0.091955 | 0.106394 | 首版packed慢1.16× |

不能将局部5.42×写成整模型speedup。大M需要复用同一decoded tile：首版BM32在M294上有10个M tiles，同一weight tile会被多次逻辑lookup/decode；实际HBM流量还受cache影响，目前没有硬件counter。下一版按[N,K]连续地址lookup、提高M方向复用、按12种shape测量tile配置。压缩只减少weight表示，BF16 MMA的数学工作量不变，还增加lookup/decode成本；大M未必从weight compression获得latency收益。

检查官方PyTorch v2.7.1代码发现一个明确的算术不等价点：`amax / 7`的CPU-scalar除法采用FP32 reciprocal后相乘，首版Triton用了`div_rn(amax,7)`。这是需要修复的差异，尚需真实输入GPU验证它是否解释本次失败；不放宽A4 gate。[官方执行分支](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu#L32-L44)。即使都保留FP32，更换division/rounding实现也可能改变离散A4 code；更不能从“参数是BF16”推断全BF16 arithmetic无损。

旧job的15秒NVML采样中，measuring阶段GPU utilization中位数：legacy 41%（36样本）、BF16 42%（8样本）、decoded export 42%（33样本）。这些稀疏样本不是SM占用率或Tensor Core利用率，不能单独判定compute/launch/bandwidth瓶颈。`aten::cat`累计125.223 GB allocation也不是峰值显存或HBM实测流量。

<!-- derived_tables -->

## Latency统计

| mode | metric | n | mean ms | p50 ms | p95 ms | min ms | max ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bf16 | full_query_wall_ms | 80 | 1479.927 | 1481.098 | 1538.208 | 1410.534 | 1547.976 |
| bf16 | denoising_latency_ms | 80 | 1414.149 | 1415.860 | 1474.293 | 1348.498 | 1482.096 |
| legacy | full_query_wall_ms | 80 | 7089.547 | 7083.761 | 7179.290 | 6906.369 | 7377.352 |
| legacy | denoising_latency_ms | 80 | 7019.844 | 7011.546 | 7110.800 | 6839.319 | 7307.700 |
| inference_dense | full_query_wall_ms | 80 | 6613.259 | 6714.148 | 6937.389 | 5951.424 | 7009.864 |
| inference_dense | denoising_latency_ms | 80 | 6544.891 | 6647.794 | 6867.507 | 5884.235 | 6939.818 |
| stream_decode | full_query_wall_ms | 2 | 10796.021 | 10796.021 | 10866.894 | 10717.274 | 10874.768 |
| stream_decode | denoising_latency_ms | 2 | 10729.795 | 10729.795 | 10798.873 | 10653.041 | 10806.549 |

## 独立context preprocessing

| mode | n | mean ms | p50 ms | p95 ms | min ms | max ms |
| --- | --- | --- | --- | --- | --- | --- |
| bf16 | 40 | 45.816 | 39.432 | 79.873 | 11.785 | 124.454 |
| legacy | 40 | 72.141 | 46.483 | 155.825 | 12.111 | 447.223 |
| inference_dense | 40 | 11.604 | 11.471 | 13.037 | 9.677 | 13.340 |
| stream_decode | 2 | 11.686 | 11.686 | 11.862 | 11.491 | 11.882 |

## prep + query派生两段总和

| mode | n | mean ms | p50 ms | p95 ms | min ms | max ms |
| --- | --- | --- | --- | --- | --- | --- |
| bf16 | 80 | 1525.743 | 1528.191 | 1581.431 | 1445.645 | 1602.228 |
| legacy | 80 | 7161.688 | 7152.214 | 7298.265 | 7006.100 | 7487.466 |
| inference_dense | 80 | 6624.863 | 6726.305 | 6947.636 | 5962.425 | 7021.458 |
| stream_decode | 2 | 10807.708 | 10807.708 | 10878.756 | 10728.765 | 10886.650 |

## query wall减denoising interval的逐query差值

| mode | n | mean ms | p50 ms | p95 ms | min ms | max ms |
| --- | --- | --- | --- | --- | --- | --- |
| bf16 | 80 | 65.778 | 65.826 | 69.021 | 59.573 | 92.101 |
| legacy | 80 | 69.703 | 69.660 | 70.461 | 64.163 | 105.855 |
| inference_dense | 80 | 68.369 | 69.454 | 70.390 | 62.125 | 72.328 |
| stream_decode | 2 | 66.226 | 66.226 | 68.020 | 64.233 | 68.220 |

## Memory：decimal GB

| mode | warm allocated | query baseline max | query peak allocated max | query peak reserved max | load peak allocated | load peak reserved |
| --- | --- | --- | --- | --- | --- | --- |
| bf16 | 24.865422 | 24.866024 | 25.130502 | 25.419579 | 24.831867 | 25.027412 |
| legacy | 28.210058 | 28.210660 | 28.474890 | 28.749857 | 29.082743 | 29.729227 |
| inference_dense | 24.874590 | 24.875192 | 25.139490 | 25.608323 | 24.901912 | 25.297945 |
| stream_decode | 16.174151 | 16.174754 | 17.088417 | 17.444110 | 24.831867 | 25.027412 |

## Encoding组件

| component | bytes | GiB | bits/target weight |
| --- | --- | --- | --- |
| books | 2514944 | 0.002342224 | 0.003344164 |
| row_scales | 4147598 | 0.003862752 | 0.005515132 |
| indices | 3008159232 | 2.801566601 | 4.000000681 |
| scale_exponents | 188013792 | 0.175101489 | 0.250005149 |

## Stream存储

| stream | modules | weight values | BF16 GiB | encoded GiB |
| --- | --- | --- | --- | --- |
| video | 306 | 4996792320 | 9.307250977 | 2.475809455 |
| action | 307 | 1019492352 | 1.898952484 | 0.507033093 |
| proprio | 1 | 32768 | 0.000061035 | 0.000030518 |

## Intrusive stage spans

| mode | stage | calls | event elapsed ms |
| --- | --- | --- | --- |
| bf16 | _denoise_video | 10 | 1745.908 |
| bf16 | _denoise_action_with_video_cache | 10 | 1709.313 |
| bf16 | video_cache_prefill | 1 | 166.648 |
| legacy | _denoise_video | 10 | 7083.032 |
| legacy | _denoise_action_with_video_cache | 10 | 7044.958 |
| legacy | video_cache_prefill | 1 | 720.861 |
| inference_dense | _denoise_video | 10 | 6558.829 |
| inference_dense | _denoise_action_with_video_cache | 10 | 6520.898 |
| inference_dense | video_cache_prefill | 1 | 644.358 |

## 全部614层按family聚合的intrusive envelope

| mode | stream | family | modules | calls | event elapsed ms |
| --- | --- | --- | --- | --- | --- |
| bf16 | action | cross_attn | 120 | 1200 | 345.620 |
| bf16 | action | ffn | 60 | 600 | 173.420 |
| bf16 | action | other | 7 | 70 | 20.364 |
| bf16 | action | self_attn | 120 | 1200 | 345.198 |
| bf16 | proprio | other | 1 | 1 | 0.785 |
| bf16 | video | cross_attn | 120 | 1320 | 375.313 |
| bf16 | video | ffn | 60 | 660 | 234.599 |
| bf16 | video | other | 6 | 65 | 19.821 |
| bf16 | video | self_attn | 120 | 1320 | 377.233 |
| legacy | action | cross_attn | 120 | 1200 | 2431.624 |
| legacy | action | ffn | 60 | 600 | 1213.456 |
| legacy | action | other | 7 | 70 | 127.223 |
| legacy | action | self_attn | 120 | 1200 | 2418.211 |
| legacy | proprio | other | 1 | 1 | 14.509 |
| legacy | video | cross_attn | 120 | 1320 | 2689.868 |
| legacy | video | ffn | 60 | 660 | 1391.750 |
| legacy | video | other | 6 | 65 | 129.363 |
| legacy | video | self_attn | 120 | 1320 | 2679.860 |
| inference_dense | action | cross_attn | 120 | 1200 | 2237.653 |
| inference_dense | action | ffn | 60 | 600 | 1108.867 |
| inference_dense | action | other | 7 | 70 | 120.851 |
| inference_dense | action | self_attn | 120 | 1200 | 2229.102 |
| inference_dense | proprio | other | 1 | 1 | 21.558 |
| inference_dense | video | cross_attn | 120 | 1320 | 2475.706 |
| inference_dense | video | ffn | 60 | 660 | 1285.519 |
| inference_dense | video | other | 6 | 65 | 115.641 |
| inference_dense | video | self_attn | 120 | 1320 | 2466.154 |

## Decoded详细query全部已汇总GEMM shapes

| stream | M | N | K | calls | event elapsed ms |
| --- | --- | --- | --- | --- | --- |
| video | 294 | 3072 | 3072 | 1991 | 385.615 |
| action | 32 | 3072 | 1024 | 1200 | 227.503 |
| video | 129 | 3072 | 3072 | 671 | 128.508 |
| action | 32 | 1024 | 3072 | 600 | 119.691 |
| action | 129 | 3072 | 1024 | 600 | 113.992 |
| video | 294 | 14336 | 3072 | 330 | 109.616 |
| video | 294 | 3072 | 14336 | 330 | 102.099 |
| action | 32 | 1024 | 4096 | 300 | 59.724 |
| action | 32 | 4096 | 1024 | 300 | 56.369 |
| other | 128 | 4096 | 4096 | 96 | 17.441 |
| other | 128 | 10240 | 4096 | 48 | 9.549 |
| other | 128 | 4096 | 10240 | 24 | 5.082 |
| video | 294 | 18432 | 3072 | 11 | 3.626 |
| video | 129 | 3072 | 4096 | 11 | 2.049 |
| video | 294 | 3072 | 256 | 11 | 2.047 |
| action | 32 | 7 | 1024 | 10 | 2.036 |
| action | 32 | 1024 | 7 | 10 | 2.015 |
| video | 294 | 192 | 3072 | 10 | 1.934 |
| action | 129 | 1024 | 4096 | 10 | 1.890 |
| action | 129 | 1024 | 1024 | 10 | 1.886 |
| action | 1 | 1024 | 1024 | 10 | 1.882 |
| action | 1 | 1024 | 256 | 10 | 1.862 |
| action | 1 | 6144 | 1024 | 10 | 1.852 |
| proprio | 1 | 4096 | 8 | 1 | 0.233 |

## Decoded operator前缀中的ATen记录（非完整trace）

| op | calls | self-device ms | self-CPU ms | cumulative self-device-memory GB |
| --- | --- | --- | --- | --- |
| aten::cat | 45653 | 493.220 | 844.066 | 125.222754 |
| aten::addmm | 6436 | 303.312 | 491.541 | 2.165836 |
| aten::add | 52850 | 278.198 | 512.944 | 66.584755 |
| aten::sub | 45035 | 229.904 | 293.665 | 59.658279 |
| aten::fill_ | 101513 | 221.410 | 353.203 | 0.000000 |
| aten::mul | 28548 | 202.768 | 248.906 | 69.155052 |
| aten::mm | 6593 | 154.201 | 469.611 | 18.313376 |
| aten::copy_ | 22129 | 135.727 | 118.448 | 0.000000 |
| aten::div | 19401 | 110.019 | 175.365 | 35.568526 |
| aten::amax | 6436 | 81.182 | 71.952 | 0.006709 |
| aten::_efficient_attention_forward | 1261 | 41.410 | 11.495 | -0.001004 |
| aten::mean | 2573 | 36.849 | 27.631 | 0.006666 |
| aten::abs | 12922 | 30.695 | 52.337 | 0.000000 |
| aten::clamp | 6436 | 28.626 | 50.147 | 17.455844 |
| aten::round | 6436 | 28.283 | 38.657 | 17.409874 |
| aten::where | 8980 | 25.522 | 61.707 | -0.000598 |
| aten::gt | 6461 | 19.158 | 60.837 | 0.003688 |
| aten::native_layer_norm | 1900 | 14.905 | 19.438 | 0.000000 |
| aten::pow | 2635 | 10.382 | 27.556 | 6.056406 |
| aten::cudnn_convolution | 41 | 7.223 | 1.768 | 0.104721 |
| aten::rsqrt | 2569 | 6.614 | 15.862 | 0.002652 |
| aten::gelu | 651 | 6.020 | 8.104 | 2.871761 |

## 每mode NVML稀疏采样，含loading与intrusive

| mode | samples | sampled max used MiB |
| --- | --- | --- |
| bf16 | 48 | 24799 |
| legacy | 105 | 28853 |
| inference_dense | 59 | 24981 |
| stream_decode | 11 | 20897 |

NVML全job共224个样本，另有1条在mode开始前。GPU-Util采样不能替代SM占用率、Tensor Core利用率或HBM带宽计数。
