# C80 R128 A+Z+KL fused VQ：实现与完整测量结果

主实验 `25943391.pbs101`：PBS F/Exit0、artifact exit0、PIPELINE_COMPLETE及完整面板检查均通过。申请walltime 06:00:00，实际运行00:54:45，完成后释放allocation。normal→g1；scheduler实际分配1 GPU/16 CPU/110 GB（提交请求120 GB，经scheduler规范化）。

结论：input fusion获得1.862934×的policy-call p50加速，并保持完整80条action数值一致。Packed VQ kernel与direct/meta loader已实现并完成测量，显著降低query和启动显存，但本版packed GEMM没有速度收益，也没有保持原checkpoint的完整action输出；不能作为已验证的等价替换。

## 固定协议与计时边界

冻结source checkpoint `25921795.pbs101 / reader_kl`，新增训练更新0、closed-loop episodes 0。614个Linears（306 video、307 action、1 proprio），每query调用6436次；四个量化模式的正式计时调用数各514880。

五组均为同一A100-SXM4-40GB（UUID d62df84e-f017-3a07-5492-ec26b529f468），40 contexts3000–3039×seeds0/1，每组80 queries、fresh process、3 warmups、batch1、32×7 actions、9 frames、10 video+10 action steps。deterministic=True、fill_uninitialized_memory=True、TF32=False、compile=False、CUDA graph=False；BF16 reduced-precision-reduction flag记录为True。

Policy-call E2E：prepared GPU datum→infer_action返回并同步，包括内部prompt/text、image latent encoding、schedule和denoising。排除外部observation读取/预处理、environment及robot motion。每次返回32步action，不等于执行了32步机器人运动。详细profiler在主计时结束后另跑，不纳入下表。

## 同GPU完整80-query结果

GPU GB=10^9 bytes；weight storage GiB=2^30 bytes。Speedup=BF16 p50/candidate p50，低于1表示慢于BF16。

| Mode | n | Mean s | p50 s | p95 s | Speedup vs BF16 | Warm allocated GB | Query peak allocated GB | Query peak reserved GB | Load peak allocated GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 80 | 1.451284 | 1.450439 | 1.505931 | 1.000000× | 24.865422 | 25.130502 | 25.419579 | 24.831867 |
| eager | 80 | 6.547441 | 6.527626 | 6.695142 | 0.222200× | 24.874590 | 25.139422 | 25.685918 | 24.901912 |
| input_fused | 80 | 3.520413 | 3.503950 | 3.724216 | 0.413944× | 24.874590 | 25.139422 | 25.685918 | 24.901912 |
| packed_fused | 80 | 7.097266 | 7.092558 | 7.161170 | 0.204502× | 16.066148 | 16.329718 | 16.426992 | 24.831867 |
| packed_direct | 80 | 7.099053 | 7.094879 | 7.151960 | 0.204435× | 16.172259 | 16.437519 | 16.538141 | 16.138769 |

Input fusion相对eager p50加速1.862934×，latency下降46.32%；相对BF16仍慢2.416倍。GPU仍持有614个dense target weights，所以显存没有压缩常驻收益。
Packed+direct query peak相对BF16下降34.59%；direct loader的load peak相对普通packed loader下降35.01%。两个packed模式均为614 encoded modules、0 dense target weights；profiler实际出现6436次_linear_kernel。
Packed_direct的warm/query内存比普通packed约高0.11 GB，这项固定差异未做storage级归因；不据此推断算术或KV格式变化。两者完整80条action误差为0。Reserved含allocator缓存，不能加到allocated上。

Load seconds（按本轮实际执行顺序）：input_fused=462.543, eager=80.958, packed_fused=80.370, packed_direct=24.268, bf16=90.448。首组冷文件cache与后续cache状态不同，loader工作量也不同，不计算公平cold-start speedup。

## 完整80条数值结果

下表相对冻结eager量化checkpoint；max abs及full32 action RMSE含全部7维，motor RMSE只含前6维。First10对应每query前10步；gripper判据为输出>0.5，共80×10=800个决策。

| Mode | Action max abs | Full32 action RMSE | First10 motor RMSE | Full32 motor RMSE | First10 gripper disagreement |
|---|---:|---:|---:|---:|---:|
| bf16 | 2.015625 | 0.0912011713 | 0.0462718196 | 0.0564000569 | 0.6250% |
| eager | 0 | 0 | 0 | 0 | 0.0000% |
| input_fused | 0 | 0 | 0 | 0 | 0.0000% |
| packed_fused | 2.0078125 | 0.0786525011 | 0.0449622646 | 0.0487507731 | 0.7500% |
| packed_direct | 2.0078125 | 0.0786525011 | 0.0449622646 | 0.0487507731 | 0.7500% |

Input_fused相对eager完整80条action max/RMSE均为0。Direct loader相对相同packed算术完整80条也均为0。614个standalone weight decoder比较的是BF16原始int16位模式，全部一致；这不等于packed GEMM或最终action bitwise一致。

Packed相对eager的前10步motor RMSE=0.0449623，gripper分歧6/800；相对BF16的对应指标为0.0454130和7/800。Eager相对BF16则为0.0462718和5/800。不能由这些fixed-input数值推断closed-loop成功率优劣。

独立anchor（context2000/seed0，不属于80条held-out面板）：input_fused action/future误差均0；packed两个模式的action max abs=0.1220703125、future max abs=2.4453125、future NRMSE=0.1822309196。

自动summary的parity_passed表示预先规定的input-fusion/loader gate及测量流程完成，不表示packed相对eager数值等价。packed_outputs_bitwise=False，closed_loop_validated=False。

## Storage账本

| 项目 | GiB |
|---|---:|
| 目标BF16 weights | 11.206264496 |
| Packed target encoding | 2.982873065 |
| Encoding+Smooth/sign/U | 2.991408997 |
| 未量化model parameters | 11.902287906 |
| Whole bundle唯一tensor payload | 14.893702424 |
| 实际inference_bundle.pt archive | 14.895613806 |
| 原dense唯一tensor payload | 23.108552402 |

目标weight values=6,016,317,440；encoding=4.258865125 bits/weight；含transforms=4.271052497 bits/weight。目标encoding压缩3.756869×，whole tensor payload压缩1.551565×、减少35.55%。共享storage只计一次。
整模型没有接近4×，因为目标BF16 weight bytes只占原dense唯一tensor bytes的48.49%，其余11.90 GiB仍未量化。Archive不包含外部cached VAE/text/tokenizer assets，不等于整个可搬运部署目录大小。

## 实际热点与首版kernel局限

Packed_direct的额外intrusive profiler query（不是主计时分解）：

| Operator/kernel | Calls | Self-device ms |
|---|---:|---:|
| _linear_kernel | 6436 | 5749.842 |
| _smooth_h_kernel | 6456 | 59.412 |
| _a4_kernel | 6436 | 45.586 |
| aten::mm | 6593 | 126.197 |
| aten::cat | 634 | 5.533 |
| aten::fill_ | 50027 | 104.527 |

输入fusion后，cat由旧实现intrusive query的45653次降至634次，div由19401次降至93次，sub由45035次降至27次。旧/new profiler instrumentation不同，不能用其耗时直接计算主计时的加速；call结构与完整主计时1.86×收益共同支持输入融合有效。

当前VQ GEMM kernel self-device约5.750s，远高于fused H约0.059s、A4约0.046s及全部aten::mm约0.126s。当前主要热点已转到packed GEMM内部。ATen行和底层CUDA kernel行可能表示同一工作，不相加；没有Nsight硬件counter，不把该5.75s进一步硬分为HBM、lookup、ldexp、转置或MMA。

本版12代表shape×12 tiling候选全部完成数值gate和计时后冻结dispatch。候选均未胜过对应dense microbench；大M video FFN尤其差。Packed格式减少weight表示，并不减少BF16 MMA FLOPs；gather/decode、布局转换、K-loop pipeline和M方向复用均有额外成本。本轮没有实现INT8/INT4整数MMA。

## Precision归因：独立六shape诊断

诊断job `25944399.pbs101`：gdev、A100-SXM4-40GB，PBS/marker成功，6/6 shapes均finite，新增整模型forward=0、训练更新=0。只使用相同checkpoint的actual weights与fixed random BF16 inputs，不把不同GPU的诊断当latency比较。

六组中切换allow_bf16_reduced_precision_reduction True/False，cuBLAS结果全部raw BF16 bitwise一致。因而本次测得差异不能归因于这个toggle；这不证明所有cuBLAS路径或任意shape都不受该flag影响。

| M,N,K | CuBLAS vs packed NRMSE | CuBLAS vs dense Triton NRMSE | Packed vs dense Triton NRMSE |
|---|---:|---:|---:|
| 294,3072,14336 | 0.00265038293 | 0.00265054242 | 5.57274871e-05 |
| 294,14336,3072 | 3.96208379e-05 | 0 | 3.96208379e-05 |
| 294,3072,3072 | 3.81436257e-05 | 0 | 3.81436294e-05 |
| 129,3072,3072 | 4.08016349e-05 | 0 | 4.08016458e-05 |
| 32,3072,1024 | 3.45755143e-05 | 2.56895546e-05 | 2.46296186e-05 |
| 32,1024,3072 | 0.00012104035 | 0.000121045516 | 1.43562056e-06 |

关键：在K14336的video FFN上，移除VQ decode、直接读同一BF16 dense weight的Triton GEMM仍与cuBLAS相差约0.0026505 NRMSE；packed与dense Triton的差异小得多（约5.57e-5）。现有证据更支持GEMM backend的数值路径，而非weight存储内容变化，是该局部差异的主要候选。它没有完全排除所有inline-decoder细节，也没有将完整80条轨迹变化因果归于某一层。

额外BF16 bias probe（M32,N1024,K3072）：packed相对F.linear NRMSE约0.0025153，而F.linear相对“先BF16 matmul输出、再BF16 bias相加”的NRMSE约4.575e-5。Bias/epilogue的舍入次序值得单独匹配；这项结果不是对cuBLAS内部具体指令顺序的证明。

FP32 operands→F.linear→BF16 output也只是较高精度对照，不是数学精确oracle。PyTorch2.7.1的gemm_and_bias有独立cuBLASLt路径，使用FP32 compute与BF16矩阵类型；仅有相同operand和accumulator dtype，仍不保证与自写Triton有相同reduction/epilogue。[官方源码](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/cuda/CUDABlas.cpp#L1188-L1318)。

## 交付状态与后续实现判断

已交付：FP32-preserving Smooth/sign/H128与whole-row A4 fusion；614层native packed VQ tile-decode/BF16 Tensor Core GEMM原型；直接empty/meta loader；冻结dispatch、全部五组80条、storage及profile/numerical receipts。源码位于实验根目录（本文件上两级）的fused_input.py、fused_vq.py、kernel_runtime.py、kernel_worker.py、finalize.py。

当前可保留的结果是input fusion：此面板输出误差0并有1.86×相对eager收益。Direct/meta loader已证明启动峰值下降及相对相同packed算术的输出一致。Packed GEMM本版仍是实验候选，既慢于eager，也引入完整action变化，不作为当前best checkpoint的已验证替代。

下一版应优先匹配GEMM的reduction/bias epilogue，再优化codebook staging、coalescing、M复用和K-loop pipeline，必要时转CUDA/CUTLASS SM80。若后续优先要求保留cuBLAS数值，可另评估single-pass Triton完整layer decode+cuBLAS F.linear作为后备；它仍会生成临时dense weight，不能冒称为当前fused VQ kernel。本轮未运行该后备路线。

本轮没有把中间算术全面降为BF16，也没有新增closed-loop验证。FP32 arithmetic可以留在寄存器中，1.86×收益来自减少launch/临时张量，不需要先牺牲中间精度；“所有参数BF16”不足以证明全BF16 arithmetic或更换GEMM backend无损。

## 证据入口

主summary.json、parity.json、completion_evidence.json；五个mode的summary/samples；四个量化mode的validation；per_layer/decoder_*的614层位模式结果；tuning.json与selected_configs.json；packed_direct/operators.json。完整.pt与Chrome trace留在compute filesystem，未下载。Precision收据在../../precision_probe/pulled/25944399.pbs101/precision.json。RESULTS.generated.md保留原自动生成报告，remote artifacts未改动。
