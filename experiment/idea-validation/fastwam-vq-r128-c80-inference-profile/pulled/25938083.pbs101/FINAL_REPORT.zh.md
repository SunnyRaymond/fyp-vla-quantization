# C80 R128 A+Z+reader KL：inference-only export 与 profiling 最终报告

成功 job `25938083.pbs101`，PBS F/Exit0，artifact exit0 与 PIPELINE_COMPLETE 均通过。Source checkpoint `25921795.pbs101 / reader_kl`，新增训练更新0。
短任务25938321.pbs101因长任务先启动而取消，qdel返回0并核验F；监控已停止。

## 结论

Export已能恢复原模型数值，并且不再依赖训练checkpoint或source banks。仅做decoded-dense export能清理重复状态，不能获得低bit计算加速。Packed-resident诊断已经实现实际显存下降，但逐层完整decode进一步增加耗时。下一步应先减少输入变换的小算子，再实现packed VQ tile decode与BF16 GEMM融合。

## 数值与测量协议

同一张NVIDIA A100-SXM4-40GB，Torch2.7.1+cu128 / CUDA12.8；各模式独立fresh process，3次warmup。主要面板固定40 contexts（3000..3039）×2 seeds，每组80次；batch1、32 action horizon、9 video frames、10 video+10 action denoising steps，torch.compile/CUDA graph关闭。
测量边界为prepared GPU datum到action output，包含policy内部prompt/text encoding；排除observation读取/预处理、环境step和机器人动作执行。Observation-to-datum另存每个sample的preprocessing_wall_ms。主计时无详细profiler，所有模式使用相同QueryMeter的两次CUDA event和wall计时。
Decoded export与legacy的80个完整32×7 action逐一比较，最大绝对误差0；保存的完整action/future anchor在legacy、decoded export、packed-resident三条路径均为0。Packed-resident仅验证2000/seed0与3000/seed0两个诊断query，二者action误差均为0，未将其伪装成80次结果。

## 完整80-query主表

| Mode | n | p50 wall s | p95 wall s | speedup vs BF16 | warm allocated GB | query peak allocated GB | query peak reserved GB |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 reference | 80 | 1.4811 | 1.5382 | 1.0000x | 24.8654 | 25.1305 | 25.4196 |
| Current legacy VQ | 80 | 7.0838 | 7.1793 | 0.2091x | 28.2101 | 28.4749 | 28.7499 |
| Inference export, decoded BF16 | 80 | 6.7141 | 6.9374 | 0.2206x | 24.8746 | 25.1395 | 25.6083 |

以上全部来自最终成功job，未与前一轮25937618的部分结果拼接。Speedup=BF16 p50/该模式p50，低于1即更慢。GPU GB使用10^9 bytes，后续storage GiB使用2^30 bytes。
Decoded export的query peak比legacy低11.71%，回到接近BF16的水平；其p50仍为BF16的4.53倍。这轮单次固定顺序测量不用于宣称数个百分点改善具有统计显著性。

## Packed-resident：只作2-query诊断

Warm allocated **16.1742 GB**；query peak allocated **17.0884 GB**；peak reserved **17.4441 GB**。
Warm resident较BF16低34.95%；共享的3000/seed0 query上peak allocated低32.00%。两个query耗时分别见samples，范围10.7173–10.8748秒。这是真实compressed weight常驻证据，但当前执行仍逐Linear完整decode成BF16，再调F.linear；不是fused VQ kernel，也不是完整面板speedup结果。

## 加载瞬态：与query峰值分开

| Mode | reported load seconds | load-stage peak allocated GB | load-stage peak reserved GB |
|---|---:|---:|---:|
| bf16 | 455.291 | 24.8319 | 25.0274 |
| legacy | 592.628 | 29.0827 | 29.7292 |
| inference_dense | 87.970 | 24.9019 | 25.2979 |
| stream_decode | 81.088 | 24.8319 | 25.0274 |

这些load字段的工作范围不相同：legacy包括source恢复和两次prepare anchor验证，不能计算公平cold-start speedup。Packed路径的启动峰值仍约24.83GB，因为factory先构造完整BF16模型再移除目标weights；不能把17.09GB query peak当启动预算。后续loader应直接构造空/meta目标weights并安装编码。

## Weight存储

| 项目 | GiB |
|---|---:|
| Target BF16 weights | 11.206264 |
| Packed target encoding | 2.982873 |
| Encoding + Smooth/sign/U | 2.991409 |
| Unquantized model parameters | 11.902288 |
| Whole bundle tensor payload | 14.893702 |
| Actual inference_bundle.pt archive | 14.895614 |
| Original dense unique tensor payload | 23.108552 |

614个目标Linear、6,016,317,440个weight values。Encoding为4.258865 bits/target weight，包含Smooth/sign/U后4.271052 bits/target weight。目标encoding压缩3.7569x；整模型unique tensor payload压缩1.5516x，减少35.55%。Shared parameter aliases只计一次；没有把optimizer或重复state引用作为BF16部署基线。
Bundle包括未量化persistent state、最终编码、transforms、固定FP32 U、normalizer与minimal resolved config，排除目标dense BF16 weights、训练masters、optimizer和frozen reader。Runtime仍需要FastWAM代码和cached base VAE/text/tokenizer assets；14.896GiB archive不等于完整可搬运部署目录大小，外部cache未计入。
Remote bundle：`/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-inference-profile-20261023-r2/results/export/inference_bundle.pt`。完整大文件和Chrome traces留在compute filesystem；只拉回小型JSON/Markdown/log receipts。

## 耗时瓶颈

单独的intrusive query观察到614个目标Linear被调用6436次，Smooth/H、U、A4各进入6436次，aten::cat调用45653次。H128七级butterfly、FP32转换和A4 reduction/round/clamp产生大量小算子及中间张量。Legacy已常驻decoded weights，其DecodedWeight.forward只返回dense view，所以当前慢不能归因于每次都在解码weight。

| Component | intrusive CUDA-event elapsed ms |
|---|---:|
| smooth_signed_h | 5923.336 |
| a4_quant_dequant | 2362.363 |
| learned_r | 1293.875 |

全部被捕获的F.linear GEMM envelopes（含target与非target）合计1358.499ms；它与component/stage/Linear事件相互嵌套，而且CUDA事件包含launch间隙和profiler开销。以上不能相加、不能当无hook主延迟的分解百分比，也不能用来计算Amdahl加速上界。KV cache实际payload为108,380,160 bytes（约103.36MiB），保持BF16。

## 已确定的kernel实现顺序

1. 融合FP32 Smooth/sign/H128；先保留原FP32 U GEMM。再融合整行A4 maxabs、safe divisor、round/clamp、乘回scale与BF16 cast。保持当前whole-K per-token scale，不改activation clipping或B32 activation格式。
2. 实现packed VQ tile decode + BF16 Tensor Core GEMM。读取d4的两枚uint8 index，FP32相加两个BF16 codewords，应用原BF16 row gain与B32 exponent的ldexp scale，在MMA前round成BF16；FP32累加，保留bias/output dtype。避免在global memory生成整层weight。
3. 先用Triton原型验证数值与收益；若gather/layout/register pressure成为瓶颈，再实现CUDA/CUTLASS SM80专用版本。大M video、小M action和M129 context projection分别dispatch，小型tail模块可用BF16 fallback并披露coverage。
4. 另改loader的dense初始化与重复base weight读取，降低启动峰值。INT8/INT4/FP4属于另一个需要重新表示/校准的baseline，当前任意BF16 codewords不能直接作为整数MMA operands。

| Target stream | M | N | K | calls/query | intrusive GEMM ms |
|---|---:|---:|---:|---:|---:|
| video | 294 | 3072 | 3072 | 1991 | 385.615 |
| action | 32 | 3072 | 1024 | 1200 | 227.503 |
| video | 129 | 3072 | 3072 | 671 | 128.508 |
| action | 32 | 1024 | 3072 | 600 | 119.691 |
| action | 129 | 3072 | 1024 | 600 | 113.992 |
| video | 294 | 14336 | 3072 | 330 | 109.616 |
| video | 294 | 3072 | 14336 | 330 | 102.099 |
| action | 32 | 1024 | 4096 | 300 | 59.724 |
| action | 32 | 4096 | 1024 | 300 | 56.369 |

这些是本次实际shape，video主query M=294、action M=32，context/proprio拼接投影M=129。不能套用M=1的LLM decode kernel。Raw analyzer把恰好名为proprio_encoder的单个M1/N4096/K8行归到other；reviewed plan已用module/storage receipt纠正为tiny fallback，未改动原始结果。其余other GEMM保留为未量化组件开销。
详细decoder、layout、AQLM可复用边界和验收顺序见项目根目录KERNEL_DESIGN.zh.md；机器可读最终方案见REVIEWED_KERNEL_PLAN.json。

## 验证与边界

GPU telemetry配置每15秒采集，log中共224条样本，保存utilization和used/total memory；采样最大显存不是精确allocator peak。所有模型读取、export、profiling与重IO均在PBS allocation内执行。
成功收据：completion_evidence.json、parity.json、summary.json、per_layer/*.json。未重训，未新增闭环episodes，未实现native fused VQ或native INT4/INT8 arithmetic kernel，因此目前没有native-kernel speedup结论。
