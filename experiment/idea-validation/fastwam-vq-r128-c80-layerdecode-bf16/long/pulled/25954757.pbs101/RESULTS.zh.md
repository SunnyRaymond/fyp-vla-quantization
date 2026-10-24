# C80 R128 A+Z+KL：逐层解码及 BF16 对照结果

结论：第一轮达成省显存、相对原量化 eager 降 latency、输出对齐。推荐保留第一轮。两种 BF16 变体本轮均更慢且改变输出，没有直接采用的依据。

## 测量范围与来源

冻结 checkpoint `25921795.pbs101 / reader_kl`；没有训练更新，没有新增 closed-loop episode。新增三种模式各完成40 contexts（3000–3039）×2 seeds，共80 queries；全部在同一张 A100-SXM4-40GB（UUID 341e5ae0-9682-7b35-e706-1e9346eeafbd）运行。614个目标Linears，每模式正式调用514880次；614个独立BF16 scratch decoder及12种F.linear shape均通过精确预检。
按用户要求，BF16/eager/input_fused复用 `25943391.pbs101` 的完整面板。参考与新增模式均为A100，allocation UUID不同；这些对照是历史结果复用，不声称重新进行了同一allocation的配对计时。三个新增模式在同一allocation内按先第一轮、后两个BF16变体顺序执行。
Policy-call latency：prepared GPU datum到infer_action返回并同步，包含内部prompt/text/image-latent及10+10步denoising；排除外部observation preprocessing、environment/robot执行。batch1、32×7actions、9frames、3warmups。主计时无profiler，compile/CUDA graph关闭。deterministic及fill开启，TF32关闭，BF16 reduction flag开启。

## Latency 与显存

GPU GB=10^9 bytes；storage GiB=2^30 bytes。Peak为80次query中的最大值；reserved包含allocated，不能相加。

| 模式 | Mean s | p50 s | p95 s | Peak allocated GB | Peak reserved GB | Warm allocated GB | Load peak GB |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 reference（复用） | 1.451284 | 1.450439 | 1.505931 | 25.130502 | 25.419579 | 24.865422 | 24.831867 |
| 原量化 eager（复用） | 6.547441 | 6.527626 | 6.695142 | 25.139422 | 25.685918 | 24.874590 | 24.901912 |
| 输入融合 dense（复用） | 3.520413 | 3.503950 | 3.724216 | 25.139422 | 25.685918 | 24.874590 | 24.901912 |
| 第一轮：逐层解码 + FP32 transform | 3.593032 | 3.532909 | 3.936028 | 16.550766 | 16.651387 | 16.285505 | 16.251951 |
| 第二轮：BF16 U | 4.136921 | 4.166183 | 4.359765 | 16.550831 | 16.651387 | 16.285571 | 16.252016 |
| 第二轮：BF16 input-transform | 4.023239 | 3.982432 | 4.258963 | 16.554446 | 16.655581 | 16.289185 | 16.255631 |

第一轮相对原量化eager：p50加速 1.847663×、latency下降 45.88%；query peak减少 8.588656GB（34.16%）；load peak下降 34.74%。
与此前输入融合dense版p50 3.504s相比，第一轮为3.533s，基本维持其速度，同时压缩常驻显存。不同allocation和单轮顺序不足以把约0.8%的差异作为稳定因果效应。第一轮仍慢于原BF16 reference的1.450s，不能写成相对BF16加速。
第一轮显存中只保留614份encoding及一个113,246,208bytes（113.25MB）的可复用BF16 weight scratch，目标module.weight均为None。每次Linear用单次Triton解码写入scratch，然后走原cuBLAS F.linear。scratch模式要求顺序、单CUDA stream调用；并发stream需各自buffer。

## 输出变化

以原量化eager（等于第一轮）为参考。motor只含前6维，full32 action含7维。gripper使用前10步>0.5判据，共800个决策。

| 新模式 | Action max abs | Full32 action RMSE | First10 motor RMSE | Full32 motor RMSE | Gripper分歧 /800 | Anchor future NRMSE |
|---|---:|---:|---:|---:|---:|---:|
| 第一轮：逐层解码 + FP32 transform | 0 | 0 | 0 | 0 | 0 | 0 |
| 第二轮：BF16 U | 2 | 0.0760847032 | 0.0456158407 | 0.0518559143 | 3 | 0.25814414 |
| 第二轮：BF16 input-transform | 2 | 0.094297193 | 0.045823589 | 0.0611508824 | 4 | 0.227304906 |

第一轮完整80条action的max/RMSE均0；独立context2000/seed0的action、future max abs与future NRMSE也均0。这里是数值一致；decoder另做过raw BF16 bitwise比较。
BF16 U相对第一轮p50慢17.92%，更广BF16 input-transform慢12.72%。两种变体都改变了当前checkpoint输出；这不能直接推断任务成功率一定下降，也不能称为准确度无损。

## BF16 到底改了什么

BF16 U：U及其GEMM输入转BF16，使用原生BF16 Tensor Core GEMM、保留FP32 accumulator；将U输出转回FP32后进入原A4。Smooth/H及A4算术保留FP32，以单独评估U。
BF16 input-transform：进一步将Smooth scale、H butterfly中间值、H/U全局输出及A4相关值按BF16保存/舍入。除法、reduction及rounding helper仍有FP32；weight decoder保持原FP32相加缩放→BF16舍入。它并非整模型所有操作全部原生BF16。
BF16诊断所需的原FP32 reference transforms/rotations在本轮仍驻留，故BF16 input-transform约多3.68MB。这不能作为部署清理后小参数存储的精确上限；与16.55GB整体相比很小。第一轮没有新增这份BF16副本。

## Profiler 提示

以下来自正式计时后额外的intrusive query。ATen与CUDA leaf可能重叠；不相加，不当作E2E严格分解。

| 项目 | 第一轮 ms / calls | BF16 U ms / calls | BF16 input ms / calls |
|---|---:|---:|---:|
| _decode_kernel | 286.306 / 6436 | 333.049 / 6436 | 332.330 / 6436 |
| aten::addmm | 262.545 / 6436 | 307.819 / 6436 | 306.457 / 6436 |
| aten::mm | 138.458 / 6593 | 65.923 / 6593 | 64.677 / 6593 |
| aten::copy_ | 53.186 / 9259 | 141.887 / 22109 | 60.091 / 9259 |
| aten::fill_ | 119.113 / 50027 | 177.296 / 62877 | 126.636 / 50027 |

U改BF16后，全部aten::mm的self-device时间从约138ms降到66ms，但copy_从9259次增至22109次，fill从50027次增至62877次。更广BF16数据流消除了部分来回转换，但其H kernel也比FP32版慢。直接改dtype没有带来本轮E2E收益。
同时，同一decode与主Linear在后两个profiler query也更慢；固定顺序、运行状态和profiler扰动仍可能影响时间。没有硬件counter或独立反向顺序复测，故不把全部端到端差异因果归给某一个cast或BF16本身。
新路径把weight解码和主GEMM交回高效独立kernel，避免旧native packed GEMM的主要问题；当前trace中decode累计约286ms，主addmm约263ms。后续若继续加速，应结合CPU dispatch、重复小GEMM、allocation/转换和融合评估，而非仅依据FP32标签。

## Weight storage

三种实验复用同一个inference bundle，未生成新权重训练结果或另一个BF16 archive。
| 项目 | GiB |
|---|---:|
| 原目标BF16 weights | 11.206264496 |
| Packed encoding | 2.982873065 |
| Encoding + Smooth/sign/U | 2.991408997 |
| 其余未量化model parameters | 11.902287906 |
| Whole bundle unique tensor payload | 14.893702424 |
| inference_bundle.pt实际文件 | 14.895613806 |
| 原dense model unique tensor payload | 23.108552402 |

目标开销包含transforms为4.271052497bpw，目标weight约3.746×压缩，whole tensor约1.551565×压缩。共享storage去重；外部cached VAE/text/tokenizer assets不在archive总数中。

## 执行与验收

Winner `25954757.pbs101`，normal→g1，申请6h，实际26m27s；PBS F/Exit0、artifact exit0、PIPELINE_COMPLETE及controller verify通过。短道 `25954759.pbs101` 申请1h59，在Q时被精确qdel，返回0并核实F。原子启动锁防止两份模型同时运行。GPU利用率/显存15秒采样写入job log。gpt-6-luna/xhigh监控已停止。
第一轮全部gate通过后才运行两个BF16变体。每个新增mode均有614编码模块、0 dense目标weights、514880个Linear与decode调用；完整80条样本及有限值已复核。对照未重跑。
r1在用户要求复用对照后取消；r2因提交目录命名问题在模型计算前退出；两者不计入本次性能结果。相关终态在superseded_r*.json与attempts/保留。

证据：summary.json、stage1_gate.json、kernel_validation.json、completion_evidence.json、各mode的summary/samples/validation/operators，以及上层race_status.json/monitor_status.json。完整action.pt与Chrome trace保留在集群。
