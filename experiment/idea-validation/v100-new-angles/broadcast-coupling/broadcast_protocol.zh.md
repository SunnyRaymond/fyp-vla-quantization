# Broadcast activation rounding：单步空间相关性 screen

2026-09-13，读新 numerical outputs 前冻结。C02 same-locus/same-error-budget + C04 broadcast coupling。原 encoder-W4 versus predictor-W4 案保留 prior no-go；本案的 weights 全为 FP32，仅改变相同 broadcasted activation 的 stochastic rounding 空间耦合。原案尚未实测，本案不是修复一个 empirical no-go。Generic correlated-rounding novelty 未确证；不宣称新 quantizer。

## 可证伪假设

DINO-WM Wall 把 proprio/action embedding 沿196个 image patches复制。对20个复制坐标应用相同 A4 quantizer，当量化误差不再对所有 patches 完全相干，单步 predictor visual error 可能下降。比较 Shared（量化后broadcast）与 Spatial（broadcast后量化、固定三个draw的空间重排）；FP predictor、FP visual channels、scale、每个patch在三个draw中的量化输入集合完全相同。保持每patch marginal set和总 input squared-error **严格匹配**，不拿不同模块的误差来解释 broadcast。

## 输入与固定执行

复用 teacher_bias_ready2/manifest.json SHA256 6dae677a0e7763896d735495500aea78a9254e59b67fa47817bb9bb9bb1f5c14；仅用每条的第一帧 observation 与第一组 normalized action（5 primitive actions）。Wall valid124..129→1035/1534/1158/203/1837/1095；不读取旧 teacher scientific outputs。checkpoint/source/helper 与该manifest相同且重新核验，raw proprio2→embedding10、action10、visual384，concat_dim1、num_hist1、196patch、total404。

所有 weights FP32/eval/no_grad、无训练。`z=model.encode(obs0,actions[:,:1])` 固定形状[1,1,196,404]；20个tail coords逐patch必须exact equal。Scale按每state的20D vector maxabs/7，zero vector scale1；一个scale不会与别state/draw计算绑定。signed A4 levels[-7,7]。CPU torch float32 uniform[20]使用三个独立 seed2501/2502/2503（六state复用同U）；在CPU float32做 u=x/scale, lower=floor(u), code=clip(lower+(U<fraction),-7,7)，送CUDA的FP32 dequantized inputs。CPU scalar quantization preparation在本次GPU allocation内执行；不在本机/head计算。

Shared draw d 将base_quant[d,20]复制到全部196patch。Spatial draw d 的patch p用 base_quant[(d+offset[p])%3]。offset是CPU torch.randperm(196,seed2601)对 arange(196)%3 的重排，六states共用。每patch跨三draw的multiset完全相同；不声称patch独立随机。Primary按全部三draw均值，不能选择有利draw。

每state一次FP，一次FP identity copy，一次RTN-before，一次RTN-after，以及Shared3/Spatial3共10个单步predict calls。FP target为未量化的同一 model.predict(z) visual first384；不使用记录future/goal/return。输出固定逐call batch1，避免batch/kernel混杂。RTN-before/after输入与visual输出必须exact equal，FP copy output allclose(atol1e-6,rtol0)。用 predictor pre-hook 保存actualseeninput并核对与预期逐元素相同；完整model state digest before/after一致。

## 判据

所有 source/allocation/quantization/marginal/total-input-MSE/readback/no-op/finite gates通过；六state全完成。每state Shared平均visual MSE>1e-12且三个base draws至少两种不同，Spatial input与Shared至少某patch不同；六个都binding，否则 inconclusive_binding。Gain=(Eshared−Espatial)/Eshared；至少5/6 gain>=.10 => scope_limited_preliminary_go，否则 mechanism_no_go。输入 matched-MSE 误差容差绝对1e-12+相对1e-7（CPU float64重算）；其余multiset/code/dequant/readback用exact comparison。

这只证明或反驳特定 activation coupling 对该checkpoint单步FP visual fidelity的影响。Spatial破坏每次forward中的embedding exact replicas是实验自变量，本案不把它隐藏成误差更小；不保证真实future/action/planner/success，也不宣称 compressed-storage、latency、native kernel或通用novelty。原有temporal weight persistence/antithetic ensemble不扩验。

GPU一次5分钟，内部240秒，外部270秒；CPU verification一次5分钟。结果无论go/no-go/inconclusive均STOP；不能追加draw/scale/任务/horizon。不完整不分析partial science。有明确实现错误才另立repair gate。

## raw.npz 精确合同

float32：initial_z[6,196,404]，base_quant[6,3,20]，uniform[3,20]，scale[6]，rtn_vector[6,20]，seen_tail[6,2,3,196,20]，pred_visual[6,2,3,196,384]，fp_visual[6,196,384]，fp_copy_visual[6,196,384]，rtn_visual[6,2,196,384]。int8 codes[6,3,20]；int64 offset[196]、valid_indices[6]=124..129、trajectory_ids[6]=[1035,1534,1158,203,1837,1095]；bool completed[6]。arm order Shared,Spatial；rtn_visual order before,after。不要求schema数组。

CPU独立replay重建quantizer/RNG/实际tail输入、比较逐patch三draw排序multiset，重算input误差预算和visual MSE/gain。engineering.json另保留完整source/checkpoint/helper/输入/producer/protocol/FPweightsdigest/actualhook checks和rawSHA。大数组留compute filesystem，只下载小JSON结论。
