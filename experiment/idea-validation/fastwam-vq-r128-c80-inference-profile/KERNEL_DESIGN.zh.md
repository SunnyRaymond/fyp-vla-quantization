# C80 R128 A+Z+reader KL：保持 checkpoint 的部署与 kernel 方案

当前状态：profiling job `25938083.pbs101` 已成功完成（PBS F/Exit0、artifact与summary gates通过）。Decoded export完整80-query action parity与完整action/future anchor均为0；packed-resident两条诊断action及anchor误差也为0。该文是根据实测确定的实现方案，尚未实现native fused VQ kernel。

## 先把三种“量化部署”分开

1. **Packed file**：磁盘上保存 codebooks、indices、B32 exponents 和 row gains。文件变小并不保证推理显存或耗时下降。
2. **Decoded dense inference**：加载时解码一次，GPU 常驻 BF16 weights；去除训练 masters、reader、optimizer 和重复编码。这是当前 export 的正确性参考。
3. **Packed resident inference**：GPU 常驻编码，执行 Linear 时直接消费编码。只有 tile decode 与 GEMM 融合，才能避免每次先生成整层 BF16 weight。当前 `stream_decode` 只验证压缩常驻与数值，仍逐层完整解码，不能作为 fused kernel 性能结论。

三者均保持现有 checkpoint；这轮不重新量化、不校准新格式、不训练。

## 本轮实测与实现优先级

同一A100，三组各80次无详细profiler的query：BF16 p50 1.4811s / peak allocated25.1305GB；legacy VQ 7.0838s /28.4749GB；decoded inference export 6.7141s /25.1395GB。Export清理了重复状态，但仍是BF16的4.53倍耗时。最终主表仅使用本轮结果，不拼接旧job。

Packed-resident仅两条预声明诊断query：warm16.1742GB、query peak17.0884GB，耗时10.7173–10.8748s。它证明真实压缩常驻可行，但完整weight临时decode不适合争取speedup。其load-stage峰值仍24.8319GB，factory需要后续改造。

Intrusive单query中，614个目标Linear共调用6436次，Smooth/H、U、A4各进入6436次，aten::cat共45653次。H128的七级stack/butterfly产生大量小算子。CUDA event elapsed及嵌套profiler时间不能相加或解释为无hook主延迟占比。因此执行顺序确定为：先融合输入路径的小算子，随后实现packed-weight consumer，并独立处理loader启动峰值。

| Kernel priority shape | M | N | K | calls/query |
|---|---:|---:|---:|---:|
| video | 294 | 3072 | 3072 | 1991 |
| action | 32 | 3072 | 1024 | 1200 |
| video | 129 | 3072 | 3072 | 671 |
| action | 32 | 1024 | 3072 | 600 |
| action | 129 | 3072 | 1024 | 600 |
| video | 294 | 14336 | 3072 | 330 |
| video | 294 | 3072 | 14336 | 330 |
| action | 32 | 1024 | 4096 | 300 |
| action | 32 | 4096 | 1024 | 300 |

这些是实测shape：主video M294、action M32，text+proprio context projection M129；不沿用LLM的M1 matvec配置。完整性能表及证据边界见 `pulled/25938083.pbs101/FINAL_REPORT.zh.md`，机器可读reviewed方案见同目录 `REVIEWED_KERNEL_PLAN.json`。

## 必须保持的数值顺序

对 Linear 的输入，现有顺序为：

```text
FP32(X) / Smooth scale
→ saved signs
→ normalized H128 butterfly（不足128的尾部遵循原实现）
→ saved FP32 U（仅完整128块；不足128的尾部不乘U）
→ 每个 token 整行 max(abs(.)) / 7
→ round-to-nearest-even、clamp[-7,7]、乘回 scale
→ BF16 operand
```

Activation 的 scale 覆盖整个 K。把它改成每32或128元素一个 scale，会改变当前模型；不能在“保持 checkpoint 的 kernel 优化”中悄悄这样做。全零行的 safe divisor 为1，乘回的真实 scale 仍为0。

Weight 的每4个数来自两个 uint8 indices，分别查两个 `256×4` BF16 codebooks。对输出行 n、输入列 k：

```text
v = FP32(book0[index0, k%4]) + FP32(book1[index1, k%4])
s = ldexp(FP32(row_gain[n]), int(exponent[n,k//32]) - 127)
w = BF16(v * s)
y = BF16-operand × BF16-w，FP32 accumulation，原 bias/输出 dtype
```

必须在 MMA 前把 weight 取整到 BF16。直接让 FP32 codeword 参与乘加、把 scale 挪到 GEMM 输出端，或分别计算两个 codebook 的 GEMM 后再相加，都可能改变原模型的 BF16 rounding。

## 建议的实现顺序

### 第一步：先减少输入路径的小算子

现有 PyTorch H128 butterfly 会创建大量 slice、cat、add/sub 等操作，随后 U 和 A4 又引入中间张量及 launch。先做两个边界清楚的融合：

- 将 Smooth 除法、sign 和七级 H128 butterfly 融成一个 kernel，保持各级 FP32 顺序及归一化。
- 保留现有 FP32 U GEMM；将其后的整行 absmax、safe-scale、round、clamp、乘回和 BF16 cast 融成一个 row-wise kernel。

这一步保留 cuBLAS GEMM，改动面小，适合先确认 launch 数和中间张量的下降。U 不在首版中改成 TF32/BF16，也不直接把 H 与 U 合成预乘矩阵；后者虽然代数相同，但浮点累加顺序不同。A4 的 row-wise reduction 与按 K tile 的 GEMM 不天然同边界，因此首版无需追求“一个 kernel 做完全部事情”。

### 第二步：packed VQ tile decode + BF16 Tensor Core GEMM

接口只需要 BF16 activation、两个 codebooks、indices、exponents、row gains、bias 与输出；GPU 上不保留目标 Linear 的完整 decoded weight。

对每个 M/N output tile：沿 K 循环读取编码块，查表并按上述顺序生成局部 BF16 weight tile，立即送入 BF16 MMA；FP32 accumulator 最后加 bias 并输出。生成的 weight tile 只在寄存器或 shared memory 中短暂存在。

- `BLOCK_K` 为32的倍数，同时满足 d4 索引边界；尾部 load/store 加 mask。
- 编码dtype/shape、finite values和保留exponent255的检查在安装时完成一次。当前eager `decode_vq` 每次还会调用 `check_encoding` 并触发host同步；这类诊断开销不应留在fused kernel热路径中。
- 每层两个 BF16 codebooks 共4096 bytes，适合复用；缓存的具体位置依据编译后 register/shared-memory 开销决定。
- indices 可离线按 N/K tile 重排，保留两个原始 uint8 值；每个 d4 用16 bits，即索引部分4 bits/weight。B32 exponent 和 row gains 必须另外计入。
- 首版以 Triton `tl.dot` 的 BF16 输入、FP32 accumulator 做原型；如果 lookup/layout conversion 或 register pressure 成为主要限制，再写 CUDA/CUTLASS SM80 专用实现。不要先引入两套框架。
- 实际 video 大 M 与 action 小 M 分开 dispatch，tile 参数从本次 profile 的 M/N/K 表决定。不能照抄 LLM 的 M=1 GEMV 配置，也不能把非目标 text encoder GEMM 当 VQ Linear 来优化。
- 对很小、收益不足的模块保留 BF16 fallback；必须在 coverage 表中披露覆盖的模块数、weight values 和实际调用时间，不能只报告最快的一层。

Loader 也需要跟进：当前 factory 仍先构造完整 BF16 模型，再替换目标 weights。因此 packed 常驻下降之后，load transient peak 仍可能偏高。真正部署版应让目标 Linear 从空/meta weight 构造，直接安装编码；这属于加载路径优化，不改变 codewords。不能用 warm resident 的下降冒充启动峰值也已经下降。

当前 factory 还会访问 cached base VAE/text/tokenizer assets。完整 bundle 的 tensor state 与“脱离现有缓存、可独立搬运的部署目录”也要区分；本轮 archive bytes 不包含外部缓存。后续 loader 应从 bundle 构造已有参数，避免重复读取 base weight 文件，tokenizer等非weight资产另行打包。

Triton 的 blocked matmul 与 `tl.dot` 支持这条原型路线；NVIDIA 文档也列出 SM80 的 BF16 TensorOp/FP32 累加支持。它们提供实现基础，不保证我们的 VQ gather kernel 能胜过 cuBLAS。[Triton matmul](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)、[tl.dot](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)、[CUTLASS functionality](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/functionality.html)。

### 现成 AQLM kernel 能复用到什么程度

检查了 AQLM 官方 kernel selector 和 CUDA 源码：专用2×8-bit CUDA分支匹配 `2×256×1×8` codebook，即 d8；其 `code2x8_matmat` 对每个输入行调用 matvec，另一条 dequant 路径先生成整层 weight 再调 `F.linear`。因此它们是查表、shared-memory布局和dispatch的参考，不是当前 d4/B32 格式可直接使用的 fused Tensor Core GEMM。[selector](https://github.com/Vahe1994/AQLM/blob/main/inference_lib/src/aqlm/inference_kernels/kernel_selector.py)、[CUDA wrapper](https://github.com/Vahe1994/AQLM/blob/main/inference_lib/src/aqlm/inference_kernels/cuda_kernel.cpp)。

尤其需要改的有三处：d8→d4索引/向量布局；输出行scale→K维每32元素scale；原kernel的BF16 codeword相加与输出端scaling→我们要求的FP32 codeword相加、B32 scale、再BF16 rounding。复制kernel然后只换shape，不能保证数值一致。逐行matvec也会对 video 的 M=294 重复launch，不能把其LLM decode speedup外推到本模型。[CUDA decoder](https://github.com/Vahe1994/AQLM/blob/main/inference_lib/src/aqlm/inference_kernels/cuda_kernel.cu)。

### 第三步：再决定是否需要 INT8 路线

本模型的 indices 表示任意 BF16 codewords，不是 INT4/INT8 weight 值。因此，“约4 bits/weight 的存储”与“INT4 Tensor Core 计算”是两回事。现有 A4 解码值可以用整数码加 scale 表达，但 weight 仍需匹配整数 GEMM 的表示；直接把两个 index 当整数乘是错误的。

首版建议以 **compressed-weight BF16 compute** 达成真实显存节省，再判断延迟是否值得进一步改格式。INT8 可以成为后续独立 baseline，但会引入新的 weight rounding 与数值验证，不能称作原 checkpoint 的无损 kernel 替换。

同样，B32 exponent 与 MX 格式共享部分设计思想，但两个 additive codebooks 并不是 MXFP4。Triton 的原生 block-scaled FP4/FP8 tutorial 面向 NVIDIA compute capability10 等支持硬件，不能把它作为当前 A100 的直接执行路径。[官方 block-scaled matmul](https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html)。

## 实施时的最小验收

1. Tile decoder 与当前 `decode_vq` 的 BF16 weight 比较，包括非整除尾部、zero row gain、合法 exponent 边界；要求 decoded weight bitwise 一致。
2. 输入融合与当前 input operand 比较，尤其整行 maxabs、全零行、round ties 与 H128 尾部。GEMM 累加顺序改变则另外报告输出误差，不能承诺 bitwise 相同。
3. 完整40 contexts×2 seeds 的配对 action replay，以及保存的完整 action/future anchor；不缩成少数有利输入。若 kernel 改变数值，后续另做闭环验证。
4. Fresh process 测 GPU resident、query peak allocated/reserved 与 load transient；确认没有完整 decoded weights、source banks 或训练 masters 常驻。只看 packed 文件大小不能通过此项。
5. Kernel microbenchmark 与完整 policy latency 分开，后者包含输入 transform、A4、decode/MMA、attention/cache 和其余组件。使用无 profiler hooks 的80-query主面板，报告 p50/p95、同 GPU BF16 comparator 和真实 coverage。单次 intrusive trace 只用于定位，嵌套时间不可相加。

当前仅完成 export/profiling 与设计，不实现或宣称 native fused VQ kernel speedup；也不新增闭环实验。
