# C80 R128 A+Z+reader KL inference export 与 profiling

PBS job `25938083.pbs101`；source `25921795.pbs101` / reader_kl；new training updates = 0。

导出：`/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-inference-profile-20261023-r2/results/export/inference_bundle.pt`。Architecture 与 cached VAE/text/tokenizer assets 仍为运行依赖。

完整40 contexts ×2 seeds action parity max abs = 0；另通过保存的完整 action/future anchor。

| Mode | n | policy p50 ms | p95 ms | speedup vs BF16 | peak allocated GB | peak reserved GB | resident GB |
|---|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 80 | 1481.098 | 1538.208 | 1.000x | 25.1305 | 25.4196 | 24.8654 |
| legacy | 80 | 7083.761 | 7179.290 | 0.209x | 28.4749 | 28.7499 | 28.2101 |
| inference_dense | 80 | 6714.148 | 6937.389 | 0.221x | 25.1395 | 25.6083 | 24.8746 |
| stream_decode | 2 | 10796.021 | 10866.894 | 0.137x | 17.0884 | 17.4441 | 16.1742 |

Latency 从已准备的 GPU datum 到 action output；观察预处理另报，排除环境和机器人动作执行。
stream_decode 只有2个诊断 queries，逐Linear临时解码完整weight，不作主speedup结果。所有模式使用BF16 GEMM；未实现native VQ low-bit arithmetic。

Packed encoding payload：2.982873 GiB；4.258865 bits/target weight。
加入 transforms/U：2.991409 GiB；4.271052 bits/target weight。
Whole inference bundle tensor payload：14.893702 GiB；实际 archive：14.895614 GiB。

详细 trace 为单独 intrusive query；nested timing 不能相加，不能替代上表80次无hook测量。

| Highest-cost GEMM stream | M | N | K | calls | intrusive GPU ms |
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
| video | 294 | 18432 | 3072 | 11 | 3.626 |
| video | 129 | 3072 | 4096 | 11 | 2.049 |
| video | 294 | 3072 | 256 | 11 | 2.047 |

实现方案：首版保持当前数值，采用 indices tile-load → 双codeword FP32相加 → B32 scale → BF16取整 → BF16 MMA/FP32累加 → bias；移除global memory里的完整decoded weight。
按上表实际shapes选择dispatch；再根据已测Smooth/H/U/A4时间决定融合输入变换。当前checkpoint不直接改成INT8/INT4/FP4。

内存的load transient peak、warm resident、query peak分别保存；不得把packed archive压缩比解释为整模型显存压缩比。
该报告提供固定输入的部署与性能证据，未新增闭环episodes。
