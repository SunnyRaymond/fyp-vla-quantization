# Inference-only export：storage receipt

Source：`25921795.pbs101 / reader_kl`；export writer：`25937618.pbs101`。Export已由成功job `25938083.pbs101` 完成reload、完整80-query action parity与完整action/future anchor验证，误差均为0。完整性能结果见 `pulled/25938083.pbs101/FINAL_REPORT.zh.md`。

614 个目标 Linear，6,016,317,440 个 weight values。无新增训练更新。

| 项目 | bytes | GiB |
|---|---:|---:|
| Target BF16 weight reference | 12,032,634,880 | 11.206264 |
| books | 2,514,944 | 0.002342 |
| row_scales | 4,147,598 | 0.003863 |
| indices | 3,008,159,232 | 2.801567 |
| scale_exponents | 188,013,792 | 0.175101 |
| Packed target subtotal | 3,202,835,566 | 2.982873 |
| Smooth scales and signs | 9,034,315 | 0.008414 |
| Fixed FP32 U | 131,072 | 0.000122 |
| Target encoding with transforms/U | 3,212,000,953 | 2.991409 |
| Unquantized model parameters | 12,779,984,326 | 11.902288 |
| Processor stats | 5,928 | 0.000006 |
| Whole bundle tensor payload | 15,991,991,207 | 14.893702 |
| Actual inference_bundle.pt archive | 15,994,043,538 | 14.895614 |
| Original dense model tensor payload | 24,812,619,206 | 23.108552 |

目标 weights 的有效编码位宽为 **4.258865 bits/weight**；包含 Smooth/sign/U 后为 **4.271052 bits/weight**。索引部分为4 bits/weight；总量还包括B32 exponents、row gains、codebooks。
目标 weight payload 压缩比 **3.7569×**。Whole-model tensor payload 压缩比 **1.5516×**，减少 **35.55%**。

Whole-model比较使用两端 unique tensor storage。实际archive大小单独给出；没有将原训练checkpoint/optimizer文件大小混作BF16部署文件大小。共享parameter alias按底层storage只计一次，padding按实际storage计入。原模型logical state引用合计约45.54GiB，但存在大量alias，不能当实际weight占用。

| Stream | Modules | Weight values | Encoded bytes | GiB |
|---|---:|---:|---:|---:|
| video | 306 | 4,996,792,320 | 2,658,380,160 | 2.475809 |
| action | 307 | 1,019,492,352 | 544,422,638 | 0.507033 |
| proprio | 1 | 32,768 | 32,768 | 0.000031 |

未量化部分约11.902GiB，占完整bundle大多数。即使目标weights大幅压缩，整模型的压缩也受这一部分限制。以上是存储账，不是GPU peak memory结果。

Bundle包含目标VQ编码、未量化model persistent state、固定transforms/U、normalizer和minimal resolved config。目标dense BF16 weights、FP32 training masters、optimizer、frozen reader被排除。推理仍需要FastWAM代码和cached base VAE/text/tokenizer assets。Archive大小不包含外部cache，因此不能据此声称整个可搬运部署目录的磁盘用量也是14.896GiB；后续loader应避免重复加载base weight文件，并单独打包tokenizer等非weight资产。

Remote bundle（保留在compute filesystem，不下载大文件）：

`/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-inference-profile-20261023-r2/results/export/inference_bundle.pt`
