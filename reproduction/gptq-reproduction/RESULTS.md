# GPTQ 复现结果

本次复现已完成。实验调用 official `IST-DASLab/gptq` commit `2d65066eeb06a5c9ff5184d8cebdf33662c67faf`，没有修改 GPTQ 核心算法。运行设备为 1 × NVIDIA A100-SXM4-40GB；GPU 作业没有使用 2–4 卡配置。

## OPT-125M smoke

| 方法 | WikiText-2 PPL | 论文参考值 |
|---|---:|---:|
| FP16 | 27.6547 | 27.65 |
| RTN 4-bit | 37.2784 | 37.28 |
| GPTQ 4-bit | 31.5202 | 31.12 |

FP16 与论文参考值相差约 0.005，GPTQ 4-bit 明显优于 RTN 4-bit，因此通过 smoke gate。

## OPT-6.7B

| 方法 | WikiText-2 PPL | 论文参考值 | 总耗时 | 量化耗时 | Peak allocated |
|---|---:|---:|---:|---:|---:|
| FP16 | 10.8606 | 10.86 | 124.2 s | — | 5.83 GiB |
| RTN 4-bit | 12.0989 | 12.10 | 70.3 s | — | 5.95 GiB |
| GPTQ 4-bit | 11.3842 | 11.39 | 966.9 s | 902.2 s | 8.15 GiB |
| GPTQ 3-bit | 15.2132 | 14.86 | 964.8 s | 883.6 s | 8.15 GiB |

GPTQ 4-bit 的结果与论文参考值非常接近，并将 RTN 4-bit 的 PPL 从 12.0989 降到 11.3842。GPTQ 3-bit 仍明显优于 3-bit RTN 的预期趋势；本次没有运行 3-bit RTN。

## Compute accounting

- CPU preparation：`16165215.pbs101`，57 分 55 秒，`ngpus=0`。
- GPU smoke：`16167997.pbs101`，实际 walltime 2 分 18 秒，申请 1 GPU，成功退出。
- GPU main：`16168060.pbs101`，实际 walltime 37 分 41 秒，申请 1 GPU，成功退出。
- 按 1 张 GPU × 实际 walltime 计算，GPU 用时约 **0.666 GPU-hours**；按 ASPIRE2A 的 64 SU / GPU-hour 约 **42.65 SU**。PBS 的 `resources_used.ngpus` 字段显示为 0，但两份作业的 `Resource_List.ngpus=1`，且日志中的 `nvidia-smi` 确认实际 GPU 为 A100-SXM4-40GB；因此使用 walltime 计算更可靠。`myusage` 在本次查询时仍显示旧缓存。
- GPU allocator peak 最高约 8.15 GiB；这不是整卡 `nvidia-smi` 总占用。

## Reproducibility boundary

- Calibration：C4 第一 train shard，seed 0，128 × 2048 tokens；数据和 model revisions 记录在 `manifest.json`。
- Evaluation：WikiText-2 raw test，固定 2048-token windows，实际计入 286,720 tokens。
- Environment：见 `environment.txt`，核心版本为 PyTorch 2.1.2+cu118、Transformers 4.31.0、Datasets 2.14.7。
- 本次验证的是 quantization accuracy 和可复现实验耗时。量化后的权重仍以浮点 Tensor 表示量化网格值，没有验证 packed checkpoint 的磁盘压缩率，也没有验证专用 3-bit inference kernel 的速度。
- `CUDA extension not installed.` 是预期提示，因为本次目标是 GPTQ accuracy；没有运行官方专用 3-bit kernel benchmark。

完整 PBS accounting 和运行日志已保存在 `logs/`；结果 JSON 位于 `results/`。
