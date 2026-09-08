# 06 — Towards Accessible Physical AI: LoRA-Based Fine-Tuning of VLA Models for Real-World Robot Control

- 定位：consumer GPU 上的 low-cost VLA adaptation 與 NF4 deployment case study。
- Source：[arXiv 2512.11921](https://arxiv.org/abs/2512.11921)
- Local PDF：[paper-arxiv-v1.pdf](paper-arxiv-v1.pdf)

## 為何納入

Paper 以 VLA real-robot control 為主題，明確使用 4-bit NF4、double quantization 與 LoRA，並報告 RTX 4060 上的 VRAM、latency、control rate 和 success。

## Background / Problem

大型 VLA 很難在 8 GB consumer GPU 上 fine-tune 與 inference。作者希望建立一條低成本 SO-101 robot pipeline，研究 demonstration count 與 vision encoder unfreezing 對 real task 的影響。

## Method 與 Key Innovation

- Paper 描述一個約 3.1B 的 SmolVLA-like model：Phi-2 language backbone 加 SigLIP-SO400M vision encoder。
- 以 LoRA rank 8 做 adaptation，weights 採 4-bit NF4 與 double quantization，compute 使用 FP16。
- 使用 dual cameras 與 SO-101，測 20/50/100/200 demonstrations；比較 frozen 與 unfrozen vision encoder。

## Main Results

- 200 demonstrations：frozen vision encoder 74%，unfrozen 76%。
- 約 22.2 predictions/s、45 ms mean latency；paper 報告 20 Hz control。
- Peak VRAM 約 6.8 GB、mean 約 6.2 GB，可在 RTX 4060 8 GB 運行。

## Limitations / Evidence Boundary

只有 button-pressing 單一 task、單一 robot，沒有 full-precision 或其他 quantizer 的 matched baseline，也沒有 long-horizon test。Paper 對 model architecture / parameter count 的描述應在引用 numerical claims 前與官方 SmolVLA checkpoint 獨立核對；因此本 corpus 只把它當 feasibility evidence，不作 method ranking。

## Why It Matters

它補足「研究室是否能用 commodity GPU 做 VLA adaptation」的 practical question，並提醒 fine-tuning memory、inference latency 與 task diversity 是不同證據。

## Reading Route

- 20 minutes：hardware setup、quantization config、demo-count result。
- 60 minutes：核對 dataset collection、action interface、success definition 與 model provenance。

## Reading Questions（不附答案）

1. NF4 是 storage format 還是所有 inference arithmetic 都是 4-bit？
2. 20 Hz control 與 22.2 predictions/s 的 measurement boundary 是否一致？
3. Vision encoder unfreezing 的 2-point gain 是否超過 trial uncertainty？
4. 單一 button task 能否代表 general VLA deployment？
5. 官方 checkpoint 的 architecture 與 paper 描述是否一致？

## Meeting Card

這是一個 8 GB GPU feasibility case，不是能證明某個 quantization algorithm 優越的 comparative study。
