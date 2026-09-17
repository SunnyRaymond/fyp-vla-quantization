# 05 — SQAP-VLA: A Synergistic Quantization-Aware Pruning Framework for High-Performance VLA Models

- 定位：training-free W4A4 quantization 與 visual token pruning co-design。
- Source：[arXiv 2509.09090](https://arxiv.org/abs/2509.09090)
- Local PDF：[paper-arxiv-v1.pdf](paper-arxiv-v1.pdf)

## 為何納入

SQAP-VLA 直接對 CogACT VLA 同時進行 weight-activation quantization 與 token pruning，並量測 closed-loop success、memory 與 latency。

## Background / Problem

Quantization 會改變 token importance；token pruning 也會改變 activation distribution。若兩者獨立套用，原本在 full precision 上選出的 salient tokens 未必仍適合 low-bit model。

## Method 與 Key Innovation

- 保留對 quantization perturbation 不敏感的 top-k tokens。
- 以 3D-to-2D end-effector projection 建立 robot-aware ring，保留操作區域。
- 使用 Farthest Point Sampling 維持 spatial coverage。
- 對 Q/K activations 用 Hadamard transform 處理 outliers，再配合 channel-wise quantization。
- 目標設定為 CogACT W4A4 加約 40% token pruning。

## Main Results

- SIMPLER Visual Matching average：FP16 74.8%，SQAP-VLA 79.3%；Variant Aggregation：61.3→64.4%。
- Paper 報告約 1.93× overall speedup；peak memory 14.3→7.6 GB。
- LLM backbone acceleration 約 2.56×；W4A4 與 pruning 的 isolated gains 約 2.09× 與 1.21×。

## Limitations / Evidence Boundary

只在 CogACT 與四個 SIMPLER tasks 上評估；quantization 與 pruning 同時改動，headline gain 不能全部歸因於 quantization。Quantized success 高於 FP16 缺少足夠 seeds / confidence intervals 時，應視為可能的 rollout variance，而不是穩定 accuracy improvement。

## Why It Matters

它指出 VLA efficiency methods 會互相干擾，因而需要 joint design。對實際 deployment，這比孤立地比較 bit-width 更接近真實系統。

## Reading Route

- 20 minutes：pipeline figure、token criteria、main result table。
- 60 minutes：核對各 component ablation、robot-aware ring 的假設與 latency measurement boundary。

## Reading Questions（不附答案）

1. Quantization-insensitive token 是否等同 task-critical token？
2. 3D-to-2D projection 誤差會如何影響 pruning？
3. 為何 Q/K outliers 對 token pruning 特別重要？
4. 1.93× 中 quantization 與 pruning 各自貢獻多少？
5. 若換成 multi-camera Pi-0.5，robot-aware ring 還成立嗎？

## Meeting Card

SQAP-VLA 的重點是 quantization 與 token pruning 不能獨立最佳化；它們共同改變 token saliency 與 runtime bottleneck。

