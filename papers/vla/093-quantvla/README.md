# 09 — QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models

- 定位：面向 flow-matching / DiT action head 的 selective W4A8 PTQ。
- Source：[CVPR 2026 final](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_QuantVLA_Scale-Calibrated_Post-Training_Quantization_for_Vision-Language-Action_Models_CVPR_2026_paper.html)
- Local PDF：[paper-cvpr-final.pdf](paper-cvpr-final.pdf)

## 為何納入

QuantVLA 是專門 VLA PTQ paper，直接量化 Pi-0.5 與 GR00T N1.5 的 language backbone 和 DiT action module，並提供 module-level memory 與 closed-loop LIBERO results。

## Background / Problem

DiT action head 的 attention logits 與 output magnitude 對 quantization scale 很敏感；若整個 LLM + DiT 直接 W4A8，success 可能大幅下降。問題不只是 outlier，而是 perception-to-action interface 和 denoising dynamics 的 scale mismatch。

## Method 與 Key Innovation

- `Selective Quantization Layout`：integerize LLM linear layers 與 DiT MLP；DiT attention Q/K/V/O 保留 floating point。
- `Attention Temperature Matching (ATM)`：用 per-head scalar 對齊 quantized 與 full-precision attention-logit standard deviation。
- `Output Head Balancing (OHB)`：用 per-layer scalar 對齊 output RMS。
- Scalars 可 fold 回 dequantization scales，不增加新的 inference operators。

## Main Results

- Pi-0.5：FP 97.1%、LLM+DiT memory 4.27 GB；QuantVLA W4A8 97.6%、1.28 GB，約 70% relative saving。
- GR00T N1.5：FP 86.5%、2.02 GB；QuantVLA 88.0%、0.91 GB，約 55% saving。
- Pi-0.5 W4A4 average 約 95.3%。
- 將所有 DiT/LLM linears 直接量化的 baseline 可跌到約 76.3%，凸顯 action-head attention sensitivity。

## Limitations / Evidence Boundary

Final paper 的主 evidence 是 simulation；沒有 main real-robot study。Memory 數字針對 LLM+DiT target modules，不等於 total process peak VRAM。Quantized model 偶爾高於 FP 的小差距不應在沒有 matched uncertainty 分析時視為 intrinsic improvement。

## Why It Matters

QuantVLA 提供一個 pragmatic architecture-aware answer：先保護 DiT attention，再用 scale matching 取得大部分 W4A8 benefit。它是理解 HoloQ-VLA 為何要進一步處理 full-stack W4A4 的直接前置閱讀。

## Reading Route

- 20 minutes：Figure 1、三個 components、main LIBERO table。
- 60 minutes：追蹤 ATM / OHB 如何 fold、哪些 operators 維持 FP、memory scope 與 calibration size。

## Reading Questions（不附答案）

1. ATM 與 OHB 校正的是 calibration-time statistics 還是 inference-time dynamic values？
2. 保留 DiT attention floating point 佔多少 latency / memory？
3. Module footprint 與 peak VRAM 差多少？
4. W4A4 失效主要來自 weights 還是 activations？
5. QuantVLA 能否與 action-aware bit allocation 結合？

## Meeting Card

QuantVLA 用 selective layout + scale matching 避免 DiT action head collapse，是目前最清楚的 W4A8 VLA PTQ baseline 之一。

