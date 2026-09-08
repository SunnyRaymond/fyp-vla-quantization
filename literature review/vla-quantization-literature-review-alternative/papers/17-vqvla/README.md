# 17 — VQVLA: A Motion-Aware Vector Quantization Framework with Centroid Reuse for Efficient VLA Inference

- 定位：MotionVQ algorithm + merged-centroid GEMM + custom accelerator 的 algorithm-hardware co-design。
- Source：[arXiv 2607.24148](https://arxiv.org/abs/2607.24148)
- Local PDF：[paper-arxiv-v1.pdf](paper-arxiv-v1.pdf)

## 為何納入

VQVLA 直接對 OpenVLA、OpenVLA-OFT、RDT、Pi-0、GR00T 的 transformer weights 做 Vector Quantization，並以 task motion state 動態選 precision，再設計專用硬體。

## Background / Problem

傳統 weight quantization 主要減少 memory traffic；Vector Quantization 以 codebook centroids 表示 weight vectors，還能利用 repeated centroid 避免 multiplication。但 static codebook precision 沒有利用 robot 在 coarse transition 與 fine execution states 的不同 noise tolerance。

## Method 與 Key Innovation

- `MotionVQ`：用前一步 3D action magnitude 區分 execution / transition state。
- Offline 建兩組 codebooks：high precision `VQ[256,2,256]` 約 4.125 bits；low precision `VQ[128,2,64]` 約 3.125 bits，runtime 依 state 切換。
- `Merged-Centroid Vectorized GEMM`：同 column 聚合 shared centroid 的 inputs 做 spatial reuse；跨 columns cache hot-centroid products 做 temporal reuse。
- Custom accelerator 直接處理 codebook/index representation 與 state predictor。

## Main Results

- Five VLA workloads：平均 success rate 下降約 2.5 points；retrieving weights 的 memory traffic 平均降 79.4%，multiplications 平均降 54.8%。
- Cycle-level simulation + 28 nm synthesis、再 scale 到 7 nm：報告相對 A100 約 6.5×、相對 Dadu-Corki 約 2.8×；per-action latency 約 30–60 ms。
- Paper 報告相對 A100 約 75.5× energy efficiency，custom design power 約為 A100 的 6.4%。

## Limitations / Evidence Boundary

Headline latency / energy 主要來自 cycle simulator、Verilog synthesis 與 process-node scaling，不是 fabricated chip 或 commodity edge device measurement。Paper PDF 雖標 MICRO 2026，但 ISBN/DOI 是 placeholder；本 corpus 按 arXiv preprint 處理。Motion magnitude 作 sensitivity proxy 在 contact-rich 或 stationary-but-uncertain states 可能失效。

## Why It Matters

它展示 Vector Quantization 只有配合 centroid-aware compute 才能同時省 memory 與 multiplications；也把 robot state 直接放入 precision schedule。

## Reading Route

- 20 minutes：MotionVQ、Figure 14–18、evaluation methodology。
- 60 minutes：codebook layout、spatial/temporal reuse、cycle simulator assumptions、area/power scaling。

## Reading Questions（不附答案）

1. 前一步 action magnitude 是否足以預測下一步 sensitivity？
2. 兩套 codebooks 的 storage 是否計入 memory saving？
3. GPU-VQVLA 為何比原 A100 execution 慢 45.2%？
4. Cycle simulator 如何校準到真實 end-to-end VLA？
5. Centroid reuse 能否在現有 GPU kernels 上部分實現？

## Meeting Card

VQVLA 的核心是「quantized representation 要改變 GEMM dataflow」；但最亮眼的 speed/energy 數字仍是 simulated custom hardware evidence。
