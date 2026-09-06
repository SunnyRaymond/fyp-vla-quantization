# 14 — HoloQ-VLA: Uniform W4A4 Quantization of Vision-Language-Action Models

- 定位：training-free、uniform full-stack W4A4，包含整個 DiT action head attention。
- Source：[arXiv 2605.28803](https://arxiv.org/abs/2605.28803)
- Local PDF：[paper-arxiv-v3.pdf](paper-arxiv-v3.pdf)

## 為何納入

HoloQ-VLA 的 central claim 就是把 Pi-0.5 與 GR00T N1.5 的 language backbone + entire DiT action head 全部量化到 W4A4，並在 simulation 與 bimanual robot 上測試。

## Background / Problem

在 uniform W4A4 下，weight outliers 與 activation outliers 的最佳 rotation 方向可能互相衝突；DiT activations 又隨 denoising step 改變 range。先前方法通常保留 DiT attention 或整個 action head 的較高 precision。

## Method 與 Key Innovation

- `SVD · Hadamard composite rotation`：SVD factor 對齊 weight geometry，Hadamard factor 再分散 residual activation outliers。
- Zigzag channel permutation 與 block-wise rotation，限制 rotation metadata / compute。
- DiT 使用 per-step activation scale table，處理 8 個 Euler denoising steps 的 range drift。
- LLM side 採 GPTQ-style weight quantization；DiT side採 RTN；只用 10 trajectories calibration。

## Main Results

- LIBERO：GR00T N1.5 W4A4 full-stack 87.8% vs FP16 86.5%；Pi-0.5 98.0% vs FP16 97.1%。
- Pi-0.5 static footprint（含 metadata）：5.41→1.39 GB，saving 74.2%。
- Real bimanual ARX R5，5 tasks、每 task 10 rollouts：HoloQ-VLA average progress score 51.0、FP16 49.6、QuantVLA W4A8 25.0。
- Ablation 顯示 composite rotation 是主要 gain；per-step scaling 對 long-horizon suite 再有較大幫助。

## Limitations / Evidence Boundary

Real-world metric 是 stage-wise progress score，不是 binary task success。Quantized score 小幅高於 FP 應視為 sampling variance / policy stochasticity 的可能結果，尚需 independent reproduction。Paper 沒有提供 end-to-end low-bit latency；uniform precision 是否更快仍取決於 kernels 與 rotation folding。

## Why It Matters

它直接挑戰「DiT attention 必須保留 high precision」的保守 layout，提供 full-stack W4A4 的 concrete mechanism，也把 weight / activation outlier conflict 講得很清楚。

## Reading Route

- 20 minutes：Table 1–3、composite rotation、per-step scaling。
- 60 minutes：推導 transform folding、比較 QuantVLA layout、核對 metadata 與 real-world scoring rubric。

## Reading Questions（不附答案）

1. SVD 與 Hadamard 的順序為何重要？
2. Per-step scale table 是否對 different sampler / step count 可轉移？
3. 1.39 GB 是 static footprint 還是 peak VRAM？
4. Progress score 為何可能高於 FP16？
5. 沒有 low-bit kernel benchmark 時，uniform W4A4 的 runtime claim能到哪一步？

## Meeting Card

HoloQ-VLA 是目前最激進的 full-stack W4A4 route：用兩種 rotation 分別處理 weight / activation outliers，再用 per-step scales 修正 DiT drift。
