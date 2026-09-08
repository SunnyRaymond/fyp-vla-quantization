# 20 — SpecVLA: Algorithm-Architecture Co-Design for Efficient VLA Inference via Speculative Inference and Verification

- 定位：以 quantized small VLA verifier 支援 long-action speculation 的 heterogeneous co-design。
- Source：[arXiv 2608.15636](https://arxiv.org/abs/2608.15636)
- Local PDF：[paper-arxiv-v1.pdf](paper-arxiv-v1.pdf)

## 為何納入

SpecVLA 直接從 OpenVLA / RDT 建構 block-wise mixed-precision quantized `sVLA`，讓它在 active states 驗證 full VLA 的 speculative actions；quantization 是整個 verification path 的必要元件。

## Background / Problem

Action chunking 能 amortize expensive VLA calls，但 long chunk 在接近 object、contact 或 placement 時容易因 stale observation 失敗。作者將 robot execution 分 active / inactive states：inactive 可長 speculation，active 則逐 action 驗證。

## Method 與 Key Innovation

- Full VLA 在 GPU 預測長 action sequence；quantized `sVLA` 在 robotic-specific hardware 做短步 verification。
- 以連續 visual features 的 residual `ΔX` 建 sVLA；block absolute sum 低於 `Tz` 設 0-bit，中間用 4-bit，高於 `Th` 用 8-bit。
- Verification 比較 full VLA predicted action 與 sVLA action 的 weighted normalized L1 distance，超過 threshold 便 rollback。
- Speculative dataflow 讓 VLA prediction、sVLA verification 與 robot execution overlap。

## Main Results

- sVLA residual blocks 平均約 35.8% zero、59.3% 4-bit、4.9% 8-bit。
- OpenVLA LIBERO quantization comparison：custom W8A3-style setting報告 98/97/97/93 success，約 10.22% FLOPs、8.78× vs FP16 GPU；這依賴 proposed hardware。
- Full SpecVLA 在 OpenVLA / RDT tasks 的 average accepted action length 約 7.1、rollback 約 9.5%，path length 約在 baseline ±2%。
- Cycle-simulated heterogeneous system：平均約 2.9× vs A100、1.9× vs Dadu-Corki-ADAP，約 61 ms/action。

## Limitations / Evidence Boundary

Full VLA 本身保持 full precision；量化的是 verifier path，所以不能和 whole-model PTQ 的 memory saving 直接比較。Headline speedup 由 long action speculation、parallel dataflow 與 custom accelerator共同產生。Hardware results 來自 cycle simulator + 28 nm synthesis，不是 fabricated chip。PDF 的 MICRO 2026 ISBN/DOI 是 placeholder，故按 arXiv preprint 處理；沒有 physical-robot trial。

## Why It Matters

它提出不同的用法：quantized VLA 不一定取代 full model，也可以成為高頻 verifier。這把 compression 變成 reliability-aware scheduling 元件。

## Reading Route

- 20 minutes：Figure 1、sVLA construction、Tables 1–4。
- 60 minutes：active-state detector、verification threshold、rollback semantics、dataflow 與 cycle-simulator assumptions。

## Reading Questions（不附答案）

1. sVLA 與 full VLA 同時錯誤時 verification 是否會 false accept？
2. 0/4/8-bit residual thresholds 如何跨 tasks 泛化？
3. Quantized verifier 的 memory 是否納入 full system footprint？
4. Speculation / quantization / hardware 各自的 isolated gain是多少？
5. 61 ms/action 在 contact-rich real robot 上是否仍成立？

## Meeting Card

SpecVLA 把 quantized model當作 high-frequency verifier，而非 full-model replacement；但最強 speedup 仍建立在 simulated custom hardware 上。
