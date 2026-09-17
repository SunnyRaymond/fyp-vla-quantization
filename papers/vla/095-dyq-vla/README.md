# 11 — DyQ-VLA: Temporal-Dynamic-Aware Quantization for Embodied Vision-Language-Action Models

- 定位：依 robot execution phase 在 runtime 切換 activation precision 的 dynamic PTQ framework。
- Source：[arXiv 2603.07904](https://arxiv.org/abs/2603.07904)
- Local PDF：[paper-arxiv-v2.pdf](paper-arxiv-v2.pdf)

## 為何納入

DyQ-VLA 直接量化 OpenVLA weights / activations，並以 LIBERO、physical robot、end-to-end latency 與 peak memory 評估 temporal dynamic precision。

## Background / Problem

Static quantization 必須按 task 中最敏感的 moment 選 bit-width，因此在 free-space coarse motion 期間浪費 precision。真正困難是 runtime 如何用低成本 signal 判斷何時需要 BF16 fallback。

## Method 與 Key Innovation

- Weights 固定為 INT4，避免 runtime weight swapping；activations 在 `{INT2, INT4, INT8, BF16}` 間切換。
- `Motion Fineness` 捕捉 macro trend，`Angular Jerk` 捕捉短暫 sensitivity spike；paper 報告它們與 post-hoc sensitivity 的 correlation 約 0.90 / 0.87。
- Hysteresis-based switching 避免 sensor noise 造成頻繁抖動。
- Custom CUTLASS kernels、fused activation quantization、zero-copy flag 與 asynchronous CPU-GPU dispatch 將 scheduling overhead 隱藏在 vision prefill。

## Main Results

- LIBERO 40 tasks、每 suite 200 trials，A100：BF16 average 76.5%，DyQ-VLA 76.1%；peak memory 15.2→4.7 GB，約 1.49× speedup。
- 與 QVLA 4.3 GB / 76.0% 相比，DyQ-VLA 是相近 Pareto point，而非壓倒性 accuracy 優勢。
- Physical robot：Atomic 86.7→86.7%、Spatial 76.7→73.3%、Composite 70.0→66.7%；speedup 約 1.32×–1.43×。
- Dynamic system overhead 報告小於 0.5 ms 並由 asynchronous execution 隱藏，state 約小於 0.1 MB。

## Limitations / Evidence Boundary

Kinematic proxies 假設小 motion / 高 jerk 與 precision need 對齊；contact-rich 或 unfamiliar embodiment 可能破壞這個關係。A100 結果不能直接代表 edge hardware。Performance 對 threshold 與 window size 敏感，paper 顯示過度追求 speed 可令 complex-task success 降約 15.4 points。

## Why It Matters

它把 mixed precision 從固定 layer/channel map 推到 time-varying control state，是 VLA quantization 與 online scheduling 真正交會的 paper。

## Reading Route

- 20 minutes：Figure 2–4、W4AX design、Tables I–IV。
- 60 minutes：檢查 sensitivity ground truth、proxy correlation、hysteresis、kernel dispatch 與 real-robot protocol。

## Reading Questions（不附答案）

1. Motion Fineness / Angular Jerk 是否使用上一步 predicted action，會否形成 feedback bias？
2. BF16 fallback 的 worst-case latency 是多少？
3. Dynamic bit switching 對 control jitter 有何影響？
4. 4.7 GB 是否包含所有 runtime buffers？
5. 如何在 contact-rich task 上獨立驗證 proxy validity？

## Meeting Card

DyQ-VLA 的新意是「precision 隨 execution phase 變」，但收益成立的前提是 kinematic proxy 與 custom runtime 都可靠。

