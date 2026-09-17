# 13 — ActQuant: Sub-4-bit Action-Guided Quantization for Vision-Language-Action Models

- 定位：hardware-friendly、action-guided、sub-4-bit weight-only PTQ，加上 native C/C++ runtime。
- Source：[arXiv 2605.24011](https://arxiv.org/abs/2605.24011) · [official code](https://github.com/arashakb/ActQuant)
- Local PDF：[paper-arxiv-v3.pdf](paper-arxiv-v3.pdf)

## 為何納入

ActQuant 專門處理 OpenVLA-OFT 與 Pi-0.5 的 2–4 bits-per-weight quantization，並把 algorithm、low-bit kernels、LIBERO、UR3 與三種 hardware 放在同一 study。

## Background / Problem

Generic AWQ/GPTQ 與 QVLA 在 sub-4-bit 常快速 collapse。過細 channel-wise mixed precision 又可能造成 irregular memory access；真正需要的是同時足夠細緻、又能 dispatch 到 dense integer kernel 的 granularity。

## Method 與 Key Innovation

- `Inter-Tensor Bit Allocation`：用 Hilbert-Schmidt Independence Criterion (HSIC) 衡量 matrix output 與 action label 的 dependence，為每個 weight matrix 選一個 bit-width。
- `Intra-Tensor Scale Optimization`：以 `Action-Mixed Fisher` 混合 action-head 與 language-head signal，優化 block scales / zero points。
- Quantize vision encoder 與 language backbone；projector / action head 保持 full precision。
- `OmniModel.cpp` 將 PyTorch VLA 轉成 native C/C++ / GGML graph，直接使用 low-bit kernels。

## Main Results

- OpenVLA-OFT：FP16 96.9%、14.3 GB；ActQuant 3.0 bpw 95.0%、3.2 GB；2.5 bpw 90.1%、2.7 GB（約 5.3× compression）。
- Pi-0.5：3.0 bpw 94.8% vs FP16 97.0%；2.5 bpw 85.7%。
- UR3，4 tasks、每 task 10 trials：Pi-0.5 FP16 77.5%、6.7 GB；ActQuant 3.0 bpw 75.0%、2.7 GB。
- OpenVLA Q4_K_M native runtime：A6000 233→153 ms、M4 Pro 1319→999 ms、AGX Thor 598→465 ms。

## Limitations / Evidence Boundary

主要是 weight-only；activation compute 與 action head memory 保持較高 precision。Quantizer accuracy 與 OmniModel.cpp runtime gain 是兩個 components，不能把 native-runtime speedup 全歸因於 ActQuant score。Real-robot sample 每 task 10 trials。

## Why It Matters

它同時處理 algorithm granularity 與 executable kernel granularity，並把研究推入真正 sub-4-bit regime，是很適合做 implementation-oriented follow-up 的 paper。

## Reading Route

- 20 minutes：Figure 2、Table 1–3、method overview。
- 60 minutes：HSIC score、Action-Mixed Fisher、GGML block formats、conversion verification 與 hardware comparison。

## Reading Questions（不附答案）

1. HSIC 與 final action deviation 各自捕捉什麼 sensitivity？
2. Per-tensor bit-width 為何比 per-channel 更 hardware-friendly？
3. Action-Mixed Fisher 中 language loss 的權重如何選？
4. Model memory 是否包含 quantization metadata？
5. 如何拆分 ActQuant algorithm 與 OmniModel.cpp 的 latency contribution？

## Meeting Card

ActQuant 的貢獻不只是在 2.5–3 bpw 保留 action，而是把 precision granularity 限制在現有 dense low-bit kernels 真能執行的範圍。

