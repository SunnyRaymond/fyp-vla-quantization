# 19 — Bit-Flip Attacks on Vision-Language-Action Models: Action-Decoding Architecture Shapes the Vulnerability

- 定位：quantized VLA stored-weight integrity / security study。
- Source：[arXiv 2608.15475](https://arxiv.org/abs/2608.15475)
- Local PDF：[paper-arxiv-v1.pdf](paper-arxiv-v1.pdf)

## 為何納入

Paper 的 attack surface 就是 deployed INT8 / INT4 VLA weights，並在 OpenVLA、OpenVLA-OFT、Pi-0、Pi-0.5 上比較不同 action decoders 的 closed-loop vulnerability。

## Background / Problem

Quantization 把 weights 存成低 precision integers，形成可被 logical bit flip / Rowhammer-style fault 影響的 surface。Random corruption 看似無害，但 gradient-selected bits 可能集中在少數 action-generating layers，並透過 closed loop 放大。

## Method 與 Key Innovation

- 對 per-output-channel symmetric INT8 weights 的 exact bit changes 計算 quantization-aware gradient gain。
- Direct regression / token heads 使用 action-aligned loss；flow-matching heads 使用固定方向 `manifold-escape` objective，讓多步 decoder 的 action displacement 保持一致方向。
- Candidate surface 包含所有 action-generating transformer linears，而不是只預先指定 action head。
- Defense 以 adaptive re-attack 評估 selective integrity protection。

## Main Results

- Discrete OpenVLA LIBERO-Spatial：3 selected flips 令 88% clean success 變 0/50；300 random flips 仍約近 clean。
- Direct regression OpenVLA-OFT 可在 1 flip collapse；evaluated Pi-0 / Pi-0.5 flow heads 約需 100–300 selected flips。
- Real 6-DoF Pi-0.5 task：K=100 emulated INT8-equivalent patches 0/20；clean 14/20；equal-count global-random 16/20。
- Protect OpenVLA early layer L1 的 3.1% weights 後，K=100 仍有 60% success；action head + L1 約 5.3% protection 將 open-loop break threshold 由 3 提到 100 flips。

## Limitations / Evidence Boundary

這是 logical fault susceptibility，不是完整 physical Rowhammer delivery demonstration；actual reachability 依 device、memory placement、ECC 與 fault profile。Selective protection 對 direct head 有效，但不直接轉移到 flow heads。Security result 不代表一般 quantization accuracy 較差。

## Why It Matters

它把 VLA quantization evaluation 從 average success 擴展到 weight integrity。模型即使 clean accuracy 幾乎不掉，也可能存在高度 concentrated failure surface。

## Reading Route

- 20 minutes：threat model、Tables 1–8、limitations。
- 60 minutes：推導 bit gain、比較 decoder objectives、核對 confidence intervals、real-robot patch boundary 與 adaptive defense。

## Reading Questions（不附答案）

1. 為何 direct/token heads 的 collapse budget 比 flow heads 小？
2. Logical INT8 flips 與真實 hardware Rowhammer 的距離有多大？
3. 3.1% protected weights 應用 ECC、checksum 還是 duplication？
4. Quantization group size 如何改變 bit-flip magnitude？
5. Calibration frames 改變時 vulnerable layers 是否穩定？

## Meeting Card

Quantized VLA 的 deployment boundary不只 accuracy / speed，還包括 stored-weight integrity；少數 selected bits 可形成 closed-loop catastrophic failure。

