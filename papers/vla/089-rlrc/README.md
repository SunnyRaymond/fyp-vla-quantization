# 04 — RLRC: Reinforcement Learning-based Recovery for Compressed Vision-Language-Action Models

- 定位：structured pruning 後以 SFT + RL 恢復，再選擇性加入 NF4 quantization。
- Source：[arXiv 2506.17639](https://arxiv.org/abs/2506.17639)
- Local PDF：[paper-arxiv-v2.pdf](paper-arxiv-v2.pdf)

## 為何納入

RLRC 的完整 pipeline 明確包含 VLA compression 與 optional 4-bit NF4，並用 latency/memory ablation 顯示量化不一定帶來 speedup。

## Background / Problem

Structured pruning 能減少真正執行的 layers/parameters，但會破壞 action policy。只用 behavior cloning recovery 可能無法恢復 closed-loop reward，因此作者加入 PPO 與 critic warm-up。

## Method 與 Key Innovation

1. 對 VLA language backbone 做 structured pruning。
2. 以 SFT 恢復基本 imitation behavior。
3. 以 PPO + behavior-cloning regularization 做 closed-loop recovery。
4. 在已縮小模型上測試 bitsandbytes NF4，分析額外 memory saving 與 dequantization cost。

## Main Results

- OpenVLA-OFT 由約 7B 壓到 2B，latency 149.23→65.59 ms；LIBERO average 97.1→97.6。
- 在 recovery model 上再加 NF4：memory 4.02→1.86 GB，但 latency 65.59→86.58 ms，且 Spatial / Long 約由 97.8 / 94.8 降到 95.2 / 91.4。
- Paper 報告 RLRC 約 320 GPU-hours，明顯高於某些小模型 training baseline。

## Limitations / Evidence Boundary

Quantization 不是核心 recovery mechanism，而是 optional last stage。RL recovery 需要 simulation throughput；對 Pi-0 / Pi-0.5 這種 language 與 action expert 緊密耦合的 architecture，pruning 後可能出現難以恢復的 degradation。Real-world transfer 仍不足。

## Why It Matters

這是 corpus 中最直接的「更小不一定更快」證據：低 bit weight 若沒有適合的 kernels，dequantization overhead 可以抵消 memory saving。任何 FYP 都應同時測 wall-clock latency。

## Reading Route

- 20 minutes：pipeline、main table、NF4 ablation。
- 60 minutes：讀 pruning criterion、critic warm-up、reward 與不同 VLA architectures 的 recovery 差異。

## Reading Questions（不附答案）

1. RL recovery 修復的是 representation 還是 closed-loop state distribution？
2. NF4 latency regression 主要來自 kernel、batch size 還是 hardware？
3. 320 GPU-hours 是否值得換取 2B model？
4. 對 action expert pruning 失敗是否可由 sensitivity profiling 預測？
5. 如何把 pruning 與 VLA-aware PTQ 做 factorial ablation？

## Meeting Card

RLRC 說明 compression pipeline 的主角可能是 pruning + RL；NF4 可再省 memory，卻可能同時降低 success 並增加 latency。

