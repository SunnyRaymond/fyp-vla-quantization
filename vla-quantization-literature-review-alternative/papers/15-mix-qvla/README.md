# 15 — Mix-QVLA: Task-Evidence-Aware Mixed-Precision Quantization of VLA Models

- 定位：用 internal task-evidence preservation 與 temporal sensitivity 做 layer-wise mixed-precision PTQ。
- Source：[arXiv 2606.19565](https://arxiv.org/abs/2606.19565)
- Local PDF：[paper-arxiv-v1.pdf](paper-arxiv-v1.pdf)

## 為何納入

Mix-QVLA 直接量化 OpenVLA / OpenVLA-OFT，並將 action endpoint 之外的 vision、projection、reasoning、pre-action evidence 納入 bit allocation。

## Background / Problem

兩個 models 可能輸出相近 action，卻依賴完全不同的 internal evidence；quantized policy 若已破壞 visual grounding，短期 action 相近也可能在後續 state 崩潰。只看 final action deviation 無法定位哪個 functional boundary 失真。

## Method 與 Key Innovation

- 固定 full-precision action-token sequence 為 reference decision。
- 在 vision encoder output、projector output、language-policy representation、pre-action representation 四個 boundaries 建立 normalized gradient-weighted evidence maps。
- 用 evidence mass 與 attribution distribution distortion 比較 FP / quantized variants，再以 soft-bottleneck 聚合。
- 將 trajectory progress 分 bins，加入 worst-phase temporal sensitivity；最後以 binary optimization 在 model-size / BitOps constraints 下從 `{2,4,8,16}` 選 layer precision。

## Main Results

- OpenVLA W4A4-budget：76.3% vs BF16 76.5%；4.0 vs 15.2 GB；約 1.52×。
- OpenVLA-OFT W4A4-budget：96.3% vs BF16 97.1%；4.1 vs 15.4 GB；約 1.52×。
- Weight-only OpenVLA W4A16：76.6%、4.1 GB；OpenVLA-OFT 96.9%、4.2 GB。
- Ablation：task evidence + temporal evidence 76.3%，各自單獨約 75.9% / 75.6%。

## Limitations / Evidence Boundary

只有 OpenVLA-style policies 與 LIBERO simulation，沒有 real robot 或其他 action-decoding family。Calibration 需要額外 forward/backward passes；evidence map 是 diagnostic correlation，不是 causal guarantee。Allocation 在 calibration 後固定，並非 runtime dynamic precision。

## Why It Matters

它把 sensitivity 從 output error 擴展到 decision-supporting evidence，適合研究「量化後 action 看似正常，但 grounding 已悄悄改變」的 hidden failure。

## Reading Route

- 20 minutes：Figure 1、four boundaries、Tables 1–3。
- 60 minutes：teacher-forced objective、evidence normalization、soft-bottleneck、temporal bins 與 optimization constraints。

## Reading Questions（不附答案）

1. Evidence maps 是否對 reference action token 的選擇敏感？
2. Worst temporal bin 會否過度保護少數 layers？
3. 0.3-point ablation gain 是否超過 rollout variance？
4. Calibration backward-pass cost 有多大？
5. Evidence distortion 能否預測 real-robot failure type？

## Meeting Card

Mix-QVLA 問的不是「action 差多少」，而是「支持 full-precision action 的 internal evidence 是否仍存在」。
