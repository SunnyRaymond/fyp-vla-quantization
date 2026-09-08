# VLA Quantization Literature Review

Snapshot：2026-09-03。本文是 strict-scope narrative literature review，不是宣稱窮盡所有出版物的 formal systematic review。

## Research question 與 scope

核心問題是：**如何在降低 VLA 的 weight memory、activation memory 與 inference cost 時，保留 closed-loop control fidelity，並把 algorithmic compression 轉成真實 hardware latency、control frequency、energy 與安全收益？**

Inclusion criteria：paper 必須明確以 VLA 為研究對象，且 quantization 必須實際作用於 model weights、activations、residual computation 或 deployed integer representation；需要至少提供方法、ablation、deployment measurement 或 security analysis 之一。Supporting system papers 只有在直接部署並評估 quantized VLA 時才保留。

Exclusion criteria：只做 action tokenization、Vector Quantization latent/action codebook、離散 action bins，而沒有 model quantization；只做 generic LLM/VLM quantization 而沒有 VLA experiment；只在 related work 提及 quantization；撤回或結果明顯未完成的 manuscript。

## Executive synthesis

1. **VLA quantization 不是把 LLM quantizer 原封不動套上去。** LLM 常用的 layer-wise reconstruction error 沒有建模 action error 經 environment feedback、action chunk 與 task horizon 累積的方式。`SQIL`、`QVLA`、`HB-VLA`、`DyQ-VLA`、`DA-PTQ`、`ActQuant` 與 `Mix-QVLA` 都在不同層次把 sensitivity 重新連到 action、trajectory 或 task evidence。
2. **最敏感的 module 隨 action-decoding architecture 改變。** Discrete-token OpenVLA、direct regression OpenVLA-OFT、flow-matching Pi-0.5 與 DiT-based GR00T N1.5 不應共享一個盲目的 quantization layout。`QuantVLA` 保留 DiT attention 為 floating point；`HoloQ-VLA` 則以 composite rotation 與 per-step activation scaling 進一步嘗試 uniform full-stack W4A4。
3. **Memory reduction 不保證 latency reduction。** `RLRC` 的 optional NF4 把 memory 再降，但 dequantization overhead 令 latency 反而惡化；`ActQuant`、`DyQ-VLA`、`VQVLA` 與 `SpecVLA` 都需要 runtime kernels 或 hardware mapping 才把 low-bit format 變成 speedup。
4. **目前最清楚的兩條 method 路線是 training-heavy recovery 與 training-free PTQ。** 前者包括 `SQIL`、`BitVLA`、`RLRC`，通常有較高 recovery ceiling 但需要 QAT、distillation 或 RL；後者包括 `QVLA`、`HB-VLA`、`QuantVLA`、`DyQ-VLA`、`DA-PTQ`、`ActQuant`、`HoloQ-VLA`、`Mix-QVLA`，部署成本較低但高度依賴 calibration coverage。
5. **真正的 deployment problem 是 model、runtime、control loop 與 hardware 的 co-design。** `SQAP-VLA`、`LiteVLA-Edge`、`Embodied.cpp`、`VQVLA`、`FlashDrive`、`SpecVLA` 顯示 quantization 只是一個 lever；vision reuse、token pruning、kernel fusion、speculative decoding、action chunking 與 heterogeneous execution 可能佔更大收益。
6. **Evidence 仍然不成熟。** 20 篇中只有 4 篇已有可核對的 peer-reviewed final，14 篇是 2026 work。多數 comparison 使用不同 checkpoints、hardware、episode counts 與 action interfaces；有些 speedup 來自 simulator 或 custom accelerator model，而不是可購買 hardware 的 end-to-end measurement。

## Search and screening record

### Discovery passes

- Multi-index pass：使用 `arXiv`、`DBLP`、`OpenAlex`、`OpenReview`、`Semantic Scholar`、`Crossref`，queries 包含 `vision language action quantization`、`VLA quantization robot`、`low-bit vision-language-action`、`post-training quantization VLA`、`quantized vision language action`，year range 2024–2026。工具回傳 207 unique records，並標示 107 cross-source duplicates 已合併。
- Focused arXiv pass：`all:"vision-language-action" AND all:quantization`，回傳 48 records；再以 title、abstract、paper body 與 source status 做 semantic screening。
- `OpenReview` client 在本機未安裝；`Semantic Scholar` 在查詢時回傳 HTTP 429；`OpenAlex` 曾回傳 HTTP 504 後重試。這些失敗已保留在 limitation，而不是把 missing results 當作 zero evidence。
- 最終納入 20 篇。兩個 discovery passes 高度重疊，沒有可靠的 record-level union，因此不虛構單一 PRISMA total。

### 具代表性的排除項目

| Record | 排除理由 |
|---|---|
| EaqVLA | arXiv record 標示作者要求撤回，abstract 仍含未完成的 placeholder result。 |
| VQ-VLA、LAPA、SALT / Lost in Reconstruction、NAC、RotVLA、QuoVLA、QDepth-VLA | 主要是 action/latent representation quantization 或 action tokenizer，不是 model weight/activation quantization。 |
| CogACT、HybridVLA | 使用 discrete action bins，但 paper 本身不研究 model quantization。 |
| SmoothQuant、AWQ、GPTQ、QuaRot、DuQuant | 可作 quantization baselines，但沒有直接 VLA evaluation，因此不作 corpus paper。 |
| XS-VLA、ResTacVLA、Hermite Tokens | quantized vocabulary、tactile Vector Quantization 或 token representation，不符合 strict model-quantization scope。 |

## Comparison matrix

`Result` 全部是 authors-reported；它們只在同一 row 的 evaluation context 內成立。

| # | Method class | Target / precision | Training | Evaluation / hardware | Representative result | Evidence boundary |
|---:|---|---|---|---|---|---|
| 01 | Baseline study | OpenVLA INT8 / INT4 weight-only | No retraining | 8 Bridge V2 tasks; GPU profiling | INT4：7.0 GB、71.9%; BF16：16.8 GB、71.3% | Small task subset; establishes the non-monotonic precision/runtime problem |
| 02 | Action-saliency QAT | OpenVLA / Pi-0, W4 or W4A4 variants | QAT + distillation | LIBERO, UR5, Jetson AGX Orin | OpenVLA INT4：4.0 GB、374.7 ms vs BF16 15.2 GB、955.2 ms | Strong deployment evidence; training required |
| 03 | Native low-bit architecture | LLM W1.58A8; vision W1.58A8 | Multi-stage training + distillation | LIBERO, real robot, A100 | 1.4 GB、96.0 LIBERO average、73 ms/action chunk | Not drop-in PTQ; substantial training budget |
| 04 | Compression + recovery | Structured pruning + RL; optional NF4 | SFT + PPO | LIBERO / ManiSkill, RTX 5880 Ada | 7B→2B; 149.23→65.59 ms; optional NF4 reduces memory but slows to 86.58 ms | Quantization is optional final stage |
| 05 | PTQ + token pruning | CogACT W4A4 + 40% tokens | Training-free | SIMPLER, RTX 3090 | Peak memory 14.3→7.6 GB; 1.93×; four-task averages improve | Narrow model/task scope; combined gains not quantization-only |
| 06 | Affordable deployment | SmolVLA-like 3.1B, NF4 | LoRA fine-tuning | Single real task, RTX 4060 8 GB | 200 demos：76%; 45 ms; peak VRAM 6.8 GB | Single task, no matched baseline; architecture provenance needs checking |
| 07 | Action-centric PTQ | Per-output-channel mixed weight bits; activation uniform | Training-free calibration | LIBERO, small real-world study, RTX 4090 / RTX 4070 | OpenVLA W4A4 retains 99.3%, 28.2% memory, 1.47× | Canonical ICLR final; local PDF is arXiv fallback |
| 08 | Binary PTQ | 1-bit weights, BF16 activations | Training-free calibration | LIBERO, SIMPLER, Mobile ALOHA, A800 | Pi-0.5 footprint 4.6→0.83 GB; whole-model success 97.1→87.9 | Extreme compression; action head remains fragile |
| 09 | Scale-calibrated PTQ | LLM + DiT MLP W4A8; attention floating point | Training-free calibration | LIBERO, A100 | Pi-0.5：97.1→97.6; target-module memory 4.27→1.28 GB | Published CVPR final; memory is scoped to LLM+DiT modules |
| 10 | GGUF deployment | 256M backbone, Q4_K_M | FP32 LoRA then PTQ | Simulated closed loop, Jetson Orin-class | 150.5 ms, 6.64 Hz, low jitter | No task-success benchmark; hardware naming is internally inconsistent |
| 11 | Temporal dynamic PTQ | Static W4 weights; A2/A4/A8/BF16 switching | Calibration + custom kernels | LIBERO + real robot, A100 | 4.7 GB vs 15.2 GB; 1.49×; 76.1 vs 76.5 average | Threshold and kinematic-proxy dependence |
| 12 | Drift-aware PTQ | CogACT W4A8 / BF16 mixed precision | 512-trajectory calibration | SIMPLER, RTX 5090 | 48.9% vs FP 51.3%; 42.5% memory reduction; reported 54.8% speedup | Preprint uses placeholder ACM template; no physical-robot test |
| 13 | Sub-4-bit PTQ | Vision + LLM mixed 2–4 bpw; head full precision | 60-episode calibration | LIBERO, UR3, A6000 / M4 Pro / AGX Thor | OpenVLA-OFT 2.5 bpw：2.7 GB, 90.1%; native runtime up to 1.5× | Strong low-bit study; runtime and quantizer gains should be separated |
| 14 | Uniform full-stack PTQ | LLM + entire DiT W4A4 | 10-trajectory calibration | LIBERO, bimanual robot, H100 | Pi-0.5：98.0 vs FP16 97.1; static footprint 5.41→1.39 GB | Real-world metric is progress score, not binary success; no latency result |
| 15 | Task-evidence PTQ | Layer-wise mixed {2,4,8,16}; W4A4 budget | Offline forward/backward calibration | LIBERO, A100 | OpenVLA-OFT：96.3 vs 97.1; 4.1 vs 15.4 GB; 1.52× | OpenVLA-family only; evidence maps are diagnostic, not causal |
| 16 | Portable runtime | C++ 8/6/4-bit configurations | Runtime conversion | Pi-0.5, GR00T N1.7, HY-VLA | 1.05×–2.70× and 7%–77% lower VRAM across settings | Results normalized; quantizer details and absolute VLA setup are limited |
| 17 | VQ accelerator | MotionVQ 4.125/3.125 average bits | Offline codebooks + custom hardware | Five VLA models; cycle simulator + synthesis | 2.5-point average success drop; 6.5× vs A100; 30–60 ms/action | Hardware speed/energy mostly simulated and node-scaled |
| 18 | Full-pipeline co-design | Alpamayo 1.5 backbone W4A8; action expert BF16 | Targeted streaming fine-tuning | Driving dataset + AlpaSim; five GPUs | 717→151 ms overall; quantization alone 176→151 ms | Quantization is one component, not the source of the full 4.7× |
| 19 | Quantized-weight security | Per-channel INT8/INT4 stored weights | Attack calibration only | LIBERO, SIMPLER, real robot, A800 | Direct/token heads collapse in 1–5 selected flips; flow heads need ~100–300 | Logical faults; not an end-to-end Rowhammer delivery demonstration |
| 20 | Quantized verifier co-design | sVLA residual blocks 0/4/8-bit | Model construction + custom hardware | OpenVLA / RDT; cycle simulator + synthesis | 2.9× vs A100, ~61 ms/action, comparable success | Full VLA stays full precision; speedup relies on speculative scheduling and simulated NPU |

## 核心 synthesis

### 1. Baseline：precision、accuracy 與 speed 不是單調關係

`OpenVLA` 的 quantization ablation 是這個 literature 的起點：INT8 並沒有自然優於 INT4，甚至可能因 runtime overhead 與 closed-loop distribution shift 同時輸掉 accuracy 和 speed。這說明 bit-width 只是 format；真正結果由 quantized modules、scales、kernel availability、memory bandwidth 與 policy feedback 共同決定。

後續 papers 對這個 failure mode 給出不同解釋。`SQIL` 把問題定位在 mission-critical states；`QVLA` 與 `ActQuant` 認為 layer/channel/tensor 對 action prediction 的貢獻不均；`QuantVLA` 與 `HoloQ-VLA` 聚焦 DiT action head 的 activation range 與 denoising-step drift；`DyQ-VLA`、`DA-PTQ`、`Mix-QVLA` 則直接把 time、trajectory 或 internal task evidence 放進 sensitivity definition。

### 2. Training-heavy methods 與 PTQ 的真正 trade-off

`SQIL`、`BitVLA`、`RLRC` 的共同優點是可用 imitation learning、distillation 或 RL 主動恢復 policy，而不只接受 calibration 的局部 approximation。代價是 data、GPU-hours、implementation complexity 與 checkpoint portability。`BitVLA` 尤其接近「重新設計並訓練一個 low-bit VLA」，不是把現有 7B checkpoint 一鍵壓縮。

PTQ methods 的賣點是 pretrained policy frozen，但「training-free」不等於「data-free」或「cost-free」。`QVLA` 要測 action-space sensitivity；`DA-PTQ` 使用 512 trajectories；`ActQuant` 需要 forward/backward action-aware statistics；`HoloQ-VLA` 雖只用 10 trajectories，仍需 rotation 與 per-step scale calibration；`Mix-QVLA` 額外需要 gradient-weighted evidence maps。部署前應把 calibration time、memory 與 data coverage 一併記錄。

### 3. Action head 是核心，但不能只說「action head 很敏感」

在 flow-matching / DiT VLA 中，action head 的 iterative dynamics 可放大量化誤差。`QuantVLA` 的 pragmatic 策略是只量化 DiT MLP 並保留 Q/K/V/O；`HoloQ-VLA` 的更激進主張是透過 `SVD · Hadamard` composite rotation 和 per-step scaling，把整個 DiT attention 也帶到 W4A4。兩者不是直接矛盾：前者展示 selective layout 的穩健性，後者展示更複雜 transform 下 full-stack uniform precision 的可能性。

`Bit-Flip Attacks` 提供另一個 architecture-sensitive evidence：direct regression / discrete-token heads 在 selected INT8 bit flips 下只需 1–5 flips 即可 collapse，而 evaluated flow heads 約需 100–300。這不是說 flow models 普遍安全，而是說 attack propagation、quantization robustness 與 integrity protection 必須按 decoder architecture 分析。

### 4. Static sensitivity 正在轉向 temporal 與 task-conditional sensitivity

`QVLA` 的 channel-wise action deviation 是第一層 action-aware signal；`DyQ-VLA` 以 Motion Fineness 和 Angular Jerk 決定 runtime activation precision；`DA-PTQ` 用 structural Jacobian 和 trajectory-level drift 做 mixed-precision allocation；`Mix-QVLA` 則比較 full-precision 與 quantized policy 在多個 functional boundaries 的 task-evidence maps。

這條演進的共同直覺是：free-space transport、contact、grasp alignment 與 long-horizon recovery 對 numerical noise 的 tolerance 不同。不過，動態方法帶來新的 failure surface：thresholds、sensor noise、out-of-distribution kinematics、kernel switching、CPU/GPU synchronization，以及 calibration trajectory 是否涵蓋真正的 critical states。

### 5. Algorithmic compression 必須落到 runtime

`RLRC` 是最清楚的反例：更低 memory 的 NF4 variant 反而更慢。`ActQuant` 因此把 per-tensor bit allocation 約束成 dense low-bit kernels 能處理的形狀，並用 `OmniModel.cpp` 做 native deployment。`VQVLA` 和 `SpecVLA` 更進一步設計 custom accelerator，但它們的 headline speedup 主要來自 cycle simulation / synthesized hardware，不能等同於現成 Jetson 的實測。

`FlashDrive` 也提醒不要把整個 pipeline speedup 歸因於 quantization：它的 4.7× 同時包含 streaming KV-cache reuse、speculative reasoning、adaptive flow-step caching、CUDA Graph 與 kernel fusion；W4A8 只把已優化的 176.0 ms 再降到 151.4 ms。良好的 report 應拆開 model-only、kernel-only 與 end-to-end gains。

### 6. Edge deployment 需要更完整的 measurement contract

一個足夠可信的 VLA quantization evaluation 至少要同時報告：static model footprint、peak VRAM、end-to-end latency、control frequency、latency variance、closed-loop success、action chunk length、energy/power、hardware、runtime、calibration set、episode count 與 confidence interval。現有 papers 通常只覆蓋其中一部分。

`SQIL` 的 Jetson 結果相對完整，包含 latency、memory 和 energy；`LiteVLA-Edge` 有 timing 和 jitter，但沒有 matched task success；`Embodied.cpp` 有跨 model 的 normalized runtime/memory/success，但缺少足夠 absolute context；`Accessible Physical AI` 有 consumer GPU real-task feasibility，卻只有單一任務且沒有 matched baseline。因此「能跑」與「量化後仍可靠地完成任務」必須分開表述。

## 主要 contradictions 與 interpretation

- **Quantized model 偶爾高於 full precision。** OpenVLA INT4、SQAP-VLA、QuantVLA、HoloQ-VLA 都出現局部 improvement。最保守的 interpretation 是 rollout variance、regularization effect 或 calibration luck；沒有 matched seeds、足夠 rollouts 與 confidence intervals 時，不應聲稱 quantization 本身提升 policy quality。
- **Uniform W4A4 到底是否可行？** 在 naive baselines 上常 collapse；`HoloQ-VLA` 在 two DiT families 報告可行。結論應是「需要 architecture-specific outlier transforms 與 timestep calibration，且尚待 independent reproduction」，不是「所有 VLA 都能直接 W4A4」。
- **Dynamic precision 是否一定更好？** `DyQ-VLA` 在相近 memory 下接近 `QVLA`，主要價值是把 precision 與 execution phase 對齊；其收益依賴 custom backend 與 kinematic proxy，不能從 accuracy table 單獨推導。
- **Hardware co-design 的 speedup 是否可直接採用？** `VQVLA`、`SpecVLA` 的 custom hardware results 對 architecture research 有價值，但與 commercially available GPU/edge device 的 wall-clock measurement 屬於不同 evidence level。

## Research gaps

1. 缺少同一 checkpoint、同一 runtime、同一 GPU/edge device、同一 calibration set 與同一 rollout seeds 的 broad comparison。
2. 很少 paper 同時量測 total process peak memory、KV-cache、vision encoder、action head、runtime workspace 與 metadata；只報 target-module footprint 容易高估可部署性。
3. 真實 robot evaluation 通常 task 數少、每 task 約 10–30 trials，且很少報 confidence intervals、failure taxonomy 與 long-duration thermal/power behavior。
4. Calibration shift 尚未充分研究：camera change、new embodiment、contact-rich tasks、sensor degradation 與 long-horizon recovery 可能改寫 sensitivity map。
5. Dynamic quantization 的 control stability、bit-switch hysteresis、worst-case latency 與 scheduling jitter 缺乏統一 protocol。
6. Security 幾乎空白；`Bit-Flip Attacks` 顯示 integrity 是獨立於 average accuracy 的 deployment requirement，但目前只有 logical fault evidence。
7. 缺少 quantization 與 action chunking、cache reuse、token pruning、speculative decoding 之間的 factorial ablation，因而很難知道各 component 的 interaction。

## 適合作為 FYP 的可執行方向

最實際的 project shape 是：選定一個 open VLA checkpoint（例如 OpenVLA-OFT），固定 LIBERO version、episodes、seeds 與 one GPU；比較 BF16、generic W4A16、VLA-aware W4A16/W4A4；同時記錄 success rate、per-step/action-chunk latency、peak VRAM、control frequency 和 power。核心 contribution 可以放在 **phase-aware calibration / bit allocation 是否在 matched runtime 下仍優於 static PTQ**，而不是再提出一個沒有 kernel support 的 sensitivity score。

最低可行 experiment：

1. 重現 OpenVLA 或 QVLA 的 BF16 / INT4 baseline。
2. 用 task phase（free-space、pre-contact、contact、post-contact）標記 calibration frames。
3. 比較 random calibration、uniform temporal calibration、phase-balanced calibration。
4. 不改 kernel 時先測 closed-loop robustness；只有在 accuracy signal 穩定後再做 runtime integration。
5. 報告 mean、variance、failure type、peak memory、latency distribution 和 power，不只報 model size。

## Source and version map

- [OpenVLA — PMLR final](https://proceedings.mlr.press/v270/kim25c.html)
- [SQIL — ICCV 2025 final](https://openaccess.thecvf.com/content/ICCV2025/html/Park_Saliency-Aware_Quantized_Imitation_Learning_for_Efficient_Robotic_Control_ICCV_2025_paper.html)
- [BitVLA — arXiv](https://arxiv.org/abs/2506.07530)
- [RLRC — arXiv](https://arxiv.org/abs/2506.17639)
- [SQAP-VLA — arXiv](https://arxiv.org/abs/2509.09090)
- [Towards Accessible Physical AI — arXiv](https://arxiv.org/abs/2512.11921)
- [QVLA — arXiv identity](https://arxiv.org/abs/2602.03782)
- [HB-VLA — arXiv](https://arxiv.org/abs/2602.13710)
- [QuantVLA — CVPR 2026 final](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_QuantVLA_Scale-Calibrated_Post-Training_Quantization_for_Vision-Language-Action_Models_CVPR_2026_paper.html)
- [LiteVLA-Edge — arXiv](https://arxiv.org/abs/2603.03380)
- [DyQ-VLA — arXiv](https://arxiv.org/abs/2603.07904)
- [DA-PTQ — arXiv](https://arxiv.org/abs/2604.11572)
- [ActQuant — arXiv](https://arxiv.org/abs/2605.24011)
- [HoloQ-VLA — arXiv](https://arxiv.org/abs/2605.28803)
- [Mix-QVLA — arXiv](https://arxiv.org/abs/2606.19565)
- [Embodied.cpp — arXiv](https://arxiv.org/abs/2607.02501)
- [VQVLA — arXiv](https://arxiv.org/abs/2607.24148)
- [FlashDrive — arXiv](https://arxiv.org/abs/2608.12932)
- [Bit-Flip Attacks on VLA Models — arXiv](https://arxiv.org/abs/2608.15475)
- [SpecVLA — arXiv](https://arxiv.org/abs/2608.15636)

## Review limitations and disclosure

這份 review 截止於 2026-09-03。Search APIs 的 rate limits 與 unavailable sources 已在上文披露；沒有把 citation count 當作 quality proxy。數字以 local PDFs 的 authors-reported results 為準，沒有把不同 evaluation protocols 的 score 做 meta-analysis。

本 review 由 AI 協助 discovery、screening、PDF extraction、evidence structuring 與 drafting；paper selection、version labels、關鍵 claims、排除邊界與 local files 已逐項核對。所有 synthesis 與 project suggestions 都是 reviewer-level interpretation，不是原 paper 的直接結論。
