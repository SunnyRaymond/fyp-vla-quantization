# 8-week reading and weekly-meeting plan

这份安排假设每周有一次组会。目标不是“每篇都读完”，而是每周至少形成一个可以被追问、可以承认边界的 technical story。

## Weekly cadence

### Minimum route — 3 hours / week

1. **Core paper, 90 min**：Figure 1、Method overview、主结果表、Limitations。
2. **Companion paper, 45 min**：只回答 WHY / HOW / WHAT。
3. **Comparison, 20 min**：写出两篇的 shared assumption 与关键差异。
4. **Meeting card, 25 min**：完成 5 个 bullet。

最低完成线：即使没读完，也必须能说清 `Problem → Mechanism → Evidence → Limitation → Question`。

### Standard route — 6 hours / week

- 两篇都按 60–90 minute route 阅读。
- 复述一条核心 equation / architecture path。
- 核对两张表或两个 ablation，不只引用 abstract headline。
- 准备 5 slides，并保留一页 backup evidence。

### Stretch route — 9+ hours / week

- 读 appendix / supplementary material。
- 检查 code 或 implementation detail。
- 做一次小型 reproduction / profiling proposal。
- 把方法映射到 VLA deployment，提出可验证 hypothesis。

## Schedule

### Priority insertion — 李老师 2026-09-04 assignment

这三篇先作为一个独立 reading sprint，不等待原 8-week cycle 走完：

| Session | Core | Contrast | Meeting theme | Minimum deliverable |
|---|---|---|---|---|
| A | [HBVLA](papers/20-hbvla/README.md) | [BitVLA](papers/06-bitvla/README.md) + [OPTQ/GPTQ](papers/10a-gptq-optq/README.md) | native low-bit training vs action-aware 1-bit PTQ；weight memory vs actual closed-loop retention | 画 `rectified Hessian → column partition → Haar W1`；报告 Mobile ALOHA 12.5–23.4 pp absolute drop；列出 2.93× latency headline 缺失的 protocol |
| B | [WorldCache — Content-Aware](papers/21-worldcache-content-aware/README.md) | 回看 world/video path | static background、small moving objects、stale features 与 late denoising | 用一句话分别解释 `CFC/SWD/OFA/ATS`；从 Table 4 画 speed–quality path；明确 EgoDex 不是 policy success |
| C | [WorldCache — Heterogeneous Token](papers/22-worldcache-heterogeneous-token-caching/README.md) | [Content-Aware WorldCache](papers/21-worldcache-content-aware/README.md) | deep-block feature reuse vs token-level heterogeneous prediction | 手算一例 curvature；解释 `CHTP/CAS`；列出 A800/H200、model、schedule、granularity 为什么不 matched |
| Synthesis | [Three-paper targeted scan](research/THREE_PAPER_SCAN_2026-09-04.md) | 三篇一起 | `precision allocation + compute reuse + temporal error` | 一张 6-column comparison；一个 `quantization × caching` matched experiment；一个交给李老师的问题 |

若本周只有 90 minutes：三篇各走 20-minute route，再用 30 minutes 填 synthesis table。不要为了追完整 related work 而扩张 core。

| Week | Core | Companion | Meeting theme | Minimum deliverable |
|---|---|---|---|---|
| Foundation before Week 1 | [Flow Matching](papers/17-flow-matching/README.md) | 预览 [π0](papers/02-pi0/README.md) 的 flow loss | `probability path → conditional velocity target → ODE sampling` | 推导 linear conditional `OT` path；解释 simulation-free training 为什么不等于 one-step inference；把 $x_t,u_t$ 映射到 $A^\tau,A-\epsilon$ |
| Representation bridge before Week 1 | [Self-Flow](papers/18-self-flow/README.md) | [REPA / iREPA → VLA representation alignment](papers/19-repa-irepa-vla-alignment/README.md)，再读 optional [mechanism critique](papers/18-self-flow/README.md#83-direct-post-publication-mechanism-challenge) | `external frozen teacher → internal EMA teacher → Dual-Timestep Scheduling → data-augmentation challenge` | 画出 student/teacher/noise path；核对 Tables 1-4 与 Appendix E；解释 `Attention Separation` 检验了什么、没有检验什么 |
| 1 | [OpenVLA](papers/01-openvla/README.md) | [π0](papers/02-pi0/README.md) | `discrete action tokens` vs `flow matching action expert` | 画出两个 input-to-action pipeline；解释 data scale 与 openness 的 trade-off |
| Bridge before Week 2 | [FAST](papers/02a-fast/README.md) | 回看 [π0](papers/02-pi0/README.md) | `naive scalar tokens` vs `DCT+BPE trajectory tokens` vs `continuous flow chunks` | 解释 FAST 五步 encoding；区分 training efficiency 与 inference latency |
| 2 | [π0.5](papers/03-pi05/README.md) | [π*0.6](papers/04-pistar06/README.md) | 从 `imitation learning` 到 `open-world generalization` 与 `experience-driven improvement` | 区分 co-training、hierarchical inference、offline RL、online corrections |
| Systems bridge after Week 2 | [FlashVLA](papers/03a-flashvla/README.md) | 回看 [π0.5](papers/03-pi05/README.md) | `isolated flow chunks` → `staggered streaming buffer`；policy latency vs control frequency | 画出 clean-to-noisy buffer 与 causal direction；解释 `19.9× / 3.5× / 2.43× / 30 Hz` 的不同 denominator |
| Robot RL systems bridge after Week 2 | [FlashSAC](papers/18a-flashsac/README.md) | 回看 [π*0.6](papers/04-pistar06/README.md) 的 offline RL / advantage conditioning | `PPO on-policy data → SAC replay → scaled off-policy learning`；sample efficiency vs wall-clock vs control cadence | 画出 replay/critic/target path；解释 UTD、1024/4096 parallel environments、50 Hz policy 与 200 Hz PD control；审计 Table 14 reward confound |
| Bridge after Week 2 | [MEM](papers/04a-mem/README.md) | 回看 [π0.5](papers/03-pi05/README.md) + [π*0.6](papers/04-pistar06/README.md) | `generalization` vs `learning from experience` vs `remembering context` | 画出 high-level memory update 与 low-level action path；解释 video/proprioceptive/text 三种 memory 的 timescale 分工 |
| 3 | [FP8 Formats](papers/07-fp8-formats/README.md) | [Microscaling Data Formats](papers/08-microscaling-formats/README.md) | 从 per-tensor scaling 到 block scaling | 手算 E4M3/E5M2 dynamic range 取舍；画 MX block + shared scale |
| Standalone core before Week 4 | [Outlier Suppression](papers/16-outlier-suppression/README.md) | [LLM.int8 / bitsandbytes](papers/06d-llm-int8-bitsandbytes/README.md)，然后预览 [SmoothQuant](papers/10-smoothquant/README.md) | `route outliers → suppress outliers → migrate quantization difficulty` | 推导 Gamma Migration；解释 Token-Wise Clipping；用 SmoothQuant matched table 界定 large-LLM transfer gap |
| 4 | [SmoothQuant](papers/10-smoothquant/README.md) | [AWQ](papers/11-awq/README.md) | Equivalent transformation 如何重新分配 quantization difficulty | 比较 activation smoothing 与 activation-aware weight scaling；说明 W8A8 vs weight-only |
| Bridge after Week 4 | [OPTQ/GPTQ](papers/10a-gptq-optq/README.md) | [2026 quantization survey](papers/10b-llm-quantization-survey-2026/README.md) + [2025 low-bit survey](papers/10c-low-bit-llm-survey/README.md) | second-order compensation 与 field taxonomy | 推导 $H=2XX^\top$；把 GPTQ/SmoothQuant/AWQ 放入 lifecycle × object × error-control map |
| Bridge before Week 5 | [QuaRot](papers/12a-quarot/README.md) | 回看 [SpinQuant](papers/12-spinquant/README.md) | fixed/randomized Hadamard rotation → learned rotation | 画出 QuaRot 的 fused 与 online paths；用 matched Table 5 rows 解释 SpinQuant 的 research gap |
| 5 | [SpinQuant](papers/12-spinquant/README.md) | [HAQ](papers/13-haq/README.md) | `representation transform` vs `hardware-aware bit allocation` | 解释 learned rotation 为何保持 full-precision function；解释 hardware feedback 为什么不能用 FLOPs 替代 |
| 6 | [TurboQuant](papers/14-turboquant/README.md) | [NVFP4 pretraining](papers/09-nvfp4-pretraining/README.md) | 极低 bit-width 下的 distortion、scaling 与 training stability | 分开 theoretical distortion guarantee、KV-cache evidence、native hardware recipe |
| Standalone core before Week 7 | [LoRA](papers/15-lora/README.md) | 回看 [OpenVLA](papers/01-openvla/README.md) + [OpenVLA-OFT](papers/06c-openvla-oft/README.md) | `full fine-tuning → low-rank update → VLA adaptation` | 推导 $W_0+BA$；分开 trainable parameters、task-specific storage、base-model serving cost；提出 component-wise LoRA ablation |
| 7 | [BitVLA](papers/06-bitvla/README.md) | [ω-0](papers/05-omega0/README.md) | 对老师标记为“may not be that good”的论文做 critical reading | 每篇给出 strongest contribution、strongest unsupported leap、最需要的 missing baseline |
| Bridge after Week 7 | [OpenVLA-OFT](papers/06c-openvla-oft/README.md) + [LLaVA](papers/06b-llava/README.md) | [Apprentice](papers/06a-apprentice-quantization-distillation/README.md) + [LLM.int8/bitsandbytes](papers/06d-llm-int8-bitsandbytes/README.md) | 拆开 BitVLA inherited components 与 actual contribution | 解释 Quantize-then-Distill prior art、three camera views、bnb PTQ boundary；提出一个 matched BitVLA + flow experiment |
| 8 | 全部 | Researcher watchlist | VLA × quantization × deployment synthesis | 一张 design-space map；一个可执行的 FYP hypothesis；三项最小实验 |

## Why this order

1. 先读 Flow Matching 的 `FM → CFM → linear conditional OT path`，再看 `π0`；这样不会把 action expert 的 loss 当成 Physical Intelligence 凭空提出的公式。编号 `#17` 是新增顺序，不是 chronological order。
2. 再读 Self-Flow，把 `Flow Matching` 与 internal representation learning 接起来；随后用 `From SRA to Self-Flow` 检验 causal explanation，而不是把 original performance result 与 later mechanism critique 混成一个结论。
3. 再用 OpenVLA 与 π0 建立 discrete autoregression 和 continuous flow 两种 action representation。
4. 用 FAST 理解 trajectory compression，以及 π0.5 为什么把 all-discrete pretraining 与 flow post-training 分成两阶段。
5. 用 FlashVLA 把 π0.5 的 flow action expert 推进到真实 deployment scheduling：同一方法必须同时测 policy latency、observation staleness、control frequency 与 closed-loop success。
6. 用 FlashSAC 建立 Robot RL systems baseline：把 replay/sample reuse、UTD、critic stability、wall-clock、simulator throughput、policy frequency 与 low-level control frequency 分开，再讨论 VLA/offline-RL adaptation。
7. 再看 π series 如何分别把 generalization、learning from experience 与 multi-scale memory 加进系统；MEM bridge 还会暴露 temporal context 对 VLA memory/latency 的新压力。
8. 进入 Week 4 前先读 Outlier Suppression：从 `LayerNorm γ` amplifier 与 token importance 出发，再看 SmoothQuant 怎样把 equivalent scaling 推到 large-LLM W8A8。编号 `#16` 是新增顺序，不是 chronological order。
9. 之后沿 number format → outlier suppression → equivalent transform → rotation / mixed precision → ultra-low precision，避免把 quantization method 当成互不相干的技巧。
10. 在评估 BitVLA adaptation 前，把 LoRA 作为 standalone core foundation：它澄清 low-rank update 减少的是 training states 与 task delta，不是 frozen base model 的全部 inference cost。
11. 最后才读 BitVLA 与 ω-0：此时你已经有足够 baseline knowledge 做 critique，而不是被 headline 带走。
12. Week 7 后的 companion bridge 不增加 core count；它只追溯 BitVLA 的 LLaVA curriculum、OpenVLA-OFT action head、low-precision distillation prior 与 bitsandbytes PTQ baseline。
13. Week 4 后的 quantization bridge 也不增加 core count；GPTQ 补齐 second-order weight-only PTQ，两个 survey 只作为 field map 与 lookup reference。
14. Week 2 后的 FlashVLA、FlashSAC 与 MEM bridges 都不增加 core count；它们分别补 deployment scheduling、off-policy Robot RL systems、`π₀.₆` temporal memory branch。
15. 2026-09-04 的三篇按 `HBVLA/HB-VLA → Content-Aware WorldCache → Heterogeneous Token WorldCache` 插入：先读与你 FYP 最直接的 1-bit PTQ，再用第一篇 WorldCache 建立 caching skeleton，最后比较 token-level heterogeneous predictor。两篇同名论文必须带 subtitle/arXiv ID。

## 5-slide meeting template

1. **Problem and setting**：任务、输入输出、为什么现有方法不够。
2. **Core mechanism**：只画一张你自己能解释的 diagram / equation。
3. **Evidence**：一张主结果表 + 一项最有诊断力的 ablation。
4. **Limitations**：authors' stated limitations 与你的 critique 分开。
5. **Takeaway and question**：对 FYP 的一个影响，以及希望组里讨论的一个问题。

## If a week collapses

不要补读到下一周失控。改用以下 recovery rule：

- 只完成 Core paper 的 `20-minute route` 与一张 evidence table。
- Companion paper 只读 abstract、Figure 1、结论；标记为 `skimmed`，不要假装读完。
- 组会中直接说明 `unread / skimmed / deep-read` 的范围。
- Week 8 预留为 buffer；只有真正影响 FYP hypothesis 的遗漏才回补。

## End-of-cycle decision

第 8 周不要再加 paper。用已有阅读回答：

1. VLA 的实际 deployment bottleneck 是 parameter memory、activation/KV memory、latency，还是 data / control frequency？
2. 现有 LLM quantization assumption 哪些能直接迁移到 VLA，哪些会被 continuous action、temporal feedback 或 embodiment 打破？
3. 最小可行实验应优先测 `accuracy-only`，还是 `success rate + latency + memory + power` 的 multi-objective trade-off？
4. LoRA 应分配给 language backbone、visual encoder/projector 还是 action expert？其 rank 与 target-module choice 是否应由 closed-loop evidence 决定？
5. VLA 中的 large activation 是 removable outlier，还是 rare corrective-control signal？哪种 calibration/rollout evidence 能决定是否安全 clipping？
6. 在 matched VLA checkpoint 下，probability path、timestep sampling、solver steps 与 low precision 怎样共同影响 endpoint action error 和 closed-loop success？
7. 若引入 off-policy replay/critic，UTD、buffer composition、reward shaping 与 quantization 如何共同影响 critic stability、sample efficiency 与 wall-clock，而不是只改变 inference latency？
8. 若将 quantization 与 world-model caching 组合，low-bit noise 会怎样改变 drift/curvature detector、cache hit rate、action choice 与 closed-loop error accumulation？


## 2026-09-05 师兄报告追加路线

独立整数编号 [#23 WaterSIC](papers/23-watersic/README.md) 与 [#24 GRACE](papers/24-grace/README.md) 已加入主 Reading List。

- **本周 minimum route（3 hours）**：WaterSIC 75 min → GRACE 75 min → 30 min 填两张 Meeting Cards，区分 rate–distortion allocation、teacher information selection 与尚未验证的 VLA transfer。
- **深入 route（另加 3 hours）**：WaterSIC theorem assumptions / actual entropy storage 90 min；GRACE quantization scope / matched ablations / optimizer configuration 90 min。
- 第二份 VLA_idea 的任务按 [独立三方向阅读计划](../vla-ideas-reading-guide/README.md) 进行；无需把 30 项塞进这份原有 8-week plan。
