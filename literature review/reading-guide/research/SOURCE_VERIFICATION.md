# Source verification report

> Snapshot: 2026-09-04. `Grade` 表示该 source 对当前阅读目的的可用性，不表示方法优劣。

## Overall assessment

- Core paper/report identities: 22（原 19 篇加上李老师新增指定的 `#20` HBVLA、`#21` Content-Aware WorldCache 与 `#22` Heterogeneous Token WorldCache）
- VLA systems companion: 1（FlashVLA, arXiv:2608.27384 v1）
- Robot RL systems companion: 1（FlashSAC, RSS 2026 Outstanding Paper Award / arXiv:2604.04539 v2）
- PI-series companion: 1（MEM）
- BitVLA dependency/prior-art companions: 4（Apprentice、LLaVA、OpenVLA-OFT、LLM.int8）
- Quantization companions: 8（OPTQ/GPTQ、QuaRot、two surveys、original RaBitQ、multi-bit RaBitQ extension、Revisiting RaBitQ and TurboQuant、DRIVE/EDEN technical note）
- Representation-alignment companions: 12（REPA、iREPA，以及 ten direct/adjacent VLA/WAM works）
- Self-Flow mechanism companion: 1（From SRA to Self-Flow, arXiv:2607.02508 v1）
- Within the 22-paper active core: 15 peer-reviewed / accepted conference papers；7 arXiv preprints / official technical reports
- All four BitVLA companions have peer-reviewed conference records；the local Apprentice copy uses arXiv v1 because unattended OpenReview PDF acquisition was blocked
- Supporting official specification: OCP Microscaling Formats (MX) Specification v1.0
- Rejected identity: `NVFP8` as a distinct formally defined format/paper（未找到权威定义，不据此虚构 entry）

## Quality matrix

| Entry | Canonical status | Verification | Grade | Main caveat |
|---|---|---|---|---|
| OpenVLA | CoRL 2024 / PMLR | official proceedings + arXiv full text | A | Evaluation tasks and real-robot setup differ from later π-series tasks |
| π0 | RSS 2025 | official proceedings + arXiv/official full text | A | Internal data and custom progress metrics still limit independent reproduction |
| FAST | RSS 2025 | official proceedings + arXiv/official full text + released FAST+ tokenizer | A | Training is faster but autoregressive action inference is slower; universal-tokenizer evidence is not arbitrary-embodiment policy generalization |
| π0.5 | CoRL 2025 oral / PMLR | official proceedings + arXiv/official full text | A | Internal data and evaluation protocol limit independent reproduction |
| π*0.6 | Physical Intelligence technical report | official PDF + project page | B | Teacher/value/reward pipeline and real-world infrastructure are not fully reproducible externally |
| ω-0 | arXiv:2608.06375 | arXiv full text | B | Very recent preprint; benchmark and dataset are author-controlled |
| BitVLA | arXiv:2506.07530 | arXiv full text + official code link | B | Pretraining/data scale differs from strong VLA baselines; memory claim is not a full deployment study |
| FP8 Formats for Deep Learning | arXiv:2209.05433 / industry whitepaper | arXiv full text + official vendor announcement | B | Multi-company authors define and evaluate their proposed format; early hardware claims are vendor-linked |
| Microscaling Data Formats for Deep Learning | arXiv:2310.10537 | arXiv full text + Microsoft Research page | B | Preprint; breadth of tasks is stronger evidence for format viability than for every deployment setting |
| Pretraining Large Language Models with NVFP4 | arXiv:2509.25149 | arXiv full text + NVIDIA documentation | B | NVIDIA-authored, hardware-specific recipe; headline efficiency requires Blackwell-class support |
| SmoothQuant | ICML 2023 / PMLR | official proceedings + arXiv full text | A | LLM W8A8 evidence does not automatically imply closed-loop VLA robustness |
| AWQ | MLSys 2024 | official proceedings + arXiv full text | A | Weight-only results mix algorithmic accuracy and TinyChat systems optimization |
| SpinQuant | ICLR 2025 | official conference record + arXiv full text | A | Rotation learning/calibration cost and hardware kernels matter in practice |
| HAQ | CVPR 2019 | official CVF record + arXiv full text | A | CNN-era hardware simulator and RL search do not transfer unchanged to transformer/VLA stacks |
| TurboQuant | ICLR 2026 | official proceedings final + archived arXiv v1 | A | Theory targets vector distortion; arXiv-v1 RaBitQ timing table used an allegedly asymmetric baseline and is absent from the proceedings final |
| LoRA | ICLR 2022 | Microsoft Research venue record + DBLP/OpenReview link + arXiv v2 full text + official code | A | GPT-3 scale claims use closed infrastructure; small-rank and no-extra-latency conclusions are conditional on target/task and merged single-adapter serving |
| Outlier Suppression | NeurIPS 2022 | official proceedings main + supplemental + arXiv metadata + official code | A | Original evidence is BERT/RoBERTa/BART; no random-seed error bars or end-to-end deployment metrics, and later SmoothQuant baselines show failure on 100B-scale LLMs |
| Flow Matching for Generative Modeling | ICLR 2023, Top 25% | official ICLR record + OpenReview identity + arXiv v2 verified full text | A | Main evidence is image generation; simulation-free training still uses ODE integration for sampling, and conditional OT paths do not guarantee a globally optimal/straight marginal flow |
| Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis | ICML 2026 poster | official ICML accepted-paper list + OpenReview identity `HoThWhfxiK` + arXiv v1 verified full text + official project/code | A | Local artifact is arXiv v1 rather than proceedings; author-run multi-modal results lack seed intervals, training-throughput/memory/power accounting, and independent replication; `SIMPLER` uses a small fixed task list and two evaluation repeats |
| HBVLA: Pushing 1-Bit Post-Training Quantization for Vision-Language-Action Models | arXiv:2602.13710 v2 preprint | arXiv metadata + verified 9-page local v2 full text | B | No official code or venue was verified; local v2 repeatedly cites an appendix that is absent; real-world success drops 12.5–23.4 pp and the 2.93× latency headline lacks a complete table/protocol in the PDF |
| WorldCache: Content-Aware Caching for Accelerated Video World Models | ECCV 2026 accepted; arXiv:2603.22286 v1 | arXiv metadata/local full text + official ECCV listing + project/code | A | Local artifact is arXiv v1 rather than proceedings; quality is mainly automatic video/world-model evaluation, and EgoDex evidence is prediction fidelity rather than policy-in-loop success |
| WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching | ICML 2026 / PMLR 306; arXiv:2603.06331 v2 | arXiv metadata + official ICML accepted-paper listing + verified 26-page local full text + official code | A | Author-run A800 results lack uncertainty and closed-loop action evaluation; layer-wise cache latency comparisons are partly confounded by CPU offloading |

## VLA systems companion quality matrix

| Entry | Canonical status | Verification | Grade | Main caveat |
|---|---|---|---|---|
| FlashVLA: Streaming Action Decoding for Fast and Asynchronous VLA Inference | arXiv:2608.27384 v1 | arXiv metadata + verified local full text + official code/checkpoint repository | B | Four-day-old preprint at snapshot time; author-run benchmarks, 8-H200 fine-tuning, limited real-world trials, and no peak-memory/power/tail-latency report; `FlashVLA` name collides with arXiv:2505.21200, and the live repository's LingBot-VLA table differs from arXiv v1 Table 5 |

## Robot RL systems companion quality matrix

| Entry | Canonical status | Verification | Grade | Main caveat |
|---|---|---|---|---|
| FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control | RSS 2026 Outstanding Paper Award / arXiv:2604.04539 v2 | official RSS paper and awards pages + arXiv v2 full text + official project/code repository | A | Composite author-run benchmark; baseline environment-step budgets are not always matched; UTD denominator and sim-to-real reward-description inconsistencies remain unresolved; no peak-memory, power, tail-latency, or independent physical replication report |

## Representation-alignment companion quality matrix

| Entry | Canonical status | Verification | Grade | Main caveat |
|---|---|---|---|---|
| REPA / iREPA → VLA/WAM prior-art route | 2 peer-reviewed method papers + 10 direct/adjacent robotics papers, versioned at 2026-08-31 | 12 local full texts + per-paper source inventory and search report in `papers/19-repa-irepa-vla-alignment/` | mixed A/B | This is a mechanism/prior-art map, not a matched benchmark; external teachers, policies, datasets, action interfaces, and environment protocols differ across works |

## Self-Flow mechanism companion quality matrix

| Entry | Canonical status | Verification | Grade | Main caveat |
|---|---|---|---|---|
| From SRA to Self-Flow: Data Augmentation or Self-Supervision? | arXiv:2607.02508 v1 | arXiv metadata + verified 10-page local full text + released code path | B | Unreviewed preprint; controlled mechanism test uses a smaller ImageNet/SiT-B setup and does not rerun Self-Flow's multi-modal, scaling, or robotics experiments |

## PI-series companion quality matrix

| Entry | Canonical status | Verification | Grade | Main caveat |
|---|---|---|---|---|
| MEM | Physical Intelligence technical report / arXiv:2603.03596 v2 | official project page + official PDF + arXiv metadata/full text | B | Internal model/data/tasks and 10-rollout evaluations limit independent reproduction; no official code, weights, or dataset release was verified as of 2026-08-25 |

## BitVLA companion quality matrix

| Entry | Canonical status | Verification | Grade | Main caveat |
|---|---|---|---|---|
| Apprentice | ICLR 2018 / arXiv:1711.05852 | arXiv full text + OpenReview venue record | A | Direct conceptual predecessor for quantization-aware Knowledge Distillation, not the exact BitVLA recipe |
| LLaVA / Visual Instruction Tuning | NeurIPS 2023 Oral | NeurIPS venue final + arXiv/project records | A | Original LLaVA is a visual assistant; BitVLA inherits the curriculum, not the robot-policy architecture |
| OpenVLA-OFT | RSS 2025 | RSS proceedings + arXiv v2 + official project/code | A | Three-image benchmark is ALOHA-specific; action throughput is chunk-amortized, not query/control rate |
| LLM.int8 / bitsandbytes | NeurIPS 2022 | NeurIPS venue final + current official bitsandbytes documentation | A | Formal paper defines the 8-bit path; BitVLA's 4-bit bitsandbytes configuration is not fully specified |

## Quantization companion quality matrix

| Entry | Canonical status | Verification | Grade | Main caveat |
|---|---|---|---|---|
| OPTQ (commonly known as GPTQ) | ICLR 2023 | ISTA Published Version + OpenReview venue record + arXiv v2 + official code | A | Weight-only layer-reconstruction evidence and paper-specific batch-1 kernel do not establish universal end-to-end speedup |
| QuaRot | NeurIPS 2024 | official proceedings final + arXiv v2 + official code | A | Fixed/random rotations are not task-loss optimized; speed/memory evidence is a custom-kernel single-block RTX 3090 benchmark, not a full-model VLA deployment study |
| A Survey of Quantization in LLM | JCST 41(1), 2026 | official journal page + venue PDF + DOI metadata | A | Narrative survey without reproducible systematic-search or quality-assessment protocol |
| A Survey of Low-bit Large Language Models | Neural Networks 192, 2025 | DOI/publisher record + arXiv v3 full text | A | Local copy is an author manuscript; framework/kernel support tables age quickly |
| RaBitQ | SIGMOD 2024 / DOI `10.1145/3654970` | official SIGMOD listing + arXiv v1 full text + author code | A | Original target is ANN distance/inner-product estimation; it does not establish neural-network inference speedup |
| Practical and Asymptotically Optimal Quantization... | SIGMOD 2025 / DOI `10.1145/3725413` | venue/DOI metadata + arXiv v1 full text + author code/library | A | Local copy is the author preprint; neural-network PTQ integration remains a synthesis rather than a paper result |
| Revisiting RaBitQ and TurboQuant | VecDB@VLDB 2026 workshop poster / arXiv:2604.19528v2 | official workshop programme + arXiv v2 + released code | B | Authors are the RaBitQ team; private-email evidence and reproduction results were not independently rerun here |
| A Note on TurboQuant and the Earlier DRIVE/EDEN Line of Work | arXiv:2604.18555v1 | arXiv v1 full text | B | Author-produced prior-art note; method-overlap argument is not an independent plagiarism finding |

## Non-paper technical materials

| Material | Role | Status |
|---|---|---|
| OCP Microscaling Formats (MX) Specification v1.0 | Normative definitions for MXFP8/MXFP4 | `official-technical-material` |
| NVIDIA Transformer Engine NVFP4 documentation | Implementation recipe and supported layout | `official-technical-material` |
| NVIDIA technical blogs | Hardware/performance context | `official-technical-material`; vendor claims must be attributed |

## Identity resolutions

1. `π0.6` in the advisor list maps to **π*0.6**, not a paper canonically titled plain π0.6. **MEM** is a separate `π₀.₆` memory extension and is now represented as a companion rather than merged with the assigned entry.
2. `MXFP8` and `MXFP4` are concrete formats in the OCP MX specification; the paper-level reading is **Microscaling Data Formats for Deep Learning**.
3. `NVFP4` has official NVIDIA documentation and a paper-level reading, **Pretraining Large Language Models with NVFP4**.
4. `NVFP8` was not verified as a distinct NVIDIA format/specification. Some informal model/checkpoint contexts use the term for FP8, but the authoritative material found uses `FP8`, `MXFP8`, and `NVFP4`.
5. `Quantize-then-Distill` is BitVLA's name for its VLM-specific recipe, but the broad `full-precision teacher → low-precision student` concept predates it; Apprentice Scheme-C is the closest verified predecessor in this companion pass.
6. `bitsandbytes` is a software library, not one numeric format. BitVLA citation `[9]` resolves to LLM.int8 for the 8-bit path; the paper text does not resolve whether its reported “INT4” baseline used FP4 or NF4 or which auxiliary settings were active.
7. BitVLA Fig. 6's three images resolve to the ALOHA/OpenVLA-OFT setup: one third-person/top-down camera plus two wrist cameras. This differs from BitVLA LIBERO's two-view input and Franka's one-view real-world input.
8. ICLR 2023 uses the title `OPTQ`; arXiv v2 and the official implementation use `GPTQ`. They are one work, not two papers.
9. “Latest survey” is snapshot-scoped: JCST 2026 is the newest peer-reviewed survey found whose title/scope is dedicated to LLM quantization; the Neural Networks 2025 survey is broader and more comprehensive but slightly older.
10. `QuaRot` is the direct fixed/randomized Hadamard-rotation predecessor used by `SpinQuant`; it remains an optional companion and does not change the advisor-assigned core count.
11. `FlashVLA` is ambiguous without an identifier. The local VLA systems companion is arXiv:2608.27384, which uses streaming action decoding; arXiv:2505.21200 is *Think Twice, Act Once* and uses action reuse plus visual-token selection.
12. `LoRA` resolves to Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv:2106.09685 v2 / ICLR 2022. The local copy is arXiv v2 because OpenReview blocked unattended PDF acquisition; venue identity and artifact version are therefore recorded separately. It is user-promoted active core `#15`, not a retroactive claim that Professor Li assigned it.
13. `Outlier Suppression` resolves to Wei et al., *Outlier Suppression: Pushing the Limit of Low-bit Transformer Language Models*, NeurIPS 2022 / arXiv:2209.13325. The local canonical artifacts are the 13-page official proceedings paper and 10-page official supplemental; arXiv v1-v3 remain one work cluster. It is user-promoted active core `#16`, not a retroactive advisor assignment.
14. `Flow Matching` resolves to Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023 / arXiv:2210.02747. The local artifact is the 28-page arXiv v2 author manuscript because OpenReview blocked unattended PDF acquisition; the official ICLR page separately establishes venue identity and the Top 25% label. It is user-promoted active core `#17`, not a retroactive advisor assignment.
15. `FlashSAC` resolves to Kim et al., *FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control*, RSS 2026 / arXiv:2604.04539 v2. The official award label is **Outstanding Paper Award**. Task-count claims remain surface/version bound: the RSS paper page says `50+`, arXiv v2/project page say `60+`, and the live repository says `100+`; they are not merged into one historical paper claim.
16. `Self-Flow` resolves to Chefer et al., *Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis*, ICML 2026 poster / arXiv:2603.06507 v1, OpenReview ID `HoThWhfxiK`. The local 37-page artifact is arXiv v1 because unattended OpenReview PDF acquisition returned HTTP 403. It is user-promoted active core `#18`. Jiang et al.'s *From SRA to Self-Flow* is a separate July 2026 mechanism critique, not an alternative title or revised version of the ICML paper.
17. `HBVLA` in the user request resolves to *HBVLA: Pushing 1-Bit Post-Training Quantization for Vision-Language-Action Models*, arXiv:2602.13710 v2. The paper identity is unhyphenated; the PDF often labels the method `HB-VLA`. It is active core `#20`; no verified venue or official code is claimed.
18. `WorldCache` is a name collision. `#21` is Nawaz et al., *Content-Aware Caching for Accelerated Video World Models*, arXiv:2603.22286 v1 / ECCV 2026 accepted. `#22` is Feng et al., *Accelerating World Models for Free via Heterogeneous Token Caching*, arXiv:2603.06331 v2 / ICML 2026. They are distinct works, not two versions of one paper.
19. `For Free` in `#22` means no base-model retraining and very low controller overhead in the authors' setup. It does not mean zero memory, integration, quality risk, energy cost, or universally free acceleration.

## Conflict-of-interest and interpretation notes

- Format papers and NVIDIA technical materials are partly produced by the organizations designing the formats/hardware. This is relevant intellectual/institutional interest, not evidence of invalidity.
- Physical Intelligence evaluates its own models on internal real-robot tasks. Treat results as primary empirical evidence, but do not read them as independent benchmark certification.
- MEM's “up to 15 minutes” context is primarily carried by compressed long-term text memory; the dense visual-history experiment extends to 54 seconds, so these horizons should not be conflated.
- BitVLA combines a native low-bit backbone, different pretraining, OFT-style action adaptation, and custom kernels; comparisons with bitsandbytes PTQ baselines are not bit-width-only controlled experiments.
- The TurboQuant dispute has two separate claim sets: Gao et al. focuses on RaBitQ attribution, theory characterization, baseline symmetry, and reproducibility; Ben-Basat et al. focuses on earlier DRIVE/EDEN method lineage and scale choices. Neither paper is an independent misconduct investigation.
- FlashVLA's authors evaluate their own decoder, released code, checkpoints, and real-robot setup. This is primary evidence with normal intellectual/institutional interest, not independent benchmark certification.
- FlashVLA repository `main` at commit `5227b039ebd4f6b5cad0c27d2d6098932f0f7ed3` updates LingBot-VLA success/time-per-step values relative to arXiv v1 Table 5; the two surfaces are versioned separately rather than numerically merged.
- FlashSAC's authors evaluate their own algorithm, simulator integrations, and sim-to-real system. This is primary empirical evidence with ordinary intellectual interest, not independent benchmark certification.
- LoRA is proposed and evaluated by Microsoft authors using Microsoft GPT-3 infrastructure. This is normal intellectual/institutional interest; it strengthens the need to keep the closed GPT-3 reproducibility boundary visible rather than implying invalidity.
- Outlier Suppression is proposed and evaluated by authors affiliated with SenseTime Research and academic institutions, using their own method/code. This is ordinary intellectual/institutional interest, not evidence of invalidity; the paper's results remain primary rather than independent validation.
- Flow Matching is proposed and evaluated by Meta AI (FAIR) and Weizmann Institute of Science authors on their own ImageNet/CIFAR setups. This is ordinary intellectual/institutional interest; the results are primary author-reported evidence rather than independent replication.
- Self-Flow is proposed and evaluated by Black Forest Labs and MIT authors using Black Forest Labs architectures, data mixtures, and infrastructure. This is ordinary intellectual/institutional interest; the reported multi-modal and `SIMPLER` results are primary evidence rather than independent certification. The later mechanism critique comes from the SRA author group and is also primary author-produced evidence, so the two claim sets remain attributed separately.
- HB-VLA and both WorldCache methods are evaluated by their proposing teams. Their latency, memory, generation, simulation, and robot results are primary evidence, not independent benchmark certification. Recent 2026 status also limits independent-replication evidence.
- Preprint status is explicit. `verified-full-text` means the source was read and claims were traced; it does not mean peer review or independent replication.

## Verification limitations

- No retraction signal was identified in the authoritative records checked, but this is not a formal retraction-certificate workflow.
- Closed datasets, robot hardware, proprietary training mixtures, and vendor-specific kernels prevent full independent reproduction for several entries.
- FlashVLA was too recent for a meaningful independent-replication or citation audit at the 2026-08-31 snapshot; the installed Semantic Scholar search endpoint also returned HTTP 429 during the exact-title search.
- The FlashSAC exact-title search found the target through arXiv and DBLP, but `openreview-py` was unavailable and Semantic Scholar returned HTTP 429. Identity and award status were therefore closed with the official RSS paper/awards pages, arXiv v2, and the official project/code repository; the installed `paper-search` CLI did not support the documented `--json` option.
- The local FlashSAC arXiv v2 PDF passes `%PDF-`, 42-page pypdf reopen/page-count/unencrypted/first-page-title checks, visual render inspection, SHA-256 inventory, and ARS structural preflight (`PASS`, 42/42 pages). The paper internally gives `2/1024` in Section 4.1/Figure 8 but `2/2048` in Appendix Table 9, and describes sim-to-real reward coefficients as identical in Section 5.4 while Appendix D/Table 14 shows different weights and shaping; the reading note records rather than silently resolves these conflicts.
- The LoRA scripted search encountered arXiv and Semantic Scholar HTTP 429 responses, missing `openreview-py`, and an OpenReview browser-verification challenge. Identity was closed with arXiv, Microsoft Research, DBLP, and official code; the local binary remains arXiv v2 rather than the OpenReview PDF.
- The Outlier Suppression scripted search encountered arXiv and Semantic Scholar HTTP 429 responses plus missing `openreview-py`. DBLP/OpenAlex/Crossref found the exact work; identity and venue version were closed with official NeurIPS proceedings, arXiv metadata, and the paper-linked code repository. The installed `paper-search` CLI did not support the documented `--json` option, so the report records semantic screening without an unfiltered JSON export.
- The Outlier Suppression paper reports no random-seed error bars and does not provide production end-to-end latency, peak-memory, tail-latency, or power measurements. `SmoothQuant` Tables 2-4 are retained as later direct comparison evidence, not merged into the original authors' claims.
- The Flow Matching OpenReview forum and PDF endpoints returned HTTP 403 to unattended acquisition. Venue identity was closed with the official ICLR record and OpenReview ID; the local binary is explicitly arXiv v2. The paper reports ImageNet/CIFAR NLL, FID and NFE but no random-seed uncertainty or hardware-level latency/memory/power evaluation, and it contains no robotics experiment.
- The Self-Flow exact-title search found the target on arXiv and the official ICML accepted-paper list, but `openreview-py` was unavailable, the OpenReview forum/PDF endpoints presented an HTTP 403 browser challenge, Semantic Scholar returned HTTP 429, and DBLP returned transient HTTP 503 responses. The installed `paper-search` CLI also lacked its documented `--json` flag, so unfiltered metadata was recovered with the programmatic API for arXiv/OpenAlex/Crossref. The local arXiv v1 paper passes `%PDF-`, 37-page pypdf reopen/page-count/unencrypted/first-page-title checks, visual inspection, and ARS structural preflight (`PASS`, 37/37 pages); the 10-page critique passes the same structural checks and preflight (`PASS`, 10/10 pages). Neither artifact is an independent replication.
- The 2026-09-04 targeted search for HB-VLA and the two WorldCache titles returned 41 noisy unique raw records; exact-title/scope screening retained the three assigned work identities. `openreview-py` was unavailable, Semantic Scholar returned HTTP 429 for the HB-VLA and heterogeneous-token queries, and OpenAlex returned a transient HTTP 504. Identity was closed with direct arXiv, official conference pages, project/code repositories, and the local full texts rather than treating the aggregate search as complete coverage.
- The three new local files pass `%PDF-`, pypdf reopen/page-count/unencrypted/first-page-title checks, visual title/results-page inspection, SHA-256 inventory, and ARS structural preflight: HB-VLA `PASS` 9/9 pages, Content-Aware WorldCache `PASS` 33/33, Heterogeneous Token WorldCache `PASS` 26/26. These checks establish readable acquisition and locators, not numerical reproduction.
- HB-VLA's local v2 ends after references despite repeated appendix pointers, and no official code was verified. The missing implementation detail is retained as an unresolved reproducibility boundary rather than reconstructed from assumptions.
- The two WorldCache speedups are not directly comparable: `#21` primarily uses H200 with Cosmos/WAN and 35-step video-world settings; `#22` uses A800 with Voyager/Aether and different schedules/resolutions/cache granularity. Neither paper reports closed-loop policy success under the caching intervention.
- No official MEM code, model weights, or dataset release link was verified on the project page or arXiv record as of the snapshot date.
- The list is advisor-specified rather than systematic-review-selected; coverage reflects the assignment, not the entire field.
- The 2026 survey search is a point-in-time source-verification pass, not an exhaustive systematic review; newer preprints and runtime support may appear after 2026-08-25.
