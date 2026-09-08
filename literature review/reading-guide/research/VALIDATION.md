# Delivery validation

> Snapshot: 2026-09-04（李老师新增指定 HBVLA 与两篇不同的 WorldCache；各自单开整数 `#20`、`#21`、`#22`；active core expanded to 22）

## Inventory

- Original advisor-list items represented in the master guide: **18/18** (`OpenVLA`; three PI entries; `ω-0`; `BitVLA`; `FP8`; four named format labels; five named quantization algorithms; two researchers).
- Core paper / technical-report identities after deduplication and the three 2026-09-04 additions: **22**.
- Per-core-paper README files present: **22/22**.
- Local canonical core-paper PDFs present and structurally readable: **22/22**.
- Official core-paper supplemental artifacts present and structurally readable: **1/1**（Outlier Suppression）。
- VLA systems companion README files present: **1/1**.
- Local VLA systems companion PDFs present and structurally readable: **1/1**.
- Robot RL systems companion README files present: **1/1**.
- Local Robot RL systems companion PDFs present and structurally readable: **1/1**.
- PI-series companion README files present: **1/1**.
- Local PI-series companion PDFs present and structurally readable: **1/1**.
- BitVLA companion README files present: **4/4**.
- Local BitVLA companion PDFs present and structurally readable: **4/4**.
- Quantization companion reading entries present: **8/8**（the two foundational RaBitQ papers share one README; the two TurboQuant dispute papers share one comparison README）.
- Local quantization companion PDFs present and structurally readable: **8/8**.
- Representation-alignment companion reading entries present: **12/12**（REPA、iREPA、10 direct/adjacent VLA/WAM papers）。
- Local representation-alignment companion PDFs present and structurally readable: **12/12**.
- Self-Flow mechanism companion reading entries present: **1/1**（From SRA to Self-Flow）。
- Local Self-Flow mechanism companion PDFs present and structurally readable: **1/1**.
- Official companion specifications present and structurally readable: **2/2**.
- Archived core-paper versions retained for version audit: **1**（TurboQuant arXiv v1）。
- Total local PDFs after this pass: **54 / 514,301,964 bytes / 490.48 MiB**（22 canonical core + 1 core-paper supplemental + 1 archived core version + 1 VLA systems companion + 1 Robot RL systems companion + 1 PI-series companion + 4 BitVLA companions + 8 quantization companions + 12 representation-alignment companions + 1 Self-Flow mechanism companion + 2 specifications）。
- Within the 22-paper active core, peer-reviewed / accepted conference records: **15**.
- Within the 22-paper active core, preprints / official technical reports: **7**.
- BitVLA companions with peer-reviewed conference records: **4/4**.
- Optional researcher-watchlist papers: **8**, explicitly excluded from the core count；`OPTQ/GPTQ` and `QuaRot` are now locally prepared as companions, leaving **6** watchlist-only papers.

## Structural checks

- Every paper note contains: identity/status, background, problem, method, results with locator language, limitations, project relevance, 20-minute route, 60–90/90-minute route, reading questions, meeting card, and evidence boundary.
- Markdown files under `reading-guide/`: **53**；local Markdown links checked: **304**；broken targets: **0**.
- Final artifacts contain **0** unresolved drafting placeholders.
- The master guide links all 22 core paper notes, the 8-week schedule, the new three-paper targeted scan, source-verification report, method note, NVFP8 clarification, and researcher watchlist.
- The master guide additionally links one VLA systems, one Robot RL systems, one PI-series, four BitVLA, eight quantization, twelve representation-alignment, and one Self-Flow mechanism companion readings without inflating the 22-paper active core count.
- PDF inventory records page count, byte size, SHA-256, local-version status, and acquisition source for every downloaded file.

## Identity and version checks

- The bare `π₀.₆` advisor entry is mapped to `π*₀.₆`; `π₀.₆-MEM` is separately acquired and documented as a companion rather than merged with that assigned entry.
- MEM's local file is the Physical Intelligence official project PDF; canonical arXiv metadata is v2, and the binary provenance distinction is explicit.
- arXiv and venue versions of the same work are merged rather than double-counted.
- `MXFP8/MXFP4`: OCP MX specification is treated as normative; the microscaling paper is treated as empirical evidence.
- `NVFP4`: vendor-defined format/recipe; not represented as an OCP/IEEE standard.
- `NVFP8`: unresolved checkpoint/product label; no fabricated paper or datatype entry.
- `OPTQ` and commonly used name `GPTQ` are treated as one locally prepared companion paper; the local PDF is the ICLR 2023 Published Version.
- `QuaRot` is recorded as a NeurIPS 2024 venue-final companion and linked as the fixed/randomized Hadamard-rotation bridge into `SpinQuant`.
- `FlashVLA`: the local entry is arXiv:2608.27384 v1 and is explicitly separated from the different action-reuse/token-pruning method also named `FlashVLA` in arXiv:2505.21200.
- `FlashSAC`: the canonical venue identity is RSS 2026 and the official award label is Outstanding Paper Award; the local 42-page artifact is arXiv:2604.04539 v2. It passes `%PDF-`, pypdf reopen/page-count/unencrypted/first-page-title, SHA-256, visual render inspection, and ARS structural preflight (`PASS`, 42/42 pages). The note keeps the official RSS `50+`, arXiv/project `60+`, and live-repository `100+` task-count claims version bound and flags the paper's UTD-denominator and sim-to-real reward-description conflicts.
- `LoRA`: the canonical venue identity is ICLR 2022; the local 26-page artifact is arXiv:2106.09685 v2 because OpenReview blocked unattended PDF acquisition. The note separates venue identity, arXiv artifact version, and the scope of the paper's no-extra-latency claim.
- `Outlier Suppression`: the canonical venue identity is NeurIPS 2022 Main Conference Track; the local 13-page proceedings final and 10-page official supplemental both pass `%PDF-`, page-count, unencrypted, first-page-text, SHA-256, and ARS structural preflight checks. arXiv:2209.13325 v1-v3 remains one work cluster. Original claims, later SmoothQuant direct comparison, and VLA synthesis are separated.
- `Flow Matching`: the canonical venue identity is ICLR 2023 with an official Top 25% label; the local 28-page artifact is arXiv:2210.02747 v2 because OpenReview returned HTTP 403 to unattended PDF acquisition. It passes `%PDF-`, pypdf reopen/page-count/unencrypted/first-page-title, SHA-256, visual render inspection, and ARS structural preflight (`PASS`, 28/28 pages). The note separates marginal FM, practical CFM, conditional OT, ImageNet evidence, and the later algebraic mapping into `pi0`.
- `Self-Flow`: the canonical venue identity is ICML 2026 poster `65011`, with OpenReview ID `HoThWhfxiK`; the local 37-page artifact is arXiv:2603.06507 v1 because OpenReview returned HTTP 403 to unattended PDF acquisition. It passes `%PDF-`, pypdf reopen/page-count/unencrypted/first-page-title, SHA-256, visual inspection of the title, method, main results, `SIMPLER`, limitations, and Appendix E pages, plus ARS structural preflight (`PASS`, 37/37 pages). The separate 10-page arXiv:2607.02508 v1 critique passes the same checks and ARS structural preflight (`PASS`, 10/10 pages), and is labelled as a mechanism companion rather than a revised Self-Flow version.
- `HBVLA` / method `HB-VLA`: the local file is arXiv:2602.13710 v2, 9 pages. It passes `%PDF-`, pypdf reopen/page-count/unencrypted/first-page-title, SHA-256, visual inspection, and ARS structural preflight (`PASS`, 9/9 pages). No venue or official code is claimed. The PDF's repeated appendix pointers are flagged because this v2 ends after references and contains no appendix; the `2.93×` latency headline remains protocol-incomplete.
- `WorldCache — Content-Aware`: the local file is arXiv:2603.22286 v1, 33 pages; ECCV 2026 acceptance is recorded separately from the binary version. It passes the same PDF checks and ARS structural preflight (`PASS`, 33/33 pages). Its H200 PAI-Bench/EgoDex evidence is explicitly labelled as video/world-model evaluation rather than closed-loop policy success.
- `WorldCache — Heterogeneous Token`: the local file is arXiv:2603.06331 v2, 26 pages, with ICML 2026 / PMLR 306 identity. It passes the same PDF checks and ARS structural preflight (`PASS`, 26/26 pages). It is separated from the content-aware paper by full subtitle/arXiv ID; its A800 speedup is not directly compared with the H200 paper.
- The JCST 2026 survey is recorded as the latest dedicated peer-reviewed snapshot found; the Neural Networks 2025 survey is recorded as the more comprehensive reference, with its local arXiv-v3 manuscript boundary explicit.
- `TurboQuant`: official ICLR 2026 proceedings final is local and canonical; the note records the arXiv-v1 to final Table 1 change (`49.44 → 49.74` for the 2.5-bit LongBench average) and the removal of arXiv-v1 timing Table 2.
- `RaBitQ`: original arXiv v1 is structurally verified; SIGMOD 2024 identity and DOI `10.1145/3654970` are recorded separately.
- `Practical and Asymptotically Optimal Quantization...`: multi-bit arXiv v1 is structurally verified; SIGMOD 2025 identity and DOI `10.1145/3725413` are recorded separately.
- `Revisiting RaBitQ and TurboQuant`: arXiv v2 is verified and the VecDB@VLDB 2026 poster acceptance is confirmed from the official workshop programme.
- `A Note on TurboQuant and the Earlier DRIVE/EDEN Line of Work`: arXiv v1 is verified and kept separate from the RaBitQ baseline dispute.
- `Quantize-then-Distill`: exact BitVLA name/implementation is separated from prior quantization-aware Knowledge Distillation; Apprentice Scheme-C is recorded as direct conceptual predecessor, not an exact original recipe.
- `three 224×224 images`: resolved as ALOHA's one third-person/top-down plus two wrist-camera views; kept distinct from BitVLA's two-view LIBERO and one-view Franka settings.
- `bitsandbytes`: software-library scope is separated from LLM.int8; BitVLA's underspecified 4-bit FP4/NF4 configuration is explicitly flagged.

## Evidence boundary

This validation checks inventory, note structure, local navigation, identity/version discipline, and the presence of source locators. It does not constitute independent experimental replication or an academic-misconduct finding. Private correspondence, proprietary robot data, custom real-world evaluation, vendor hardware, and unavailable kernels remain external reproducibility constraints.
