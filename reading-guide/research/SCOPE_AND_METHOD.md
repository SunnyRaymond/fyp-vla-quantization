# Scope and verification method

## Research question

`ReadingList.md` 具体要求阅读哪些 paper / technical document？怎样把它们组织成能支持每周组会的 staged curriculum？

## Inclusion rules

1. `ReadingList.md` 直接给出 title 或 canonical link 的论文。
2. 明确命名的 series（`π0 / π0.5 / π0.6`）中与该名称直接对应的 primary report。
3. 明确命名的 format family（`FP8 / MXFP8 / MXFP4 / NVFP8 / NVFP4`）的 defining paper/specification；若该名称不是正式 format，则记录 identity resolution，而不虚构论文。
4. 用户明确要求“单开一个数字”的 foundational paper 可进入 active core；必须记录它是 user-promoted core，而不是 silently retrofitted advisor assignment。
5. 用户明确要求准备的 VLA / Robot RL systems paper 可作为 separately labelled companion；它必须填补 execution、latency、control-loop 或 training-systems gap，且不得被表述为 advisor-assigned core。
6. 用户转述李老师新指定并要求“每篇单开一个序号”的论文，进入 active core；按加入指南的顺序分配下一个整数，同时记录 exact title、version 与 name collision。

## Exclusion rules

1. 只是在相关工作中出现、但老师未点名的论文，不进入 core count。
2. Prof. Song Han 与 Prof. Dan Alistarh 的全部 publications 不自动视为 assigned papers。
3. Secondary summaries 不能作为 method/result 的唯一 evidence。
4. 无法在 primary/official source 核验的条目不进入已核验清单。

## Version and deduplication rules

- 同一工作的 arXiv、conference、project page 合并为一个 canonical paper entry。
- 优先使用最新可核验 paper version；venue metadata 与 arXiv version 分开记录。
- Blog / specification / documentation 明确标成 non-paper technical material。

## Search record

- Initial core-list search date: 2026-08-22 (Asia/Singapore)
- Companion update: 2026-08-25; user-requested GPTQ and MEM acquisitions plus a point-in-time search for recent peer-reviewed LLM quantization surveys. Companion additions do not alter the core count.
- Companion update: 2026-08-29; user-requested TurboQuant controversy audit, official ICLR final acquisition, and separate RaBitQ and DRIVE/EDEN critical companions. These additions do not alter the core count.
- Companion update: 2026-08-29; original RaBitQ and its multi-bit extension acquired with a separate integration analysis covering weight-only PTQ, KV cache, activations, kernels, and theory-transfer limits. These additions do not alter the core count.
- Companion update: 2026-08-31; user-requested FlashVLA exact-paper lookup for arXiv:2608.27384, full-text acquisition, name-collision resolution against arXiv:2505.21200, and a VLA systems reading scaffold. This addition does not alter the core count.
- Companion update: 2026-08-31; user-requested FlashSAC exact-paper search, RSS 2026 Outstanding Paper Award verification, arXiv:2604.04539 v2 acquisition, official project/code audit, and a critical Robot RL systems reading scaffold. This addition does not alter the core count.
- Companion update: 2026-08-31; representation-alignment prior-art pass acquired `REPA`, `iREPA`, and ten direct/adjacent `VLA/WAM` papers under `papers/19-repa-irepa-vla-alignment/`. These twelve papers remain a separately labelled companion route and do not alter the core count.
- Core expansion: 2026-08-31; user-requested original LoRA exact-paper lookup and explicit promotion to standalone integer `#15`. ICLR 2022 venue identity, arXiv:2106.09685 v2 local artifact, Microsoft Research record, DBLP result, and official code were cross-checked. The original 15-paper advisor-aligned inventory therefore becomes a 16-paper active core; LoRA is user-promoted rather than advisor-assigned.
- Core expansion: 2026-08-31; user-requested Outlier Suppression exact-paper lookup and explicit promotion to standalone integer `#16`. NeurIPS 2022 Main Conference identity, official 13-page paper, official 10-page supplemental, arXiv:2209.13325 v1-v3 metadata, and paper-linked code were cross-checked. The active core therefore becomes 17; Outlier Suppression is user-promoted rather than advisor-assigned.
- Core expansion: 2026-08-31; user-requested *Flow Matching for Generative Modeling* preparation and explicit promotion to standalone integer `#17`. ICLR 2023 venue identity, official Top 25% label, OpenReview ID `PqvMRDCJT9t`, arXiv:2210.02747 v1-v2 metadata, and the 28-page arXiv v2 local artifact were cross-checked. The active core therefore becomes 18; Flow Matching is user-promoted rather than advisor-assigned.
- Core expansion: 2026-08-31; user-requested ICML 2026 `Self-Flow` exact-paper lookup and explicit promotion to standalone integer `#18`. The target was resolved to Chefer et al., *Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis*: official ICML poster `65011`, OpenReview ID `HoThWhfxiK`, arXiv:2603.06507 v1, official project/code, and the 37-page arXiv v1 local artifact were cross-checked. The active core therefore becomes 19; Self-Flow is user-promoted rather than advisor-assigned. The later arXiv:2607.02508 mechanism challenge is retained as a separately labelled critical companion and does not change the core count.
- Core expansion: 2026-09-04; 李老师 assigned three exact papers and the user requested one new integer per paper. `HBVLA` (method label `HB-VLA`) is stored as `#20` using arXiv:2602.13710 v2; *WorldCache: Content-Aware Caching for Accelerated Video World Models* is `#21` using arXiv:2603.22286 v1 with ECCV 2026 acceptance verified separately; *WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching* is `#22` using arXiv:2603.06331 v2 with ICML 2026 / PMLR 306 identity. The two `WorldCache` entries are distinct works, not versions. Active core becomes 22.
- Sources: arXiv, DBLP, OpenAlex, Crossref, official project/code pages, official publisher/venue pages, official vendor specifications/documentation, official faculty/lab profiles. OpenReview, Semantic Scholar, and arXiv API degradation is recorded in the exact-paper search reports.
- Search strings and per-item evidence are recorded in `research/phase2/` and the dated [three-paper targeted scan](THREE_PAPER_SCAN_2026-09-04.md).

## Planned evidence labels

- `verified-full-text`
- `verified-metadata`
- `official-technical-material`
- `unverified`
