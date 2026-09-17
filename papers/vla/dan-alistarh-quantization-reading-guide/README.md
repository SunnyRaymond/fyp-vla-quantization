# Dan Alistarh Quantization Reading Guide (2022–2025)

> Evidence snapshot: 2026-08-31  
> Scope: papers authored or co-authored by Dan Alistarh, published from 2022 onward, with direct relevance to quantization, low-bit execution, or compression-aware deployment.

## What is ready

This folder contains eight verified official PDFs and one self-reading note for each paper. The main route is deliberately limited to six papers; two specialized papers are kept as side branches so that Mixture-of-Experts and deployment benchmarking do not interrupt the core method progression.

- [PDF inventory and checksums](PDF_INVENTORY.md)
- [Search and selection record](SEARCH_AND_SELECTION.md)
- [Validation report](VALIDATION.md)

## Core route

| Order | Paper | Venue | Why it is here | Local note |
|---|---|---|---|---|
| 1 | Optimal Brain Compression | NeurIPS 2022 | Establishes the layer-wise second-order reconstruction framework behind later work | [Read](papers/01-optimal-brain-compression/README.md) |
| 2 | OPTQ / GPTQ | ICLR 2023 | Scales one-shot second-order weight quantization to 175B-parameter Transformers | [Read](papers/02-optq-gptq/README.md) |
| 3 | AQLM | ICML 2024 | Moves from scalar quantization to learned additive codebooks below 3 bits per parameter | [Read](papers/03-aqlm/README.md) |
| 4 | QuaRot | NeurIPS 2024 | Uses function-preserving rotations to make end-to-end W4A4KV4 quantization viable | [Read](papers/04-quarot/README.md) |
| 5 | HIGGS | NAACL 2025 | Connects layer-wise MSE to end-to-end perplexity and derives data-free, non-uniform quantization | [Read](papers/05-higgs/README.md) |
| 6 | HALO | NeurIPS 2025 | Extends rotation-based low precision from inference into forward, backward, communication, and activation storage | [Read](papers/06-halo/README.md) |

The conceptual progression is:

`second-order local reconstruction` → `LLM-scale weight-only PTQ` → `extreme vector/codebook compression` → `rotation-based W4A4KV4 inference` → `theory-guided data-free bit allocation` → `low-precision fine-tuning`

## Specialized side branches

| Paper | Venue | Read when | Local note |
|---|---|---|---|
| QMoE | MLSys 2024 | You need Mixture-of-Experts, sub-1-bit storage, or compression-format/kernel co-design | [Read](papers/07-qmoe/README.md) |
| “Give Me BF16 or Give Me Death”? | ACL 2025 | You need deployment decisions across FP8, INT8, INT4, latency, throughput, GPU count, and cost | [Read](papers/08-bf16-tradeoffs/README.md) |

## Recommended schedules

### 90-minute sampler

1. OPTQ / GPTQ: Abstract, Sections 3.1–3.4, Tables 2 and 6 — 25 minutes.
2. QuaRot: Figure 1, method overview, main W4A4KV4 results — 25 minutes.
3. HIGGS: Abstract, linearity theorem statement, Tables 1–3, Limitations — 25 minutes.
4. Use the remaining 15 minutes to write one comparison across objective, calibration data, numerical format, and real runtime evidence.

### One-day route (about 5 hours)

1. Optimal Brain Compression — 45 minutes.
2. OPTQ / GPTQ — 60 minutes.
3. AQLM — 60 minutes.
4. QuaRot — 60 minutes.
5. HIGGS — 60 minutes.
6. Choose either HALO or the BF16 deployment audit — 45 minutes.

### Two-week route

- Week 1: read the six core papers in order and answer only the Reading Questions in each note.
- Week 2: reproduce one small comparison on a model you can run locally. Record model, calibration corpus, group size, kernel, batch size, prompt length, GPU, peak memory, time-to-first-token, inter-token latency, throughput, and task quality. Then read the side branch that best matches the result.

## How to read this body of work

Keep four evidence layers separate:

1. **Representation**: scalar, vector, codebook, ternary, integer, or floating-point format.
2. **Optimization**: RTN, second-order compensation, learned codebooks, rotations, or dynamic bit allocation.
3. **Execution**: whether compatible kernels actually reduce latency, memory traffic, communication, or GPU count.
4. **Evaluation boundary**: model family, calibration source, task suite, batch size, sequence length, and hardware.

A lower nominal bit count is not automatically a better deployment. Metadata, scales, codebooks, packing, dequantization, Hadamard overhead, KV cache, peak memory, and workload shape all matter.

## Optional 2026 watchlist

These are recent Dan Alistarh co-authored papers found during the search but not added to the core folder because they have not yet accumulated the same publication/runtime context as the selected route:

- Statistically-Lossless Quantization of Large Language Models
- MatGPTQ: Matrix-Level Post-Training Quantization for Large Language Models
- ECO: Efficient Quantization-Aware Training with Orthogonal Transformations

Treat them as a refresh list, not as required reading for the current route.

## Evidence boundary

Paper-specific numerical statements in the notes are transcribed from the local official PDFs. “Why it matters” and FYP suggestions are synthesis, not author claims. The Reading Questions are intentionally left unanswered so that the folder works as a self-reading companion.

AI assistance was used to search, organize, validate, and scaffold the reading notes; paper identity, venue, PDF integrity, and reported numbers were checked against the local source files.
