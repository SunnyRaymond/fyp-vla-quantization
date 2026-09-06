# Self-Flow exact-paper search report

> **Search date:** 2026-08-31 (Asia/Singapore)  
> **Window:** 2025-2026  
> **Target:** identify the ICML 2026 work known as `Self-Flow`, acquire the correct full text, and separate it from similarly named papers.  
> **Detailed reading package:** [reading-guide/papers/18-self-flow/README.md](reading-guide/papers/18-self-flow/README.md)

## Search execution

The installed `paper-search` CLI was run with the query union `Self-Flow | Self Flow` across arXiv, DBLP, OpenAlex, OpenReview, Semantic Scholar, and Crossref.

- Raw per-source hits: arXiv 20, DBLP 20, OpenAlex 20, OpenReview 0, Semantic Scholar 0, Crossref 20.
- Unified result set: 43 unique records after 37 cross-source duplicate merges.
- Semantic screening against the user's exact request retained 2 records and filtered 41 as unrelated uses of “self” and “flow.”
- `openreview-py` was unavailable, so the CLI OpenReview connector returned no records.
- Semantic Scholar returned HTTP 429 after bounded retries.
- DBLP returned transient HTTP 503 responses during retries but still produced results.
- The installed CLI has no `--json` option. A programmatic exact-title recovery query across arXiv, OpenAlex, and Crossref saved 30 unfiltered raw records in [search-results-unfiltered.json](reading-guide/papers/18-self-flow/search-results-unfiltered.json); only the exact arXiv work was relevant.
- Official acceptance was checked separately against the [ICML 2026 accepted-paper list](https://icml.cc/Downloads/2026), which links the title to poster `65011`, and against OpenReview ID `HoThWhfxiK`.

## Semantically retained results

| # | Work | Status at snapshot | Why retained | Primary source |
|---:|---|---|---|---|
| 1 | *Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis* — Hila Chefer, Patrick Esser, Dominik Lorenz, Dustin Podell, Vikash Raja, Vinh Tong, Antonio Torralba, Robin Rombach | ICML 2026 accepted poster; arXiv:2603.06507 v1 | This is the method called `Self-Flow` and the paper the user meant | [ICML list](https://icml.cc/Downloads/2026) · [OpenReview](https://openreview.net/forum?id=HoThWhfxiK) · [arXiv](https://arxiv.org/abs/2603.06507v1) |
| 2 | *From SRA to Self-Flow: Data Augmentation or Self-Supervision?* — Dengyang Jiang, Mengmeng Wang, Harry Yang, Jingdong Wang | arXiv:2607.02508 v1 | Direct post-publication mechanism critique; not the ICML paper | [arXiv](https://arxiv.org/abs/2607.02508v1) |

## Overview

The target identity is closed: `Self-Flow` refers to Chefer et al.'s ICML 2026 poster. The similarly named Jiang et al. paper is a later arXiv preprint that studies whether `Dual-Timestep Scheduling` works through cross-noise self-supervision or noise-state data augmentation.

## Trends

This is an exact-identity search, not a field-wide literature review. The two retained records form a short chronological sequence: a March 2026 method paper followed by a July 2026 mechanism challenge. The main methodological tension is not whether two-timestep training improves the reported setup, but why it improves it.

## Key themes

1. **Internal representation learning:** `Self-Flow` replaces a frozen external encoder with an `EMA teacher` inside the generative model (paper 1).
2. **Dual-Timestep Scheduling:** tokens from one sample receive two noise levels while preserving per-token marginal timestep sampling (papers 1-2).
3. **Mechanism identification:** the original paper emphasizes cleaner-to-noisier token interaction; the critique attributes much of the gain to data augmentation (papers 1-2).
4. **Multi-modal and embodied transfer:** the original paper evaluates image, video, audio, mixed multi-modal training, and `SIMPLER` joint video-action prediction (paper 1).

## Keywords frequency

Counts use normalized title tokens from the two retained records.

| Keyword | Count |
|---|---:|
| self / self-supervised / self-supervision | 4 |
| flow | 2 |
| matching | 1 |
| multi-modal | 1 |
| data augmentation | 1 |

## Most cited by accepted paper

The available search snapshot returned zero citations for these very recent 2026 records. Citation counts are therefore not used to rank them.

| Rank | Title | Year | Citations in snapshot |
|---:|---|---:|---:|
| 1 | Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis | 2026 | 0 |

## Most cited by first author

| Rank | Author | Papers in set | Total citations in snapshot |
|---:|---|---:|---:|
| 1 | Hila Chefer | 1 | 0 |
| 2 | Dengyang Jiang | 1 | 0 |

## Recommendations for reading

1. [Flow Matching for Generative Modeling](reading-guide/papers/17-flow-matching/README.md) — understand the base velocity-regression objective and ODE sampling boundary.
2. [Self-Flow](reading-guide/papers/18-self-flow/README.md) — read the method, multi-modal results, and `SIMPLER` appendix as the main target.
3. [From SRA to Self-Flow](reading-guide/papers/18-self-flow/critique-from-sra-to-self-flow-arxiv-v1.pdf) — use it after the original paper to test the scheduler's causal explanation.

## Version and source boundary

- Local `paper-arxiv-v1.pdf` is the 37-page arXiv v1 author manuscript, not an ICML proceedings binary.
- ICML 2026 acceptance is established by the official accepted-paper list; the linked OpenReview identity is `HoThWhfxiK`.
- Local `critique-from-sra-to-self-flow-arxiv-v1.pdf` is a 10-page arXiv v1 preprint and is not counted as a second core paper.
- The two papers' claims are kept separate; the critique does not retroactively change what the original authors claimed.

