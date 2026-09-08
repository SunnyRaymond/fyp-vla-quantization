# Search report: REPA / iREPA transfer to VLA

> **Search date:** 2026-08-31  
> **Coverage:** 2023–2026  
> **Databases attempted:** arXiv, DBLP, OpenAlex, OpenReview, Semantic Scholar, Crossref  
> **Selection rule:** keep papers that implement, closely approximate, or create a necessary counter-baseline for representation alignment in robot control

## Queries

### Exact/direct query

```text
"representation alignment" VLA robot policy | REPA robot policy diffusion action
```

Returned source counts before cross-source merging:

| Source | Hits |
|---|---:|
| arXiv | 20 |
| DBLP | 0 |
| OpenAlex | 20 |
| OpenReview | 0 |
| Semantic Scholar | 10 |
| Crossref | 20 |

The tool reported 65 unique records and 5 cross-source merges before manual relevance filtering.

### Broad-mechanism query

```text
vision language action pretrained visual encoder feature distillation |
VLA auxiliary visual reconstruction DINO |
flow matching VLA visual representation supervision
```

| Source | Hits |
|---|---:|
| arXiv | 30 |
| DBLP | 0 |
| OpenAlex | 30 |
| OpenReview | 0 |
| Semantic Scholar | 0 |
| Crossref | 30 |

The tool reported 83 unique records and 7 cross-source merges before manual relevance filtering.

### Programmatic recovery query set

Because the installed CLI did not expose the documented `--json` option, the same skill's programmatic API was used with these five queries:

```text
representation alignment VLA robot policy REPA diffusion action
future latent representation alignment robot policy
spatial representation alignment vision language action
frozen visual teacher VLA cosine alignment
world action model representation alignment
```

The unfiltered recovery file is [search-results-unfiltered.json](search-results-unfiltered.json). It contains 138 deduplicated records and the complete source payload available from that run.

| Source | Raw hits in recovery run |
|---|---:|
| arXiv | 50 |
| DBLP | 2 |
| OpenAlex | 50 |
| Semantic Scholar | 10 |
| Crossref | 50 |

## API and environment notes

- `OpenReview` was unavailable because `openreview-py` was not installed.
- `Semantic Scholar` returned repeated HTTP 429 rate limits; only the first recovery query returned results.
- `OpenAlex` and `DBLP` returned intermittent HTTP 504/503 responses and were retried by the search runtime.
- The results below were therefore verified against primary proceedings/arXiv full texts instead of treating search metadata as decisive.

## Manually verified relevant set

| Work | Year | Venue/status at snapshot | Keep class | Verification basis |
|---|---:|---|---|---|
| Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think (`REPA`) | 2025 | ICLR 2025 Oral | Method origin | official proceedings final |
| What Matters for Representation Alignment: Global Information or Spatial Structure? (`iREPA`) | 2026 | ICLR 2026 | Method refinement | official proceedings final |
| FLARE: Robot Learning with Implicit World Modeling | 2025 | CoRL 2025 | Direct | PMLR final; method explicitly cites/adapts REPA |
| Spatial Forcing: Implicit Spatial Representation Alignment for Vision-Language-Action Model | 2025/2026 | arXiv v2; ICLR 2026 work | Direct | local arXiv v2 full text |
| ReconVLA: Reconstructive Vision-Language-Action Model as Effective Robot Perceiver | 2025/2026 | arXiv v1; AAAI 2026 record | Adjacent | reconstruction, not teacher-feature alignment |
| FRAPPE: Infusing World Modeling into Generalist Policies via Multiple Future Representation Alignment | 2026 | arXiv v1 | Direct/extension | local full text |
| Robot-DIFT: Learning Task-Generalizable Robot Skills from Diffusion Features | 2026 | arXiv | Adjacent/direct distillation | primary arXiv record; robot policy feature distillation |
| FutureVLA: Joint Visuomotor Prediction for Vision-Language-Action Model | 2026 | arXiv v1 | Partial/extension | local full text |
| VEGA: Visual Encoder Grounding Alignment for Spatially-Aware Vision-Language-Action Models | 2026 | arXiv v1 | Direct | local full text |
| HARP-VLA: Human Action Representations for Generalizable Vision-Language-Action Models | 2026 | arXiv | Adjacent | human/robot domain representation alignment, not REPA teacher loss |
| Making Foresight Actionable: Repurposing Representation Alignment in World Action Models (`AGRA`) | 2026 | arXiv v1 | Direct WAM | local full text; literal repurposing |
| SAM3D-Guided Object-Centric Representation Alignment for Vision-Language-Action Models | 2026 | arXiv v1 | Direct/object-centric | local full text |
| Mind-VLA: Instruction-Aware Spatial Representation Alignment for Vision-Language-Action Models | 2026 | arXiv v2 | Direct/instruction-aware | local full text |
| Robust-WAM: Bridging Generative Pretraining and Semantic Foresight in World-Action Models | 2026 | arXiv v1 | Direct WAM/counter-baseline | local full text |

## Excluded result classes

The following result types appeared in broad searches but were not counted as implementations of the proposed idea:

- text-to-image/video representation-alignment papers without robot control;
- ordinary vision-language contrastive pretraining without action prediction;
- robot-policy methods that concatenate/fuse 3D features as inputs but do not align hidden features;
- world models operating wholly in a pretrained latent space without a separate alignment objective;
- action-language semantic alignment that does not use an external visual teacher;
- surveys, workshop summaries, project pages, patents, and duplicate arXiv/venue records;
- papers where “alignment” means trajectory synchronization, coordinate calibration, human preference, or cross-embodiment data correspondence.

## Negative-search audit for exact iREPA transfer

Searches included exact combinations of `iREPA` with `Vision-Language-Action`, robot policy, robot learning, `VLA`, and `WAM`. In addition, every local candidate PDF was searched for the literal term `iREPA`.

Result:

- only the iREPA paper itself contained the term;
- several robot papers cite or describe `REPA`;
- none of the checked robot papers verified the exact `3×3 Conv projector + spatial normalization` pair.

This supports a **search-bounded absence statement**, not a universal proof of absence.

## Most informative primary links

- [REPA — ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d9e42b4d7163931f3689d6d6fbaa11d0-Abstract-Conference.html)
- [iREPA — ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3929a7785bd56f57edcff0152ab41289-Abstract-Conference.html)
- [FLARE — CoRL 2025](https://proceedings.mlr.press/v305/zheng25a.html)
- [Spatial Forcing](https://arxiv.org/abs/2510.12276)
- [FRAPPE](https://arxiv.org/abs/2602.17259)
- [FutureVLA](https://arxiv.org/abs/2603.10712)
- [VEGA](https://arxiv.org/abs/2605.10485)
- [AGRA](https://arxiv.org/abs/2606.12217)
- [SAM3D-Guided](https://arxiv.org/abs/2607.25912)
- [Mind-VLA](https://arxiv.org/abs/2608.04633)
- [Robust-WAM](https://arxiv.org/abs/2608.05903)
- [ReconVLA](https://arxiv.org/abs/2508.10333)
- [Robot-DIFT](https://arxiv.org/abs/2602.11934)
- [HARP-VLA](https://arxiv.org/abs/2605.31234)

## Synthesis

### Overview

The literature has moved from current-frame spatial alignment, through future-token latent prediction, to object-conditioned 3D and future-semantic alignment. The broad method family is no longer novel.

### Trends

1. Frozen visual teachers are increasingly spatial/3D-aware (`VGGT`, `DINOv2-FiT3D`, `SAM3D`).
2. Alignment targets move closer to the action interface: future tokens, action-stream queries, or intermediate video features.
3. Several methods remove the teacher/projector at inference, but some retain learnable query tokens.
4. Newer papers condition alignment on future time, object identity, or language instructions.

### Key themes

- current vs future target;
- global `CLS` vs local patch structure;
- visual encoder vs LLM tokens vs action/video `DiT` hidden states;
- 2D, 3D, object-centric, and multi-view alignment;
- optimization acceleration vs final success vs OOD robustness;
- training-only overhead vs deployed latency.

### Keyword frequency in the retained conceptual set

The dominant repeated concepts are `representation alignment`, `future representation`, `spatial`, `3D`, `frozen teacher`, `cosine loss`, `VLA`, `WAM`, and `flow matching`. Frequency is descriptive only; it is not evidence of novelty or effectiveness.

### Most-cited accepted paper

The search APIs returned incomplete and rate-limited citation metadata, so no current “most cited” claim is reported. The accepted anchor papers are `REPA` (ICLR 2025), `FLARE` (CoRL 2025), and `iREPA` (ICLR 2026).

### Most-cited by first author

Not reported because the citation snapshot was incomplete and would be misleading.

### Recommendations

1. Read `REPA → iREPA → FLARE → Spatial Forcing → VEGA → AGRA` in that order.
2. Treat `Robust-WAM` as counter-evidence to the assumption that patch structure is always best.
3. If pursuing the project, run the exact iREPA component ablation before adding 3D, future, or object-conditioning complexity.
4. Re-run the search immediately before proposal submission because this is a fast-moving 2026 topic.
