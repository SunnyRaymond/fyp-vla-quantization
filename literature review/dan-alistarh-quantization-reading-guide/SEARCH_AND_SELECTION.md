# Search and Selection Record

## Search question

Which Dan Alistarh papers published from 2022 onward are most useful for learning modern neural-network quantization as a coherent progression rather than as an unranked bibliography?

## Inclusion criteria

- Publication year 2022 or later.
- Dan Alistarh is an author or co-author.
- Quantization, low-bit computation, or a compression/deployment question driven by quantization is central to the paper.
- The paper contributes at least one of: method, theory, representation, kernel/runtime system, or broad deployment evidence.
- An official proceedings or institutional PDF can be obtained and validated.

## Exclusion and deferral criteria

- Papers mainly about pruning, sparsity, federated learning, or distributed optimization were excluded from the quantization route.
- Closely related papers such as SpQR, QUIK, PV-Tuning, QTIP-related work, and other 2024–2026 variants were deferred to keep the route readable. This is not a judgment that they are unimportant.
- SparseGPT is influential but its central object is pruning, so it is not treated as a core quantization paper here.
- 2026 papers are listed only as an optional watchlist because the selected route already covers the method lineage through 2025.

## Reproducible search

The installed paper-search script was checked before use. Its current interface does not support the documented `--json` flag, so the search was run in text mode:

```powershell
python C:\Users\Raymond\.codex\skills\paper-search\scripts\search_papers.py --queries "Dan Alistarh quantization|Dan Alistarh low-bit compression|Dan Alistarh GPTQ AQLM QuaRot" --start-year 2022 --end-year 2026 --max-papers 10
```

The run produced 50 deduplicated candidate records from arXiv, DBLP, OpenAlex, Crossref, and attempted OpenReview and Semantic Scholar. Some providers returned connection-reset, DNS, HTTP 504, missing-package, or HTTP 429 errors. Candidate identity and final selection were therefore rechecked against official proceedings and the ISTA publication listing rather than accepted from aggregator metadata alone.

## Selected corpus and rationale

| Status | Paper | Decision rationale |
|---|---|---|
| Core | Optimal Brain Compression | General foundation for efficient OBS-style post-training pruning and quantization |
| Core | OPTQ / GPTQ | Direct continuation that makes the second-order approach practical for very large Transformers |
| Core | AQLM | Representative extreme-compression shift from scalar grids to learned additive codebooks |
| Core | QuaRot | Representative end-to-end weight, activation, and KV-cache quantization with rotations |
| Core | HIGGS | Adds a theoretical bridge from local error to perplexity plus data-free and variable-bitwidth methods |
| Core | HALO | Changes the problem from inference-only quantization to low-precision fine-tuning and communication |
| Side branch | QMoE | Important but specialized to massive Mixture-of-Experts and custom sub-1-bit formats |
| Side branch | “Give Me BF16 or Give Me Death”? | Broad empirical audit that is best read after at least one method paper |

## Source hierarchy used

1. Official conference proceedings PDF and metadata.
2. ISTA institutional publication record or author-group listing.
3. arXiv only when useful for identity cross-checking.
4. Aggregator records only for discovery, never as the final version boundary.

## Coverage advisory

This is an author-centered reading route, not a systematic review of the entire quantization field. It is intentionally concentrated on LLMs, post-training quantization, rotations, low-bit kernels, and Dan Alistarh’s collaborator network. It does not provide balanced coverage of training-aware quantization, vision-only models, mobile accelerators, or competing research groups.
